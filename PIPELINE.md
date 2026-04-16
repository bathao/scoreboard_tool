# Current Pipeline Implementation

## Purpose
Snapshot of the production pipeline as currently implemented in code. Update this
file whenever the pipeline structure changes (new step added, step removed, signal
source swapped, model replaced, etc.).

This file describes **what is**, not **what should be**:
- `ROADMAP_PRODUCTION.md` = long-term target (what we want to be)
- `PIPELINE.md` = current implementation (what we are now) ← **this file**
- `PROJECT_PROGRESS.md` = daily work log (how we got here)
- `PROJECT_ACTION_PLAN.md` = operational board (what's next)

## Top-level Flow
```
A. Setup (Web UI)
    ↓
B. Initial Job Pipeline   (staged, with operator confirmation pauses)
   ├─ Stage 1: trim_input + detect_sets           → PAUSE (confirm sets)
   ├─ Stage 2: detect_rallies per set              → PAUSE (confirm rally counts)
   └─ Stage 3: export_clips + predict_winners      → auto → review
    ↓
C. Review (operator UI loop)
    ↓
D. Export (preview / final scoreboard render)
```

---

## Phase A — Setup (Web UI)

| Step | Action | Input | Output |
|------|--------|-------|--------|
| A1 | Browse and select raw video | filesystem | `raw_video_path` |
| A2 | (Optional) Click "Identify Players" | `raw_video_path` + face DB | Pre-filled `player_a_name`, `player_b_name` |
| A3 | Fill setup form: `best_of`, `trim_start_sec`, `player_a_starts_near` | UI form | `MatchJob` created on disk |

**Files involved:**
- `web_ui/app.py` — HTTP routes
- `web_ui/templates.py` — UI markup
- `backend/production_jobs.py` — `MatchJob` dataclass and persistence

---

## Phase B — Initial Job Pipeline

Entry point: `run_initial_job_pipeline()` in `backend/production_pipeline.py`.
Five sequential steps; all must succeed for the job to reach review state.

### Step 1/5 — `trim_input`

Cut the raw video from `trim_start_sec` to end using FFmpeg with NVIDIA hardware
encoding (`h264_nvenc`).

| Property | Value |
|----------|-------|
| Function | `trim_input_video()` in `backend/production_pipeline.py` |
| Input | `raw_video_path`, `trim_start_sec` |
| Output artifact | `working_video.mp4` in job directory |
| GPU | Required (`h264_nvenc`) |
| Skip condition | `trim_start_sec <= 0.0001` → just copies the raw file |

### Step 2/5 — `identify_players`

Detects the **table ROI** and runs a face-DB scan to resolve NEAR and FAR player
identity. The table-ROI detector runs **unconditionally** (even when face
identification is skipped) so that Step 3 can reuse the ROI without detecting it
again on the same video.

| Property | Value |
|----------|-------|
| Functions | `detect_table_roi_and_player_zone()` + `quick_identify_players_standalone()` in `backend/player_identification.py` |
| Input | `working_video.mp4`, `pose_weights_path`, `table_weights_path`, `FaceDB` |
| Output | `TableROI` (passed to Step 3) + updated `player_a_name`, `player_b_name` on the MatchJob |
| Models | YOLOv8x-table + YOLOv8x-pose + ArcFace (`w600k_r50.onnx`) |
| GPU | Required (CUDA enforced for both ONNX and torch) |
| Skip condition | Face scan is skipped if both names are already provided; table-ROI detection always runs |

**What it does internally:**
1. Detect table ROI (YOLOv8x-table) — runs once, shared with Step 3
2. Derive player zone = table bbox expanded X +30%, Y +110% on each side
3. Scan early window for FAR player (rank=1) — filter by player zone
4. Scan early set-1 window for NEAR player (rank=0) as the primary pass
5. If needed, run a post-swap rescue scan for Player 2 on rank=1 using clean chunk voting
6. If identity is still unresolved, leave it as unknown for operator input/enrollment later

**Note:** Uses display crop (simple resize 224→112) for embedding to match the
enrollment method (Session 5 fix).

### Step 3/5 — `detect_rallies`

The slowest step. Loads YOLO models once, extracts multi-stream energy signals,
applies hysteresis segmentation, refines endpoints, and samples player positions.

| Property | Value |
|----------|-------|
| Functions | `build_rally_timeline()` in `scripts/generate_rally_timeline.py` + `populate_player_positions()` in `backend/set_boundary.py` |
| Input | `working_video.mp4`, YOLO table + pose weights, `best_of` |
| Output artifact | `timeline.json` (RallyTimeline) |
| GPU | Required (NVDEC for video decode + CUDA for YOLO) |
| Default mode | `player` (YOLO pose wrist velocity as primary signal) |

**Reuses the Table ROI detected in Step 2** — avoids running YOLOv8x-table a
second time on the same video. If Step 2 failed to produce a ROI (rare), Step 3
falls back to detecting the table ROI itself.

**What it does internally (a single conceptual step — four sub-phases):**

1. **Extract multi-stream signals** (`extract_multistream_signals`): decode the
   video once on GPU (NVDEC) and produce three parallel energy signals:
   - **Table energy** — frame-to-frame motion inside the table ROI
   - **Ball energy** — ball candidate motion in an expanded ROI
   - **Player energy** — YOLO pose wrist velocity per side (default:
     `role_tracker` source; alternative: `nearest_two`)
   - **Fused** — `max(table, player * gain_p, ball * gain_b)`

2. **Segment rallies** (`detect_multistream_rallies`): hysteresis state machine
   on the chosen energy signal.
   - Smoothing: 1D Gaussian (kernel=11, σ=3.0) on GPU
   - Normalization: 10th–95th percentile clipping
   - Player-mode thresholds: `high=0.22`, `low=0.09`, `max_gap=1.35s`
   - Post-processing: split long segments on dips, merge artifact runs

3. **Build rally points** (`_build_points_with_active_windows`): convert segments
   into `RallyTimelinePoint` objects, tag let-rallies, attach service-attempt
   indices and `active_start` / `active_end` windows.

4. **Refine endpoints** (`_refine_points_with_endpoint_signals`): tighten each
   rally's `t_end` using 12 support series (action_a/b, exchange, terminal body
   language, dead reset, ball-only false tail, etc.). Records `endpoint_mode`
   and `endpoint_confidence` on each point.

5. **Sample player positions** (`populate_player_positions`): sample a few frames
   per rally, run YOLO pose on the top-2 bodies, and store mean X positions on
   each rally point. Used by **Signal 3** (side-swap detection) of the
   set-boundary algorithm in Step 5.

> **Removed (Session 6):** An older signal-based winner inference layer
> (`_annotate_points_with_winner_fusion_v2`) used to run here. It was removed
> because Step 5's Qwen3-VL adapter overwrites every winner field it produced.
> If a fallback is ever needed when the adapter fails on a specific clip,
> implement a per-clip try/except in Step 5 rather than re-introducing this
> layer.

### Step 4/5 — `export_clips`

Cut the working video into per-rally MP4 clips in parallel.

| Property | Value |
|----------|-------|
| Function | `export_review_clips()` in `backend/production_pipeline.py` |
| Input | `working_video.mp4` + `RallyTimeline` |
| Output artifact | `review_clips/{point_id}.mp4` (one per rally) |
| Encoder | `libx264 -preset veryfast` (CPU, 8 parallel workers) |
| Why CPU | NVENC is reserved for the trim step; parallel CPU is faster for many small clips |

### Step 5/5 — `predict_winners`

Run a Qwen3-VL base model with a LoRA adapter (PEFT) on each scoring rally clip,
then apply set numbers and validate the score.

| Property | Value |
|----------|-------|
| Class | `WinnerAdapterPredictor` in `backend/production_pipeline.py` |
| Base model | `models/Qwen3-VL-4B-Instruct` |
| Adapter | `models/adapters/qwen3vl4b_table_tennis_pilot_4ep_cache_v2/checkpoint-108` |
| Input | One MP4 clip per scoring rally |
| Output artifacts | Predictions JSONL + updated `timeline.json` |
| GPU | Required; prefers `bfloat16` on CUDA |

**What it does internally (three sub-phases — all small post-processing on the timeline):**

1. **Run adapter inference**: predict `{winner, loser, taxonomy, last_hitter}`
   for each scoring rally clip.
   - Adapter output is **NEAR/FAR-relative** (trained with `player_a = NEAR`
     in Set 1)
   - `near_player_for_rally()` remaps those labels to actual `player_a` /
     `player_b`, accounting for set-swap and mid-deciding-set swap at score 5

2. **Apply set numbers** (`apply_set_numbers` in `backend/set_boundary.py`):
   assign `set_number` to each rally using three signals:

   | Signal | Source | Strength |
   |--------|--------|----------|
   | 1. Score rule | 11+ points with 2-point lead | Strongest when winners are correct |
   | 2. Inter-rally gap | Gap > 60s between rallies | Independent of winner correctness |
   | 3. Side swap | Player X-position jump (from Step 3 position sampling) | Independent geometric signal |

3. **Validate score** (`build_score_validation` in `backend/score_validation.py`):
   check that the score progression is plausible, set job status to
   `ready_for_final` or `needs_review`.

---

## Phase C — Review (operator)

Function: `review_job_point()` in `backend/production_pipeline.py`.
For each rally the operator can:

| Action | Effect |
|--------|--------|
| `keep` | Accept the AI prediction as-is |
| `set_winner` | Override the winner manually (`player_a` or `player_b`) |
| `mark_let` | Mark this rally as a let — does not count toward score |

After every action the timeline is re-saved and `build_score_validation()` runs
again. Status flips between `needs_review` and `ready_for_final`.

---

## Phase D — Export

### D1 — `render_job_preview`
Quick preview render. Allowed only when at least one rally has a known winner.

| Property | Value |
|----------|-------|
| Function | `render_job_preview()` → `render_scoreboard_video()` |
| Output | `preview.mp4` in job directory |

### D2 — `export_job_final_video`
Final delivery. Allowed only when `final_export_ready=True` (every rally has a
known winner or has been resolved by the operator).

| Property | Value |
|----------|-------|
| Function | `export_job_final_video()` → `render_scoreboard_video()` |
| Output | `outputs/{job_id}__final_scoreboard.mp4` (repo root) |
| Renderer | `backend/rendering.py` |
| Includes | Scoreboard overlay, set scores, rally counter, audio merge |

---

## Key Dependencies

| Dependency | Used by |
|------------|---------|
| FFmpeg + `h264_nvenc` (NVDEC/NVENC) | Step 1 trim, Step 3 video decode, D1/D2 render |
| YOLOv8x table weights | Step 2 table ROI detection (shared with Step 3) |
| YOLOv8x-pose weights | Step 2 face alignment, Step 3 player energy + position sampling |
| ArcFace `w600k_r50.onnx` (onnxruntime-gpu CUDA) | Step 2 face embedding |
| Qwen3-VL-4B-Instruct + LoRA adapter (PEFT) | Step 5 winner prediction |
| Face DB (`data/players/faces.json`) | Step 2 matching |

## GPU Enforcement

CUDA is **required** at multiple points (Session 4 enforcement):

| Component | Check |
|-----------|-------|
| `FaceEmbedder` | Raises `RuntimeError` if `CUDAExecutionProvider` not available |
| `quick_identify_players_standalone` | `torch.cuda.is_available()` check before loading YOLO |
| `populate_player_positions` | Same `torch.cuda` check |
| `build_rally_timeline` | `torch.cuda.is_available()` check at entry |
| `WinnerAdapterPredictor` | Loads model with `device_map="auto"`, prefers `bfloat16` on CUDA |

CUDA DLL resolution on Windows: `os.add_dll_directory(torch_lib_path)` is called
before `import onnxruntime` to make PyTorch's bundled `cublasLt64_12.dll`
discoverable to onnxruntime-gpu (Session 5 fix).

## Side-Swap Detection (CLI helper for Step 3 debug)

Script: `scripts/detect_side_swap.py`. Independent of Step 3 rally detection.
Determines the timestamp at which the two players have physically swapped sides
of the table — i.e. the boundary between Set N and Set N+1 (or the mid-set
swap at total score 5 in a deciding set).

Useful when Step 3 produces a wrong set boundary on multi-set continuous video,
because side-swap is a guaranteed physical event in table tennis (rule of ITTF)
and provides ground truth for splitting the timeline.

### Inputs (relies on Step 1 + Step 2 having run)
- Video file
- Face DB with the two playing identities enrolled (Step 2 invariant: names
  are always known by the time Step 3 runs)
- YOLOv8x-table + YOLOv8x-pose + ArcFace weights

### Algorithm

1. **Detect Table ROI + player zone** (reuses
   `detect_table_roi_and_player_zone()` from `backend/player_identification.py`).
   Compute `table_center_x = roi.x + roi.w / 2`.

2. **Sample frames every `sample_step` seconds** (default 2 s) through the full
   video. Per frame:
   - Run YOLOv8x-pose; filter detections to those whose bbox center lies inside
     the player zone (excludes adjacent-table players)
   - Take the top-2 bodies by area
   - Per body, compute `cx`; classify side as `L` if `cx < table_center_x`
     else `R`
   - Run ArcFace via the display-crop method (matches enrollment); match against
     the face DB; assign identity (player name or `None`)

3. **Auto-select the two main players** as the two identities with the most
   matched samples (do not assume the face DB has exactly two records).

4. **Per-identity side timeline**: list of `(t, side)` for each player.

5. **Smoothed side at time `t`**: dominant side in a `±window` second sliding
   window (default `window = 10 s`); requires `min_samples = 2` and
   `min_majority_frac = 0.6`. Returns `None` when the data is sparse or evenly
   split (transitioning).

6. **Baseline (Set 1) sides**: dominant side per player in `[baseline_start,
   baseline_end]` (default 10–60 s). The two baseline sides must be opposite;
   if not, abort.

7. **Search for swap** (walk forward from `baseline_end`):

   At each timestamp `t`, classify the candidate as one of:

   | Mode | Condition |
   |------|-----------|
   | `both`   | Both players' smoothed sides observed flipped |
   | `a-only` | Player A clearly flipped; Player B has no contrary evidence (sparse data, e.g. back-to-camera) — accept by symmetry: two players physically must be on opposite sides |
   | `b-only` | Mirror of `a-only` |

   Then verify **stability**: state must not return to the baseline state for
   the next `stability_seconds` (default 15 s). The first stable candidate
   wins.

8. **Backtrack** from the swap-detected timestamp: walk backward in `step`
   increments; continue past `None` (no data) but stop on any explicit return
   to the baseline state. The earliest non-baseline timestamp is
   `T_swap_start`.

### Output
- `T_swap_start` (seconds): the timestamp from which the post-swap state begins
- Set 1 timeline range: `t < T_swap_start`
- Set 2 timeline range: `t >= T_swap_start`
- The algorithm reports the swap window `[T_last_set1, T_swap_start]` so the
  operator can verify visually

### Verified result on `inputs/raw_matches/2_sets.mp4` (`2026-04-15`)
- Detected `T_swap_start = 172.00 s` — confirmed correct by operator
- Mode used: `a-only` (player B was facing away from the camera too often for
  reliable face matching during the transition window)
- Step 3 rally detection on the same input is still buggy (24 rallies, wrong
  per-set counts — see Known Issues), but this script gives a trustworthy
  set boundary independent of rally detection

### CLI usage
```
python scripts/detect_side_swap.py --video <path> [--sample-step 2.0] [--smooth-window 10.0] [--stability-seconds 15.0]
```

---

## Internal `current_step` Values

The `MatchJob.current_step` field uses machine-readable values. UI progress
tracking in `web_ui/progress.py` and the staged pipeline in
`backend/production_pipeline.py` depend on these exact strings.

### Staged pipeline (current — debug-first GUI with operator pauses)

```
run_pipeline_stage_trim_and_detect_sets()
  ├─ trim_input          (running)       — trim video (ffmpeg nvenc)
  ├─ detect_sets         (running)       — side-swap detection (reuse Table ROI)
  └─ confirm_sets        (awaiting_confirmation)  ← PAUSE

run_pipeline_stage_detect_rallies()
  ├─ detect_rallies      (running)       — per-set clip cut + rally detection
  └─ confirm_rallies     (awaiting_confirmation)  ← PAUSE

run_pipeline_stage_predict()
  ├─ export_review_clips          (running)  — cut per-rally clips
  ├─ predict_winners_with_adapter (running)  — Qwen3-VL inference
  └─ ai_ready                     (needs_review / ready_for_final)
```

| `current_step` | `status` | Stage | Action |
|----------------|----------|-------|--------|
| `"trim_input"` | `running` | 1 | Trim video |
| `"detect_sets"` | `running` | 1 | Side-swap detection |
| `"confirm_sets"` | `awaiting_confirmation` | — | **PAUSE**: confirm set count + swap times |
| `"detect_rallies"` | `running` | 2 | Per-set rally detection |
| `"confirm_rallies"` | `awaiting_confirmation` | — | **PAUSE**: confirm per-set rally counts |
| `"export_review_clips"` | `running` | 3 | Cut per-rally clips |
| `"predict_winners_with_adapter"` | `running` | 3 | Qwen3-VL winner prediction |
| `"ai_ready"` | `needs_review` | — | Pipeline done, enter review |
| `"review_updated"` | `needs_review` / `ready_for_final` | — | After each review action |

### Legacy pipeline (old — runs A-to-Z without pauses)

Still available via `run_initial_job_pipeline()` and `POST /jobs/<id>/run`.

| `current_step` | Step |
|----------------|------|
| `"trim_input"` | Step 1/5 |
| `"player_identification"` | Step 2/5 |
| `"generate_rally_timeline"` | Step 3/5 |
| `"export_review_clips"` | Step 4/5 |
| `"predict_winners_with_adapter"` | Step 5/5 |
| `"ai_ready"` | Done |

### Export + other (shared by both flows)

| `current_step` | `status` |
|----------------|----------|
| `"final_export"` | `running` |
| `"final_export_complete"` | `completed` |
| `"preview_ready"` | varies |
| `"preview_skipped_no_known_winner"` | `needs_review` |
| `"failed"` | `failed` |

Do not rename `current_step` values without also updating `web_ui/progress.py`
and any saved job files that reference them.

## Known Issues

See `PROJECT_PROGRESS.md` for current known bugs:
- **Multi-set continuous video** - rally detection (Step 3/5) fails when a
  single input contains multiple sets concatenated. Each set in isolation works
  correctly. Likely cause: energy normalization or hysteresis thresholds do not
  cope with the 60–120s inter-set break.
- **Current failure signature on `inputs/raw_matches/2_sets.mp4` (`2026-04-15`)**
  - Command run: `python scripts/debug_set_boundaries.py --video inputs/raw_matches/2_sets.mp4 --best-of 3 --trim 0`
  - Current code output:
    - total rallies detected = `24`
    - set 1 = `9` rallies
    - set 2 = `15` rallies
    - detected side-swap boundary = index `9` (`pt_0009 -> pt_0010`)
    - detected swap window = about `200.67s -> 204.44s`
  - Operator verdict: all three are wrong for this input
    - wrong set-1 rally count
    - wrong set-2 rally count
    - wrong swap timing
  - Practical meaning: do not treat `2_sets.mp4` set counts or swap timing as
    trustworthy until Step 3 rally detection is fixed.
