# Current Pipeline Implementation

## Purpose
Snapshot of the pipeline contract and the current implementation shape. Update
this file whenever the pipeline structure changes (new step added, step removed,
signal source swapped, model replaced, etc.).

This file describes the agreed active pipeline direction. If code has not caught
up yet, mark that gap explicitly instead of hiding it:
- `ROADMAP_PRODUCTION.md` = long-term target (what we want to be)
- `PIPELINE.md` = active pipeline contract + implementation snapshot ← **this file**
- `PROJECT_PROGRESS.md` = daily work log (how we got here)
- `PROJECT_ACTION_PLAN.md` = operational board (what's next)

## Top-level Flow

The Web UI direction is now **debug-mode first**, not a single black-box
production run. The pipeline must stop at several checkpoints so the operator
can verify the output before the next step consumes it.

```
A. Setup (Web UI)
   Select raw video + enter Trim Start
    |
B. Identify Players button
   Step 1: Trim input video                         -> auto, no pause
   Step 2: Identify players                         -> PAUSE / confirm or enroll
    |
C. Run AI Pipeline button
   Step 3: Detect rallies + side/set state          -> auto, no operator timeline preview
   Step 4: Predict winners                          -> auto, then hand off to review
   Step 5: GUI confirm winners                      -> first operator preview/review loop
   Step 6: Final output video with scoreboard       -> completed artifact
```

Main rule: the GUI flow starts from **Identify Players**, not from **Run AI
Pipeline**. The Identify step first trims the raw video, then identifies players
on the trimmed working video. After the operator confirms player names, **Run AI
Pipeline** starts from Step 3 only.

---

## Phase A — Setup (Web UI)

| Step | Action | Input | Output |
|------|--------|-------|--------|
| A1 | Browse and select raw video | filesystem | `raw_video_path` |
| A2 | Enter `trim_start_sec` | UI form | Trim timestamp used by Step 1 |
| A3 | Click `Identify Players` | `raw_video_path`, `trim_start_sec`, face DB | Trimmed working video + detected player names |
| A4 | Confirm/enroll/edit players and fill setup fields | UI form | `MatchJob` created on disk when Run AI Pipeline is clicked |

Player recognition is not just a setup shortcut. The `Identify Players` button
runs Step 1 and Step 2 before a match job starts the heavier AI pipeline.

**Files involved:**
- `web_ui/app.py` — HTTP routes
- `web_ui/templates.py` — UI markup
- `backend/production_jobs.py` — `MatchJob` dataclass and persistence

---

## Phase B — Debug-Mode Job Pipeline

This is the agreed Web UI pipeline contract. It is intentionally more
interactive than the old production-style `run everything` flow.

### Step 1/6 — `trim_input`

Cut the raw video from `trim_start_sec` to the end. This creates the working
video that all later steps consume.

| Property | Value |
|----------|-------|
| Function | `trim_input_video()` in `backend/production_pipeline.py` |
| Input | `raw_video_path`, `trim_start_sec` |
| Output artifact | `working_video.mp4` in job directory |
| GPU | Required. Uses an NVENC smoke check first, then CUDA hardware decode + NVIDIA NVENC encode (`h264_nvenc`) when trimming |
| Skip condition | `trim_start_sec <= 0.0001` -> copy the raw file only after the NVENC smoke check passes |
| UI checkpoint | None. Step 1 runs automatically when `Identify Players` is clicked and continues to Step 2 |

Current FFmpeg trim path:

```text
ffmpeg -y -hwaccel cuda -hwaccel_output_format cuda -ss <trim_start_sec> -i <raw_video_path> -map 0:v:0 -map 0:a? -c:v h264_nvenc -preset p1 -c:a copy -movflags +faststart <working_video_path>
```

If CUDA/NVENC is not available, Step 1 fails and the pipeline stops. There is
no CPU fallback for this tool.

### Step 2/6 — `identify_players`

Detect the table ROI and identify the two players. This step is a first-class
debug checkpoint because wrong identities poison winner prediction, review
labels, and scoreboard names downstream.

| Property | Value |
|----------|-------|
| Functions | `detect_table_roi_and_player_zone()` + `quick_identify_players_standalone()` in `backend/player_identification.py` |
| Input | trimmed `working_video.mp4`, `pose_weights_path`, `table_weights_path`, `FaceDB` |
| Output artifacts | `TableROI`, player-zone diagnostics, `player_a_name`, `player_b_name`, face crops for unknowns |
| Models | YOLOv8x-table + YOLOv8x-pose + ArcFace (`w600k_r50.onnx`) |
| GPU | Required (CUDA enforced for both ONNX and torch) |
| UI checkpoint | Show detected players, confidence/evidence, face crops, and table/player-zone diagnostics |

**Rules:**
1. Detect table ROI once and reuse it in Step 3.
2. Derive player zone from the table ROI to exclude adjacent-table players.
3. If a player is unresolved, keep the result as `unknown`.
4. Never assign a player by guessing or by using a "only other DB person" fallback.
5. The operator can confirm, manually name, or enroll a new player before Step 3.
6. `Run AI Pipeline` must reuse the already-trimmed working video and start from Step 3.

**Note:** Uses display crop (simple resize 224→112) for embedding to match the
enrollment method.

### Step 3/6 — `detect_rallies`

Detect set boundaries and the rally timeline. In the current Web UI code,
Step 3 is not one continuous black-box step. It is split into two staged
sub-steps with two operator pauses:

```text
Run AI Pipeline
  -> Step 3.1 detect_sets
  -> PAUSE confirm_sets
  -> Step 3.2 detect_rallies
  -> PAUSE confirm_rallies
  -> Step 4 predict_winners
```

| Property | Value |
|----------|-------|
| Stage entrypoints | `run_pipeline_stage_detect_sets()` + `run_pipeline_stage_detect_rallies()` in `backend/production_pipeline.py` |
| Rally detector | `build_rally_timeline()` in `scripts/generate_rally_timeline.py` |
| Input | `working_video.mp4`, confirmed players, `TableROI`, `best_of` |
| Output artifacts | `set_clips/setN.mp4`, merged `timeline.json`, `timeline_summary.detected_sets`, `timeline_summary.per_set_rallies` |
| GPU | Required (NVDEC for video decode + CUDA for YOLO) |
| Default mode | `player` (YOLO pose wrist velocity as primary signal) |
| UI checkpoints | Pause 1: confirm total rally/LET start-times. Later pauses TBD |

Current debug warning:
- Step 3.1 / Step 3.2 / Step 3.3 are **not good enough yet** on `2_sets.mp4`.
- The canonical `summary.md` format is still rough and needs more iteration.
- Step 3.3 correctly blocks invalid output, but the upstream rally/side-state
  data still has many failures. Treat all current Step 3 results as debugging
  artifacts, not production-ready output.

#### Step 3.1 — `detect_total_rallies`

Code path: `run_pipeline_stage_detect_sets()` orchestrates the stage, while the
Step 3.1 rally/LET/server review logic lives in
`backend/step3_rally_start_review.py`.
Runtime state while running: `status="running"`, `current_step="detect_total_rallies"`.

Current sub-steps:
1. Load `working_video.mp4` prepared by Step 1 + Step 2.
2. Reuse `job.timeline_summary["table_roi"]` from Identify Players if present.
3. If no cached ROI exists, call `detect_table_roi_and_player_zone()` to detect the table again.
4. If table ROI is missing or invalid, store an error and pause at `confirm_total_rallies` instead of continuing blindly.
5. Run the existing, already-debugged start-time detector on the full working video via `build_rally_timeline()`. Do not reimplement rally start-time, endpoint, or LET detection in Step 3.1.
6. Save the raw detector timeline to `step3_1_total_rally_timeline.json`.
7. Build one chronological review list from both sources:
   `timeline.points` for scoring rallies, plus `analysis_metadata["excluded_let_starts"]` and `analysis_metadata["unattached_trailing_let_starts"]` for LET/non-scoring rallies.
8. Keep the initial Step 2 name map as provisional debug context only: role `A` = initial near-side player, role `B` = initial far-side player. This mapping is **not enough after a side swap** and must not be treated as current NEAR/FAR.
9. Apply the existing serve-order engine (`_infer_player_serve_mode_from_starter_roles`) as a review guard. If double-serve order shows a singleton scoring run between two complete runs from the other player, add a `needs_review` marker in the gap instead of silently accepting the missing start.
10. Count review rows as `scoring + LET/non-scoring + needs_review`. Confirmed detector starts remain available as `detected_total`.
11. Export one annotated JPG per detected/review start-time into `step3_1_rally_start_frames/`.
12. Do **not** run player side-state identification here. Step 3.1 is detector-only: total rally/LET start-times plus serve-order review markers.
13. Export `rally_start_times.csv` in the same folder, including `starter_role` for debugging and initial Step 2 server mapping only.
14. Export the merged start-time event list to `step3_1_rally_start_events.json`.
15. Save summary and events into `job.timeline_summary["detected_total_rallies"]`.
16. Pause for operator review at `status="awaiting_confirmation"`, `current_step="confirm_total_rallies"`.

Rule-driven repair loop for Step 3.1:
1. First pass: scan the full working video once to get the total chronological list of scoring rallies and LET/non-scoring starts.
2. Rule audit: compare the output against basic table-tennis rules, especially the 2-serve order and the fact that LET does not advance service.
3. Targeted rescan only: when the rule audit finds a suspicious gap or conflict, rescan only that `gap_start -> gap_end` window, with small padding if needed. Do not rescan the whole clip just to repair one suspected missing rally.
4. Finalize Step 3.1: update the same Step 3.1 summary/CSV/JSON with recovered candidates or explicit `needs_review` markers, then pause again for operator review.

Current command-line support for the targeted repair loop:
- Full first pass: `scripts/step3_1_rally_start_review.py --video ... --start ... --end ...`
- Targeted repair only after an existing first pass: add `--rescan-only --rescan-review-gaps`.
- The rescan window source must come from rule-audit fields such as `gap_start`, `gap_end`, `source_gap_start`, and `source_gap_end`; never scan unrelated time ranges.
- Step 3.1 side-id is off by default. For temporary combined debug only, use `--with-side-id`; normal pipeline should run Step 3.2 separately.

Operator feedback from `2_sets.mp4` full summary:
- `rally_0013` in the first segment is accepted by the operator.
- From `rally_0016` onward in the combined report, the displayed player names are suspected wrong because the players have already swapped sides.
- Therefore any user-facing Step 3.1/Step 3.2 report must include a separate `current_side` column (`NEAR`, `FAR`, or `unknown`) for each rally start.
- `A/B` is a detector/debug role only. It must not be used as a proxy for current side after a swap.
- `current_side` must be detected independently at each rally start time using player identification / side evidence around that timestamp. Do **not** infer `current_side` by first detecting one side-swap timestamp and then globally remapping the rest of the video.
- The existing side-swap detector can be used later for set-boundary logic, but it is not the source of truth for the per-rally `NEAR/FAR` column in this Step 3.1/3.2 review report.
- Whenever the pipeline rescans or re-identifies players around a rally/gap, it must also detect which side that player is currently standing on at that exact timestamp. Identity without side is insufficient for multi-set clips.
- Per-rally side detection does **not** need to identify both players. It is enough to identify one of the two trusted Step 2 players at that rally start, determine whether that identified player is currently `NEAR` or `FAR`, then infer the other player's side by exclusion because exactly two players are in the match.
- The side-id scan must be spatially constrained to the main match player zone only: `table ROI + 25% table width on X + 110% table height on Y`. Do not scan the full frame for faces/players, because adjacent tables and audience faces can create false matches.
- During this phase, identity matching is restricted to the two trusted Step 2 player names. The matcher must not choose arbitrary FaceDB players.
- Side-id must be conservative: a face result is accepted only when it has enough accepted samples and similarity quality. Weak evidence must become `unknown/needs_review`, not a guessed player/side.
- Side-id must be start-anchored. First scan only around `start_time`. If that is weak, post-rally extension is allowed only when the next rally starts soon enough that the gap is not a likely set break / side swap. A long gap after the rally blocks `t_end + N` evidence, because that evidence belongs to players walking/swapping sides, not to the rally start.
- Jersey/color evidence is auxiliary by default. Do not use jersey-only fallback as source of truth unless explicitly enabled for a match with clearly distinct shirts, because similar red/yellow jerseys can be confidently wrong.
- Until side state is applied, post-swap player-name mapping should be considered provisional and must be review-marked instead of silently trusted.

Single entrypoint rule:
- GUI and command-line debug must use the same Step 3.1 engine in `backend/step3_rally_start_review.py`.
- The command-line wrapper is `scripts/step3_1_rally_start_review.py`; it should differ from GUI only by arguments/input source, not by detector logic.

Explicit non-goals for Step 3.1:
- Do not split Set 1 / Set 2.
- Do not finalize side swap or set split inside the first-pass detector.
- Do not claim current NEAR/FAR from initial `A/B`; current side must come from a dedicated side-state scan.
- Do not predict winners.
- Do not write `timeline_review.json` yet, because that would push the GUI into winner-review mode too early.

LET source-of-truth:
- Use the existing player-path LET logic in `backend/ai_multistream_rally.py`, especially `_detect_player_sandwich_rallies_from_diagnostics()`, `_infer_forced_let_indices_from_starter_roles()`, and `_repair_double_serve_role_singletons()`.
- `scripts/generate_rally_timeline.py` deliberately excludes LET segments from `timeline.points` and stores them in `analysis_metadata["excluded_let_starts"]` or `analysis_metadata["unattached_trailing_let_starts"]`.
- Step 3.1 must only merge those existing detector outputs for operator review. It must not introduce a new LET classifier or relabel LET from scratch.

Pause after Step 3.1:

| Pause | Runtime state | GUI shows | Next click does |
|-------|---------------|-----------|-----------------|
| `confirm_total_rallies` | `status="awaiting_confirmation"`, `current_step="confirm_total_rallies"` | Total scoring/LET count, exported start-time images, CSV/JSON paths | Click Next to run Step 3.2 side-state detection |

After Step 3.1 finishes, the pipeline should continue into Step 3.2/3.3 automatically in production flow. Command-line debug may still inspect Step 3.1 artifacts while the algorithm is being developed.

#### Step 3.2 — `detect_side_state`

Step 3.2 is the next required sub-step after the Step 3.1 start-time review.
It must resolve side state and set split before winner prediction consumes the
timeline.

Required Step 3.2 outputs:
1. For each Step 3.1 rally start, run local identity/side evidence around that start timestamp.
2. Detect at least one trusted Step 2 player in that local window when possible; if one player is identified with side `NEAR`/`FAR`, infer the other player's side by exclusion.
3. Add `current_side` for the server/player row: `NEAR`, `FAR`, or `unknown`.
4. Add `side_evidence_source`, for example `single_player_face_id_at_start`, `single_player_body_id_at_start`, or `unknown`.
5. Re-map server player names using per-rally identity-side evidence, not a global side-swap remap and not initial `A/B`.
6. Keep any post-swap identity/side ambiguity as `unknown` or `needs_review`; do not guess.
7. Update the human summary table to show:
   `id | kind | start_time | server | current_side | note | image`.
8. Keep `starter_role` available in CSV/JSON for debugging, but hide or de-emphasize it in the human-facing summary.
9. Human-facing Markdown reports must show only timestamps in the final input
   video's timeline (`source_t_start` / `source_t_end`). Segment/debug clip time
   exists only to speed up internal scans and must stay out of Markdown tables
   to avoid confusing operator review.
10. Step 3.2 creates the canonical Step 3 human summary. Step 3.3 must update
    that same Markdown summary after audit/rescan instead of creating a second
    user-facing report. JSON/CSV files may remain separate technical artifacts.

Step 3.2 two-pass scan:
1. Fast pass (`start_anchor`): scan a short window around `start_time` with the standard FPS/thresholds.
2. Safe extension (`safe_post_rally_extension`): if fast pass is weak, extend only when the next rally starts soon enough that the gap is not a likely break/side-swap. Never scan into the next rally.
3. Retry pass (`unknown_retry_start_anchor`): only for rows still `unknown`, scan a longer start-anchored window at higher FPS (default `12s`, `8 FPS`, min `3` accepted face samples). This retry remains break-safe: if the rally is followed by a long gap / side-swap candidate, the retry is capped at the rally end instead of using post-rally evidence.
4. Strong-candidate promotion: if a row only missed the sample-count threshold but has 3 strong face samples (`best/avg/margin` all high), promote it instead of leaving it unknown.
5. Continuity fill (`side_continuity_fill`): if direct scans still leave `unknown`, fill only when neighboring known side maps are consistent, or when a terminal unknown run has no long gap that could hide a side swap. These rows are marked `inferred`, not face-identified.
6. Goal: reduce unknowns as much as possible before Step 3.3. Guardrail remains unchanged: if scan evidence and continuity are both unsafe, keep `unknown`; do not guess.

Production rule: Step 3 must not rely on operator preview/confirmation. It must produce a complete, internally consistent rally timeline before Step 4 starts. If Step 3 cannot resolve a required field safely, it should fail/pause with an explicit error for developer debugging, not ask the operator to manually approve a partial timeline.

Implementation note:
- Step 3.2 is now a separate code path: `build_step3_2_side_state_review()` in `backend/step3_rally_start_review.py`.
- CLI wrapper: `scripts/step3_2_side_state_review.py --video ... --events-json ...`.
- GUI review for Step 3 artifacts is debug-only while developing. In the intended production/debug-mode flow requested by the operator, the first user-facing preview is Step 5 after winner prediction.
- `current_side`/server remap must come from per-rally evidence. Do not reintroduce `--side-swap-cutoff` or any global cutoff-based remap for this report.
- Side evidence uses face evidence first. Jersey-anchor evidence may be exported for debugging, but jersey-only identity must not override weak/missing face evidence by default.

Known failure fixed from `2_sets.mp4`:
- `rally_0016` at `02:52.188` was incorrectly reported as `Nguyễn Bá Thảo=FAR` because the previous rule accepted only 2 weak face samples (`best_sim≈0.413`) from a short window. Correct human-verified side is `FAR=Trần Quang Vinh`, `NEAR=Nguyễn Bá Thảo`.
- Fix: constrain player ROI to `table ROI +25% X/+110% Y`, widen the local scan through `t_end + 4s`, require at least 4 accepted face samples, require stronger best/average similarity, and leave weak rows as `unknown/needs_review` instead of filling a wrong name.
- `rally_0015` at `02:23.176` exposed the opposite failure mode: full batch scanned through `t_end + 4s`, crossing the confirmed side-swap break starting around `02:28`. Correct start-time side is `FAR=Nguyễn Bá Thảo`, `NEAR=Trần Quang Vinh`; post-rally evidence after the set-ending rally must not be assigned backward to the rally start.
- Fix: side-id now uses `start_anchor` first and blocks `safe_post_rally_extension` whenever the next-start gap is long (`side-id-break-gap-sec`, default `12s`) or the event is terminal. Extension is also guarded so it never scans into the next rally.

#### Step 3.3 — `logic_audit_and_repair`

Step 3.3 consumes the Step 3.2 `summary.md` / events JSON and validates whether
the side-state/server timeline is internally legal enough to be trusted by Step
4. This is a code-owned audit step, not an operator preview step.

Core rule: Step 3.3 audits by **real server identity** (`server_player_key/name`)
after Step 3.2 has mapped `starter_role` + `current_side`. Do not audit by raw
`A/B` alone, because `A/B` is only a tracker side role and changes meaning after
players swap sides.

Required Step 3.3 behavior:
1. Read Step 3.2 events, including `kind`, `source_t_start`, `server_player_key`, `current_side`, and `side_evidence_status`.
2. Apply basic table-tennis service rules:
   - Scoring rallies advance the service count.
   - LET/non-scoring rows do **not** advance the service count.
   - Before deuce, service should run in two scoring points per player.
   - After the deuce threshold is reached, service switches every point.
3. Treat likely set boundaries as audit resets only when there is enough evidence: long gap, side-state swap, and at least the minimum legal scoring count before the boundary.
4. If a player appears to serve only once in the middle of a normal two-serve block, or the other player appears to serve three times, mark the local events as suspicious.
5. Recursively call the Step 3.2 side-state scan only on suspicious event IDs / neighboring rows. Do **not** rescan the whole video.
6. Re-run the Step 3.3 audit after each targeted Step 3.2 repair pass until the timeline is legal or the repair limit is reached.
7. If the remaining issue indicates a missing rally start rather than a side-state error, mark it as requiring Step 3.1 targeted gap-rescan. Step 3.2 must not invent new rally starts.
8. Step 3.3 is a hard gate. Only when `logic_ok=true` and `blocking_issue_count=0`
   can the pipeline continue to Step 3.4. If any blocking issue remains, the
   pipeline must stop here and keep repairing/debugging Step 3.2/Step 3.1
   inputs. Do not continue to winner prediction with a logically invalid
   summary.
9. When Step 3.3 finishes cleanly, its output summary/events are the trusted Step 3 timeline source for Step 3.4.
10. The user-facing result of Step 3.3 is an updated Step 3.2 summary file:
    one Markdown table for the whole Step 3 state, including total rally/LET
    counts, start time, server, `server_side`, opponent, `opponent_side`, and
    logic gate status. Do not repeat the same player name multiple times in one
    row just to show both `server` and full NEAR/FAR assignment.

Implementation note:
- Backend entrypoint: `build_step3_3_logic_audit_review()` in `backend/step3_rally_start_review.py`.
- CLI wrapper: `scripts/step3_3_logic_audit.py --video ... --events-json <step3_2_events.json>`.
- Audit function: `audit_step3_side_state_logic()`.
- The final report uses dedicated `step3_3_*` fields so old Step 3.1/3.2 debug flags do not become the source of truth.

#### Step 3.4 — `detect_set_boundaries_side_swaps`

Step 3.4 is allowed to run **only after Step 3.3 passes 100%**. Its job is to
detect set boundaries / side swaps from the audited Step 3.3 events, assign set
metadata, then produce the final rally timeline consumed by Step 4 winner
prediction.

Gate rule:
- If Step 3.3 has `logic_ok=false` or `blocking_issue_count > 0`, Step 3.4 must
  not run.
- Do not silently bypass this gate.
- Do not ask the operator to approve an invalid Step 3 timeline.
- Keep the pipeline blocked at Step 3.3 until targeted Step 3.2/Step 3.1 repair
  produces a logically valid summary.

Required Step 3.4 behavior:
1. Consume only the audited Step 3.3 events where `logic_ok=true`.
2. Detect set boundaries from verified side-state transitions, service-order
   resets, legal set-completion constraints, and long break gaps.
3. Detect side swaps after every completed set.
4. Detect deciding-set side swap when the total score in the deciding set reaches 5:
   set 3 of BO3, set 5 of BO5, or set 7 of BO7.
5. Assign `set_number`, `set_rally_index`, `set_score_index`, and current
   `NEAR/FAR` side state to each scoring/LET row.
6. Export a final Step 3 timeline JSON that Step 4 can consume directly.
7. If set boundary / side-swap evidence is inconsistent, stop at Step 3.4 with
   an explicit debug error. Do not guess and do not continue to Step 4.

Implementation status:
- Not implemented yet.
- This is the next missing Step 3 sub-step before Step 4.

No Step 3 GUI checkpoint:
- There is no `GUI confirm final rally timeline` step.
- The operator does not preview the rally timeline before winner prediction.
- Any Step 3 report/summary is for developer debugging only.
- The first operator-facing preview/review is Step 5, after Step 4 winner prediction has already run.

### Step 4/6 — `predict_winners`

Generate AI winner predictions for detected scoring rallies. Clip export is an
internal sub-step here; it is not a separate top-level Web UI step.

| Property | Value |
|----------|-------|
| Clip function | `export_review_clips()` in `backend/production_pipeline.py` |
| Predictor class | `WinnerAdapterPredictor` in `backend/production_pipeline.py` |
| Base model | `models/Qwen3-VL-4B-Instruct` |
| Adapter | `models/adapters/qwen3vl4b_table_tennis_pilot_4ep_cache_v2/checkpoint-108` |
| Input | Final Step 3 `timeline.json` + one exported MP4 clip per scoring rally |
| Output artifacts | `review_clips/{point_id}.mp4`, predictions JSONL, updated `timeline.json` |
| GPU | Required. Review clips use CUDA/NVENC, and Qwen3-VL is pinned to `cuda:0` with `bfloat16`; no `device_map="auto"` CPU offload |
| UI checkpoint | Show prediction summary: known/review/unknown counts and low-confidence examples |

**What it does internally:**
1. Cut one review clip per scoring rally using FFmpeg CUDA/NVENC.
2. Run adapter inference to predict `{winner, loser, taxonomy, last_hitter}`.
3. Remap position-relative labels to actual player names using set-side state.
4. Apply set numbers and score validation after predictions are attached.

### Step 5/6 — `confirm_winners_in_gui`

The operator reviews AI predictions and resolves every scoring rally needed for
the final scoreboard.

| Property | Value |
|----------|-------|
| Function | `review_job_point()` in `backend/production_pipeline.py` |
| Input | Predicted `timeline.json`, review clips, player names |
| Output artifact | Reviewed/resolved `timeline.json` |
| UI checkpoint | Review loop until every scoring rally is accepted, corrected, marked let, or intentionally blocked |

For each rally the operator can:

| Action | Effect |
|--------|--------|
| `keep` | Accept the AI prediction as-is |
| `set_winner` | Override the winner manually (`player_a` or `player_b`) |
| `mark_let` | Mark this rally as a let — does not count toward score |

After every action, the timeline is re-saved and score validation runs again.
Final export remains blocked until all required scoring rallies are resolved.

### Step 6/6 — `final_output_video`

Render the final scoreboard video from the reviewed timeline.

| Property | Value |
|----------|-------|
| Function | `export_job_final_video()` → `render_scoreboard_video()` |
| Input | `working_video.mp4`, reviewed `timeline.json`, player names |
| Output | `outputs/{job_id}__final_scoreboard.mp4` |
| Renderer | `backend/rendering.py` |
| Includes | Scoreboard overlay, set scores, rally counter, audio merge |
| Gate | Allowed only when `final_export_ready=True` |

Preview render may still exist as a convenience tool, but it is not a top-level
pipeline step. The canonical end state is the final scoreboard video.

---

## Key Dependencies

| Dependency | Used by |
|------------|---------|
| FFmpeg + `h264_nvenc` (NVDEC/NVENC) | Step 1 trim, Step 3 video decode, Step 4 clip export, Step 6 render |
| YOLOv8x table weights | Step 2 table ROI detection (shared with Step 3) |
| YOLOv8x-pose weights | Step 2 face alignment, Step 3 player energy + position sampling |
| ArcFace `w600k_r50.onnx` (onnxruntime-gpu CUDA) | Step 2 face embedding |
| Qwen3-VL-4B-Instruct + LoRA adapter (PEFT) | Step 4 winner prediction |
| Face DB (`data/players/faces.json`) | Step 2 matching |

## GPU Enforcement

CUDA is **required** at multiple points (Session 4 enforcement):

Project hardware rule: this tool is optimized for the local NVIDIA RTX 5060 Ti.
Heavy video and AI steps should use GPU acceleration by default. Do not silently
fall back to CPU for GPU-required steps; fail clearly instead.

| Component | Check |
|-----------|-------|
| `FaceEmbedder` | Raises `RuntimeError` if `CUDAExecutionProvider` not available |
| `quick_identify_players_standalone` | `torch.cuda.is_available()` check before loading YOLO |
| `populate_player_positions` | Same `torch.cuda` check |
| `build_rally_timeline` | `torch.cuda.is_available()` check at entry |
| `WinnerAdapterPredictor` | Requires `torch.cuda.is_available()`, sets CUDA device 0, loads `bfloat16` model on `cuda:0` |

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
- Face DB / confirmed player identities when available. If identity evidence is
  insufficient, Step 3 must stop with an explicit developer/debug error instead
  of guessing or asking the operator to approve Step 3 manually.
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
tracking in `web_ui/progress.py` and the debug-mode pipeline in
`backend/production_pipeline.py` should depend on these exact strings.

### Debug-mode pipeline target

These are the canonical step names for the 6-step Web UI flow. Some existing
code still uses legacy names; when implementation catches up, update code and
progress rendering together.

Current implementation note: Step 1 + Step 2 run before `MatchJob` creation via
the Identify Players scan (`scan_id`). Their progress is shown by the scan log
panel, not by `MatchJob.current_step`. The MatchJob created by `Run AI Pipeline`
reuses the trimmed working video and starts from Step 3.

| `current_step` | `status` | Step | Action |
|----------------|----------|------|--------|
| `"trim_input"` | `running` | 1 | Trim video |
| `"identify_players"` | `running` | 2 | Detect table ROI and identify players |
| `"confirm_players"` | `awaiting_confirmation` | 2 | **PAUSE**: confirm/enroll/manual-name players |
| `"detect_rallies"` | `running` | 3 | Detect set/rally timeline |
| `"confirm_rallies"` | `awaiting_confirmation` | 3 | **PAUSE**: confirm set split and per-set rally counts |
| `"predict_winners"` | `running` | 4 | Export review clips internally and run winner model |
| `"confirm_predictions"` | `awaiting_confirmation` | 4 | **PAUSE**: inspect AI prediction summary before review loop |
| `"confirm_winners"` | `needs_review` | 5 | GUI review loop for winner confirmation/correction |
| `"review_updated"` | `needs_review` / `ready_for_final` | 5 | After each review action |
| `"final_export"` | `running` | 6 | Render final scoreboard video |
| `"final_export_complete"` | `completed` | 6 | Final output video is ready |

### Legacy implementation values

The old A-to-Z path may still appear in existing job files or code paths. Treat
these as compatibility names, not the target Web UI design.

| Legacy `current_step` | New conceptual owner |
|-----------------------|----------------------|
| `"player_identification"` | Step 2: `identify_players` |
| `"generate_rally_timeline"` | Step 3: `detect_rallies` |
| `"export_review_clips"` | Internal sub-step of Step 4: `predict_winners` |
| `"predict_winners_with_adapter"` | Internal sub-step of Step 4: `predict_winners` |
| `"ai_ready"` | Step 5: `confirm_winners` |
| `"preview_ready"` | Optional preview helper, not a top-level step |
| `"preview_skipped_no_known_winner"` | Optional preview helper, not a top-level step |
| `"failed"` | Shared terminal state |

Do not rename `current_step` values in code without also updating
`web_ui/progress.py` and any saved job files that reference them.

## Known Issues

See `PROJECT_PROGRESS.md` for current known bugs:
- **Multi-set continuous video** - rally detection (Step 3/6) fails when a
  single input contains multiple sets concatenated. Latest debugging shows the
  issue is not only the continuous-video boundary: per-set clips cut from
  `2_sets.mp4` still produce wrong rally counts. Treat this as a Step 3 rally
  detector accuracy problem.
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
- **Latest split-and-detect result on `2_sets.mp4` (`2026-04-16`)**
  - Side-swap split recovered more rallies than the continuous run.
  - Detected Set 1 = `13` scoring rallies, Set 2 = `17` scoring rallies.
  - Operator verdict: still wrong.
  - Practical meaning: Step 3 must be fixed to produce a correct timeline
    automatically before Step 4 winner prediction.
