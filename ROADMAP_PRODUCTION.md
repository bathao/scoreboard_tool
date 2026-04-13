# Scoreboard Tool - Production Roadmap

## Purpose
This file defines the long-term production target for the project.

Use this file for:
- final goal
- guiding architecture
- non-negotiable design rules
- production promotion criteria

Do not use this file as a daily work log.

## Final Goal
- Input:
  - one raw full-match table-tennis clip, usually `30-40` minutes
  - fixed-tripod camera per clip
  - typical input quality: `1K/2K`, `60fps`
- Output:
  - one rendered `1080p` video with the correct scoreboard
  - correct points, sets, and match result
  - player names automatically resolved — no manual input required
- Quality target:
  - **100% fully automated** — zero manual correction required on a correctly-recorded clip
  - operator input should only be needed once per new player (first enrollment)
  - after the first enrollment, the same player is recognized automatically in all future clips

### How to reach 100% automation — three-track strategy

**Track 1 — Player Identity System**
- Build a persistent face DB from the first clip where a player appears.
- Operator provides the player's name once (first enrollment); the system extracts and stores face embeddings automatically.
- Store face crops from each processed clip to progressively strengthen the identity record.
- From the second clip onward the same player is recognized without any operator input.
- Per-clip jersey binding covers set-boundary re-tracking without re-running face detection.

**Track 2 — Scoreboard Tool (current work)**
- Detect rallies, infer winners, validate score/state, render scoreboard video.
- Near-term: produce a usable output now, accept manual review while winner model is weak.
- Long-term: reduce manual review to zero as the winner model improves via Track 3.

**Track 3 — Rally Dataset and Model Improvement**
- Every reviewed clip feeds reviewed rally data into a growing training dataset.
- Once the dataset is large enough, fine-tune the winner VLM on reviewed data.
- Each training cycle should measurably reduce the manual-review rate on new clips.
- Repeat: new clips → reviewed data → fine-tune → lower review rate → new clips.

## Operational Scope
- This is an internal project, not a public multi-user product.
- The default operating model is local processing on one machine.
- The system is designed to process one video at a time.
- Production v1 is scoped to one full match clip at a time.
- Mid-match fragments or partial clips are debug-only inputs, not part of the production critical path.
- Do not optimize the architecture around multi-tenant, distributed, or concurrent batch use.

## Product Definition
The system is not just a rally detector.

The production system should:
- detect the main table correctly
- understand when rallies start and end
- infer the point winner with bounded risk
- validate score/state before applying a result
- render a usable final video with minimal manual cleanup

## Operator Workflow
- The system should provide a local Web UI for the internal user.
- The local Web UI is the primary operating path for production use.
- The Web UI should allow the user to:
  - select the input video
  - run the processing pipeline
  - review only low-confidence rally outcomes
- The Web UI should support a per-job operating mode:
  - `output only` — finish the scoreboard video, no dataset writeback required
  - `output + dataset` — finish the video and also persist reviewed rallies for future training
- Near-term success is:
  - produce a correct usable scoreboard video now
  - collect reviewed rally data while doing that work
- CLI scripts should be treated as secondary tools for narrow debug only.
  Once a CLI step proves useful, fold it back into shared backend services and the Web UI flow.
- The review flow should support active learning:
  - if the AI result is correct, the operator keeps it with one click
  - if the AI result is wrong, the operator corrects it with one click
- Human input should be exception-driven, not full manual annotation.
- Target manual-review rate: less than `2%` of rallies as the long-term mature target.
- The minimum required operator input for scoreboard completion is: who won this rally?
- After a user correction, the code must automatically recompute all downstream score changes:
  - next points, set progression, match progression, final rendered scoreboard state.
- The product should support two output modes:
  - `preview render` — allowed while review-needed rallies still exist, but must surface warnings
  - `final export` — must require all review-needed rallies to be resolved

## Guiding Architecture
- `Table ROI` is the primary scene anchor.
- Winner inference is constrained inference, not a single-model guess.
- Production logic should combine: table timing, player behavior, optional ball/audio signals, score/state validation.
- Ball detection is optional strong evidence, not a hard dependency.
- Offline reasoning is preferred when it improves identity stability and sequence consistency.

## Non-Negotiable Design Rules
- Prefer architecture-first solutions over local patches.
- Fix bugs at the owning layer: detector bugs → detector design; state bugs → state logic; rendering must not hide upstream failures.
- After the Web UI exists, do not let ad-hoc CLI flows become a second production path.
- When the operator asks to export review rallies, the export must be a clean end-to-end rerun from the original input video.
- Always verify important claims against the code before treating them as true.
- The agent must retain the right to challenge or correct a request when it conflicts with the code, the current architecture, or the final goal.
- Reject workaround directions that only mask symptoms:
  - display hold, frozen boxes, render-only trackers, fake continuity bridges, narrow hacks that overfit one clip
- When evidence is weak, prefer `missing`, `unknown`, or `review` over a forced wrong answer.

## Signal Stack
### 1. Table Stream
- Mandatory. Used for: rally timing, dead-time detection, bounce and motion context.

### 2. Player Streams
- Mandatory for high-quality winner inference.
- `Stream 2` tracks `Player A`, `Stream 3` tracks `Player B`.
- Used for: serve preparation, swing/motion cues, reset behavior, point-end behavior.

### 3. Global Context
- Lightweight full-frame context for: idle periods, ball retrieval, scene sanity checks, neighboring-table disambiguation.

### 4. Optional Strong Signals
- Ball trajectory / bounce, audio cues.
- Current `ball tracking V0`: classical small-object motion tracking inside expanded Table ROI.
  Frame differencing → small-blob extraction → short tracklets across nearby frames.
- Ball signals should be used as bounded evidence for merging false-split rallies and bounce context.
- Ball tracking must stay optional: the production path must still function when ball evidence is weak.

### 5. Validation Layer
- Production output must pass through score/state validation.
- This layer must: accept safe evidence, flag weak evidence, block contradictory updates, gate `final export`.

## Ball Tracking Doctrine
- Search only inside an expanded `Table ROI`, not across the full frame.
- Motion is more reliable than appearance for a ball this small — do not use appearance-heavy MOT.
- Do not promote a ball-tracking direction unless it improves checked regression clips without damaging the table-first baseline.

## Player Tracking Doctrine
- Player tracking is role tracking, not just left/right detection.
- `Player A` and `Player B` must be modeled relative to the tracked table.
- Tracker state must distinguish: `visible`, `occluded`, `missing`.
- Initial role seeding must be evidence-driven — do not trust frame `0` by default.
- `true leave` and `short occlusion` are different events and must not share the same fallback.
- A wrong neighboring-table capture is worse than `missing`.

## Winner Inference Doctrine
- Winner inference should use multiple signals, not one brittle cue.
- The system should support three decision outcomes: safe auto-apply, human review, blocked/unknown.
- Score/state validation must be part of the decision path, not only post-hoc reporting.
- Human review should request the minimum possible input — ask only for the winner of the uncertain rally.
- A user correction must be treated as authoritative input for that rally, and the pipeline must propagate the resulting scoreboard changes automatically.
- Review-needed rallies may be visible in a `preview render`, but they must block `final export` until resolved.
- Winner taxonomy should be treated as first-class supervision:
  - each reviewed rally should store: `winner`, `loser`, `taxonomy`, `last_hitter`
  - the same underlying event should reuse the same taxonomy label across different sets and matches
- Maintain a versioned reviewed-data bundle:
  - exported frozen rally clips + reviewed JSONL manifest + stable IDs
  - stored under `dataset/reviewed_matches/<match_id>/set_<nn>/`
  - treat this bundle as a production asset: reproducible, versionable, safe to reuse for fine-tuning
- The active-learning loop:
  1. run the current pipeline on a new match
  2. review predicted rally winners in the Web UI
  3. keep correct rallies or fix wrong rallies with one click
  4. auto-append reviewed rallies into the rolling fine-tune dataset
  5. once collection is large enough, train an adapted model
  6. use that model as the next pre-labeler for later matches
- Training-time augmentation such as `horizontal flip` is allowed: treat as extra training view, not a new reviewed label.
- Once a reviewed adapter branch is green, move the active winner path to the trained adapter — not kept as a parallel prompt-only branch.

## Player Identification System (Two-Tier)

### Goal
Automatically identify player names from the input video.
The operator enters a name only once per new player (first enrollment).
All subsequent clips recognize the same player without any manual input.

### Design

**Tier 1 — Global Identity (persistent face DB)**
- ArcFace (InsightFace ONNX `w600k_r50.onnx`) extracts 512-dim face embeddings.
- Face DB stored in `data/players/faces.json`; grows permanently across all processed clips.
- Same player recognized across different match days and venues.

**Tier 2 — Local Session (per-video jersey binding)**
- Within one video, jersey color is stable for the full match.
- Once a face is matched (Tier 1), bind identity to that jersey HSV histogram.
- For Set 2+: skip face re-detection; re-bind using jersey matching only (fast, no YOLO face alignment).

### Standalone Scan (production entry point)
`quick_identify_players_standalone(video_path, pose_weights_path, face_db)`:
- Runs as **Step 1b** in the pipeline — after trim, before rally detection.
- No rally timeline needed: uses fixed time windows.
  - FAR player: t=1–40s at 4 fps, rank=1 (faces camera from match start)
  - NEAR player: t=1–300s at 1 fps, rank=0, filtered to exclude FAR rank-flip contamination
- Returns `IdentificationResult` with face crops stored for unknown players.

### Unknown Player Enrollment (Web UI)
- "Nhận diện cầu thủ" button on setup form → background AJAX scan.
- If player not in DB: face crop displayed inline → operator enters name → POST `/api/enroll-player`.
- After enrollment: player name fields auto-fill, embedding saved to `faces.json`.
- Operator only needs to enroll once; all future clips recognize the player automatically.

### Failure modes and fallbacks
- Jersey colors too similar → flag "ambiguous", fall back to manual name entry.
- No face captured → fall back to manual name entry.
- Pipeline never blocked: identification failure → graceful fallback to user-entered names.

### UI Principle
- Player names are used everywhere in the UI — no NEAR/FAR positional labels shown to operator.
- The internal pipeline still tracks which player is currently on the near/far side per set,
  but this is invisible to the operator; they always see and click actual player names.

### Integration
- Entry point: `quick_identify_players_standalone()` in `backend/player_identification.py`
- Face DB: `data/players/faces.json`
- Web UI API: `POST /api/identify-players`, `GET /api/identify-players/{id}`, `POST /api/enroll-player`

### Future: YOLO signal integration
- Currently YOLO tracks players as largest/smallest bbox (positional, not identity-aware).
- After side swaps, rank flips and the YOLO player signal loses player identity.
- Planned: use jersey histograms from Tier 2 to anchor player identity in the YOLO signal path,
  giving the winner inference model stable player-identity context across sets.

## Production Baseline Promotion Rule
Do not promote a new algorithm direction unless it does all of the following:
- fixes the target bug at root cause
- preserves the current regression guardrails
- does not reintroduce neighboring-table capture
- does not rely on render-layer masking
- is validated on more than one clip or regression window

## Project Phases

### Phase 1 — Stable Table + Player Baseline ✓
- Table-first pipeline, stable rally detection, set boundary detection, player identity system.
- Player identification: ArcFace face DB + standalone scan as Step 1b in pipeline.
- Exit criteria met: regression clips run end-to-end, player names auto-resolved, no NEAR/FAR confusion.

### Phase 2 — Winner Inference Upgrade (current)
- Raise winner quality using multi-signal fusion; Web UI review + export flow functional.
- Current: trained adapter deployed, manual review rate still high, reviewed dataset seed = 71 rallies.
- Exit: winner accuracy reaches operational target; dataset growing via production use.
- Next steps:
  - Run first real match (`match_vinh_001__full.mp4`) end-to-end through Web UI
  - Wire reviewed corrections into dataset storage
  - YOLO player signal: integrate jersey identity to stabilize tracking across side swaps

### Phase 3 — Reviewed Dataset + SFT
- Build a reviewed winner dataset large enough for real supervision (target: 200–500, then >1000).
- Fine-tune the current winner model on reviewed data.
- Establish repeatable active-learning loop: review UI → reviewed dataset → training → upgraded model.
- Exit: adapted model beats prompt-only baseline; Web UI correction flow auto-feeds finetune collection.

### Phase 4 — Production Hardening
- Make the system repeatable across new venues and clips.
- Exit: stable batch usage, better observability, low-risk auto decisions.

## Document Map
- `ROADMAP_PRODUCTION.md` — why the project exists, what the final system must look like, what rules must not be violated
- `PROJECT_ACTION_PLAN.md` — current execution plan, big goals broken into shippable tasks, done/doing/todo board
- `PROJECT_PROGRESS.md` — daily work log, experiments, failures, resume points
