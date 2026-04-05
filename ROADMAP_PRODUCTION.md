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
  - one raw full-match table-tennis clip, usually `20-30` minutes
  - fixed-tripod camera per clip
  - typical input quality: `1K/2K`, `60fps`
- Output:
  - one rendered `1080p` video with the correct scoreboard
  - correct points, sets, and match result
- Quality target:
  - rally + winner pipeline with target `95%+`
  - manual correction should be limited to low-confidence cases

## Operational Scope
- This is an internal project, not a public multi-user product.
- The default operating model is local processing on one machine.
- The system is designed to process one video at a time.
- Production v1 is scoped to one full match clip at a time.
- Mid-match fragments or partial clips are debug-only inputs, not part of the production critical path.
- Do not optimize the architecture around:
  - multi-tenant server usage
  - distributed processing
  - concurrent batch handling as a primary requirement

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
- The Web UI should allow the user to:
  - select the input video
  - run the processing pipeline
  - review only low-confidence rally outcomes
- The default review asset should be a short rally clip.
- Human input should be exception-driven, not full manual annotation.
- If AI confidence for a rally winner is too low, the system should ask only:
  - who won this rally?
- Target manual-review rate:
  - less than `2%` of rallies
- The user should only provide the winner for that rally.
- After a user correction, the code must automatically recompute all downstream score changes:
  - next points
  - set progression
  - match progression
  - final rendered scoreboard state
- The product should support two output modes:
  - `preview render`
  - `final export`
- `preview render` may be allowed while review-needed rallies still exist, but must surface warnings clearly.
- `final export` must require all review-needed rallies to be resolved.

## Guiding Architecture
- `Table ROI` is the primary scene anchor.
- Winner inference is constrained inference, not a single-model guess.
- Production logic should combine:
  - table timing
  - player behavior
  - optional ball/audio signals
  - score/state validation
- Ball detection is optional strong evidence, not a hard dependency.
- Offline reasoning is preferred when it improves identity stability and sequence consistency.

## Non-Negotiable Design Rules
- Prefer architecture-first solutions over local patches.
- Fix bugs at the owning layer:
  - detector / tracker bugs -> detector / tracker design
  - state / score bugs -> state / score logic
  - rendering must not hide upstream failures
- When the operator asks to export review rallies for `set1 / set2 / set3 / set4`, the export must be a clean end-to-end rerun:
  - start from the original input video
  - rerun the required pipeline stages from scratch
  - produce the final rally clips from that fresh run
  - do not reuse intermediate JSON artifacts from earlier partial runs
- Always verify important claims and requests against the code in this repository before treating them as true.
- A statement from the operator should be accepted as true only when:
  - it matches the current code
  - or the code does not contain enough information to confirm or reject it
- Do not blindly mirror assumptions, status, or architecture claims into project documents without checking the code first.
- The agent must retain the right to challenge or correct a request when:
  - it conflicts with the code
  - it conflicts with the current architecture
  - it moves the project away from the final goal
- Reject workaround directions that only mask symptoms:
  - display hold
  - frozen boxes
  - render-only trackers
  - fake continuity bridges without identity evidence
  - narrow hacks that overfit one clip and weaken the system
- When evidence is weak, prefer `missing`, `unknown`, or `review` over a forced wrong answer.

## Signal Stack
### 1. Table Stream
- Mandatory.
- Used for:
  - rally timing
  - dead-time detection
  - bounce and motion context

### 2. Player Streams
- Mandatory for high-quality winner inference.
- `Stream 2` tracks `Player A`.
- `Stream 3` tracks `Player B`.
- Used for:
  - serve preparation
  - swing / motion cues
  - reset behavior
  - point-end behavior

### 3. Global Context
- Lightweight full-frame context is useful for:
  - idle periods
  - ball retrieval
  - scene sanity checks
  - neighboring-table disambiguation

### 4. Optional Strong Signals
- ball trajectory / bounce
- audio cues
- current `ball tracking V0` direction should stay:
  - inside an expanded `Table ROI`
  - motion-first
  - secondary to the table stream, not a replacement for it

## Ball Tracking Doctrine
- Ball tracking should search only in an expanded `Table ROI` / playing lane, not across the full frame by default.
- The current preferred `V0` direction is classical small-object motion tracking:
  - frame differencing inside the expanded table crop
  - small-blob candidate extraction
  - short motion continuity / short tracklets across nearby frames
- Do not treat appearance-heavy MOT methods as the main ball solution:
  - the ball is too small
  - motion is more reliable than appearance re-identification
- Ball signals should be used as bounded evidence for:
  - merging false split rallies
  - bounce / dead-ball context
  - future winner-inference support
- Ball tracking must stay optional strong evidence:
  - it may improve rally quality
  - but the production path must still function when ball evidence is weak or missing
- Do not promote a ball-tracking direction unless it improves checked regression clips without damaging the table-first baseline.

### 5. Validation Layer
- Production output must pass through score/state validation.
- This layer must be able to:
  - accept safe evidence
  - flag weak evidence
  - block contradictory updates
  - gate `final export` when required reviews are unresolved

## Player Tracking Doctrine
- Player tracking is role tracking, not just left/right detection.
- `Player A` and `Player B` must be modeled relative to the tracked table.
- Tracker state must distinguish:
  - `visible`
  - `occluded`
  - `missing`
- Initial role seeding must be evidence-driven:
  - do not trust frame `0` by default
  - allow deferred seeding from a bootstrap window
  - backfill early frames only when identity linkage is real
  - ambiguous early frames may stay `missing`
- `true leave` and `short occlusion` are different events and must not share the same fallback.
- A wrong neighboring-table capture is worse than `missing`.
- Near-side and far-side roles may need different cues and thresholds.

## Winner Inference Doctrine
- Winner inference should use multiple signals, not one brittle cue.
- Local multimodal review is allowed as a secondary reviewer:
  - one local vision model may inspect rally frames
  - one local reasoning model may judge structured evidence
  - they should support review / debug / bounded correction, not replace the owning detector blindly
- The system should support three decision outcomes:
  - safe auto-apply
  - human review
  - blocked / unknown
- Score/state validation must be part of the decision path, not only post-hoc reporting.
- Human review should request the minimum possible input:
  - ask only for the winner of the uncertain rally
  - never ask the user to manually recalculate later points or sets
- A user correction must be treated as authoritative input for that rally, and the pipeline must propagate the resulting scoreboard changes automatically.
- Review-needed rallies may still be visible in a `preview render`, but they must block `final export` until resolved.

## Production Baseline Promotion Rule
Do not promote a new algorithm direction unless it does all of the following:
- fixes the target bug at root cause
- preserves the current regression guardrails
- does not reintroduce neighboring-table capture
- does not rely on render-layer masking
- is validated on more than one clip or regression window

## Project Phases
### Phase 1. Stable Table + Player Baseline
Goal:
- build a trustworthy table-first pipeline with stable `Player A / Player B` tracking

Exit condition:
- the main regression clips can run end-to-end without obvious tracker-role corruption

### Phase 2. Winner Inference Upgrade
Goal:
- raise rally boundary and winner quality using multi-signal fusion

Exit condition:
- winner accuracy and rally quality reach operational target on labeled benchmarks

### Phase 3. Production Hardening
Goal:
- make the system repeatable across new venues and clips

Exit condition:
- stable batch usage, better observability, low-risk auto decisions

## Current Production Stance
- The default production direction remains table-first.
- `backend/ai_multistream_rally.py` and `scripts/generate_rally_timeline.py` are still experimental.
- For the current algorithm-change cycle:
  - keep `table / ROI-first` unchanged as the production reference
  - keep the current `ball tracking V0` implementation unchanged as bounded secondary evidence
  - focus rally-detector debugging only on the independent `player-only / YOLO player-signal` path
  - defer fusion-policy tuning until the `player` path is materially healthier
- Current `player-only` boundary logic has been temporarily reset away from the failed full-rally state machine:
  - the previous long-`active` state-machine experiment on `set4` is rejected as a rally detector
  - the current debug direction is now `start-first`
  - it still runs only when `mode=player` and `player_signal_source=role_tracker`
  - it still does **not** redesign role assignment:
    - `Stream 2` still maps to `Player A`
    - `Stream 3` still maps to `Player B`
    - role assignment still comes from the existing offline role tracker
  - the current temporary algorithm is:
    1. detect every `Toss & Serve` start image independently from player behavior
    2. treat those detections as `rally_count + let_count`
    3. define provisional `active` time only between one detected start and the next detected start
    4. detect `LET` inside that bounded interval
    5. compute final rally count as:
       - `total starts - total LET`
  - the current `start-first` detector uses per-role pose / bbox signals:
    - crouch / ready posture
    - reach toward the table
    - serve cue
    - upper-body activity
    - footwork
    - opponent-ready context
    - same-role vs opposite-role dominance
  - `LET` remains represented by segment flags:
    - `rally_label_let`
    - `let_no_score`
  - `LET` segments are still excluded from score conversion in the current contract layer
  - this branch is still a debug / benchmark path only and is not a promotion candidate yet
- `ball tracking V0` is currently an experimental secondary signal:
  - it may support conservative rally-boundary merge / validation
  - it is not yet the promoted production baseline
- Local `Qwen` review paths are currently experimental:
  - `qwen3-vl` may be used for frame / boundary review
  - `qwen3` may be used for structured reasoning
  - these paths are debug tools until they prove benchmark value
- Experimental paths should not replace the production path until they beat it on benchmarks and regression clips.

## Document Map
- `ROADMAP_PRODUCTION.md`
  - why the project exists
  - what the final production system must look like
  - what rules must not be violated
- `PROJECT_ACTION_PLAN.md`
  - the current execution plan
  - big goals broken into small shippable tasks
- `PROJECT_PROGRESS.md`
  - daily progress
  - pass / fail experiments
  - artifacts
  - resume point for the next session
