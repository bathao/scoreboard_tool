# Project Action Plan

## Purpose
This file is the execution plan for the project.

Use this file for:
- the practical architecture used to finish the product
- current layer status
- small tasks that can be finished in one session
- deciding what is `done`, `doing`, and `todo`

Do not use this file as:
- the final product doctrine
- the experiment log

## Planning Model
To reach the final goal, the project is best organized into 6 runtime layers:

1. `Local AI / Perception`
2. `Decision / Validation`
3. `Domain Core (Score Engine)`
4. `Session / Application`
5. `Video Processing / Rendering`
6. `Operator Interface (Web UI)`

## Important Architecture Note
- `AI Contract` is still important, but it is not the best top-level runtime layer anymore.
- In the current codebase, it behaves more like a shared contract / schema artifact used by:
  - `Decision / Validation`
  - `Session / Application`
  - `Operator Interface`
- Benchmarking / regression safety is also critical, but it is a cross-cutting discipline, not a runtime layer.

## Status Legend
- `[todo]`
- `[doing]`
- `[done]`
- `[deferred]`
- `[blocked]`
- `[rejected]`

## v1 Production Scope
- Production v1 supports:
  - one full match clip
  - local processing
  - one video at a time
- Production v1 does not require:
  - mid-match fragment support
  - partial-clip scoreboard context recovery
- Fragment clips remain useful for debugging only.

## Current Detector Action Plan
This is the temporary action plan for the experimental `YOLO player` rally path.

### Current Work-Cycle Constraint
- `[doing]` keep `table / ROI-first` unchanged as the current production reference detector
- `[doing]` keep the current `ball tracking V0` implementation unchanged as a secondary reference signal
- `[doing]` restrict current algorithm-change work to the independent `multistream / YOLO player-signal` rally path
- `[deferred]` do not spend the current cycle on fusion-rule tuning or `Qwen` boundary/split tuning until the `player` detector is materially stronger

### Step 1. Detect rally starts from images
- `[doing]` detect every `Toss & Serve` start image independently from player behavior
- `[doing]` use those start images to estimate:
  - `total rally starts + total LET starts`
- `[doing]` improve the start detector until start-state detection is visually obvious and effectively exact on checked debug windows
- `[doing]` treat this as the current priority over full-rally `active/end` state-machine tuning
- `[done]` current checked regression suite `set1..4` is now operator-accepted for starter detection in the independent `YOLO player` path
  - scope:
    - `starter = rally + let`
  - accepted counts:
    - `set1 = 15`
    - `set2 = 19`
    - `set3 = 18`
    - `set4 = 23`
  - latest pushed checkpoint:
    - commit `5e94835`
    - `Detect accurate starters for all 4 sets (server + let)`
  - latest targeted regression:
    - `31 passed, 1 warning`

### Step 2. Detect `LET` and subtract it
- `[todo]` detect `LET` only after the start-image detector is stable enough
- `[todo]` infer `serve mode` before localizing `LET`
  - use the accepted `starter_role` sequence as the primary signal
  - infer:
    - `double-serve mode` when the game is likely still before `10-10`
    - `single-serve mode` when the game is likely already after `10-10`
  - preferred read:
    - `AA -> BB -> AA -> BB` patterns support `double-serve mode`
    - `A -> B -> A -> B` patterns support `single-serve mode`
- `[todo]` derive mandatory `LET` counts from same-server run length, assuming starter detection is already exact
  - if `double-serve mode` is active:
    - a same-server run of length `L` implies `max(0, L - 2)` mandatory `LET`
  - if `single-serve mode` is active:
    - a same-server run of length `L` implies `max(0, L - 1)` mandatory `LET`
- `[todo]` use local post-start vision / timing cues only to localize which starter(s) inside a forced run are the actual `LET`
  - do not use those cues to re-open the starter list itself
- `[todo]` compute:
  - `total rallies = total starts - total LET`
- `[todo]` keep `active` bounded strictly between:
  - one detected start
  - and the next detected start
- `[todo]` do not allow a free-running long `active` state outside that bounded interval

## Artifact Flow
This is the simplest way to read the project plan.

The product should move through these concrete artifacts:

1. `Raw Full-Match Video`
   - Source artifact selected by the user or engineer
   - Scope for production v1:
     - one full match clip
     - local processing
     - one video at a time

2. `Draft Rally JSON`
   - Produced from raw video by the current draft-generation path
   - Minimum useful contents:
     - table ROI
     - ordered rally list
     - `t_start`
     - `t_end`
     - confidence
     - flags
     - winner may still be `unknown`
   - This is the main deliverable of `Layer 1 - Local AI / Perception`

3. `Reviewable Draft JSON`
   - The draft JSON after winner inference and decision/validation metadata are attached
   - Must be able to express for each rally:
     - safe auto-apply
     - human review
     - blocked / unknown
   - Must also carry score/state validation results
   - This is the main deliverable of `Layer 2 - Decision / Validation`

4. `Corrected / Validated Match State`
   - The state after one or more reviewed rally winners are corrected
   - The corrected winner must be persisted
   - Downstream points / sets / match state must be replayed deterministically
   - This is where `Layer 3 - Domain Core` becomes authoritative

5. `Review Assets + Preview Render`
   - Short rally clips for flagged reviews
   - Preview video with scoreboard and visible warning state when unresolved reviews remain
   - This is the first operator-facing output

6. `Final Export`
   - Final rendered `1080p` video
   - Must use the corrected and validated scoreboard state
   - Must be blocked until all required reviews are resolved

### Practical Reading Rule
- If you are confused by layers, read the project as:
  - `video -> draft JSON -> reviewable JSON -> corrected state -> preview -> final export`
- Layers describe ownership.
- Stages describe implementation order.
- Artifacts describe what must exist between stages.

## Per-Artifact Action Checklist
This section answers a different question:
- not just `what artifact should exist?`
- but `what work should we do, in order, to reach that artifact reliably?`

### A. Engineering Path To Reach `Draft Rally JSON`
- Production target artifact:
  - one draft JSON covering one full match clip
- Engineering reality:
  - the current production path can already export a full-match draft in one run
  - but quality work should not jump directly from `raw video` to `full-match acceptance`
- Current code paths:
  - `table / ROI-first` draft path exists and is the current production baseline
  - `multistream / YOLO player-signal` draft path exists but is still experimental
  - `table_ball_refined` now exists as an experimental path:
    - table-first segmentation
    - optional `Player A / Player B` streams
    - optional classical `ball tracking V0` merge support
  - standalone `ball-only` draft mode now exists experimentally:
    - benchmark-only compare path
    - tuned with standalone ball-tracking profile
    - not a promoted production baseline
  - local `Qwen` review paths now exist experimentally:
    - `qwen3-vl` for frame / boundary inspection
    - `qwen3` for structured reasoning over review payloads
- Preferred Stage 1 strategy from now:
  1. run `Table ROI / table-first` rally detection as an independent path
  2. run `multistream / YOLO player-signal` rally detection as an independent path
  3. run standalone `ball-only / ball tracking online` rally detection as an independent path
  4. fuse / validate the 3 independent rally lists into one final rally list for the video
  5. only after Stage 1 rally quality is effectively solved, continue to winner inference
- Current-cycle execution override:
  1. do not retune `table / ROI-first` in this cycle
  2. do not retune `ball tracking V0` in this cycle
  3. use `table` and `ball` outputs only as fixed comparison references while debugging `player`
  4. make rally-algorithm changes only in the independent `YOLO player` path before resuming fusion work
- Temporary current-cycle note for the `player` detector:
  - stop treating the failed long-`active` state machine as the current optimization target
  - first optimize `Toss & Serve` start-image detection
  - use `start_count = rally_count + let_count`
  - then add `LET` subtraction as the next pass
  - bound any later `active` state between consecutive detected starts only
- Important correction:
  - do not optimize only for `rally count`
  - Stage 1 should optimize for:
    - ordered rally list
    - boundary quality
    - count correctness
  - a correct count with wrong rally boundaries is not sufficient
- Use this checklist:
  1. `[done]` keep the current `table / ROI-first` draft export as the working production baseline
  2. `[done]` independent `Table ROI / table-first` rally detection already exists and is the checked reference detector for Stage 1
  3. `[doing]` run `1 set -> rally draft` with the independent `multistream / YOLO player-signal` path
  4. `[done]` benchmark `1 set -> rally draft` with the independent standalone `ball-only / ball tracking online` path on `set1`, `set2`, `set3`, `set4`
  5. `[doing]` compare the 3 independent set-level drafts against manual review or GT, using `table / ROI-first` and `ball-only` as fixed references while `player` is being debugged
  6. `[deferred]` define the fusion / validation rule that merges the 3 rally lists into one final rally list only after the independent `player` path is trustworthy enough
  7. `[todo]` benchmark the fused rally list against the independent paths
  8. `[todo]` fix root-cause failures at the owning layer of the weakest path
  9. `[todo]` add a focused regression test for each fixed failure
  10. `[todo]` rerun the affected full set
  11. `[todo]` rerun the regression set list:
     - `set1`
     - `set2`
     - `set3`
     - `set4`
  12. `[todo]` benchmark rally segmentation / draft quality on representative full-match clips
  13. `[todo]` only then promote the best independent or fused path as the new baseline
- Practical rule:
  - for engineering, go from:
    - `debug window -> one set -> regression sets -> full match`
  - do not treat `one successful full-match export` as sufficient proof of quality
  - `100%` may be used as a local engineering target on a small checked regression set, but it is not the global production promise
  - treat `99.99% rally count` only as a stretch engineering target, not as the only acceptance rule
- Current experimental read:
  - role streams are useful as secondary evidence, but current `table_refined` logic is not yet a promotion candidate
  - conservative `ball tracking V0` currently looks more promising as a draft-quality assist:
    - `set2` improved from `20 -> 19`
    - `set1 / set3 / set4` stayed neutral on rally count
  - checked explanation for the current `ball tracking V0` behavior:
    - `set2` had exactly one eligible merge pair under the current conservative gates:
      - `160.03 -> 166.57`
      - combined duration `6.54s`
      - normalized ball window was just strong enough:
        - peak about `0.434`
        - mean about `0.192`
    - `set1` had no adjacent pair inside the current merge-gap gate
    - `set3` and `set4` did have contiguous `split_long` neighbors, but their combined durations were above the current `8.5s` cap, so the merge logic never fired
  - practical read:
    - current `ball tracking V0` is only helping short false-split repair
    - it is not currently addressing long-segment correction or missing-rally recovery
  - experimental standalone `ball-only v7` now exists as a benchmark-only compare path:
    - `set1`: `18`
    - `set2`: `20`
    - `set3`: `20`
    - `set4`: `22`
  - practical read for standalone `ball-only`:
    - it no longer collapses on `set1 / set3`
    - but it still over-counts known clips and is not the promoted baseline
  - current `Qwen` review passes are not promotion candidates yet:
    - `set4` merge-only review regressed strongly
    - the older `set4` split-review `v0` report stayed neutral because the candidate pass it used was too strict
    - a fresh `skip-models` rerun on `2026-03-20` now surfaces `11` split candidates on `set4`
    - the current blocker is choosing which candidates the review stack should accept, not `0-candidate` extraction

### B. Engineering Path From `Draft Rally JSON` To Winner-Labeled JSON
- Goal:
  - turn raw draft output into winner-labeled JSON with review metadata
- Gate before starting this stage:
  - the video must already have a rally list that is effectively stable on checked benchmarks
  - do not move to winner work while Stage 1 still has major rally-boundary uncertainty
- Use this checklist:
  1. `[todo]` define or improve the first winner-candidate path from player / pose / YOLO-derived signals if benchmark data shows it helps
  2. `[doing]` use `Ollama local` as refinement / second-opinion on weak or unresolved rallies
  3. `[doing]` benchmark local `Qwen3-VL + Qwen3` review passes on debug clips before trusting them for correction
  4. `[todo]` assign decision status for each rally:
     - auto-apply
     - review
     - blocked / unknown
  5. `[todo]` attach score/state validation output
  6. `[todo]` define the authoritative correction payload
  7. `[todo]` make one unresolved rally representable without ambiguity
  8. `[todo]` export `Reviewable Draft JSON`

### C. After `Reviewable Draft JSON` exists
- Goal:
  - turn one reviewed winner correction into authoritative corrected match state
- Use this checklist:
  1. `[todo]` apply one corrected rally winner
  2. `[todo]` persist the correction in the draft contract
  3. `[todo]` replay the score engine deterministically
  4. `[todo]` verify downstream recompute:
     - later points
     - set progression
     - match progression
  5. `[todo]` add replay regression tests around score-edge cases
  6. `[todo]` export `Corrected Final JSON`

### D. After `Corrected / Validated Match State` exists
- Goal:
  - make the operator-facing outputs usable
- Use this checklist:
  1. `[todo]` generate short rally clips for flagged reviews
  2. `[todo]` produce a `preview render`
  3. `[todo]` surface unresolved-review warnings in preview
  4. `[todo]` keep `final export` blocked while required review items remain unresolved

### E. After `Review Assets + Preview Render` exists
- Goal:
  - complete the human review loop and unlock `Final Export`
- Use this checklist:
  1. `[todo]` provide the minimum local Web UI
  2. `[todo]` let the user select one full match video
  3. `[todo]` let the user inspect flagged rallies through short rally clips
  4. `[todo]` let the user submit only:
     - who won this rally?
  5. `[todo]` refresh corrected state, validation state, and export readiness
  6. `[todo]` allow `final export` only when required reviews are resolved

## Definition of Done (v1)
Production v1 is done only when all of these are true:
- the user selects one full match video in the local Web UI
- the system runs locally on one video at a time
- the system produces draft output plus confidence / review state
- the system can produce a `preview render` even when warnings remain
- the user reviews only flagged rallies using a short rally clip
- the user answers only:
  - who won this rally?
- the system automatically recomputes:
  - later points
  - set progression
  - match progression
  - render state
- the system produces a `final export` only after all required reviews are resolved
- review rate stays below `5%`

## Done Rule
- The project is considered `v1 done` when:
  - every item in `Critical Path to v1 Done` is effectively completed
  - the `Definition of Done (v1)` is demonstrated on a real full-match clip
- Items under:
  - `Deferred / Debug-Only`
  - `After v1 / Hardening`
  do not block `v1 done`

## Layer Summary
### 1. Local AI / Perception
- Current read:
  - early but real
  - main technical bottleneck

### 2. Decision / Validation
- Current read:
  - partially present in code
  - not yet product-grade

### 3. Domain Core (Score Engine)
- Current read:
  - strongest layer
  - already useful in the current pipeline

### 4. Session / Application
- Current read:
  - basic orchestration exists
  - missing correction-driven product flow

### 5. Video Processing / Rendering
- Current read:
  - partial but real
  - final render path exists

### 6. Operator Interface (Web UI)
- Current read:
  - not implemented yet
  - exists only as a product requirement

## Recommended Work Order From Now
1. stabilize `Local AI / Perception`
2. complete `Decision / Validation`
3. keep `Domain Core` deterministic under correction
4. clean up `Session / Application` orchestration
5. harden `Video Processing / Rendering`
6. build `Operator Interface (Web UI)` on top of the stabilized local pipeline

## Critical Path to v1 Done
Follow these stages in order. If all stages pass, the project reaches `v1 done`.

### Stage 1. Raw video -> draft JSON baseline gate
- Primary artifact:
  - `Draft Rally JSON`
- Working method:
  - `debug window -> one set -> regression sets -> full match`
- Stage 1 rally strategy:
  1. independent `Table ROI / table-first` detector
  2. independent `multistream / YOLO player-signal` detector
  3. independent standalone `ball-only / ball tracking online` detector
  4. fusion / validation across the 3 detectors to produce the final rally list
- Current-cycle override:
  - keep `table / ROI-first` unchanged
  - keep `ball tracking V0` unchanged
  - change the rally algorithm only in the independent `YOLO player` detector before returning to fusion work
- Stage 1 acceptance rule:
  - the target is not only `correct rally count`
  - the target is:
    - correct ordered rally list
    - correct rally boundaries
    - count close enough to near-perfect on the checked regression set
- Temporary priority note:
  - `set1 1:34 -> 1:47` is temporarily ignored for the current work cycle
  - do not treat this as fixed
  - bring it back before baseline promotion / Stage 1 exit
- `[todo]` keep one short debug-window export path for fast iteration
- `[todo]` require one full-set check before accepting a local fix
- `[deferred]` isolate and fix the root cause of `set1 1:34 -> 1:47`
- `[done]` independent `Table ROI / table-first` path already exists and is the checked Stage 1 reference detector
- `[done]` benchmark the independent `multistream / YOLO player-signal` path on `set1`, `set2`, `set3`, `set4`
  - current benchmark counts are:
    - `set1`: historical pre-`v9` benchmark `13`
    - `set2`: `17`
    - `set3`: `13`
    - `set4`: `22`
- `[done]` benchmark the independent standalone `ball-only v7` path on `set1`, `set2`, `set3`, `set4`
  - current benchmark counts are:
    - `set1`: `18`
    - `set2`: `20`
    - `set3`: `20`
    - `set4`: `22`
  - treat `table` and `ball` as temporarily stable reference detectors for the current debug cycle
  - do not treat this as a promoted final baseline yet
- `[doing]` debug and improve the independent `multistream / YOLO player-signal` path
  - current read:
    - latest `set1` start-first snapshot `sandwich_v9` now reports `12`
    - `set1` is still under-detecting and is waiting for manual image review on the saved `12` starts
    - `set2`: under-count at `17`
    - `set3`: under-count at `13`
    - `set4`: over-count at `22`
  - current goal:
    - first make `player` a usable independent `start-image` detector
    - use detected starts to estimate `rally + let`
    - then add `LET` subtraction
    - only after that return to full player-rally segmentation and 3-detector fusion
- `[deferred]` define the fusion / validation rule that merges the 3 detector outputs into one final rally list
- `[deferred]` benchmark `Qwen3-VL + Qwen3` review passes on `set4` again only after the independent `player` detector is in a healthier state
- `[todo]` add the focused regression test for that failure
- `[todo]` rerun full `set1`, `set2`, `set3`, `set4`
- Exit gate:
  - `set1` blocker is fixed
  - `set2 / set3 / set4` do not regress on full-match review

### Stage 2. Draft JSON -> reviewable JSON gate
- Primary artifact:
  - `Reviewable Draft JSON`
- `[todo]` define the authoritative correction payload
- `[todo]` define the three runtime decision outcomes:
  - auto-apply
  - review
  - blocked / unknown
- `[todo]` define preview vs final-export gating
- Exit gate:
  - one unresolved rally can be represented, reviewed, corrected, and revalidated without ambiguity

### Stage 3. Reviewable JSON -> corrected match state gate
- Primary artifact:
  - `Corrected / Validated Match State`
- `[todo]` replay the score engine after one corrected rally winner
- `[todo]` verify deterministic downstream recompute:
  - points
  - sets
  - match result
- `[todo]` add regression tests for match-edge cases
- Exit gate:
  - one correction reliably updates all downstream score state

### Stage 4. Corrected state -> preview artifacts gate
- Primary artifacts:
  - `Short Rally Review Clips`
  - `Preview Render`
- `[todo]` generate short rally clips for flagged reviews
- `[todo]` support `preview render` with warning state
- `[todo]` support `final export` only when required reviews are resolved
- Exit gate:
  - the product can show a useful review clip and produce the correct output mode

### Stage 5. Preview artifacts -> operator review loop gate
- Primary artifact:
  - `Operator Review Session`
- `[todo]` implement the minimum local Web UI:
  - select one full match video
  - start processing
  - inspect flagged rallies
  - submit corrected winner
- Exit gate:
  - one internal user can complete the review loop without touching raw JSON

### Stage 6. Operator-reviewed state -> final export gate
- Primary artifact:
  - `Final Export`
- `[todo]` run one full local production flow:
  - input full match
  - preview
  - review flagged rallies
  - final export
- `[todo]` verify review rate target on the chosen acceptance clip/run
- Exit gate:
  - the system satisfies `Definition of Done (v1)`

## Layer 1 - Local AI / Perception
### Current State
- This layer covers:
  - table ROI detection
  - rally segmentation
  - winner-related AI signals
  - player tracking / player streams
- Current experimental sub-directions inside this layer now include:
  - role-aware table refinement
  - classical `ball tracking V0` inside expanded `Table ROI`
  - standalone `ball-only` rally draft benchmarking
  - local `Qwen3-VL + Qwen3` review scripts for boundary / split debug
- The codebase already has real implementations here, but the layer is still far from production target quality.
- Current clean tracker baseline:
  - `v12 deferred seed bootstrap`

### Already Done
- `[done]` table ROI detection is wired into the production draft path
- `[done]` motion-based rally draft generation exists
- `[done]` basic AI winner refinement exists through `scripts/ai_refine_draft.py`
- `[done]` offline player tracking exists for `Player A / Player B` debugging
- `[done]` deferred seeding fixed the obvious `set3` frame-0 seeding failure
- `[done]` experimental `table_refined` path exists for role-aware table segmentation checks
- `[done]` classical `ball tracking V0` exists as an optional secondary signal in the experimental draft path
- `[done]` experimental standalone `ball-only` draft mode now exists for benchmark-only compare
- `[done]` local `Qwen3-VL` and `Qwen3` review scripts now exist for debug-only rally review experiments
- `[done]` `review_rally_splits_qwen.py` now supports `--skip-models` so split candidates can be benchmarked without Qwen decisions

### In Progress
- `[doing]` debug the independent `multistream / YOLO player-signal` rally detector:
  - current temporary scope is narrowed to `Toss & Serve` start-image detection
  - compare `player` start candidates against visually obvious serve-start frames
  - confirm that the detected starts cover `rally + let`
  - keep the current checked review artifact at:
    - `debug_report/Vinh_set1_rally_start_candidates_v9_review/`
  - keep the current checked JSON snapshot at:
    - `matches/Vinh_set1_stage1_player_independent_sandwich_v9.json`
  - do not trust the previous long-`active` state-machine behavior as the working direction
- `[doing]` keep `table / ROI-first` and standalone `ball-only` as fixed comparison references while the current algorithm work stays inside the independent `player` path

### Remaining Work
- `[deferred]` temporarily ignore the `set1` bug around `1:34 -> 1:47`
  - do not treat it as fixed
  - bring it back before promoting a new baseline
- `[deferred]` return to 3-detector fusion only after the independent `player` detector is trustworthy enough on checked sets
- `[deferred]` return to `Qwen` split / boundary tuning only after the current `player` rally-algorithm change is checked
- `[todo]` keep the engineering loop fast:
  - short debug window first
  - then one full set
  - then full regression set list
- `[todo]` write a focused regression test for the `set1` `Player B` failure
- `[todo]` fix that bug at the owning layer, not with render masking or continuity hacks
- `[todo]` rerun full `set1`, `set2`, `set3`, `set4` after a real fix
- `[todo]` keep `set2 / set3 / set4` full-match regression mandatory before promoting a new tracker baseline
- `[todo]` benchmark rally segmentation quality on representative clips
- `[todo]` benchmark the fused rally list against each independent detector path
- `[todo]` verify boundary quality, not only rally count, for `ball tracking V0`
- `[todo]` verify boundary quality, not only rally count, for standalone `ball-only v7`
- `[todo]` decide whether standalone `ball-only` should stay benchmark-only or feed bounded evidence back into the table-first path
- `[todo]` tune `ball_gap_merge` conservatively so it improves real table splits without masking bad parent segments
- `[todo]` keep `Qwen` review outputs as debug-only until they show real benchmark gains
- `[todo]` benchmark which fresh `set4` split candidates are real and define conservative accept / reject gates
  - fresh `skip-models` rerun on `2026-03-20` surfaced `11` candidates across `pt_0001 / pt_0002 / pt_0004 / pt_0010 / pt_0014 / pt_0016`
  - treat the old `0-candidate` `v0` report as stale
- `[todo]` integrate `Player A / Player B` streams into winner inference if the current benchmark cannot meet v1 quality targets without them

## Layer 2 - Decision / Validation
### Current State
- This layer decides what the system is allowed to believe and apply.
- It should own:
  - confidence buckets
  - review routing
  - score validation
  - correction propagation rules
- The current codebase already contains pieces of this:
  - `backend/ai_contract.py`
  - `backend/score_validation.py`
- But the layer is not complete enough for the final product workflow yet.

### Already Done
- `[done]` draft match / draft point schema exists
- `[done]` confidence bucket helper exists
- `[done]` human-review detection helper exists
- `[done]` score validation exists with `expected_scope = any | set | match`
- `[done]` correction audit structure exists in the draft contract
- `[done]` the current architecture already has a natural place for review gating in validation, not in rendering

### In Progress
- `[doing]` define the minimal low-confidence correction flow around one rally winner

### Remaining Work
- `[todo]` define the authoritative correction payload:
  - rally id
  - corrected winner
  - correction source
- `[todo]` define the three runtime decision outcomes clearly:
  - auto-apply
  - review
  - blocked / unknown
- `[todo]` define what must be recomputed automatically after a correction
- `[todo]` make score validation part of the correction loop, not just post-hoc reporting
- `[todo]` define the output gating policy clearly:
  - `preview render` allowed with warnings
  - `final export` blocked until required reviews are resolved
- `[todo]` add tests for:
  - invalid correction payload rejection
  - correction replay consistency
  - review-rate accounting

## Layer 3 - Domain Core (Score Engine)
### Current State
- This is the strongest layer in the codebase.
- It already powers timeline replay and render-state construction.
- The main remaining risk is not core set logic, but correction replay integration.

### Already Done
- `[done]` set logic
- `[done]` best-of handling (`BO3 / BO5 / BO7`)
- `[done]` deterministic replay
- `[done]` timeline build
- `[done]` snapshot generation
- `[done]` engine is already used in timeline/render and score validation

### Remaining Work
- `[todo]` audit replay correctness after a corrected rally winner changes downstream state
- `[todo]` add regression tests near:
  - end of set
  - start of next set
  - match-ending point
- `[todo]` keep the score engine as the single source of truth after correction

## Layer 4 - Session / Application
### Current State
- This layer orchestrates the local pipeline.
- The current code already has real orchestration:
  - `scripts/run_production_pipeline.py`
  - `backend/match_session.py`
- But there is still no clean app-level correction workflow for the final product.

### Already Done
- `[done]` basic pipeline orchestration exists for:
  - generate draft
  - refine winners
  - render final video
- `[done]` load / reset / replay / export concepts exist in the session utilities

### Remaining Work
- `[todo]` define one stable app/service flow for:
  - load draft
  - apply human correction
  - replay score engine
  - persist updated draft
  - export final render
- `[todo]` make correction handling idempotent and repeatable
- `[todo]` support two app-level output paths:
  - `preview render`
  - `final export`
- `[todo]` keep AI logic out of this layer as much as possible

## Layer 5 - Video Processing / Rendering
### Current State
- This layer already has meaningful code:
  - draft generation from video
  - final scoreboard render
  - audio mux
  - debug exports
- But it is not yet polished enough for the final operator workflow.

### Already Done
- `[done]` video-to-draft generation path exists
- `[done]` final render path exists
- `[done]` `1080p` render path exists
- `[done]` original audio mux exists
- `[done]` debug video export exists for tracker work

### Remaining Work
- `[todo]` audit timestamp sync from raw video through draft, correction, and final render
- `[todo]` make render consume corrected scoreboard state cleanly after human correction
- `[todo]` generate short rally clips for low-confidence review
- `[todo]` support `preview render` with unresolved-review warnings
- `[todo]` support `final export` only after required reviews are resolved
- `[todo]` make the final render path usable after correction without manual score editing
- `[todo]` harden the current render path for long real clips, not only debug runs

## Layer 6 - Operator Interface (Web UI)
### Current State
- This layer does not exist yet in implementation.
- No actual local Web UI code was found in the repository.
- This layer should stay thin and operator-focused.

### Already Done
- `[done]` the local Web UI requirement is defined in `ROADMAP_PRODUCTION.md`
- `[done]` the intended operator workflow is now reflected in planning

### Remaining Work
- `[todo]` choose the local UI approach
- `[todo]` define the first usable screens:
  - select one video
  - start processing
  - inspect flagged rallies
  - submit corrected winner
- `[todo]` define what the UI shows for each flagged rally using a short rally clip as the default review asset:
  - rally id
  - timestamps
  - confidence bucket
  - score validation state
  - correction history
- `[todo]` wire one correction action to:
  - update the draft
  - replay the score engine
  - refresh review state
  - enable final render
- `[todo]` keep the UI local-only and single-video-at-a-time

## Cross-Cutting Workstreams
These are mandatory, but they are not runtime layers.

### A. Benchmark / Regression Discipline
- `[todo]` maintain a small regression list of clips and critical timestamps
- `[todo]` add tests whenever a bug is fixed at root cause
- `[todo]` benchmark rally and winner quality on representative full-match clips for v1 acceptance

### B. Shared AI Contract / Data Schema
- `[todo]` reject invalid payloads early
- `[todo]` make correction history and review flags stable enough for UI and export flows

## After v1 / Hardening
- `[deferred]` broaden benchmark coverage beyond the minimum v1 acceptance set
- `[deferred]` keep draft schema backward-compatible across future product versions when possible
- `[deferred]` add more observability and QC reporting once the core local workflow is stable

## Deferred / Debug-Only
- `[deferred]` partial-match / fragment context support for production flow
- `[deferred]` mid-match-start score engine context recovery
- `[deferred]` any clip-scope handling needed only for debug fragments
- `[deferred]` audio cues unless they become a direct blocker for rally/winner quality
- `[deferred]` promotion of experimental multistream role / ball rally logic unless it clearly beats the table-first production path

## Target Product Workflow
This is the user-facing flow the full system must eventually support:

1. `[todo]` user opens the local Web UI
2. `[todo]` user selects one full match video input
3. `[todo]` system runs the local pipeline
4. `[todo]` system auto-applies high-confidence rally winners
5. `[todo]` system may produce a `preview render` with warnings when confidence issues remain
6. `[todo]` system asks the user only when confidence is low
7. `[todo]` user reviews flagged rallies through short rally clips
8. `[todo]` user answers only:
   - who won this rally?
9. `[todo]` system automatically recomputes:
   - later points
   - set progression
   - match progression
   - final render state
10. `[todo]` system allows `final export` only after required review items are resolved

## Product Targets To Preserve
- `[todo]` manual review rate should stay below `5%` of rallies
- `[todo]` the user must never recalculate score manually
- `[todo]` the default workflow stays local and single-video-at-a-time

## Best Next Small Tasks
1. `[done]` lock the independent `YOLO player` `starter + LET` baseline on checked `set1`, `set2`, `set3`, `set4`
   - accepted current results:
     - `set1 = 14 rallies`, `LET = 1`
     - `set2 = 19 rallies`, `LET = 0`
     - `set3 = 18 rallies`, `LET = 0`
     - `set4 = 20 rallies`, `LET = 3`
   - latest kept drafts:
     - `matches/Vinh_set1_stage1_player_independent_sandwich_with_starter_role.json`
     - `matches/Vinh_set2_stage1_player_independent_sandwich_with_starter_role.json`
     - `matches/Vinh_set3_stage1_player_independent_sandwich_with_starter_role.json`
     - `matches/Vinh_set4_stage1_player_independent_sandwich_with_starter_role.json`
   - latest targeted regression:
     - `33 passed, 1 warning`
2. `[done]` export `starter_role` and infer `LET` from the reviewed player-path starter sequence
   - serving-law constraints are allowed as the global rule
   - no timestamp-specific hardcoding is allowed
   - `LET` localization still depends on general timing / pose evidence inside each same-server run
3. `[done]` repair the `set3` false `LET` issue at the `starter_role` layer
   - root cause:
     - wrong `starter_role` created a fake `BBB | A | BB` shape
   - current fix:
     - conservative `double-serve` role-singleton repair runs before `LET` inference
   - accepted repaired `set3` serve pattern:
     - `BB | AA | BB | AA | BB | AA | BB | AA | BB`
4. `[doing]` keep the accepted `starter + LET` baseline frozen while redefining `active` strictly between consecutive accepted starts
   - do not retune `table / ROI-first`
   - do not retune `ball tracking V0`
   - do not reopen accepted `set1..4` starter boundaries or `LET` labels without new operator evidence
5. `[todo]` move downstream point / winner / score logic onto the frozen rally list from task `#1`
6. `[todo]` return to 3-detector fusion only after the `player` detector remains trustworthy beyond the `start-first` stage
