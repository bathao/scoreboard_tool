# Project Progress

## Purpose
This file is the working log for the project.

Use this file for:
- current baseline
- what was tried
- what passed
- what failed
- artifacts worth reopening
- the exact resume point for the next session

Do not use this file as the long-term architecture spec.

## Current Status
- Date:
  - `2026-03-21`
- Current production draft baseline:
  - `table / ROI-first`
- Current tracker baseline:
  - `v12 deferred seed bootstrap`
- Current code state:
  - production draft path remains table-first
  - last pushed checkpoint:
    - commit `6764139`
    - `Add ball-only rally benchmark and Qwen review tooling`
  - experimental multistream code now includes:
    - role-aware table refinement
    - standalone `player-only` draft mode for benchmark-only compare
    - classical `ball tracking V0`
    - standalone `ball-only` draft mode for benchmark-only compare
  - local `Qwen` review support now also exists:
    - `qwen3-vl:8b` installed in Ollama for vision review
    - `qwen3:14b` installed in Ollama for reasoning review
    - default local vision model config was switched from `llama3.2-vision` to `qwen3-vl:8b`
    - `review_rally_splits_qwen.py` now supports `--skip-models` for candidate-only benchmarking
  - role and ball paths remain experimental and are not promoted baselines yet
- Current last confirmed test result:
  - full suite:
    - `65 passed, 1 warning`
  - multistream rally tests:
    - `14 passed, 1 warning`
  - tracker tests:
    - `15 passed, 1 warning`
  - note:
    - these were confirmed on `2026-03-21` after the independent `table / player / ball` draft-export work

## Work Log - `2026-03-21`
### Experiments That Passed
- `independent 3-detector draft export baseline`
  - the Stage 1 independent detector paths now all exist and can export the same draft JSON contract:
    - `table / ROI-first`
    - `player-only / YOLO player-signal`
    - `ball-only / ball tracking`
  - draft JSON now carries:
    - `summary.total_rallies`
    - detector provenance in `analysis_metadata`
  - benchmark counts on the checked set list are:
    - `set1`
      - `table`: `14`
      - `player`: `13`
      - `ball`: `18`
    - `set2`
      - `table`: `20`
      - `player`: `17`
      - `ball`: `20`
    - `set3`
      - `table`: `18`
      - `player`: `13`
      - `ball`: `20`
    - `set4`
      - `table`: `18`
      - `player`: `22`
      - `ball`: `22`
- `set4 qwen split-candidate extraction rerun`
  - added `--skip-models` support in `scripts/review_rally_splits_qwen.py`
  - candidate-only rerun surfaced `11` split candidates on `set4`
  - this confirmed the blocker moved from `0-candidate extraction` to accept / reject quality
- `ball-only standalone draft mode`
  - added a real `--mode ball` path in `scripts/generate_draft_multistream.py`
  - `ball-only` mode now disables player streams and uses a standalone ball-tracking profile
  - added dedicated multistream tests for `ball-only` mode and the post-pass pair-merge behavior
- `ball-only standalone tuning v7`
  - tuned standalone ROI scope, track confirmation, motion weighting, and ball-only segmentation gates
  - added a conservative `ball_pair_merge` post-pass for short contiguous `split_long` pairs
  - final `ball-only v7` benchmark counts are:
    - `set1`: `18`
    - `set2`: `20`
    - `set3`: `20`
    - `set4`: `22`
- `test coverage refresh`
  - multistream rally tests now pass at `10 passed, 1 warning`
  - full suite now passes at `61 passed, 1 warning`

### Experiments That Failed Or Were Rejected
- `raw ball-only port`
  - first standalone `ball-only` run produced:
    - `set1`: `7`
    - `set2`: `22`
    - `set3`: `2`
  - rejected as unstable
- `over-strict standalone motion-first profile`
  - one standalone tuning branch improved `set3` to `22`
  - but dropped `set2` to `12`
  - rejected as too aggressive
- `pre-post-pass standalone profile`
  - one intermediate standalone profile reached:
    - `set1`: `18`
    - `set2`: `21`
    - `set3`: `22`
    - `set4`: `27`
  - rejected as still over-splitting `set4`
- `set4 qwen accept / reject logic`
  - candidate extraction is now working
  - but no accepted split policy is benchmarked enough yet
  - keep the `Qwen` outputs debug-only for now

### Main Findings From Today
- basic independent rally detection is now completed for all 3 Stage 1 detector families:
  - `table`
  - `player`
  - `ball`
- `table / ROI-first` remains the strongest reference detector today:
  - `set1`: local GT-aligned at `14`
  - `set2`: still close to the known target at `20`
  - `set3 / set4`: remains the conservative reference count
- current `player-only` path is usable as an independent benchmark detector, but is not yet strong enough:
  - it under-counts clearly on `set1 / set2 / set3`
  - it over-counts on `set4`
- `ball-only` can now produce a real standalone rally draft and no longer collapses on `set1 / set3`
- the best current standalone benchmark is `ball-only v7`:
  - `set1`: `18`
  - `set2`: `20`
  - `set3`: `20`
  - `set4`: `22`
- standalone `ball-only` is still not a promoted baseline:
  - `set2` known target `19` is still matched better by conservative `table_ball_refined`
  - `set1` local GT `14` is still well below `ball-only v7 = 18`
  - `set4` debug target `20` is still below `ball-only v7 = 22`
- the safe current use of standalone `ball-only` is:
  - benchmark / diagnosis
  - possible bounded evidence for future split / merge logic
  - not replacement for the `table / ROI-first` production baseline
- `set4` qwen split-candidate extraction is no longer blocked by `0-candidate` generation
- review quality, not extraction, is now the real bottleneck for the local `Qwen` split path

## Work Log - `2026-03-16`
### Experiments That Passed
- `role-stream extraction for experimental multistream draft`
  - wired `Player A / Player B` streams into the experimental draft path
  - aligned the multistream compare path back to the production-style table energy extraction
- `table_refined safety check on set1`
  - restored `set1` count back to `14`
  - stopped the earlier role-fused under-segmentation from becoming the default compare path
- `classical ball tracking V0`
  - added ROI-scoped ball motion extraction inside expanded `Table ROI`
  - used frame differencing + small-blob candidates + short motion continuity
  - exposed it through `table_ball_refined`
- `set2 ball-assisted merge benchmark`
  - baseline `table` count was `20`
  - conservative `table_ball_refined` improved the result to `19`
  - operator-confirmed target for `set2` is `19` rallies with score `11-8`
- `set1 / set3 / set4 regression check for ball V0`
  - `set1`: stayed `14`
  - `set3`: stayed `18`
  - `set4`: stayed `18`
  - conservative ball support did not worsen rally count on those sets
- `test coverage refresh`
  - added / kept unit coverage for role-series behavior and ball-assisted merge behavior
  - full test suite stayed green
- `local Qwen model setup`
  - installed `qwen2.5vl:7b`, then `qwen3-vl:8b`
  - kept `qwen3:14b` as the local reasoning model
  - switched local vision defaults in code to `qwen3-vl:8b`

### Experiments That Failed Or Were Rejected
- `raw role-fused activation as the main detector`
  - on `set1`, the initial role-fused path dropped to `8` rallies
  - rejected as a production-direction candidate
- `role quiet-gap refine as a promotion candidate`
  - did not improve `set2`
  - on `set3`, it created an extra split in a window later identified by operator review as idle
  - kept only as experimental debug logic, not a promoted path
- `first ball-merge thresholds`
  - the first `ball tracking V0` pass on `set2` over-merged down to `15`
  - thresholds were tightened before keeping the current `19`-rally artifact
- `stale set2 GT file`
  - repo GT file with `10` rallies was wrong / incomplete
  - it was removed instead of being treated as current truth
- `set4 qwen merge-only review`
  - baseline `table` output for `set4` was `18`
  - `qwen3-vl + qwen3` merge-only boundary review dropped it to `14`
  - rejected as clearly worse than the baseline
- `set4 qwen split-review v0`
  - first split-review pass stayed at `18`
  - candidate generation was too strict and found `0` review candidates
- `set4 qwen split-review v1`
  - a softer candidate-generator rewrite was started
  - rerun was interrupted before a valid result was recorded
  - do not treat the interrupted run as evidence

### Main Findings From Today
- `Player A / Player B` role streams are useful as secondary evidence, not as the main rally-activation signal
- classical `ball tracking V0` is currently a better rally-segmentation assist than the current role-refine logic
- the safe current use of ball evidence is:
  - conservative split merge support
  - not standalone rally detection
- `set2` is the first confirmed case where ball support is better than the current table-only baseline on rally count
- `set1 / set3 / set4` show that the current ball logic is at least neutral on count, but boundary quality still needs manual checking
- local `Qwen` review can be useful as a debug tool, but current prompts / candidate generation are not trustworthy enough for auto boundary changes
- `set4` is the wrong place for merge-only review:
  - baseline is already under-counting
  - the real need is split / missing-rally recovery

### Important Operator-Provided Clarifications
- true `set2` target:
  - `19` rallies
  - score `11-8`
- true `set4` target for debug compare:
  - `20` rallies
  - use this only for compare / debug, not as a clip-specific code rule
- `set3` debug note:
  - window `11s -> 22s` was reported as idle by operator review
  - treat that as debug evidence for diagnosis only
  - do not turn it into a clip-specific hard-coded rule

## Work Log - `2026-03-10`
### Experiments That Passed
- `full-set render check on old baseline`
  - exported full `set1`, `set2`, `set3`, `set4`
  - confirmed `set2` and `set4` were already relatively good
  - confirmed `set3` was the obvious outlier that needed first attention
- `set3 opening-window probe`
  - re-ran `set3 0s -> 10s`
  - confirmed the failure started at frame `0`
  - confirmed `Player A` was being seeded onto the neighboring-table far-side person
- `deferred seed bootstrap`
  - changed role seeding so the tracker could wait for better evidence instead of trusting frame `0`
  - improved `set3` from the first seconds
  - full `set3` became temporarily acceptable again
- `regression check after deferred seed`
  - re-exported full `set1`, `set2`, `set4` with the new baseline
  - confirmed `set2` and `set4` stayed relatively good
  - confirmed the main remaining visible issue moved back to `set1 1:34 -> 1:47`
- `artifact cleanup`
  - removed older debug outputs
  - kept only the latest full-set `v12 deferred_seed` artifacts
- `baseline rollback discipline`
  - after later failed tuning attempts, the code was rolled back to the clean `v12 deferred_seed` baseline
  - tracker tests still pass on that rolled-back baseline

### Experiments That Failed Or Were Rejected
- `old seed logic on set3`
  - failed from frame `0`
  - wrong early seed caused a full-set identity cascade
- `ownership-tuning branch after v12`
  - not accepted as a new baseline
  - later role-ownership changes were removed
- `depth-only explanation for set1 1:34 -> 1:47`
  - tested conceptually against the observed candidates
  - rejected as too weak because wrong and true `Player B` candidates overlapped too much in depth
- `x-only / right-side hard threshold explanation`
  - rejected because true `Player B` can also stand far right in clips that currently look good
- `quick local guardrail style fixes`
  - rejected because they would act like patchwork instead of fixing the owning layer

### Main Findings From Today
- `set3` root cause was `initial role seeding`, not occlusion or render
- `set1 1:34 -> 1:47` is a different class of problem from `set3`
- the current unresolved `set1` bug should not be solved by:
  - render masking
  - frozen boxes
  - continuity hacks
  - one-off hard thresholds without broader support
- the clean end-of-day code should remain `v12 deferred seed bootstrap` until a better root-cause fix is proven on regression clips

## What The Current Baseline Solves
- `set3` no longer seeds `Player A` onto the neighboring-table far-side person at frame `0`
- the tracker is allowed to stay effectively unseeded early instead of forcing the wrong `A`
- `Stream 2 / Player A` and `Stream 3 / Player B` are now good enough to use as meaningful player streams for later winner inference work

## Current Outputs Kept
Keep the latest full-set debug outputs and current rally-benchmark artifacts:
- `debug_report/Vinh_set1_persondet_offline_fullset_v12_deferred_seed.mp4`
- `debug_report/Vinh_set2_persondet_offline_fullset_v12_deferred_seed.mp4`
- `debug_report/Vinh_set3_persondet_offline_fullset_v12_deferred_seed.mp4`
- `debug_report/Vinh_set4_persondet_offline_fullset_v12_deferred_seed.mp4`
- `matches/Vinh_set1_role_compare_table_draft.json`
- `matches/Vinh_set1_table_ball_refined_v0.json`
- `matches/Vinh_set2_role_compare_table_draft_v2.json`
- `matches/Vinh_set2_table_ball_refined_v0b.json`
- `matches/Vinh_set3_role_compare_table_draft_v2.json`
- `matches/Vinh_set3_table_ball_refined_v0.json`
- `matches/Vinh_set4_role_compare_table_draft_v0.json`
- `matches/Vinh_set4_table_ball_refined_v0.json`
- `matches/Vinh_set4_qwen_boundary_reviewed_v0.json`
- `matches/Vinh_set4_qwen_boundary_review_report_v0.json`
- `matches/Vinh_set4_qwen_split_reviewed_v0.json`
- `matches/Vinh_set4_qwen_split_review_report_v0.json`
- `matches/Vinh_set4_qwen_split_reviewed_candidates_only_v1.json`
- `matches/Vinh_set4_qwen_split_review_report_candidates_only_v1.json`
- `scripts/review_rally_boundaries_qwen.py`
- `scripts/review_rally_splits_qwen.py`
- `outputs/smoke_set1_ball_only_v7_20260320.json`
- `outputs/smoke_set2_ball_only_v7_20260320.json`
- `outputs/smoke_set3_ball_only_v7_20260320.json`
- `outputs/smoke_set4_ball_only_v7_20260320.json`
- `matches/Vinh_set3_table_role_refined_debug_compare.json`
- `matches/Vinh_set3_rally_timestamps_only.json`
- Render times remembered:
  - `set3`
    - `2026-03-10 18:58`
  - `set1`
    - `2026-03-10 19:13`
  - `set2`
    - `2026-03-10 19:26`
  - `set4`
    - `2026-03-10 19:42`

## Current Quality Read
- `set2`
  - operator-confirmed truth is `19` rallies with score `11-8`
  - current `table` baseline is `20`
  - current `ball tracking V0` result is `19`
  - current standalone `ball-only v7` result is `20`
  - current read:
    - conservative `table_ball_refined` is still the strongest checked count result on the known target
    - standalone `ball-only` is close enough to keep benchmarking, but it is not better than the current ball-assisted merge path
- `set4`
  - current `table` baseline is `18`
  - current `ball tracking V0` result is also `18`
  - current standalone `ball-only v7` result is `22`
  - operator-provided debug truth is `20`
  - current `qwen` merge-only review result is `14`
  - current `qwen` split-review v0 result is still `18`
  - current `qwen` split candidate-only rerun now surfaces `11` candidates
  - current read:
    - merge-only `Qwen` review is clearly worse
    - standalone `ball-only` recovers missing-rally pressure better than table-only, but still over-counts
    - split-review direction is still unresolved
- `set3`
  - current `table` baseline is `18`
  - current `table_ball_refined` result is also `18`
  - current standalone `ball-only v7` result is `20`
  - the earlier `table_refined` role path produced `19`, which is not trusted as an improvement
  - no local GT yet
  - current read:
    - standalone `ball-only` is the first non-collapsing standalone benchmark on this set
- `set1`
  - still has an important unresolved bug around `1:34 -> 1:47`
  - during rally, `Player B` can jump to a wrong person behind / outside the real playing lane
  - current `table` baseline is `14`
  - current `ball tracking V0` result is also `14`
  - current standalone `ball-only v7` result is `18`
  - local GT count is `14`
  - current read:
    - standalone `ball-only` no longer collapses
    - but it still over-counts against local GT

## Active Open Issues
### `set1` full match
- Bug window:
  - about `1:34 -> 1:47`
- Symptom:
  - rally is still happening
  - tracker switches `Stream 3 / Player B` to the wrong outsider
  - tracker may switch repeatedly during that window
- Current status:
  - not fixed
  - no workaround promoted
  - code was intentionally rolled back to clean `v12 deferred_seed` baseline before continuing

### `set3` table-path debug concern
- Debug window:
  - about `11s -> 22s`
- Current read:
  - operator review reported this window as idle
  - both baseline table output and role-refined debug output need caution in this area
  - this should guide diagnosis only, not become a clip-specific rule

### `set4` missing-rally debug concern
- Current read:
  - baseline `table` under-counts at `18`
  - debug target is `20`
  - current `Qwen` merge-only review made this much worse
  - current `Qwen` split-review logic still needs better candidate generation before the models can help

### `standalone ball-only` promotion question
- Current read:
  - `ball-only v7` is the best current standalone benchmark
  - but it is still not strong enough to replace the table-first baseline
  - the next decision is whether it should stay debug-only or feed bounded evidence back into the table-first path

## Latest Successful Direction
### `local Qwen review setup`
- Why it was added:
  - to test whether local multimodal review could help debug rally-boundary errors
- What changed:
  - installed local `qwen3-vl:8b` for vision review
  - kept local `qwen3:14b` for reasoning review
  - switched default local vision model config to `qwen3-vl:8b`
  - added scripts:
    - `scripts/review_rally_boundaries_qwen.py`
    - `scripts/review_rally_splits_qwen.py`
- Validation remembered:
  - setup is working locally
  - but review quality is not yet good enough to trust for automatic rally-boundary updates

### `table-first + classical ball gap-merge V0`
- Why it was added:
  - role streams alone were not yet improving rally segmentation reliably
  - `set2` looked like a likely split-repair candidate
- What changed:
  - added `backend/ai_ball_tracking.py`
  - ball signal is extracted only in expanded `Table ROI`
  - used:
    - frame differencing
    - small-blob candidate filtering
    - short motion continuity
  - current use is conservative:
    - only to merge likely false splits in `table_ball_refined`
- Validation remembered:
  - `set1` stayed `14`
  - `set2` improved `20 -> 19`
  - `set3` stayed `18`
  - `set4` stayed `18`
  - full suite stayed `57 passed, 1 warning`

### `standalone ball-only v7`
- Why it was added:
  - to test whether ball motion alone could produce a usable rally draft without collapsing on `set1 / set3`
- What changed:
  - added a real `ball-only` draft mode in the multistream path
  - added standalone ball-tracking profile tuning
  - added a conservative post-pass merge for short contiguous split-pair artifacts
- Validation remembered:
  - `set1`: `18`
  - `set2`: `20`
  - `set3`: `20`
  - `set4`: `22`
  - full suite stayed `61 passed, 1 warning`
  - current read:
    - much better than the first raw standalone runs
    - still not a promoted production baseline

### `v12 deferred seed bootstrap`
- Why it was added:
  - `set3` failed from frame `0`
  - the old seed logic trusted the wrong early left/right pair
- What changed:
  - seed pair selection now uses accumulated offline evidence
  - near-side `A` evidence is explicitly preferred
  - the tracker can delay seeding instead of forcing the wrong match pair
- Validation remembered:
  - `set3 0s -> 10s` probe improved
  - `set2` and `set4` stayed acceptable on quick checks
  - end-of-day code was intentionally reset back to this version after later failed experiments

## Today Also Completed Outside Tracker Code
- planning docs were restructured into:
  - `ROADMAP_PRODUCTION.md`
  - `PROJECT_ACTION_PLAN.md`
  - `PROJECT_PROGRESS.md`
- `PROJECT_ACTION_PLAN.md` now distinguishes:
  - product artifact flow
  - engineering checklist
  - critical path to `v1 done`
- Stage 1 is now framed explicitly as:
  - existing independent `table / ROI-first` reference detector
  - independent `multistream / YOLO player-signal` detector
  - independent standalone `ball-only` detector
  - fusion / validation across the 3 detector outputs before winner work
- `table / ROI-first` is not treated as new work:
  - it already exists
  - it remains the checked Stage 1 reference detector

## Historical Milestones Worth Remembering
### `v2 fixA_missing`
- Key lesson:
  - when `Player A` truly leaves frame, `missing` is correct and better than borrowing a neighboring-table player

### `v8 global occlusion solver`
- Key lesson:
  - global reasoning helped short occlusion, but only when ambiguity stayed manageable

### `v10 multimode role memory`
- Key lesson:
  - role identity cannot rely on a single averaged prototype

### `v11 role ownership`
- Key lesson:
  - continuity alone is not enough
  - role ownership must be explicit and role-specific

## Rejected Directions
- bridge-tracklet hacks
  - created fake continuity
- display-hold / frozen-box logic
  - hid symptoms without fixing identity
- render-only tracker ideas
  - wrong layer
- aggressive segment fragmentation
  - exploded ambiguity and regressed role stability
- one-off local guardrail tweaks without architectural support
  - may help locally but are not a trustworthy baseline

## Important Root-Cause Findings
- full-box area alone is not a safe identity gate
- post-hoc hard filtering after assignment can create conflicting truths
- global reasoning is useful only when it reduces ambiguity
- `true leave` and `short occlusion` are different states
- `Player A` and `Player B` need role-specific modeling

## Resume Point For The Next Session
1. Start from the current code where:
   - production draft baseline is still `table / ROI-first`
   - tracker baseline is still `v12 deferred seed bootstrap`
   - Stage 1 now has 3 explicit detector paths:
     - `table / ROI-first` as the already-existing reference detector
     - `multistream / YOLO player-signal` as an experimental independent detector now benchmarked on `set1..4`
     - standalone `ball-only v7` as the experimental independent detector already benchmarked on `set1..4`
   - conservative `table_ball_refined` remains experimental
   - local `Qwen` review scripts exist and split-candidate extraction can run in `--skip-models` mode
2. Do not re-open `table / ROI-first` as if it still needs to be created:
   - it already exists
   - use it as detector `#1` in the Stage 1 compare / fusion plan
3. Next critical Stage 1 work is:
   - compare `table`, `multistream`, and standalone `ball-only` on the same reviewed sets
   - align their rally lists by time overlap, not only by count
   - optimize for ordered rally list and boundary quality, not count alone
4. After the 3-detector compare is stable enough:
   - define the first fusion / validation rule that merges the 3 rally lists into one final rally list
   - benchmark the fused list against the independent detectors
5. Do not promote standalone `ball-only` as the new baseline yet:
   - it is still benchmark / diagnosis code
   - it still over-counts known clips
6. For `set4`, treat the debug compare target as:
   - `20` rallies
   - compare-only, not a clip-specific rule
7. Continue the unfinished `Qwen` split-review direction as a side debug track:
   - define a conservative accept / reject policy on the fresh `11-candidate` list
   - only keep completed reruns as evidence
   - do not let it displace the Stage 1 3-detector compare / fusion path
8. Keep the deferred `set1` tracker failure visible:
   - `1:34 -> 1:47`
   - do not treat it as fixed
   - do not let detector benchmarking hide it
9. Do not fix it with:
   - render smoothing
   - frozen boxes
   - continuity hacks
   - clip-specific thresholds without broader justification
10. If a real root-cause fix is found, rerun full `set1`, `set2`, `set3`, and `set4`.

## End-Of-Session Update Rule
At the end of each work session:
- update `PROJECT_PROGRESS.md` with:
  - new baseline
  - test result
  - outputs kept
  - open issues
  - exact next resume point
- update `PROJECT_ACTION_PLAN.md` if priorities or task status changed
- update `ROADMAP_PRODUCTION.md` only if architecture or doctrine changed
