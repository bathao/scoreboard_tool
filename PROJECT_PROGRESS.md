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
  - `2026-03-16`
- Current production draft baseline:
  - `table / ROI-first`
- Current tracker baseline:
  - `v12 deferred seed bootstrap`
- Current code state:
  - production draft path remains table-first
  - experimental multistream code now includes:
    - role-aware table refinement
    - classical `ball tracking V0`
  - role and ball paths remain experimental and are not promoted baselines yet
- Current test result:
  - full suite:
    - `57 passed, 1 warning`
  - multistream rally tests:
    - `6 passed, 1 warning`
  - tracker tests:
    - `15 passed, 1 warning`

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

### Main Findings From Today
- `Player A / Player B` role streams are useful as secondary evidence, not as the main rally-activation signal
- classical `ball tracking V0` is currently a better rally-segmentation assist than the current role-refine logic
- the safe current use of ball evidence is:
  - conservative split merge support
  - not standalone rally detection
- `set2` is the first confirmed case where ball support is better than the current table-only baseline on rally count
- `set1 / set3 / set4` show that the current ball logic is at least neutral on count, but boundary quality still needs manual checking

### Important Operator-Provided Clarifications
- true `set2` target:
  - `19` rallies
  - score `11-8`
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
  - current read:
    - ball-assisted merge is better than table-only on rally count
- `set4`
  - current `table` baseline is `18`
  - current `ball tracking V0` result is also `18`
  - no local GT yet
- `set3`
  - current `table` baseline is `18`
  - current `table_ball_refined` result is also `18`
  - the earlier `table_refined` role path produced `19`, which is not trusted as an improvement
  - no local GT yet
- `set1`
  - still has an important unresolved bug around `1:34 -> 1:47`
  - during rally, `Player B` can jump to a wrong person behind / outside the real playing lane
  - current `table` baseline is `14`
  - current `ball tracking V0` result is also `14`
  - local GT count is `14`

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

## Latest Successful Direction
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
   - `ball tracking V0` remains experimental
2. Re-open the real failure window in `set1`:
   - `1:34 -> 1:47`
3. Keep the experimental ball path conservative:
   - use it for diagnosis and merge support
   - do not promote it without boundary-quality checks beyond count
4. Investigate in this order:
   - raw detections and true player presence
   - tracklet linkage
   - role ownership / role observation selection
5. Analyze why `ball tracking V0` helped `set2` but stayed neutral on `set1 / set3 / set4`.
6. Do not fix it with:
   - render smoothing
   - frozen boxes
   - continuity hacks
   - clip-specific thresholds without broader justification
7. If a real root-cause fix is found, rerun full `set1`, `set2`, `set3`, and `set4`.

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
