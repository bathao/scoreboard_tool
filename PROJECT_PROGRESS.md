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
  - `2026-03-10`
- Current code baseline:
  - `v12 deferred seed bootstrap`
- Current code state:
  - rolled back to the code that produced `v12 deferred_seed` outputs
  - later ownership-tuning experiments were removed
- Current tracker test result:
  - `tests/test_offline_player_tracker.py`
  - `15 passed, 1 warning`

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
Only keep the latest full-set debug outputs:
- `debug_report/Vinh_set1_persondet_offline_fullset_v12_deferred_seed.mp4`
- `debug_report/Vinh_set2_persondet_offline_fullset_v12_deferred_seed.mp4`
- `debug_report/Vinh_set3_persondet_offline_fullset_v12_deferred_seed.mp4`
- `debug_report/Vinh_set4_persondet_offline_fullset_v12_deferred_seed.mp4`
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
  - relatively good
- `set4`
  - relatively good
- `set3`
  - much better after deferred seeding
  - full set is temporarily acceptable
- `set1`
  - still has an important unresolved bug around `1:34 -> 1:47`
  - during rally, `Player B` can jump to a wrong person behind / outside the real playing lane

## Active Open Issue
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

## Latest Successful Direction
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
1. Start from the current `v12 deferred seed bootstrap` code, not the ownership-tuning experiment that was rolled back.
2. Re-open the real failure window in `set1`:
   - `1:34 -> 1:47`
3. Investigate in this order:
   - raw detections and true player presence
   - tracklet linkage
   - role ownership / role observation selection
4. Do not fix it with:
   - render smoothing
   - frozen boxes
   - continuity hacks
   - clip-specific thresholds without broader justification
5. If a real root-cause fix is found, rerun full `set1`, `set2`, `set3`, and `set4`.

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
