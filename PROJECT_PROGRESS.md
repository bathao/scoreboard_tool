# Project Progress

This file supersedes dated progress logs for the player-tracking work in this repository.
Use this as the single resume point for the project.

## Current Focus
- Stabilize `Player A / Player B` tracking for the main table.
- Keep the tracker correct on `Vinh_set1.mp4`, especially the debug window `50s -> 90s`.
- Preserve this invariant:
  - `wrong player` is worse than `missing`
  - `true leave` must become `missing`
  - short occlusion should recover when identity evidence is trustworthy

## Non-Negotiable Tracker Laws
- Do not use render-only fixes to hide upstream tracker failures.
- Do not use bridge hacks or fake continuity.
- Do not regress `Player A true leave -> missing` in order to fix short occlusion.
- Do not let `Stream 2` or `Stream 3` jump to a neighboring-table player.
- `missing` must be allowed when all visible candidates are worse than no assignment.

## Main Debug Setup
- Video:
  - `Vinh_set1.mp4`
- Default debug window:
  - `50s -> 90s`
- Important relative timestamps inside that window:
  - `8s` relative = about absolute `58s`
  - `25s` relative = about absolute `75s`
  - `29s` relative = about absolute `79s`
  - `34s` relative = about absolute `84s`
  - `36s` relative = about absolute `86s`

## Current Code Direction
- Person detector:
  - `weights/yolov8s.pt`
- Tracker architecture:
  - offline tracklets
  - global A/B assignment
  - role-specific observation selection
  - explicit occlusion timeline
- Key files:
  - `backend/offline_player_tracker.py`
  - `scripts/debug_multi_stream.py`
  - `tests/test_offline_player_tracker.py`

## Current Best Working State
- Current working artifact for the latest direction:
  - `debug_report/Vinh_set1_persondet_offline_50s_to_90s_v11_role_ownership.mp4`
- This direction adds:
  - role-specific ownership gating
  - stronger rejection of neighboring-table candidates
  - true-leave protection in role-gap bridging
  - regression coverage for `A true leave` and `B` continuity
- Current test result:
  - `49 passed`

## Latest Verified Result On The Real Debug Window
- Probe run on the correct `50s -> 90s` window shows:
  - `tracklets = 58`
  - `A-assigned low-center tracklets = []`
    - meaning the obvious neighboring-table `A` ownership bug is no longer present in assignment
  - `A missing` intervals now exist again:
    - `7.86s -> 13.16s`
    - `24.62s -> 27.98s`
    - `35.70s -> 36.24s`
    - `36.77s -> 38.87s`
  - direct probes at the three user-reported timestamps:
    - relative `8s`: `A = missing`
    - relative `25s`: `A = missing`
    - relative `36s`: `A = missing`
- This is the key latest improvement:
  - the previous bug where `Stream 2` captured the neighboring-table player during true leave is now fixed in the tracked debug window

## Important Operator Note
- A wrong validation output was rendered once for `0s -> 40s`:
  - `debug_report/Vinh_set1_persondet_offline_0s_to_40s_v11_role_ownership.mp4`
- That file was generated from the wrong time window for this bug.
- Do not use it to judge the `8s / 25s / 36s` issue discussed here.
- The correct validation file is:
  - `debug_report/Vinh_set1_persondet_offline_50s_to_90s_v11_role_ownership.mp4`

## Historical Baseline To Keep
- The most important early baseline remains:
  - `debug_report/Vinh_set1_persondet_offline_50s_to_90s_v2_fixA_missing.mp4`
- Why it mattered:
  - when `Player A` truly walked out of frame, `Stream 2` became `missing`
  - it did not jump to a neighboring-table player
- This baseline remained the reference guardrail for all later work.

## Successful Directions Worth Keeping

### 1. Offline person-based tracker
- This was the first major step forward over older pose-first / online directions.
- Good outcomes:
  - much better overall tracking quality
  - much better speed
  - a usable offline continuity prior

### 2. Global occlusion solver on top of the offline tracker
- Artifact:
  - `debug_report/Vinh_set1_persondet_offline_50s_to_90s_v8_global_occlusion_solver.mp4`
- Good outcome:
  - much better handling of the short foreground occlusion around absolute `79s`
- Important lesson:
  - this direction was useful, but it exposed weaknesses in later role export logic

### 3. Multi-mode role memory
- Artifact:
  - `debug_report/Vinh_set1_persondet_offline_50s_to_90s_v10_multimode_role_memory.mp4`
- Good outcome:
  - fixed the `34s` regression where `A` was still visible but became `missing`
  - fixed the `36s -> 38s` regression where a compact-but-valid representation of `A` was not accepted
- Important lesson:
  - role identity cannot rely on one averaged prototype

### 4. Role ownership gating
- Artifact:
  - `debug_report/Vinh_set1_persondet_offline_50s_to_90s_v11_role_ownership.mp4`
- Good outcome:
  - removed the current form of `A` jumping to the neighboring table on true leave
  - reintroduced proper `missing` at the reported `8s / 25s / 36s` relative timestamps
- Important lesson:
  - continuity alone is not enough
  - role ownership must be explicit and role-specific

## Failed / Rejected Directions

### 1. Zone-based person detection
- Attempt:
  - detect independently in near/far player zones
- Outcome:
  - helped one `Player B` case
  - regressed `Player A`
  - boxes became too small / unstable
- Status:
  - rejected

### 2. Partial-box guardrail tuning alone
- Attempt:
  - suppress tiny partial boxes
  - tweak guardrails around suspicious detections
- Outcome:
  - helped locally
  - did not solve the deeper identity / occlusion problem
- Status:
  - insufficient

### 3. Bridge-tracklet hacks
- Attempt:
  - create bridging continuity across missing gaps
- Outcome:
  - duplicated roles
  - revived neighboring-table swaps
  - created fake continuity
- Status:
  - rejected

### 4. Display-hold timeline hacks
- Artifacts:
  - `debug_report/Vinh_set1_persondet_offline_50s_to_90s_v5_occlusion_hold_anchor.mp4`
  - `debug_report/Vinh_set1_persondet_offline_50s_to_90s_v6_occlusion_hold_longer.mp4`
- Outcome:
  - only froze the last good box
  - did not solve true identity continuity
  - risked regressing correct `missing`
- Status:
  - rejected

### 5. Local render-only tracker
- Artifact:
  - `debug_report/Vinh_set1_persondet_offline_50s_to_90s_v7_local_occlusion_tracker.mp4`
- Outcome:
  - visually smoother than display hold
  - still the wrong layer
  - still not trustworthy enough
- Status:
  - rejected

### 6. Segment timeline solver
- Artifact:
  - `debug_report/Vinh_set1_persondet_offline_50s_to_90s_v9_segment_timeline_solver.mp4`
- What it changed:
  - split raw tracklets into many shorter segments
  - solved role assignment over segment paths
- Why it looked promising:
  - it unified tracklet continuity and role export more cleanly on paper
- Why it failed in practice:
  - over-fragmented the scene
  - exploded the `50s -> 90s` window from `58` tracklets to about `130` segment units
  - weakened long continuity priors
  - made too many left-side paths look plausible for `A`
  - regressed early `Stream 3 / Player B`
  - made `Stream 2 / Player A` jump across neighboring-table people
- Rollback artifact:
  - `debug_report/Vinh_set1_persondet_offline_50s_to_90s_v8_restored_after_v9_rollback.mp4`
- Status:
  - rejected

## Important Root-Cause Findings To Remember

### A. Full-box area is not a trustworthy identity gate
- A normal table-tennis action can enlarge the full person box:
  - reaching for the ball
  - serve follow-through
  - leaning toward the table
- Hard-gating on full-box area caused false `A missing` when the player was still validly visible.

### B. Post-hoc hard role filtering was architecturally wrong
- The old `_filter_role_observations(...)` could erase observations from an already assigned role tracklet.
- That created multiple competing truths:
  - raw continuity
  - role assignment
  - post-hoc export filter
- This was one root cause of the later `34s` regression.

### C. Global reasoning is useful only if it reduces ambiguity
- The failed segment solver did not fail because global reasoning is bad.
- It failed because it created too many ambiguous units.
- Rule:
  - do not add an identity layer unless it simplifies the scene

### D. True leave and short occlusion are different states
- They must not be handled by the same fallback behavior.
- Short occlusion:
  - keep identity alive when evidence is strong
- True leave:
  - prefer `missing` over borrowing another player

### E. `A` and `B` need role-specific modeling
- `A` is near-camera:
  - larger box
  - stronger lower-body / shoe evidence
  - stronger bottom-center cue
- `B` is farther:
  - smaller box
  - easier to lose under table occlusion
  - needs softer thresholds and different emphasis

## Current Implementation Summary
- Role assignment now uses role-specific ownership checks.
- Observation selection now rejects candidates that violate role ownership.
- True-leave protection now blocks some false `occluded` bridges across edge exit / re-entry patterns.
- Added regression tests for:
  - `A true leave -> missing instead of neighboring-table capture`
  - `B` continuity not dropping under the new ownership rules

## Known Remaining Risks
- The latest fix is verified on the key `50s -> 90s` window, but not yet promoted as a final full-match answer.
- Full `set1` should still be reviewed visually for:
  - early `Stream 3 / Player B`
  - any delayed `A` reacquire after long leave
  - any new edge cases outside the tracked debug window
- Any future change must re-check:
  - `29s` relative short foreground occlusion
  - `8s / 25s / 36s` relative true-leave cases

## Artifacts Worth Remembering
- Baseline:
  - `debug_report/Vinh_set1_persondet_offline_50s_to_90s_v2_fixA_missing.mp4`
- Occlusion improvement:
  - `debug_report/Vinh_set1_persondet_offline_50s_to_90s_v8_global_occlusion_solver.mp4`
- Failed segment refactor:
  - `debug_report/Vinh_set1_persondet_offline_50s_to_90s_v9_segment_timeline_solver.mp4`
- Multi-mode identity:
  - `debug_report/Vinh_set1_persondet_offline_50s_to_90s_v10_multimode_role_memory.mp4`
- Current role-ownership output:
  - `debug_report/Vinh_set1_persondet_offline_50s_to_90s_v11_role_ownership.mp4`
- Wrong-window artifact to ignore for this bug:
  - `debug_report/Vinh_set1_persondet_offline_0s_to_40s_v11_role_ownership.mp4`

## Resume Point
1. Start from the current `v11 role ownership` direction.
2. Use the correct validation clip:
   - `debug_report/Vinh_set1_persondet_offline_50s_to_90s_v11_role_ownership.mp4`
3. Keep the existing regression guardrails:
   - short occlusion around relative `29s`
   - true leave around relative `8s / 25s / 36s`
   - no neighboring-table capture
   - no `Player B` regression
4. If a new bug appears now, investigate in this order:
   - raw linker / true presence
   - role ownership
   - role observation selection
5. Do not revive rejected directions:
   - bridge hacks
   - display hold
   - render-only tracker
   - aggressive segment fragmentation
