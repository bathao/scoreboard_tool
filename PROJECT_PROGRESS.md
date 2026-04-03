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
  - `2026-04-03`
- Current production baseline:
  - `table / ROI-first`
- Current tracker baseline:
  - endpoint refinement on top of the accepted `starter + LET + active window` player-path baseline
- Current code state:
  - production draft path remains table-first
  - current accepted independent `YOLO player` start + `LET` baseline on checked sets:
    - `set1 = 14 rallies`, `LET = 1`
    - `set2 = 19 rallies`, `LET = 0`
    - `set3 = 18 rallies`, `LET = 0`
    - `set4 = 20 rallies`, `LET = 3`
    - scope:
      - accepted on the reviewed `set1..4` suite only
      - `starter_role` is exported in the latest player-path drafts
      - `LET` is inferred after starter detection
  - latest pushed endpoint checkpoint:
    - commit `0990559`
    - `Stabilize first ten set4 endpoint reviews`
  - current endpoint regression suite:
    - `matches/ground_truth/timeline_regression_suite.json`
    - `set4_frozen_full`
      - required no-regression suite
    - `set1_reviewed_first6`
      - required no-regression suite from the accepted reviewed first-six batch
  - current accepted `set4` endpoint checkpoint covers:
    - full `pt_0001 .. pt_0020`
    - accepted `t_end` list:
      - `7.474`
      - `15.949`
      - `28.762`
      - `37.237`
      - `58.859`
      - `82.449`
      - `93.760`
      - `102.536`
      - `127.361`
      - `135.469`
      - `146.113`
      - `156.990`
      - `168.268`
      - `180.080`
      - `191.224`
      - `202.402`
      - `223.223`
      - `232.766`
      - `241.742`
      - `260.093`
  - experimental multistream code now includes:
    - role-aware table refinement
    - standalone `player-only` draft mode for benchmark-only compare
    - experimental `player-only` start-image candidate mining for `Toss & Serve`
    - classical `ball tracking V0`
    - standalone `ball-only` draft mode for benchmark-only compare
  - local `Qwen` review support now also exists:
    - `qwen3-vl:8b` installed in Ollama for vision review
    - `qwen3:14b` installed in Ollama for reasoning review
    - default local vision model config was switched from `llama3.2-vision` to `qwen3-vl:8b`
    - `review_rally_splits_qwen.py` now supports `--skip-models` for candidate-only benchmarking
  - latest kept rally timeline outputs are:
    - `matches/Vinh_set1_rally_timeline.json`
    - `matches/Vinh_set2_rally_timeline.json`
    - `matches/Vinh_set3_rally_timeline.json`
    - `matches/Vinh_set4_rally_timeline.json`
  - older debug outputs still exist only as reference material and do not change the promoted baseline
  - current work-cycle constraint:
    - keep `table / ROI-first` unchanged
    - keep `ball tracking V0` unchanged
    - change the rally algorithm only in the independent `YOLO player` path for now
    - keep accepted `set4` endpoints frozen while tuning `set1`
  - role and ball paths remain experimental and are not promoted baselines yet
- Current last confirmed test result:
  - latest targeted multistream + contract tests:
    - `49 passed, 1 warning`
  - latest contract tests:
    - `3 passed, 1 warning`
  - note:
    - targeted tests were re-run on `2026-04-03` with `.venv\Scripts\python.exe -m pytest`
    - the warning remains a `.pytest_cache` permission warning inside the workspace
    - no fresh full-suite rerun was completed after the current endpoint cycle

## Work Log - `2026-04-03`
### Experiments That Passed
- `set1 pt_0007 improved without regressing frozen suites`
  - new current result:
    - `pt_0007 = 69.336`
    - `endpoint_mode = terminal_body_split_start`
  - previous current value before this pass:
    - `71.471`
  - current interpretation:
    - `pt_0007` is not another `ball-only false tail`
    - it behaves more like an `inside-run terminal split`
  - guardrails stayed green after the change:
    - `set4_frozen_full = 20/20`
    - `set1_reviewed_first6 = 6/6`
- `endpoint regression guardrails stayed usable`
  - current frozen machine-checkable suites remain:
    - `set4_frozen_full`
    - `set1_reviewed_first6`
  - these are still the required keep/drop decision criteria before keeping any endpoint patch
- `review batch for the next set1 expansion was produced`
  - current batch under review:
    - `debug_report/Vinh_set1_rally_clips_endpoint_pt7_10_current`
  - current batch values are:
    - `pt_0007 = 69.336`
    - `pt_0008 = 74.808`
    - `pt_0009 = 90.757`
    - `pt_0010 = 95.662`

### Experiments That Failed
- `pt_0009 is still the main unresolved blocker in the current batch`
  - operator feedback remains:
    - `pt_0009` is still late by about `6s`
  - current status:
    - no accepted improvement yet
    - current output remains:
      - `pt_0009 = 90.757`
      - `endpoint_mode = dead_reset_run_start`
- `two quick trace attempts failed before the real pt_0009 analysis even started`
  - first attempt:
    - called `extract_multistream_signals()` with the wrong `mode` argument
    - result:
      - `TypeError`
  - second attempt:
    - used the wrong table-weight path
    - result:
      - `FileNotFoundError`
- `the first full real-signal trace of pt_0009 was interrupted`
  - after fixing the API call and weights path, a long full-signal trace for `pt_0009` was launched on real `set1` data
  - result:
    - the run took several minutes and was interrupted before completion
    - no usable diagnostic artifact was saved from that attempt
  - conclusion:
    - `pt_0009` still needs a proper real-signal archetype trace
    - do not guess the fix from thresholds alone

### Current Diagnosis
- `pt_0007` and `pt_0009` are different endpoint archetypes
  - `pt_0007` responded to a narrow `terminal_body_split` style fix
  - `pt_0009` still appears to be a separate case and should not be forced through the same rule blindly
- `pt_0009` is now the highest-priority endpoint debug target
  - do not widen current rules until its archetype is identified on real support series
- `set4_frozen_full` and `set1_reviewed_first6` must continue to stay green
  - these suites remain the hard stop before keeping any further patch

### Exact Resume Point
- first:
  - rerun the real-signal `pt_0009` trace with the correct table weights:
    - `weights/yolov8x_table.pt`
- then:
  - dump the real `competitive / dead / terminal` runs for `pt_0009`
  - decide whether the next fix should be:
    - `inside-run split`
    - `post-body pseudo-live suppression`
    - or a new narrow archetype
- keep/drop rule:
  - keep a patch only if:
    - `set4_frozen_full` stays green
    - `set1_reviewed_first6` stays green
  - do not freeze `pt_0007 .. pt_0010` until the whole batch is reviewed
  - do not do another blind threshold sweep
  - do not repeat the long full trace without using the correct weights path and a long enough timeout

## Work Log - `2026-04-02`
### Experiments That Passed
- `set1 first six endpoints are now frozen as a guardrail`
  - accepted current `set1 pt_0001 .. pt_0006` values are now:
    - `9.176`
    - `12.880`
    - `29.630`
    - `38.405`
    - `47.981`
    - `57.858`
  - accepted current endpoint modes are:
    - `pt_0001 = dead_reset_run_start`
    - `pt_0002 = dead_reset_run_start`
    - `pt_0003 = dead_reset_run_start`
    - `pt_0004 = dead_reset_run_start`
    - `pt_0005 = dead_reset_run_start`
    - `pt_0006 = ball_only_false_tail_start`
  - `matches/ground_truth/timeline_regression_suite.json` was updated so:
    - `set1_reviewed_first6` is now `required`
- `endpoint regression suite scaffolding now exists`
  - added regression manifest:
    - `matches/ground_truth/timeline_regression_suite.json`
  - added checker:
    - `scripts/check_timeline_regression.py`
  - added pure comparison helper:
    - `backend/endpoint_regression.py`
  - added unit coverage:
    - `tests/test_endpoint_regression.py`
- `set4 frozen guardrail is now machine-checkable`
  - current checker result:
    - `set4_frozen_full` passes `20/20`
    - `max_abs_diff = 0.000s`
- `ball-only false-tail archetype is now separated from the general dead-reset path`
  - added a narrow endpoint branch for:
    - high ball
    - very low table/live/effective interaction
    - high reset
    - high terminal-body
    - weak long-gap pseudo-resume afterward
  - current effect:
    - `set1 pt_0006` now lands at the accepted `57.858`
    - `set4_frozen_full` still passes `20/20`

### Experiments That Failed
- `trying to fix set1 pt_0006 by global endpoint rules still caused set4 regressions`
  - broad and medium-scope endpoint tweaks could improve `set1 pt_0006`
  - but the same tweaks regressed accepted `set4` points such as:
    - `pt_0003`
    - `pt_0015`
    - `pt_0016`
    - `pt_0017`
    - `pt_0020`
  - conclusion:
    - stop global threshold tuning
    - move to a mode-based endpoint engine with explicit regression checks
- `reopened-after-early-dead terminal-body branch`
  - a branch that allowed `terminal_body_split_start` to ignore an earlier viable dead-run if a later competitive run existed was tested
  - result:
    - it improved some synthetic behavior
    - but it regressed frozen `set4` points such as:
      - `pt_0003`
      - `pt_0009`
      - `pt_0015`
      - `pt_0016`
  - action taken:
    - reject that branch
    - return `terminal_body_split_start` to the stable behavior

### Current Diagnosis
- `set4_frozen_full` and `set1_reviewed_first6` are now both frozen no-regression suites
- the next useful work is batch-wise `set1` expansion:
  - review `pt_0007 .. pt_0010`
  - then promote them into the suite
- the current architecture direction is still correct:
  - use archetype-specific endpoint branches
  - keep guardrails machine-checkable
  - avoid global threshold tuning

### Exact Resume Point
- run:
  - `.venv\\Scripts\\python.exe scripts\\check_timeline_regression.py`
- expected current state:
  - `set4_frozen_full` must stay green
  - `set1_reviewed_first6` must also stay green
- next implementation step:
  - export and review `set1 pt_0007 .. pt_0010`
  - only keep a patch if both frozen suites remain green

## Work Log - `2026-04-01`
### Experiments That Passed
- `set4 endpoint accepted checkpoint for the full set`
  - operator accepted the full `set4` rally boundary pass after iterative endpoint tuning
  - accepted final `t_end` values are:
    - `pt_0001 = 7.474`
    - `pt_0002 = 15.949`
    - `pt_0003 = 28.762`
    - `pt_0004 = 37.237`
    - `pt_0005 = 58.859`
    - `pt_0006 = 82.449`
    - `pt_0007 = 93.760`
    - `pt_0008 = 102.536`
    - `pt_0009 = 127.361`
    - `pt_0010 = 135.469`
    - `pt_0011 = 146.113`
    - `pt_0012 = 156.990`
    - `pt_0013 = 168.268`
    - `pt_0014 = 180.080`
    - `pt_0015 = 191.224`
    - `pt_0016 = 202.402`
    - `pt_0017 = 223.223`
    - `pt_0018 = 232.766`
    - `pt_0019 = 241.742`
    - `pt_0020 = 260.093`
  - latest kept rally timeline:
    - `matches/Vinh_set4_rally_timeline.json`
  - checkpoint committed and pushed:
    - `0990559`
    - `Stabilize first ten set4 endpoint reviews`
- `open-tail endpoint guard fixed the last-rally early-cut issue`
  - root case:
    - `pt_0020` is the final rally in the set
    - an early `dead-run` was being accepted even though strong late rally activity continued in the open tail
  - current fix:
    - add an `open-tail` rescue path so late strong live evidence can reject an early dead-run on the final rally
    - restrict the final fallback to stronger late competitive runs instead of blindly taking the weakest last fragment
  - result:
    - `pt_0020` moved from the too-early `257.691` to the accepted `260.093`
    - `pt_0001 .. pt_0019` remained unchanged
- `terminal body-language cues materially improved endpoint quality`
  - `face_hidden` was upgraded from pure missing-face logic to also use collapsed shoulder / hip span as a profile-turn proxy
  - `face_touch` remained as a light `wipe face / casual recovery` cue
  - `terminal_body_pair` remained useful as a pair-level disengagement cue
  - this branch was especially important for fixing the long-running `pt_0010` endpoint issue
- `accepted endpoint guardrail is now explicit`
  - later endpoint experiments must not regress accepted `pt_0001 .. pt_0010`
  - this rule is now the practical promotion blocker for later set4 endpoint changes

### Experiments That Failed
- `broad resume / reset threshold tuning after pt_0010`
  - multiple branches tried to improve `pt_0012 / pt_0015`
  - result:
    - they did not materially improve those two points
    - one branch regressed accepted points such as `pt_0003 / pt_0008`
    - another branch pulled `pt_0008` back toward a worse endpoint again
  - action taken:
    - reject those branches
    - restore workspace and latest `set4` JSON to the previously accepted checkpoint
- `adding reach / net-approach support directly into terminal endpoint cues`
  - hypothesis:
    - `pickup / lunge / moving toward the ball` might help terminate `pt_0012 / pt_0015`
  - result:
    - did not improve `pt_0012`
    - did not improve `pt_0015`
    - regressed `pt_0002`
  - action taken:
    - reject and restore

### Current Diagnosis
- the reviewed `set4` endpoint pass is now good enough to freeze as the current downstream baseline
- the next blocker is no longer `set4 endpoint`
- the next blocker is downstream `winner / point / score` logic on top of the frozen rally list

### Exact Resume Point
- keep the current accepted `set4` JSON unchanged as the frozen downstream baseline
- start moving downstream onto the accepted rally list:
  - winner inference
  - point flow
  - score progression
- avoid reopening `set4` endpoint unless new operator evidence appears

## Work Log - `2026-03-26`
### Experiments That Passed
- `starter_role` export and `LET` subtraction now work together on the reviewed four-set suite
  - accepted current results are:
    - `set1 = 14 rallies`, `LET = 1`
    - `set2 = 19 rallies`, `LET = 0`
    - `set3 = 18 rallies`, `LET = 0`
    - `set4 = 20 rallies`, `LET = 3`
  - latest kept rally timelines are:
    - `matches/Vinh_set1_rally_timeline.json`
    - `matches/Vinh_set2_rally_timeline.json`
    - `matches/Vinh_set3_rally_timeline.json`
    - `matches/Vinh_set4_rally_timeline.json`
  - accepted current `set4` `LET` timestamps are:
    - `01:51.778`
    - `01:55.048`
    - `03:36.516`
- `set3` role reassignment fix removed the false `LET` pattern
  - root cause:
    - `LET` was being inferred from a wrong `starter_role` run shape, not from missing starters
    - the suspicious shape looked like:
      - `BBB | A | BB`
  - fix:
    - add a conservative `double-serve` role-singleton repair before `LET` inference
    - only flip the edge of a `BBB | A | BB` or `BB | A | BBB` pattern when the edge candidate looks like a late follow-up, not a clean serve
  - accepted repaired `set3` serve pattern is now:
    - `BB | AA | BB | AA | BB | AA | BB | AA | BB`
  - accepted repaired timestamps that changed role interpretation:
    - `00:38.405`
    - `03:08.722`
  - result after the repair:
    - `set3 = 18 rallies`
    - `LET = 0`
  - targeted regression after the repair:
    - `.venv\Scripts\python.exe -m pytest tests/test_multistream_rally.py -q`
    - result:
      - `33 passed, 1 warning`

## Work Log - `2026-03-24`
### Experiments That Passed
- `independent YOLO player starter detector accepted on checked set1..4`
  - final accepted starter counts on the checked set suite are now:
    - `set1 = 15`
    - `set2 = 19`
    - `set3 = 18`
    - `set4 = 23`
  - scope of this acceptance:
    - `starter = rally + let`
    - only for the independent `multistream / YOLO player-signal` path
  - the last `set4` duplicate false positive at `4.872s` was removed
  - latest accepted `set4` review artifact:
    - `debug_report/Vinh_set4_rally_start_candidates_feedback_probe_v2/`
  - latest accepted probe artifacts:
    - `matches/Vinh_set1_stage1_player_independent_sandwich_set4_feedback_probe_v2.json`
    - `matches/Vinh_set2_stage1_player_independent_sandwich_set4_feedback_probe_v2.json`
    - `matches/Vinh_set3_stage1_player_independent_sandwich_set4_feedback_probe_v2.json`
    - `matches/Vinh_set4_stage1_player_independent_sandwich_set4_feedback_probe_v2.json`
  - regression remembered:
    - `set1 / set2 / set3` stayed exactly unchanged on timestamp lists while fixing `set4`
  - targeted regression after the final guard:
    - `.venv\Scripts\python.exe -m pytest tests/test_multistream_rally.py tests/test_ai_contract.py`
    - result:
      - `31 passed, 1 warning`
  - committed and pushed checkpoint:
    - commit `5e94835`
    - `Detect accurate starters for all 4 sets (server + let)`
- `set2 v14 operator review ingestion`
  - saved operator feedback at:
    - `debug_report/Vinh_set2_rally_start_candidates_v14_review/operator_feedback.json`
  - operator accepted all unmentioned `set2 v14` start images as correct
  - operator marked these `set2 v14` candidates as false positives:
    - `#5` at `29.062s`
    - `#8` at `56.323s`
    - `#12` at `111.979s`
    - `#14` at `124.124s`
    - `#20` at `192.693s`
    - `#22` at `202.002s`
  - operator description of the rejected frames:
    - no one is holding the ball
    - the ball is already on the table
    - both players are already in live chop / drive / loop preparation, not serve preparation
- `selector guard against already-live exchange false positives`
  - `backend/ai_multistream_rally.py` now rejects an additional class of `player_sandwich` starts where the scene is already rally-active before the chosen candidate
  - the new local rejection shape focuses on:
    - weak `pre_ready`
    - high `pre_live`
    - high live exchange continuation after the candidate
    - low / moderate dominance for the supposed server
  - also added a stricter high-action rejection for the `already-live attack` pattern seen in `set2`
- `targeted regression after set2 feedback`
  - `.venv\Scripts\python.exe -m pytest tests/test_multistream_rally.py tests/test_ai_contract.py`
  - result:
    - `28 passed, 1 warning`
- `set2 rerun after the feedback-driven selector change`
  - saved probe artifact:
    - `matches/Vinh_set2_stage1_player_independent_sandwich_v15_probe.json`
  - result:
    - `total_rallies = 16`
    - `LET = 0`
  - compared with `matches/Vinh_set2_stage1_player_independent_sandwich_v14.json`:
    - removed exactly the operator-rejected start timestamps:
      - `29.062`
      - `56.323`
      - `111.979`
      - `124.124`
      - `192.693`
      - `202.002`
    - added no new timestamps
- `set3 operator review ingestion`
  - saved operator feedback at:
    - `debug_report/Vinh_set3_rally_start_candidates_v14_review/operator_feedback.json`
  - operator accepted all unmentioned `set3 v14` start images as correct
  - operator marked these `set3 v14` candidates as false positives:
    - `#6` at `70.504s`
    - `#9` at `101.535s`
    - `#11` at `109.776s`
    - `#14` at `143.110s`
    - `#15` at `144.811s`
    - `#17` at `160.227s`
    - `#18` at `162.262s`
    - `#21` at `197.330s`
    - `#22` at `198.498s`
- `post-set2 selector tuning on set3 with set2 guardrail preserved`
  - kept the `set2`-driven live-exchange rejection in place
  - added a narrow `strong follow-up` exception so the selector can keep a stronger near-next start instead of over-pruning all high-live cases
  - added two narrower rejection patterns for:
    - `weak_opponent_mid_rally`
    - `post_rally_freeze`
  - targeted regression after the tuning:
    - `.venv\Scripts\python.exe -m pytest tests/test_multistream_rally.py tests/test_ai_contract.py`
    - result:
      - `28 passed, 1 warning`
- `set2 + set3 cross-check on the new selector snapshot`
  - saved probe artifacts:
    - `matches/Vinh_set2_stage1_player_independent_sandwich_v17_probe.json`
    - `matches/Vinh_set3_stage1_player_independent_sandwich_v17_probe.json`
  - `set2` stayed exactly aligned with the accepted `v15` probe:
    - `16` rallies
    - no timestamps added
    - no timestamps removed
  - `set3` improved from `24` to `18` while preserving accepted starts
  - removed reviewed `set3` false positives:
    - `70.504`
    - `109.776`
    - `143.110`
    - `144.811`
    - `162.262`
    - `198.498`
  - no reviewed-correct `set3` timestamp was removed
  - remaining reviewed `set3` hard false positives are now:
    - `101.535`
    - `160.227`
    - `197.330`

### Main Findings From Today
- the most important `set2` false-positive family is no longer generic `stroke-like` motion
- the sharper failure mode is:
  - the selector is picking frames that are already inside a live exchange
  - ball state is visually inconsistent with serve prep
  - player motion remains strong before and after the chosen timestamp
- the reviewed `set2` labels were enough to recover the draft count from `22` to `16` without introducing extra starts
- the additional `set3` review labels were enough to push `set3` from `24` to `18` without breaking the current `set2` guardrail
- the remaining `set3` mistakes are now concentrated in only `3` hard cases:
  - `101.535`
  - `160.227`
  - `197.330`
- the next debug direction should no longer be broad threshold tuning
- the next likely win is a more local duplicate / cluster suppression pass around nearby starts, while keeping `set2` frozen as the promotion guardrail

## Work Log - `2026-03-23`
### Experiments That Passed
- `player-only start-first tuning on set1`
  - the independent `YOLO player` path was iterated through `sandwich_v4 .. sandwich_v9`
  - the current kept snapshot is:
    - `matches/Vinh_set1_stage1_player_independent_sandwich_v9.json`
  - current `set1` result:
    - `total_rallies = 12`
    - kept `t_start` list:
      - `3.670`
      - `10.210`
      - `20.254`
      - `25.158`
      - `43.377`
      - `54.288`
      - `81.581`
      - `92.259`
      - `104.204`
      - `125.659`
      - `135.836`
      - `145.145`
  - the false positive around `44.845s` is currently removed
- `start detector quality improvements`
  - the raw miner is now more `prep-first`
  - the selector now uses stronger `pre-ready / pre-live / post-growth / live-exchange` confirmation
  - `opponent_ready` remains a guard against rally-active false positives
  - `clean prep rescue` now recovers visually good serve-prep starts that had weak `opponent_ready`
  - narrower cross-role merge windows now preserve nearby real starts instead of swallowing them
- `saved operator-review artifact for tomorrow`
  - the current kept review folder is:
    - `debug_report/Vinh_set1_rally_start_candidates_v9_review/`
  - it contains:
    - `12` start images
    - `1` CSV with the current detector scores per candidate
- `targeted regression rerun on current snapshot`
  - `tests/test_multistream_rally.py` + `tests/test_ai_contract.py`:
    - `27 passed, 1 warning`
  - the only observed warning is still `.pytest_cache` access denied

### Experiments That Failed Or Were Rejected
- `player-only sandwich v1 on set1`
  - severe over-split at `34` rallies
- `player-only sandwich v7 on set1`
  - over-corrected and collapsed to only `3` rallies
- `player-only sandwich v9 on set1`
  - still under-detects clearly for this set
  - it is a better debug snapshot, not a promotion candidate

### Main Findings From Today
- the current bottleneck is still `start` selection quality, not JSON export plumbing
- the most useful next signal is manual review on the saved `12` `set1` start images
- keep `table / ROI-first` frozen for this cycle
- keep `ball tracking V0` frozen for this cycle
- keep all algorithm work inside the independent `YOLO player` path until the reviewed start detector is healthier

## Work Log - `2026-03-22`
### Experiments That Passed
- `workspace state verification`
  - local HEAD is still `e7ea372`
  - current branch focus remains aligned with the docs:
    - production draft baseline is still `table / ROI-first`
    - the `player` path is still in the `start-first` debug phase
- `targeted regression rerun on current workspace`
  - `tests/test_multistream_rally.py` + `tests/test_ai_contract.py`:
    - `22 passed, 1 warning`
  - `tests/test_offline_player_tracker.py`:
    - `15 passed, 1 warning`
  - the only observed warning is `.pytest_cache` access denied
- `artifact verification for latest debug outputs`
  - `matches/Vinh_set3_stage1_player_independent_v2.json` still reports:
    - `total_rallies = 4`
  - `matches/Vinh_set4_stage1_player_independent_v2.json` still reports:
    - `total_rallies = 4`
  - `matches/Vinh_set4_qwen_split_review_report_v0b.json` reports:
    - `input_count = 18`
    - `output_count = 18`
    - `candidate_count = 2`
  - current conclusion stays the same:
    - no new promoted baseline
    - no accepted `Qwen` split policy yet
- `player-only sandwich detector wiring`
  - `mode=player` with `player_signal_source=role_tracker` now routes through a new independent `player_sandwich` detector
  - the new detector currently uses:
    - pose-driven start candidates
    - swing confirmation window
    - posture-reset end detection
    - short-rally `LET` classification
    - forced close at `Start(n+1)` when no clean end is found
- `targeted sandwich detector test coverage`
  - added focused tests for:
    - reset-based close
    - forced close at next start
    - short `LET`
    - `mode=player` dispatch
  - current test result:
    - `tests/test_multistream_rally.py`: `24 passed, 1 warning`
    - `tests/test_ai_contract.py`: `2 passed, 1 warning`

### Experiments That Failed Or Were Rejected
- `player-only v2 full-run artifacts`
  - the saved `set3` and `set4` `v2` artifacts both still collapse to only `4` rallies
  - treat them as failed experimental outputs, not forward progress for Stage 1 quality
- `set4 qwen split-review v0b`
  - the saved `v0b` rerun still keeps the output at `18`
  - only `2` candidates were actually reviewed in the saved report
  - this still does not recover the missing-rally gap versus the debug target `20`
- `player-only sandwich v1 on set1`
  - output artifact:
    - `matches/Vinh_set1_stage1_player_independent_sandwich_v1.json`
  - current result:
    - `total_rallies = 34`
    - `scoring_rallies = 34`
    - `non_scoring_rallies = 0`
  - current local GT read for `set1` is still `14`
  - current failure shape:
    - severe over-split
    - many adjacent rallies are separated by tiny gaps
    - `LET` logic did not fire on this first set1 run
  - quick symptom check on the saved JSON:
    - `16` segments are shorter than `3s`
    - `26` segments are shorter than `5s`
    - `29` inter-segment gaps are below `0.2s`
  - current read:
    - the detector now runs end-to-end and exports valid JSON
    - but start anchors are still too dense and are chaining false boundaries

### Main Findings From Today
- docs are now aligned with the current local HEAD and verified targeted test state
- no new evidence today changes the production baseline away from `table / ROI-first`
- the next resume point remains:
  - keep `table / ROI-first` frozen for this cycle
  - keep `ball tracking V0` frozen for this cycle
  - improve the `player` `start-first` detector
  - reduce false-positive start anchors before trusting sandwich closure behavior
  - add `LET` subtraction
  - then return to 3-detector fusion
- `Qwen` split review remains deferred until the independent `player` detector is materially healthier

## Work Log - `2026-03-21`
### Experiments That Passed
- `player-only start-first detector reset`
  - the experimental `YOLO player` branch was reset away from the failed long-`active` rally state machine
  - the current detector now focuses only on finding `Toss & Serve` start images
  - new code exports `PlayerRallyStartCandidate` items from per-role signals:
    - crouch / ready posture
    - reach toward the table
    - serve cue
    - upper-body activity
    - footwork
    - opponent-ready context
    - same-role vs opposite-role dominance
  - the current temporary counting doctrine is now:
    - `total starts = rallies + LET`
    - `total rallies = total starts - LET`
    - `active` should later be bounded only between consecutive detected starts
- `player-only start-image export tooling`
  - added `scripts/export_player_rally_start_candidates.py`
  - this exports one annotated image + timestamp per detected start candidate
  - current kept artifacts:
    - `debug_report/Vinh_set4_rally_start_candidates_v1_first80/`
    - `debug_report/Vinh_set4_rally_start_candidates_v1_full/`
  - current counts from that detector are:
    - first `80s`: `18` start candidates
    - full `set4`: `72` start candidates
- `checked serve-start examples on set4`
  - the new start-image detector now catches the operator-confirmed examples:
    - `3.103s`
    - `12.279s`
    - `25.859s`
    - `33.967s`
  - these are exported as annotated images and recorded in CSV under the new debug-report folders
- `experimental player-only state machine v2 wiring`
  - `backend/ai_multistream_rally.py` now contains a role-aware player-only state machine that uses:
    - `motion`
    - `crouch / ready`
    - `serve`
    - `upper-body`
    - `footwork`
    - `reach / catch proxy`
    - `net-approach proxy`
  - the branch still uses the existing role tracker:
    - `Stream 2 = Player A`
    - `Stream 3 = Player B`
  - role assignment logic in `backend/offline_player_tracker.py` was **not** changed in this branch
- `let-label contract plumbing`
  - experimental player-only segments can now carry:
    - `rally_label_let`
    - `let_no_score`
  - the contract layer now skips those segments when converting to scoring `RallyEvent`
- `player-state debug artifact export`
  - added `scripts/export_player_state_machine_debug.py`
  - this can export:
    - annotated MP4 with:
      - `Player A / Player B` boxes and keypoints
      - current state-machine phase
      - feature values used by the detector
      - segment timeline overlay
    - per-sampled-frame CSV diagnostics
  - current artifacts:
    - `debug_report/Vinh_set4_player_state_machine_debug_first80.mp4`
    - `debug_report/Vinh_set4_player_state_machine_debug_first80.csv`
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
  - added a real `--mode ball` path in `scripts/generate_rally_timeline.py`
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
  - multistream rally tests now pass at `20 passed, 1 warning`
  - latest previously confirmed full suite remains `70 passed, 1 warning`

### Experiments That Failed Or Were Rejected
- `player-only state machine v2 on set4`
  - current status is `very poor`
  - full-run artifact:
    - `matches/Vinh_set4_stage1_player_independent_v2.json`
  - full clip output collapsed to only `4` rallies:
    - `0.934 -> 43.877`
    - `47.180 -> 209.576`
    - `212.746 -> 249.983`
    - `256.022 -> 264.764`
  - first `80s` debug window still collapsed to only `2` rallies:
    - `3.103 -> 43.844`
    - `47.214 -> 79.980`
  - this is not remotely realistic rally timing for table tennis and must be treated as a failed experimental state
  - current failure shape:
    - severe over-merge
    - `active` state held too long
    - `dead_now` can be true while `live_now` is also true
  - likely immediate debug target:
    - `live / dead / end-casual` interaction
    - not role-assignment redesign
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
- the current experimental `player-only state machine v2` is not usable for rally segmentation quality
- the `player` branch is now being reset to a simpler `start-first` doctrine:
  - first detect all visually obvious `Toss & Serve` starts
  - treat those starts as `rally + let`
  - then detect `LET` as a subtraction pass
  - only after that define `active` between consecutive starts
- the latest failure on `set4` is dominated by `over-merge`, not by a proven `Player A / Player B` stream-mapping bug
- current evidence says:
  - `Stream 2 / Stream 3` role assignment remained unchanged
  - the new bug is in the decision logic that keeps rallies alive too long
- the new start-image detector is already materially better for debug than the old full-rally state machine:
  - it catches the operator-confirmed start examples around `3s`, `12s`, `25s`, and `34s`
  - it still over-generates many additional candidates and is not yet a rally counter
- debug next session should focus on:
  - pruning false-positive start candidates
  - stabilizing `start_count = rally + let`
  - then adding `LET` subtraction before returning to bounded `active` logic
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
- `matches/Vinh_set1_stage1_player_independent_sandwich_v9.json`
- `matches/Vinh_set3_stage1_player_independent_v2.json`
- `matches/Vinh_set4_stage1_player_independent_v2.json`
- `matches/Vinh_set4_qwen_split_reviewed_v0b.json`
- `matches/Vinh_set4_qwen_split_review_report_v0b.json`
- `debug_report/Vinh_set1_rally_start_candidates_v9_review/`
- `debug_report/Vinh_set4_rally_start_candidates_v1_first80/`
- `debug_report/Vinh_set4_rally_start_candidates_v1_full/`
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
  - current `qwen` split-review `v0b` result is also still `18`
  - the saved `v0b` report only reviews `2` split candidates
  - current `qwen` split candidate-only rerun now surfaces `11` candidates
  - current read:
    - merge-only `Qwen` review is clearly worse
    - standalone `ball-only` recovers missing-rally pressure better than table-only, but still over-counts
    - split-review direction is still unresolved
    - candidate accept / reject quality is still the bottleneck
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
  - saved `Qwen` split-review `v0b` still keeps the output at `18`
  - current `Qwen` split-review logic still needs trustworthy candidate acceptance before the models can help

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
   - the independent `YOLO player` path now has an accepted reviewed `starter + LET` baseline on checked `set1..4`:
     - `set1 = 14 rallies`, `LET = 1`
     - `set2 = 19 rallies`, `LET = 0`
     - `set3 = 18 rallies`, `LET = 0`
     - `set4 = 20 rallies`, `LET = 3`
   - latest kept player-path drafts are:
     - `matches/Vinh_set1_rally_timeline.json`
     - `matches/Vinh_set2_rally_timeline.json`
     - `matches/Vinh_set3_rally_timeline.json`
     - `matches/Vinh_set4_rally_timeline.json`
   - Stage 1 now has 3 explicit detector paths:
     - `table / ROI-first` as the already-existing reference detector
   - `multistream / YOLO player-signal` now has a reviewed `start + LET` baseline on `set1..4`
   - standalone `ball-only v7` is still benchmark / compare code only
   - conservative `table_ball_refined` remains experimental
   - local `Qwen` review scripts still exist but are not the current critical path
2. Do not re-open `table / ROI-first` as if it still needs to be created:
   - it already exists
   - use it as detector `#1` in the Stage 1 compare / fusion plan
3. Next critical Stage 1 work is:
   - do not change `table / ROI-first` in this cycle
   - do not retune `ball tracking V0` in this cycle
   - use both only as fixed references while debugging `player`
   - keep the accepted `set1..4` `starter + LET` baseline frozen as the current guardrail
   - preserve the key `set3` lesson:
     - wrong `LET` can come from wrong `starter_role`, not from missing starters
     - the conservative `double-serve` role-singleton repair must remain before `LET` inference
   - only now redefine `active` between consecutive accepted starts
   - then continue downstream point / winner / score work on top of that fixed rally list
4. After the `player` start-first branch is stable enough:
   - compare `table`, `multistream`, and standalone `ball-only` on the same reviewed sets
   - align their rally lists by time overlap, not only by count
   - optimize for ordered rally list and boundary quality, not count alone
5. After the 3-detector compare is stable enough:
   - define the first fusion / validation rule that merges the 3 rally lists into one final rally list
   - benchmark the fused list against the independent detectors
6. Do not promote standalone `ball-only` as the new baseline yet:
   - it is still benchmark / diagnosis code
   - it still over-counts known clips
7. For the reviewed player-path suite, keep these accepted compare targets visible:
   - `set1 = 14 rallies`, `LET = 1`
   - `set2 = 19 rallies`, `LET = 0`
   - `set3 = 18 rallies`, `LET = 0`
   - `set4 = 20 rallies`, `LET = 3`
8. Continue the unfinished `Qwen` split-review direction only after the current `player` work is checked:
   - the saved `v0b` rerun still keeps the output at `18` with only `2` reviewed candidates
   - define a conservative accept / reject policy on the broader candidate list before trusting model output
   - only keep completed reruns as evidence
   - do not let it displace the current independent `YOLO player` debug cycle
9. Keep the deferred `set1` tracker failure visible:
   - `1:34 -> 1:47`
   - do not treat it as fixed
   - do not let detector benchmarking hide it
10. Do not fix it with:
   - render smoothing
   - frozen boxes
   - continuity hacks
   - clip-specific thresholds without broader justification
11. If a real root-cause fix is found, rerun full `set1`, `set2`, `set3`, and `set4`.

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

## 2026-04-03
- `all four reviewed sets now have accepted rally timelines`
  - `set1 = 14 rallies`, `LET = 1`
  - `set2 = 19 rallies`, `LET = 0`
  - `set3 = 18 rallies`, `LET = 0`
  - `set4 = 20 rallies`, `LET = 3`
- `set2 pt_0019` was fixed as an `open-tail fragmented resume`
  - new endpoint: `203.737`
  - this change preserved `set1` and `set4` exactly
- `set3 pt_0009` was fixed as a `post_dead_plateau_start`
  - new endpoint: `110.210`
  - this change preserved `set1`, `set2`, and `set4` exactly
- `canonical naming cleanup`
  - active generator entrypoint is now `scripts/generate_rally_timeline.py`
  - active checker entrypoint is now `scripts/check_timeline_regression.py`
  - active outputs are now:
    - `matches/Vinh_set1_rally_timeline.json`
    - `matches/Vinh_set2_rally_timeline.json`
    - `matches/Vinh_set3_rally_timeline.json`
    - `matches/Vinh_set4_rally_timeline.json`
- verification
  - `50 passed, 1 warning`
  - `scripts/check_timeline_regression.py`:
    - `set4_frozen_full = 20/20`
    - `set1_reviewed_first6 = 6/6`
- next resume point
  - expand the regression suite toward full `set1..4`
  - start the first `winner / point-state` pass on top of the frozen rally timelines
