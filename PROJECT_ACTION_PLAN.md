# Project Action Plan

## Purpose
Use this file as the short operational board for the project.

Keep this file focused on:
- what is `done`
- what is `doing`
- what is `todo`
- what is `deferred`
- what must not regress

Put detailed explanations, experiments, failures, and resume notes in:
- `PROJECT_PROGRESS.md`

## Status Legend
- `[done]`
- `[doing]`
- `[todo]`
- `[deferred]`
- `[blocked]`
- `[rejected]`

## Current Baseline
- Production baseline:
  - `table / ROI-first`
- Current algorithm-change area:
  - independent `multistream / YOLO player-signal` path only
- Keep unchanged for now:
  - `table / ROI-first`
  - `ball tracking V0`
- Current accepted `starter + LET` baseline:
  - `set1 = 14 rallies`, `LET = 1`
  - `set2 = 19 rallies`, `LET = 0`
  - `set3 = 18 rallies`, `LET = 0`
  - `set4 = 20 rallies`, `LET = 3`
- Latest pushed checkpoint:
  - commit `4e70e0f`
  - `Detect rallies correctly for all four reviewed sets`
- Latest local checkpoint:
  - commit `bfe9fa9`
  - `Freeze temporary rally timestamps for all four sets`
- Endpoint regression suite baseline:
  - `set1_frozen_full` = required no-regression suite
  - `set2_frozen_full` = required no-regression suite
  - `set3_frozen_full` = required no-regression suite
  - `set4_frozen_full` = required no-regression suite
- Current focus:
  - keep the current post-followup rally timestamps for `set1..4` frozen
  - resume winner detection using `Transformers native-video`
  - use `Qwen3-VL-4B-Instruct` as the main model
  - keep `Qwen3-VL-8B-Instruct` as backup only if `4B` is not good enough

## Done
- `[done]` starter detection accepted on reviewed `set1..4`
- `[done]` `starter_role` export works in the latest player-path timelines
- `[done]` `LET` inference and subtraction accepted on reviewed `set1..4`
- `[done]` `active window` layer implemented
- `[done]` `accepted_start -> next accepted_start` ownership window is implemented
- `[done]` preceding `LET` attempts are attached to the next accepted rally
- `[done]` `set4` endpoint is now operator-accepted for the full set
- `[done]` accepted current `set4` endpoint values for `pt_0001 .. pt_0010`:
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
- `[done]` accepted current `set4` endpoint values for `pt_0011 .. pt_0020`:
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
- `[done]` endpoint regression manifest created:
  - `matches/ground_truth/timeline_regression_suite.json`
- `[done]` endpoint regression checker created:
  - `scripts/check_timeline_regression.py`
- `[done]` rally detection is accepted for `set1`
- `[done]` rally detection is accepted for `set2`
- `[done]` rally detection is accepted for `set3`
- `[done]` rally detection is accepted for `set4`
- `[done]` canonical rally timeline naming is now in place for the active flow
- `[done]` legacy `Draft* / draft / generate_draft*` naming has been removed from the active rally timeline flow
- `[done]` canonical timeline regression naming is now in place:
  - `backend/timeline_regression.py`
  - `scripts/check_timeline_regression.py`
  - `tests/test_timeline_regression.py`
  - `tests/test_rally_timeline_contract.py`
- `[done]` `set3 pt_0009` is fixed without regressing `set1 / set2 / set4`
- `[done]` full `set1..4` rally baseline is now frozen in `timeline_regression_suite.json`
- `[done]` current `set4` rally endtime patch is operator-accepted for review use
- `[done]` accepted current fresh `set4` review batch:
  - `debug_report/Vinh_set4_fresh_full_rallies_endtime_debug_current_v2`
- `[done]` current post-followup full-rally review batches are temporarily accepted:
  - `debug_report/Vinh_set1_fresh_full_rallies_post_followup_current`
  - `debug_report/Vinh_set2_fresh_full_rallies_post_followup_current`
  - `debug_report/Vinh_set3_fresh_full_rallies_post_followup_current`
  - `debug_report/Vinh_set4_fresh_full_rallies_post_followup_current`
- `[done]` freeze the current post-followup rally timestamps into:
  - `matches/Vinh_set1_rally_timeline.json`
  - `matches/Vinh_set2_rally_timeline.json`
  - `matches/Vinh_set3_rally_timeline.json`
  - `matches/Vinh_set4_rally_timeline.json`
  - `matches/ground_truth/timeline_regression_suite.json`

## Doing
- `[doing]` keep the current post-followup `set1..4` timelines frozen as the temporary accepted timestamp baseline
- `[doing]` resume `detect winner = Transformers native-video` on top of the frozen `set1..4` rally boundaries
- `[doing]` use `Qwen3-VL-4B-Instruct` as the active main path for winner work
- `[doing]` keep `Qwen3-VL-8B-Instruct` downloaded as backup only
- `[doing]` start the next winner cycle from `set1`, then expand to `set2 / set3 / set4`
- `[doing]` use only these four folders as the active review artifacts:
  - `debug_report/Vinh_set1_fresh_full_rallies_post_followup_current`
  - `debug_report/Vinh_set2_fresh_full_rallies_post_followup_current`
  - `debug_report/Vinh_set3_fresh_full_rallies_post_followup_current`
  - `debug_report/Vinh_set4_fresh_full_rallies_post_followup_current`

## Todo
- `[todo]` if rally boundaries are reopened later, rerun from source video end-to-end and do not reuse intermediate timeline JSON files
- `[todo]` keep winner inference strictly downstream of the frozen `set1..4` timestamp checkpoint
- `[todo]` build `score progression` on top of accepted rallies + inferred winners
- `[todo]` run the same pipeline on a single long multi-set input, not only split sets
- `[todo]` start correction / UI flow only after rally + winner + score are usable

## Deferred
- `[deferred]` do not retune `table / ROI-first` in this cycle
- `[deferred]` do not retune `ball tracking V0` in this cycle
- `[deferred]` do not work on 3-detector fusion yet
- `[deferred]` do not spend this cycle on `Qwen` boundary/split tuning
- `[deferred]` do not start Web UI work yet
- `[deferred]` do not start Web UI / correction UX work until winner / score logic is usable enough
- `[deferred]` do not try to rescue `winner_fusion_v2_layer_ab` as the primary path in this cycle
- `[deferred]` do not reintroduce any `Ollama`-based winner path

## Rejected
- `[rejected]` broad threshold tuning on `ball / ROI / reset / resume` as the main fix
- `[rejected]` treating `weak tail cluster` suppression as the main fix by itself
- `[rejected]` any patch that improves one rally by dragging many endpoints toward the next starter
- `[rejected]` any patch that regresses accepted `set4` endpoints
- `[rejected]` repeating full-signal blind threshold sweeps before first classifying the endpoint archetype
- `[rejected]` treating the current `set1` review bucket as proof that winner detection is already usable
- `[rejected]` promoting `auto` thresholds before fixing the current `blocked` coverage problem
- `[rejected]` continuing to spend the main winner cycle on threshold tuning inside the current Layer A/B heuristic path

## Guardrails
- Never reopen accepted `starter` or `LET` labels without new operator evidence.
- Never accept an endpoint patch that regresses accepted `set4` rally boundaries.
- Keep the accepted `set4` endpoint baseline frozen while moving downstream.
- Winner phase must preserve accepted rally `t_start / t_end` from the earlier detection phase `100%`.
- Winner scripts may read frozen rally boundaries, but must never rewrite them.
- When exporting review rallies for `set1 / set2 / set3 / set4`, always rerun from the original video input end-to-end.
- Do not build review-rally exports from reused intermediate JSON artifacts from earlier partial runs.
- Run `scripts/check_timeline_regression.py` after each endpoint patch.
- Never infer winner from body-language cues alone.
- Winner inference must be constrained by rally-end evidence and score/state validation.
- Keep winner iteration on `set1` from changing accepted rally boundaries.
- Keep detailed experiment logs in `PROJECT_PROGRESS.md`, not here.
