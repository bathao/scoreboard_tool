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
  - commit `2587d6f`
  - `Set1 and set4 rally detection is correct`
- Endpoint regression suite baseline:
  - `set4_frozen_full` = required no-regression suite
  - `set1_reviewed_first6` = required no-regression suite from the accepted reviewed first-six batch
- Current focus:
  - freeze full `set1..4` rally baseline and prepare winner / score work

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
- `[done]` `set3 pt_0009` is fixed without regressing `set1 / set2 / set4`

## Doing
- `[doing]` keep the accepted `set1 / set2 / set3 / set4` rally timelines frozen while moving downstream
- `[doing]` use the new endpoint regression suite before every endpoint change
- `[doing]` expand the regression suite from the current required guardrails toward full `set1..4`
- `[doing]` prepare the first `winner / point-state` pass on top of the frozen rally timelines

## Todo
- `[todo]` add stable `set2` and `set3` checkpoint suites into `timeline_regression_suite.json`
- `[todo]` promote a full `set1..4` rally timeline baseline
- `[todo]` start `winner / point-state` inference on top of the frozen rally timelines
- `[todo]` build `score progression` on top of accepted rallies + inferred winners
- `[todo]` run the same pipeline on a single long multi-set input, not only split sets
- `[todo]` start correction / UI flow only after rally + winner + score are usable
- `[todo]` define the first winner-candidate path on top of the frozen rally list
- `[todo]` attach winner / point-state outputs to the current rally timeline contract
- `[todo]` begin score progression work on top of accepted rally windows
- `[todo]` return to fusion only after the independent `player` path is stable enough

## Deferred
- `[deferred]` do not retune `table / ROI-first` in this cycle
- `[deferred]` do not retune `ball tracking V0` in this cycle
- `[deferred]` do not work on 3-detector fusion yet
- `[deferred]` do not spend this cycle on `Qwen` boundary/split tuning
- `[deferred]` do not start Web UI work yet
- `[deferred]` do not start Web UI / correction UX work until winner / score logic is usable enough

## Rejected
- `[rejected]` broad threshold tuning on `ball / ROI / reset / resume` as the main fix
- `[rejected]` treating `weak tail cluster` suppression as the main fix by itself
- `[rejected]` any patch that improves one rally by dragging many endpoints toward the next starter
- `[rejected]` any patch that regresses accepted `set4` endpoints
- `[rejected]` repeating full-signal blind threshold sweeps before first classifying the endpoint archetype

## Guardrails
- Never reopen accepted `starter` or `LET` labels without new operator evidence.
- Never accept an endpoint patch that regresses accepted `set4` rally boundaries.
- Keep the accepted `set4` endpoint baseline frozen while moving downstream.
- Run `scripts/check_timeline_regression.py` after each endpoint patch.
- Keep detailed experiment logs in `PROJECT_PROGRESS.md`, not here.
