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
- Production draft baseline:
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
  - commit `0990559`
  - `Stabilize first ten set4 endpoint reviews`

## Done
- `[done]` starter detection accepted on reviewed `set1..4`
- `[done]` `starter_role` export works in latest player-path drafts
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

## Doing
- `[doing]` move downstream logic onto the frozen rally list:
  - winner inference
  - point flow
  - score progression
- `[doing]` keep the accepted `starter + LET + active + set4 endpoint` baseline frozen while starting downstream work

## Todo
- `[todo]` define the first winner-candidate path on top of the frozen rally list
- `[todo]` attach winner / point-state outputs to the current draft contract
- `[todo]` begin score progression work on top of accepted rally windows
- `[todo]` return to fusion only after the independent `player` path is stable enough

## Deferred
- `[deferred]` do not retune `table / ROI-first` in this cycle
- `[deferred]` do not retune `ball tracking V0` in this cycle
- `[deferred]` do not work on 3-detector fusion yet
- `[deferred]` do not spend this cycle on `Qwen` boundary/split tuning
- `[deferred]` do not start Web UI work yet
- `[deferred]` do not start Web UI / correction UX work until winner / score logic is usable enough

## Blockers
- `[blocked]` no major blocker inside `set4` endpoint at this checkpoint
- `[blocked]` the next real blocker is that downstream `winner / score` logic has not been promoted yet

## Rejected
- `[rejected]` broad threshold tuning on `ball / ROI / reset / resume` as the main fix
- `[rejected]` treating `weak tail cluster` suppression as the main fix by itself
- `[rejected]` any patch that improves one rally by dragging many endpoints toward the next starter
- `[rejected]` any patch that regresses accepted `set4` endpoints

## Guardrails
- Never reopen accepted `starter` or `LET` labels without new operator evidence.
- Never accept an endpoint patch that regresses accepted `set4` rally boundaries.
- Keep the accepted `set4` endpoint baseline frozen while moving downstream.
- Keep detailed experiment logs in `PROJECT_PROGRESS.md`, not here.
