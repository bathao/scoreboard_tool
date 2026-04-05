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

## Work Log - `2026-04-05` (Set4 Endtime Reopened)
### What Was Confirmed
- `set4 did not regress from later code drift`
  - a clean rerun from `Vinh_set4.mp4` with current code reproduced the same `set4` rally boundaries as the accepted checkpoint
  - a clean rerun using historical commit `ddb0ba8` also reproduced the same early-ending points
  - conclusion:
    - the problem is in the accepted `set4` baseline itself, not a later regression

### Fresh Debug Inputs
- `set4 signal bundle was extracted for direct endpoint debugging`
  - local artifact:
    - `matches/checks/_set4_signals_current.pkl`
- `raw detector end is often much later than refined end`
  - strongest pattern:
    - `dead_reset_run_start` is deciding `19/20` points
  - several operator-flagged points showed strong continuation after the chosen dead run
    - the continuation often failed only because of:
      - `gap`
      - `duration`
      - `reset_mean`
    - or because a later stronger dead run existed but the scorer locked onto the first short dead blip

### Patch Attempt
- `dead_reset_run_start now skips some early dead blips`
  - new logic:
    - if a short/early dead run is followed by fragmented continuation
    - or by a later stronger dead run shortly after
    - let the loop continue and evaluate the later dead run instead of returning immediately
- `fresh rerun was executed from source video`
  - output timeline:
    - `matches/checks/Vinh_set4_rally_timeline_set4_endtime_debug_current.json`
  - fresh full-rally review artifact:
    - `debug_report/Vinh_set4_fresh_full_rallies_endtime_debug_current`
    - `debug_report/Vinh_set4_fresh_full_rallies_endtime_debug_current/rally_clips.csv`

### Current Result Of The Patch
- `moved later on several operator-flagged points`
  - `pt_0003: 28.762 -> 30.664`
  - `pt_0004: 37.237 -> 39.740`
  - `pt_0006: 82.449 -> 85.552`
  - `pt_0008: 102.536 -> 105.238`
  - `pt_0010: 135.469 -> 139.306`
  - `pt_0011: 146.113 -> 151.685`
  - `pt_0017: 223.223 -> 225.992`
  - `pt_0019: 241.742 -> 243.410`
- `did not move some flagged points yet`
  - `pt_0013`
  - `pt_0018`
  - `pt_0020`
- `also moved several points the operator did not flag yet`
  - `pt_0002: 15.949 -> 21.488`
  - `pt_0005: 58.859 -> 66.967`
  - `pt_0007: 93.760 -> 98.432`
  - `pt_0009: 127.361 -> 128.595`
  - `pt_0014: 180.080 -> 184.584`
- `operator review after the fresh batch`
  - remaining early-end points are now only:
    - `pt_0013`
    - `pt_0018`
    - `pt_0020`
  - all other shifted points in the fresh batch are currently acceptable

### Follow-up Patch (`v2`)
- `pt_0013 / pt_0018 / pt_0020 were targeted with narrower fixes`
  - `pt_0013`
    - buffered the chosen dead-run start slightly inside the same dead run
  - `pt_0018`
    - allow a very long dead run to reopen if a strong late exchange tail appears before the next accepted point
  - `pt_0020`
    - use the true video duration instead of `signals.timestamps[-1]`
    - relax open-tail strong-run selection for long final runs
- `fresh rerun v2 from source video was completed`
  - timeline:
    - `matches/checks/Vinh_set4_rally_timeline_set4_endtime_debug_current_v2.json`
  - review batch:
    - `debug_report/Vinh_set4_fresh_full_rallies_endtime_debug_current_v2`
    - `debug_report/Vinh_set4_fresh_full_rallies_endtime_debug_current_v2/rally_clips.csv`
- `v2 changes versus the previous fresh batch`
  - `pt_0012: 156.990 -> 157.224`
  - `pt_0013: 168.268 -> 168.435`
  - `pt_0018: 232.766 -> 239.840`
  - `pt_0020: 260.093 -> 264.731`
- `operator decision after reviewing v2`
  - `set4 rallies ok`
  - current code is accepted to use for fresh reruns on:
    - `set1`
    - `set2`
    - `set3`

### Current Diagnosis
- `the current set4 endtime patch is accepted for the next review cycle`
  - winner work stays paused
  - next step is no longer more set4 debugging
  - next step is fresh end-to-end reruns for:
    - `set1`
    - `set2`
    - `set3`
- `set4 now appears to have at least three endtime archetypes`
  - `early dead blip followed by fragmented continuation`
  - `early short dead blip followed by a later stronger dead run`
  - `cases still not improved`
    - likely different archetypes:
      - `pt_0013`
      - `pt_0018`
      - `pt_0020`

### Exact Resume Point
- commit the current set4-accepted code
- then rerun fresh from source video:
  - `set1`
  - `set2`
  - `set3`
- export full rally clips from those fresh runs for operator review

## Work Log - `2026-04-05` (Fresh Set1-3 Review Runs)
### What Was Completed
- `current set4-accepted code was committed`
  - commit:
    - `3ec0d2a`
  - message:
    - `Accept set4 rally endtime fixes`
- `set1 / set2 / set3 were rerun fresh from original videos`
  - `set1`
    - timeline:
      - `matches/checks/Vinh_set1_rally_timeline_fresh_review_current.json`
    - total rallies:
      - `14`
  - `set2`
    - timeline:
      - `matches/checks/Vinh_set2_rally_timeline_fresh_review_current.json`
    - total rallies:
      - `19`
  - `set3`
    - timeline:
      - `matches/checks/Vinh_set3_rally_timeline_fresh_review_current.json`
    - total rallies:
      - `18`
- `full review clips were exported from those fresh runs`
  - `set1`
    - `debug_report/Vinh_set1_fresh_full_rallies_current`
    - `debug_report/Vinh_set1_fresh_full_rallies_current/rally_clips.csv`
  - `set2`
    - `debug_report/Vinh_set2_fresh_full_rallies_current`
    - `debug_report/Vinh_set2_fresh_full_rallies_current/rally_clips.csv`
  - `set3`
    - `debug_report/Vinh_set3_fresh_full_rallies_current`
    - `debug_report/Vinh_set3_fresh_full_rallies_current/rally_clips.csv`

### Exact Resume Point
- operator should review the fresh full-rally batches:
  - `set1`
  - `set2`
  - `set3`
- next debugging cycle should follow operator feedback on those fresh exports

## Work Log - `2026-04-05` (Set2/Set3 Endtime Follow-up)
### Operator Feedback
- fresh reruns still had three obvious early-end points:
  - `set2 pt_0019`
  - `set3 pt_0001`
  - `set3 pt_0002`

### Debug Diagnosis
- `set3 pt_0002`
  - `terminal_body_split_start` was still firing on body runs that overlapped real continuation
  - the immediate post-body continuation around `31.465 -> 32.499` was being mislabeled as weak tail
  - a later dead run around `37.304 -> 38.138` also overlapped a real late continuation run
- `set3 pt_0001`
  - `dead_reset_run_start` was starting inside an exchange that still ran past `6.139`
  - the correct fix was not to skip to a much later dead run, but to delay the chosen dead start inside the same dead run past the embedded exchange blips
- `set2 pt_0019`
  - `last_exchange_support` was ending at the last competitive run even though a narrow body-supported tail persisted after it

### Patch Summary
- added a stronger guard against `terminal_body_split_start` on body runs whose own window still looks live
- added a `strong_post_body_continuation` check so immediate post-body continuation is not treated as weak pseudo-tail
- changed dead-run handling so embedded exchange blips within the chosen dead run delay the dead start instead of skipping to a much later dead run
- added an overlap guard so a dead run is rejected when a strong competitive continuation still overlaps it
- added a narrow `last_exchange_body_tail_end` fallback for the `set2 pt_0019` archetype

### Fresh Rerun Result
- reran fresh from source video again:
  - `matches/checks/Vinh_set2_rally_timeline_endtime_followup_current.json`
  - `matches/checks/Vinh_set3_rally_timeline_endtime_followup_current.json`
- updated target points now land at:
  - `set2 pt_0019: 203.737 -> 205.038`
    - mode:
      - `last_exchange_body_tail_end`
  - `set3 pt_0001: 6.139 -> 7.074`
    - mode:
      - `dead_reset_run_start`
  - `set3 pt_0002: 31.431 -> 38.372`
    - mode:
      - `last_exchange_support`

### Review Artifact
- targeted fresh clips from source video were exported for the three points only:
  - `debug_report/targeted_set2_set3_endtime_followup_current`
  - `debug_report/targeted_set2_set3_endtime_followup_current/rally_clips.csv`

### Exact Resume Point
- operator should review only:
  - `set2 pt_0019`
  - `set3 pt_0001`
  - `set3 pt_0002`
- if all three are accepted:
  - expand review back to the fresh full-rally batches for `set2` and `set3`

## Work Log - `2026-04-05` (Cross-Set Timing Verification After Follow-up Patch)
### What Was Checked
- reran fresh from source video again for all four sets with the current follow-up patch:
  - `matches/checks/Vinh_set1_rally_timeline_post_followup_verify.json`
  - `matches/checks/Vinh_set2_rally_timeline_post_followup_verify.json`
  - `matches/checks/Vinh_set3_rally_timeline_post_followup_verify.json`
  - `matches/checks/Vinh_set4_rally_timeline_post_followup_verify.json`
- compared only `t_start / t_end` against the current review baselines

### Result
- `t_start`
  - unchanged on all four sets
- `t_end`
  - changed on extra points outside the three target points, so the patch is not globally safe yet

### Time Diff Summary
- `set1`
  - `pt_0005: 52.452 -> 52.753`
  - `pt_0007: 69.336 -> 71.471`
- `set2`
  - `pt_0005: 36.270 -> 37.004`
  - `pt_0008: 76.243 -> 76.576`
  - `pt_0009: 86.386 -> 86.920`
  - `pt_0016: 174.574 -> 174.941`
  - `pt_0017: 186.820 -> 187.220`
  - `pt_0019: 203.737 -> 205.038`
- `set3`
  - `pt_0001: 6.139 -> 7.074`
  - `pt_0002: 31.431 -> 38.372`
  - `pt_0008: 93.760 -> 94.628`
  - `pt_0010: 126.393 -> 126.493`
  - `pt_0013: 169.803 -> 170.337`
- `set4`
  - `pt_0001: 7.474 -> 8.408`
  - `pt_0004: 39.740 -> 40.641`
  - `pt_0007: 98.432 -> 99.166`
  - `pt_0016: 202.402 -> 202.836`
  - `pt_0017: 225.992 -> 226.660`

### Conclusion
- the follow-up patch fixed the three target points in the right direction
- but it also pulled later on multiple non-target points across `set1..4`
- current patch should be treated as a narrow debug branch, not the new global baseline

## Work Log - `2026-04-05` (Fresh Full-Rally Exports After Follow-up Patch)
### What Was Completed
- reran all four sets fresh from source video again with the current follow-up patch:
  - `matches/checks/Vinh_set1_rally_timeline_fresh_review_post_followup_full.json`
  - `matches/checks/Vinh_set2_rally_timeline_fresh_review_post_followup_full.json`
  - `matches/checks/Vinh_set3_rally_timeline_fresh_review_post_followup_full.json`
  - `matches/checks/Vinh_set4_rally_timeline_fresh_review_post_followup_full.json`
- exported fresh full-rally review clips from those new runs:
  - `debug_report/Vinh_set1_fresh_full_rallies_post_followup_current`
  - `debug_report/Vinh_set2_fresh_full_rallies_post_followup_current`
  - `debug_report/Vinh_set3_fresh_full_rallies_post_followup_current`
  - `debug_report/Vinh_set4_fresh_full_rallies_post_followup_current`

### Exact Resume Point
- operator should review the fresh post-followup full-rally batches for:
  - `set1`
  - `set2`
  - `set3`
  - `set4`
- next step should be decided from operator review of whether the current patch trends positive or negative overall

## Work Log - `2026-04-05` (Temporary Freeze Of Current Set1-4 Timestamps)
### Operator Decision
- the current post-followup full-rally batches are now temporarily accepted as the working baseline
- freeze the current rally timestamps for:
  - `set1`
  - `set2`
  - `set3`
  - `set4`
- winner work remains paused

### Freeze Action
- copied the current fresh post-followup timelines into the canonical files:
  - `matches/Vinh_set1_rally_timeline.json`
  - `matches/Vinh_set2_rally_timeline.json`
  - `matches/Vinh_set3_rally_timeline.json`
  - `matches/Vinh_set4_rally_timeline.json`
- regenerated the regression manifest from those canonical timelines:
  - `matches/ground_truth/timeline_regression_suite.json`
  - version:
    - `2026-04-05`

### Frozen Review Artifacts
- keep only these full-rally review folders in `debug_report`:
  - `debug_report/Vinh_set1_fresh_full_rallies_post_followup_current`
  - `debug_report/Vinh_set2_fresh_full_rallies_post_followup_current`
  - `debug_report/Vinh_set3_fresh_full_rallies_post_followup_current`
  - `debug_report/Vinh_set4_fresh_full_rallies_post_followup_current`

### Exact Resume Point
- boundary timestamps for `set1..4` are now frozen again at this temporary checkpoint
- if work resumes later:
  - start from this frozen checkpoint
  - and treat any new boundary edit as a deliberate re-open of the baseline

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

## Work Log - `2026-04-04`
### Experiments That Passed
- `winner-side naming cleanup is now complete in the active flow`
  - active code now uses:
    - `scripts/generate_rally_timeline.py`
    - `backend/rally_timeline_contract.py`
    - `scripts/check_timeline_regression.py`
  - active flow no longer uses the old `draft / Draft*` naming
- `candU + review` was removed from the active winner path
  - current rule:
    - if `winner_candidate = unknown`
    - then `winner_decision` must be `blocked`
  - result:
    - review buckets are now actionable by construction
- `decisive actor resolver improved set2 pt_0014`
  - before:
    - `pt_0014 = candA review`
    - operator feedback:
      - wrong side
  - after the resolver + actor-supported review pass:
    - `pt_0014 = candB review`
    - `winner_confidence ~= 0.635`
  - frozen rally guardrails stayed green:
    - `set1_frozen_full = 14/14`
    - `set2_frozen_full = 19/19`
    - `set3_frozen_full = 18/18`
    - `set4_frozen_full = 20/20`
- `set2 winner review batch was regenerated after the fix`
  - current folder:
    - `debug_report/Vinh_set2_rally_clips_winner_fusion_v2_labeled_current`
- `reply-window no-reply branch increased set2 winner hypothesis coverage`
  - new narrow winner branch:
    - if a strong decisive actor exists
    - and the opponent never produces a real reply window
    - allow `review` even when `ball/live` stay active briefly after the final shot
  - current set2 result after rerun:
    - `17/19 blocked`
    - `2/19 review`
    - `0/19 auto`
  - current reviewed rallies are:
    - `pt_0002 -> player_b review`
    - `pt_0014 -> player_b review`
  - this is still not usable overall, but it is better than the previous:
    - `18/19 blocked`
    - `1/19 review`

### Experiments That Failed
- `winner fusion v2 is still not usable on set2`
  - after the latest pass:
    - `17/19 blocked`
    - `2/19 review`
    - `0/19 auto`
  - conclusion:
    - the latest patch fixed one wrong reviewed point and added one more reviewable rally
    - but did not solve the broader coverage problem

### Current Diagnosis
- `set2` does not mostly fail because of side inversion anymore
  - `pt_0014` proved the actor-side resolver was part of the problem
  - `pt_0002` proved that some rallies were blocked only because the system required post-shot collapse to be too immediate
- `set2` still mainly fails because Layer A/B do not create enough rally-level hypotheses
  - most points still end at:
    - `ambiguous_end`
    - `candU`
    - `blocked`
- `coverage`, not `auto`, remains the correct next target
  - do not spend time promoting `review -> auto`
  - first increase the number of rallies that land on plausible `candA / candB`
- `the current heuristic Layer A/B path is still too weak as the main winner engine`
  - even after the latest improvements, coverage is still far below usable
  - conclusion:
    - stop treating this branch as the primary implementation target
    - pivot the next winner cycle to a local VLM-first workflow

### Exact Resume Point
- next cycle should start with:
  - local VLM winner planning and implementation on `set1`
  - frozen rally boundaries remain the fixed input
- first rollout plan:
  - export short end-clips + ordered frame packs from `set1`
  - define the local VLM prompt and strict JSON output schema
  - run the local model on `set1`
  - compare output with operator review before expanding to `set2..4`
- keep/drop rule:
  - keep rally boundaries frozen:
    - `set1_frozen_full = 14/14`
    - `set2_frozen_full = 19/19`
    - `set3_frozen_full = 18/18`
    - `set4_frozen_full = 20/20`
  - keep the current heuristic winner branch only as a reference baseline, not as the main path

## Work Log - `2026-04-04` (Local VLM Start)
### Experiments That Passed
- `local VLM winner scaffolding is now in the repo`
  - contract additions in:
    - `backend/rally_timeline_contract.py`
    - new optional fields:
      - `winner_reason`
      - `winner_model`
  - structured Ollama client in:
    - `backend/ai_ollama_client.py`
  - local VLM winner runner in:
    - `scripts/refine_rally_winners.py`
  - current behavior:
    - uses `qwen3-vl:8b`
    - uses the second half of each rally
    - exports an ordered frame-grid evidence pack
    - expects structured JSON:
      - `winner`
      - `confidence`
      - `decision`
      - `reason`
- `contract/client tests passed after the local-VLM additions`
  - command:
    - `.venv\Scripts\python.exe -m pytest tests\test_ai_ollama_client.py tests\test_rally_timeline_contract.py -q`
  - result:
    - `6 passed, 1 warning`
- `frozen rally suite stayed green after the local-VLM scaffolding`
  - command:
    - `.venv\Scripts\python.exe scripts\check_timeline_regression.py`
  - result:
    - `set1_frozen_full = 14/14`
    - `set2_frozen_full = 19/19`
    - `set3_frozen_full = 18/18`
    - `set4_frozen_full = 20/20`

### Experiments That Did Not Finish
- `first local VLM run on set1 was interrupted`
  - command started:
    - `scripts/refine_rally_winners.py --timeline matches/Vinh_set1_rally_timeline.json --out matches/checks/Vinh_set1_rally_timeline_local_vlm_trial.json --image-dir debug_report/Vinh_set1_local_vlm_evidence --model qwen3-vl:8b`
  - result:
    - interrupted by operator before completion
    - no trusted output timeline was produced
  - partial artifact currently visible:
    - `debug_report/Vinh_set1_local_vlm_evidence/pt_0001_winner_evidence_grid.jpg`

### Additional Results After The Lighter Pack Change
- `lighter 4-frame local-VLM rerun progressed further but still is not usable`
  - command:
    - `scripts/refine_rally_winners.py --timeline matches/Vinh_set1_rally_timeline.json --out matches/checks/Vinh_set1_rally_timeline_local_vlm_trial.json --image-dir debug_report/Vinh_set1_local_vlm_evidence_run3 --model qwen3-vl:8b --resume-if-exists`
  - timeout:
    - `30 minutes`
  - progress reached before timeout:
    - `4` rallies were written to the trial output
    - evidence grids were exported for:
      - `pt_0001`
      - `pt_0002`
      - `pt_0003`
      - `pt_0004`
      - `pt_0005` image also exists, but timeline checkpoint shows only `4` processed
  - current trial output:
    - `matches/checks/Vinh_set1_rally_timeline_local_vlm_trial.json`
  - current per-rally local-VLM result on the processed points:
    - `winner_candidate = unknown`
    - `winner_confidence = 0.0`
    - `winner_decision = blocked`
  - interpretation:
    - the bottleneck is no longer only speed
    - the local VLM prompt / JSON extraction / evidence representation is not yet producing useful structured output

### Single-Rally Debug Result
- `pt_0001` was debugged directly against `qwen3-vl:8b`

## Work Log - `2026-04-05` (Qwen3 Two-Pass Recovery)
### Experiments That Passed
- `qwen3-vl:8b` now works better through a 2-pass local stack`
  - pass 1:
    - qwen3 sees the ordered rally evidence images
    - `content` may still come back empty
    - `thinking` still contains the visual analysis
  - pass 2:
    - the same qwen3 model is called again with that `thinking`
    - it condenses the analysis into compact JSON
  - result:
    - the pipeline can now recover:
      - `winner_candidate`
      - `winner_score_a`
      - `winner_score_b`
      - short textual reason
- `winner score fields were added to the rally timeline contract`
  - new per-rally fields:
    - `winner_score_a`
    - `winner_score_b`
- `set1 pt_0001..pt_0004` now produce differentiated qwen3 outputs`
  - output file:
    - `matches/checks/Vinh_set1_rally_timeline_local_vlm_qwen3_scores2_pt1_4.json`
  - current results:
    - `pt_0001 -> player_a, a=0.95, b=0.05, review`
    - `pt_0002 -> player_a, a=0.95, b=0.05, review`
    - `pt_0003 -> player_b, a=0.00, b=1.00, review`
    - `pt_0004 -> player_a, a=0.70, b=0.30, review`
- `set1 review clips were exported for the new qwen3 batch`
  - folder:
    - `debug_report/Vinh_set1_local_vlm_qwen3_scores2_pt1_4_clips`
  - csv:
    - `debug_report/Vinh_set1_local_vlm_qwen3_scores2_pt1_4_clips/rally_clips.csv`
- `frozen rally suite stayed green after the qwen3 2-pass recovery`
  - `set1_frozen_full = 14/14`
  - `set2_frozen_full = 19/19`
  - `set3_frozen_full = 18/18`
  - `set4_frozen_full = 20/20`

### Important Interpretation
- `set1` true final score is `11-3` tilted toward `near/player_a`
  - therefore a high number of `near` winner predictions is not automatically a bad sign
  - the correct evaluation target is:
    - beat the trivial `always-near` baseline
    - find the minority `far` wins reliably
- current first useful qwen3 evidence of minority-class recall:
  - `pt_0003` is now predicted as `player_b/far` with full confidence

### Exact Resume Point
- get operator review on:
  - `pt_0001..pt_0004` from the new 2-pass qwen3 batch
- if that direction is correct:
  - expand the same 2-pass qwen3 path across more of `set1`
  - use `winner_score_a / winner_score_b` to rank likely `far` wins first

## Work Log - `2026-04-05` (Late Stop / Next Resume)
### Latest Stable Status
- `2-pass qwen3` is now the active local-VLM path for winner research
  - pass 1:
    - qwen3 looks at ordered rally evidence images
  - pass 2:
    - qwen3 condenses the pass-1 `thinking` into compact JSON
- current first reviewable `set1` batch:
  - `matches/checks/Vinh_set1_rally_timeline_local_vlm_qwen3_scores2_pt1_4.json`
  - review clips:
    - `debug_report/Vinh_set1_local_vlm_qwen3_scores2_pt1_4_clips`
  - current outputs:
    - `pt_0001 -> near, a=0.95, b=0.05, review`
    - `pt_0002 -> near, a=0.95, b=0.05, review`
    - `pt_0003 -> far, a=0.00, b=1.00, review`
    - `pt_0004 -> near, a=0.70, b=0.30, review`
- frozen rally suite is still preserved:
  - `set1_frozen_full = 14/14`
  - `set2_frozen_full = 19/19`
  - `set3_frozen_full = 18/18`
  - `set4_frozen_full = 20/20`

### Important Evaluation Note
- operator confirmed:
  - real `set1` final score is `11-3`
  - it is tilted toward `near/player_a`
- implication:
  - many `near` predictions on `set1` are not automatically bad
  - the correct benchmark is now:
    - can the model beat a trivial `always-near` baseline
    - can the model identify the minority `far` wins reliably

### Resume Point For Next Session
- wait for operator feedback on:
  - `pt_0001`
  - `pt_0002`
  - `pt_0003`
  - `pt_0004`
  - from:
    - `debug_report/Vinh_set1_local_vlm_qwen3_scores2_pt1_4_clips`
- after that:
  - keep `2-pass qwen3`
  - expand only on `set1`
  - use `winner_score_a / winner_score_b` to rank likely `far` wins first
  - do not touch frozen rally boundaries

## Work Log - `2026-04-05` (Transformers Native-Video Prep)
### Experiments That Passed
- `Hugging Face core stack is now installed in the project .venv`
  - installed:
    - `transformers 5.5.0`
    - `accelerate 1.13.0`
    - `safetensors 0.7.0`
    - `sentencepiece 0.2.1`
    - `einops 0.8.2`
    - `av 17.0.0`
- `CUDA / GPU check stayed good`
  - `torch 2.12.0.dev + cu128`
  - `cuda_available = True`
  - GPU:
    - `NVIDIA GeForce RTX 5060 Ti`
    - `~15.93 GB VRAM`
- `transformers has direct Qwen3-VL support in this env`
  - verified available classes:
    - `Qwen3VLProcessor`
    - `Qwen3VLForConditionalGeneration`
    - `Qwen3VLVideoProcessor`
- `official HF checkpoints were verified by metadata lookup`
  - `Qwen/Qwen3-VL-4B-Instruct`
  - `Qwen/Qwen3-VL-8B-Instruct`

### Important Conclusion
- if we try the `Transformers native-video` path on this machine:
  - the first practical checkpoint should be:
    - `Qwen/Qwen3-VL-4B-Instruct`
  - reason:
    - it is much safer for `16 GB VRAM` in a first video POC
  - `Qwen/Qwen3-VL-8B-Instruct` remains the preferred larger option only after the smaller POC proves workable

### Exact Resume Point
- next narrow experiment can be:
  - `set1 pt_0001`
  - same short rally clip
  - first native-video POC through `Transformers`
  - start with `Qwen/Qwen3-VL-4B-Instruct`

## Work Log - `2026-04-05` (Transformers Native-Video POC)
### Experiments That Passed
- `Qwen3-VL-4B-Instruct` was downloaded locally into:
  - `models/Qwen3-VL-4B-Instruct`
- `Qwen3-VL` processor accepted true native-video input`
  - local clip used:
    - `debug_report/pt_0001_winner_window_ratio67.mp4`
  - message format using:
    - `type: "video"`
    - local `.mp4` path
  - processor generated:
    - `pixel_values_videos`
    - `video_grid_thw`
- `default native-video settings were too heavy for 16GB VRAM`
  - `Qwen3-VL-4B-Instruct` loaded successfully
  - generation failed with CUDA OOM under the processor defaults
- `reduced native-video token budget works on this GPU`
  - successful settings for `set1 pt_0001`:
    - `fps = 1`
    - `min_frames = 4`
    - `max_frames = 4`
    - `size.shortest_edge = 1024`
    - `size.longest_edge = 1048576`
  - resulting video grid:
    - `video_grid_thw = [2, 24, 42]`
  - resulting native-video output:
    - `player_a`

### Important Interpretation
- native-video through `Transformers` is now confirmed viable on local hardware
  - but only after constraining the video token budget
- this is a real improvement over the Ollama `videos` path
  - the Ollama path could not prove that the model was actually reading the clip
  - the `Transformers` path truly decoded the clip and reached generation

### Exact Resume Point
- compare on the same rally:
  - `2-pass qwen3` image-pack result for `set1 pt_0001`
  - `Transformers native-video` result for `set1 pt_0001`
- if the direction still looks good:
  - expand the reduced-token native-video POC to:
    - `set1 pt_0001..pt_0004`

## Work Log - `2026-04-05` (8B Backup Downloaded)
### Experiments That Passed
- `Qwen3-VL-8B-Instruct` is now downloaded locally`
  - path:
    - `models/Qwen3-VL-8B-Instruct`
  - key weight shards:
    - `model-00001-of-00004.safetensors`
    - `model-00002-of-00004.safetensors`
    - `model-00003-of-00004.safetensors`
    - `model-00004-of-00004.safetensors`

### Current Position
- native-video work now has both checkpoints available locally:
  - `models/Qwen3-VL-4B-Instruct`
  - `models/Qwen3-VL-8B-Instruct`
- intended usage:
  - `4B` is the active main path for native-video winner work
  - `8B` is downloaded as backup only
  - `8B` is not part of the active flow right now

### Active Flow Snapshot
- active winner path:
  - `Transformers native-video`
  - model:
    - `Qwen3-VL-4B-Instruct`
- comparison baseline only:
  - `2-pass qwen3-vl:8b` image-pack path
- parked backup:
  - `Qwen3-VL-8B-Instruct`

## Work Log - `2026-04-05` (Native-Video Full Set1 Batch)
### Experiments That Passed
- `full set1 native-video batch ran end-to-end with the frozen rally windows`
  - runner:
    - `scripts/refine_rally_winners_native_video.py`
  - active model:
    - `models/Qwen3-VL-4B-Instruct`
  - winner window rule:
    - use the last `2/3` of each frozen rally
  - reduced video-token settings that fit `RTX 5060 Ti 16GB`:
    - `fps = 1`
    - `min_frames = 4`
    - `max_frames = 4`
    - `size.shortest_edge = 1024`
    - `size.longest_edge = 1048576`
  - output JSON:
    - `matches/checks/Vinh_set1_rally_timeline_native_video_qwen3vl4b_full.json`
  - review clips:
    - `debug_report/Vinh_set1_native_video_qwen3vl4b_full`
  - CSV:
    - `debug_report/Vinh_set1_native_video_qwen3vl4b_full/rally_clips.csv`

### Current Result
- `technical path works, quality is not good enough yet`
  - the batch completed on all `14` rallies of `set1`
  - current model output is:
    - `14/14 = player_a / near`
  - current filenames therefore all look like:
    - `pt_000x__pick_near__native_video_qwen3vl4b.mp4`

### Interpretation
- `Transformers native-video` is now a real working path, not just a prep stage
- current weakness is no longer transport/runtime
  - it is prediction quality
- next evaluation should focus on:
  - whether this path can beat the trivial `always-near` baseline on `set1`
  - especially whether it can recover the minority `far` wins

## Work Log - `2026-04-05` (Native-Video Window Bug Found)
### Findings
- `the native-video clip builder was not fully honoring the agreed winner window rule`
  - intended rule:
    - use the last `2/3` of each frozen rally
  - actual bug in `scripts/refine_rally_winners_native_video.py`:
    - the helper still applied `max_window_sec = 4.0`
    - long rallies were silently clipped to `4s`

### Fix
- `the default native-video window cap is now disabled`
  - `--max-window-sec` default changed from `4.0` to `0.0`
  - `0.0` now means:
    - do not cap
    - keep the full ratio-derived window

### Quick Verification
- `set1 pt_0001`
  - duration `5.506s`
  - native-video window now `5.506 -> 9.176`
  - length `3.670s`
- `set1 pt_0011`
  - duration `8.842s`
  - native-video window now `107.151 -> 113.046`
  - length `5.895s`
- `set1 pt_0014`
  - duration `5.372s`
  - native-video window now `146.936 -> 150.517`
  - length `3.581s`

### Next Step
- rerun the native-video `set1` batch with the corrected uncapped `2/3` winner window before judging model quality further

## Work Log - `2026-04-05` (Native-Video Full Set1 Rerun After Window Fix)
### Experiments That Passed
- `the full set1 native-video batch was rerun with the corrected uncapped 2/3 winner window`
  - output JSON:
    - `matches/checks/Vinh_set1_rally_timeline_native_video_qwen3vl4b_ratio67_uncapped.json`
  - review clips:
    - `debug_report/Vinh_set1_native_video_qwen3vl4b_ratio67_uncapped`
  - CSV:
    - `debug_report/Vinh_set1_native_video_qwen3vl4b_ratio67_uncapped/rally_clips.csv`
- `the corrected clip timings now match the intended rule`
  - example:
    - `pt_0011`
    - corrected CSV now shows:
      - `clip_start = 107.151`
      - `clip_end = 113.046`
    - exported clip duration measured by `ffprobe`:
      - `5.906s`
    - this matches the intended uncapped `2/3` winner window

### Current Result
- `quality is still unchanged after the timing fix`
  - rerun result is still:
    - `14/14 = player_a / near`

### Interpretation
- the earlier review complaint about clip timing was valid
- that timing bug is now fixed
- from this point onward, any judgment about native-video quality should use only the corrected uncapped artifact

## Work Log - `2026-04-05` (Native-Video Window Rule Updated Again)
### Rule Change
- `winner window rule is now adaptive by rally length`
  - if rally duration `<= 4s`:
    - keep the full frozen rally
  - if rally duration `> 4s`:
    - use the last `2/3` of the frozen rally
  - keep the `max-window-sec = 0` behavior:
    - no silent cap on long rallies

### Quick Verification
- `set1 pt_0002`
  - duration `2.703s`
  - window now `10.177 -> 12.880`
  - full rally kept
- `set1 pt_0006`
  - duration `3.570s`
  - window now `54.288 -> 57.858`
  - full rally kept
- `set1 pt_0011`
  - duration `8.842s`
  - window now `107.151 -> 113.046`
  - still uses uncapped last `2/3`

### Next Step
- rerun the native-video `set1` batch with this adaptive rule before collecting more winner feedback

## Work Log - `2026-04-05` (Native-Video Full Set4 Batch)
### Experiments That Passed
- `full set4 native-video batch ran with the adaptive winner-window rule`
  - rule:
    - rally `<= 4s` -> keep full rally
    - rally `> 4s` -> use last `2/3`
  - output JSON:
    - `matches/checks/Vinh_set4_rally_timeline_native_video_qwen3vl4b_ratio67_adaptive.json`
  - review clips:
    - `debug_report/Vinh_set4_native_video_qwen3vl4b_ratio67_adaptive`
  - CSV:
    - `debug_report/Vinh_set4_native_video_qwen3vl4b_ratio67_adaptive/rally_clips.csv`

### Current Result
- `technical path completed on all 20 rallies`
- `prediction quality is still collapsed to one side`
  - current result:
    - `20/20 = player_a / near`

### Interpretation
- `set4` confirms the current native-video path is not just overfitting to the near-heavy scoreline of set1`
- even on a different set, the present prompt/setup still collapses to `player_a`

## Work Log - `2026-04-05` (Native-Video Full Set4 Rerun)
### Current Review Artifact
- latest full `set4` rerun:
  - `debug_report/Vinh_set4_native_video_qwen3vl4b_ratio67_adaptive_rerun`
  - `debug_report/Vinh_set4_native_video_qwen3vl4b_ratio67_adaptive_rerun/rally_clips.csv`
  - `matches/checks/Vinh_set4_rally_timeline_native_video_qwen3vl4b_ratio67_adaptive_rerun.json`

### Result
- rerun result stayed unchanged:
  - `20/20 = player_a / near`

## Work Log - `2026-04-05` (Boundary Freeze Guardrail Strengthened)
### Experiments That Passed
- `winner scripts now assert that frozen rally boundaries stay unchanged`
  - scripts covered:
    - `scripts/refine_rally_winners.py`
    - `scripts/refine_rally_winners_native_video.py`
  - enforced invariant:
    - winner phase may update winner fields only
    - it must not modify accepted rally `id / t_start / t_end`

## Work Log - `2026-04-05` (Return To Pure Set4 Boundary Review)
### Experiments That Passed
- `all in-progress JSONs in matches/checks were cleared before the next review cycle`
- `full set4 rallies were exported again directly from the original input video`
  - no winner model used
  - no `2/3` cropping
  - each clip is exactly the frozen `t_start -> t_end`
  - review folder:
    - `debug_report/Vinh_set4_frozen_full_rallies`
  - CSV:
    - `debug_report/Vinh_set4_frozen_full_rallies/rally_clips.csv`
  - count:
    - `20` clips

### Current Working Rule
- if the pure full-rally export still looks cut in the middle of a rally, the problem is upstream in the frozen boundary baseline, not in winner processing

## Work Log - `2026-04-05` (Winner Work Paused, Return To Endtime)
### Operator Review
- `set4` pure full-rally export still appears to end too early on multiple rallies
  - reported points:
    - `pt_0003`
    - `pt_0004`
    - `pt_0006`
    - `pt_0008`
    - `pt_0010`
    - `pt_0011`
    - `pt_0013`
    - `pt_0017`
    - `pt_0018`
    - `pt_0019`
    - `pt_0020`

### Decision
- `winner detection is paused`
- `native-video / local-VLM winner experiments are not the current priority anymore`
- `the project returns upstream to rally endtime / boundary correction first`

### Resume Point
- use only:
  - `debug_report/Vinh_set4_frozen_full_rallies`
- re-check and fix the accepted `set4` endtime baseline before resuming any winner work

## Work Log - `2026-04-05` (Set4 Drift Check Against Accepted Commit)
### Experiments That Passed
- `set4 was rerun cleanly from the original video with the current code`
  - command path:
    - `scripts/generate_rally_timeline.py`
  - input:
    - `Vinh_set4.mp4`
  - output:
    - `matches/checks/Vinh_set4_rally_timeline_rerun_current_code.json`

### Result
- `the current rerun matched the accepted checkpoint exactly`
  - compare target:
    - commit `4e70e0f`
    - file `matches/Vinh_set4_rally_timeline.json`
  - diff result:
    - `20/20` points match
    - `diff_count = 0`
  - fields checked:
    - `t_start`
    - `t_end`
    - `active_start`
    - `active_end`
    - `search_upper_bound`
    - `endpoint_mode`
    - `starter_role`
    - `preceding_let_count`

### Interpretation
- the current code has not drifted away from the previously accepted `set4` checkpoint
- if the current full-rally review still looks early-ended, then the issue is no longer:
  - `current code drift`
- it is instead one of:
  - the accepted `set4` baseline itself was wrong
  - or the earlier operator acceptance missed those early-cut rallies

## Work Log - `2026-04-05` (Historical Commit ddb0ba8 Rechecked)
### Experiments That Passed
- `set4 was also rerun fresh from the original video with historical commit ddb0ba8`
  - review artifact:
    - `debug_report/Vinh_set4_fresh_full_rallies_ddb0ba8`
  - CSV:
    - `debug_report/Vinh_set4_fresh_full_rallies_ddb0ba8/rally_clips.csv`

### Result
- `ddb0ba8 reproduces the same early-ending set4 rallies`
- practical conclusion:
  - `ddb0ba8` should no longer be treated as a trustworthy “correct set4 baseline”

### Decision
- stop winner work completely for now
- return the project focus to `set4 endtime / rally boundary` debugging only

## Work Log - `2026-04-05` (New Export Rule)
### Rule Added
- for any operator-requested rally export on `set1 / set2 / set3 / set4`:
  - rerun from the original video input end-to-end
  - do not reuse intermediate JSON from earlier partial runs

### Consequence
- the recent `set4` full-rally export should be treated only as a temporary diagnostic artifact
  - it was built from the frozen timeline JSON
  - so it does not satisfy the new stricter export rule
  - evidence image:
    - `debug_report/Vinh_set1_local_vlm_evidence_run3/pt_0001_winner_evidence_grid.jpg`
  - result with a simple descriptive prompt:
    - the model described the image correctly
    - so the image itself is readable to the model
  - result with the current strict winner prompt:
    - `message.content` came back as malformed / incomplete JSON
    - `message.thinking` was very long
    - the model explicitly appeared confused by the time labels
  - concrete confusion pattern from the raw trace:
    - frame labels are relative:
      - `F1 +0.00s`
      - `F2 +0.92s`
      - ...
    - footer also shows:
      - `winner_window=6.42s->9.18s`
    - the model reasoned as if the frames might be outside or before the winner window
  - conclusion:
    - current failure is not only winner logic
    - it is also:
      - prompt design
      - evidence-pack labeling
      - structured-output reliability

### Current Diagnosis
- the next useful step is no longer architecture work
  - the first local VLM pass can now be executed directly
- the current partial `set1` local-VLM artifacts should not be trusted
  - rerun the first pass cleanly from the start in the next session
- the first attempted local-VLM run also showed a practical performance issue
  - the previous evidence pack was too heavy for a smooth full-set pass
  - current default was reduced for the next run:
    - `4` frames
    - `480x270` per frame
    - `2x2` grid
- the next blocker is now:
  - inspect a single rally's raw model response
  - determine whether failure comes from:
    - prompt design
    - JSON extraction
    - or evidence-pack quality

### Exact Resume Point
- first:
  - do not launch another full `set1` local-VLM run yet
  - instead run one rally only, with raw-response logging enabled
  - start with:
    - `pt_0001` raw-response debugging is now done
    - next use `pt_0001` again only after simplifying the evidence labels
    - then retry on `pt_0002`
- then:
  - inspect:
    - the evidence grid
    - the raw Qwen response
    - the parsed JSON fields
  - next concrete fix should be:
    - remove conflicting absolute-vs-relative time labels from the evidence grid
    - simplify the winner prompt
    - log raw `content` and `thinking` for the next one-rally test
  - only after one-rally debugging works:
    - rerun a small batch
    - then rerun full `set1`
- do not:
  - touch frozen rally boundaries
  - treat the interrupted or timed-out trial output as proof that the local-VLM path is working

## Work Log - `2026-04-04` (Local VLM Model Pivot)
### Experiments That Passed
- `single-rally direct model comparison now points to qwen2.5vl:7b`
  - same evidence image:
    - `debug_report/Vinh_set1_local_vlm_single_debug_onlygrid/pt_0001_winner_evidence_grid.jpg`
  - `qwen3-vl:8b` result on the simplified winner prompt:
    - `message.content = ''`
    - `message.thinking` non-empty
    - not usable as a clean winner label source in the current local stack
  - `qwen2.5vl:7b` result on the same image and prompt:
    - returned a direct winner label:
      - `player_a`
    - no long structured-output failure observed on that direct call
- `active local-VLM defaults were switched to qwen2.5vl:7b`
  - active client default:
    - `backend/ai_ollama_client.py`
  - active winner refine script default:
    - `scripts/refine_rally_winners.py`
  - winner inference metadata string now targets:
    - `local_vlm_qwen2_5vl_7b_second_half_framepack_v1`

### Current Diagnosis
- `qwen3-vl:8b` is no longer the best first model for this task on this machine
  - not because it cannot read the image
  - but because its winner-response path is not producing clean usable output
- `qwen2.5vl:7b` is now the correct next primary local-VLM candidate
  - it already produced a direct `player_a` label on the same evidence image and prompt
- the next useful test is now:
  - run the full local winner script path on exactly one rally with `qwen2.5vl:7b`
  - verify:
    - parsed winner label
    - decision bucket
    - stored JSON fields

### Exact Resume Point
- first:
  - rerun `scripts/refine_rally_winners.py` on `set1 pt_0001` only
  - use:
    - `qwen2.5vl:7b`
  - write to a fresh trial JSON
- then:
  - inspect the stored point fields:
    - `winner_candidate`
    - `winner_confidence`
    - `winner_decision`
    - `winner_reason`
    - `winner_model`
- only after that:
  - decide whether to batch more `set1` rallies

## Work Log - `2026-04-04` (qwen2.5vl Script Path Check)
### Experiments That Passed
- `the full local-VLM script path now works with qwen2.5vl:7b`
  - one-rally run:
    - `set1 pt_0001`
  - output:
    - `winner_candidate = player_a`
    - `winner_confidence = 0.64`
    - `winner_decision = review`
    - `winner_model = qwen2.5vl:7b`
  - trial file:
    - `matches/checks/Vinh_set1_rally_timeline_local_vlm_qwen25_trial.json`
  - evidence image:
    - `debug_report/Vinh_set1_local_vlm_qwen25_single/pt_0001_winner_evidence_grid.jpg`
- `small-batch speed is now acceptable with qwen2.5vl:7b`
  - `set1 pt_0001..pt_0004` completed in a few seconds
  - trial file:
    - `matches/checks/Vinh_set1_rally_timeline_local_vlm_qwen25_trial4.json`

### Experiments That Failed
- `current qwen2.5vl prompt/output path looks strongly biased toward player_a`
  - `set1 pt_0001..pt_0004` all returned:
    - `winner_candidate = player_a`
    - `winner_decision = review`
    - `winner_confidence = 0.64`
  - `set2 pt_0014` also returned:
    - `winner_candidate = player_a`
    - even though the earlier heuristic review path had already improved this point toward `player_b`
  - conclusion:
    - the local-VLM path now works technically
    - but the current prompt/evidence design is likely introducing a near-side bias

### Current Diagnosis
- the main blocker is no longer:
  - empty VLM output
  - or unacceptable speed on tiny batches
- the new blocker is:
  - side bias / default-answer bias toward `player_a`
- next useful work should focus on:
  - prompt redesign
  - frame-pack redesign
  - explicit bias checks on rallies expected to favor `player_b`

### Exact Resume Point
- first:
  - keep `qwen2.5vl:7b` as the active local winner model
  - do not go back to `qwen3-vl:8b` as the primary path
- then:
  - redesign the winner prompt to reduce default-answer bias
  - test at least:
    - one rally likely won by `player_a`
    - one rally likely won by `player_b`
    - one ambiguous rally
- only after that:
  - batch more `set1`

## Work Log - `2026-04-04` (Table-Centered Crop Trial)
### Experiments That Passed
- `winner evidence pack now crops around the table play zone`
  - evidence frames are no longer full-frame only
  - current crop expands around the frozen table ROI so the top and bottom players occupy much more of the grid
- `first labeled set1 review clips were exported for operator review`
  - folder:
    - `debug_report/Vinh_set1_local_vlm_qwen25_crop_trial4_clips`
  - exported points:
    - `pt_0001`
    - `pt_0002`
    - `pt_0003`
    - `pt_0004`

### Experiments That Failed
- `the crop+prompt change appears to have overcorrected the earlier near-side bias`
  - previous small-batch behavior:
    - all first four `set1` rallies leaned `player_a`
  - current small-batch behavior after the crop+prompt change:
    - all first four `set1` rallies lean `player_b`
  - additional check:
    - `set2 pt_0014` also leaned `player_b`
  - conclusion:
    - the local-VLM path is now technically stable enough to review
    - but side bias is still dominating the output

### Current Diagnosis
- the main question is no longer whether local VLM can run
  - it can
- the main question is now:
  - whether the current evidence-pack and prompt design are forcing a side prior
- the next useful input is operator review on the first labeled `set1` clips
  - this will tell us whether the new far-side lean is:
    - a real improvement
    - or just the opposite bias

### Exact Resume Point
- first:
  - get operator feedback on:
    - `debug_report/Vinh_set1_local_vlm_qwen25_crop_trial4_clips`
- then:
  - depending on feedback:
    - keep the crop and soften the prompt
    - or redesign the frame-pack again

## Work Log - `2026-04-04` (qwen3-vl Stack Fix)
### Experiments That Passed
- `qwen3-vl:8b` now works through the real script path
  - key fix was not a model swap
  - key fix was local-stack behavior:
    - do not send one stitched grid as the only image
    - send `4` ordered frame images separately
    - increase token budget substantially
    - allow fallback winner extraction from `thinking` when `content` is empty
- `set1 pt_0001` now runs end-to-end with qwen3-vl
  - output file:
    - `matches/checks/Vinh_set1_rally_timeline_local_vlm_qwen3_fix_pt1.json`
  - result:
    - `winner_candidate = player_b`
    - `winner_decision = review`
    - `winner_model = qwen3-vl:8b`
- `first qwen3 set1 mini-batch is now reviewable`
  - output file:
    - `matches/checks/Vinh_set1_rally_timeline_local_vlm_qwen3_fix_pt1_4.json`
  - clip folder:
    - `debug_report/Vinh_set1_local_vlm_qwen3_fix_pt1_4_clips`
  - current mini-batch outputs:
    - `pt_0001 -> lean_far`
    - `pt_0002 -> lean_near`
    - `pt_0003 -> lean_far`
    - `pt_0004 -> lean_near`

### Current Diagnosis
- the earlier conclusion `qwen3-vl:8b is unusable` was too strong
  - it was largely a stack/protocol issue
- qwen3 is now back as the primary local-VLM path
- the next blocker is now model quality, not transport:
  - does the new `set1` mini-batch actually lean the right side often enough

### Exact Resume Point
- first:
  - get operator feedback on:
    - `debug_report/Vinh_set1_local_vlm_qwen3_fix_pt1_4_clips`
- then:
  - tune prompt / frame-pack based on those first four qwen3 labels

## Work Log - `2026-04-04` (qwen3 First Review Feedback)
### Experiments That Passed
- `first human review of the qwen3 mini-batch now exists`
  - operator feedback on `set1 pt_0001..pt_0004`:
    - `pt_0001 = wrong`
    - `pt_0002 = correct`
    - `pt_0003 = wrong`
    - `pt_0004 = wrong`

### Current Diagnosis
- `qwen3-vl` is now usable enough to produce reviewable winner hypotheses
  - but current accuracy on the first four reviewed `set1` rallies is still poor
- the strongest concrete hypothesis from operator feedback is:
  - `winner_window = 1/2` may be too short on rallies whose accepted `t_end` is a bit late
  - especially `pt_0001` may need more pre-end context for the model to understand the last real exchange

### Exact Resume Point
- first:
  - change the local-VLM default winner window from `1/2` to `2/3` of the rally
- then:
  - rerun `set1 pt_0001..pt_0004` with `qwen3-vl:8b`
  - export a new labeled clip batch for operator comparison

## Work Log - `2026-04-04` (qwen3 Two-Thirds Window Trial)
### Experiments That Passed
- `local-VLM default winner window was widened from 1/2 to 2/3 of the rally`
  - active script default now uses:
    - `window_ratio = 2/3`
- `the qwen3 set1 mini-batch was rerun with the wider winner window`
  - output file:
    - `matches/checks/Vinh_set1_rally_timeline_local_vlm_qwen3_fix_pt1_4_ratio67.json`
  - clip folder:
    - `debug_report/Vinh_set1_local_vlm_qwen3_fix_pt1_4_ratio67_clips`
  - current outputs:
    - `pt_0001 -> lean_near`
    - `pt_0002 -> lean_near`
    - `pt_0003 -> lean_near`
    - `pt_0004 -> lean_near`

### Current Diagnosis
- widening the window from `1/2` to `2/3` did materially change the model behavior
  - the earlier mixed pattern:
    - `far, near, far, near`
  - became:
    - `near, near, near, near`
- this strongly suggests that winner-window length is a first-order factor for local-VLM behavior
- the next decision now depends on operator review:
  - whether the wider context actually improves correctness enough

### Exact Resume Point
- first:
  - get operator feedback on:
    - `debug_report/Vinh_set1_local_vlm_qwen3_fix_pt1_4_ratio67_clips`
- then:
  - decide whether `2/3` should stay as default
  - or whether window ratio needs to become adaptive instead of fixed

## Work Log - `2026-04-05` (Strict Prompt Trial)
### Experiments That Passed
- `qwen3 prompt was tightened to reduce posture-only winner guesses`
  - new prompt explicitly says:
    - do not decide only from who walks away, relaxes, or picks up the ball
    - use the last clear shot plus visible failed return
    - otherwise answer `unknown`
- `the stricter prompt changed the first reviewed set1 batch in a meaningful way`
  - output file:
    - `matches/checks/Vinh_set1_rally_timeline_local_vlm_qwen3_ratio67_prompt2_pt1_4.json`
  - clip folder:
    - `debug_report/Vinh_set1_local_vlm_qwen3_ratio67_prompt2_pt1_4_clips`
  - current outputs:
    - `pt_0001 -> unknown`
    - `pt_0002 -> lean_near`
    - `pt_0003 -> unknown`
    - `pt_0004 -> unknown`

### Current Diagnosis
- this is the first sign that the prompt can change the model in the desired direction
  - instead of confidently forcing one side on almost every point
  - it now abstains on several rallies
- coverage is lower, but false-positive pressure also appears lower
- the next decision should be made from operator review:
  - is this conservative shift better than the previous overconfident near/far swings

### Exact Resume Point
- first:
  - get operator feedback on:
    - `debug_report/Vinh_set1_local_vlm_qwen3_ratio67_prompt2_pt1_4_clips`
- then:
  - if the new conservative behavior is better:
    - keep this prompt family
    - and work on increasing coverage
  - otherwise:
    - adjust evidence pack again

## Work Log - `2026-04-05` (Forced 8x2 Evidence Trial)
### Experiments That Passed
- `qwen3 winner evidence pack was expanded to 8 time steps with dual views`
  - each time step now has:
    - `full` frame
    - `crop` frame
  - qwen3 receives ordered separate images instead of only one stitched grid
- `forced A/B choice is now active for qwen3 path`
  - current prompt no longer allows `unknown`
  - model is asked to choose:
    - `player_a`
    - or `player_b`
  - confidence is then parsed or defaulted for review use
- `set1 pt_0001..pt_0004 rerun completed with the richer evidence pack`
  - output file:
    - `matches/checks/Vinh_set1_rally_timeline_local_vlm_qwen3_forced8x2_pt1_4.json`
  - current outputs:
    - `pt_0001 -> player_a, 0.50`
    - `pt_0002 -> player_a, 0.50`
    - `pt_0003 -> player_a, 0.50`
    - `pt_0004 -> player_a, 0.95`

### Current Diagnosis
- richer evidence did change the behavior again
  - compared with the earlier conservative prompt run, the model is now willing to decide on every point
- but current pattern is still suspicious:
  - the first four reviewed set1 rallies all lean `player_a`
  - three of them have only floor confidence
  - one rally has high confidence and therefore deserves extra scrutiny

### Exact Resume Point
- first:
  - export and review the new `8x2 forced-choice` clips for:
    - `pt_0001..pt_0004`
- then:
  - determine whether the richer evidence improved correctness enough
  - especially inspect the high-confidence `pt_0004`

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
    - `backend/timeline_regression.py`
  - added unit coverage:
    - `tests/test_timeline_regression.py`
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
    - `.venv\Scripts\python.exe -m pytest tests/test_multistream_rally.py tests/test_rally_timeline_contract.py`
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
  - `.venv\Scripts\python.exe -m pytest tests/test_multistream_rally.py tests/test_rally_timeline_contract.py`
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
    - `.venv\Scripts\python.exe -m pytest tests/test_multistream_rally.py tests/test_rally_timeline_contract.py`
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
  - `tests/test_multistream_rally.py` + `tests/test_rally_timeline_contract.py`:
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
  - `tests/test_multistream_rally.py` + `tests/test_rally_timeline_contract.py`:
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
    - `tests/test_rally_timeline_contract.py`: `2 passed, 1 warning`

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

## 2026-04-04
- `winner / point-state v1 started on set1`
  - added new rally fields:
    - `point_end_event`
    - `winner_candidate`
    - `winner_confidence`
    - `winner_decision`
  - `winner` itself is still only auto-applied when confidence is high enough
- `winner inference doctrine implemented in code`
  - use `end-of-rally evidence -> point-end event -> winner candidate -> decision`
  - keep `table timing + player behavior` mandatory
  - keep `ball` as optional strong evidence
  - keep body-language as secondary confidence only
- `set1 winner v1 rerun`
  - output:
    - `matches/Vinh_set1_rally_timeline.json`
  - summary:
    - `winner_known_rallies = 0`
    - `winner_candidate_known_rallies = 6`
    - `winner_review_rallies = 7`
    - `winner_blocked_rallies = 7`
  - first visible candidate examples:
    - `pt_0001 -> player_a review 0.762 clean_winner_like`
    - `pt_0006 -> player_a review 0.735 clean_winner_like`
    - `pt_0008 -> player_a review 0.669 rally_error_like`
    - `pt_0009 -> player_a review 0.584 rally_error_like`
- verification
  - `50 passed, 1 warning`
  - `scripts/check_timeline_regression.py`:
    - `set1_frozen_full = 14/14`
    - `set2_frozen_full = 19/19`
    - `set3_frozen_full = 18/18`
    - `set4_frozen_full = 20/20`
- current read
  - this is a safe starting point for winner work
  - it does not yet auto-resolve any rally
  - the next useful step is to review `set1` winner candidates and tighten event classes before expanding to the other sets
- `winner v1 rerun expanded to set2 / set3 / set4`
  - outputs:
    - `matches/Vinh_set2_rally_timeline.json`
    - `matches/Vinh_set3_rally_timeline.json`
    - `matches/Vinh_set4_rally_timeline.json`
  - current summaries:
    - `set2: candidate_known=1, review=5, blocked=14, auto=0`
    - `set3: candidate_known=5, review=6, blocked=12, auto=0`
    - `set4: candidate_known=5, review=9, blocked=11, auto=0`
- `winner review assets exported`
  - `debug_report/Vinh_set1_rally_clips_winner_current`
  - `debug_report/Vinh_set2_rally_clips_winner_current`
  - `debug_report/Vinh_set3_rally_clips_winner_current`
  - `debug_report/Vinh_set4_rally_clips_winner_current`
  - filenames now include:
    - winner candidate
    - decision bucket
    - confidence
    - coarse end-event class
- verification after full rerun
  - `scripts/check_timeline_regression.py`:
    - `set1_frozen_full = 14/14`
    - `set2_frozen_full = 19/19`
    - `set3_frozen_full = 18/18`
    - `set4_frozen_full = 20/20`
- next resume point
  - review the winner-labeled clips with the operator
  - classify which current candidates are directionally right vs clearly wrong
  - only then tighten event classes / confidence thresholds
  - do not start score progression until winner candidates are substantially healthier

## 2026-04-04 (winner fusion v2 kickoff after naming cleanup)

- cleaned the active flow naming so the canonical path now uses only `rally timeline` terms
  - moved the active implementation onto `scripts/generate_rally_timeline.py`
  - moved the active contract onto `backend/rally_timeline_contract.py`
  - renamed regression helpers to `timeline_regression`
  - renamed tests to `tests/test_rally_timeline_contract.py` and `tests/test_timeline_regression.py`
- verified the rename refactor did not break the frozen rally baseline
  - `52 passed, 1 warning`
  - `set1_frozen_full = 14/14`
  - `set2_frozen_full = 19/19`
  - `set3_frozen_full = 18/18`
  - `set4_frozen_full = 20/20`
- started `Winner Fusion v2` implementation on top of the frozen baseline
  - split winner inference into:
    - common winner-search window
    - Layer A `Physics / Event`
    - Layer B `Player Interaction`
    - fused final decision
  - updated `analysis_metadata.winner_inference_mode` to `winner_fusion_v2_layer_ab`
- removed `winner v1` naming from the active code path
  - winner annotation now goes through the explicit `fusion v2` functions only
- reran `set1` through the new winner path
  - rally boundaries stayed frozen
  - current `set1` winner summary:
    - `winner_known_rallies = 1`
    - `winner_candidate_known_rallies = 6`
    - `winner_review_rallies = 7`
    - `winner_blocked_rallies = 6`
  - notable first-pass outputs:
    - `pt_0001 -> rally_error_like / player_a / 0.836 / auto`
    - `pt_0006 -> rally_error_like / player_a / 0.807 / review`
- next resume point:
  - inspect the first `winner_fusion_v2_layer_ab` pass on `set1`
  - decide whether `pt_0001` should really stay `auto`
  - then tighten Layer A/B fusion gates before expanding v2 beyond `set1`

## 2026-04-04 (winner fusion v2 reality check)

- exported `set1` winner-labeled rally clips and reviewed the labeling format
- fixed one important labeling bug:
  - `candU + review` is no longer allowed
  - if `winner_candidate = unknown`, decision now goes straight to `blocked`
- reran `set1`
  - current `set1` summary after the gate fix:
    - `winner_known_rallies = 1`
    - `winner_candidate_known_rallies = 6`
    - `winner_review_rallies = 4`
    - `winner_blocked_rallies = 9`
- reran `set2` with the same `winner_fusion_v2_layer_ab` path
  - rally boundaries stayed frozen:
    - `set1_frozen_full = 14/14`
    - `set2_frozen_full = 19/19`
    - `set3_frozen_full = 18/18`
    - `set4_frozen_full = 20/20`
  - winner result on `set2` is poor:
    - `18/19 blocked`
    - `1/19 review`
    - the only reviewed rally `pt_0014` is wrong by operator inspection
- conclusion:
  - `Winner Fusion v2` is still a research prototype
  - it is not usable yet, even on `set1`, because blocked coverage is still far too high
  - the main problem is not `auto` thresholding
  - the main problem is that Layer A/B still fail to create useful `candidate A/B` hypotheses for most rallies
- next resume point:
  - debug `set2 pt_0014` first
  - then debug a small batch of `set2` blocked rallies that look obvious by eye
  - focus on increasing hypothesis coverage
  - do not spend the next cycle trying to promote more `auto`
