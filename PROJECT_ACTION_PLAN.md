# Project Action Plan

## Purpose
This is the short operational board for the current work.

Keep this file short. It should answer:
- what is being worked on now
- what is blocked
- what is next
- what must not regress

Detailed history belongs in `PROJECT_PROGRESS.md`.

Step 3-specific debug details belong in `STEP3_DEBUG.md`.

The active pipeline contract belongs in `PIPELINE.md`.

## Status Legend
- `[done]`
- `[doing]`
- `[todo]`
- `[blocked]`
- `[deferred]`
- `[rejected]`

## Current Focus

- `[doing]` Debug Step 3.1 start-time detection on `inputs/raw_matches/2_sets.mp4`.
- `[blocked]` Step 3.2 side-state reliability is blocked because latest Step 3.1 over-counts rallies.
- `[blocked]` Step 3.3 logic audit is paused until Step 3.1 total rally count is trustworthy.
- `[blocked]` Step 3.4 set-boundary / side-swap detection is not implemented yet and must wait for Step 3.1/3.2 to stabilize.
- `[blocked]` Step 4 winner prediction must not run on `2_sets.mp4` while Step 3 is invalid.

## Immediate Next Steps

1. Reopen the latest chunked Step 3.1 output:
   `runtime_jobs/debug_2_sets_step31_chunked151_latest/step3_1_review/summary.md`.
2. Mark the `38 detected starts` output as rejected: total rally count is too high.
3. Identify which rows are false positives or unsafe rule-only repairs.
4. Demote `serve_order_gap_auto_repair` rows back to suspect/review unless a
   targeted visual rescan confirms the start.
5. Keep the chunked detector improvements that recovered real starts like
   `00:18` LET, `00:32`, `01:03`, and `02:13`, but remove whatever caused
   over-detection.
6. Rerun Step 3.1 only. Do not rerun Step 3.2 until Step 3.1 total is accepted.

## Current Artifacts

- Active input: `inputs/raw_matches/2_sets.mp4`
- Step 3 debug board: `STEP3_DEBUG.md`
- Step 3.1 latest summary:
  `runtime_jobs/debug_2_sets_step31_chunked151_latest/step3_1_review/summary.md`
- Step 3.1 latest events:
  `runtime_jobs/debug_2_sets_step31_chunked151_latest/step3_1_review/step3_1_rally_start_events.json`
- Latest Step 3.1 result: `38 detected starts = 36 scoring + 2 LET/non-scoring + 4 rule-conflict rows`
- Latest Step 3.1 verdict: rejected, total count is too high.
- Previous Step 3.2 summary kept only as artifact:
  `runtime_jobs/debug_2_sets_full/step3_2_side_state_review/summary.md`
- Latest repaired Step 3.2 summary:
  `runtime_jobs/debug_2_sets_step31_chunked151_latest/step3_2_side_state_review/summary.md`
- Latest repaired Step 3.2 result: invalid as baseline because Step 3.1 input over-counts rallies.

## Must Not Regress

- GPU-only runtime rule remains mandatory. No CPU fallback for pipeline steps that require GPU.
- Production input is long / full-match video. Short clips are only for faster
  development/debug and must never become an algorithm assumption.
- Unknown player/server remains `unknown`. Do not assign by guessing.
- Step 2 confirmed player names are trusted input for Step 3 side-state logic.
- Step 3.1 owns rally/LET `start_time` detection.
- Step 3.2 must not create or move start times.
- Step 3.3 must not create new rally starts; it can only audit and request Step 3.1 repair.
- Rule-only gap repair must not become a confirmed rally without visual confirmation.
- Do not assume short-clip accuracy transfers to long multi-set input. Long
  clips change normalization, include side swaps/walking/noise, and can break
  role/serve-order assumptions.
- Step 4 must not consume an invalid Step 3 timeline.

## Deferred

- `[deferred]` Step 3.3 recursive audit/rescan loop.
- `[deferred]` Step 3.4 detect set boundaries / side swaps.
- `[deferred]` Web UI Step 3 integration cleanup.
- `[deferred]` Winner prediction improvements.
