# Step 3 Debug Board

## Purpose
Use this file for the active Step 3 debugging state only.

Keep here:
- current blocker
- ground-truth notes for active videos
- latest debug commands and artifact paths
- hypotheses to test next
- what must not regress

Do not use this file for the long-term product target. That belongs in
`ROADMAP_PRODUCTION.md`.

Do not use this file as the pipeline contract. That belongs in `PIPELINE.md`.

## Current Status - 2026-04-22

Active input:
- `inputs/raw_matches/2_sets.mp4`

Production input rule:
- Production will always run on long / full-match videos.
- Short clips are only a development/debug shortcut to iterate faster.
- Any Step 3 algorithm that only works on short clips is not production-ready.
- Fixes may use short clips to isolate a bug, but final validation must be on
  the long working video.

GUI debug mode:
- The setup screen now has `Debug Step 3 only`.
- This skips Step 1 trim and Step 2 identify.
- The operator manually enters Player A / Player B, treated as trusted Step 2
  ground truth for this debug job.
- The GUI runs Step 3.1 then Step 3.2 in one background task.
- It pauses at `confirm_side_state` and displays the generated
  `step3_2_side_state_review/summary.md` in the browser.
- The same debug setup now has two run buttons:
  - `Run Step 3.1 Only`: stops at `confirm_total_rallies` for start-time /
    LET debugging.
  - `Run Step 3.1 + 3.2`: runs both stages and stops at side-state summary.
- The Step 3.1 pause view shows a start-time audit table with total starts,
  scoring count, LET/non-scoring count, needs-review rows, and each detected
  start/end timestamp in `mm:ss.mmm` format.
- Player names must already exist in FaceDB if Step 3.2 side identification is
  expected to resolve `NEAR/FAR`.
- Debug logs now include phase-level heartbeat messages after pose detection,
  during signal alignment, segment detection, LET/scoring merge, endpoint
  refinement, Step 3.2 local side scans, frame export, and summary writing.

Current gate:
- Step 3 is blocked before Step 4.
- Step 3.1 is still not accepted.
- Step 3.2 / Step 3.3 / Step 3.4 are blocked until Step 3.1 total rally count is correct.
- The next debugging target remains Step 3.1 start-time detection and false-positive removal.

Highest-priority blocker:
- Step 3.1 start-time detection is not reliable enough.
- If Step 3.1 misses or invents rally/LET start times, every downstream step
  fails: side-state detection, serve-order logic audit, set-boundary detection,
  winner prediction, and final scoreboard rendering.

Latest clean rerun from empty cache:
- Cache folder reset: `runtime_jobs/debug_2_sets_full/`
- Step 3.1 output:
  - `33 total`
  - `28 scoring`
  - `5 LET/non-scoring`
  - `4 needs-review`
  - summary: `runtime_jobs/debug_2_sets_full/step3_1_review/summary.md`
- Step 3.2 output:
  - `identified=23`
  - `inferred=5`
  - `unknown=5`
  - summary: `runtime_jobs/debug_2_sets_full/step3_2_side_state_review/summary.md`
- Step 3.3 was not rerun after the cache reset.

Operator verdict:
- Current Step 3.1/3.2 output is worse than an earlier working point.
- The number of detected rallies is wrong.
- LET detection is wrong.
- `summary.md` is still not good enough for productive review.

Latest GUI Step 3.1 feedback on `2_sets.mp4`:
- Job reviewed: `runtime_jobs/20260422T123744Z__2_sets/`
- GUI Step 3.1 output:
  - `28 total`
  - `24 scoring`
  - `4 LET/non-scoring`
  - `5 needs-review`
- Operator-confirmed problems in the reviewed portion:
  - `rally_0003` at about `00:18` is actually `LET`.
  - Missing scoring rally at about `00:32`.
  - `rally_0005` at about `00:41` is **not** LET.
  - `rally_0006` at about `00:52` is **not** LET.
  - Missing scoring rally at about `01:03`.
  - Missing scoring rally at about `02:02`.
  - Missing scoring rally at about `02:13`.
  - Missing scoring rally at about `02:51`.
  - Missing scoring rally at about `02:56`.
  - Missing scoring rally at about `03:08`.
  - Missing scoring rally at about `03:16`.
  - Later portion has not been fully reviewed yet, but current miss rate is
    already unacceptable.
- Important comparison: older segmented debug run
  `runtime_jobs/debug_2_sets_0_151/step3_1_review/summary.md` did much better
  on the same early source window. It detected `00:32`, `01:03`, `02:02`
  as a `needs_review` gap marker, and `02:13`, while the latest full-video GUI
  run missed those starts.
- Current suspicion:
  - Full-video Step 3.1 behavior regressed versus bounded-segment debug.
  - LET labels are being forced by serve-order overflow logic
    (`let_inferred_forced_serve_order`) rather than reliably inferred from the
    local video action.
  - Step 3.1 must be debugged at the raw candidate / active segment level
    before Step 3.2 or Step 3.3 are touched again.

Why short input looked correct while long input failed:
- The detector normalizes and thresholds motion/pose/ball signals over the
  whole input it receives. A short bounded clip has local normalization, so weak
  but real serve-start signals stay visible. A long multi-set clip has more
  unrelated motion, side-swap movement, walks, adjacent-table activity, and
  post-point noise; those can change the global scale and suppress weak starts.
- Role tracking is easier inside one short segment. In a long clip, players
  swap sides, walk around, and temporarily leave the ready/serve posture. The
  raw `A/B` role series can drift or become inconsistent across the set break,
  so serve-order inference sees a worse role sequence.
- LET / serve-order inference is downstream of start detection. If the long
  scan misses one real start, the 2-serve rule sees an impossible pattern and
  may relabel the wrong event as LET or create a gap marker. In a short clip
  with fewer misses, the same logic appears correct.
- The old short debug path effectively isolated a stable game state: same table
  ROI, fewer side-swap frames, fewer spectators/adjacent-table distractions, and
  fewer long gaps. The full clip mixes Set 1, side swap, Set 2, and off-rally
  behavior into one signal.
- Therefore the apparent "short input works / long input fails" is not because
  the video content changed. It is because the detector's assumptions are not
  invariant to input duration and match-state changes.

Implementation update after this feedback:
- Step 3.1 now uses a chunked overlap detector path in
  `backend/step3_rally_start_review.py`.
- Current detector id: `chunked_overlap_local_visual_let_v2_151s`.
- The full working video is split into `151s` windows with `10s` overlap.
  Each window still runs the same existing `build_rally_timeline()` detector;
  Step 3.1 then maps chunk-local timestamps back to the full input timeline
  and de-duplicates overlap candidates.
- Rationale: the older bounded `0-151s` debug run was materially better than
  the full-video run, so Step 3.1 should preserve that local-normalization
  behavior instead of scanning the whole clip as one global signal.
- LET detection is automatic. LET rows come from the existing player-sandwich
  path in `backend/ai_multistream_rally.py`, using local abort evidence and
  serve-order replay inference after the chunk-local start list is dense.
  The old full-video false LET issue came from applying serve-order inference
  to an incomplete global start list.

Latest implementation test:
- Test artifact:
  `runtime_jobs/debug_2_sets_step31_chunked151_latest/step3_1_review/summary.md`
- Events JSON:
  `runtime_jobs/debug_2_sets_step31_chunked151_latest/step3_1_review/step3_1_rally_start_events.json`
- Result:
  - `38 total review rows`
  - `38 detected starts`
  - `36 scoring`
  - `2 LET/non-scoring`
  - `4 needs-review / rule-conflict rows`
- Operator verdict:
  - `38 detected starts` is **wrong** and too high versus the real match.
  - This run is not accepted as Step 3.1 output.
  - The current implementation improved several previously missed timestamps,
    but it also introduced over-detection / false-positive starts.
- Confirmed improvements against the operator feedback:
  - `00:18.252` is now detected as `LET`.
  - `00:32.099` is now detected.
  - `00:41.375` is scoring, not LET.
  - `00:52.286` is scoring, not LET.
  - `01:03.797` and `01:05.265` are detected.
  - `02:13.867`, `02:51/02:52`, `02:56`, `03:09`, and `03:16` are detected.
- Remaining Step 3.1 issue:
  - `02:02.306` and `02:49.478` are now auto-promoted scoring starts from
    strong double-serve singleton-gap evidence, with
    `source="serve_order_gap_auto_repair"`.
  - The remaining 4 review rows are rule-conflict rows later in the clip, not
    singleton-gap missing-start markers.
- Important lesson:
  - Auto-promoting a serve-order gap directly into a scoring rally can inflate
    the total rally count.
  - Table-tennis serve-order rules are useful to locate suspicious windows, but
    they are not sufficient proof that a rally start exists.
  - A gap marker should trigger targeted visual rescan / visual confirmation
    before it becomes a confirmed scoring start.
  - Recovered starts must carry explicit evidence. If the visual detector cannot
    confirm the start, keep it as `needs_review` / `suspect_gap`, not as a real
    scoring row.
  - Step 3.1 must optimize for both: no missed starts and no invented starts.
- What did work:
  - Chunked/bounded scanning recovered several real timestamps that the full
    global scan missed.
  - Running on local windows appears closer to the older good behavior than
    scanning the whole video as one signal.
  - The old false LET rows at `00:41` and `00:52` were removed in this test.
- What is still wrong:
  - Total count is too high.
  - Some newly created rows are likely false positives or unsafe rule repairs.
  - The current Step 3.2 rerun is not useful as a quality signal because it is
    based on a wrong Step 3.1 timeline.

Latest Step 3.2 rerun on the repaired Step 3.1 events:
- Summary:
  `runtime_jobs/debug_2_sets_step31_chunked151_latest/step3_2_side_state_review/summary.md`
- Result:
  - `38 total starts`
  - `36 scoring`
  - `2 LET/non-scoring`
  - side evidence: `23 identified`, `3 inferred`, `12 unknown`
  - `4` rule-conflict rows remain
- Interpretation:
  - This Step 3.2 run is not a valid downstream baseline because its Step 3.1
    input over-counts rallies.
  - Do not spend more time optimizing Step 3.2 until Step 3.1 total rally count
    is correct.

## Current Commands

Clean full Step 3.1 rerun:

```powershell
.\.venv\Scripts\python.exe scripts\step3_1_rally_start_review.py `
  --video inputs\raw_matches\2_sets.mp4 `
  --start 0 `
  --end 376.046 `
  --out-dir runtime_jobs\debug_2_sets_full\step3_1_review `
  --player-a-name "Trần Quang Vinh" `
  --player-b-name "Nguyễn Bá Thảo" `
  --force `
  --force-clip
```

Step 3.2 only, using Step 3.1 events:

```powershell
.\.venv\Scripts\python.exe scripts\step3_2_side_state_review.py `
  --video inputs\raw_matches\2_sets.mp4 `
  --events-json runtime_jobs\debug_2_sets_full\step3_1_review\step3_1_rally_start_events.json `
  --out-dir runtime_jobs\debug_2_sets_full\step3_2_side_state_review `
  --player-a-name "Trần Quang Vinh" `
  --player-b-name "Nguyễn Bá Thảo"
```

Step 3.3 logic audit is intentionally paused for now.

## Next Debug Target

Debug Step 3.1 start-time detection before touching Step 3.2/3.3/3.4 again.

Questions to answer:
- Which real rally/LET starts in `2_sets.mp4` are missing?
- Which detected starts are false positives?
- Which LET rows are mislabeled?
- Did the current full-clip run regress because old per-set logic was better?
- Can the older better behavior be recovered from commit history or previous
  debug artifacts?
- Which of the newly added chunked / auto-repair rows caused the total to become
  too high?
- Should `serve_order_gap_auto_repair` be demoted back to `needs_review` unless
  a targeted visual rescan confirms the start?

## Non-Regression Rules

- Do not hide Step 3.1 errors inside Step 3.2 or Step 3.3.
- Do not let Step 3.2 infer side/player from a wrong or missing start time.
- Do not let Step 3.3 create new rally starts by itself. Missing starts must go
  back to Step 3.1 targeted repair.
- Do not count a rule-only repaired gap as a confirmed rally unless there is
  visual confirmation.
- Do not continue to Step 4 while Step 3.1 start times are wrong.
- Keep unknown as unknown. Do not assign a player or server by guessing.
