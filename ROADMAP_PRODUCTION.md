# Scoreboard Tool - Production Roadmap

## Final Goal
- Input: raw table-tennis video 20-30 minutes, 1K/2K 60fps, fixed tripod per clip.
- Output: 1080p rendered video with accurate scoreboard (points/sets/match).
- Accuracy target: rally + winner pipeline >= 90%, target 95%+.
- Manual correction target: minimal, only low-confidence points.

## Phase 1 - Unified Production Pipeline
### Objective
Create one consistent flow from raw video to valid draft JSON and final render.

### Deliverables
1. Single production draft generator:
   - `scripts/generate_draft_production.py`
   - Uses:
     - table ROI detection (`ai_table_roi_dl`)
     - GPU motion extraction (`video_gpu_io + torch`)
     - segmentation (`ai_rally_segmentation`)
     - strict draft contract (`ai_contract`)
2. Stable execution sequence:
   - Draft generation
   - AI refinement
   - Optional manual review for flagged points
   - Final render + audio
3. Remove ambiguity between experimental/debug scripts and production scripts.

### Success Criteria
- A full 20-30 minute clip runs end-to-end without schema errors.
- Draft JSON is fully compatible with `backend/ai_contract.py`.
- Renderer consumes refined events and produces 1080p output.

## Phase 2 - Accuracy Upgrade to 90-95%
### Objective
Improve rally boundary quality and winner prediction quality.

### Deliverables
1. Benchmark set:
   - Build a small labeled dataset (5-10 clips).
   - Ground truth: rally start/end + winner.
2. Metrics:
   - Rally segment precision/recall/F1 (IoU-based match).
   - Winner accuracy.
   - Review-rate (% points requiring human review).
   - Tool: `scripts/evaluate_phase2.py`
3. Model/rule improvements:
   - Threshold auto-calibration per clip.
   - Multi-signal fusion (table energy + player cue confidence).
   - Confidence calibration and stricter flagging.

### Success Criteria
- Rally F1 reaches operational target.
- Winner accuracy reaches >= 90% on validation set.
- Review-rate significantly reduced while preserving correctness.

### Benchmark Run Commands
1. Single clip:
   - `python scripts/evaluate_phase2.py --pred matches/<name>_phase1_refined.json --gt matches/ground_truth/<name>_gt.json --name <name> --iou-threshold 0.5`
2. Multi-clip:
   - `python scripts/evaluate_phase2.py --manifest benchmarks/phase2_manifest.example.json --iou-threshold 0.5 --out debug_report/phase2_eval_report.json`

## Phase 3 - Production Hardening
### Objective
Make the system reliable for repeated real-world usage.

### Deliverables
1. Job profiles/config:
   - Config per venue/camera style.
2. Observability:
   - Runtime logs
   - Error categories
   - QC summary per processed clip
3. Regression safety:
   - Add tests for draft contract and pipeline compatibility.
   - Keep old clips as regression fixtures.

### Success Criteria
- Stable processing on new unseen clips with predictable quality.
- Fast troubleshooting when quality drops due to scene changes.
- Consistent output quality across multiple tournaments/venues.

## Current Recommended Run Flow
1. Generate draft:
   - `python scripts/generate_draft_production.py --video <input.mp4> --weights weights/yolov8x_table.pt --out matches/<name>_draft.json --best-of 5 --stride 2`
2. Refine winners:
   - `python scripts/ai_refine_draft.py --draft matches/<name>_draft.json --out matches/<name>_refined.json --model llama3.2-vision`
3. Render final:
   - `python scripts/final_render.py --video <input.mp4> --json matches/<name>_refined.json --out outputs/<name>_1080p_final.mp4 --unknown-winner-policy player_a`
4. One-command run:
   - `python scripts/run_production_pipeline.py --video <input.mp4> --weights weights/yolov8x_table.pt --draft-out matches/<name>_draft.json --final-out outputs/<name>_1080p_final.mp4 --best-of 5 --stride 2`

## Notes
- Existing debug scripts remain useful for diagnostics but should not be the default production path.
- Keep draft schema strict; do not bypass `save_draft_match` validation.
