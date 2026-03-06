# Ground Truth Guide (Phase 2)

## Muc dich
Tao file GT de benchmark quality cua pipeline:
- Rally detection (segment overlap)
- Winner prediction

## Dinh dang file
Dung cung schema voi `backend/ai_contract.py` (`draft_match_v1`).
Ban co the copy tu file refined du doan, sau do chinh sua `points` cho dung.

## Quy tac gan nhan toi thieu
1. Moi point bat buoc co: 
   - `id`
   - `t_start`
   - `t_end`
   - `winner` (`player_a` / `player_b` / `unknown`)
2. `t_start < t_end`
3. Thu tu point tang dan theo thoi gian.
4. Neu khong chac winner thi de `unknown` (script se bo qua trong winner accuracy).

## Cach tao nhanh GT
1. Copy file refined:
   - `matches/Vinh_set1_phase1_refined.json` -> `matches/ground_truth/Vinh_set1_gt.json`
2. Chinh sua cac doan sai:
   - tach/gop rally
   - sua `t_start`, `t_end`
   - sua `winner`
3. Lam tuong tu voi `Vinh_set2`.

## Chay benchmark
Single clip:
`python scripts/evaluate_phase2.py --pred matches/Vinh_set1_phase1_refined.json --gt matches/ground_truth/Vinh_set1_gt.json --name Vinh_set1 --iou-threshold 0.5`

Multi-clip qua manifest:
`python scripts/evaluate_phase2.py --manifest benchmarks/phase2_manifest.example.json --iou-threshold 0.5 --out debug_report/phase2_eval_report.json`
