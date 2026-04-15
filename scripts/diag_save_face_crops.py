"""Save face crops from rank=0 and rank=1 at early timestamps for visual inspection.

This script helps verify which player is at which rank (near/far) in Set 1.
Saves both the natural display crop (224x224) and the ArcFace-aligned crop (112x112).

Usage:
    python scripts/diag_save_face_crops.py --video data/input/2_sets.mp4
    python scripts/diag_save_face_crops.py --video data/input/2_sets.mp4 --t-start 1 --t-end 50 --fps 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np

from backend.player_identity import align_face_from_keypoints, FaceEmbedder, FaceDB, face_similarity
from backend.player_identification import _face_kpts_in_head_region, detect_table_roi_and_player_zone

FACE_MODEL_PATH  = ROOT / "data" / "models" / "face" / "w600k_r50.onnx"
FACE_DB_PATH     = ROOT / "data" / "players" / "faces.json"
POSE_WEIGHTS     = ROOT / "weights" / "yolov8x-pose.pt"
TABLE_WEIGHTS    = ROOT / "weights" / "yolov8x_table.pt"

_KPT_NOSE = 0
_KPT_LEYE = 1
_KPT_REYE = 2
_MIN_KPT_CONF = 0.4


def _display_crop(frame, kpts_xy, kpts_conf, out_size=224, padding=0.60):
    """Natural-orientation face crop for human display."""
    idxs = [_KPT_NOSE, _KPT_LEYE, _KPT_REYE]
    valid = [i for i in idxs if kpts_conf[i] >= _MIN_KPT_CONF]
    if len(valid) < 2:
        return None
    pts = kpts_xy[valid]
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    span = max(abs(kpts_xy[_KPT_LEYE][0] - kpts_xy[_KPT_REYE][0]) * 2.5, 40.0)
    half = span * (0.5 + padding)
    h, w = frame.shape[:2]
    x1 = max(0, int(cx - half))
    y1 = max(0, int(cy - half * 1.4))
    x2 = min(w, int(cx + half))
    y2 = min(h, int(cy + half))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True)
    parser.add_argument("--t-start", type=float, default=1.0)
    parser.add_argument("--t-end",   type=float, default=50.0)
    parser.add_argument("--fps",     type=float, default=0.5,
                        help="Sampling rate (frames per second). Default=0.5 = every 2s")
    parser.add_argument("--out-dir", default="data/face_crops/diag_rank")
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    rank0_dir = out_dir / "rank0_near"
    rank1_dir = out_dir / "rank1_far"
    rank0_dir.mkdir(parents=True, exist_ok=True)
    rank1_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading YOLO...")
    from ultralytics import YOLO
    yolo = YOLO(str(POSE_WEIGHTS))

    print(f"Loading ArcFace + FaceDB...")
    embedder = FaceEmbedder(FACE_MODEL_PATH)
    db = FaceDB(FACE_DB_PATH)
    print(f"  DB players: {[r.name for r in db.records]}")

    print(f"\nDetecting table ROI + player zone...")
    table_roi, roi = detect_table_roi_and_player_zone(str(args.video), str(TABLE_WEIGHTS))
    if roi:
        print(f"  table ROI: x={table_roi.x} y={table_roi.y} w={table_roi.w} h={table_roi.h}")
        print(f"  player zone: x=[{roi[0]:.0f},{roi[2]:.0f}]  y=[{roi[1]:.0f},{roi[3]:.0f}]")
    else:
        print(f"  table ROI detection failed — scanning full frame")

    print(f"\nScanning {args.video} from t={args.t_start}s to t={args.t_end}s at {args.fps} fps\n")

    cap = cv2.VideoCapture(str(args.video))
    step = 1.0 / args.fps
    t = args.t_start
    saved = {0: 0, 1: 0}

    while t <= args.t_end:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo.predict(frame, verbose=False, half=True, device=0)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            t += step
            continue

        boxes = results[0].boxes.xyxy.cpu().numpy()
        kpts_all = results[0].keypoints
        if kpts_all is None:
            t += step
            continue
        kpts_xy   = kpts_all.xy.cpu().numpy()
        kpts_conf = kpts_all.conf.cpu().numpy()

        # ROI filter
        if roi:
            rx1, ry1, rx2, ry2 = roi
            keep_idx = []
            for ii, b in enumerate(boxes):
                cx = (b[0]+b[2])/2; cy = (b[1]+b[3])/2
                if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                    keep_idx.append(ii)
            boxes    = boxes[keep_idx]
            kpts_xy  = kpts_xy[keep_idx]
            kpts_conf= kpts_conf[keep_idx]

        # Sort by bbox area descending
        areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
        order = sorted(range(len(boxes)), key=lambda i: -areas[i])

        # Save annotated full frame (bboxes + keypoints for top-2 bodies)
        frame_annot = frame.copy()
        # Draw ROI
        if roi:
            cv2.rectangle(frame_annot, (int(roi[0]), int(roi[1])), (int(roi[2]), int(roi[3])),
                          (255, 255, 0), 2)
            cv2.putText(frame_annot, "ROI", (int(roi[0])+4, int(roi[1])+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        colors_rank = [(0, 255, 0), (0, 0, 255)]  # rank0=green, rank1=red
        for ri, di in enumerate(order[:2]):
            b = boxes[di].astype(int)
            col = colors_rank[ri]
            cv2.rectangle(frame_annot, (b[0], b[1]), (b[2], b[3]), col, 2)
            cv2.putText(frame_annot, f"rank{ri}", (b[0], b[1]-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            # Draw nose + eyes keypoints
            for ki, kname in [(0, "N"), (1, "LE"), (2, "RE")]:
                kc = kpts_conf[di][ki]
                if kc >= 0.3:
                    kp = kpts_xy[di][ki].astype(int)
                    cv2.circle(frame_annot, tuple(kp), 5, col, -1)
                    cv2.putText(frame_annot, f"{kname}{kc:.1f}", (kp[0]+4, kp[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
        scale = 0.4
        fh = int(frame_annot.shape[0] * scale)
        fw = int(frame_annot.shape[1] * scale)
        frame_small = cv2.resize(frame_annot, (fw, fh))
        cv2.imwrite(str(out_dir / f"frame_t{t:06.1f}s.jpg"), frame_small)

        for rank_idx, det_idx in enumerate(order[:2]):
            kxy  = kpts_xy[det_idx]
            kcon = kpts_conf[det_idx]

            face_score = float(min(kcon[_KPT_NOSE], kcon[_KPT_LEYE], kcon[_KPT_REYE]))

            # Display crop (natural orientation, large)
            disp = _display_crop(frame, kxy, kcon, out_size=224, padding=0.6)
            # Aligned crop (112x112) for ArcFace
            aligned = align_face_from_keypoints(frame, kxy, kcon, out_size=112)

            # Check bbox contamination (new fix)
            bbox = boxes[det_idx]
            kpts_ok = _face_kpts_in_head_region(kxy, kcon, bbox)
            contaminated = not kpts_ok

            # Compute similarities using display crop (simple resize), matching enrollment method
            sim_str = ""
            embed_img = cv2.resize(disp, (112, 112)) if disp is not None else None
            if not contaminated and embed_img is not None and embed_img.std() >= 8.0 and face_score >= 0.4:
                emb = embedder.embed(embed_img)
                sims = []
                for r in db.records:
                    s = face_similarity(emb, r.embedding_array())
                    sims.append(f"{r.name.split()[-1]}={s:.3f}")
                sim_str = "  " + "  ".join(sims)

            out_dir_rank = rank0_dir if rank_idx == 0 else rank1_dir
            status_tag = "CONTAM" if contaminated else "OK"
            fname_base = f"t{t:06.1f}s_faceQ{face_score:.2f}_{status_tag}"

            # Save display crop (most readable for humans)
            if disp is not None:
                img_out = disp.copy()
                color = (0, 0, 255) if contaminated else (0, 255, 0)
                label_text = f"rank={rank_idx} t={t:.1f}s faceQ={face_score:.2f} {'[CONTAM]' if contaminated else ''}"
                cv2.putText(img_out, label_text,
                            (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
                if sim_str:
                    for li, chunk in enumerate(sim_str.split("  ") if "  " in sim_str else [sim_str]):
                        if chunk.strip():
                            cv2.putText(img_out, chunk.strip(), (4, 36 + li*16),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,200,255), 1)
                cv2.imwrite(str(out_dir_rank / f"{fname_base}_display.jpg"), img_out)
                saved[rank_idx] += 1

            # Save aligned crop (what ArcFace sees)
            if aligned is not None and not contaminated:
                cv2.imwrite(str(out_dir_rank / f"{fname_base}_aligned.jpg"), aligned)

            contam_tag = " [CONTAM-REJECTED]" if contaminated else ""
            print(f"  t={t:5.1f}s  rank={rank_idx}  faceQ={face_score:.2f}{sim_str}{contam_tag}")

        t += step

    cap.release()
    print(f"\nSaved rank=0 (NEAR): {saved[0]} crops → {rank0_dir}")
    print(f"Saved rank=1 (FAR):  {saved[1]} crops → {rank1_dir}")
    print(f"\nOpen these folders to see which player is at each rank.")


if __name__ == "__main__":
    main()
