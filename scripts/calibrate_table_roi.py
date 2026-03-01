# scripts/calibrate_table_roi.py
from __future__ import annotations

import argparse
from pathlib import Path

from backend.ai_table_roi import save_table_roi, detect_table_roi_classical
from backend.ai_table_roi_dl import detect_table_roi_dl, DLConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, default="input.mp4")
    ap.add_argument("--out", type=str, default="matches/table_roi_001.json")
    ap.add_argument("--dl", action="store_true", help="Use DL if available (ultralytics + local weights)")
    ap.add_argument("--weights", type=str, default="", help="Path to local YOLO weights (required if --dl)")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.out)

    if args.dl:
        if not args.weights:
            raise SystemExit("ERROR: --dl requires --weights <path_to_weights>")
        roi = detect_table_roi_dl(
            args.video,
            cfg=DLConfig(weights_path=args.weights, device=args.device),
            debug=args.debug,
        )
    else:
        roi = detect_table_roi_classical(args.video, debug=args.debug)

    save_table_roi(out_path, roi)
    print(f"Saved ROI: {out_path}")
    print(f"ROI: {roi}")


if __name__ == "__main__":
    main()