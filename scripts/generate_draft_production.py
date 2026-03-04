from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).parent.parent))

from backend.ai_contract import DraftMatch, DraftPointEvent, save_draft_match
from backend.ai_rally_segmentation import detect_rally_segments_advanced_gpu
from backend.ai_table_roi_dl import DLConfig, detect_table_roi_dl
from backend.video_gpu_io import nvdec_bgr24_stream, probe_video_ffprobe


def build_draft(
    video_path: str,
    weights_path: str,
    *,
    best_of: int = 5,
    stride: int = 2,
) -> DraftMatch:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for production draft generation.")

    if best_of <= 0 or best_of % 2 == 0:
        raise ValueError("best_of must be a positive odd number.")

    v_path = Path(video_path).resolve()
    w_path = Path(weights_path).resolve()
    if not v_path.exists():
        raise FileNotFoundError(f"Video not found: {v_path}")
    if not w_path.exists():
        raise FileNotFoundError(f"Weights not found: {w_path}")

    info = probe_video_ffprobe(str(v_path))
    roi = detect_table_roi_dl(
        str(v_path),
        cfg=DLConfig(weights_path=str(w_path), device="cuda"),
    )
    tx, ty, tw, th = roi.as_tuple()

    energies: List[float] = []
    timestamps: List[float] = []
    prev_frame_gpu = None

    frame_iter = nvdec_bgr24_stream(
        str(v_path),
        info.width,
        info.height,
        crop_roi=(tx, ty, tw, th),
    )

    for idx, frame_np in enumerate(frame_iter):
        if idx % stride != 0:
            continue

        curr_gpu = torch.from_numpy(frame_np).to("cuda").float()

        if prev_frame_gpu is not None:
            diff = torch.abs(curr_gpu - prev_frame_gpu)
            diff_max = F.max_pool2d(
                diff.permute(2, 0, 1).unsqueeze(0),
                kernel_size=3,
                stride=1,
                padding=1,
            )
            energies.append(float(diff_max.mean().item()))
            timestamps.append(float(idx / info.fps))

        prev_frame_gpu = curr_gpu

    segments = detect_rally_segments_advanced_gpu(
        energies,
        timestamps,
        effective_fps=info.fps / stride,
    )

    points: List[DraftPointEvent] = []
    for i, seg in enumerate(segments, start=1):
        points.append(
            DraftPointEvent(
                id=f"pt_{i:04d}",
                t_start=float(seg.t_start),
                t_end=float(seg.t_end),
                winner="unknown",
                confidence=float(seg.confidence),
                flags=list(seg.flags),
                source="ai",
            )
        )

    draft = DraftMatch(
        video_path=str(v_path),
        video_fps=float(info.fps),
        best_of=int(best_of),
        created_at=datetime.now(timezone.utc).isoformat(),
        roi={"x": int(tx), "y": int(ty), "w": int(tw), "h": int(th)},
        points=points,
    )
    return draft


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate production draft JSON from raw table-tennis clip."
    )
    parser.add_argument("--video", required=True, help="Path to source video")
    parser.add_argument(
        "--weights",
        default="weights/yolov8x_table.pt",
        help="Path to YOLO table weights",
    )
    parser.add_argument("--out", required=True, help="Output draft JSON path")
    parser.add_argument("--best-of", type=int, default=5, help="Match format (3/5/7)")
    parser.add_argument("--stride", type=int, default=2, help="Process every Nth frame")

    args = parser.parse_args()

    draft = build_draft(
        args.video,
        args.weights,
        best_of=args.best_of,
        stride=max(1, int(args.stride)),
    )

    out_path = Path(args.out)
    save_draft_match(out_path, draft)
    print(f"[OK] Saved draft: {out_path} | points={len(draft.points)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
