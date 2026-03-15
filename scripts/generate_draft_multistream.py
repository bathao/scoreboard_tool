from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import torch

sys.path.append(str(Path(__file__).parent.parent))

from backend.ai_contract import DraftMatch, DraftPointEvent, save_draft_match
from backend.ai_multistream_rally import detect_multistream_rallies, extract_multistream_signals


def build_draft(
    video_path: str,
    table_weights_path: str,
    *,
    pose_weights_path: str = "weights/yolov8x-pose.pt",
    best_of: int = 5,
    stride: int = 2,
    mode: str = "fused",
    player_margin_px: int = 220,
    player_fuse_gain: float = 1.0,
    player_signal_source: str = "role_tracker",
) -> DraftMatch:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for multi-stream draft generation.")

    if best_of <= 0 or best_of % 2 == 0:
        raise ValueError("best_of must be a positive odd number.")

    signals = extract_multistream_signals(
        video_path,
        table_weights_path,
        pose_weights_path=pose_weights_path,
        stride=max(1, int(stride)),
        player_margin_px=int(player_margin_px),
        player_fuse_gain=float(player_fuse_gain),
        player_signal_source=player_signal_source,
        device="cuda",
    )
    segments = detect_multistream_rallies(signals, mode=mode)

    v_path = Path(video_path).resolve()
    points: List[DraftPointEvent] = []
    for i, seg in enumerate(segments, start=1):
        flags = list(seg.flags)
        if mode == "fused":
            flags.append("multistream_fused")
        elif mode == "table_refined":
            flags.append("table_role_refined")
        flags.append(f"player_signal_{signals.player_signal_source}")
        points.append(
            DraftPointEvent(
                id=f"pt_{i:04d}",
                t_start=float(seg.t_start),
                t_end=float(seg.t_end),
                winner="unknown",
                confidence=float(seg.confidence),
                flags=sorted(set(flags)),
                source="ai",
            )
        )

    return DraftMatch(
        video_path=str(v_path),
        video_fps=float(signals.effective_fps * max(1, int(stride))),
        best_of=int(best_of),
        created_at=datetime.now(timezone.utc).isoformat(),
        roi=signals.roi.to_dict() | {
            "x": int(signals.roi.x),
            "y": int(signals.roi.y),
            "w": int(signals.roi.w),
            "h": int(signals.roi.h),
        },
        points=points,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate draft JSON using table-only or multi-stream fused segmentation.")
    parser.add_argument("--video", required=True, help="Path to source video")
    parser.add_argument("--weights", default="weights/yolov8x_table.pt", help="Path to YOLO table weights")
    parser.add_argument("--pose-weights", default="weights/yolov8x-pose.pt", help="Path to YOLO pose weights")
    parser.add_argument("--out", required=True, help="Output draft JSON path")
    parser.add_argument("--best-of", type=int, default=5)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--mode", choices=["table", "fused", "table_refined"], default="fused")
    parser.add_argument("--player-margin-px", type=int, default=220)
    parser.add_argument("--player-fuse-gain", type=float, default=1.0)
    parser.add_argument("--player-signal-source", choices=["role_tracker", "nearest_two"], default="role_tracker")
    args = parser.parse_args()

    draft = build_draft(
        args.video,
        args.weights,
        pose_weights_path=args.pose_weights,
        best_of=args.best_of,
        stride=args.stride,
        mode=args.mode,
        player_margin_px=args.player_margin_px,
        player_fuse_gain=args.player_fuse_gain,
        player_signal_source=args.player_signal_source,
    )
    out_path = Path(args.out)
    save_draft_match(out_path, draft)
    print(f"[OK] Saved {args.mode} draft: {out_path} | points={len(draft.points)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
