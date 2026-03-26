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
    ball_fuse_gain: float = 1.15,
    ball_signal_source: str = "none",
) -> DraftMatch:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for multi-stream draft generation.")

    if best_of <= 0 or best_of % 2 == 0:
        raise ValueError("best_of must be a positive odd number.")
    if mode == "player" and player_signal_source == "none":
        raise ValueError("player mode requires a real --player-signal-source")
    if mode == "ball" and ball_signal_source == "none":
        raise ValueError("ball mode requires --ball-signal-source classical")

    effective_player_signal_source = player_signal_source
    effective_ball_signal_source = ball_signal_source
    if mode == "table":
        effective_player_signal_source = "none"
        effective_ball_signal_source = "none"
    elif mode == "player":
        effective_ball_signal_source = "none"
    elif mode == "ball":
        effective_player_signal_source = "none"
    ball_tracking_profile = "standalone" if mode == "ball" else "support"

    signals = extract_multistream_signals(
        video_path,
        table_weights_path,
        pose_weights_path=pose_weights_path,
        stride=max(1, int(stride)),
        player_margin_px=int(player_margin_px),
        player_fuse_gain=float(player_fuse_gain),
        player_signal_source=effective_player_signal_source,
        ball_fuse_gain=float(ball_fuse_gain),
        ball_signal_source=effective_ball_signal_source,
        ball_tracking_profile=ball_tracking_profile,
        device="cuda",
    )
    segments = detect_multistream_rallies(signals, mode=mode)

    v_path = Path(video_path).resolve()
    points: List[DraftPointEvent] = []
    excluded_let_starts: List[dict] = []
    for i, seg in enumerate(segments, start=1):
        flags = list(seg.flags)
        if mode == "fused":
            flags.append("multistream_fused")
        elif mode == "player":
            flags.append("player_only")
        elif mode == "ball":
            flags.append("ball_only")
        elif mode == "table":
            flags.append("table_only")
        elif mode == "table_refined":
            flags.append("table_role_refined")
        elif mode == "table_ball_refined":
            flags.append("table_ball_refined")
        flags.append(f"player_signal_{signals.player_signal_source}")
        flags.append(f"ball_signal_{signals.ball_signal_source}")
        starter_role = str(seg.server_role) if getattr(seg, "server_role", None) in {"A", "B"} else None
        normalized_flags = sorted(set(flags))
        if "let_no_score" in normalized_flags or "rally_label_let" in normalized_flags:
            excluded_let_starts.append(
                {
                    "t_start": float(seg.t_start),
                    "t_end": float(seg.t_end),
                    "starter_role": starter_role,
                    "flags": normalized_flags,
                }
            )
            continue
        points.append(
            DraftPointEvent(
                id=f"pt_{len(points) + 1:04d}",
                t_start=float(seg.t_start),
                t_end=float(seg.t_end),
                starter_role=starter_role,
                winner="unknown",
                confidence=float(seg.confidence),
                flags=normalized_flags,
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
        analysis_metadata={
            "detector_mode": mode,
            "detector_group": "independent" if mode in {"table", "player", "ball"} else "experimental",
            "player_signal_source": signals.player_signal_source,
            "ball_signal_source": signals.ball_signal_source,
            "stride": max(1, int(stride)),
            "excluded_let_count": len(excluded_let_starts),
            "excluded_let_starts": excluded_let_starts,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate draft JSON using independent table/player/ball or multi-stream segmentation.")
    parser.add_argument("--video", required=True, help="Path to source video")
    parser.add_argument("--weights", default="weights/yolov8x_table.pt", help="Path to YOLO table weights")
    parser.add_argument("--pose-weights", default="weights/yolov8x-pose.pt", help="Path to YOLO pose weights")
    parser.add_argument("--out", required=True, help="Output draft JSON path")
    parser.add_argument("--best-of", type=int, default=5)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--mode", choices=["table", "player", "ball", "fused", "table_refined", "table_ball_refined"], default="fused")
    parser.add_argument("--player-margin-px", type=int, default=220)
    parser.add_argument("--player-fuse-gain", type=float, default=1.0)
    parser.add_argument("--player-signal-source", choices=["role_tracker", "nearest_two", "none"], default="role_tracker")
    parser.add_argument("--ball-fuse-gain", type=float, default=1.15)
    parser.add_argument("--ball-signal-source", choices=["none", "classical"], default="none")
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
        ball_fuse_gain=args.ball_fuse_gain,
        ball_signal_source=args.ball_signal_source,
    )
    out_path = Path(args.out)
    save_draft_match(out_path, draft)
    print(f"[OK] Saved {args.mode} draft: {out_path} | total_rallies={len(draft.points)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
