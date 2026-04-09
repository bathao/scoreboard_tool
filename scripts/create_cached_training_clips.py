from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from winner_finetune_common import load_manifest_rows, resolve_clip_path, resolve_cached_clip_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build lightweight cached 4-frame clips for Qwen3-VL training.")
    parser.add_argument("--manifest", default="dataset/collections/finetune_dataset/manifest.jsonl")
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--out-dir", default="dataset/collections/finetune_dataset/cache/qwen3vl4b_4f384_v1")
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--target-shortest-edge", type=int, default=384)
    parser.add_argument("--fps", type=float, default=1.0, help="Playback FPS for cache clips. Keep at 1.0 to preserve 4 sampled frames.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _evenly_spaced_indices(total_frames: int, num_frames: int) -> list[int]:
    if total_frames <= 1:
        return [0 for _ in range(num_frames)]
    positions = np.linspace(0, total_frames - 1, num=num_frames)
    return [int(round(float(pos))) for pos in positions]


def _resize_preserving_aspect(frame: np.ndarray, target_shortest_edge: int) -> np.ndarray:
    height, width = frame.shape[:2]
    shortest = min(height, width)
    if shortest <= 0:
        return frame
    scale = float(target_shortest_edge) / float(shortest)
    new_width = max(2, int(round(width * scale)))
    new_height = max(2, int(round(height * scale)))
    if new_width % 2 == 1:
        new_width += 1
    if new_height % 2 == 1:
        new_height += 1
    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)


def _read_frame_at(cap: cv2.VideoCapture, frame_index: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame


def _build_cached_clip(src_path: Path, dst_path: Path, num_frames: int, target_shortest_edge: int, fps: float) -> dict[str, object]:
    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source clip: {src_path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames <= 0:
            raise RuntimeError(f"Source clip reports no frames: {src_path}")
        frame_indices = _evenly_spaced_indices(total_frames, num_frames)
        frames: list[np.ndarray] = []
        last_good: np.ndarray | None = None
        for frame_index in frame_indices:
            frame = _read_frame_at(cap, frame_index)
            if frame is None:
                if last_good is None:
                    raise RuntimeError(f"Failed to decode frame {frame_index} from {src_path}")
                frame = last_good.copy()
            frame = _resize_preserving_aspect(frame, target_shortest_edge)
            last_good = frame
            frames.append(frame)

        out_height, out_width = frames[0].shape[:2]
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(dst_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (int(out_width), int(out_height)),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to create cache clip: {dst_path}")
        try:
            for frame in frames:
                writer.write(frame)
        finally:
            writer.release()

        return {
            "src_path": str(src_path).replace("\\", "/"),
            "dst_path": str(dst_path).replace("\\", "/"),
            "total_frames": total_frames,
            "sampled_frame_indices": frame_indices,
            "cache_frame_count": len(frames),
            "cache_height": out_height,
            "cache_width": out_width,
            "cache_fps": float(fps),
        }
    finally:
        cap.release()


def main() -> None:
    args = _parse_args()
    manifest_path = Path(args.manifest)
    dataset_root = Path(args.dataset_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_manifest_rows(manifest_path)
    records: list[dict[str, object]] = []
    created = 0
    skipped = 0

    for row in rows:
        src_path = resolve_clip_path(dataset_root, row)
        dst_path = resolve_cached_clip_path(out_dir, row)
        if dst_path.exists() and not bool(args.overwrite):
            skipped += 1
            records.append(
                {
                    "sample_id": row["sample_id"],
                    "status": "skipped_existing",
                    "src_path": str(src_path).replace("\\", "/"),
                    "dst_path": str(dst_path).replace("\\", "/"),
                }
            )
            continue

        details = _build_cached_clip(
            src_path=src_path,
            dst_path=dst_path,
            num_frames=int(args.num_frames),
            target_shortest_edge=int(args.target_shortest_edge),
            fps=float(args.fps),
        )
        created += 1
        records.append({"sample_id": row["sample_id"], "status": "created", **details})

    summary = {
        "schema": "winner_cache_clip_summary_v1",
        "manifest": str(manifest_path).replace("\\", "/"),
        "dataset_root": str(dataset_root).replace("\\", "/"),
        "out_dir": str(out_dir).replace("\\", "/"),
        "num_frames": int(args.num_frames),
        "target_shortest_edge": int(args.target_shortest_edge),
        "fps": float(args.fps),
        "row_count": len(rows),
        "created_count": created,
        "skipped_count": skipped,
    }
    (out_dir / "build_records.jsonl").write_text(
        "\n".join(json.dumps(record, ensure_ascii=True) for record in records) + "\n",
        encoding="utf-8",
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
