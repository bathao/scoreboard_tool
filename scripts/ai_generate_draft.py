# scripts/ai_generate_draft.py
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

# Optional deps
try:
    import cv2  # type: ignore
except Exception as e:
    cv2 = None

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None


# ----------------------------
# ROI helpers
# ----------------------------
def _to_int(v) -> Optional[int]:
    try:
        return int(round(float(v)))
    except Exception:
        return None


def parse_roi(roi_raw: Any) -> Dict[str, int]:
    """
    Normalize ROI into dict: {"x":int,"y":int,"w":int,"h":int}

    Supports:
      - dict: {x,y,w,h} or {x,y,width,height} or {left,top,right,bottom}
      - list/tuple: [x,y,w,h] or [l,t,r,b]
      - nested: roi/table/bbox/rect/region/crop/area
      - polygon points -> bbox
    """
    def from_xywh(x, y, w, h):
        x = _to_int(x); y = _to_int(y); w = _to_int(w); h = _to_int(h)
        if None in (x, y, w, h):
            return None
        if w <= 0 or h <= 0:
            return None
        return {"x": x, "y": y, "w": w, "h": h}

    def from_ltrb(l, t, r, b):
        l = _to_int(l); t = _to_int(t); r = _to_int(r); b = _to_int(b)
        if None in (l, t, r, b):
            return None
        return from_xywh(l, t, r - l, b - t)

    def bbox_from_points(pts):
        xs, ys = [], []
        for p in pts or []:
            if isinstance(p, dict):
                xs.append(_to_int(p.get("x")))
                ys.append(_to_int(p.get("y")))
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                xs.append(_to_int(p[0]))
                ys.append(_to_int(p[1]))
        xs = [v for v in xs if v is not None]
        ys = [v for v in ys if v is not None]
        if not xs or not ys:
            return None
        return from_ltrb(min(xs), min(ys), max(xs), max(ys))

    def try_parse(obj):
        # list
        if isinstance(obj, (list, tuple)):
            if len(obj) == 4:
                a = from_xywh(obj[0], obj[1], obj[2], obj[3])
                if a:
                    return a
                b = from_ltrb(obj[0], obj[1], obj[2], obj[3])
                if b:
                    return b
            return None

        # dict
        if isinstance(obj, dict):
            # direct xywh
            if all(k in obj for k in ("x", "y", "w", "h")):
                return from_xywh(obj["x"], obj["y"], obj["w"], obj["h"])
            if all(k in obj for k in ("x", "y", "width", "height")):
                return from_xywh(obj["x"], obj["y"], obj["width"], obj["height"])
            if all(k in obj for k in ("left", "top", "width", "height")):
                return from_xywh(obj["left"], obj["top"], obj["width"], obj["height"])
            # ltrb
            if all(k in obj for k in ("left", "top", "right", "bottom")):
                return from_ltrb(obj["left"], obj["top"], obj["right"], obj["bottom"])
            if all(k in obj for k in ("l", "t", "r", "b")):
                return from_ltrb(obj["l"], obj["t"], obj["r"], obj["b"])
            if all(k in obj for k in ("x1", "y1", "x2", "y2")):
                return from_ltrb(obj["x1"], obj["y1"], obj["x2"], obj["y2"])

            # polygon
            for pk in ("points", "polygon", "poly", "vertices"):
                if pk in obj and isinstance(obj[pk], list):
                    bb = bbox_from_points(obj[pk])
                    if bb:
                        return bb

            # nested common keys
            for nk in ("roi", "table", "bbox", "rect", "region", "crop", "area"):
                if nk in obj:
                    got = try_parse(obj[nk])
                    if got:
                        return got

            # shallow scan
            for v in obj.values():
                got = try_parse(v)
                if got:
                    return got

        return None

    roi = try_parse(roi_raw)
    if not roi:
        if isinstance(roi_raw, dict):
            raise ValueError(f"Unsupported ROI format. Top-level keys={list(roi_raw.keys())}")
        raise ValueError(f"Unsupported ROI format. Got type={type(roi_raw)}")
    return roi


def clamp_roi(roi: Dict[str, int], W: int, H: int) -> Dict[str, int]:
    """
    Clamp ROI to be inside frame. Prevent crop crash.
    """
    x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return {"x": x, "y": y, "w": w, "h": h}


# ----------------------------
# Video IO
# ----------------------------
@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: Optional[int]


def _cv2_open(video_path: str):
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) not installed. Please: pip install opencv-python")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    return cap


def read_video_frames_cv2(
    video_path: str,
    *,
    start_sec: float = 0.0,
    max_sec: Optional[float] = None,
    stride: int = 1,
) -> Tuple[VideoInfo, Iterator[Tuple[int, float, Any]]]:
    """
    Yields (frame_idx, timestamp_sec, frame_bgr_uint8)
    """
    cap = _cv2_open(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_count = n if n > 0 else None

    if fps <= 1e-6:
        fps = 30.0  # fallback

    # Seek
    if start_sec > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)

    info = VideoInfo(width=W, height=H, fps=float(fps), frame_count=frame_count)

    def gen():
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
        start_time = time.time()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ts = (cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0

            # stop by time window
            if max_sec is not None and ts >= (start_sec + max_sec):
                break

            if stride > 1 and (frame_idx % stride) != 0:
                frame_idx += 1
                continue

            yield frame_idx, ts, frame
            frame_idx += 1

        cap.release()
        _ = start_time

    return info, gen()


# Try to use your video_gpu_io.py if available
def read_video_frames(
    video_path: str,
    *,
    start_sec: float = 0.0,
    max_sec: Optional[float] = None,
    stride: int = 1,
) -> Tuple[VideoInfo, Iterator[Tuple[int, float, Any]]]:
    """
    Wrapper: use scripts.video_gpu_io if it exposes a compatible API.
    Fallback to cv2.
    """
    try:
        # Expect you created scripts/video_gpu_io.py
        from scripts import video_gpu_io  # type: ignore

        if hasattr(video_gpu_io, "read_video_frames"):
            info, it = video_gpu_io.read_video_frames(
                video_path,
                start_sec=start_sec,
                max_sec=max_sec,
                stride=stride,
            )
            # normalize info
            if isinstance(info, dict):
                info = VideoInfo(
                    width=int(info["width"]),
                    height=int(info["height"]),
                    fps=float(info["fps"]),
                    frame_count=info.get("frame_count"),
                )
            return info, it
    except Exception:
        pass

    return read_video_frames_cv2(video_path, start_sec=start_sec, max_sec=max_sec, stride=stride)


# ----------------------------
# Draft generation (baseline)
# ----------------------------
@dataclass
class DraftEvent:
    t: float
    kind: str
    value: float


def motion_energy(prev_gray, gray) -> float:
    # mean absolute difference
    import numpy as np  # local import to avoid hard dependency at top
    diff = cv2.absdiff(prev_gray, gray)
    return float(np.mean(diff))


def detect_events_from_motion(
    frames_iter: Iterator[Tuple[int, float, Any]],
    roi: Dict[str, int],
    *,
    debug: bool = False,
    motion_threshold: float = 6.0,
    min_gap_sec: float = 0.5,
) -> List[DraftEvent]:
    """
    Baseline: create an "event" when motion energy in ROI crosses threshold.
    Replace this with your real logic later.
    """
    if cv2 is None:
        raise RuntimeError("cv2 required for baseline motion detection.")

    x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]

    events: List[DraftEvent] = []
    prev_gray = None
    last_event_t = -1e9

    processed = 0
    for frame_idx, ts, frame in frames_iter:
        processed += 1
        crop = frame[y:y + h, x:x + w]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        if prev_gray is None:
            prev_gray = gray
            continue

        e = motion_energy(prev_gray, gray)
        prev_gray = gray

        if debug and processed % 120 == 0:
            print(f"[DEBUG] idx={frame_idx} t={ts:.3f} motion={e:.3f}")

        if e >= motion_threshold and (ts - last_event_t) >= min_gap_sec:
            events.append(DraftEvent(t=float(ts), kind="motion_peak", value=float(e)))
            last_event_t = ts

    return events


def events_to_draft_json(
    video_path: str,
    video_info: VideoInfo,
    roi: Dict[str, int],
    events: List[DraftEvent],
    best_of: int,
) -> Dict[str, Any]:
    return {
        "meta": {
            "video": str(video_path),
            "width": video_info.width,
            "height": video_info.height,
            "fps": video_info.fps,
            "frame_count": video_info.frame_count,
            "best_of": best_of,
        },
        "roi": roi,
        "events": [
            {"t": e.t, "kind": e.kind, "value": e.value}
            for e in events
        ],
    }


# ----------------------------
# Main
# ----------------------------
def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Generate draft match events from a table tennis video (baseline)."
    )
    p.add_argument("--video", required=True, help="Path to input video, e.g. Vinh_1280.mp4")
    p.add_argument("--roi", required=True, help="Path to ROI json, e.g. matches/table_roi_001.json")
    p.add_argument("--out", required=True, help="Output draft json path")
    p.add_argument("--best-of", type=int, default=5)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--stride", type=int, default=1, help="Process every Nth frame (e.g. 3 => ~20fps from 60fps)")
    p.add_argument("--start-sec", type=float, default=0.0)
    p.add_argument("--max-sec", type=float, default=None)
    p.add_argument("--motion-thr", type=float, default=6.0, help="Baseline motion threshold")
    p.add_argument("--min-gap-sec", type=float, default=0.5, help="Min seconds between events")

    args = p.parse_args(argv)

    video_path = args.video
    roi_path = Path(args.roi)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load ROI
    roi_raw = load_json(roi_path)
    if args.debug:
        print("[DEBUG] roi_raw type:", type(roi_raw))
        if isinstance(roi_raw, dict):
            print("[DEBUG] roi_raw keys:", list(roi_raw.keys()))
        else:
            print("[DEBUG] roi_raw:", roi_raw)

    roi = parse_roi(roi_raw)

    # Open video + get info
    info, frames_iter = read_video_frames(
        video_path,
        start_sec=float(args.start_sec),
        max_sec=args.max_sec,
        stride=max(1, int(args.stride)),
    )

    print(f"[INFO] Video: {video_path}  size={info.width}x{info.height}  fps={info.fps:.3f}")

    # Clamp ROI to frame bounds (IMPORTANT for your ROI that is larger than 1280x720)
    roi_clamped = clamp_roi(roi, info.width, info.height)
    if roi_clamped != roi:
        print(f"[WARN] ROI clamped from {roi} -> {roi_clamped}")
    else:
        print(f"[INFO] ROI: {roi_clamped}")

    # Wrap iterator with progress if possible
    # We don't always know frame_count in iterator (especially GPU pipeline), so tqdm without total.
    if tqdm is not None:
        frames_iter = tqdm(frames_iter, desc="Processing", unit="frame")

    t0 = time.time()
    events = detect_events_from_motion(
        frames_iter,
        roi_clamped,
        debug=bool(args.debug),
        motion_threshold=float(args.motion_thr),
        min_gap_sec=float(args.min_gap_sec),
    )
    t1 = time.time()

    elapsed = t1 - t0
    print(f"[INFO] Done. events={len(events)}  elapsed={elapsed:.2f}s")

    draft = events_to_draft_json(
        video_path=video_path,
        video_info=info,
        roi=roi_clamped,
        events=events,
        best_of=int(args.best_of),
    )

    out_path.write_text(json.dumps(draft, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] Saved: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())