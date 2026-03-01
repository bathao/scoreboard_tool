# backend/ai_rally_segmentation.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np


@dataclass(frozen=True)
class RallySegment:
    t_start: float
    t_end: float
    confidence: float
    flags: List[str]


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def detect_rally_segments_motion(
    video_path: str,
    *,
    roi: Optional[Tuple[int, int, int, int]] = None,  # x, y, w, h
    sample_fps: float = 20.0,
    smooth_window_sec: float = 0.35,
    active_threshold: float = 0.22,
    min_segment_sec: float = 1.0,
    merge_gap_sec: float = 0.7,
    debug: bool = False,
) -> Tuple[List[RallySegment], float]:
    """
    Motion-energy based segmentation.
    Works best for fixed tripod camera.

    Returns: (segments, video_fps)

    Notes:
    - This is MVP. It does not detect winner.
    - Confidence is derived from activity strength and stability.
    - Use ROI to focus on table area if you have it later.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 60.0  # fallback

    # Sampling step
    step = max(1, int(round(fps / sample_fps)))
    effective_sample_fps = fps / step

    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Cannot read first frame")

    h, w = frame.shape[:2]
    if roi is None:
        x, y, rw, rh = 0, 0, w, h
    else:
        x, y, rw, rh = roi
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        rw = max(1, min(rw, w - x))
        rh = max(1, min(rh, h - y))

    prev_gray = cv2.cvtColor(frame[y:y+rh, x:x+rw], cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)

    energies: List[float] = []
    times: List[float] = []

    frame_idx = 0
    sample_idx = 0

    while True:
        # skip frames
        if frame_idx % step != 0:
            ret = cap.grab()
            if not ret:
                break
            frame_idx += 1
            continue

        ret, frame = cap.read()
        if not ret:
            break

        cur_gray = cv2.cvtColor(frame[y:y+rh, x:x+rw], cv2.COLOR_BGR2GRAY)
        cur_gray = cv2.GaussianBlur(cur_gray, (5, 5), 0)

        diff = cv2.absdiff(cur_gray, prev_gray)
        # threshold to reduce noise
        _, diff_thr = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
        motion = float(np.mean(diff_thr) / 255.0)  # 0..1

        t = frame_idx / fps
        energies.append(motion)
        times.append(t)

        prev_gray = cur_gray
        frame_idx += 1
        sample_idx += 1

    cap.release()

    if len(energies) < 5:
        return ([], float(fps))

    # Normalize energies robustly
    e = np.array(energies, dtype=np.float32)
    # robust scale using percentiles
    p10, p90 = np.percentile(e, 10), np.percentile(e, 90)
    denom = float(max(1e-6, p90 - p10))
    e_norm = np.clip((e - p10) / denom, 0.0, 1.0)

    # Smooth
    win = max(1, int(round(smooth_window_sec * effective_sample_fps)))
    kernel = np.ones(win, dtype=np.float32) / float(win)
    e_smooth = np.convolve(e_norm, kernel, mode="same")

    # Activity mask
    active = e_smooth >= float(active_threshold)

    # Find segments from mask
    segments_idx: List[Tuple[int, int]] = []
    in_seg = False
    s = 0
    for i, a in enumerate(active):
        if a and not in_seg:
            in_seg = True
            s = i
        elif (not a) and in_seg:
            in_seg = False
            segments_idx.append((s, i - 1))
    if in_seg:
        segments_idx.append((s, len(active) - 1))

    # Convert to time segments
    raw_segments: List[Tuple[float, float, float]] = []
    for a, b in segments_idx:
        t0 = float(times[a])
        t1 = float(times[b])
        if t1 - t0 < min_segment_sec:
            continue
        strength = float(np.mean(e_smooth[a:b+1]))
        raw_segments.append((t0, t1, strength))

    # Merge close segments
    merged: List[Tuple[float, float, float]] = []
    for t0, t1, strength in raw_segments:
        if not merged:
            merged.append((t0, t1, strength))
            continue
        mt0, mt1, ms = merged[-1]
        if t0 - mt1 <= merge_gap_sec:
            # merge
            merged[-1] = (mt0, max(mt1, t1), max(ms, strength))
        else:
            merged.append((t0, t1, strength))

    # Build RallySegment with confidence/flags
    out: List[RallySegment] = []
    for idx, (t0, t1, strength) in enumerate(merged):
        flags: List[str] = []
        conf = _clamp01(0.35 + 0.65 * strength)

        dur = t1 - t0
        if dur < 1.2:
            flags.append("SEGMENT_SHORT")
            conf = min(conf, 0.65)
        if dur > 25.0:
            flags.append("SEGMENT_LONG")
            conf = min(conf, 0.75)

        out.append(RallySegment(t_start=t0, t_end=t1, confidence=conf, flags=flags))

    if debug:
        # Simple debug prints
        print(f"[seg] fps={fps:.2f}, effective_sample_fps={effective_sample_fps:.2f}, points={len(out)}")

    return (out, float(fps))