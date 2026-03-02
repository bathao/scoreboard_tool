# backend/ai_rally_segmentation.py
from __future__ import annotations
import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass(frozen=True)
class RallySegment:
    t_start: float
    t_end: float
    confidence: float
    flags: List[str]

def detect_rally_segments_gpu(
    energies: List[float],
    timestamps: List[float],
    effective_fps: float,
    *,
    smooth_window_sec: float = 0.35,
    active_threshold: float = 0.22,
    min_segment_sec: float = 1.0,
    merge_gap_sec: float = 0.8
) -> List[RallySegment]:
    """
    Process pre-calculated motion energies (from GPU) into Rally segments.
    Logic is now separated from video IO for maximum speed.
    """
    if not energies:
        return []

    e = np.array(energies, dtype=np.float32)
    # Robust normalization
    p10, p90 = np.percentile(e, 10), np.percentile(e, 90)
    denom = float(max(1e-6, p90 - p10))
    e_norm = np.clip((e - p10) / denom, 0.0, 1.0)

    # Smoothing using NumPy (fast for 1D arrays)
    win = max(1, int(round(smooth_window_sec * effective_fps)))
    kernel = np.ones(win, dtype=np.float32) / float(win)
    e_smooth = np.convolve(e_norm, kernel, mode="same")

    # Masking activity
    active = e_smooth >= float(active_threshold)

    # Find raw segments
    segments_idx: List[Tuple[int, int]] = []
    in_seg, s = False, 0
    for i, a in enumerate(active):
        if a and not in_seg:
            in_seg, s = True, i
        elif (not a) and in_seg:
            in_seg = False
            segments_idx.append((s, i - 1))
    if in_seg:
        segments_idx.append((s, len(active) - 1))

    # Merge and Filter
    raw_segments: List[Tuple[float, float, float]] = []
    for a, b in segments_idx:
        t0, t1 = timestamps[a], timestamps[b]
        if t1 - t0 < min_segment_sec:
            continue
        strength = float(np.mean(e_smooth[a:b+1]))
        raw_segments.append((t0, t1, strength))

    # Merging logic
    merged: List[Tuple[float, float, float]] = []
    for t0, t1, strength in raw_segments:
        if not merged:
            merged.append((t0, t1, strength))
            continue
        mt0, mt1, ms = merged[-1]
        if t0 - mt1 <= merge_gap_sec:
            merged[-1] = (mt0, max(mt1, t1), max(ms, strength))
        else:
            merged.append((t0, t1, strength))

    # Convert to Final Objects
    out = []
    for t0, t1, strength in merged:
        conf = float(np.clip(0.35 + 0.65 * strength, 0.0, 1.0))
        out.append(RallySegment(t_start=t0, t_end=t1, confidence=conf, flags=[]))
    
    return out