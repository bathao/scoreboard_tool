# backend/ai_rally_segmentation.py
import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class RallySegment:
    t_start: float
    t_end: float
    confidence: float
    flags: List[str]


def _merge_contiguous_artifact_runs(
    segments: List[RallySegment],
    *,
    contiguous_eps_sec: float = 0.05,
    artifact_min_dur_sec: float = 2.8,
) -> List[RallySegment]:
    if len(segments) <= 1:
        return segments

    out: List[RallySegment] = []
    i = 0
    while i < len(segments):
        run = [segments[i]]
        j = i
        while j + 1 < len(segments):
            gap = segments[j + 1].t_start - segments[j].t_end
            if gap <= contiguous_eps_sec:
                run.append(segments[j + 1])
                j += 1
                continue
            break

        if len(run) == 1:
            out.append(run[0])
        else:
            min_dur = min(r.t_end - r.t_start for r in run)
            if min_dur < artifact_min_dur_sec:
                first_dur = run[0].t_end - run[0].t_start
                # Common artifact in long rallies: tail gets over-split into tiny pieces.
                # Keep the first long chunk, merge the noisy tail only.
                if len(run) >= 4 and first_dur >= 8.0:
                    out.append(run[0])
                    tail = run[1:]
                    merged_tail = RallySegment(
                        t_start=tail[0].t_start,
                        t_end=tail[-1].t_end,
                        confidence=float(np.clip(np.median([r.confidence for r in tail]), 0.5, 1.0)),
                        flags=sorted(set([f for r in tail for f in r.flags] + ["merged_contiguous_tail_artifact"])),
                    )
                    out.append(merged_tail)
                else:
                    merged = RallySegment(
                        t_start=run[0].t_start,
                        t_end=run[-1].t_end,
                        confidence=float(np.clip(np.median([r.confidence for r in run]), 0.5, 1.0)),
                        flags=sorted(set([f for r in run for f in r.flags] + ["merged_contiguous_artifact"])),
                    )
                    out.append(merged)
            else:
                out.extend(run)

        i = j + 1

    return out


def _split_long_segment_on_dips(
    s_idx: int,
    e_idx: int,
    timestamps: List[float],
    e_norm: np.ndarray,
    *,
    low_thresh: float,
    min_dur_sec: float,
    long_segment_sec: float,
    split_gap_sec: float,
    split_low_factor: float,
    min_split_dur_sec: float,
) -> List[tuple[int, int, List[str]]]:
    seg_dur = timestamps[e_idx] - timestamps[s_idx]
    if seg_dur <= long_segment_sec:
        return [(s_idx, e_idx, [])]

    split_low = max(0.02, low_thresh * split_low_factor)

    cuts: List[int] = []
    run_start = None
    for i in range(s_idx, e_idx + 1):
        if float(e_norm[i]) < split_low:
            if run_start is None:
                run_start = i
            continue

        if run_start is not None:
            run_dur = timestamps[i - 1] - timestamps[run_start]
            if run_dur >= split_gap_sec:
                cuts.append((run_start + (i - 1)) // 2)
            run_start = None

    if run_start is not None:
        run_dur = timestamps[e_idx] - timestamps[run_start]
        if run_dur >= split_gap_sec:
            cuts.append((run_start + e_idx) // 2)

    if not cuts:
        return [(s_idx, e_idx, ["long_unsplit"])]

    boundaries = [s_idx] + sorted(set(cuts)) + [e_idx]
    raw_segments: List[list] = []
    for i in range(len(boundaries) - 1):
        a = boundaries[i]
        b = boundaries[i + 1]
        if b <= a:
            continue
        raw_segments.append([a, b, ["split_long"]])

    if not raw_segments:
        return [(s_idx, e_idx, ["long_unsplit"])]

    # Merge very short split pieces back to neighbors to avoid over-segmentation noise.
    # This is important for long rallies with brief energy dips.
    i = 0
    while i < len(raw_segments):
        a, b, flags = raw_segments[i]
        dur = timestamps[b] - timestamps[a]
        if dur >= min_split_dur_sec:
            i += 1
            continue

        if i > 0:
            raw_segments[i - 1][1] = b
            raw_segments[i - 1][2].append("merged_short_split")
            raw_segments.pop(i)
            continue

        if i + 1 < len(raw_segments):
            raw_segments[i + 1][0] = a
            raw_segments[i + 1][2].append("merged_short_split")
            raw_segments.pop(i)
            continue

        # Single tiny segment only -> fallback to unsplit.
        return [(s_idx, e_idx, ["long_unsplit"])]

    split_segments: List[tuple[int, int, List[str]]] = []
    for a, b, flags in raw_segments:
        if timestamps[b] - timestamps[a] >= min_dur_sec:
            split_segments.append((a, b, flags))

    return split_segments if split_segments else [(s_idx, e_idx, ["long_unsplit"])]

def detect_rally_segments_advanced_gpu(
    energies: List[float],
    timestamps: List[float],
    effective_fps: float,
    *,
    high_thresh: float = 0.25,
    low_thresh: float = 0.12,
    max_gap_sec: float = 2.0,
    min_dur_sec: float = 1.0,
    long_segment_sec: float = 12.0,
    split_gap_sec: float = 0.55,
    split_low_factor: float = 0.65,
    min_split_dur_sec: float = 2.0,
) -> List[RallySegment]:
    """
    Advanced Hysteresis Segmentation using GPU-based signal refinement.
    Strictly processed on CUDA for high-performance table tennis dynamics.
    """
    if not energies:
        return []

    # 1. STRICT HARDWARE CHECK
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise RuntimeError("CRITICAL: CUDA GPU required for Advanced Segmentation.")

    # 2. SIGNAL SMOOTHING (GPU Convolution)
    # FIX: Added dtype=torch.float32 to prevent Double vs Float mismatch
    signal_tensor = torch.tensor(energies, device=device, dtype=torch.float32).view(1, 1, -1)
    
    k_size, sigma = 11, 3.0
    gx = torch.arange(k_size, device=device, dtype=torch.float32) - (k_size - 1) / 2
    kernel = (torch.exp(-gx.pow(2) / (2 * sigma**2))).view(1, 1, -1)
    kernel /= kernel.sum()
    
    # Apply 1D Gaussian Smoothing on GPU
    smoothed_signal = F.conv1d(signal_tensor, kernel, padding=k_size//2).squeeze()
    
    # 3. NORMALIZATION (Robust Percentile Scaling)
    e_np = smoothed_signal.cpu().numpy()
    p10, p95 = np.percentile(e_np, 10), np.percentile(e_np, 95)
    # Prevent division by zero with 1e-6
    e_norm = np.clip((e_np - p10) / (p95 - p10 + 1e-6), 0.0, 1.0)
    
    # 4. HYSTERESIS SEGMENTATION
    rallies: List[RallySegment] = []
    active = False
    s_idx, l_idx = 0, 0

    def append_segment(start_idx: int, end_idx: int) -> None:
        if end_idx <= start_idx:
            return
        if timestamps[end_idx] - timestamps[start_idx] <= min_dur_sec:
            return

        for a_idx, b_idx, seg_flags in _split_long_segment_on_dips(
            start_idx,
            end_idx,
            timestamps,
            e_norm,
            low_thresh=low_thresh,
            min_dur_sec=min_dur_sec,
            long_segment_sec=long_segment_sec,
            split_gap_sec=split_gap_sec,
            split_low_factor=split_low_factor,
            min_split_dur_sec=min_split_dur_sec,
        ):
            seg_conf = float(np.clip(np.median(e_norm[a_idx:b_idx + 1]), 0.5, 1.0))
            rallies.append(
                RallySegment(
                    t_start=float(timestamps[a_idx]),
                    t_end=float(timestamps[b_idx]),
                    confidence=seg_conf,
                    flags=seg_flags,
                )
            )
    
    for i, val in enumerate(e_norm):
        curr_t = timestamps[i]
        if not active:
            if val > high_thresh:
                active = True
                s_idx = i
                l_idx = i
        else:
            if val > low_thresh:
                l_idx = i
            
            # Closing trigger: silence duration exceeds max_gap_sec
            if curr_t - timestamps[l_idx] > max_gap_sec:
                append_segment(s_idx, l_idx)
                active = False

    # Handle final rally if active at the end of stream
    if active:
        append_segment(s_idx, l_idx)

    rallies = _merge_contiguous_artifact_runs(rallies)
    return rallies
