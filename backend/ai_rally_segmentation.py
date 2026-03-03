# backend/ai_rally_segmentation.py
import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass(frozen=True)
class RallySegment:
    t_start: float
    t_end: float
    confidence: float
    flags: List[str]

def detect_rally_segments_advanced_gpu(
    energies: List[float],
    timestamps: List[float],
    effective_fps: float,
    *,
    high_thresh: float = 0.25,
    low_thresh: float = 0.12,
    max_gap_sec: float = 2.0,
    min_dur_sec: float = 1.0
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
    rallies = []
    active = False
    s_time, l_time = 0.0, 0.0
    
    for i, val in enumerate(e_norm):
        curr_t = timestamps[i]
        if not active:
            if val > high_thresh:
                active = True
                s_time = curr_t
                l_time = curr_t
        else:
            if val > low_thresh:
                l_time = curr_t
            
            # Closing trigger: silence duration exceeds max_gap_sec
            if curr_t - l_time > max_gap_sec:
                duration = l_time - s_time
                if duration > min_dur_sec:
                    rallies.append(RallySegment(
                        t_start=s_time, 
                        t_end=l_time, 
                        confidence=float(np.clip(val, 0.5, 1.0)),
                        flags=[]
                    ))
                active = False

    # Handle final rally if active at the end of stream
    if active:
        duration = l_time - s_time
        if duration > min_dur_sec:
            rallies.append(RallySegment(
                t_start=s_time, 
                t_end=l_time, 
                confidence=0.8, 
                flags=[]
            ))

    return rallies