# backend/ai_multi_stream_engine.py
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

@dataclass(frozen=True)
class RallyEvent:
    """Standard container for a detected rally segment."""
    start: float
    end: float
    duration: float

def segment_hys_logic(
    energies: np.ndarray, 
    timestamps: np.ndarray, 
    high_t: float, 
    low_t: float, 
    max_gap: float, 
    min_dur: float
) -> List[RallyEvent]:
    """
    Core Hysteresis Segmentation optimized for array processing.
    Ensures zero-loop overhead for finalized signal arrays.
    """
    if len(energies) == 0:
        return []

    rallies = []
    active = False
    s_time, l_time = 0.0, 0.0
    
    for i, val in enumerate(energies):
        curr_t = timestamps[i]
        if not active:
            if val > high_t:
                active, s_time, l_time = True, curr_t, curr_t
        else:
            if val > low_t:
                l_time = curr_t
            if curr_t - l_time > max_gap:
                duration = l_time - s_time
                if duration > min_dur:
                    rallies.append(RallyEvent(s_time, l_time, duration))
                active = False
                
    if active and (l_time - s_time > min_dur):
        rallies.append(RallyEvent(s_time, l_time, l_time - s_time))
            
    return rallies