# backend/ai_multi_stream_engine.py
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Deque, Tuple
from collections import deque

class MatchState(Enum):
    IDLE = 0
    PREPARE = 1
    RALLY = 2

@dataclass(frozen=True)
class FrameSignal:
    """Strictly typed container for fused signals from all streams."""
    timestamp: float
    table_energy: float
    p1_wrist_vel: float
    p2_wrist_vel: float
    p1_near_table: bool
    p2_near_table: bool
    p1_is_low: bool
    p2_is_low: bool
    p1_ratio: float  # Added for debug visualization consistency
    p2_ratio: float  # Added for debug visualization consistency

class MultiStreamStateMachine:
    """
    Main state controller. Uses pose-aware signals to identify match states.
    Strictly maintains smoothing buffers and internal monitoring variables.
    """
    def __init__(self):
        self.state = MatchState.IDLE
        # Smoothing buffers for jitter reduction
        self.p1_vel_buf: Deque[float] = deque(maxlen=5)
        self.p2_vel_buf: Deque[float] = deque(maxlen=5)
        
        # --- CALIBRATED THRESHOLDS ---
        self.RALLY_START_THRESHOLD = 0.18
        self.RALLY_END_TIMEOUT = 1.0
        self.STANCE_CLOSE_DELAY = 0.3    # Grace period before stance-based closing
        
        # State Tracking
        self.last_active_time = 0.0
        self.current_rally_start: Optional[float] = None
        
        # Monitoring variables for Debug Overlay
        self.last_fused_activity = 0.0
        self.last_p1_smoothed_vel = 0.0
        self.last_p2_smoothed_vel = 0.0

    def _get_smoothed_val(self, val: float, buf: Deque[float]) -> float:
        buf.append(val)
        return sum(buf) / len(buf)

    def update(self, signal: FrameSignal) -> Optional[Tuple[float, float]]:
        """Updates match state and returns rally duration if finalized."""
        if not isinstance(signal, FrameSignal):
            raise TypeError("Input must be a FrameSignal instance")

        # 1. Smoothing
        s_p1 = self._get_smoothed_val(signal.p1_wrist_vel, self.p1_vel_buf)
        s_p2 = self._get_smoothed_val(signal.p2_wrist_vel, self.p2_vel_buf)
        self.last_p1_smoothed_vel, self.last_p2_smoothed_vel = s_p1, s_p2

        # 2. Activity Fusion
        fused_activity = max(s_p1, s_p2) + (signal.table_energy * 0.1)
        self.last_fused_activity = fused_activity

        # Logical Check: If both players are standing straight (IDLE cue)
        both_standing = not (signal.p1_is_low or signal.p2_is_low)

        # 3. Transitions
        if self.state == MatchState.IDLE:
            if signal.p1_near_table and signal.p2_near_table and (signal.p1_is_low or signal.p2_is_low):
                self.state = MatchState.PREPARE
                
        elif self.state == MatchState.PREPARE:
            if fused_activity > self.RALLY_START_THRESHOLD:
                self.state = MatchState.RALLY
                self.current_rally_start = signal.timestamp
                self.last_active_time = signal.timestamp
            elif both_standing or not (signal.p1_near_table or signal.p2_near_table):
                self.state = MatchState.IDLE

        elif self.state == MatchState.RALLY:
            # Sustain rally on motion or action-stance
            if fused_activity > (self.RALLY_START_THRESHOLD * 0.4) or not both_standing:
                self.last_active_time = signal.timestamp
            
            # Close triggers
            timeout_reached = (signal.timestamp - self.last_active_time > self.RALLY_END_TIMEOUT)
            standing_confirmed = both_standing and (signal.timestamp - self.last_active_time > self.STANCE_CLOSE_DELAY)
            
            if timeout_reached or standing_confirmed:
                t_start = self.current_rally_start
                t_end = signal.timestamp if standing_confirmed else self.last_active_time
                
                self.state = MatchState.IDLE
                self.current_rally_start = None
                
                if t_start is not None:
                    return (t_start, t_end)
        
        return None