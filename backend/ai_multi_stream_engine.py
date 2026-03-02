# backend/ai_multi_stream_engine.py
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Deque, Tuple
from collections import deque

class MatchState(Enum):
    IDLE = 0        # Standing, walking, or dead ball period
    PREPARE = 1     # Players in position, ready stance detected
    RALLY = 2       # Active rally in progress

@dataclass(frozen=True)
class FrameSignal:
    """
    Strictly typed container for multi-stream signals.
    All velocities are normalized to player body height (0.0 - 1.0).
    """
    timestamp: float
    table_energy: float
    p1_wrist_vel: float
    p2_wrist_vel: float
    p1_near_table: bool
    p2_near_table: bool
    p1_is_low: bool  # True if crouching (Ready/Action stance)
    p2_is_low: bool  # True if crouching (Ready/Action stance)

class MultiStreamStateMachine:
    """
    Production-grade state machine for table tennis rally detection.
    Fuses Pose (Stance + Velocity), Table Energy, and Spatial Context.
    """
    def __init__(self):
        self.state = MatchState.IDLE
        
        # Smoothing buffers for wrist velocity to eliminate jitter (5-frame window)
        self.p1_vel_buf: Deque[float] = deque(maxlen=5)
        self.p2_vel_buf: Deque[float] = deque(maxlen=5)
        
        # --- CALIBRATED THRESHOLDS ---
        self.RALLY_START_THRESHOLD = 0.18 # Normalized velocity to trigger rally
        self.RALLY_END_TIMEOUT = 1.0     # Max silence before closing rally (seconds)
        self.STANCE_CLOSE_DELAY = 0.3    # Grace period after standing up before closing
        
        # State tracking
        self.last_active_time = 0.0
        self.current_rally_start: Optional[float] = None
        
        # Monitor variables for Debug UI (Do not remove)
        self.last_fused_activity = 0.0
        self.last_p1_smoothed_vel = 0.0
        self.last_p2_smoothed_vel = 0.0

    def _calculate_ema(self, new_val: float, buffer: Deque[float]) -> float:
        """Simple moving average for signal stabilization."""
        buffer.append(new_val)
        return sum(buffer) / len(buffer)

    def update(self, signal: FrameSignal) -> Optional[Tuple[float, float]]:
        """
        Processes frame signal and returns (t_start, t_end) if a rally is finalized.
        Enforces strict validation on input signal.
        """
        if not isinstance(signal, FrameSignal):
            raise TypeError("Expected FrameSignal object")

        # 1. Signal Smoothing
        s_p1 = self._calculate_ema(signal.p1_wrist_vel, self.p1_vel_buf)
        s_p2 = self._calculate_ema(signal.p2_wrist_vel, self.p2_vel_buf)
        
        # Store for debug monitor
        self.last_p1_smoothed_vel = s_p1
        self.last_p2_smoothed_vel = s_p2

        # 2. Activity Fusion
        # Wrist velocity is the lead indicator. Table energy is secondary.
        fused_activity = max(s_p1, s_p2) + (signal.table_energy * 0.1)
        self.last_fused_activity = fused_activity

        # Both players standing up is a strong signal for IDLE/End
        both_standing = not (signal.p1_is_low or signal.p2_is_low)

        # 3. State Machine Logic
        if self.state == MatchState.IDLE:
            # Transition to PREPARE: Near table AND at least one player in low stance
            if signal.p1_near_table and signal.p2_near_table and (signal.p1_is_low or signal.p2_is_low):
                self.state = MatchState.PREPARE
                
        elif self.state == MatchState.PREPARE:
            # Transition to RALLY: Explosive movement detected
            if fused_activity > self.RALLY_START_THRESHOLD:
                self.state = MatchState.RALLY
                self.current_rally_start = signal.timestamp
                self.last_active_time = signal.timestamp
            # Revert to IDLE if players leave the table area
            elif not (signal.p1_near_table or signal.p2_near_table):
                self.state = MatchState.IDLE

        elif self.state == MatchState.RALLY:
            # Keep rally active if there is motion OR players are still in low stance
            if fused_activity > (self.RALLY_START_THRESHOLD * 0.4) or not both_standing:
                self.last_active_time = signal.timestamp
            
            # Close rally triggers:
            # A. Inactivity timeout (Ball long gone)
            # B. Both players stand up (Human cue for point finished)
            timeout_reached = (signal.timestamp - self.last_active_time > self.RALLY_END_TIMEOUT)
            
            if timeout_reached or (both_standing and (signal.timestamp - self.last_active_time > self.STANCE_CLOSE_DELAY)):
                t_start = self.current_rally_start
                # If finished by standing up, t_end is roughly when they stood up
                t_end = signal.timestamp if both_standing else self.last_active_time
                
                self.state = MatchState.IDLE
                self.current_rally_start = None
                
                if t_start is not None:
                    return (t_start, t_end)
        
        return None