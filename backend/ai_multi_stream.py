# backend/ai_multi_stream_engine.py
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Deque, Tuple
from collections import deque
import numpy as np

@dataclass(frozen=True)
class FrameSignal:
    timestamp: float
    table_energy: float
    p1_vel: float # Normalized wrist + body velocity
    p2_vel: float
    p1_near: bool
    p2_near: bool

class MultiLayerSegmenter:
    """
    Handles two modes of segmentation: 
    1. Table Only (Layer 1)
    2. Fused Table + Players (Layer 1 + 2)
    """
    def __init__(self):
        # Constants for Layer 1 (Table Only)
        self.T1_HIGH = 0.25
        self.T1_LOW = 0.12
        
        # Constants for Layer 2 (Fusion)
        # We use a higher threshold because players move more pixels
        self.FUSE_HIGH = 0.35 
        self.FUSE_LOW = 0.15
        
        self.MAX_GAP = 2.0
        self.MIN_DUR = 1.0

        # State tracking for both modes
        self.t1_active = False
        self.t1_start, self.t1_last = 0.0, 0.0
        
        self.fuse_active = False
        self.fuse_start, self.fuse_last = 0.0, 0.0

    def process(self, signal: FrameSignal) -> Tuple[Optional[tuple], Optional[tuple]]:
        """
        Returns (table_only_event, fused_event)
        Each event is (start, end)
        """
        t1_event, fuse_event = None, None
        t = signal.timestamp

        # --- MODE 1: TABLE ONLY ---
        val1 = signal.table_energy
        if not self.t1_active:
            if val1 > self.T1_HIGH:
                self.t1_active, self.t1_start, self.t1_last = True, t, t
        else:
            if val1 > self.T1_LOW: self.t1_last = t
            if t - self.t1_last > self.MAX_GAP:
                if self.t1_last - self.t1_start > self.MIN_DUR:
                    t1_event = (self.t1_start, self.t1_last)
                self.t1_active = False

        # --- MODE 2: FUSED (TABLE + PLAYERS) ---
        # Players only count if they are near the table
        p1_act = signal.p1_vel if signal.p1_near else 0
        p2_act = signal.p2_vel if signal.p2_near else 0
        val2 = max(val1, p1_act, p2_act)

        if not self.fuse_active:
            if val2 > self.FUSE_HIGH:
                self.fuse_active, self.fuse_start, self.fuse_last = True, t, t
        else:
            if val2 > self.FUSE_LOW: self.fuse_last = t
            if t - self.fuse_last > self.MAX_GAP:
                if self.fuse_last - self.fuse_start > self.MIN_DUR:
                    fuse_event = (self.fuse_start, self.fuse_last)
                self.fuse_active = False

        return t1_event, fuse_event