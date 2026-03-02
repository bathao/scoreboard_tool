# backend/ai_multi_stream.py
import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class MultiStreamSignal:
    timestamp: float
    table_energy: float      # Signal from Table ROI
    player_a_activity: float # Signal from Player A Pose/Box
    player_b_activity: float # Signal from Player B Pose/Box
    is_player_in_pos: bool   # Global context check

class MultiStreamAnalyzer:
    """
    Advanced Tầng 5 Layer: Fusion of Table, Player and Global context.
    """
    def __init__(self, device="cuda"):
        self.device = device
        # Load a lightweight pose model (YOLOv8-pose)
        # self.pose_model = YOLO('yolov8n-pose.pt') 

    def analyze_frame_batch(self, 
                           full_frame_lowres: np.ndarray, 
                           table_crop: np.ndarray, 
                           player_crops: List[np.ndarray]) -> MultiStreamSignal:
        """
        Process 3 logical streams in parallel for one timestamp.
        """
        # 1. Table Energy (Pixel difference in Table ROI)
        # 2. Player Activity (Pose changes/Velocity of wrists)
        # 3. Global Context (Are players near the table?)
        
        # This returns a fused signal for the State Machine to decide
        pass

def detect_rally_v2(signals: List[MultiStreamSignal]):
    """
    State Machine Logic:
    - IDLE: Waiting for Serve
    - PREPARE: Player in position + Low movement
    - ACTIVE: High table energy + High player activity
    - END: Ball out of table ROI + Player relaxation pose
    """
    # Logic implementation here
    pass