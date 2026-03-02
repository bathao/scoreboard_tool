from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple

from backend.ai_table_roi import TableROI

@dataclass(frozen=True)
class DLConfig:
    """
    Configuration for Deep Learning based ROI detection.
    """
    weights_path: str
    conf_thres: float = 0.25
    iou_thres: float = 0.45
    device: str = "cuda"
    class_name: str = "table"

def _try_import_ultralytics():
    """Attempts to import YOLO from ultralytics library."""
    try:
        from ultralytics import YOLO # type: ignore
        return YOLO
    except ImportError:
        return None

def detect_table_roi_dl(
    video_path: str,
    *,
    cfg: Optional[DLConfig] = None,
    sample_time_sec: float = 1.0,
    max_frames: int = 3,
    debug: bool = False
) -> TableROI:
    """
    Strict YOLO-based table detection with Centrality Logic.
    
    If no table is found with sufficient confidence, it RAISES an error 
    instead of falling back to classical methods.
    """
    
    YOLO = _try_import_ultralytics()
    
    # --- PHASE 0: Environment Validation ---
    if YOLO is None:
        raise RuntimeError("CRITICAL: 'ultralytics' library not installed. Cannot run YOLO detection.")
    
    if cfg is None:
        raise ValueError("CRITICAL: DLConfig missing for detect_table_roi_dl.")

    # --- PHASE 1: Model Loading ---
    try:
        model = YOLO(cfg.weights_path)
    except Exception as e:
        raise RuntimeError(f"CRITICAL: Failed to load YOLO weights from {cfg.weights_path}. Error: {e}")

    # --- PHASE 2: Video Setup ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"CRITICAL: Cannot open video file: {video_path}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    
    video_center = np.array([W / 2, H / 2])
    max_possible_dist = np.linalg.norm(video_center)

    # Seek to sample time
    frame_idx0 = int(round(sample_time_sec * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx0)

    best_roi: Optional[TableROI] = None
    highest_selection_score = -1.0

    # --- PHASE 3: Detection & Centrality Scoring ---
    for i in range(max_frames):
        ret, frame = cap.read()
        if not ret: break

        results = model.predict(
            frame, 
            conf=cfg.conf_thres, 
            iou=cfg.iou_thres, 
            device=cfg.device, 
            verbose=False
        )

        if not results or len(results[0].boxes) == 0:
            continue

        boxes = results[0].boxes
        for box in boxes:
            xyxy = box.xyxy.cpu().numpy()[0]
            conf = float(box.conf.cpu().numpy()[0])
            
            x1, y1, x2, y2 = xyxy
            bw, bh = x2 - x1, y2 - y1
            
            # Centrality Logic
            box_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            dist_to_center = np.linalg.norm(box_center - video_center)
            centrality_score = 1.0 - (dist_to_center / max_possible_dist)
            
            # Area Importance
            area_ratio = (bw * bh) / (W * H)
            
            # Heuristic: 60% Centrality, 40% Area. Multiplied by AI Confidence.
            selection_score = ((centrality_score * 0.6) + (area_ratio * 0.4)) * conf

            if debug:
                print(f"[roi-dl] Table Candidate: Score={selection_score:.3f} | Conf={conf:.2f}")

            if selection_score > highest_selection_score:
                highest_selection_score = selection_score
                best_roi = TableROI(
                    x=int(max(0, x1)),
                    y=int(max(0, y1)),
                    w=int(min(bw, W - x1)),
                    h=int(min(bh, H - y1)),
                    confidence=conf,
                    method="dl_yolo_v8x_centralized",
                    notes=f"CentralityScore: {selection_score:.2f}"
                )

    cap.release()

    # --- PHASE 4: Strict Validation ---
    if best_roi is None:
        raise RuntimeError(
            f"CRITICAL: YOLO failed to detect ANY table in '{video_path}'.\n"
            "Check if the table is visible, or adjust 'sample_time_sec'."
        )

    if best_roi.confidence < 0.40: # Strict confidence threshold
        raise RuntimeError(
            f"CRITICAL: Detected table confidence too low ({best_roi.confidence:.2f}).\n"
            "Execution stopped to prevent incorrect rally tracking."
        )

    if debug:
        print(f"[roi-dl] FINAL SELECTION: {best_roi.as_tuple()} (Conf: {best_roi.confidence:.2f})")

    return best_roi