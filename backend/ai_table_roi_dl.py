from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field, replace
from typing import Optional, List, Tuple

from backend.ai_table_roi import TableROI

@dataclass(frozen=True)
class DLConfig:
    """
    ROI Configuration optimized for Blue Playing Surface.
    Designed to ignore the table legs and under-structure.
    """
    weights_path: str
    conf_thres: float = 0.10
    iou_thres: float = 0.45
    device: str = "cuda"

def _find_blue_surface_box(frame: np.ndarray, search_area: Tuple[int, int, int, int]) -> Optional[Tuple[int, int, int, int]]:
    """
    Focuses specifically on finding the blue rectangular playing surface 
    within a broader area identified by AI.
    """
    x, y, w, h = search_area
    # Expand the search area upwards slightly to catch the surface if YOLO caught the legs
    search_y1 = max(0, y - int(h * 0.8))
    search_y2 = min(frame.shape[0], y + h)
    
    roi_zone = frame[search_y1:search_y2, x:x+w]
    if roi_zone.size == 0: return None

    # Convert to HSV for robust blue detection
    hsv = cv2.cvtColor(roi_zone, cv2.COLOR_BGR2HSV)
    
    # Strict range for the blue table top
    lower_blue = np.array([95, 80, 50])
    upper_blue = np.array([125, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Clean the mask (remove net lines and noise)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None

    # Filter contours by area and shape
    valid_contours = []
    for cnt in contours:
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if bw > (w * 0.4) and (bw / bh) > 1.2:
            valid_contours.append((bx, by, bw, bh))

    if not valid_contours: return None

    # Pick the contour that is most central horizontally in the search zone
    valid_contours.sort(key=lambda c: abs((c[0] + c[2]/2) - (w/2)))
    bx, by, bw, bh = valid_contours[0]

    return x + bx, search_y1 + by, bw, bh

def _try_import_ultralytics():
    try:
        from ultralytics import YOLO
        return YOLO
    except ImportError:
        return None

def detect_table_roi_dl(
    video_path: str,
    *,
    cfg: Optional[DLConfig] = None,
    sample_time_sec: float = 2.0,
    max_frames: int = 5,
    debug: bool = False
) -> TableROI:
    """
    Custom Table ROI Detection:
    1. Uses YOLO to find the general 'Table' vicinity (often catches legs).
    2. Uses _find_blue_surface_box to shift the ROI UP to the blue surface.
    3. Prioritizes the central axis of the video.
    """
    YOLO = _try_import_ultralytics()
    if YOLO is None or cfg is None:
        raise RuntimeError("CRITICAL: Environment/Config Error.")

    model = YOLO(cfg.weights_path)
    cap = cv2.VideoCapture(video_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Sample at different times to get a clear view
    check_times = [2.0, 4.0, 6.0]
    best_candidate = None
    max_score = -1.0

    for ts in check_times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(ts * (cap.get(cv2.CAP_PROP_FPS) or 60.0)))
        ret, frame = cap.read()
        if not ret: break

        results = model.predict(frame, conf=cfg.conf_thres, device=cfg.device, verbose=False)
        if not results or len(results[0].boxes) == 0: continue

        for box in results[0].boxes:
            if int(box.cls[0]) == 0: continue # Skip person
            
            x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
            bw_ai, bh_ai = x2 - x1, y2 - y1

            # Ignore banners (Anything in the top 40% of the screen)
            if y1 < (H * 0.40): continue

            # --- CORRECTION STEP: PULL ROI UP TO BLUE SURFACE ---
            surface = _find_blue_surface_box(frame, (int(x1), int(y1), int(bw_ai), int(bh_ai)))
            
            if surface:
                sx, sy, sw, sh = surface
                
                # Ranking score: Horizontal Centrality (70%) + Confidence (30%)
                dist_center_x = abs((sx + sw/2) - (W/2))
                centrality = 1.0 - (dist_center_x / (W/2))
                score = centrality * 0.7 + float(box.conf[0]) * 0.3

                if score > max_score:
                    max_score = score
                    best_candidate = TableROI(
                        x=sx, y=sy, w=sw, h=sh,
                        confidence=float(box.conf[0]),
                        method="v8x_surface_corrected",
                        notes=f"Shifted from AI-box to Blue-surface"
                    )

    cap.release()

    if best_candidate is None:
        raise RuntimeError("CRITICAL: Failed to locate blue table surface. Check lighting.")

    return best_candidate