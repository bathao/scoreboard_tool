# backend/ai_table_roi_dl.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from backend.ai_table_roi import TableROI, detect_table_roi_classical


@dataclass(frozen=True)
class DLConfig:
    """
    Local-only DL config.
    You provide weights path manually (download/finetune later).
    """
    weights_path: str
    conf_thres: float = 0.25
    iou_thres: float = 0.45
    device: str = "cuda"  # or "cpu"
    class_name: str = "table"  # expected class name in model


def _try_import_ultralytics():
    try:
        from ultralytics import YOLO  # type: ignore
        return YOLO
    except Exception:
        return None


def detect_table_roi_dl(
    video_path: str,
    *,
    cfg: DLConfig,
    sample_time_sec: float = 1.0,
    max_frames: int = 3,
    debug: bool = False,
) -> TableROI:
    """
    Detect table ROI using local DL model (Ultralytics YOLO).
    Falls back to classical if ultralytics isn't available or detection fails.

    IMPORTANT:
    - This does NOT download weights.
    - You must place weights locally and pass weights_path.
    """
    YOLO = _try_import_ultralytics()
    if YOLO is None:
        if debug:
            print("[roi-dl] ultralytics not installed -> fallback classical")
        roi = detect_table_roi_classical(video_path, sample_time_sec=sample_time_sec, max_frames=5, debug=debug)
        return TableROI(**roi.to_dict(), method="classical_fallback", notes=f"dl_unavailable; {roi.notes}")  # type: ignore

    # Load model
    try:
        model = YOLO(cfg.weights_path)
    except Exception as e:
        if debug:
            print(f"[roi-dl] failed load weights '{cfg.weights_path}' -> fallback classical: {e}")
        roi = detect_table_roi_classical(video_path, sample_time_sec=sample_time_sec, max_frames=5, debug=debug)
        return TableROI(**roi.to_dict(), method="classical_fallback", notes=f"dl_load_failed; {roi.notes}")  # type: ignore

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 60.0

    frame_idx0 = int(round(sample_time_sec * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx0)

    best_roi: Optional[TableROI] = None

    for i in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Inference
        try:
            results = model.predict(
                frame,
                conf=cfg.conf_thres,
                iou=cfg.iou_thres,
                device=cfg.device,
                verbose=False,
            )
        except Exception as e:
            if debug:
                print(f"[roi-dl] predict error -> fallback classical: {e}")
            break

        if not results:
            continue

        r0 = results[0]
        # ultralytics results: boxes.xyxy, boxes.conf, boxes.cls
        if r0.boxes is None or len(r0.boxes) == 0:
            continue

        boxes = r0.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        # Choose highest confidence box
        j = int(np.argmax(confs))
        x1, y1, x2, y2 = xyxy[j].tolist()
        c = float(confs[j])

        x1 = int(max(0, round(x1)))
        y1 = int(max(0, round(y1)))
        x2 = int(max(0, round(x2)))
        y2 = int(max(0, round(y2)))
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)

        roi = TableROI(
            x=x1,
            y=y1,
            w=w,
            h=h,
            confidence=float(max(0.0, min(1.0, c))),
            method="dl_yolo",
            notes="ok",
        )

        if debug:
            print(f"[roi-dl] frame {i} conf={roi.confidence:.3f} bbox=({roi.x},{roi.y},{roi.w},{roi.h})")

        if best_roi is None or roi.confidence > best_roi.confidence:
            best_roi = roi

    cap.release()

    if best_roi is None or best_roi.confidence < 0.30:
        roi = detect_table_roi_classical(video_path, sample_time_sec=sample_time_sec, max_frames=5, debug=debug)
        return TableROI(**roi.to_dict(), method="classical_fallback", notes=f"dl_no_detection; {roi.notes}")  # type: ignore

    return best_roi