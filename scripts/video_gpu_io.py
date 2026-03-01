from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Tuple, Union

import numpy as np
import torch


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float


def _run_capture(args: List[str]) -> str:
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed ({r.returncode}): {' '.join(args)}\n{r.stderr.strip()}")
    return r.stdout.strip()


def probe_video_ffprobe(video_path: Union[str, Path], ffprobe: str = "ffprobe") -> VideoInfo:
    """
    Auto-detect width/height/fps using ffprobe.
    """
    video_path = str(video_path)
    out = _run_capture([
        ffprobe,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-of", "json",
        video_path
    ])
    data = json.loads(out)
    stream = data["streams"][0]
    w = int(stream["width"])
    h = int(stream["height"])

    rfr = stream.get("r_frame_rate", "0/1")
    num, den = rfr.split("/")
    fps = float(num) / float(den) if float(den) != 0 else 0.0

    return VideoInfo(width=w, height=h, fps=fps)


def nvdec_bgr24_stream(
    video_path: Union[str, Path],
    width: int,
    height: int,
    ffmpeg: str = "ffmpeg",
    loglevel: str = "error",
) -> Iterator[np.ndarray]:
    """
    Decode using FFmpeg NVDEC and output raw BGR frames (H,W,3) uint8.
    """
    video_path = str(video_path)
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", loglevel,
        "-hwaccel", "cuda",
        "-i", video_path,
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8)
    if proc.stdout is None:
        raise RuntimeError("Failed to open ffmpeg stdout pipe")

    frame_size = width * height * 3
    try:
        while True:
            raw = proc.stdout.read(frame_size)
            if len(raw) != frame_size:
                break
            yield np.frombuffer(raw, np.uint8).reshape((height, width, 3))
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        proc.wait()


@dataclass
class ROI:
    kind: str  # "bbox" or "poly"
    bbox: Tuple[int, int, int, int] | None = None
    poly: List[Tuple[int, int]] | None = None
    meta: Dict | None = None


def load_roi_json(path: Union[str, Path]) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_roi(roi_obj: Dict) -> ROI:
    """
    Flexible ROI parser:
      - {"x1","y1","x2","y2"}
      - {"bbox":[x1,y1,x2,y2]} or {"bbox":{"x1"...}}
      - {"points":[[x,y],...]} or {"polygon":[...]}
    """
    meta = dict(roi_obj)

    if all(k in roi_obj for k in ("x1", "y1", "x2", "y2")):
        return ROI(kind="bbox", bbox=(int(roi_obj["x1"]), int(roi_obj["y1"]), int(roi_obj["x2"]), int(roi_obj["y2"])), meta=meta)

    if "bbox" in roi_obj:
        bb = roi_obj["bbox"]
        if isinstance(bb, list) and len(bb) == 4:
            return ROI(kind="bbox", bbox=(int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])), meta=meta)
        if isinstance(bb, dict) and all(k in bb for k in ("x1", "y1", "x2", "y2")):
            return ROI(kind="bbox", bbox=(int(bb["x1"]), int(bb["y1"]), int(bb["x2"]), int(bb["y2"])), meta=meta)

    pts = None
    if "points" in roi_obj:
        pts = roi_obj["points"]
    elif "polygon" in roi_obj:
        pts = roi_obj["polygon"]

    if isinstance(pts, list) and len(pts) >= 3:
        poly = [(int(p[0]), int(p[1])) for p in pts]
        return ROI(kind="poly", poly=poly, meta=meta)

    raise ValueError("Unsupported ROI format")


def scale_roi(roi: ROI, src_w: int, src_h: int, dst_w: int, dst_h: int) -> ROI:
    sx = dst_w / float(src_w)
    sy = dst_h / float(src_h)

    if roi.kind == "bbox" and roi.bbox:
        x1, y1, x2, y2 = roi.bbox
        nb = (int(round(x1 * sx)), int(round(y1 * sy)), int(round(x2 * sx)), int(round(y2 * sy)))
        return ROI(kind="bbox", bbox=nb, meta=roi.meta)

    if roi.kind == "poly" and roi.poly:
        npoly = [(int(round(x * sx)), int(round(y * sy))) for x, y in roi.poly]
        return ROI(kind="poly", poly=npoly, meta=roi.meta)

    raise ValueError("ROI missing coords")


def auto_skip_for_target_fps(src_fps: float, target_fps: float = 10.0) -> int:
    if src_fps <= 0:
        return 1
    return max(1, int(math.floor(src_fps / target_fps)))


def crop_roi_gpu(frame_bgr_uint8_gpu: torch.Tensor, roi: ROI) -> torch.Tensor:
    """
    frame: HWC uint8 on GPU
    returns ROI as HWC uint8 on GPU
    """
    if roi.kind == "bbox" and roi.bbox:
        x1, y1, x2, y2 = roi.bbox
        return frame_bgr_uint8_gpu[y1:y2, x1:x2]

    if roi.kind == "poly" and roi.poly:
        xs = [p[0] for p in roi.poly]
        ys = [p[1] for p in roi.poly]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        return frame_bgr_uint8_gpu[y1:y2, x1:x2]

    raise ValueError("Invalid ROI")


def to_chw_float01(roi_hwc_uint8_gpu: torch.Tensor) -> torch.Tensor:
    return roi_hwc_uint8_gpu.permute(2, 0, 1).contiguous().float().div_(255.0)