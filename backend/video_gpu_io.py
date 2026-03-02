import json
import subprocess
import numpy as np
from pathlib import Path
from typing import Iterator, Union, Optional, Tuple, List, Dict
from dataclasses import dataclass

@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float

def probe_video_ffprobe(video_path: Union[str, Path], ffprobe: str = "ffprobe") -> VideoInfo:
    video_path = str(video_path)
    cmd = [
        ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-of", "json", video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    
    data = json.loads(result.stdout)
    stream = data["streams"][0]
    w = int(stream["width"])
    h = int(stream["height"])
    
    rfr = stream.get("r_frame_rate", "30/1")
    num, den = rfr.split("/")
    fps = float(num) / float(den) if float(den) != 0 else 30.0
    return VideoInfo(width=w, height=h, fps=fps)

def nvdec_bgr24_stream(
    video_path: str,
    width: int,
    height: int,
    crop_roi: Optional[Tuple[int, int, int, int]] = None,
    ffmpeg: str = "ffmpeg"
) -> Iterator[np.ndarray]:
    """
    Decode video using GPU (NVDEC) and yield BGR frames.
    """
    video_path = str(video_path)
    
    # Target dimensions
    target_w, target_h = width, height
    filter_args = []
    
    if crop_roi:
        x, y, w, h = crop_roi
        # Standard crop filter (very fast)
        filter_args = ["-vf", f"crop={w}:{h}:{x}:{y}"]
        target_w, target_h = w, h

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "error",
        "-hwaccel", "cuda",
        "-i", video_path,
        *filter_args,
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "pipe:1"
    ]
    
    # bufsize=10**8 (100MB) is safe and sufficient for 2.7K 60fps
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8)
    
    frame_size = target_w * target_h * 3
    try:
        while True:
            raw = proc.stdout.read(frame_size)
            if len(raw) != frame_size:
                break
            # Copy to memory to ensure it's writable for PyTorch
            yield np.frombuffer(raw, dtype=np.uint8).copy().reshape((target_h, target_w, 3))
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait()