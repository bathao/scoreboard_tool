from __future__ import annotations

import os
import subprocess
from pathlib import Path

import cv2

from backend.rally_timeline_contract import RallyTimeline, to_core_rally_events
from backend.timeline import build_match_timeline
from render.renderer import ScoreboardRenderer


def build_match_timeline_from_rally_timeline(timeline: RallyTimeline):
    return build_match_timeline(best_of=timeline.best_of, events=to_core_rally_events(timeline))


def render_scoreboard_video(
    *,
    input_video_path: str,
    timeline: RallyTimeline,
    output_video_path: str,
    player_a_name: str,
    player_b_name: str,
    temp_video_path: str | None = None,
) -> str:
    output_path = Path(output_video_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = Path(temp_video_path) if temp_video_path else output_path.with_name(f"{output_path.stem}__tmp_no_audio.mp4")

    match_timeline = build_match_timeline_from_rally_timeline(timeline)
    renderer = ScoreboardRenderer(
        input_path=str(Path(input_video_path)),
        output_path=str(temp_path),
        timeline=match_timeline,
        player_a_name=player_a_name,
        player_b_name=player_b_name,
    )
    render_to_1080p(renderer)
    try:
        merge_audio(str(temp_path), str(Path(input_video_path)), str(output_path))
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return str(output_path)


def render_to_1080p(renderer: ScoreboardRenderer) -> None:
    cap = cv2.VideoCapture(renderer.input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    target_w, target_h = 1920, 1080

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{target_w}x{target_h}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "h264_nvenc",
        "-preset", "p4",
        "-pix_fmt", "yuv420p",
        renderer.output_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    frame_count = 0
    state_index = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            current_time = frame_count / fps
            current_state, state_index = renderer.state_for_time(current_time, state_index)
            renderer._draw_scoreboard(frame, current_state, target_w, target_h)
            proc.stdin.write(frame.tobytes())
            frame_count += 1
    finally:
        proc.stdin.close()
        proc.wait()
    cap.release()


def merge_audio(video_no_audio: str, audio_source: str, output_file: str) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_no_audio,
        "-i",
        audio_source,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        output_file,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        message = stderr if stderr else stdout
        raise RuntimeError(message if message else f"ffmpeg failed with code {exc.returncode}") from exc
