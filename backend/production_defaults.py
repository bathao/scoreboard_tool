from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RallyTimelineDefaults:
    table_weights_path: str = "weights/yolov8x_table.pt"
    pose_weights_path: str = "weights/yolov8x-pose.pt"
    stride: int = 2
    mode: str = "player"
    player_margin_px: int = 220
    player_fuse_gain: float = 1.0
    player_signal_source: str = "role_tracker"
    ball_fuse_gain: float = 1.15
    ball_signal_source: str = "classical"


@dataclass(frozen=True)
class WinnerAdapterDefaults:
    base_model_dir: str = "models/Qwen3-VL-4B-Instruct"
    adapter_dir: str = "models/adapters/qwen3vl4b_table_tennis_pilot_4ep_cache_v2"
    fps_sample: float = 1.0
    min_frames: int = 4
    max_frames: int = 4
    size_shortest_edge: int = 384
    size_longest_edge: int = 1048576
    max_pixels: int = 262144
    max_new_tokens: int = 64


PRODUCTION_RALLY_DEFAULTS = RallyTimelineDefaults()
PRODUCTION_WINNER_DEFAULTS = WinnerAdapterDefaults()


def production_defaults_summary() -> dict[str, object]:
    return {
        "rally_timeline": asdict(PRODUCTION_RALLY_DEFAULTS),
        "winner_adapter": asdict(PRODUCTION_WINNER_DEFAULTS),
    }
