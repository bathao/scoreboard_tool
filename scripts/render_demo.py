from backend.timeline import build_match_timeline
from render.renderer import ScoreboardRenderer


def main():

    winner_sequence = (
        ["player_a"] * 11 +
        ["player_b"] * 11 +
        ["player_a"] * 11
    )

    timeline = build_match_timeline(
        best_of=5,
        winner_sequence=winner_sequence
    )

    renderer = ScoreboardRenderer(
        input_path="input.mp4",
        output_path="output.mp4",
        timeline=timeline
    )

    renderer.render()


if __name__ == "__main__":
    main()