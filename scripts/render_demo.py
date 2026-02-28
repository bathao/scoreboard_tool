from backend.models import RallyEvent
from backend.timeline import build_match_timeline
from render.renderer import ScoreboardRenderer


def main():

    events = [
        RallyEvent("player_a", 3.0),
        RallyEvent("player_b", 7.0),
        RallyEvent("player_a", 11.0),
        RallyEvent("player_a", 15.0),
    ]

    timeline = build_match_timeline(best_of=5, events=events)

    renderer = ScoreboardRenderer(
        input_path="input.mp4",
        output_path="output.mp4",
        timeline=timeline
    )

    renderer.render()


if __name__ == "__main__":
    main()