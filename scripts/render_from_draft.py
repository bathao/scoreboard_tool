import sys
from pathlib import Path

# Fix path to import from root
sys.path.append(str(Path(__file__).parent.parent))

from backend.ai_contract import load_draft_match, to_core_rally_events
from backend.timeline import build_match_timeline
from render.renderer import ScoreboardRenderer

def main():
    # 1. Configuration
    DRAFT_JSON = "matches/Vinh_1280_2min_draft.json"
    INPUT_VIDEO = "Vinh_1280_2min.mp4"
    OUTPUT_VIDEO = "output_with_scoreboard.mp4"
    
    if not Path(DRAFT_JSON).exists():
        print(f"Error: {DRAFT_JSON} not found. Run main.py first.")
        return

    print(f"--- Loading Draft: {DRAFT_JSON} ---")
    draft = load_draft_match(Path(DRAFT_JSON))

    # 2. Manual Update (Optional: You can open JSON and change "unknown" to "player_a"/"player_b")
    # For testing, let's force all "unknown" to "player_a" to see the score count up
    for p in draft.points:
        if p.winner == "unknown":
            p.winner = "player_a" # Simulation: Player A wins everything

    # 3. Convert Draft to Core Events
    print("Converting draft to core rally events...")
    core_events = to_core_rally_events(draft, require_resolved_winner=True)

    # 4. Build Timeline (Match Logic)
    print("Building match timeline (calculating scores)...")
    timeline = build_match_timeline(best_of=draft.best_of, events=core_events)

    # 5. Render Video
    print(f"Rendering video to: {OUTPUT_VIDEO} ...")
    renderer = ScoreboardRenderer(
        input_path=INPUT_VIDEO,
        output_path=OUTPUT_VIDEO,
        timeline=timeline
    )
    renderer.render()
    print("--- SUCCESS: Video rendered with scoreboard ---")

if __name__ == "__main__":
    main()