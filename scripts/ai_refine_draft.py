import sys
import cv2
import numpy as np
import argparse
from pathlib import Path
from collections import Counter

# Add root to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.ai_contract import load_draft_match, save_draft_match
from backend.ai_ollama_client import OllamaVisionClient
from backend.score_validation import build_score_validation


def extract_key_frames_grid(video_path, t_start, t_end):
    """
    Extract 3 frames and stitch them into one grid for AI to see context.
    """
    cap = cv2.VideoCapture(video_path)

    # Timestamps: Middle, Near End, and End
    times = [t_start + (t_end - t_start) * 0.5, t_end - 0.5, t_end]
    frames = []

    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (640, 360))
            frames.append(frame)

    cap.release()

    if len(frames) < 3:
        return None

    grid = np.hstack(frames)
    temp_path = "matches/temp_rally_grid.jpg"
    cv2.imwrite(temp_path, grid)
    return temp_path


def predict_winner_majority(client: OllamaVisionClient, image_path: str, votes: int):
    votes = max(1, int(votes))
    results = [client.predict_winner(image_path) for _ in range(votes)]
    c = Counter(results)
    winner, winner_count = c.most_common(1)[0]
    consensus = float(winner_count / votes)

    top_counts = [kv[1] for kv in c.most_common()]
    if len(top_counts) > 1 and top_counts[0] == top_counts[1]:
        return "unknown", 0.0, dict(c)

    return winner, consensus, dict(c)


def main():
    parser = argparse.ArgumentParser(description="Refine unknown rally winners using Ollama vision.")
    parser.add_argument("--draft", required=True, help="Path to input draft JSON")
    parser.add_argument("--out", default=None, help="Path to output refined JSON")
    parser.add_argument("--model", default="llama3.2-vision", help="Ollama vision model name")
    parser.add_argument("--votes", type=int, default=3, help="Number of AI votes per rally")
    parser.add_argument(
        "--expected-scope",
        choices=["any", "set", "match"],
        default="any",
        help="Expected clip scope for score-rule validation",
    )
    parser.add_argument(
        "--expected-final-set-score",
        default=None,
        help="Expected final set score in A-B format, e.g. 11-3",
    )
    args = parser.parse_args()

    draft_path = Path(args.draft)
    output_path = Path(args.out) if args.out else Path(str(draft_path).replace("_draft.json", "_refined.json"))

    print(f"--- AI Refinement with Ollama ({args.model}) ---")
    draft = load_draft_match(draft_path)
    client = OllamaVisionClient(model_name=args.model)

    processed_count = 0
    for p in draft.points:
        if p.winner == "unknown":
            print(f"Analyzing {p.id} ({p.t_start}s - {p.t_end}s)...")
            grid_path = extract_key_frames_grid(draft.video_path, p.t_start, p.t_end)

            if grid_path:
                prediction, consensus, raw_votes = predict_winner_majority(client, grid_path, args.votes)
                p.winner = prediction
                p.source = "ai"
                if consensus > 0:
                    p.confidence = max(float(p.confidence), float(consensus))
                if consensus < 0.67 and "winner_low_consensus" not in p.flags:
                    p.flags.append("winner_low_consensus")
                print(f"   > AI Result: {prediction} | consensus={consensus:.2f} | votes={raw_votes}")
                processed_count += 1

    draft.score_validation = build_score_validation(
        draft,
        expected_scope=args.expected_scope,
        expected_final_set_score=args.expected_final_set_score,
    )
    save_draft_match(output_path, draft)
    print(f"Validation: status={draft.score_validation.get('status')} | inferred={draft.score_validation.get('inferred_scoreline')}")
    for msg in draft.score_validation.get("issues", []):
        print(f"  - {msg}")
    print(f"\n--- DONE ---")
    print(f"Processed {processed_count} rallies. Refined JSON: {output_path}")


if __name__ == "__main__":
    main()
