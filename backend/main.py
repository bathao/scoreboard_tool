import json
from pathlib import Path

from score_engine import (
    MatchEngine,
    MatchAlreadyFinishedError,
    MatchNotFinishedError,
    InvalidWinnerCodeError,
)


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_match(filename: str = "match_001.json") -> dict:
    project_root = get_project_root()
    match_path = project_root / "matches" / filename

    if not match_path.exists():
        raise FileNotFoundError(f"Match file not found: {match_path}")

    with open(match_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    try:
        match_data = load_match()
        
        print("Best of:", match_data["best_of"])
        print("Total points:", len(match_data["points"]))

        engine = MatchEngine(match_data)
        # result = engine.process()
        try:
             result = engine.process()

        except Exception as e:
            # DEBUG SECTION
            print("\n==== DEBUG INFO ====")
            print("Sets completed:", len(engine.sets))


            for i, s in enumerate(engine.sets, 1):
                print(f"Set {i}: {s['A']} - {s['B']} (Winner: {s['winner']})")

            # Tính điểm set đang dở
            current_a = 0
            current_b = 0

            for p in match_data["points"]:
                if p == "A":
                    current_a += 1
                else:
                    current_b += 1

                # reset nếu đủ điều kiện kết thúc set
                if (current_a >= 11 or current_b >= 11) and abs(current_a - current_b) >= 2:
                    current_a = 0
                    current_b = 0

            print("Unfinished set score:", current_a, "-", current_b)
            print("====================\n")

            raise e

        print("\n==========================")
        print("✅ MATCH FINISHED")
        print("==========================")
        print("Winner:", result["match_winner"])
        print("Sets:")

        for i, s in enumerate(result["sets"], 1):
            print(f"Set {i}: {s['A']} - {s['B']} (Winner: {s['winner']})")

        print("==========================\n")

    except MatchNotFinishedError as e:
        print("❌ ERROR:", e)

    except InvalidWinnerCodeError as e:
        print("❌ INVALID POINT:", e)

    except MatchAlreadyFinishedError as e:
        print("❌ LOGIC ERROR:", e)

    except Exception as e:
        print("❌ UNEXPECTED ERROR:", e)


if __name__ == "__main__":
    main()