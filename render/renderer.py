import cv2
from typing import List
from backend.models import MatchSnapshot


class ScoreboardRenderer:

    def __init__(self, input_path: str, output_path: str, timeline: List[MatchSnapshot]):
        self.input_path = input_path
        self.output_path = output_path
        self.timeline = timeline

        if not self.timeline:
            raise ValueError("Timeline cannot be empty")

    def render(self):

        cap = cv2.VideoCapture(self.input_path)

        if not cap.isOpened():
            raise RuntimeError("Cannot open input video")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

        frame_count = 0
        state_index = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = frame_count / fps

            # advance timeline by timestamp
            while (
                state_index + 1 < len(self.timeline)
                and current_time >= self.timeline[state_index + 1].timestamp
            ):
                state_index += 1

            current_state = self.timeline[state_index]

            self._draw_scoreboard(frame, current_state, width, height)

            out.write(frame)
            frame_count += 1

        cap.release()
        out.release()

    # ----------------------------------------------------
    # DRAWING
    # ----------------------------------------------------

    def _draw_scoreboard(self, frame, state: MatchSnapshot, width: int, height: int):

        scoreboard_width = 280
        scoreboard_height = 110

        margin = 20

        # bottom-right corner
        x1 = width - scoreboard_width - margin
        y1 = height - scoreboard_height - margin
        x2 = width - margin
        y2 = height - margin

        # background box
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        font = cv2.FONT_HERSHEY_SIMPLEX
        white = (255, 255, 255)

        # Titles
        cv2.putText(frame, "PLAYER A", (x1 + 15, y1 + 30), font, 0.6, white, 2)
        cv2.putText(frame, "PLAYER B", (x1 + 15, y1 + 60), font, 0.6, white, 2)

        # Scores
        cv2.putText(
            frame,
            str(state.score_a),
            (x2 - 60, y1 + 30),
            font,
            0.9,
            white,
            2,
        )

        cv2.putText(
            frame,
            str(state.score_b),
            (x2 - 60, y1 + 60),
            font,
            0.9,
            white,
            2,
        )

        # Set score
        set_text = f"Sets: {state.sets_a} - {state.sets_b}"
        cv2.putText(
            frame,
            set_text,
            (x1 + 15, y1 + 90),
            font,
            0.6,
            white,
            2,
        )

        # If match finished
        if state.is_finished:
            winner_text = f"Winner: {state.winner}"
            cv2.putText(
                frame,
                winner_text,
                (x1 + 15, y1 - 10),
                font,
                0.7,
                (0, 255, 0),
                2,
            )