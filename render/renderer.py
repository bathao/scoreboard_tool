import cv2
from pathlib import Path


class ScoreboardRenderer:

    def __init__(self, input_path: str, output_path: str, timeline: list[dict]):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.timeline = timeline

    def render(self):

        # Validate timeline
        if not self.timeline:
            raise ValueError("Timeline is empty. Nothing to render.")

        # Validate input file existence
        if not self.input_path.exists():
            raise FileNotFoundError(
                f"Input video not found: {self.input_path}"
            )

        cap = cv2.VideoCapture(str(self.input_path))

        # Validate video can open
        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open video file: {self.input_path}"
            )

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps == 0:
            raise RuntimeError("Invalid video FPS detected.")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            fps,
            (width, height)
        )

        frame_count = 0
        rally_index = 0

        # Temporary mapping: 3 seconds per rally
        frames_per_rally = int(fps * 3)

        print("Starting render...")
        print(f"Input: {self.input_path}")
        print(f"Output: {self.output_path}")
        print(f"FPS: {fps}")
        print(f"Resolution: {width}x{height}")
        print(f"Total rallies: {len(self.timeline)}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if rally_index < len(self.timeline):
                state = self.timeline[rally_index]
            else:
                state = self.timeline[-1]

            self._draw_scoreboard(frame, state)

            out.write(frame)

            frame_count += 1

            if frame_count % frames_per_rally == 0:
                rally_index += 1

        cap.release()
        out.release()

        print("Render completed successfully.")

    def _draw_scoreboard(self, frame, state):

        height, width, _ = frame.shape

        text_score = f"A {state['score_a']} - {state['score_b']} B"
        text_sets = f"Sets: {state['sets_a']} - {state['sets_b']}"

        # ---- Box size ----
        box_width = 480
        box_height = 130
        margin = 20

        # Bottom-right positioning
        x1 = width - box_width - margin
        y1 = height - box_height - margin
        x2 = width - margin
        y2 = height - margin

        # Background box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)

        # Text positions relative to box
        score_pos = (x1 + 20, y1 + 55)
        sets_pos = (x1 + 20, y1 + 100)

        cv2.putText(
            frame,
            text_score,
            score_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (255, 255, 255),
            3,
        )

        cv2.putText(
            frame,
            text_sets,
            sets_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
        )