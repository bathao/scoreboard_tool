from __future__ import annotations

import os
from typing import List

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from backend.models import MatchSnapshot

# ---------------------------------------------------------------------------
# Font helpers
# ---------------------------------------------------------------------------

_FONT_CANDIDATES_REGULAR = [
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\calibri.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\tahoma.ttf",
]
_FONT_CANDIDATES_BOLD = [
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\calibrib.ttf",
    r"C:\Windows\Fonts\segoeuib.ttf",
    r"C:\Windows\Fonts\tahomabd.ttf",
]


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = _FONT_CANDIDATES_BOLD if bold else _FONT_CANDIDATES_REGULAR
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# ScoreboardRenderer
# ---------------------------------------------------------------------------

class ScoreboardRenderer:

    def __init__(
        self,
        input_path: str,
        output_path: str,
        timeline: List[MatchSnapshot],
        player_a_name: str = "PLAYER A",
        player_b_name: str = "PLAYER B",
        tournament_name: str = "",
        round_name: str = "",
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.timeline = timeline
        self.player_a_name = self._display_name(player_a_name, "PLAYER A")
        self.player_b_name = self._display_name(player_b_name, "PLAYER B")
        self.tournament_name = str(tournament_name or "").strip()
        self.round_name = str(round_name or "").strip()

        if not self.timeline:
            raise ValueError("Timeline cannot be empty")
        self._initial_state = MatchSnapshot(
            timestamp=0.0,
            set_number=1,
            score_a=0,
            score_b=0,
            sets_a=0,
            sets_b=0,
            is_finished=False,
            winner=None,
        )

        # Pre-load fonts (Vietnamese-capable TrueType)
        self._font_header  = _load_font(20)
        self._font_name    = _load_font(26)
        self._font_sets    = _load_font(30, bold=True)
        self._font_pts     = _load_font(30, bold=True)
        self._font_winner  = _load_font(26, bold=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _display_name(self, value: str, fallback: str) -> str:
        text = str(value or "").strip()
        if not text:
            return fallback
        if len(text) > 20:
            return f"{text[:17]}..."
        return text

    def _winner_label(self, winner: str | None) -> str:
        if winner == "player_a":
            return self.player_a_name
        if winner == "player_b":
            return self.player_b_name
        return "Unknown"

    def state_for_time(self, current_time: float, state_index: int) -> tuple[MatchSnapshot, int]:
        while (
            state_index + 1 < len(self.timeline)
            and current_time >= self.timeline[state_index + 1].timestamp
        ):
            state_index += 1
        if current_time < self.timeline[0].timestamp:
            return self._initial_state, state_index
        return self.timeline[state_index], state_index

    # ------------------------------------------------------------------
    # Text drawing — PIL-based (supports Vietnamese and full Unicode)
    # ------------------------------------------------------------------

    @staticmethod
    def _text_size(text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int]:
        """Return (width, height) of text in pixels."""
        bbox = font.getbbox(text)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    @staticmethod
    def _put_text(
        frame: np.ndarray,
        text: str,
        x: int,
        y: int,
        font: ImageFont.FreeTypeFont,
        color_bgr: tuple[int, int, int],
    ) -> None:
        """Draw Unicode text on a BGR frame; (x, y) is the top-left origin."""
        if not text:
            return
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
        bbox = font.getbbox(text)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        pad = 2
        rx1 = max(0, x - pad)
        ry1 = max(0, y - pad)
        rx2 = min(frame.shape[1], x + tw + pad * 2)
        ry2 = min(frame.shape[0], y + th + pad * 2)
        if rx2 <= rx1 or ry2 <= ry1:
            return
        region = frame[ry1:ry2, rx1:rx2].copy()
        pil_img = Image.fromarray(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        draw.text((x - rx1 - bbox[0], y - ry1 - bbox[1]), text, font=font, fill=color_rgb)
        frame[ry1:ry2, rx1:rx2] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def _put_text_cx(
        self,
        frame: np.ndarray,
        text: str,
        cx: int,
        cy: int,
        font: ImageFont.FreeTypeFont,
        color_bgr: tuple[int, int, int],
    ) -> None:
        """Draw text centred at (cx, cy)."""
        tw, th = self._text_size(text, font)
        self._put_text(frame, text, cx - tw // 2, cy - th // 2, font, color_bgr)

    # ------------------------------------------------------------------
    # Legacy render() — uses cv2 VideoWriter (kept for compatibility)
    # ------------------------------------------------------------------

    def render(self):
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            raise RuntimeError("Cannot open input video")
        fps = cap.get(cv2.CAP_PROP_FPS)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
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
            current_state, state_index = self.state_for_time(current_time, state_index)
            self._draw_scoreboard(frame, current_state, width, height)
            out.write(frame)
            frame_count += 1
        cap.release()
        out.release()

    # ------------------------------------------------------------------
    # DRAWING
    # ------------------------------------------------------------------

    def _draw_scoreboard(self, frame, state: MatchSnapshot, width: int, height: int):
        # --- Layout constants (tuned for 1920×1080) ---
        row_h     = 56
        pad_x     = 18
        name_col_w = 320
        sets_col_w = 66
        pts_col_w  = 82
        bar_w = pad_x + name_col_w + sets_col_w + pts_col_w + pad_x
        margin = 26

        white      = (255, 255, 255)
        grey       = (185, 185, 185)
        gold       = (40,  140, 180)   # BGR — displays as gold/amber on screen
        sep_color  = (75,  75,  75)

        # --- Header (single line: tournament · round) ---
        parts       = [p for p in [self.tournament_name, self.round_name] if p]
        header_text = "  ·  ".join(parts)
        header_pad  = 8
        header_h    = (22 + header_pad * 2) if header_text else 0

        player_bar_h = row_h * 2 + 2
        total_h      = header_h + player_bar_h

        # Position: bottom-right corner
        x1    = width - bar_w - margin
        y1    = height - total_h - margin
        x2    = x1 + bar_w
        y2    = y1 + total_h
        hdr_y2 = y1 + header_h
        mid_y  = hdr_y2 + row_h

        col_sets_x = x1 + pad_x + name_col_w
        col_pts_x  = col_sets_x + sets_col_w

        # --- Header strip ---
        if header_text:
            ov = frame.copy()
            cv2.rectangle(ov, (x1, y1), (x2, hdr_y2), (35, 35, 35), -1)
            cv2.addWeighted(ov, 0.90, frame, 0.10, 0, frame)
            cv2.line(frame, (x1, y1), (x2, y1), (40, 140, 180), 2)  # gold top line
            self._put_text(
                frame, header_text,
                x1 + pad_x, y1 + header_pad,
                self._font_header, grey,
            )
            cv2.line(frame, (x1, hdr_y2), (x2, hdr_y2), sep_color, 1)

        # --- Player rows background ---
        ov2 = frame.copy()
        cv2.rectangle(ov2, (x1, hdr_y2), (x2, y2), (18, 18, 18), -1)
        cv2.addWeighted(ov2, 0.88, frame, 0.12, 0, frame)

        # Accent bars (left edge, per player)
        cv2.rectangle(frame, (x1, hdr_y2),    (x1 + 5, mid_y),  (30, 100, 210), -1)  # orange-A
        cv2.rectangle(frame, (x1, mid_y + 2), (x1 + 5, y2),     (220, 140, 50), -1)  # blue-B

        # Separators
        cv2.line(frame, (x1, mid_y), (x2, mid_y), sep_color, 2)
        cv2.line(frame, (col_sets_x, hdr_y2 + 5), (col_sets_x, y2 - 5), sep_color, 1)
        cv2.line(frame, (col_pts_x,  hdr_y2 + 5), (col_pts_x,  y2 - 5), sep_color, 1)

        def _draw_row(name: str, sets: int, pts: int, row_y1: int, row_y2: int) -> None:
            cy = (row_y1 + row_y2) // 2

            # Player name — left-aligned
            tw, th = self._text_size(name, self._font_name)
            self._put_text(
                frame, name,
                x1 + pad_x + 6, cy - th // 2,
                self._font_name, white,
            )

            # Sets won — centred in sets column
            s_str = str(sets)
            self._put_text_cx(
                frame, s_str,
                col_sets_x + sets_col_w // 2, cy,
                self._font_sets, grey,
            )

            # Game points — centred in pts column
            p_str = str(pts)
            self._put_text_cx(
                frame, p_str,
                col_pts_x + pts_col_w // 2, cy,
                self._font_pts, white,
            )

        _draw_row(self.player_a_name, state.sets_a, state.score_a, hdr_y2,     mid_y)
        _draw_row(self.player_b_name, state.sets_b, state.score_b, mid_y + 2,  y2)

        # Winner banner above the box
        if state.is_finished:
            w_text = f"WINNER: {self._winner_label(state.winner)}"
            tw, th = self._text_size(w_text, self._font_winner)
            self._put_text(
                frame, w_text,
                x1, y1 - th - 8,
                self._font_winner, (100, 230, 60),
            )
