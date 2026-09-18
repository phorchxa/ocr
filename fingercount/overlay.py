"""All OpenCV drawing: hand skeletons, the gesture panel and the HUD text."""

from __future__ import annotations

import cv2
import numpy as np

from fingercount.counter import HandInfo
from fingercount.emoji import EmojiRenderer
from fingercount.gestures import HAND_CONNECTIONS

# Colors are BGR, as OpenCV expects.
YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
LIGHT_GREY = (200, 200, 200)
GREY = (180, 180, 180)
BLACK = (0, 0, 0)
PANEL_BG = (20, 20, 20)

FONT = cv2.FONT_HERSHEY_SIMPLEX
CREDIT = "made by phorchkha"
HELP_TEXT = "q: quit  |  s: screenshot"


def draw_skeleton(frame: np.ndarray, landmarks) -> None:
    """Draw landmark connections and joints for one hand, in place."""
    h, w = frame.shape[:2]
    for a, b in HAND_CONNECTIONS:
        x1, y1 = int(landmarks[a].x * w), int(landmarks[a].y * h)
        x2, y2 = int(landmarks[b].x * w), int(landmarks[b].y * h)
        cv2.line(frame, (x1, y1), (x2, y2), GREEN, 2)
    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (x, y), 4, RED, -1)


def _draw_emoji_panel(frame: np.ndarray, hands: list[HandInfo], emoji: EmojiRenderer) -> None:
    """Fixed panel in the top-right showing the current gesture(s).

    The panel never follows the hand: position is anchored to the frame.
    Up to two hands are shown stacked vertically.
    """
    h, w = frame.shape[:2]
    panel_w, panel_h = 220, 180
    pad = 20
    x0 = w - panel_w - pad
    y0 = pad + 70  # leave room for the FPS readout above

    # Backdrop.
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), PANEL_BG, thickness=-1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), YELLOW, thickness=2)
    cv2.putText(frame, "Gesture", (x0 + 12, y0 + 26), FONT, 0.7, YELLOW, 2, cv2.LINE_AA)

    if not hands:
        cv2.putText(
            frame, "(no hand)", (x0 + 20, y0 + panel_h // 2 + 10),
            FONT, 0.7, GREY, 2, cv2.LINE_AA,
        )
        return

    # Draw each detected hand's gesture: emoji on the left, name on the right.
    cell_h = (panel_h - 40) // max(len(hands), 1)
    emoji_size = min(110, cell_h - 10)
    for i, hand in enumerate(hands):
        cy = y0 + 40 + i * cell_h + cell_h // 2
        if hand.gesture is None:
            cv2.putText(frame, "?", (x0 + 30, cy + 12), FONT, 1.5, LIGHT_GREY, 3, cv2.LINE_AA)
            name = "(unknown)"
        else:
            name, char = hand.gesture
            emoji.draw(frame, char, (x0 + emoji_size // 2 + 12, cy), size=emoji_size)
        cv2.putText(
            frame, hand.label, (x0 + emoji_size + 28, cy - 6),
            FONT, 0.55, GREY, 1, cv2.LINE_AA,
        )
        cv2.putText(
            frame, name, (x0 + emoji_size + 28, cy + 18),
            FONT, 0.65, WHITE, 2, cv2.LINE_AA,
        )


def draw_overlay(
    frame: np.ndarray,
    total: int,
    hands: list[HandInfo],
    fps: float,
    emoji: EmojiRenderer,
) -> np.ndarray:
    """Draw skeletons, the finger count box, per-hand lines, panel and HUD."""
    h, w = frame.shape[:2]

    for hand in hands:
        draw_skeleton(frame, hand.landmarks)

    cv2.rectangle(frame, (10, 10), (470, 70), BLACK, thickness=-1)
    cv2.rectangle(frame, (10, 10), (470, 70), YELLOW, thickness=2)
    cv2.putText(
        frame, f"Fingers detected: {total}", (22, 52), FONT, 1.0, YELLOW, 2, cv2.LINE_AA,
    )

    y = h - 50
    for hand in reversed(hands):
        name = hand.gesture[0] if hand.gesture else "(unknown)"
        cv2.putText(
            frame, f"{hand.label} hand: {hand.count}  -  {name}",
            (15, y), FONT, 0.7, WHITE, 2, cv2.LINE_AA,
        )
        y -= 28

    _draw_emoji_panel(frame, hands, emoji)

    cv2.putText(frame, f"FPS: {fps:5.1f}", (w - 150, 35), FONT, 0.7, LIGHT_GREY, 2, cv2.LINE_AA)
    cv2.putText(frame, HELP_TEXT, (w - 320, h - 15), FONT, 0.55, GREY, 1, cv2.LINE_AA)
    cv2.putText(frame, CREDIT, (15, h - 15), FONT, 0.6, YELLOW, 2, cv2.LINE_AA)
    return frame
