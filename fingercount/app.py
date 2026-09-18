"""Webcam capture loop tying detection, rendering and keyboard handling together."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from fingercount.counter import FingerCounter
from fingercount.emoji import EmojiRenderer
from fingercount.overlay import draw_overlay

WINDOW_NAME = "Finger Count + Emoji"
KEY_QUIT = ord("q")
KEY_ESC = 27
KEY_SCREENSHOT = ord("s")


@dataclass(frozen=True)
class AppConfig:
    """Runtime settings for :func:`run`."""

    camera: int = 0
    width: int = 1280
    height: int = 720
    max_hands: int = 2
    detection_confidence: float = 0.5
    tracking_confidence: float = 0.5
    screenshot_dir: Path = field(default_factory=Path.cwd)
    window_size: tuple[int, int] = (1100, 620)


class FpsMeter:
    """Exponential moving average of instantaneous frames per second."""

    def __init__(self, smoothing: float = 0.9) -> None:
        self.smoothing = smoothing
        self.fps = 0.0
        self._prev = time.time()

    def tick(self) -> float:
        now = time.time()
        inst = 1.0 / max(now - self._prev, 1e-6)
        self._prev = now
        self.fps = self.smoothing * self.fps + (1.0 - self.smoothing) * inst if self.fps else inst
        return self.fps


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    """Open a capture device, preferring AVFoundation on macOS."""
    cap = None
    if sys.platform == "darwin":
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam {index}. On macOS, give your terminal/IDE "
            "camera access in System Settings > Privacy & Security > Camera."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def screenshot_path(directory: Path, index: int) -> Path:
    return directory / f"finger_count_shot_{index:02d}.png"


def run(config: AppConfig | None = None) -> None:
    """Run the interactive loop until the user presses q or Esc."""
    cfg = config or AppConfig()
    cap = open_camera(cfg.camera, cfg.width, cfg.height)

    emoji = EmojiRenderer()
    if emoji.font is None:
        print("Warning: no color emoji font found; emojis will not render.")

    # WINDOW_NORMAL lets the user drag to resize; KEEPRATIO preserves aspect.
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WINDOW_NAME, *cfg.window_size)

    meter = FpsMeter()
    shot_idx = 0

    try:
        with FingerCounter(
            max_hands=cfg.max_hands,
            detection_confidence=cfg.detection_confidence,
            tracking_confidence=cfg.tracking_confidence,
        ) as counter:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Failed to read frame; exiting.")
                    break

                frame = cv2.flip(frame, 1)
                total, hands = counter.process_frame(frame)
                fps = meter.tick()
                frame = draw_overlay(frame, total, hands, fps, emoji)
                cv2.imshow(WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (KEY_QUIT, KEY_ESC):
                    break
                if key == KEY_SCREENSHOT:
                    shot_idx += 1
                    path = screenshot_path(cfg.screenshot_dir, shot_idx)
                    cv2.imwrite(str(path), frame)
                    print(f"Saved {path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    run()
