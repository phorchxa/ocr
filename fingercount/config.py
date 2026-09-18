"""Runtime configuration shared by the CLI and the capture loop.

Kept free of OpenCV and MediaPipe imports so it can be unit-tested cheaply.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    """Runtime settings for :func:`fingercount.app.run`."""

    camera: int = 0
    width: int = 1280
    height: int = 720
    max_hands: int = 2
    detection_confidence: float = 0.5
    tracking_confidence: float = 0.5
    screenshot_dir: Path = field(default_factory=Path.cwd)
    window_size: tuple[int, int] = (1100, 620)
