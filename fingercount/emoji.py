"""Rendering color emoji glyphs onto OpenCV-style BGR frames."""

from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Pillow 10+ moved resampling constants under Image.Resampling.
_LANCZOS = getattr(Image, "Resampling", Image).LANCZOS

# Color emoji fonts to try, in order, across macOS, Linux and Windows.
FONT_PATHS = (
    "/System/Library/Fonts/Apple Color Emoji.ttc",
    "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
    "C:/Windows/Fonts/seguiemj.ttf",
)

# Apple Color Emoji ships bitmap strikes at fixed sizes only, and which ones
# Pillow accepts depends on the build. We try the largest workable size,
# rasterize once, then resize.
CANDIDATE_SIZES = (160, 96, 64, 48, 137, 109, 32)


def load_emoji_font(
    paths: tuple[str, ...] = FONT_PATHS,
    sizes: tuple[int, ...] = CANDIDATE_SIZES,
) -> tuple[ImageFont.FreeTypeFont | None, int]:
    """Return the first loadable ``(font, native_size)``, or ``(None, 0)``."""
    for path in paths:
        if not os.path.exists(path):
            continue
        for size in sizes:
            try:
                return ImageFont.truetype(path, size), size
            except OSError:
                continue
    return None, 0


def composite_rgba(
    frame: np.ndarray, rgba: np.ndarray, center: tuple[int, int]
) -> np.ndarray:
    """Alpha-blend an RGBA patch onto a BGR frame, centered at ``center``.

    The patch is clipped to the frame bounds, so it may hang off any edge.
    ``frame`` is modified in place and also returned for convenience.
    """
    h, w = rgba.shape[:2]
    cx, cy = center
    x0 = cx - w // 2
    y0 = cy - h // 2
    x1, y1 = x0 + w, y0 + h

    fx0 = max(x0, 0)
    fy0 = max(y0, 0)
    fx1 = min(x1, frame.shape[1])
    fy1 = min(y1, frame.shape[0])
    if fx0 >= fx1 or fy0 >= fy1:
        return frame

    sx0 = fx0 - x0
    sy0 = fy0 - y0
    sx1 = sx0 + (fx1 - fx0)
    sy1 = sy0 + (fy1 - fy0)

    patch = rgba[sy0:sy1, sx0:sx1]
    alpha = patch[:, :, 3:4].astype(np.float32) / 255.0
    bgr = patch[:, :, :3].astype(np.float32)[:, :, ::-1]

    roi = frame[fy0:fy1, fx0:fx1].astype(np.float32)
    frame[fy0:fy1, fx0:fx1] = (bgr * alpha + roi * (1.0 - alpha)).astype(np.uint8)
    return frame


class EmojiRenderer:
    """Renders color emojis onto BGR frames, caching rasterized glyphs."""

    def __init__(self) -> None:
        self.font, self.native_size = load_emoji_font()
        self._cache: dict[tuple[str, int], np.ndarray] = {}

    def _render_rgba(self, char: str, target_size: int) -> np.ndarray | None:
        key = (char, target_size)
        if key in self._cache:
            return self._cache[key]
        if self.font is None:
            return None

        canvas = Image.new(
            "RGBA", (self.native_size + 20, self.native_size + 20), (0, 0, 0, 0)
        )
        draw = ImageDraw.Draw(canvas)
        try:
            draw.text((10, 0), char, font=self.font, embedded_color=True)
        except Exception:
            return None

        bbox = canvas.getbbox()
        if bbox is None:
            return None
        cropped = canvas.crop(bbox)
        resized = cropped.resize((target_size, target_size), _LANCZOS)
        arr = np.array(resized)  # RGBA
        self._cache[key] = arr
        return arr

    def draw(
        self, frame: np.ndarray, char: str, center: tuple[int, int], size: int = 110
    ) -> np.ndarray:
        """Draw ``char`` centered at ``center``; a no-op if no font was found."""
        rgba = self._render_rgba(char, size)
        if rgba is None:
            return frame
        return composite_rgba(frame, rgba, center)
