"""
Finger Counter + Gesture Emoji using MediaPipe Tasks and OpenCV
Detects hand landmarks via webcam, counts lifted fingers, and renders the
matching emoji (thumbs up, peace, middle finger, ...) using Apple Color Emoji.

Requirements:
    pip install opencv-python mediapipe numpy pillow certifi

Tested on macOS (Apple Silicon M3) with Python 3.12.
Press 'q' to quit, 's' to save a screenshot.
"""

import os
import ssl
import time
import urllib.request

import certifi
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from PIL import Image, ImageDraw, ImageFont

from fingercount import fingers
from fingercount.gestures import HAND_CONNECTIONS, Pattern, classify_gesture

# Pillow 10+ moved resampling constants under Image.Resampling.
_LANCZOS = getattr(Image, "Resampling", Image).LANCZOS


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "hand_landmarker.task")


def ensure_model() -> None:
    if os.path.exists(MODEL_PATH):
        return
    print(f"Downloading hand landmarker model to {MODEL_PATH} ...")
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(MODEL_URL, context=ctx) as resp, \
            open(MODEL_PATH, "wb") as out:
        out.write(resp.read())
    print("Model ready.")


class EmojiRenderer:
    """Renders color emojis onto BGR frames using Apple Color Emoji."""

    FONT_PATHS = [
        "/System/Library/Fonts/Apple Color Emoji.ttc",
        "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
        "C:/Windows/Fonts/seguiemj.ttf",
    ]
    # Apple Color Emoji ships bitmap strikes at fixed sizes only, and which
    # ones Pillow accepts depends on the build. We try the largest workable
    # size, rasterize once, then resize.
    CANDIDATE_SIZES = (160, 96, 64, 48, 137, 109, 32)

    def __init__(self):
        self.font = None
        self.native_size = 0
        for path in self.FONT_PATHS:
            if not os.path.exists(path):
                continue
            for size in self.CANDIDATE_SIZES:
                try:
                    self.font = ImageFont.truetype(path, size)
                    self.native_size = size
                    break
                except OSError:
                    continue
            if self.font is not None:
                break
        self._cache: dict[tuple[str, int], np.ndarray] = {}

    def _render_rgba(self, char: str, target_size: int) -> np.ndarray | None:
        key = (char, target_size)
        if key in self._cache:
            return self._cache[key]
        if self.font is None:
            return None

        canvas = Image.new("RGBA", (self.native_size + 20, self.native_size + 20),
                           (0, 0, 0, 0))
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

    def draw(self, frame: np.ndarray, char: str,
             center: tuple[int, int], size: int = 110) -> np.ndarray:
        rgba = self._render_rgba(char, size)
        if rgba is None:
            return frame

        h, w = rgba.shape[:2]
        cx, cy = center
        x0 = cx - w // 2
        y0 = cy - h // 2
        x1, y1 = x0 + w, y0 + h

        # Clip to frame bounds.
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
        rgb = patch[:, :, :3].astype(np.float32)
        bgr = rgb[:, :, ::-1]

        roi = frame[fy0:fy1, fx0:fx1].astype(np.float32)
        frame[fy0:fy1, fx0:fx1] = (bgr * alpha + roi * (1.0 - alpha)).astype(np.uint8)
        return frame


class FingerCounter:
    # Tip-to-tip distance (normalized by palm size) below which we treat
    # the thumb and index as touching → OK sign.
    PINCH_DIST = 0.45

    def __init__(self,
                 max_hands: int = 2,
                 detection_confidence: float = 0.5,
                 tracking_confidence: float = 0.5,
                 thresholds: fingers.FingerThresholds | None = None):
        self.thresholds = thresholds or fingers.DEFAULT_THRESHOLDS
        ensure_model()
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            running_mode=mp_vision.RunningMode.VIDEO,
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._t0 = time.time()

    def match_gesture(self, landmarks, pattern):
        """Pick the best gesture for a hand. Returns (label, emoji) or None."""
        return classify_gesture(landmarks, pattern, pinch_dist=self.PINCH_DIST)

    def extended_fingers(self, landmarks) -> Pattern:
        """Return (thumb, index, middle, ring, pinky) where 1 = extended."""
        return fingers.extended_fingers(landmarks, self.thresholds)

    def _draw_hand(self, frame: np.ndarray, landmarks) -> None:
        h, w = frame.shape[:2]
        for a, b in HAND_CONNECTIONS:
            x1, y1 = int(landmarks[a].x * w), int(landmarks[a].y * h)
            x2, y2 = int(landmarks[b].x * w), int(landmarks[b].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for lm in landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

    def process_frame(self, frame: np.ndarray):
        """Process one BGR frame; returns annotated frame and per-hand info."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int((time.time() - self._t0) * 1000)
        result = self.landmarker.detect_for_video(mp_image, ts_ms)

        total_fingers = 0
        hands_info = []  # list of dicts per hand

        if result.hand_landmarks:
            for landmarks, handedness in zip(result.hand_landmarks,
                                             result.handedness, strict=True):
                label = handedness[0].category_name  # "Left" or "Right"
                pattern = self.extended_fingers(landmarks)
                count = sum(pattern)
                total_fingers += count

                gesture = self.match_gesture(landmarks, pattern)
                hands_info.append({
                    "label": label,
                    "count": count,
                    "pattern": pattern,
                    "gesture": gesture,  # None if no match
                })
                self._draw_hand(frame, landmarks)

        return frame, total_fingers, hands_info

    def release(self) -> None:
        self.landmarker.close()


def _draw_emoji_panel(frame, hands_info, emoji: EmojiRenderer):
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
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h),
                  (20, 20, 20), thickness=-1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h),
                  (0, 255, 255), thickness=2)
    cv2.putText(frame, "Gesture", (x0 + 12, y0 + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    if not hands_info:
        cv2.putText(frame, "(no hand)", (x0 + 20, y0 + panel_h // 2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2,
                    cv2.LINE_AA)
        return

    # Draw each detected hand's gesture: emoji on the left, name on the right.
    cell_h = (panel_h - 40) // max(len(hands_info), 1)
    emoji_size = min(110, cell_h - 10)
    for i, info in enumerate(hands_info):
        cy = y0 + 40 + i * cell_h + cell_h // 2
        gesture = info["gesture"]
        if gesture is None:
            cv2.putText(frame, "?",
                        (x0 + 30, cy + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (200, 200, 200), 3,
                        cv2.LINE_AA)
            name = "(unknown)"
        else:
            name, char = gesture
            emoji.draw(frame, char, (x0 + emoji_size // 2 + 12, cy),
                       size=emoji_size)
        cv2.putText(frame, f"{info['label']}",
                    (x0 + emoji_size + 28, cy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1,
                    cv2.LINE_AA)
        cv2.putText(frame, name,
                    (x0 + emoji_size + 28, cy + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
                    cv2.LINE_AA)


def draw_overlay(frame, total: int, hands_info, fps: float,
                 emoji: EmojiRenderer):
    h, w = frame.shape[:2]

    cv2.rectangle(frame, (10, 10), (470, 70), (0, 0, 0), thickness=-1)
    cv2.rectangle(frame, (10, 10), (470, 70), (0, 255, 255), thickness=2)
    cv2.putText(
        frame, f"Fingers detected: {total}",
        (22, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA,
    )

    y = h - 50
    for info in reversed(hands_info):
        name = info["gesture"][0] if info["gesture"] else "(unknown)"
        cv2.putText(
            frame,
            f"{info['label']} hand: {info['count']}  -  {name}",
            (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            cv2.LINE_AA,
        )
        y -= 28

    _draw_emoji_panel(frame, hands_info, emoji)

    cv2.putText(
        frame, f"FPS: {fps:5.1f}",
        (w - 150, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame, "q: quit  |  s: screenshot",
        (w - 320, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame, "made by phorchkha",
        (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
        cv2.LINE_AA,
    )
    return frame


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open webcam. On macOS, give your terminal/PyCharm "
            "camera access in System Settings > Privacy & Security > Camera."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    counter = FingerCounter(max_hands=2)
    emoji = EmojiRenderer()
    if emoji.font is None:
        print("Warning: no color emoji font found; emojis will not render.")

    window_name = "Finger Count + Emoji"
    # WINDOW_NORMAL: user can drag the corner to resize.
    # WINDOW_KEEPRATIO: preserves aspect ratio when resizing.
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window_name, 1100, 620)

    prev_t = time.time()
    fps = 0.0
    shot_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame; exiting.")
                break

            frame = cv2.flip(frame, 1)
            frame, total, hands_info = counter.process_frame(frame)

            now = time.time()
            inst_fps = 1.0 / max(now - prev_t, 1e-6)
            fps = 0.9 * fps + 0.1 * inst_fps if fps else inst_fps
            prev_t = now

            frame = draw_overlay(frame, total, hands_info, fps, emoji)
            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('s'):
                shot_idx += 1
                name = f"finger_count_shot_{shot_idx:02d}.png"
                cv2.imwrite(name, frame)
                print(f"Saved {name}")

    finally:
        counter.release()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
