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

# Pillow 10+ moved resampling constants under Image.Resampling.
_LANCZOS = getattr(Image, "Resampling", Image).LANCZOS


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "hand_landmarker.task")

# Connections between MediaPipe hand landmarks, used for skeleton drawing.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm base
]

# Per-finger landmark triples (mcp, pip/ip, tip) used for angle checks.
# For the thumb we use (MCP=2, IP=3, TIP=4) — IP is the analogue of PIP.
FINGER_JOINTS = {
    "thumb":  (2, 3, 4),
    "index":  (5, 6, 8),
    "middle": (9, 10, 12),
    "ring":   (13, 14, 16),
    "pinky":  (17, 18, 20),
}
FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")

# Gesture lookup keyed by (thumb, index, middle, ring, pinky), 1 = extended.
# Several patterns map to the same gesture so the matcher tolerates noisy
# detections (e.g. a thumb that the model can't quite decide on).
GESTURES = {
    (0, 0, 0, 0, 0): ("Fist",          "\U0001F44A"),  # 👊
    (1, 0, 0, 0, 0): ("Thumbs up",     "\U0001F44D"),  # 👍
    (1, 1, 0, 0, 0): ("Thumbs up",     "\U0001F44D"),  # thumb + index (common slip)
    (0, 1, 0, 0, 0): ("Pointing",      "\U0000261D"),  # ☝
    (0, 0, 1, 0, 0): ("Middle finger", "\U0001F595"),  # 🖕
    (1, 0, 1, 0, 0): ("Middle finger", "\U0001F595"),  # thumb slip
    (0, 1, 1, 0, 0): ("Peace",         "\U0000270C"),  # ✌
    (1, 1, 1, 0, 0): ("Three",         "\U0001F522"),  # 🔢
    (0, 1, 1, 1, 0): ("Three",         "\U0001F522"),
    (0, 1, 1, 1, 1): ("Four",          "\U0001F590"),  # 🖐
    (1, 1, 1, 1, 1): ("Open palm",     "\U0001F590"),
    (1, 1, 1, 1, 0): ("Open palm",     "\U0001F590"),
    (1, 0, 0, 0, 1): ("Call me",       "\U0001F919"),  # 🤙
    (0, 1, 0, 0, 1): ("Rock on",       "\U0001F918"),  # 🤘
    (1, 1, 0, 0, 1): ("Love",          "\U0001F91F"),  # 🤟
    (0, 0, 0, 0, 1): ("Pinky",         "\U0001F90F"),  # 🤏 (closest match)
    (1, 0, 0, 1, 1): ("Spider-man",    "\U0001F578"),  # 🕸 (web-shooter pose)
}

# Two-tier tolerance: if the exact pattern doesn't match a gesture, we still
# accept the closest one as long as the Hamming distance is small enough.
GESTURE_TOLERANCE = 1


def ensure_model() -> None:
    if os.path.exists(MODEL_PATH):
        return
    print(f"Downloading hand landmarker model to {MODEL_PATH} ...")
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(MODEL_URL, context=ctx) as resp, \
            open(MODEL_PATH, "wb") as out:
        out.write(resp.read())
    print("Model ready.")


def _lm_xy(lm) -> np.ndarray:
    return np.array([lm.x, lm.y], dtype=np.float32)


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle in degrees at vertex b, formed by segments b->a and b->c."""
    v1 = a - b
    v2 = c - b
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-9
    cosang = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


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
    # Straightness thresholds (degrees) for the angle at the PIP/IP joint.
    # Closer to 180° = finger is straight. Loosened so detection survives
    # noisy landmarks at angled hand orientations.
    STRAIGHT_THRESHOLD = 150.0
    THUMB_STRAIGHT_THRESHOLD = 148.0
    # Tip-to-tip distance (normalized by palm size) below which we treat
    # the thumb and index as touching → OK sign.
    PINCH_DIST = 0.45

    def __init__(self,
                 max_hands: int = 2,
                 detection_confidence: float = 0.5,
                 tracking_confidence: float = 0.5):
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
        """Pick the best gesture for a hand. Returns (label, emoji) or None.

        Order: special-case shape checks (OK, thumbs down) first, then exact
        pattern lookup, then nearest pattern within Hamming distance ≤
        GESTURE_TOLERANCE. This makes recognition forgiving when the
        landmark detector flips one finger.
        """
        wrist = _lm_xy(landmarks[0])
        index_mcp = _lm_xy(landmarks[5])
        pinky_mcp = _lm_xy(landmarks[17])
        palm_size = max(np.linalg.norm(index_mcp - wrist),
                        np.linalg.norm(pinky_mcp - wrist), 1e-6)

        # OK sign: thumb tip touching index tip, with middle/ring/pinky out.
        thumb_tip = _lm_xy(landmarks[4])
        index_tip = _lm_xy(landmarks[8])
        pinch = np.linalg.norm(thumb_tip - index_tip) / palm_size
        if pinch < self.PINCH_DIST and pattern[2] and pattern[3] and pattern[4]:
            return ("OK", "\U0001F44C")  # 👌

        # Thumbs down: only thumb extended AND thumb tip is below the wrist
        # (larger y in image space). Otherwise it's thumbs up.
        if pattern == (1, 0, 0, 0, 0):
            if thumb_tip[1] > wrist[1] + 0.02:
                return ("Thumbs down", "\U0001F44E")  # 👎

        if pattern in GESTURES:
            return GESTURES[pattern]

        # Tolerance match: pick the closest catalogued pattern.
        best = None
        best_dist = GESTURE_TOLERANCE + 1
        for ref, value in GESTURES.items():
            dist = sum(a != b for a, b in zip(ref, pattern, strict=True))
            if dist < best_dist:
                best_dist = dist
                best = value
        if best is not None and best_dist <= GESTURE_TOLERANCE:
            return best
        return None

    def extended_fingers(self, landmarks) -> tuple[int, int, int, int, int]:
        """Return (thumb, index, middle, ring, pinky) where 1 = extended.

        Orientation-invariant: checks the angle at each finger's middle joint
        plus, for the non-thumb fingers, that the tip is farther from the
        wrist than the MCP. The thumb adds a splay check against the index
        MCP so a thumb tucked across the palm is not counted as extended.
        """
        wrist = _lm_xy(landmarks[0])
        index_mcp = _lm_xy(landmarks[5])
        pinky_mcp = _lm_xy(landmarks[17])
        palm_size = max(np.linalg.norm(index_mcp - wrist),
                        np.linalg.norm(pinky_mcp - wrist), 1e-6)

        out = []
        for name in FINGER_ORDER:
            mcp_id, pip_id, tip_id = FINGER_JOINTS[name]
            mcp = _lm_xy(landmarks[mcp_id])
            pip = _lm_xy(landmarks[pip_id])
            tip = _lm_xy(landmarks[tip_id])

            angle = _angle_deg(mcp, pip, tip)

            if name == "thumb":
                # Thumb is extended if it's straight AND splayed away from
                # the index MCP (so a curled-in thumb is not counted).
                splay = np.linalg.norm(tip - index_mcp) / palm_size
                extended = angle > self.THUMB_STRAIGHT_THRESHOLD and splay > 0.55
            else:
                # Tip must be farther from wrist than MCP (rules out fingers
                # that are straight but folded toward the palm), plus the
                # joint angle check.
                farther = (np.linalg.norm(tip - wrist)
                           > np.linalg.norm(mcp - wrist) * 1.05)
                extended = angle > self.STRAIGHT_THRESHOLD and farther

            out.append(1 if extended else 0)

        return tuple(out)  # type: ignore[return-value]

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
