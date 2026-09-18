"""
Finger Counter + Gesture Emoji using MediaPipe Tasks and OpenCV
Detects hand landmarks via webcam, counts lifted fingers, and renders the
matching emoji (thumbs up, peace, middle finger, ...) using Apple Color Emoji.

Requirements:
    pip install opencv-python mediapipe numpy pillow certifi

Tested on macOS (Apple Silicon M3) with Python 3.12.
Press 'q' to quit, 's' to save a screenshot.
"""

import time

import cv2

from fingercount.counter import FingerCounter, HandInfo
from fingercount.emoji import EmojiRenderer


def _draw_emoji_panel(frame, hands_info: list[HandInfo], emoji: EmojiRenderer):
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
        gesture = info.gesture
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
        cv2.putText(frame, f"{info.label}",
                    (x0 + emoji_size + 28, cy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1,
                    cv2.LINE_AA)
        cv2.putText(frame, name,
                    (x0 + emoji_size + 28, cy + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
                    cv2.LINE_AA)


def draw_overlay(frame, total: int, hands_info: list[HandInfo], fps: float,
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
        name = info.gesture[0] if info.gesture else "(unknown)"
        cv2.putText(
            frame,
            f"{info.label} hand: {info.count}  -  {name}",
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
