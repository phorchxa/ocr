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

from fingercount.counter import FingerCounter
from fingercount.emoji import EmojiRenderer
from fingercount.overlay import draw_overlay


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
            total, hands_info = counter.process_frame(frame)

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
