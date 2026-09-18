"""
Finger Counter + Gesture Emoji using MediaPipe Tasks and OpenCV
Detects hand landmarks via webcam, counts lifted fingers, and renders the
matching emoji (thumbs up, peace, middle finger, ...) using Apple Color Emoji.

The implementation lives in the ``fingercount`` package; this file is kept
as a backwards-compatible entry point. Equivalent: ``python -m fingercount``.

Requirements:
    pip install -r requirements.txt

Tested on macOS (Apple Silicon M3) with Python 3.12.
Press 'q' to quit, 's' to save a screenshot.
"""

from fingercount.app import main

if __name__ == "__main__":
    main()
