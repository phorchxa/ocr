"""MediaPipe hand tracking wrapper that turns frames into per-hand results."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from fingercount import fingers
from fingercount.gestures import DEFAULT_PINCH_DIST, Gesture, Pattern, classify_gesture
from fingercount.model import ensure_model


@dataclass(frozen=True)
class HandInfo:
    """Everything the UI needs to know about one detected hand."""

    label: str  # "Left" or "Right" as reported by MediaPipe
    count: int  # number of extended fingers
    pattern: Pattern
    gesture: Gesture | None  # None when no catalogue entry is close enough
    landmarks: Sequence  # the 21 normalized landmarks


class FingerCounter:
    """Runs the hand landmarker on video frames and classifies each hand."""

    def __init__(
        self,
        max_hands: int = 2,
        detection_confidence: float = 0.5,
        tracking_confidence: float = 0.5,
        thresholds: fingers.FingerThresholds | None = None,
        pinch_dist: float = DEFAULT_PINCH_DIST,
        model_path: Path | str | None = None,
    ):
        self.thresholds = thresholds or fingers.DEFAULT_THRESHOLDS
        self.pinch_dist = pinch_dist
        path = ensure_model() if model_path is None else ensure_model(model_path)
        base_options = mp_python.BaseOptions(model_asset_path=str(path))
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

    def match_gesture(self, landmarks, pattern: Pattern) -> Gesture | None:
        """Pick the best gesture for a hand. Returns (label, emoji) or None."""
        return classify_gesture(landmarks, pattern, pinch_dist=self.pinch_dist)

    def extended_fingers(self, landmarks) -> Pattern:
        """Return (thumb, index, middle, ring, pinky) where 1 = extended."""
        return fingers.extended_fingers(landmarks, self.thresholds)

    def process_frame(self, frame: np.ndarray) -> tuple[int, list[HandInfo]]:
        """Detect hands in one BGR frame; returns (total fingers, hands).

        The frame is not modified. Use :mod:`fingercount.overlay` to draw.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int((time.time() - self._t0) * 1000)
        result = self.landmarker.detect_for_video(mp_image, ts_ms)

        total_fingers = 0
        hands: list[HandInfo] = []

        if result.hand_landmarks:
            for landmarks, handedness in zip(
                result.hand_landmarks, result.handedness, strict=True
            ):
                pattern = self.extended_fingers(landmarks)
                count = sum(pattern)
                total_fingers += count
                hands.append(
                    HandInfo(
                        label=handedness[0].category_name,
                        count=count,
                        pattern=pattern,
                        gesture=self.match_gesture(landmarks, pattern),
                        landmarks=landmarks,
                    )
                )

        return total_fingers, hands

    def release(self) -> None:
        self.landmarker.close()

    def __enter__(self) -> FingerCounter:
        return self

    def __exit__(self, *exc) -> None:
        self.release()
