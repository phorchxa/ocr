"""Gesture catalogue and matching logic.

A *pattern* is a 5-tuple ``(thumb, index, middle, ring, pinky)`` where ``1``
means the finger is extended. Matching is plain Python and numpy so it can
be unit-tested without MediaPipe or a webcam.
"""

from __future__ import annotations

import numpy as np

from fingercount.geometry import WRIST, lm_xy, palm_size

Pattern = tuple[int, int, int, int, int]
Gesture = tuple[str, str]  # (label, emoji character)

# Connections between MediaPipe hand landmarks, used for skeleton drawing.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm base
]


THUMB_TIP = 4
INDEX_TIP = 8

# Gesture lookup keyed by pattern. Several patterns map to the same gesture
# so the matcher tolerates noisy detections (e.g. a thumb the model can't
# quite decide on). Insertion order matters: on a tolerance tie the earlier
# entry wins.
GESTURES: dict[Pattern, Gesture] = {
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

# Shape-based gestures that cannot be expressed as a finger pattern alone.
OK_SIGN: Gesture = ("OK", "\U0001F44C")            # 👌
THUMBS_DOWN: Gesture = ("Thumbs down", "\U0001F44E")  # 👎

# If the exact pattern is not catalogued, accept the closest one as long as
# it differs in at most this many fingers.
GESTURE_TOLERANCE = 1

# Thumb-tip to index-tip distance, normalized by palm size, below which the
# two are treated as touching (OK sign).
DEFAULT_PINCH_DIST = 0.45

# How far below the wrist (normalized image y) the thumb tip must sit for a
# lone extended thumb to read as thumbs down rather than thumbs up.
THUMBS_DOWN_MARGIN = 0.02


def hamming(a: Pattern, b: Pattern) -> int:
    """Number of fingers whose extended state differs between two patterns."""
    return sum(x != y for x, y in zip(a, b, strict=True))


def lookup_gesture(pattern: Pattern, tolerance: int = GESTURE_TOLERANCE) -> Gesture | None:
    """Find the catalogued gesture for ``pattern``.

    Exact matches win. Otherwise the nearest catalogued pattern is returned
    if it is within ``tolerance`` fingers; ties go to the earlier entry.
    """
    if pattern in GESTURES:
        return GESTURES[pattern]

    best: Gesture | None = None
    best_dist = tolerance + 1
    for ref, value in GESTURES.items():
        dist = hamming(ref, pattern)
        if dist < best_dist:
            best_dist = dist
            best = value
    return best if best_dist <= tolerance else None


def classify_gesture(
    landmarks,
    pattern: Pattern,
    pinch_dist: float = DEFAULT_PINCH_DIST,
) -> Gesture | None:
    """Pick the best gesture for a hand. Returns ``(label, emoji)`` or ``None``.

    Order: shape-based special cases (OK, thumbs down) first, then the
    pattern catalogue via :func:`lookup_gesture`.
    """
    wrist = lm_xy(landmarks[WRIST])
    thumb_tip = lm_xy(landmarks[THUMB_TIP])
    index_tip = lm_xy(landmarks[INDEX_TIP])

    # OK sign: thumb tip touching index tip, with middle/ring/pinky out.
    pinch = np.linalg.norm(thumb_tip - index_tip) / palm_size(landmarks)
    if pinch < pinch_dist and pattern[2] and pattern[3] and pattern[4]:
        return OK_SIGN

    # Thumbs down: only the thumb extended and its tip below the wrist
    # (larger y in image space). Otherwise the catalogue says thumbs up.
    if pattern == (1, 0, 0, 0, 0) and thumb_tip[1] > wrist[1] + THUMBS_DOWN_MARGIN:
        return THUMBS_DOWN

    return lookup_gesture(pattern)
