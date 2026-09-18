"""Small 2-D geometry helpers operating on MediaPipe-style landmarks.

A "landmark" here is any object with ``x`` and ``y`` attributes holding
normalized image coordinates in [0, 1]. This module deliberately imports
only numpy so it can be unit-tested without MediaPipe or OpenCV installed.
"""

from __future__ import annotations

import numpy as np

# Landmark indices from the MediaPipe hand model that the helpers rely on.
WRIST = 0
INDEX_MCP = 5
PINKY_MCP = 17


def lm_xy(lm) -> np.ndarray:
    """Return the (x, y) position of a landmark as a float32 array."""
    return np.array([lm.x, lm.y], dtype=np.float32)


def angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle in degrees at vertex ``b``, formed by segments b->a and b->c."""
    v1 = a - b
    v2 = c - b
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-9
    cosang = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


def palm_size(landmarks) -> float:
    """Characteristic palm length used to normalize other distances.

    Takes the larger of wrist->index-MCP and wrist->pinky-MCP so the value
    stays stable as the hand rotates, with a small floor to avoid division
    by zero on degenerate detections.
    """
    wrist = lm_xy(landmarks[WRIST])
    index_mcp = lm_xy(landmarks[INDEX_MCP])
    pinky_mcp = lm_xy(landmarks[PINKY_MCP])
    return float(
        max(np.linalg.norm(index_mcp - wrist), np.linalg.norm(pinky_mcp - wrist), 1e-6)
    )
