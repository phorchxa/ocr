"""Per-finger extended/folded classification from hand landmarks.

Pure numpy: no MediaPipe or OpenCV imports, so it can be unit-tested with
synthetic landmark objects that only expose ``x`` and ``y``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fingercount.geometry import INDEX_MCP, WRIST, angle_deg, lm_xy, palm_size
from fingercount.gestures import Pattern

# Per-finger landmark triples (mcp, pip/ip, tip) used for angle checks.
# For the thumb we use (MCP=2, IP=3, TIP=4); IP is the analogue of PIP.
FINGER_JOINTS = {
    "thumb":  (2, 3, 4),
    "index":  (5, 6, 8),
    "middle": (9, 10, 12),
    "ring":   (13, 14, 16),
    "pinky":  (17, 18, 20),
}
FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")


@dataclass(frozen=True)
class FingerThresholds:
    """Tunable cut-offs for deciding whether a finger is extended.

    Angles are measured at the PIP (IP for the thumb) joint; closer to 180
    degrees means straighter. Defaults are loosened so detection survives
    noisy landmarks at angled hand orientations.
    """

    straight_deg: float = 150.0
    thumb_straight_deg: float = 148.0
    # Thumb tip to index-MCP distance, normalized by palm size, above which
    # the thumb counts as splayed away from the palm rather than tucked in.
    thumb_splay: float = 0.55
    # A non-thumb tip must be this many times farther from the wrist than
    # its MCP, which rules out fingers that are straight but folded down.
    tip_beyond_mcp: float = 1.05


DEFAULT_THRESHOLDS = FingerThresholds()


def extended_fingers(landmarks, thresholds: FingerThresholds = DEFAULT_THRESHOLDS) -> Pattern:
    """Return ``(thumb, index, middle, ring, pinky)`` where 1 = extended.

    Orientation-invariant: checks the angle at each finger's middle joint
    plus, for the non-thumb fingers, that the tip is farther from the wrist
    than the MCP. The thumb instead adds a splay check against the index
    MCP so a thumb tucked across the palm is not counted as extended.
    """
    wrist = lm_xy(landmarks[WRIST])
    index_mcp = lm_xy(landmarks[INDEX_MCP])
    palm = palm_size(landmarks)

    out = []
    for name in FINGER_ORDER:
        mcp_id, pip_id, tip_id = FINGER_JOINTS[name]
        mcp = lm_xy(landmarks[mcp_id])
        pip = lm_xy(landmarks[pip_id])
        tip = lm_xy(landmarks[tip_id])

        angle = angle_deg(mcp, pip, tip)

        if name == "thumb":
            splay = np.linalg.norm(tip - index_mcp) / palm
            extended = angle > thresholds.thumb_straight_deg and splay > thresholds.thumb_splay
        else:
            farther = np.linalg.norm(tip - wrist) > np.linalg.norm(mcp - wrist) * (
                thresholds.tip_beyond_mcp
            )
            extended = angle > thresholds.straight_deg and farther

        out.append(1 if extended else 0)

    return tuple(out)  # type: ignore[return-value]
