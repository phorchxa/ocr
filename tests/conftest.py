"""Shared helpers for building synthetic hands.

MediaPipe landmarks are objects with ``x``/``y`` in normalized image
coordinates (origin top-left, y grows downward). These builders lay out a
right hand pointing up the frame so tests can reason about geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest


@dataclass(frozen=True)
class LM:
    x: float
    y: float


WRIST = LM(0.5, 0.9)
# Row of MCP knuckles for index, middle, ring, pinky.
MCP_X = {"index": 0.44, "middle": 0.50, "ring": 0.56, "pinky": 0.62}
MCP_Y = 0.6
FINGER_BASE = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}


def _finger(x: float, straight: bool) -> list[LM]:
    """Return (mcp, pip, dip, tip) for one non-thumb finger."""
    if straight:
        return [LM(x, MCP_Y), LM(x, 0.45), LM(x, 0.37), LM(x, 0.30)]
    # Curled: tip folds back toward the knuckle, so the PIP angle is ~0 and
    # the tip is no farther from the wrist than the MCP.
    return [LM(x, MCP_Y), LM(x, 0.50), LM(x, 0.56), LM(x, 0.62)]


def _thumb(state: str) -> list[LM]:
    """Return (cmc, mcp, ip, tip) for the thumb.

    ``splayed``: straight and away from the palm (counts as extended).
    ``tucked``: fairly straight but lying across the palm (splay check fails).
    ``bent``: splayed away but sharply bent at the IP joint (angle check fails).
    """
    cmc = LM(0.40, 0.80)
    if state == "splayed":
        return [cmc, LM(0.30, 0.72), LM(0.22, 0.66), LM(0.14, 0.60)]
    if state == "tucked":
        return [cmc, LM(0.36, 0.72), LM(0.40, 0.68), LM(0.46, 0.66)]
    if state == "bent":
        return [cmc, LM(0.30, 0.72), LM(0.22, 0.66), LM(0.28, 0.60)]
    raise ValueError(state)


def build_hand(
    thumb: str = "tucked",
    index: bool = False,
    middle: bool = False,
    ring: bool = False,
    pinky: bool = False,
) -> list[LM]:
    """Build the 21-landmark list MediaPipe would return for this pose."""
    lms: list[LM | None] = [None] * 21
    lms[0] = WRIST
    for i, lm in enumerate(_thumb(thumb), start=1):
        lms[i] = lm
    for name, straight in (("index", index), ("middle", middle),
                           ("ring", ring), ("pinky", pinky)):
        for i, lm in enumerate(_finger(MCP_X[name], straight), start=FINGER_BASE[name]):
            lms[i] = lm
    assert all(lm is not None for lm in lms)
    return lms  # type: ignore[return-value]


@pytest.fixture
def hand():
    return build_hand
