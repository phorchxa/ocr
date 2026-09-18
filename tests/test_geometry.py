import math

import numpy as np

from fingercount.geometry import angle_deg, lm_xy, palm_size
from tests.conftest import LM, build_hand


def test_lm_xy_returns_float32_pair():
    out = lm_xy(LM(0.25, 0.75))
    assert out.dtype == np.float32
    assert out.tolist() == [0.25, 0.75]


def test_angle_deg_right_angle():
    a, b, c = np.array([1.0, 0.0]), np.array([0.0, 0.0]), np.array([0.0, 1.0])
    assert math.isclose(angle_deg(a, b, c), 90.0, abs_tol=1e-4)


def test_angle_deg_straight_line():
    a, b, c = np.array([-1.0, 0.0]), np.array([0.0, 0.0]), np.array([1.0, 0.0])
    # The denominator carries a 1e-9 epsilon, so allow a hundredth of a degree.
    assert math.isclose(angle_deg(a, b, c), 180.0, abs_tol=1e-2)


def test_angle_deg_folded_back():
    a, b, c = np.array([0.0, 1.0]), np.array([0.0, 0.0]), np.array([0.0, 2.0])
    assert math.isclose(angle_deg(a, b, c), 0.0, abs_tol=1e-2)


def test_angle_deg_degenerate_points_do_not_raise():
    p = np.array([0.3, 0.3])
    assert math.isfinite(angle_deg(p, p, p))


def test_palm_size_uses_larger_knuckle_distance():
    hand = build_hand()
    wrist = lm_xy(hand[0])
    expected = max(
        np.linalg.norm(lm_xy(hand[5]) - wrist),
        np.linalg.norm(lm_xy(hand[17]) - wrist),
    )
    assert math.isclose(palm_size(hand), float(expected), rel_tol=1e-6)


def test_palm_size_has_floor_for_collapsed_hand():
    hand = [LM(0.5, 0.5)] * 21
    assert palm_size(hand) == 1e-6
