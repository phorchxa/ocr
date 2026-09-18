from itertools import product

import pytest

from fingercount.gestures import (
    GESTURES,
    OK_SIGN,
    THUMBS_DOWN,
    classify_gesture,
    hamming,
    lookup_gesture,
)
from tests.conftest import LM, build_hand


def test_hamming_counts_differing_fingers():
    assert hamming((1, 1, 0, 0, 0), (1, 1, 0, 0, 0)) == 0
    assert hamming((1, 1, 0, 0, 0), (0, 1, 0, 0, 1)) == 2
    assert hamming((0, 0, 0, 0, 0), (1, 1, 1, 1, 1)) == 5


@pytest.mark.parametrize(("pattern", "expected"), list(GESTURES.items()))
def test_lookup_exact_match_for_every_catalogued_pattern(pattern, expected):
    assert lookup_gesture(pattern) == expected


def test_lookup_falls_back_to_nearest_pattern():
    # (0,0,1,1,1) is not catalogued; it is one finger away from "Four".
    assert lookup_gesture((0, 0, 1, 1, 1)) == GESTURES[(0, 1, 1, 1, 1)]


def test_lookup_tie_goes_to_earlier_catalogue_entry():
    # (1,0,1,0,1) is one finger from both "Middle finger" (1,0,1,0,0) and
    # "Call me" (1,0,0,0,1); the catalogue lists "Middle finger" first.
    assert lookup_gesture((1, 0, 1, 0, 1))[0] == "Middle finger"


def test_lookup_respects_tolerance():
    assert lookup_gesture((0, 0, 1, 1, 1), tolerance=0) is None


def test_every_pattern_resolves_within_default_tolerance():
    # The catalogue plus a one-finger tolerance covers all 32 patterns, so
    # the UI never shows "(unknown)" for a stable detection.
    for pattern in product((0, 1), repeat=5):
        assert lookup_gesture(pattern) is not None, pattern


def test_classify_uses_catalogue_for_plain_patterns():
    hand = build_hand(index=True, middle=True)
    assert classify_gesture(hand, (0, 1, 1, 0, 0)) == GESTURES[(0, 1, 1, 0, 0)]


def test_classify_ok_sign_when_thumb_pinches_index():
    hand = build_hand(middle=True, ring=True, pinky=True)
    # Put the thumb tip on top of the index tip.
    hand[4] = LM(hand[8].x + 0.01, hand[8].y)
    assert classify_gesture(hand, (0, 0, 1, 1, 1)) == OK_SIGN


def test_classify_pinch_alone_is_not_ok_sign():
    hand = build_hand()
    hand[4] = LM(hand[8].x + 0.01, hand[8].y)
    # Middle/ring/pinky are folded, so this should fall through to the catalogue.
    assert classify_gesture(hand, (0, 0, 0, 0, 0)) == GESTURES[(0, 0, 0, 0, 0)]


def test_classify_thumbs_down_when_thumb_tip_below_wrist():
    hand = build_hand(thumb="splayed")
    hand[4] = LM(0.5, hand[0].y + 0.1)  # below the wrist in image space
    assert classify_gesture(hand, (1, 0, 0, 0, 0)) == THUMBS_DOWN


def test_classify_thumbs_up_when_thumb_tip_above_wrist():
    hand = build_hand(thumb="splayed")
    assert classify_gesture(hand, (1, 0, 0, 0, 0)) == GESTURES[(1, 0, 0, 0, 0)]
