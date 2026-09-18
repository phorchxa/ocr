import dataclasses

import pytest

from fingercount.fingers import DEFAULT_THRESHOLDS, FingerThresholds, extended_fingers


def test_fist(hand):
    assert extended_fingers(hand()) == (0, 0, 0, 0, 0)


def test_pointing(hand):
    assert extended_fingers(hand(index=True)) == (0, 1, 0, 0, 0)


def test_peace(hand):
    assert extended_fingers(hand(index=True, middle=True)) == (0, 1, 1, 0, 0)


def test_open_palm(hand):
    open_hand = hand(thumb="splayed", index=True, middle=True, ring=True, pinky=True)
    assert extended_fingers(open_hand) == (1, 1, 1, 1, 1)


def test_thumb_tucked_across_palm_is_not_extended(hand):
    # Straight enough to pass the angle check, but too close to the index MCP.
    assert extended_fingers(hand(thumb="tucked"))[0] == 0


def test_thumb_bent_is_not_extended(hand):
    # Far from the palm, but sharply bent at the IP joint.
    assert extended_fingers(hand(thumb="bent"))[0] == 0


def test_thresholds_are_honored(hand):
    strict = FingerThresholds(straight_deg=185.0, thumb_straight_deg=185.0)
    open_hand = hand(thumb="splayed", index=True, middle=True, ring=True, pinky=True)
    assert extended_fingers(open_hand, strict) == (0, 0, 0, 0, 0)


def test_default_thresholds_are_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        DEFAULT_THRESHOLDS.straight_deg = 10.0  # type: ignore[misc]
