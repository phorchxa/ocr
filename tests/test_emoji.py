import numpy as np

from fingercount import emoji
from fingercount.emoji import EmojiRenderer, composite_rgba


def _frame(h=10, w=10, value=100):
    return np.full((h, w, 3), value, dtype=np.uint8)


def _patch(h, w, rgba):
    p = np.zeros((h, w, 4), dtype=np.uint8)
    p[:, :] = rgba
    return p


def test_opaque_patch_replaces_pixels_and_swaps_rgb_to_bgr():
    frame = _frame()
    composite_rgba(frame, _patch(2, 2, (255, 0, 0, 255)), center=(5, 5))
    # Pure red in RGB becomes (0, 0, 255) in BGR.
    assert frame[4:6, 4:6].tolist() == [[[0, 0, 255]] * 2] * 2
    # Everything outside the 2x2 patch is untouched.
    untouched = np.ones((10, 10), dtype=bool)
    untouched[4:6, 4:6] = False
    assert (frame[untouched] == 100).all()


def test_transparent_patch_leaves_frame_unchanged():
    frame = _frame()
    composite_rgba(frame, _patch(4, 4, (255, 255, 255, 0)), center=(5, 5))
    assert (frame == 100).all()


def test_half_alpha_blends_to_midpoint():
    frame = _frame(value=0)
    composite_rgba(frame, _patch(2, 2, (200, 200, 200, 128)), center=(5, 5))
    # 200 * 128/255 ~= 100.4, truncated to 100.
    assert frame[4, 4].tolist() == [100, 100, 100]


def test_patch_hanging_off_top_left_is_clipped():
    frame = _frame()
    # 4x4 patch centered at (0, 0): only its bottom-right 2x2 lands in frame.
    composite_rgba(frame, _patch(4, 4, (0, 255, 0, 255)), center=(0, 0))
    assert (frame[0:2, 0:2] == [0, 255, 0]).all()
    assert (frame[2:, :] == 100).all()
    assert (frame[:, 2:] == 100).all()


def test_patch_hanging_off_bottom_right_is_clipped():
    frame = _frame()
    composite_rgba(frame, _patch(4, 4, (0, 0, 255, 255)), center=(10, 10))
    assert (frame[8:10, 8:10] == [255, 0, 0]).all()
    assert (frame[:8, :] == 100).all()
    assert (frame[:, :8] == 100).all()


def test_patch_fully_outside_frame_is_a_noop():
    frame = _frame()
    out = composite_rgba(frame, _patch(4, 4, (255, 255, 255, 255)), center=(50, 50))
    assert out is frame
    assert (frame == 100).all()


def test_renderer_without_font_is_a_noop(monkeypatch):
    monkeypatch.setattr(emoji, "FONT_PATHS", ())
    monkeypatch.setattr(emoji, "load_emoji_font", lambda *a, **k: (None, 0))
    r = EmojiRenderer()
    assert r.font is None
    frame = _frame()
    assert r.draw(frame, "\U0001F44D", (5, 5)) is frame
    assert (frame == 100).all()


def test_load_font_returns_none_for_missing_paths(tmp_path):
    font, size = emoji.load_emoji_font(paths=(str(tmp_path / "nope.ttf"),))
    assert font is None and size == 0
