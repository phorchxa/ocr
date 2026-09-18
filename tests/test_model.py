import io

import pytest

from fingercount import model


class _FakeResponse:
    """Minimal stand-in for the object urllib.request.urlopen returns."""

    def __init__(self, body: bytes, content_length: int | None):
        self._buf = io.BytesIO(body)
        self.headers = {}
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)

    def read(self, n: int = -1) -> bytes:
        return self._buf.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _patch_urlopen(monkeypatch, body: bytes, content_length: int | None):
    def fake_urlopen(url, context=None):
        return _FakeResponse(body, content_length)

    monkeypatch.setattr(model.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(model, "_ssl_context", lambda: None)


def test_ensure_model_returns_existing_file_without_downloading(tmp_path, monkeypatch):
    existing = tmp_path / "hand_landmarker.task"
    existing.write_bytes(b"already here")

    def boom(*args, **kwargs):
        raise AssertionError("download_model should not be called")

    monkeypatch.setattr(model, "download_model", boom)
    assert model.ensure_model(existing) == existing
    assert existing.read_bytes() == b"already here"


def test_download_writes_file_and_cleans_up_temp(tmp_path, monkeypatch):
    body = b"x" * (3 * model._CHUNK + 17)  # spans several chunks
    _patch_urlopen(monkeypatch, body, len(body))
    dest = tmp_path / "sub" / "hand_landmarker.task"

    assert model.download_model(dest, "https://example.invalid/model") == dest
    assert dest.read_bytes() == body
    assert list(tmp_path.rglob("*.part")) == []


def test_truncated_download_raises_and_leaves_nothing(tmp_path, monkeypatch):
    _patch_urlopen(monkeypatch, b"only part", content_length=1000)
    dest = tmp_path / "hand_landmarker.task"

    with pytest.raises(OSError, match="Incomplete download"):
        model.download_model(dest, "https://example.invalid/model")
    assert not dest.exists()
    assert list(tmp_path.iterdir()) == []


def test_download_without_content_length_is_accepted(tmp_path, monkeypatch):
    _patch_urlopen(monkeypatch, b"chunked body", content_length=None)
    dest = tmp_path / "hand_landmarker.task"
    model.download_model(dest, "https://example.invalid/model")
    assert dest.read_bytes() == b"chunked body"


def test_ensure_model_downloads_when_missing(tmp_path, monkeypatch, capsys):
    _patch_urlopen(monkeypatch, b"fresh", 5)
    dest = tmp_path / "hand_landmarker.task"
    assert model.ensure_model(dest, "https://example.invalid/model") == dest
    assert dest.read_bytes() == b"fresh"
    assert "Model ready" in capsys.readouterr().out
