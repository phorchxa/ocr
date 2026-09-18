"""Locating and downloading the MediaPipe hand landmarker model file."""

from __future__ import annotations

import os
import ssl
import tempfile
import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_FILENAME = "hand_landmarker.task"

# Default location is the repository root, one level above this package.
# Override with the FINGERCOUNT_MODEL_PATH environment variable.
DEFAULT_MODEL_PATH = Path(
    os.environ.get(
        "FINGERCOUNT_MODEL_PATH",
        Path(__file__).resolve().parent.parent / MODEL_FILENAME,
    )
)

_CHUNK = 1 << 16


def _ssl_context() -> ssl.SSLContext:
    """TLS context using certifi's bundle when available, else system certs."""
    try:
        import certifi
    except ImportError:  # pragma: no cover - depends on the environment
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def download_model(dest: Path | str, url: str = MODEL_URL) -> Path:
    """Download the model to ``dest`` atomically.

    Bytes are streamed into a ``.part`` temp file in the same directory and
    renamed into place only after the whole body has arrived, so an
    interrupted or truncated download never leaves a corrupt model behind.
    Raises ``OSError`` if the server's Content-Length does not match.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=dest.name + ".", suffix=".part", dir=dest.parent)
    tmp = Path(tmp_name)
    try:
        with (
            os.fdopen(fd, "wb") as out,
            urllib.request.urlopen(url, context=_ssl_context()) as resp,
        ):
            expected = resp.headers.get("Content-Length")
            written = 0
            while chunk := resp.read(_CHUNK):
                out.write(chunk)
                written += len(chunk)
        if expected is not None and written != int(expected):
            raise OSError(f"Incomplete download: got {written} of {expected} bytes")
        os.replace(tmp, dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return dest


def ensure_model(path: Path | str = DEFAULT_MODEL_PATH, url: str = MODEL_URL) -> Path:
    """Return the path to a usable model file, downloading it if missing."""
    path = Path(path)
    if path.exists():
        return path
    print(f"Downloading hand landmarker model to {path} ...")
    download_model(path, url)
    print("Model ready.")
    return path
