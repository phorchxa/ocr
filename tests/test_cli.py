from pathlib import Path

import pytest

from fingercount import __version__
from fingercount.cli import parse_args
from fingercount.config import AppConfig


def test_no_arguments_gives_defaults():
    assert parse_args([]) == AppConfig()


def test_all_options_override_defaults(tmp_path):
    cfg = parse_args([
        "--camera", "1",
        "--width", "640",
        "--height", "480",
        "--max-hands", "1",
        "--detection-confidence", "0.7",
        "--tracking-confidence", "0.3",
        "--screenshot-dir", str(tmp_path),
    ])
    assert cfg == AppConfig(
        camera=1, width=640, height=480, max_hands=1,
        detection_confidence=0.7, tracking_confidence=0.3, screenshot_dir=tmp_path,
    )
    assert isinstance(cfg.screenshot_dir, Path)


@pytest.mark.parametrize("argv", [
    ["--detection-confidence", "1.5"],
    ["--tracking-confidence", "-0.1"],
    ["--max-hands", "3"],
    ["--width", "0"],
    ["--height", "-5"],
    ["--camera", "abc"],
])
def test_invalid_values_exit_with_usage_error(argv, capsys):
    with pytest.raises(SystemExit) as exc:
        parse_args(argv)
    assert exc.value.code == 2
    assert "usage:" in capsys.readouterr().err


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        parse_args(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_help_mentions_keys(capsys):
    with pytest.raises(SystemExit):
        parse_args(["--help"])
    out = capsys.readouterr().out
    assert "--screenshot-dir" in out and "Esc" in out
