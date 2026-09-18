"""Command-line parsing for the ``fingercount`` entry point."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from fingercount import __version__
from fingercount.config import AppConfig

_DEFAULTS = AppConfig()


def _confidence(text: str) -> float:
    value = float(text)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(f"must be between 0 and 1, got {text}")
    return value


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {text}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fingercount",
        description=(
            "Count extended fingers and show the matching gesture emoji from your webcam. "
            "Press q or Esc to quit, s to save a screenshot."
        ),
    )
    parser.add_argument(
        "--camera", type=int, default=_DEFAULTS.camera,
        help="capture device index (default: %(default)s)",
    )
    parser.add_argument(
        "--width", type=_positive_int, default=_DEFAULTS.width,
        help="requested capture width in pixels (default: %(default)s)",
    )
    parser.add_argument(
        "--height", type=_positive_int, default=_DEFAULTS.height,
        help="requested capture height in pixels (default: %(default)s)",
    )
    parser.add_argument(
        "--max-hands", type=int, choices=(1, 2), default=_DEFAULTS.max_hands,
        help="how many hands to track (default: %(default)s)",
    )
    parser.add_argument(
        "--detection-confidence", type=_confidence, default=_DEFAULTS.detection_confidence,
        metavar="0..1", help="minimum hand detection confidence (default: %(default)s)",
    )
    parser.add_argument(
        "--tracking-confidence", type=_confidence, default=_DEFAULTS.tracking_confidence,
        metavar="0..1", help="minimum landmark tracking confidence (default: %(default)s)",
    )
    parser.add_argument(
        "--screenshot-dir", type=Path, default=None, metavar="DIR",
        help="where the s key saves screenshots (default: current directory)",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> AppConfig:
    """Parse ``argv`` (or ``sys.argv[1:]``) into an :class:`AppConfig`."""
    args = build_parser().parse_args(argv)
    screenshot_dir = args.screenshot_dir if args.screenshot_dir is not None else Path.cwd()
    return AppConfig(
        camera=args.camera,
        width=args.width,
        height=args.height,
        max_hands=args.max_hands,
        detection_confidence=args.detection_confidence,
        tracking_confidence=args.tracking_confidence,
        screenshot_dir=screenshot_dir,
    )
