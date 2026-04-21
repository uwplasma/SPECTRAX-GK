#!/usr/bin/env python3
"""Plot saved SPECTRAX-GK runtime outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from spectraxgk.plotting import plot_saved_output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Path to a saved runtime artifact bundle")
    parser.add_argument("--out", type=Path, default=None, help="Optional output figure path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out = plot_saved_output(args.path, out=args.out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
