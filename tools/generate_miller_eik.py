#!/usr/bin/env python3
"""Generate an imported Miller ``*.eiknc.nc`` file from a runtime TOML config."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from spectraxgk.io import load_runtime_from_toml
from spectraxgk.geometry.miller_eik import generate_runtime_miller_eik


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config", type=Path, help="Runtime TOML config with geometry.model='miller'"
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="Optional explicit output path"
    )
    parser.add_argument(
        "--geometry-helper-repo",
        "--gx-repo",
        dest="geometry_helper_repo",
        type=Path,
        default=None,
        help="Compatibility option retained for older invocations; the in-repo backend ignores it.",
    )
    parser.add_argument(
        "--geometry-helper-python",
        "--gx-python",
        dest="geometry_helper_python",
        type=str,
        default=None,
        help="Optional Python interpreter override recorded in the runtime geometry config.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even if the target already exists",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg, _ = load_runtime_from_toml(args.config)
    if args.geometry_helper_python is not None or args.geometry_helper_repo is not None:
        cfg = replace(
            cfg,
            geometry=replace(
                cfg.geometry,
                geometry_helper_python=args.geometry_helper_python,
                geometry_helper_repo=(
                    None
                    if args.geometry_helper_repo is None
                    else str(args.geometry_helper_repo)
                ),
            ),
        )
    out = generate_runtime_miller_eik(cfg, output_path=args.out, force=bool(args.force))
    print(out)


if __name__ == "__main__":
    main()
