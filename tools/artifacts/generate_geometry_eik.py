#!/usr/bin/env python3
"""Generate imported-geometry EIK files from runtime TOML configurations."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from spectraxgk.geometry.miller_eik import generate_runtime_miller_eik
from spectraxgk.geometry.vmec_eik import generate_runtime_vmec_eik
from spectraxgk.workflows.runtime.toml import load_runtime_from_toml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="geometry", required=True)

    vmec = subcommands.add_parser(
        "vmec",
        help="Generate a VMEC-derived *.eik.nc file from a runtime TOML config.",
    )
    vmec.add_argument("--config", required=True, type=Path, help="Runtime TOML config")
    vmec.add_argument("--out", type=Path, default=None, help="Optional output path")
    vmec.add_argument(
        "--force", action="store_true", help="Regenerate even if the target exists"
    )

    miller = subcommands.add_parser(
        "miller",
        help="Generate an imported Miller *.eiknc.nc file from a runtime TOML config.",
    )
    miller.add_argument(
        "config",
        type=Path,
        nargs="?",
        help="Runtime TOML config; may also be supplied with --config.",
    )
    miller.add_argument(
        "--config",
        dest="config_option",
        type=Path,
        default=None,
        help="Runtime TOML config.",
    )
    miller.add_argument("--out", type=Path, default=None, help="Optional output path")
    miller.add_argument(
        "--geometry-helper-repo",
        type=Path,
        default=None,
        help="Optional external helper repository recorded in the geometry config.",
    )
    miller.add_argument(
        "--geometry-helper-python",
        type=str,
        default=None,
        help="Optional Python interpreter override recorded in the geometry config.",
    )
    miller.add_argument(
        "--force", action="store_true", help="Regenerate even if the target exists"
    )
    return parser


def _with_miller_helper_overrides(cfg, args: argparse.Namespace):
    if args.geometry_helper_python is None and args.geometry_helper_repo is None:
        return cfg
    return replace(
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


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.geometry == "vmec":
        cfg, _ = load_runtime_from_toml(args.config)
        out = generate_runtime_vmec_eik(
            cfg, output_path=args.out, force=bool(args.force)
        )
        print(out)
        return 0
    if args.geometry == "miller":
        config = args.config_option if args.config_option is not None else args.config
        if config is None:
            raise SystemExit("miller requires a runtime TOML config")
        cfg, _ = load_runtime_from_toml(config)
        cfg = _with_miller_helper_overrides(cfg, args)
        out = generate_runtime_miller_eik(
            cfg, output_path=args.out, force=bool(args.force)
        )
        print(out)
        return 0
    raise ValueError(f"unknown geometry subcommand: {args.geometry}")


if __name__ == "__main__":
    raise SystemExit(main())
