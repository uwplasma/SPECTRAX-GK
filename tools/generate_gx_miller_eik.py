#!/usr/bin/env python3
"""Generate a GX-compatible Miller ``*.eiknc.nc`` file from a runtime TOML config."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from spectraxgk.io import load_runtime_from_toml
from spectraxgk.miller_eik import generate_runtime_miller_eik


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Runtime TOML config with geometry.model='miller'")
    parser.add_argument("--out", type=Path, default=None, help="Optional explicit output path")
    parser.add_argument("--gx-repo", type=Path, default=None, help="Optional GX repository override")
    parser.add_argument("--gx-python", type=str, default=None, help="Optional Python interpreter override")
    parser.add_argument("--force", action="store_true", help="Regenerate even if the target already exists")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg, _ = load_runtime_from_toml(args.config)
    if args.gx_python is not None:
        cfg = replace(cfg, geometry=replace(cfg.geometry, gx_python=args.gx_python))
    out = generate_runtime_miller_eik(cfg, output_path=args.out, gx_repo=args.gx_repo, force=bool(args.force))
    print(out)


if __name__ == "__main__":
    main()
