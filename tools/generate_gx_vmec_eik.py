#!/usr/bin/env python3
"""Generate a GX-compatible ``*.eik.nc`` file from a SPECTRAX runtime TOML."""

from __future__ import annotations

import argparse

from spectraxgk.io import load_runtime_from_toml
from spectraxgk.vmec_eik import generate_runtime_vmec_eik


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate GX VMEC geometry from a runtime TOML config.")
    parser.add_argument("--config", required=True, help="Path to a SPECTRAX runtime TOML config")
    parser.add_argument("--out", default=None, help="Optional explicit output path for the generated *.eik.nc file")
    parser.add_argument("--gx-repo", default=None, help="Optional GX repository path for gx_geo_vmec.py")
    parser.add_argument("--force", action="store_true", help="Regenerate even if the target file already exists")
    args = parser.parse_args()

    cfg, _ = load_runtime_from_toml(args.config)
    out = generate_runtime_vmec_eik(cfg, output_path=args.out, gx_repo=args.gx_repo, force=bool(args.force))
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
