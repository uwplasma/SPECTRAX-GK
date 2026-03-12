"""Inspect legacy grouped GX cETG output files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spectraxgk.gx_legacy_output import load_gx_legacy_cetg_output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gx_nc", type=Path, help="Legacy GX cETG NetCDF file (e.g. cetg_smoke.nc)")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out = load_gx_legacy_cetg_output(args.gx_nc)
    summary = {
        "samples": int(out.time.size),
        "ky": int(out.ky.size),
        "kx": int(out.kx.size),
        "kz": int(out.kz.size),
        "W_shape": list(out.W.shape),
        "Phi2_shape": list(out.Phi2.shape),
        "qflux_shape": list(out.qflux.shape),
        "pflux_shape": list(out.pflux.shape),
        "t_min": float(out.time[0]) if out.time.size else 0.0,
        "t_max": float(out.time[-1]) if out.time.size else 0.0,
    }
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
