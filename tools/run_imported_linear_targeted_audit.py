#!/usr/bin/env python3
"""Run targeted imported-linear audits one ky at a time with resumable per-ky caches."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import pandas as pd
from netCDF4 import Dataset


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gx", type=Path, required=True, help="Path to GX .out.nc file.")
    p.add_argument("--geometry-file", type=Path, required=True, help="Path to geometry file passed to compare_gx_imported_linear.py.")
    p.add_argument("--gx-input", type=Path, default=None, help="Optional GX input file.")
    p.add_argument("--out", type=Path, required=True, help="Combined CSV output.")
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("tools_out") / "imported_linear_cache",
        help="Directory for per-ky CSV cache artifacts.",
    )
    p.add_argument("--ky", type=float, nargs="*", default=None, help="Explicit ky values to run.")
    p.add_argument(
        "--max-kys",
        type=int,
        default=None,
        help="If set and --ky is omitted, only run the first N positive ky values from the GX file.",
    )
    p.add_argument("--Nl", type=int, default=None)
    p.add_argument("--Nm", type=int, default=None)
    p.add_argument("--tprim", type=float, default=3.0)
    p.add_argument("--fprim", type=float, default=1.0)
    p.add_argument("--tau-e", type=float, default=1.0, dest="tau_e")
    p.add_argument("--damp-ends-amp", type=float, default=0.1)
    p.add_argument("--damp-ends-widthfrac", type=float, default=1.0 / 8.0)
    p.add_argument("--mode-method", choices=("z_index", "max"), default="z_index")
    p.add_argument("--rel-floor-fraction", type=float, default=1.0e-2)
    p.add_argument(
        "--sample-step-stride",
        type=int,
        default=1,
        help="Subsample saved GX diagnostic samples by this stride inside compare_gx_imported_linear.py.",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="If set, only score the first N selected GX diagnostic samples per ky.",
    )
    p.add_argument("--reuse-cache", action="store_true", help="Reuse existing per-ky CSV rows if present.")
    return p


def _positive_ky(path: Path) -> list[float]:
    root = Dataset(path, "r")
    try:
        ky = [float(v) for v in root.groups["Grids"].variables["ky"][:] if float(v) > 0.0]
    finally:
        root.close()
    return ky


def _ky_tag(ky: float) -> str:
    return f"{float(ky):0.4f}".replace(".", "p")


def _combine_csvs(cache_dir: Path, ky_values: list[float], out: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for ky in ky_values:
        csv_path = cache_dir / f"ky_{_ky_tag(ky)}.csv"
        if not csv_path.exists():
            continue
        rows.append(pd.read_csv(csv_path))
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not df.empty and "ky" in df.columns:
        df = df.sort_values("ky").reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def main() -> None:
    args = build_parser().parse_args()
    here = Path(__file__).resolve().parent

    gx = args.gx.expanduser().resolve()
    geometry_file = args.geometry_file.expanduser().resolve()
    gx_input = None if args.gx_input is None else args.gx_input.expanduser().resolve()
    out = args.out.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.ky:
        ky_values = [float(v) for v in args.ky]
    else:
        ky_values = _positive_ky(gx)
        if args.max_kys is not None:
            ky_values = ky_values[: max(0, int(args.max_kys))]

    for ky in ky_values:
        cache_csv = cache_dir / f"ky_{_ky_tag(ky)}.csv"
        if args.reuse_cache and cache_csv.exists():
            _combine_csvs(cache_dir, ky_values, out)
            continue

        cmd = [
            sys.executable,
            str(here / "compare_gx_imported_linear.py"),
            "--gx",
            str(gx),
            "--geometry-file",
            str(geometry_file),
            "--out",
            str(cache_csv),
            "--cache-dir",
            str(cache_dir),
            "--ky",
            str(float(ky)),
            "--sample-step-stride",
            str(int(args.sample_step_stride)),
            "--tprim",
            str(float(args.tprim)),
            "--fprim",
            str(float(args.fprim)),
            "--tau-e",
            str(float(args.tau_e)),
            "--damp-ends-amp",
            str(float(args.damp_ends_amp)),
            "--damp-ends-widthfrac",
            str(float(args.damp_ends_widthfrac)),
            "--mode-method",
            str(args.mode_method),
            "--rel-floor-fraction",
            str(float(args.rel_floor_fraction)),
        ]
        if args.Nl is not None:
            cmd += ["--Nl", str(int(args.Nl))]
        if args.Nm is not None:
            cmd += ["--Nm", str(int(args.Nm))]
        if args.max_samples is not None:
            cmd += ["--max-samples", str(int(args.max_samples))]
        if args.reuse_cache:
            cmd += ["--reuse-cache"]
        if gx_input is not None:
            cmd += ["--gx-input", str(gx_input)]
        subprocess.run(cmd, check=True)
        _combine_csvs(cache_dir, ky_values, out)

    df = _combine_csvs(cache_dir, ky_values, out)
    print(df.to_string(index=False))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
