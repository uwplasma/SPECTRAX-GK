#!/usr/bin/env python3
"""Compare internal Miller geometry output against original GX numerics.

This script generates two ``*.eiknc.nc`` files from the same runtime TOML:
1) Original GX ``geometry_modules/miller/gx_geo.py``
2) SPECTRAX-GK internal Miller backend

It then reports per-variable error metrics.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib

import numpy as np


def _write_gx_miller_input(path: Path, request: object) -> None:
    text = "\n".join(
        [
            "debug = false",
            "",
            "[Dimensions]",
            f"ntheta = {int(request.ntheta)}",
            f"nperiod = {int(request.nperiod)}",
            "",
            "[Domain]",
            f"boundary = \"{str(request.boundary)}\"",
            f"y0 = {float(request.y0)!r}",
            "",
            "[Geometry]",
            'geo_option = "miller"',
            f"rhoc = {float(request.rhoc)!r}",
            f"qinp = {float(request.qinp)!r}",
            f"shat = {float(request.shat)!r}",
            f"Rmaj = {float(request.Rmaj)!r}",
            f"R_geo = {float(request.R_geo)!r}",
            f"shift = {float(request.shift)!r}",
            f"akappa = {float(request.akappa)!r}",
            f"akappri = {float(request.akappri)!r}",
            f"tri = {float(request.tri)!r}",
            f"tripri = {float(request.tripri)!r}",
            f"betaprim = {float(request.betaprim)!r}",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def _load_netcdf_vars(path: Path, names: list[str]) -> dict[str, np.ndarray]:
    from netCDF4 import Dataset

    out: dict[str, np.ndarray] = {}
    with Dataset(path, "r") as ds:
        for name in names:
            if name in ds.variables:
                out[name] = np.asarray(ds.variables[name][:], dtype=float)
    return out


def _metric(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: internal={a.shape}, gx={b.shape}")
    diff = np.abs(a - b)
    denom = np.maximum(np.abs(b), 1.0e-30)
    rel = diff / denom
    return float(np.max(diff)), float(np.mean(diff)), float(np.max(rel))


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare internal Miller backend vs GX output.")
    parser.add_argument("--config", required=True, help="Path to runtime TOML config (geometry.model must be miller).")
    parser.add_argument("--gx-repo", required=True, help="Path to GX repository.")
    parser.add_argument("--gx-python", default=sys.executable, help="Python executable used to run GX script.")
    parser.add_argument("--atol", type=float, default=5.0e-6, help="Maximum absolute error tolerance.")
    parser.add_argument("--rtol", type=float, default=5.0e-4, help="Maximum relative error tolerance.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root / "src") not in sys.path:
        sys.path.insert(0, str(repo_root / "src"))

    from spectraxgk.io import load_runtime_from_toml
    from spectraxgk.miller_eik import build_gx_miller_geometry_request, generate_runtime_miller_eik

    cfg, _ = load_runtime_from_toml(args.config)
    if str(cfg.geometry.model).strip().lower() != "miller":
        raise ValueError("Config must set geometry.model='miller' for this comparator")

    request = build_gx_miller_geometry_request(cfg)

    with tempfile.TemporaryDirectory(prefix="miller_parity_") as tmp:
        tmpdir = Path(tmp)
        gx_in = tmpdir / "gx_miller_geometry.in"
        gx_out = tmpdir / "gx_miller_geometry.eik.out"
        gx_nc = tmpdir / "gx_miller_geometry.eiknc.nc"
        internal_nc = tmpdir / "internal_miller.eiknc.nc"

        _write_gx_miller_input(gx_in, request)

        gx_script = Path(args.gx_repo).expanduser().resolve() / "geometry_modules" / "miller" / "gx_geo.py"
        proc = subprocess.run(
            [args.gx_python, str(gx_script), str(gx_in), str(gx_out)],
            cwd=str(gx_script.parent),
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            print("GX execution failed", file=sys.stderr)
            print(proc.stdout, file=sys.stderr)
            print(proc.stderr, file=sys.stderr)
            return 2
        if not gx_nc.exists():
            print(f"GX output not found: {gx_nc}", file=sys.stderr)
            return 2

        cfg_internal = replace(cfg, geometry=replace(cfg.geometry, geometry_backend="internal", geometry_file=str(internal_nc)))
        generate_runtime_miller_eik(cfg_internal, output_path=internal_nc, force=True)

        vars_to_compare = [
            "theta",
            "bmag",
            "gradpar",
            "grho",
            "aprime",
            "gds2",
            "gds21",
            "gds22",
            "gbdrift",
            "gbdrift0",
            "cvdrift",
            "cvdrift0",
            "jacob",
            "drhodpsi",
            "kxfac",
            "Rmaj",
            "q",
            "shat",
        ]

        gx_data = _load_netcdf_vars(gx_nc, vars_to_compare)
        in_data = _load_netcdf_vars(internal_nc, vars_to_compare)

        failed = False
        print("variable,max_abs,mean_abs,max_rel")
        for name in vars_to_compare:
            if name not in gx_data or name not in in_data:
                print(f"{name},MISSING,MISSING,MISSING")
                failed = True
                continue
            try:
                max_abs, mean_abs, max_rel = _metric(in_data[name], gx_data[name])
            except ValueError as exc:
                print(f"{name},SHAPE_MISMATCH,SHAPE_MISMATCH,{exc}")
                failed = True
                continue
            print(f"{name},{max_abs:.6e},{mean_abs:.6e},{max_rel:.6e}")
            if not (math.isfinite(max_abs) and math.isfinite(max_rel)):
                failed = True
            if max_abs > float(args.atol) and max_rel > float(args.rtol):
                failed = True

        print(f"gx_file={gx_nc}")
        print(f"internal_file={internal_nc}")
        return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
