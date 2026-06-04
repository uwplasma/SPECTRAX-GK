#!/usr/bin/env python3
"""Patch missing scalar metadata in VMEC-JAX ``wout`` files.

Some VMEC-JAX-generated NetCDF files can carry valid Fourier geometry while
leaving scalar summary fields such as ``Aminor_p`` and ``aspect`` at zero.
SPECTRAX-GK runtime VMEC geometry generation needs a positive reference minor
radius. This utility fills only those scalar metadata fields from a simple LCFS
Fourier-boundary estimate and leaves the equilibrium Fourier coefficients
unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from netCDF4 import Dataset


def _scalar_variable(ds: Dataset, name: str) -> Any:
    if name not in ds.variables:
        raise KeyError(f"{ds.filepath()} is missing variable {name!r}")
    return ds.variables[name]


def _estimate_lcfs_metadata(ds: Dataset, *, ntheta: int, nphi: int) -> dict[str, float]:
    xm = np.asarray(_scalar_variable(ds, "xm")[:], dtype=float)
    xn = np.asarray(_scalar_variable(ds, "xn")[:], dtype=float)
    rmnc = np.asarray(_scalar_variable(ds, "rmnc")[-1, :], dtype=float)
    theta = np.linspace(0.0, 2.0 * np.pi, int(ntheta), endpoint=False)
    phi = np.linspace(0.0, 2.0 * np.pi, int(nphi), endpoint=False)
    phase = (
        xm[None, None, :] * theta[:, None, None]
        - xn[None, None, :] * phi[None, :, None]
    )
    r_lcfs = np.sum(rmnc[None, None, :] * np.cos(phase), axis=-1)
    r_min = float(np.nanmin(r_lcfs))
    r_max = float(np.nanmax(r_lcfs))
    aminor = 0.5 * (r_max - r_min)
    rmajor = 0.5 * (r_max + r_min)
    if not math.isfinite(aminor) or aminor <= 0.0:
        raise ValueError("could not infer a positive LCFS minor radius")
    if not math.isfinite(rmajor) or rmajor <= 0.0:
        rmajor = float(np.nanmean(r_lcfs))
    aspect = rmajor / aminor if aminor > 0.0 else float("nan")
    volume = 2.0 * math.pi**2 * rmajor * aminor**2
    return {
        "Aminor_p": float(aminor),
        "Rmajor_p": float(rmajor),
        "aspect": float(aspect),
        "volume_p": float(volume),
        "r_lcfs_min": r_min,
        "r_lcfs_max": r_max,
    }


def patch_wout(path: Path, *, ntheta: int = 128, nphi: int = 128, force: bool = False) -> dict[str, Any]:
    """Patch one WOUT file in place and return a JSON-safe report."""

    with Dataset(path, "r+") as ds:
        estimates = _estimate_lcfs_metadata(ds, ntheta=ntheta, nphi=nphi)
        before: dict[str, float | None] = {}
        after: dict[str, float | None] = {}
        patched: dict[str, float] = {}
        for name in ("Aminor_p", "Rmajor_p", "aspect", "volume_p"):
            var = _scalar_variable(ds, name)
            old = float(np.asarray(var[:]))
            before[name] = old
            new = float(estimates[name])
            should_patch = bool(force) or (not math.isfinite(old)) or abs(old) <= 0.0
            if should_patch:
                var[...] = new
                patched[name] = new
                after[name] = new
            else:
                after[name] = old
    return {
        "path": str(path),
        "kind": "vmec_jax_wout_metadata_patch",
        "claim_level": "metadata_patch_only_fourier_geometry_unchanged",
        "before": before,
        "after": after,
        "patched": patched,
        "estimates": estimates,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wout", nargs="+", type=Path)
    parser.add_argument("--ntheta", type=int, default=128)
    parser.add_argument("--nphi", type=int, default=128)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--out-json", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reports = [
        patch_wout(path, ntheta=int(args.ntheta), nphi=int(args.nphi), force=bool(args.force))
        for path in args.wout
    ]
    payload = {"kind": "vmec_jax_wout_metadata_patch_report", "reports": reports}
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
