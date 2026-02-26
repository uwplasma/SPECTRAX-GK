#!/usr/bin/env python3
"""Compare GX s-alpha geometry coefficients against SPECTRAX-GK geometry."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from spectraxgk.config import CycloneBaseCase
from spectraxgk.geometry import SAlphaGeometry


def _summary(name: str, ref: np.ndarray, test: np.ndarray) -> None:
    diff = test - ref
    rel = np.where(ref != 0.0, diff / ref, np.nan)
    max_abs = float(np.nanmax(np.abs(diff)))
    max_rel = float(np.nanmax(np.abs(rel)))
    print(f"{name:10s} max|diff|={max_abs:.3e} max|rel|={max_rel:.3e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gx-out", type=Path, required=True, help="Path to GX .out.nc")
    parser.add_argument("--drift-scale", type=float, default=1.0, help="Drift scaling for SPECTRAX geometry")
    args = parser.parse_args()

    root = Dataset(args.gx_out, "r")
    try:
        geom_grp = root.groups["Geometry"]
        grids = root.groups["Grids"]
    except KeyError as exc:
        root.close()
        raise ValueError(f"{args.gx_out} missing Geometry/Grids groups") from exc

    theta = np.asarray(grids.variables["theta"][:], dtype=float)
    bmag_ref = np.asarray(geom_grp.variables["bmag"][:], dtype=float)
    bgrad_ref = np.asarray(geom_grp.variables["bgrad"][:], dtype=float)
    gds2_ref = np.asarray(geom_grp.variables["gds2"][:], dtype=float)
    gds21_ref = np.asarray(geom_grp.variables["gds21"][:], dtype=float)
    gds22_ref = np.asarray(geom_grp.variables["gds22"][:], dtype=float)
    cv_ref = np.asarray(geom_grp.variables["cvdrift"][:], dtype=float)
    gb_ref = np.asarray(geom_grp.variables["gbdrift"][:], dtype=float)
    cv0_ref = np.asarray(geom_grp.variables["cvdrift0"][:], dtype=float)
    gb0_ref = np.asarray(geom_grp.variables["gbdrift0"][:], dtype=float)
    gradpar_ref = np.asarray(geom_grp.variables["gradpar"][:], dtype=float)
    root.close()

    cfg = CycloneBaseCase()
    geom_cfg = replace(cfg.geometry, drift_scale=float(args.drift_scale))
    geom = SAlphaGeometry.from_config(geom_cfg)

    theta_j = np.asarray(theta, dtype=float)
    bmag = np.asarray(geom.bmag(theta_j))
    bgrad = np.asarray(geom.bgrad(theta_j))
    gds2, gds21, gds22 = geom.metric_coeffs(theta_j)
    cv, gb, cv0, gb0 = geom.drift_coeffs(theta_j)
    gradpar = np.full_like(theta_j, geom.gradpar())

    print(f"GX file: {args.gx_out}")
    print(f"Geometry: q={geom.q} s_hat={geom.s_hat} eps={geom.epsilon} R0={geom.R0}")
    _summary("bmag", bmag_ref, bmag)
    _summary("bgrad", bgrad_ref, bgrad)
    _summary("gds2", gds2_ref, np.asarray(gds2))
    _summary("gds21", gds21_ref, np.asarray(gds21))
    _summary("gds22", gds22_ref, np.asarray(gds22))
    _summary("cvdrift", cv_ref, np.asarray(cv))
    _summary("gbdrift", gb_ref, np.asarray(gb))
    _summary("cvdrift0", cv0_ref, np.asarray(cv0))
    _summary("gbdrift0", gb0_ref, np.asarray(gb0))
    _summary("gradpar", gradpar_ref, gradpar)


if __name__ == "__main__":
    main()
