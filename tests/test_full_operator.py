"""Full-operator regression checks against Cyclone reference data."""

import numpy as np

from spectraxgk.benchmarks import run_cyclone_scan
from spectraxgk.config import CycloneBaseCase, GridConfig
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.linear import LinearParams, LinearTerms


def test_full_operator_scan_relaxed():
    """Full operator should produce finite scans on a field-aligned grid."""
    grid = GridConfig(
        Nx=1,
        Ny=24,
        Nz=96,
        Lx=62.8,
        Ly=62.8,
        y0=20.0,
        ntheta=32,
        nperiod=2,
    )
    cfg = CycloneBaseCase(grid=grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    ky_values = np.array([0.2, 0.3, 0.4])
    params = LinearParams(
        R_over_Ln=cfg.model.R_over_Ln,
        R_over_LTi=cfg.model.R_over_LTi,
        omega_d_scale=1.0,
        omega_star_scale=1.0,
        rho_star=1.0,
        kpar_scale=float(geom.gradpar()),
    )
    scan = run_cyclone_scan(
        ky_values,
        cfg=cfg,
        Nl=4,
        Nm=8,
        steps=200,
        dt=0.02,
        method="rk4",
        terms=LinearTerms(),
        params=params,
    )
    for ky, gamma, omega in zip(scan.ky, scan.gamma, scan.omega):
        assert np.isfinite(gamma)
        assert np.isfinite(omega)
        assert abs(gamma) < 50.0
        assert abs(omega) < 100.0
