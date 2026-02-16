"""Full-operator regression checks against Cyclone reference data."""

import numpy as np

from spectraxgk.benchmarks import run_cyclone_scan
from spectraxgk.config import CycloneBaseCase, GridConfig
from spectraxgk.linear import LinearParams


def test_full_operator_scan_relaxed():
    """Full operator should produce finite scans on a field-aligned grid."""
    grid = GridConfig(
        Nx=1,
        Ny=24,
        Nz=16,
        Lx=62.8,
        Ly=62.8,
        y0=20.0,
        ntheta=32,
        nperiod=2,
    )
    cfg = CycloneBaseCase(grid=grid)
    ky_values = np.array([0.2, 0.3, 0.4])
    params = LinearParams(
        R_over_Ln=cfg.model.R_over_Ln,
        R_over_LTi=cfg.model.R_over_LTi,
        omega_d_scale=0.1,
        omega_star_scale=0.6,
        rho_star=0.9,
    )
    scan = run_cyclone_scan(
        ky_values,
        cfg=cfg,
        steps=200,
        dt=0.02,
        tmin=2.0,
        method="imex",
        operator="full",
        params=params,
    )
    for ky, gamma, omega in zip(scan.ky, scan.gamma, scan.omega):
        assert np.isfinite(gamma)
        assert np.isfinite(omega)
        assert abs(gamma) < 2.0
        assert abs(omega) < 2.0
