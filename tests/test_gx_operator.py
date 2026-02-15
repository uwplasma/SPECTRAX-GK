"""GX-operator regression checks against Cyclone reference data."""

import numpy as np

from spectraxgk.benchmarks import load_cyclone_reference, run_cyclone_scan
from spectraxgk.config import CycloneBaseCase, GridConfig


def test_gx_operator_scan_relaxed():
    """GX operator should produce finite scans on a GX-matched grid."""
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
    scan = run_cyclone_scan(
        ky_values,
        cfg=cfg,
        steps=200,
        dt=0.02,
        tmin=2.0,
        method="imex",
        operator="gx",
    )
    ref = load_cyclone_reference()
    for ky, gamma, omega in zip(scan.ky, scan.gamma, scan.omega):
        idx = int(np.argmin(np.abs(ref.ky - ky)))
        assert np.isfinite(gamma)
        assert np.isfinite(omega)
        assert np.isclose(abs(gamma), ref.gamma[idx], rtol=20.0)
        assert np.isclose(abs(omega), ref.omega[idx], rtol=20.0)
