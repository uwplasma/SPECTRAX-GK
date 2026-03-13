from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from spectraxgk.config import GeometryConfig, GridConfig, InitializationConfig, TimeConfig
from spectraxgk.geometry import load_gx_geometry_netcdf
from spectraxgk.runtime_config import (
    RuntimeConfig,
    RuntimeNormalizationConfig,
    RuntimePhysicsConfig,
    RuntimeSpeciesConfig,
)
from spectraxgk.vmec_eik import generate_runtime_vmec_eik


@pytest.mark.integration
def test_vmec_roundtrip_gate_is_deterministic(tmp_path: Path) -> None:
    vmec_file = os.environ.get("SPECTRAXGK_VMEC_FILE", "").strip()
    if not vmec_file:
        pytest.skip("Set SPECTRAXGK_VMEC_FILE to enable VMEC roundtrip parity gate.")

    gx_repo = os.environ.get("SPECTRAXGK_GX_REPO", "").strip() or None

    cfg = RuntimeConfig(
        grid=GridConfig(
            Nx=1,
            Ny=8,
            Nz=64,
            Lx=62.8,
            Ly=62.8,
            boundary="linked",
            y0=10.0,
            ntheta=64,
            nperiod=1,
        ),
        time=TimeConfig(t_max=1.0, dt=0.1, method="rk4", use_diffrax=False, fixed_dt=True),
        geometry=GeometryConfig(
            model="vmec",
            vmec_file=vmec_file,
            geometry_file=None,
            torflux=0.64,
            npol=1.0,
            alpha=0.0,
            gx_repo=gx_repo,
        ),
        init=InitializationConfig(init_field="density", init_amp=1.0e-6),
        species=(RuntimeSpeciesConfig(name="ion", charge=1.0, mass=1.0, tprim=3.0, fprim=1.0),),
        physics=RuntimePhysicsConfig(
            linear=True,
            nonlinear=False,
            adiabatic_electrons=True,
            tau_e=1.0,
            electrostatic=True,
            electromagnetic=False,
            beta=0.0,
            collisions=False,
        ),
        normalization=RuntimeNormalizationConfig(contract="kinetic", diagnostic_norm="gx"),
    )

    out1 = tmp_path / "geom1.eik.nc"
    out2 = tmp_path / "geom2.eik.nc"
    generate_runtime_vmec_eik(cfg, output_path=out1, force=True)
    generate_runtime_vmec_eik(cfg, output_path=out2, force=True)

    g1 = load_gx_geometry_netcdf(out1)
    g2 = load_gx_geometry_netcdf(out2)

    # This is a determinism gate, not a physics gate: any drift here will break
    # VMEC-backed parity workflows in hard-to-debug ways.
    for name in (
        "theta",
        "bmag_profile",
        "gds2_profile",
        "gds21_profile",
        "gds22_profile",
        "cv_profile",
        "gb_profile",
        "jacobian_profile",
        "grho_profile",
    ):
        a = np.asarray(getattr(g1, name))
        b = np.asarray(getattr(g2, name))
        np.testing.assert_allclose(a, b, rtol=0.0, atol=0.0)
