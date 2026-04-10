from __future__ import annotations

import os
from pathlib import Path
import pytest
from unittest.mock import MagicMock

from spectraxgk.config import GeometryConfig, GridConfig, InitializationConfig, TimeConfig
from spectraxgk.miller_eik import (
    build_miller_geometry_request,
    build_gx_miller_geometry_request,
    generate_runtime_miller_eik,
)
from spectraxgk.from_gx.miller import _request_attr
from spectraxgk.runtime_config import (
    RuntimeConfig,
    RuntimeNormalizationConfig,
    RuntimePhysicsConfig,
    RuntimeSpeciesConfig,
)


def _miller_runtime_cfg(tmp_path: Path, *, geometry_file: str | None = None) -> RuntimeConfig:
    return RuntimeConfig(
        grid=GridConfig(
            Nx=32,
            Ny=16,
            Nz=24,
            Lx=62.8,
            Ly=62.8,
            boundary="linked",
            y0=10.0,
            ntheta=24,
            nperiod=1,
        ),
        time=TimeConfig(t_max=1.0, dt=0.1, method="rk3", use_diffrax=False, fixed_dt=True),
        geometry=GeometryConfig(
            model="miller",
            geometry_file=geometry_file,
            q=1.4,
            s_hat=0.8,
            rhoc=0.5,
            R0=2.77778,
            R_geo=2.77778,
            shift=0.0,
            akappa=1.0,
            akappri=0.0,
            tri=0.0,
            tripri=0.0,
            betaprim=0.0,
        ),
        init=InitializationConfig(init_field="density", init_amp=1.0e-6),
        species=(RuntimeSpeciesConfig(name="ion", charge=1.0, mass=1.0, tprim=2.49, fprim=0.8),),
        physics=RuntimePhysicsConfig(
            linear=False,
            nonlinear=True,
            adiabatic_electrons=True,
            tau_e=1.0,
            electrostatic=True,
            electromagnetic=False,
            beta=0.0,
            collisions=False,
        ),
        normalization=RuntimeNormalizationConfig(contract="cyclone", diagnostic_norm="gx"),
    )


def test_build_miller_geometry_request_creates_expected_request(tmp_path: Path) -> None:
    cfg = _miller_runtime_cfg(tmp_path)
    request = build_miller_geometry_request(cfg)
    
    assert request.q == 1.4
    assert request.s_hat == 0.8
    assert request.rhoc == 0.5
    assert request.ntheta == 24
    assert request.nperiod == 1


def test_build_gx_miller_geometry_request_alias_still_resolves(tmp_path: Path) -> None:
    cfg = _miller_runtime_cfg(tmp_path)
    request = build_gx_miller_geometry_request(cfg)

    assert request.q == 1.4


def test_generate_runtime_miller_eik_invokes_internal_generator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_path = tmp_path / "geom.eiknc.nc"
    cfg = _miller_runtime_cfg(tmp_path, geometry_file=str(out_path))
    
    mock_gen = MagicMock(return_value=out_path.resolve())
    monkeypatch.setattr("spectraxgk.miller_eik.generate_miller_eik_internal", mock_gen)
    monkeypatch.setattr("spectraxgk.miller_eik.internal_miller_backend_available", lambda: True)

    out = generate_runtime_miller_eik(cfg)

    assert out == out_path.resolve()
    assert mock_gen.called
    _, kwargs = mock_gen.call_args
    request = kwargs["request"]
    assert request.ntheta == 24
    assert request.q == 1.4


def test_internal_miller_request_attr_accepts_runtime_aliases() -> None:
    class Req:
        q = 1.4
        s_hat = 0.8

    req = Req()
    assert _request_attr(req, "qinp", "q") == 1.4
    assert _request_attr(req, "shat", "s_hat") == 0.8
