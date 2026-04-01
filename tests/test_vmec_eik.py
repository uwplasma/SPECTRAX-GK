from __future__ import annotations

import os
from pathlib import Path
import sys
import pytest
import numpy as np
from unittest.mock import MagicMock
from dataclasses import replace

from spectraxgk.config import GeometryConfig, GridConfig, InitializationConfig, TimeConfig
from spectraxgk.runtime_config import (
    RuntimeConfig,
    RuntimeNormalizationConfig,
    RuntimePhysicsConfig,
    RuntimeSpeciesConfig,
)
from spectraxgk.vmec_eik import (
    build_gx_vmec_geometry_request,
    generate_runtime_vmec_eik,
)
from spectraxgk.from_gx.vmec import internal_vmec_backend_available


def _vmec_runtime_cfg(tmp_path: Path, *, geometry_file: str | None = None) -> RuntimeConfig:
    vmec_path = tmp_path / "wout_test.nc"
    vmec_path.write_text("stub", encoding="utf-8")
    return RuntimeConfig(
        grid=GridConfig(
            Nx=1,
            Ny=8,
            Nz=32,
            Lx=62.8,
            Ly=62.8,
            boundary="linked",
            y0=10.0,
            ntheta=32,
            nperiod=1,
        ),
        time=TimeConfig(t_max=1.0, dt=0.1, method="rk4", use_diffrax=False, fixed_dt=True),
        geometry=GeometryConfig(
            model="vmec",
            vmec_file=str(vmec_path),
            geometry_file=geometry_file,
            torflux=0.64,
            npol=2.0,
            alpha=0.1,
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


def test_build_gx_vmec_geometry_request_creates_expected_request(tmp_path: Path) -> None:
    cfg = _vmec_runtime_cfg(tmp_path)
    request = build_gx_vmec_geometry_request(cfg)
    
    assert request.vmec_file == str(Path(cfg.geometry.vmec_file).resolve())
    assert request.torflux == 0.64
    assert request.npol == 2.0
    assert request.ntheta == 32
    assert request.alpha == 0.1
    assert request.y0 == pytest.approx(10.0)
    assert request.z == (1.0, -1.0) # Ion + adiabatic electron


def test_build_gx_vmec_geometry_request_expands_env_vmec_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    vmec_path = tmp_path / "wout_env.nc"
    vmec_path.write_text("stub", encoding="utf-8")
    monkeypatch.setenv("HSX_VMEC_FILE", str(vmec_path))
    cfg = _vmec_runtime_cfg(tmp_path)
    cfg = replace(cfg, geometry=replace(cfg.geometry, vmec_file="$HSX_VMEC_FILE"))

    request = build_gx_vmec_geometry_request(cfg)

    assert request.vmec_file == str(vmec_path.resolve())


def test_generate_runtime_vmec_eik_invokes_internal_generator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_path = tmp_path / "geom.eik.nc"
    cfg = _vmec_runtime_cfg(tmp_path, geometry_file=str(out_path))
    
    mock_gen = MagicMock(return_value=out_path.resolve())
    monkeypatch.setattr("spectraxgk.vmec_eik.generate_vmec_eik_internal", mock_gen)
    monkeypatch.setattr("spectraxgk.vmec_eik.internal_vmec_backend_available", lambda: True)

    out = generate_runtime_vmec_eik(cfg)

    assert out == out_path.resolve()
    assert mock_gen.called
    _, kwargs = mock_gen.call_args
    request = kwargs["request"]
    assert request.ntheta == 32
    assert request.vmec_file == str(Path(cfg.geometry.vmec_file).resolve())


def test_internal_vmec_backend_available_detects_env_provided_booz_xform_jax(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pkg_root = tmp_path / "booz_xform_jax_checkout"
    pkg_dir = pkg_root / "src" / "booz_xform_jax"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "__init__.py").write_text("class Booz_xform:\n    pass\n", encoding="utf-8")

    for name in ("booz_xform_jax", "booz_xform"):
        sys.modules.pop(name, None)
    original_sys_path = list(sys.path)
    monkeypatch.setenv("BOOZ_XFORM_JAX_PATH", str(pkg_root))
    monkeypatch.delenv("GX_BOOZ_XFORM_JAX_PATH", raising=False)

    try:
        assert internal_vmec_backend_available() is True
    finally:
        sys.path[:] = original_sys_path
        for name in ("booz_xform_jax", "booz_xform"):
            sys.modules.pop(name, None)
