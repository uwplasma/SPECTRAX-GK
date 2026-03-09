"""Tests for unified runtime config and TOML loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from spectraxgk.io import load_runtime_from_toml
from spectraxgk.runtime_config import RuntimeConfig


def test_runtime_config_to_dict_contains_sections() -> None:
    cfg = RuntimeConfig()
    d = cfg.to_dict()
    assert set(d) == {
        "grid",
        "time",
        "geometry",
        "init",
        "species",
        "physics",
        "collisions",
        "normalization",
        "terms",
    }
    assert len(d["species"]) == 1


def test_runtime_defaults_match_gx_reference() -> None:
    cfg = RuntimeConfig()
    assert cfg.geometry.drift_scale == 1.0
    assert cfg.normalization.diagnostic_norm == "gx"
    assert cfg.normalization.flux_scale == 1.0
    assert cfg.collisions.p_hyper_m is None
    assert cfg.collisions.damp_ends_amp == pytest.approx(0.1)
    assert cfg.collisions.damp_ends_widthfrac == pytest.approx(0.125)


def test_load_runtime_from_toml_roundtrip(tmp_path: Path) -> None:
    toml = """
[[species]]
name = "ion"
charge = 1.0
mass = 1.0
density = 1.0
temperature = 1.0
tprim = 2.49
fprim = 0.8
kinetic = true

[[species]]
name = "electron"
charge = -1.0
mass = 0.00027248
density = 1.0
temperature = 1.0
tprim = 2.49
fprim = 0.8
kinetic = true

[grid]
Nx = 1
Ny = 8
Nz = 16

[physics]
electromagnetic = true
use_apar = true
adiabatic_electrons = false
beta = 0.2

[normalization]
contract = "kbm"
omega_star_scale = 0.7
"""
    path = tmp_path / "runtime.toml"
    path.write_text(toml, encoding="utf-8")
    cfg, data = load_runtime_from_toml(path)
    assert isinstance(data, dict)
    assert cfg.grid.Ny == 8
    assert cfg.physics.electromagnetic
    assert cfg.physics.use_apar
    assert not cfg.physics.adiabatic_electrons
    assert cfg.physics.beta == pytest.approx(0.2)
    assert cfg.normalization.contract == "kbm"
    assert cfg.normalization.omega_star_scale == pytest.approx(0.7)
    assert len(cfg.species) == 2
    assert cfg.species[1].charge == pytest.approx(-1.0)


def test_gx_aligned_kbm_runtime_examples_keep_end_damping_enabled() -> None:
    cfg_dir = Path(__file__).resolve().parents[1] / "examples" / "configs"
    paths = [
        cfg_dir / "runtime_kbm_nonlinear_gx.toml",
        cfg_dir / "runtime_kbm_nonlinear_gx_seed.toml",
        cfg_dir / "runtime_kbm_nonlinear_gx_short.toml",
        cfg_dir / "runtime_kbm_nonlinear_gx_short_lockin.toml",
        cfg_dir / "runtime_kbm_nonlinear_gx_t100.toml",
        cfg_dir / "runtime_kbm_nonlinear_gx_t100_nx4ny8_dt9e4.toml",
    ]
    for path in paths:
        cfg, _ = load_runtime_from_toml(path)
        assert cfg.terms.end_damping == pytest.approx(1.0), path.name


def test_load_runtime_from_toml_keeps_imported_geometry_fields(tmp_path: Path) -> None:
    toml = """
[[species]]
name = "ion"
charge = 1.0
mass = 1.0
density = 1.0
temperature = 1.0
tprim = 3.0
fprim = 1.0
kinetic = true

[grid]
Nx = 1
Ny = 12
Nz = 32

[geometry]
model = "gx-netcdf"
geometry_file = "/tmp/w7x.eik.nc"

[physics]
adiabatic_electrons = true
electromagnetic = false

[run]
ky = 0.3
Nl = 8
Nm = 12
solver = "gx_time"
"""
    path = tmp_path / "runtime_w7x.toml"
    path.write_text(toml, encoding="utf-8")

    cfg, data = load_runtime_from_toml(path)

    assert isinstance(data, dict)
    assert cfg.geometry.model == "gx-netcdf"
    assert cfg.geometry.geometry_file == "/tmp/w7x.eik.nc"
    assert cfg.physics.adiabatic_electrons is True


def test_w7x_imported_geometry_example_toml_loads() -> None:
    path = Path(__file__).resolve().parents[1] / "examples" / "configs" / "runtime_w7x_linear_imported_geometry.toml"

    cfg, data = load_runtime_from_toml(path)

    assert isinstance(data, dict)
    assert cfg.geometry.model == "gx-netcdf"
    assert cfg.geometry.geometry_file is not None
    assert cfg.physics.adiabatic_electrons is True
    assert cfg.normalization.diagnostic_norm == "gx"


def test_load_runtime_from_toml_accepts_desc_eik_geometry_alias(tmp_path: Path) -> None:
    toml = """
[geometry]
model = "desc-eik"
geometry_file = "/tmp/w7x-desc.eik.nc"
"""
    path = tmp_path / "runtime_desc.toml"
    path.write_text(toml, encoding="utf-8")

    cfg, _ = load_runtime_from_toml(path)

    assert cfg.geometry.model == "desc-eik"
    assert cfg.geometry.geometry_file == "/tmp/w7x-desc.eik.nc"
