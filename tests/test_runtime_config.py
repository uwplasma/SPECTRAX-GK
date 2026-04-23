"""Tests for unified runtime config and TOML loading."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

from spectraxgk.io import load_runtime_from_toml
from spectraxgk.runtime_config import RuntimeConfig


def _load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
        "expert",
        "output",
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

[expert]
fixed_mode = true
iky_fixed = 1
ikx_fixed = 0

[init]
init_file = "/tmp/restart.bin"
init_file_scale = 5.0
init_file_mode = "add"

[normalization]
contract = "kbm"
omega_star_scale = 0.7

[output]
path = "tools_out/runtime_case"
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
    assert cfg.expert.fixed_mode is True
    assert cfg.expert.iky_fixed == 1
    assert cfg.expert.ikx_fixed == 0
    assert cfg.init.init_file == str(Path("/tmp/restart.bin").resolve())
    assert cfg.init.init_file_scale == pytest.approx(5.0)
    assert cfg.init.init_file_mode == "add"
    assert cfg.output.path == str((tmp_path / "tools_out" / "runtime_case").resolve())
    assert len(cfg.species) == 2
    assert cfg.species[1].charge == pytest.approx(-1.0)


def test_gx_aligned_kbm_runtime_examples_keep_end_damping_enabled() -> None:
    cfg_dir = Path(__file__).resolve().parents[1] / "examples" / "nonlinear" / "axisymmetric"
    paths = [
        cfg_dir / "runtime_kbm_nonlinear.toml",
        cfg_dir / "runtime_kbm_nonlinear_seed.toml",
        cfg_dir / "runtime_kbm_nonlinear_short.toml",
        cfg_dir / "runtime_kbm_nonlinear_short_lockin.toml",
        cfg_dir / "runtime_kbm_nonlinear_t100.toml",
        cfg_dir / "runtime_kbm_nonlinear_t100_nx4ny8_dt9e4.toml",
    ]
    for path in paths:
        cfg, _ = load_runtime_from_toml(path)
        assert cfg.terms.end_damping == pytest.approx(1.0), path.name


def test_linear_axisymmetric_runtime_examples_keep_parity_collision_contract() -> None:
    cfg_dir = Path(__file__).resolve().parents[1] / "examples" / "linear" / "axisymmetric"
    expected = {
        "cyclone.toml": (1.0, 2.0, 1.0, 0.0),
        "etg.toml": (1.0, 2.0, 1.0, 0.0),
        "runtime_cyclone.toml": (1.0, 2.0, 1.0, 0.0),
        "runtime_etg.toml": (1.0, 2.0, 1.0, 0.0),
        "runtime_kaw.toml": (1.0, 2.0, 0.0, 1.0),
        "runtime_kbm.toml": (1.0, 2.0, 1.0, 0.0),
    }
    for name, (nu_h, nu_l, hyper_const, hyper_kz) in expected.items():
        cfg, _ = load_runtime_from_toml(cfg_dir / name)
        assert cfg.collisions.nu_hermite == pytest.approx(nu_h), name
        assert cfg.collisions.nu_laguerre == pytest.approx(nu_l), name
        assert cfg.collisions.hypercollisions_const == pytest.approx(hyper_const), name
        assert cfg.collisions.hypercollisions_kz == pytest.approx(hyper_kz), name


def test_etg_nonlinear_pilot_example_keeps_two_species_full_gk_contract() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "nonlinear"
        / "axisymmetric"
        / "runtime_etg_nonlinear.toml"
    )

    cfg, data = load_runtime_from_toml(path)

    assert isinstance(data, dict)
    assert len(cfg.species) == 2
    assert cfg.physics.linear is False
    assert cfg.physics.nonlinear is True
    assert cfg.physics.electrostatic is True
    assert cfg.physics.electromagnetic is False
    assert cfg.physics.adiabatic_ions is False
    assert cfg.physics.adiabatic_electrons is False
    assert cfg.grid.Lx == pytest.approx(1.25)
    assert cfg.init.gaussian_init is True
    assert cfg.init.init_single is False
    assert cfg.collisions.hypercollisions_const == pytest.approx(0.0)
    assert cfg.collisions.hypercollisions_kz == pytest.approx(1.0)
    assert data["run"]["ky"] == pytest.approx(5.0)
    assert cfg.output.path == str((path.parents[3] / "tools_out" / "etg_nonlinear_runtime").resolve())


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
solver = "explicit_time"
"""
    path = tmp_path / "runtime_w7x.toml"
    path.write_text(toml, encoding="utf-8")

    cfg, data = load_runtime_from_toml(path)

    assert isinstance(data, dict)
    assert cfg.geometry.model == "gx-netcdf"
    assert cfg.geometry.geometry_file == str(Path("/tmp/w7x.eik.nc").resolve())
    assert cfg.physics.adiabatic_electrons is True


def test_w7x_imported_geometry_example_toml_loads() -> None:
    path = Path(__file__).resolve().parents[1] / "examples" / "linear" / "non-axisymmetric" / "runtime_w7x_linear_imported_geometry.toml"

    cfg, data = load_runtime_from_toml(path)

    assert isinstance(data, dict)
    assert cfg.geometry.model == "gx-netcdf"
    assert cfg.geometry.geometry_file is not None
    assert cfg.init.init_field == "density"
    assert cfg.physics.adiabatic_electrons is True
    assert cfg.normalization.diagnostic_norm == "gx"


def test_w7x_nonlinear_imported_geometry_example_toml_loads() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "nonlinear"
        / "non-axisymmetric"
        / "runtime_w7x_nonlinear_imported_geometry.toml"
    )

    cfg, data = load_runtime_from_toml(path)

    assert isinstance(data, dict)
    assert cfg.geometry.model == "vmec-eik"
    assert cfg.geometry.geometry_file is not None
    assert cfg.physics.nonlinear is True
    assert cfg.physics.adiabatic_electrons is True
    assert cfg.physics.collisions is True
    assert cfg.terms.collisions == pytest.approx(1.0)
    assert cfg.terms.nonlinear == pytest.approx(1.0)
    assert "steps" not in data.get("run", {})
    assert cfg.output.path == str((path.parents[3] / "tools_out" / "w7x_nonlinear_imported_runtime").resolve())


def test_w7x_nonlinear_imported_geometry_builder_keeps_collision_contract() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "nonlinear"
        / "non-axisymmetric"
        / "w7x_nonlinear_imported_geometry.py"
    )
    mod = _load_module_from_path("w7x_nonlinear_imported_geometry", path)
    cfg = mod.build_w7x_nonlinear_cfg("/tmp/w7x.eik.nc", dt=0.1, t_max=200.0)
    assert cfg.physics.collisions is True
    assert cfg.terms.collisions == pytest.approx(1.0)
    assert cfg.terms.hypercollisions == pytest.approx(1.0)
    assert cfg.collisions.D_hyper == pytest.approx(0.05)


def test_hsx_nonlinear_vmec_geometry_example_toml_loads() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "nonlinear"
        / "non-axisymmetric"
        / "runtime_hsx_nonlinear_vmec_geometry.toml"
    )

    cfg, data = load_runtime_from_toml(path)

    assert isinstance(data, dict)
    assert cfg.geometry.model == "vmec"
    assert cfg.geometry.vmec_file is not None
    assert cfg.geometry.vmec_file == "$HSX_VMEC_FILE"
    assert cfg.geometry.gx_python is None
    assert cfg.geometry.torflux == pytest.approx(0.64)
    assert cfg.physics.nonlinear is True
    assert cfg.physics.adiabatic_electrons is True
    assert cfg.physics.collisions is True
    assert cfg.terms.collisions == pytest.approx(1.0)
    assert cfg.terms.nonlinear == pytest.approx(1.0)
    assert "steps" not in data.get("run", {})
    assert cfg.output.path == str((path.parents[3] / "tools_out" / "hsx_nonlinear_vmec_runtime").resolve())


def test_hsx_nonlinear_vmec_geometry_builder_keeps_collision_contract() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "nonlinear"
        / "non-axisymmetric"
        / "hsx_nonlinear_vmec_geometry.py"
    )
    mod = _load_module_from_path("hsx_nonlinear_vmec_geometry", path)
    cfg = mod.build_hsx_nonlinear_cfg(
        "/tmp/hsx.nc",
        geometry_file=None,
        gx_repo=None,
        gx_python=None,
        torflux=0.64,
        alpha=0.0,
        npol=1.0,
        dt=0.1,
        t_max=200.0,
    )
    assert cfg.physics.collisions is True
    assert cfg.terms.collisions == pytest.approx(1.0)
    assert cfg.terms.hypercollisions == pytest.approx(1.0)
    assert cfg.collisions.D_hyper == pytest.approx(0.05)


def test_hsx_nonlinear_vmec_wrapper_defaults_to_config_path(monkeypatch: pytest.MonkeyPatch) -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "nonlinear"
        / "non-axisymmetric"
        / "hsx_nonlinear_vmec_geometry.py"
    )
    mod = _load_module_from_path("hsx_nonlinear_vmec_geometry_main", path)

    captured: dict[str, object] = {}

    def fake_run_nonlinear_case(config_path, **kwargs):
        captured["config_path"] = Path(config_path)
        captured["kwargs"] = kwargs
        return 0

    monkeypatch.setattr(mod, "run_nonlinear_case", fake_run_nonlinear_case)
    monkeypatch.setattr(
        sys,
        "argv",
        ["hsx_nonlinear_vmec_geometry.py", "--steps", "200"],
    )

    rc = mod.main()

    assert rc == 0
    assert captured["config_path"] == mod.CONFIG
    assert captured["kwargs"]["steps"] == 200


def test_w7x_nonlinear_vmec_geometry_example_toml_loads() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "nonlinear"
        / "non-axisymmetric"
        / "runtime_w7x_nonlinear_vmec_geometry.toml"
    )

    cfg, data = load_runtime_from_toml(path)

    assert isinstance(data, dict)
    assert cfg.geometry.model == "vmec"
    assert cfg.geometry.vmec_file is not None
    assert cfg.geometry.vmec_file == "$W7X_VMEC_FILE"
    assert cfg.geometry.gx_python is None
    assert cfg.geometry.torflux == pytest.approx(0.64)
    assert cfg.physics.nonlinear is True
    assert cfg.physics.adiabatic_electrons is True
    assert cfg.physics.collisions is True
    assert cfg.terms.collisions == pytest.approx(1.0)
    assert "steps" not in data.get("run", {})
    assert cfg.output.path == str((path.parents[3] / "tools_out" / "w7x_nonlinear_vmec_runtime").resolve())


def test_load_runtime_from_toml_resolves_relative_runtime_paths_against_config_dir(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    toml = """
[geometry]
model = "vmec"
vmec_file = "../vmec/wout.nc"
geometry_file = "../geom/run.eik.nc"
torflux = 0.64

[init]
init_file = "../restart/state.bin"

[output]
path = "../out/run.out.nc"
restart_to_file = "../out/run.restart.nc"
restart_from_file = "../out/run.resume.nc"
"""
    path = cfg_dir / "runtime.toml"
    path.write_text(toml, encoding="utf-8")

    cfg, _ = load_runtime_from_toml(path)

    assert cfg.geometry.vmec_file == str((tmp_path / "vmec" / "wout.nc").resolve())
    assert cfg.geometry.geometry_file == str((tmp_path / "geom" / "run.eik.nc").resolve())
    assert cfg.init.init_file == str((tmp_path / "restart" / "state.bin").resolve())
    assert cfg.output.path == str((tmp_path / "out" / "run.out.nc").resolve())
    assert cfg.output.restart_to_file == str((tmp_path / "out" / "run.restart.nc").resolve())
    assert cfg.output.restart_from_file == str((tmp_path / "out" / "run.resume.nc").resolve())


def test_secondary_slab_example_toml_loads() -> None:
    path = Path(__file__).resolve().parents[1] / "examples" / "benchmarks" / "runtime_secondary_slab.toml"

    cfg, data = load_runtime_from_toml(path)

    assert isinstance(data, dict)
    assert cfg.geometry.model == "slab"
    assert cfg.geometry.s_hat == pytest.approx(1.0e-8)
    assert cfg.physics.linear is True
    assert cfg.physics.nonlinear is False
    assert cfg.physics.adiabatic_electrons is True


def test_cetg_reference_example_toml_loads() -> None:
    path = Path(__file__).resolve().parents[1] / "examples" / "nonlinear" / "axisymmetric" / "runtime_cetg_reference.toml"

    cfg, data = load_runtime_from_toml(path)

    assert isinstance(data, dict)
    assert cfg.geometry.model == "slab"
    assert cfg.physics.reduced_model == "cetg"
    assert cfg.physics.adiabatic_ions is True
    assert cfg.physics.adiabatic_electrons is False
    assert cfg.physics.tau_fac == pytest.approx(1.0)
    assert cfg.physics.z_ion == pytest.approx(1.0)
    assert cfg.expert.dealias_kz is True


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
    assert cfg.geometry.geometry_file == str(Path("/tmp/w7x-desc.eik.nc").resolve())


def test_load_runtime_from_toml_accepts_vmec_gx_python(tmp_path: Path) -> None:
    toml = """
[geometry]
model = "vmec"
vmec_file = "/tmp/wout_test.nc"
torflux = 0.64
gx_python = "python3"
"""
    path = tmp_path / "runtime_vmec.toml"
    path.write_text(toml, encoding="utf-8")

    cfg, _ = load_runtime_from_toml(path)

    assert cfg.geometry.model == "vmec"
    assert cfg.geometry.vmec_file == str(Path("/tmp/wout_test.nc").resolve())
    assert cfg.geometry.gx_python == "python3"


def test_load_runtime_from_toml_accepts_miller_geometry_fields(tmp_path: Path) -> None:
    toml = """
[geometry]
model = "miller"
rhoc = 0.5
q = 1.4
s_hat = 0.8
R0 = 2.77778
R_geo = 2.77778
shift = 0.0
akappa = 1.0
akappri = 0.0
tri = 0.0
tripri = 0.0
betaprim = 0.0
gx_python = "python3"
"""
    path = tmp_path / "runtime_miller.toml"
    path.write_text(toml, encoding="utf-8")

    cfg, _ = load_runtime_from_toml(path)

    assert cfg.geometry.model == "miller"
    assert cfg.geometry.rhoc == pytest.approx(0.5)
    assert cfg.geometry.R_geo == pytest.approx(2.77778)
    assert cfg.geometry.akappa == pytest.approx(1.0)
    assert cfg.geometry.tripri == pytest.approx(0.0)
    assert cfg.geometry.gx_python == "python3"


def test_cyclone_nonlinear_gx_miller_example_toml_loads() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "nonlinear"
        / "axisymmetric"
        / "runtime_cyclone_nonlinear_miller.toml"
    )

    cfg, data = load_runtime_from_toml(path)

    assert isinstance(data, dict)
    assert cfg.geometry.model == "miller"
    assert cfg.geometry.q == pytest.approx(1.4)
    assert cfg.geometry.s_hat == pytest.approx(0.8)
    assert cfg.geometry.rhoc == pytest.approx(0.5)
    assert cfg.physics.nonlinear is True
    assert cfg.physics.adiabatic_electrons is True


def test_miller_zonal_response_example_uses_merlo_case_iii_contract() -> None:
    path = Path(__file__).resolve().parents[1] / "examples" / "benchmarks" / "runtime_miller_zonal_response.toml"

    cfg, data = load_runtime_from_toml(path)

    assert isinstance(data, dict)
    assert cfg.expert.source == "default"
    assert cfg.expert.phi_ext == pytest.approx(0.0)
    assert cfg.init.init_field == "density"
    assert cfg.init.init_amp == pytest.approx(1.0e-6)
    assert cfg.output.save_for_restart is True
    assert cfg.geometry.q == pytest.approx(1.389)
    assert cfg.geometry.s_hat == pytest.approx(0.751)
    assert cfg.geometry.akappa == pytest.approx(1.4723)
    assert cfg.geometry.tri == pytest.approx(-0.0070)
    assert cfg.geometry.shift == pytest.approx(-0.1569)
    assert cfg.grid.Nz == 32
    assert data["run"]["Nl"] == 4
    assert data["run"]["Nm"] == 24
    assert data["run"]["dt"] == pytest.approx(0.005)
    assert data["run"]["kx"] == pytest.approx(0.05)
    assert data["run"]["ky"] == pytest.approx(0.0)
