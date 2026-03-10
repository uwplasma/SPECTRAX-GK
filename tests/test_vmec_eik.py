from __future__ import annotations

from pathlib import Path
import sys

import pytest

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
    write_gx_vmec_geometry_input,
)


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
            gx_repo=str(tmp_path / "gx"),
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


def test_write_gx_vmec_geometry_input_emits_expected_contract(tmp_path: Path) -> None:
    cfg = _vmec_runtime_cfg(tmp_path)
    request = build_gx_vmec_geometry_request(cfg)
    input_path = tmp_path / "gx_vmec.toml"
    assert cfg.geometry.vmec_file is not None

    write_gx_vmec_geometry_input(request, input_path)

    text = input_path.read_text(encoding="utf-8")
    assert "[Dimensions]" in text
    assert "ntheta = 32" in text
    assert '[Geometry]' in text
    assert 'geo_option = "vmec"' in text
    assert f'vmec_file = "{Path(cfg.geometry.vmec_file).resolve()}"' in text
    assert "torflux = 0.64" in text
    assert "npol = 2.0" in text
    assert "x0 =" not in text
    assert "[species]" in text
    assert 'type = ["ion", "electron"]' in text


def test_build_gx_vmec_geometry_request_leaves_x0_unset_for_gx_defaults(tmp_path: Path) -> None:
    cfg = _vmec_runtime_cfg(tmp_path)

    request = build_gx_vmec_geometry_request(cfg)

    assert request.x0 is None


def test_build_gx_vmec_geometry_request_resolves_relative_vmec_file_from_gx_repo(
    tmp_path: Path,
) -> None:
    gx_repo = tmp_path / "gx"
    vmec_path = gx_repo / "benchmarks" / "nonlinear" / "w7x" / "wout_w7x.nc"
    vmec_path.parent.mkdir(parents=True, exist_ok=True)
    vmec_path.write_text("stub", encoding="utf-8")
    cfg = _vmec_runtime_cfg(tmp_path)
    cfg = RuntimeConfig(
        grid=cfg.grid,
        time=cfg.time,
        geometry=GeometryConfig(
            model="vmec",
            vmec_file="benchmarks/nonlinear/w7x/wout_w7x.nc",
            geometry_file=cfg.geometry.geometry_file,
            torflux=cfg.geometry.torflux,
            npol=cfg.geometry.npol,
            alpha=cfg.geometry.alpha,
            gx_repo=str(gx_repo),
        ),
        init=cfg.init,
        species=cfg.species,
        physics=cfg.physics,
        normalization=cfg.normalization,
        collisions=cfg.collisions,
        terms=cfg.terms,
    )

    request = build_gx_vmec_geometry_request(cfg)

    assert request.vmec_file == str(vmec_path.resolve())


def test_build_gx_vmec_geometry_request_expands_env_vmec_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    vmec_path = tmp_path / "wout_env.nc"
    vmec_path.write_text("stub", encoding="utf-8")
    monkeypatch.setenv("HSX_VMEC_FILE", str(vmec_path))
    cfg = _vmec_runtime_cfg(tmp_path)
    cfg = RuntimeConfig(
        grid=cfg.grid,
        time=cfg.time,
        geometry=GeometryConfig(
            model="vmec",
            vmec_file="$HSX_VMEC_FILE",
            geometry_file=cfg.geometry.geometry_file,
            torflux=cfg.geometry.torflux,
            npol=cfg.geometry.npol,
            alpha=cfg.geometry.alpha,
            gx_repo=cfg.geometry.gx_repo,
        ),
        init=cfg.init,
        species=cfg.species,
        physics=cfg.physics,
        normalization=cfg.normalization,
        collisions=cfg.collisions,
        terms=cfg.terms,
    )

    request = build_gx_vmec_geometry_request(cfg)

    assert request.vmec_file == str(vmec_path.resolve())


def test_generate_runtime_vmec_eik_invokes_gx_script_and_creates_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _vmec_runtime_cfg(tmp_path, geometry_file=str(tmp_path / "geom.eik.nc"))
    assert cfg.geometry.gx_repo is not None
    assert cfg.geometry.geometry_file is not None
    gx_script = Path(cfg.geometry.gx_repo) / "geometry_modules" / "pyvmec" / "gx_geo_vmec.py"
    gx_script.parent.mkdir(parents=True, exist_ok=True)
    gx_script.write_text("# stub", encoding="utf-8")

    called: dict[str, object] = {}

    def _fake_run(cmd, check, capture_output, text, cwd):  # type: ignore[no-untyped-def]
        called["cmd"] = cmd
        called["cwd"] = cwd
        out = Path(cmd[-1])
        out.write_text("generated", encoding="utf-8")

        class _Result:
            returncode = 0
            stdout = "ok"
            stderr = ""

        return _Result()

    monkeypatch.setattr("spectraxgk.vmec_eik.subprocess.run", _fake_run)

    out = generate_runtime_vmec_eik(cfg)

    assert out == Path(cfg.geometry.geometry_file).resolve()
    assert out.exists()
    assert called["cmd"] is not None
    cmd = called["cmd"]
    assert isinstance(cmd, list)
    assert cmd[0] == sys.executable
    assert str(gx_script) in cmd


def test_generate_runtime_vmec_eik_uses_configured_python_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _vmec_runtime_cfg(tmp_path, geometry_file=str(tmp_path / "geom.eik.nc"))
    cfg = RuntimeConfig(
        grid=cfg.grid,
        time=cfg.time,
        geometry=GeometryConfig(
            model="vmec",
            vmec_file=cfg.geometry.vmec_file,
            geometry_file=cfg.geometry.geometry_file,
            torflux=cfg.geometry.torflux,
            npol=cfg.geometry.npol,
            alpha=cfg.geometry.alpha,
            gx_repo=cfg.geometry.gx_repo,
            gx_python="python3",
        ),
        init=cfg.init,
        species=cfg.species,
        physics=cfg.physics,
        normalization=cfg.normalization,
        collisions=cfg.collisions,
        terms=cfg.terms,
    )
    assert cfg.geometry.gx_repo is not None
    gx_script = Path(cfg.geometry.gx_repo) / "geometry_modules" / "pyvmec" / "gx_geo_vmec.py"
    gx_script.parent.mkdir(parents=True, exist_ok=True)
    gx_script.write_text("# stub", encoding="utf-8")

    called: dict[str, object] = {}

    def _fake_run(cmd, check, capture_output, text, cwd):  # type: ignore[no-untyped-def]
        called["cmd"] = cmd
        out = Path(cmd[-1])
        out.write_text("generated", encoding="utf-8")

        class _Result:
            returncode = 0
            stdout = "ok"
            stderr = ""

        return _Result()

    monkeypatch.setattr("spectraxgk.vmec_eik.subprocess.run", _fake_run)

    out = generate_runtime_vmec_eik(cfg, force=True)

    assert out.exists()
    cmd = called["cmd"]
    assert isinstance(cmd, list)
    assert cmd[0] == "python3"


def test_generate_runtime_vmec_eik_regenerates_explicit_output_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_path = tmp_path / "geom.eik.nc"
    out_path.write_text("existing", encoding="utf-8")
    cfg = _vmec_runtime_cfg(tmp_path, geometry_file=str(out_path))
    assert cfg.geometry.gx_repo is not None
    gx_script = Path(cfg.geometry.gx_repo) / "geometry_modules" / "pyvmec" / "gx_geo_vmec.py"
    gx_script.parent.mkdir(parents=True, exist_ok=True)
    gx_script.write_text("# stub", encoding="utf-8")

    called: dict[str, object] = {}

    def _fake_run(cmd, check, capture_output, text, cwd):  # type: ignore[no-untyped-def]
        called["cmd"] = cmd
        called["cwd"] = cwd
        out = Path(cmd[-1])
        out.write_text("generated", encoding="utf-8")

        class _Result:
            returncode = 0
            stdout = "ok"
            stderr = ""

        return _Result()

    monkeypatch.setattr("spectraxgk.vmec_eik.subprocess.run", _fake_run)

    out = generate_runtime_vmec_eik(cfg)

    assert out == out_path.resolve()
    assert out.read_text(encoding="utf-8") == "generated"
    assert called["cmd"] is not None


def test_build_gx_vmec_geometry_request_requires_torflux(tmp_path: Path) -> None:
    cfg = _vmec_runtime_cfg(tmp_path)
    cfg_bad = RuntimeConfig(
        grid=cfg.grid,
        time=cfg.time,
        geometry=GeometryConfig(
            model="vmec",
            vmec_file=cfg.geometry.vmec_file,
            torflux=None,
            gx_repo=cfg.geometry.gx_repo,
        ),
        init=cfg.init,
        species=cfg.species,
        physics=cfg.physics,
        normalization=cfg.normalization,
        terms=cfg.terms,
    )

    with pytest.raises(ValueError, match="geometry.torflux"):
        build_gx_vmec_geometry_request(cfg_bad)
