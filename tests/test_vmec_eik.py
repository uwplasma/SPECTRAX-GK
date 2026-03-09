from __future__ import annotations

from pathlib import Path

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

    write_gx_vmec_geometry_input(request, input_path)

    text = input_path.read_text(encoding="utf-8")
    assert "[Dimensions]" in text
    assert "ntheta = 32" in text
    assert '[Geometry]' in text
    assert 'geo_option = "vmec"' in text
    assert f'vmec_file = "{Path(cfg.geometry.vmec_file).resolve()}"' in text
    assert "torflux = 0.64" in text
    assert "npol = 2.0" in text
    assert "[species]" in text
    assert 'type = ["ion", "electron"]' in text


def test_generate_runtime_vmec_eik_invokes_gx_script_and_creates_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _vmec_runtime_cfg(tmp_path, geometry_file=str(tmp_path / "geom.eik.nc"))
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
    assert str(gx_script) in cmd


def test_generate_runtime_vmec_eik_reuses_existing_output_without_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_path = tmp_path / "geom.eik.nc"
    out_path.write_text("existing", encoding="utf-8")
    cfg = _vmec_runtime_cfg(tmp_path, geometry_file=str(out_path))

    def _fail(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("subprocess.run should not be called when the output already exists")

    monkeypatch.setattr("spectraxgk.vmec_eik.subprocess.run", _fail)

    out = generate_runtime_vmec_eik(cfg)

    assert out == out_path.resolve()
    assert out.read_text(encoding="utf-8") == "existing"


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
