from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

from spectraxgk.config import GeometryConfig, GridConfig, InitializationConfig, TimeConfig
from spectraxgk.miller_eik import (
    build_gx_miller_geometry_request,
    generate_runtime_miller_eik,
    write_gx_miller_geometry_input,
)
from spectraxgk.runtime_config import RuntimeConfig, RuntimeNormalizationConfig, RuntimePhysicsConfig, RuntimeSpeciesConfig


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
            gx_repo=str(tmp_path / "gx"),
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


def test_write_gx_miller_geometry_input_emits_expected_contract(tmp_path: Path) -> None:
    cfg = _miller_runtime_cfg(tmp_path)
    request = build_gx_miller_geometry_request(cfg)
    input_path = tmp_path / "gx_miller.in"

    write_gx_miller_geometry_input(request, input_path)

    text = input_path.read_text(encoding="utf-8")
    assert "[Dimensions]" in text
    assert "ntheta = 24" in text
    assert "nperiod = 1" in text
    assert "[Domain]" in text
    assert 'boundary = "linked"' in text
    assert "y0 = 10.0" in text
    assert '[Geometry]' in text
    assert 'geo_option = "miller"' in text
    assert "rhoc = 0.5" in text
    assert "qinp = 1.4" in text
    assert "Rmaj = 2.77778" in text


def test_generate_runtime_miller_eik_invokes_gx_script_and_creates_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _miller_runtime_cfg(tmp_path, geometry_file=str(tmp_path / "geom.eiknc.nc"))
    assert cfg.geometry.gx_repo is not None
    assert cfg.geometry.geometry_file is not None
    gx_script = Path(cfg.geometry.gx_repo) / "geometry_modules" / "miller" / "gx_geo.py"
    gx_script.parent.mkdir(parents=True, exist_ok=True)
    gx_script.write_text("# stub", encoding="utf-8")

    called: dict[str, object] = {}

    def _fake_run(cmd, check, capture_output, text, cwd):  # type: ignore[no-untyped-def]
        called["cmd"] = cmd
        called["cwd"] = cwd
        input_path = Path(cmd[2])
        generated = input_path.with_suffix("").with_suffix(".eiknc.nc")
        generated.write_text("generated", encoding="utf-8")

        class _Result:
            returncode = 0
            stdout = "ok"
            stderr = ""

        return _Result()

    monkeypatch.setattr("spectraxgk.miller_eik.subprocess.run", _fake_run)

    out = generate_runtime_miller_eik(cfg)

    assert out == Path(cfg.geometry.geometry_file).resolve()
    assert out.exists()
    cmd = called["cmd"]
    assert isinstance(cmd, list)
    assert cmd[0] == sys.executable
    assert str(gx_script) in cmd


def test_generate_runtime_miller_eik_uses_configured_python_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _miller_runtime_cfg(tmp_path, geometry_file=str(tmp_path / "geom.eiknc.nc"))
    cfg = RuntimeConfig(
        grid=cfg.grid,
        time=cfg.time,
        geometry=GeometryConfig(
            model="miller",
            geometry_file=cfg.geometry.geometry_file,
            q=cfg.geometry.q,
            s_hat=cfg.geometry.s_hat,
            rhoc=cfg.geometry.rhoc,
            R0=cfg.geometry.R0,
            R_geo=cfg.geometry.R_geo,
            shift=cfg.geometry.shift,
            akappa=cfg.geometry.akappa,
            akappri=cfg.geometry.akappri,
            tri=cfg.geometry.tri,
            tripri=cfg.geometry.tripri,
            betaprim=cfg.geometry.betaprim,
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
    gx_script = Path(cfg.geometry.gx_repo) / "geometry_modules" / "miller" / "gx_geo.py"
    gx_script.parent.mkdir(parents=True, exist_ok=True)
    gx_script.write_text("# stub", encoding="utf-8")

    called: dict[str, object] = {}

    def _fake_run(cmd, check, capture_output, text, cwd):  # type: ignore[no-untyped-def]
        called["cmd"] = cmd
        input_path = Path(cmd[2])
        generated = input_path.with_suffix("").with_suffix(".eiknc.nc")
        generated.write_text("generated", encoding="utf-8")

        class _Result:
            returncode = 0
            stdout = "ok"
            stderr = ""

        return _Result()

    monkeypatch.setattr("spectraxgk.miller_eik.subprocess.run", _fake_run)

    out = generate_runtime_miller_eik(cfg, force=True)

    assert out.exists()
    cmd = called["cmd"]
    assert isinstance(cmd, list)
    assert cmd[0] == "python3"


def test_real_gx_miller_geometry_matches_clean_gx_output_if_repo_available() -> None:
    netcdf4 = pytest.importorskip("netCDF4")
    from spectraxgk.geometry import load_gx_geometry_netcdf

    gx_repo = Path("/Users/rogeriojorge/local/gx")
    if not gx_repo.exists():
        pytest.skip("local GX checkout unavailable")

    benchmark_in = gx_repo / "benchmarks" / "nonlinear" / "cyclone" / "cyclone_miller_adiabatic_electrons.in"
    benchmark_out = gx_repo / "benchmarks" / "nonlinear" / "cyclone" / "cyclone_miller_adiabatic_electrons_correct.out.nc"
    if not benchmark_in.exists() or not benchmark_out.exists():
        pytest.skip("clean GX Miller benchmark assets unavailable")

    cfg = RuntimeConfig(
        grid=GridConfig(Nx=192, Ny=64, Nz=24, Lx=176.25000339052914, Ly=62.8, boundary="linked", y0=28.2, ntheta=24, nperiod=1),
        time=TimeConfig(t_max=1.0, dt=0.1, method="rk3", use_diffrax=False, fixed_dt=True),
        geometry=GeometryConfig(
            model="miller",
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
            gx_repo=str(gx_repo),
        ),
        init=InitializationConfig(),
        species=(RuntimeSpeciesConfig(),),
        physics=RuntimePhysicsConfig(),
        normalization=RuntimeNormalizationConfig(),
    )

    out = generate_runtime_miller_eik(cfg, force=True)
    geom = load_gx_geometry_netcdf(out)
    root = netcdf4.Dataset(benchmark_out)
    try:
        g = root.groups["Geometry"]
        mapping = {
            "bmag": "bmag_profile",
            "gds2": "gds2_profile",
            "gds21": "gds21_profile",
            "gds22": "gds22_profile",
            "cvdrift": "cv_profile",
            "gbdrift": "gb_profile",
            "cvdrift0": "cv0_profile",
            "gbdrift0": "gb0_profile",
            "grho": "grho_profile",
        }
        for gname, aname in mapping.items():
            gx = np.asarray(g.variables[gname][:], dtype=float)
            sp = np.asarray(getattr(geom, aname), dtype=float)
            assert gx.shape == sp.shape
            assert np.allclose(sp, gx, rtol=1.0e-5, atol=1.0e-7), gname
    finally:
        root.close()
