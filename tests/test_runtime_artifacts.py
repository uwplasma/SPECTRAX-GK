from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from spectraxgk.analysis import ModeSelection
from spectraxgk.config import GridConfig, TimeConfig
from spectraxgk.diagnostics import SimulationDiagnostics, ResolvedDiagnostics
from spectraxgk.runtime import RuntimeLinearResult, RuntimeNonlinearResult
from spectraxgk.runtime_config import RuntimeConfig, RuntimeOutputConfig
from spectraxgk.runtime_artifacts import (
    load_runtime_nonlinear_gx_diagnostics,
    run_runtime_nonlinear_with_artifacts,
    write_runtime_linear_artifacts,
    write_runtime_nonlinear_artifacts,
)


def test_write_runtime_linear_artifacts_writes_bundle(tmp_path: Path) -> None:
    result = RuntimeLinearResult(
        ky=0.2,
        gamma=0.3,
        omega=-0.4,
        selection=ModeSelection(ky_index=1, kx_index=2, z_index=3),
        t=np.asarray([0.1, 0.2, 0.3]),
        signal=np.asarray([1.0, 2.0, 4.0]),
        state=np.zeros((1, 2, 3), dtype=np.complex64),
        z=np.asarray([-1.0, 0.0, 1.0]),
        eigenfunction=np.asarray([0.5 + 0.0j, 1.0 + 0.2j, 0.5 + 0.1j]),
        fit_window_tmin=0.1,
        fit_window_tmax=0.3,
        fit_signal_used="phi",
    )

    paths = write_runtime_linear_artifacts(tmp_path / "linear_run", result)

    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    assert summary["kind"] == "linear"
    assert summary["gamma"] == 0.3
    assert summary["fit_window_tmin"] == 0.1
    assert summary["fit_window_tmax"] == 0.3
    assert summary["fit_signal_used"] == "phi"
    assert summary["has_eigenfunction"] is True
    assert summary["selection"]["ky_index"] == 1
    csv_lines = Path(paths["timeseries"]).read_text(encoding="utf-8").splitlines()
    assert csv_lines[0] == "t,signal_real,signal_imag,signal_abs"
    assert Path(paths["timeseries"]).exists()
    assert Path(paths["eigenfunction"]).exists()
    assert Path(paths["state"]).exists()


def test_write_runtime_linear_artifacts_splits_complex_signal_columns(tmp_path: Path) -> None:
    result = RuntimeLinearResult(
        ky=0.2,
        gamma=0.3,
        omega=-0.4,
        selection=ModeSelection(ky_index=1, kx_index=2, z_index=3),
        t=np.asarray([0.1, 0.2]),
        signal=np.asarray([1.0 + 2.0j, 3.0 + 4.0j]),
        state=None,
        fit_signal_used="phi",
    )

    paths = write_runtime_linear_artifacts(tmp_path / "linear_complex", result)

    rows = Path(paths["timeseries"]).read_text(encoding="utf-8").splitlines()
    assert rows[0] == "t,signal_real,signal_imag,signal_abs"
    data = np.loadtxt(paths["timeseries"], delimiter=",", skiprows=1)
    np.testing.assert_allclose(data[:, 1], np.asarray([1.0, 3.0]))
    np.testing.assert_allclose(data[:, 2], np.asarray([2.0, 4.0]))
    np.testing.assert_allclose(data[:, 3], np.abs(np.asarray([1.0 + 2.0j, 3.0 + 4.0j])))


def test_write_runtime_nonlinear_artifacts_preserves_csv_target(tmp_path: Path) -> None:
    diag = SimulationDiagnostics(
        t=np.asarray([0.1, 0.2]),
        dt_t=np.asarray([0.1, 0.1]),
        dt_mean=np.asarray(0.1),
        gamma_t=np.asarray([0.01, 0.02]),
        omega_t=np.asarray([0.03, 0.04]),
        Wg_t=np.asarray([1.0, 1.1]),
        Wphi_t=np.asarray([2.0, 2.1]),
        Wapar_t=np.asarray([0.5, 0.6]),
        heat_flux_t=np.asarray([3.0, 3.1]),
        particle_flux_t=np.asarray([4.0, 4.1]),
        energy_t=np.asarray([3.5, 3.8]),
        heat_flux_species_t=np.asarray([[3.0], [3.1]]),
        particle_flux_species_t=np.asarray([[4.0], [4.1]]),
        phi_mode_t=None,
    )
    result = RuntimeNonlinearResult(
        t=np.asarray([0.1, 0.2]),
        diagnostics=diag,
        state=np.zeros((2, 2), dtype=np.complex64),
        ky_selected=0.2,
        kx_selected=0.0,
    )

    csv_path = tmp_path / "diag.csv"
    paths = write_runtime_nonlinear_artifacts(csv_path, result)

    assert paths["diagnostics"] == str(csv_path)
    header = csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert "heat_flux_s0" in header
    assert "particle_flux_s0" in header
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    assert summary["kind"] == "nonlinear"
    assert summary["n_samples"] == 2
    assert Path(paths["state"]).exists()


def test_write_runtime_nonlinear_artifacts_writes_gx_netcdf_bundle(tmp_path: Path) -> None:
    netcdf4 = pytest.importorskip("netCDF4")
    Dataset = netcdf4.Dataset

    diag = SimulationDiagnostics(
        t=np.asarray([0.0, 0.1], dtype=float),
        dt_t=np.asarray([0.05, 0.05], dtype=float),
        dt_mean=np.asarray(0.05),
        gamma_t=np.asarray([0.0, 0.0], dtype=float),
        omega_t=np.asarray([0.0, 0.0], dtype=float),
        Wg_t=np.asarray([1.0, 1.1], dtype=float),
        Wphi_t=np.asarray([2.0, 2.1], dtype=float),
        Wapar_t=np.asarray([0.0, 0.0], dtype=float),
        heat_flux_t=np.asarray([3.0, 3.1], dtype=float),
        particle_flux_t=np.asarray([4.0, 4.1], dtype=float),
        energy_t=np.asarray([3.0, 3.2], dtype=float),
        heat_flux_species_t=np.asarray([[3.0], [3.1]], dtype=float),
        particle_flux_species_t=np.asarray([[4.0], [4.1]], dtype=float),
        turbulent_heating_t=np.asarray([8.0, 8.1], dtype=float),
        turbulent_heating_species_t=np.asarray([[8.0], [8.1]], dtype=float),
        phi_mode_t=None,
        resolved=ResolvedDiagnostics(
            Phi2_kxt=np.ones((2, 8), dtype=float),
            Phi2_kyt=np.ones((2, 8), dtype=float),
            Phi2_kxkyt=np.ones((2, 8, 8), dtype=float),
            Phi2_zt=np.ones((2, 6), dtype=float),
            Phi2_zonal_t=np.ones((2,), dtype=float),
            Phi2_zonal_kxt=np.ones((2, 8), dtype=float),
            Phi2_zonal_zt=np.ones((2, 6), dtype=float),
            Wg_kxst=np.ones((2, 1, 8), dtype=float),
            Wg_kyst=np.ones((2, 1, 8), dtype=float),
            Wg_kxkyst=np.ones((2, 1, 8, 8), dtype=float),
            Wg_zst=np.ones((2, 1, 6), dtype=float),
            Wg_lmst=np.ones((2, 1, 8, 4), dtype=float),
            Wphi_kxst=np.ones((2, 1, 8), dtype=float),
            Wphi_kyst=np.ones((2, 1, 8), dtype=float),
            Wphi_kxkyst=np.ones((2, 1, 8, 8), dtype=float),
            Wphi_zst=np.ones((2, 1, 6), dtype=float),
            Wapar_kxst=np.zeros((2, 1, 8), dtype=float),
            Wapar_kyst=np.zeros((2, 1, 8), dtype=float),
            Wapar_kxkyst=np.zeros((2, 1, 8, 8), dtype=float),
            Wapar_zst=np.zeros((2, 1, 6), dtype=float),
            HeatFlux_kxst=np.ones((2, 1, 8), dtype=float),
            HeatFlux_kyst=np.ones((2, 1, 8), dtype=float),
            HeatFlux_kxkyst=np.ones((2, 1, 8, 8), dtype=float),
            HeatFlux_zst=np.ones((2, 1, 6), dtype=float),
            HeatFluxES_kxst=np.full((2, 1, 8), 2.0, dtype=float),
            HeatFluxES_kyst=np.full((2, 1, 8), 2.0, dtype=float),
            HeatFluxES_kxkyst=np.full((2, 1, 8, 8), 2.0, dtype=float),
            HeatFluxES_zst=np.full((2, 1, 6), 2.0, dtype=float),
            HeatFluxApar_kxst=np.full((2, 1, 8), 3.0, dtype=float),
            HeatFluxApar_kyst=np.full((2, 1, 8), 3.0, dtype=float),
            HeatFluxApar_kxkyst=np.full((2, 1, 8, 8), 3.0, dtype=float),
            HeatFluxApar_zst=np.full((2, 1, 6), 3.0, dtype=float),
            HeatFluxBpar_kxst=np.full((2, 1, 8), 4.0, dtype=float),
            HeatFluxBpar_kyst=np.full((2, 1, 8), 4.0, dtype=float),
            HeatFluxBpar_kxkyst=np.full((2, 1, 8, 8), 4.0, dtype=float),
            HeatFluxBpar_zst=np.full((2, 1, 6), 4.0, dtype=float),
            ParticleFlux_kxst=np.ones((2, 1, 8), dtype=float),
            ParticleFlux_kyst=np.ones((2, 1, 8), dtype=float),
            ParticleFlux_kxkyst=np.ones((2, 1, 8, 8), dtype=float),
            ParticleFlux_zst=np.ones((2, 1, 6), dtype=float),
            ParticleFluxES_kxst=np.full((2, 1, 8), 5.0, dtype=float),
            ParticleFluxES_kyst=np.full((2, 1, 8), 5.0, dtype=float),
            ParticleFluxES_kxkyst=np.full((2, 1, 8, 8), 5.0, dtype=float),
            ParticleFluxES_zst=np.full((2, 1, 6), 5.0, dtype=float),
            ParticleFluxApar_kxst=np.full((2, 1, 8), 6.0, dtype=float),
            ParticleFluxApar_kyst=np.full((2, 1, 8), 6.0, dtype=float),
            ParticleFluxApar_kxkyst=np.full((2, 1, 8, 8), 6.0, dtype=float),
            ParticleFluxApar_zst=np.full((2, 1, 6), 6.0, dtype=float),
            ParticleFluxBpar_kxst=np.full((2, 1, 8), 7.0, dtype=float),
            ParticleFluxBpar_kyst=np.full((2, 1, 8), 7.0, dtype=float),
            ParticleFluxBpar_kxkyst=np.full((2, 1, 8, 8), 7.0, dtype=float),
            ParticleFluxBpar_zst=np.full((2, 1, 6), 7.0, dtype=float),
            TurbulentHeating_kxst=np.full((2, 1, 8), 8.0, dtype=float),
            TurbulentHeating_kyst=np.full((2, 1, 8), 8.0, dtype=float),
            TurbulentHeating_kxkyst=np.full((2, 1, 8, 8), 8.0, dtype=float),
            TurbulentHeating_zst=np.full((2, 1, 6), 8.0, dtype=float),
        ),
    )
    state = np.zeros((1, 4, 8, 8, 8, 6), dtype=np.complex64)
    fields = type("Fields", (), {"phi": np.zeros((8, 8, 6), dtype=np.complex64), "apar": None, "bpar": None})()
    result = RuntimeNonlinearResult(
        t=np.asarray([0.0, 0.1]),
        diagnostics=diag,
        fields=fields,
        state=state,
        ky_selected=0.2,
        kx_selected=0.0,
    )

    cfg = RuntimeConfig(
        grid=GridConfig(Nx=8, Ny=8, Nz=6, Lx=1.0, Ly=1.0),
        time=TimeConfig(gx_real_fft=True),
    )

    paths = write_runtime_nonlinear_artifacts(tmp_path / "probe.out.nc", result, cfg)

    assert Path(paths["out"]).exists()
    assert Path(paths["restart"]).exists()
    assert Path(paths["big"]).exists()

    with Dataset(paths["out"], "r") as root:
        assert set(root.groups) == {"Diagnostics", "Geometry", "Grids", "Inputs"}
        assert root.dimensions["kx"].size == 5
        assert root.dimensions["ky"].size == 3
        assert "Phi2_t" in root.groups["Diagnostics"].variables
        assert "Phi2_kxt" in root.groups["Diagnostics"].variables
        assert "Wg_st" in root.groups["Diagnostics"].variables
        assert "Wg_kyst" in root.groups["Diagnostics"].variables
        assert "Wg_lmst" in root.groups["Diagnostics"].variables
        assert "HeatFlux_st" in root.groups["Diagnostics"].variables
        np.testing.assert_allclose(root.groups["Diagnostics"].variables["HeatFluxES_st"][:], np.full((2, 1), 16.0))
        np.testing.assert_allclose(root.groups["Diagnostics"].variables["HeatFluxApar_st"][:], np.full((2, 1), 24.0))
        np.testing.assert_allclose(root.groups["Diagnostics"].variables["HeatFluxBpar_st"][:], np.full((2, 1), 32.0))
        np.testing.assert_allclose(root.groups["Diagnostics"].variables["TurbulentHeating_st"][:], np.full((2, 1), 64.0))
        assert "ParticleFluxBpar_kxkyst" in root.groups["Diagnostics"].variables
        assert "TurbulentHeating_kxkyst" in root.groups["Diagnostics"].variables

    with Dataset(paths["restart"], "r") as root:
        assert root.dimensions["Nkx"].size == 5
        assert root.dimensions["Nky"].size == 3
        assert root.variables["G"].shape[-1] == 2
        assert "time" in root.variables

    with Dataset(paths["big"], "r") as root:
        assert "Phi" in root.groups["Diagnostics"].variables
        assert "PhiXY" in root.groups["Diagnostics"].variables
        assert "Density" in root.groups["Diagnostics"].variables
        assert "Upar" in root.groups["Diagnostics"].variables
        assert "Tpar" in root.groups["Diagnostics"].variables
        assert "Tperp" in root.groups["Diagnostics"].variables
        assert "ParticleDensity" in root.groups["Diagnostics"].variables

    loaded = load_runtime_nonlinear_gx_diagnostics(paths["out"])
    np.testing.assert_allclose(np.asarray(loaded.Wg_t), np.asarray([1.0, 1.1], dtype=float))
    np.testing.assert_allclose(np.asarray(loaded.turbulent_heating_t), np.asarray([64.0, 64.0], dtype=float))
    assert loaded.resolved is not None
    assert loaded.resolved.HeatFluxApar_kxst is not None
    assert loaded.resolved.TurbulentHeating_kxst is not None


def test_run_runtime_nonlinear_with_artifacts_uses_restart_if_exists(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []
    out_path = tmp_path / "resume.out.nc"
    restart_path = tmp_path / "resume.restart.nc"
    restart_path.write_bytes(b"stub")

    cfg = RuntimeConfig(
        time=TimeConfig(dt=0.1, t_max=0.2),
        output=RuntimeOutputConfig(
            path=str(out_path),
            restart_if_exists=True,
            restart_with_perturb=True,
            restart_scale=0.25,
            restart_to_file=str(restart_path),
            restart_from_file=str(restart_path),
            nsave=1,
        ),
    )

    diag = SimulationDiagnostics(
        t=np.asarray([0.1]),
        dt_t=np.asarray([0.1]),
        dt_mean=np.asarray(0.1),
        gamma_t=np.asarray([0.0]),
        omega_t=np.asarray([0.0]),
        Wg_t=np.asarray([1.0]),
        Wphi_t=np.asarray([0.0]),
        Wapar_t=np.asarray([0.0]),
        heat_flux_t=np.asarray([0.0]),
        particle_flux_t=np.asarray([0.0]),
        energy_t=np.asarray([1.0]),
    )

    def _fake_run_runtime_nonlinear(run_cfg, **kwargs):
        calls.append(
            {
                "init_file": run_cfg.init.init_file,
                "init_file_mode": run_cfg.init.init_file_mode,
                "init_file_scale": run_cfg.init.init_file_scale,
                "steps": kwargs.get("steps"),
            }
        )
        return RuntimeNonlinearResult(
            t=np.asarray([0.1]),
            diagnostics=diag,
            state=np.zeros((1, 1, 1, 1, 1, 1), dtype=np.complex64),
            fields=type("Fields", (), {"phi": np.zeros((1, 1, 1), dtype=np.complex64), "apar": None, "bpar": None})(),
            ky_selected=0.2,
            kx_selected=0.0,
        )

    def _fake_write_runtime_nonlinear_artifacts(_out, _result, _cfg):
        restart_path.write_bytes(b"stub")
        return {"out": str(out_path), "restart": str(restart_path)}

    monkeypatch.setattr("spectraxgk.runtime_artifacts.run_runtime_nonlinear", _fake_run_runtime_nonlinear)
    monkeypatch.setattr("spectraxgk.runtime_artifacts.write_runtime_nonlinear_artifacts", _fake_write_runtime_nonlinear_artifacts)

    _result, _paths = run_runtime_nonlinear_with_artifacts(
        cfg,
        out=out_path,
        ky_target=0.2,
        steps=2,
        diagnostics=True,
    )

    assert calls[0]["init_file"] == str(restart_path)
    assert calls[0]["init_file_mode"] == "add"
    assert calls[0]["init_file_scale"] == 0.25
    assert calls[1]["init_file"] == str(restart_path)
    assert calls[1]["init_file_mode"] == "replace"
    assert calls[1]["init_file_scale"] == 1.0
