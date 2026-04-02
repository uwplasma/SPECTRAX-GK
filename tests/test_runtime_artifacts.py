from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from spectraxgk.analysis import ModeSelection
from spectraxgk.diagnostics import GXDiagnostics
from spectraxgk.runtime import RuntimeLinearResult, RuntimeNonlinearResult
from spectraxgk.runtime_artifacts import write_runtime_linear_artifacts, write_runtime_nonlinear_artifacts


def test_write_runtime_linear_artifacts_writes_bundle(tmp_path: Path) -> None:
    result = RuntimeLinearResult(
        ky=0.2,
        gamma=0.3,
        omega=-0.4,
        selection=ModeSelection(ky_index=1, kx_index=2, z_index=3),
        t=np.asarray([0.1, 0.2, 0.3]),
        signal=np.asarray([1.0, 2.0, 4.0]),
        state=np.zeros((1, 2, 3), dtype=np.complex64),
    )

    paths = write_runtime_linear_artifacts(tmp_path / "linear_run", result)

    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    assert summary["kind"] == "linear"
    assert summary["gamma"] == 0.3
    assert summary["selection"]["ky_index"] == 1
    assert Path(paths["timeseries"]).exists()
    assert Path(paths["state"]).exists()


def test_write_runtime_nonlinear_artifacts_preserves_csv_target(tmp_path: Path) -> None:
    diag = GXDiagnostics(
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
