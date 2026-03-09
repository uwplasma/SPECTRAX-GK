"""Tests for staged secondary-instability helpers."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from spectraxgk.config import GridConfig, InitializationConfig, TimeConfig
from spectraxgk.diagnostics import GXDiagnostics
from spectraxgk.runtime_config import (
    RuntimeConfig,
    RuntimeExpertConfig,
    RuntimePhysicsConfig,
    RuntimeSpeciesConfig,
    RuntimeTermsConfig,
)
from spectraxgk.secondary import (
    build_secondary_stage2_config,
    run_secondary_modes,
    run_secondary_seed,
    write_restart_state,
)


def _base_cfg() -> RuntimeConfig:
    return RuntimeConfig(
        species=(RuntimeSpeciesConfig(name="ion"),),
        grid=GridConfig(Nx=4, Ny=4, Nz=8, Lx=125.66370614359172, Ly=62.83185307179586, boundary="periodic", y0=10.0),
        time=TimeConfig(t_max=2.0, dt=1.0, method="sspx3", use_diffrax=False),
        init=InitializationConfig(init_field="density", init_amp=1.0, gaussian_init=False, init_single=True),
    )


def test_build_secondary_stage2_config_sets_restart_controls(tmp_path) -> None:
    cfg = _base_cfg()
    out = build_secondary_stage2_config(cfg, restart_file=tmp_path / "seed.bin")
    assert out.physics == RuntimePhysicsConfig(linear=False, nonlinear=True)
    assert out.terms == RuntimeTermsConfig(nonlinear=1.0)
    assert out.expert == RuntimeExpertConfig(fixed_mode=True, iky_fixed=1, ikx_fixed=0)
    assert out.init.init_file == str(tmp_path / "seed.bin")
    assert out.init.init_file_scale == 500.0
    assert out.init.init_file_mode == "add"
    assert out.init.init_single is False
    assert out.time.method == "sspx3"
    assert out.time.dt == 0.01


def test_write_restart_state_roundtrip(tmp_path) -> None:
    state = (np.arange(12, dtype=np.float32) + 1j * np.arange(12, dtype=np.float32)).astype(np.complex64)
    path = write_restart_state(tmp_path / "restart.bin", state)
    restored = np.fromfile(path, dtype=np.complex64)
    assert np.allclose(restored, state)


def test_run_secondary_modes_uses_requested_targets(monkeypatch) -> None:
    captured: list[tuple[float, float]] = []

    class _Result:
        def __init__(self) -> None:
            t = np.array([0.1], dtype=float)
            self.diagnostics = GXDiagnostics(
                t=t,
                dt_t=t,
                dt_mean=t[0],
                gamma_t=np.array([1.5]),
                omega_t=np.array([-0.25]),
                Wg_t=t * 0.0,
                Wphi_t=t * 0.0,
                Wapar_t=t * 0.0,
                heat_flux_t=t * 0.0,
                particle_flux_t=t * 0.0,
                energy_t=t * 0.0,
            )

    def _fake_runner(*args, **kwargs):
        captured.append((float(kwargs["ky_target"]), float(kwargs["kx_target"])))
        return _Result()

    monkeypatch.setattr("spectraxgk.secondary.run_runtime_nonlinear", _fake_runner)
    rows = run_secondary_modes(_base_cfg(), modes=((0.0, -0.05), (0.1, 0.05)), Nl=3, Nm=8)
    assert captured == [(0.0, -0.05), (0.1, 0.05)]
    assert rows[0].gamma == 1.5
    assert rows[1].omega == -0.25


def test_secondary_stage2_sideband_grows_on_short_window() -> None:
    cfg = _base_cfg()
    with TemporaryDirectory() as tmpdir:
        restart = Path(tmpdir) / "seed.bin"
        run_secondary_seed(cfg, restart_path=restart, ky_target=0.1, Nl=3, Nm=8)
        stage2 = build_secondary_stage2_config(cfg, restart_file=restart, t_max=1.0)
        row = run_secondary_modes(stage2, modes=((0.1, 0.05),), Nl=3, Nm=8, steps=100, sample_stride=10)[0]

    assert row.gamma > 1.0
