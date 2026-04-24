from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from spectraxgk.diagnostics import SimulationDiagnostics
from spectraxgk.runtime_chunks import _format_duration, run_adaptive_gx_chunk_loop
from spectraxgk.terms.config import FieldState


def _diag(times: list[float]) -> SimulationDiagnostics:
    t = np.asarray(times, dtype=float)
    zeros = np.zeros_like(t)
    ones = np.ones_like(t)
    return SimulationDiagnostics(
        t=t,
        dt_t=np.full_like(t, 0.1),
        dt_mean=np.asarray(0.1),
        gamma_t=zeros,
        omega_t=zeros,
        Wg_t=ones,
        Wphi_t=2.0 * ones,
        Wapar_t=zeros,
        heat_flux_t=zeros,
        particle_flux_t=zeros,
        energy_t=3.0 * ones,
    )


def test_format_duration_compacts_minutes_and_hours() -> None:
    assert _format_duration(5.0) == "00:05"
    assert _format_duration(65.0) == "01:05"
    assert _format_duration(3665.0) == "1:01:05"


def test_run_adaptive_gx_chunk_loop_reports_wall_eta(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = iter([0.0, 0.0, 10.0, 10.0, 10.0, 25.0, 25.0])
    monkeypatch.setattr("spectraxgk.runtime_chunks.time.perf_counter", lambda: next(clock))

    messages: list[str] = []
    chunks = iter(
        [
            (
                np.asarray([0.5, 1.0]),
                _diag([0.5, 1.0]),
                np.asarray([1.0]),
                FieldState(phi=np.asarray([1.0 + 0.0j])),
            ),
            (
                np.asarray([0.25, 0.5]),
                _diag([0.25, 0.5]),
                np.asarray([2.0]),
                FieldState(phi=np.asarray([2.0 + 0.0j])),
            ),
        ]
    )

    result = run_adaptive_gx_chunk_loop(
        integrate_chunk=lambda _show_progress: next(chunks),
        t_max=1.5,
        chunk_steps=16,
        label="nonlinear",
        show_progress=True,
        status_callback=messages.append,
    )

    assert messages[0] == "starting adaptive nonlinear integration in chunks of 16 steps up to t_max=1.5"
    assert "progress= 66.7%" in messages[1]
    assert "chunk_wall=00:10" in messages[1]
    assert "elapsed=00:10" in messages[1]
    assert "eta=00:05" in messages[1]
    assert "progress=100.0%" in messages[2]
    assert "eta=00:00" in messages[2]
    np.testing.assert_allclose(np.asarray(result.diagnostics.t), np.asarray([0.5, 1.0, 1.25, 1.5]))
    np.testing.assert_allclose(np.asarray(result.state), np.asarray([2.0]))
    np.testing.assert_allclose(np.asarray(result.fields.phi), np.asarray([2.0 + 0.0j]))


def test_run_adaptive_gx_chunk_loop_rejects_stalled_time_progress() -> None:
    with pytest.raises(RuntimeError, match="made no time-step progress"):
        run_adaptive_gx_chunk_loop(
            integrate_chunk=lambda _show_progress: (
                np.asarray([0.0]),
                _diag([0.0]),
                np.asarray([0.0]),
                FieldState(phi=np.asarray([0.0 + 0.0j])),
            ),
            t_max=1.0,
            chunk_steps=8,
            label="test",
        )


def test_run_adaptive_gx_chunk_loop_rejects_nonfinite_diagnostics() -> None:
    bad = replace(_diag([0.5]), Wphi_t=np.asarray([np.nan]))

    with pytest.raises(RuntimeError, match=r"non-finite diagnostics in Wphi_t at sample 0"):
        run_adaptive_gx_chunk_loop(
            integrate_chunk=lambda _show_progress: (
                np.asarray([0.5]),
                bad,
                np.asarray([0.0]),
                FieldState(phi=np.asarray([0.0 + 0.0j])),
            ),
            t_max=1.0,
            chunk_steps=8,
            label="test",
        )
