from __future__ import annotations

import numpy as np
import pytest

from spectraxgk.benchmark_scan import (
    ScanFitWindowPolicy,
    apply_auto_fit_scan_policy,
    indexed_float_value,
    indexed_scan_value,
    normalize_fit_signal,
    normalize_solver_key,
    resolve_scan_mode_method,
    scan_window_valid,
    should_use_ky_batch,
)
from spectraxgk.linear import LinearParams


def test_scan_policy_normalizes_keys_and_auto_fit_side_effects() -> None:
    assert normalize_solver_key(" Auto ") == "auto"
    assert normalize_solver_key(" gx_time ") == "gx_time"
    assert normalize_fit_signal(" Density ") == "density"
    assert apply_auto_fit_scan_policy(
        "auto", streaming_fit=True, mode_only=True
    ) == (False, False)
    assert apply_auto_fit_scan_policy(
        "phi", streaming_fit=True, mode_only=True
    ) == (True, True)

    with pytest.raises(ValueError, match="fit_signal"):
        normalize_fit_signal("bad")


def test_scan_policy_indexes_values_and_coerces_mode_method() -> None:
    assert indexed_float_value(None, 0) is None
    assert indexed_float_value([1, 2], 1) == pytest.approx(2.0)
    assert indexed_float_value(np.array([0.1, 0.2]), 0) == pytest.approx(0.1)
    assert indexed_scan_value(("a", "b"), 1) == "b"
    assert indexed_scan_value(np.array([3, 4]), 0) == 3
    assert indexed_scan_value(5, 9) == 5

    assert resolve_scan_mode_method("project", mode_only=True) == "z_index"
    assert resolve_scan_mode_method("max", mode_only=True) == "max"
    assert resolve_scan_mode_method("project", mode_only=False) == "project"


def test_scan_policy_window_and_batch_eligibility() -> None:
    t = np.linspace(0.0, 1.0, 5)
    assert scan_window_valid(t, 0.25, 0.75)
    assert not scan_window_valid(t, None, 0.75)
    assert not scan_window_valid(t, 0.2, 0.3)

    assert should_use_ky_batch(
        ky_batch=4, solver_key="time", dt=0.1, steps=10, tmin=None, tmax=None
    )
    assert not should_use_ky_batch(
        ky_batch=4,
        solver_key="krylov",
        dt=0.1,
        steps=10,
        tmin=None,
        tmax=None,
    )
    assert not should_use_ky_batch(
        ky_batch=4,
        solver_key="time",
        dt=np.array([0.1, 0.2]),
        steps=10,
        tmin=None,
        tmax=None,
    )
    with pytest.raises(ValueError, match="ky_batch"):
        should_use_ky_batch(
            ky_batch=0, solver_key="time", dt=0.1, steps=10, tmin=None, tmax=None
        )


def test_scan_fit_window_policy_recovers_synthetic_mode() -> None:
    dt = 0.1
    t = np.arange(120) * dt
    signal = np.exp((0.12 - 0.31j) * t)
    policy = ScanFitWindowPolicy(
        tmin=[2.0],
        tmax=[9.0],
        auto_window=False,
        min_points=10,
    )

    gamma, omega = policy.fit_signal(
        signal,
        idx=0,
        dt=dt,
        stride=1,
        params=LinearParams(),
        diagnostic_norm="none",
    )

    assert gamma == pytest.approx(0.12, rel=1e-3, abs=1e-3)
    assert omega == pytest.approx(0.31, rel=1e-3, abs=1e-3)


def test_scan_fit_window_policy_falls_back_to_auto_when_window_is_invalid() -> None:
    dt = 0.1
    t = np.arange(80) * dt
    signal = np.exp((0.08 - 0.2j) * t)
    policy = ScanFitWindowPolicy(
        tmin=100.0,
        tmax=101.0,
        auto_window=False,
        window_fraction=0.5,
        min_points=10,
        require_positive=True,
    )

    gamma, omega = policy.fit_signal(
        signal,
        idx=0,
        dt=dt,
        stride=1,
        params=LinearParams(),
        diagnostic_norm="none",
    )

    assert gamma == pytest.approx(0.08, rel=1e-3, abs=1e-3)
    assert omega == pytest.approx(0.2, rel=1e-3, abs=1e-3)
