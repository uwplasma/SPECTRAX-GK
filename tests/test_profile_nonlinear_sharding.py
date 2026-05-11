from __future__ import annotations

import importlib.util
from pathlib import Path

import jax.numpy as jnp
import pytest

from spectraxgk.terms.config import FieldState


def _load_tool_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "profile_nonlinear_sharding.py"
    spec = importlib.util.spec_from_file_location("profile_nonlinear_sharding", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_profile_nonlinear_sharding_parser_defaults_to_tracked_artifact() -> None:
    mod = _load_tool_module()
    args = mod.build_parser().parse_args([])

    assert args.out_json == mod.DEFAULT_OUT
    assert args.sharding == "auto"
    assert args.sharding_options is None
    assert args.method == "rk2"
    assert args.warmups == 1
    assert args.repeats == 3


def test_profile_nonlinear_sharding_helpers_report_stats_and_unique_specs() -> None:
    mod = _load_tool_module()

    stats = mod._time_stats([3.0, 1.0, 2.0])

    assert stats["min"] == 1.0
    assert stats["median"] == 2.0
    assert stats["mean"] == 2.0
    assert stats["max"] == 3.0
    assert mod._sharding_specs("auto", "ky,kx,ky,z") == ["auto", "ky", "kx", "z"]


def test_profile_nonlinear_sharding_reports_best_identity_candidate() -> None:
    mod = _load_tool_module()

    best = mod._best_identity_preserving_candidate(
        {
            "auto": {
                "identity_gate_pass": True,
                "engineering_speedup_median": 0.8,
                "state_sharding_active": True,
            },
            "kx": {
                "identity_gate_pass": True,
                "engineering_speedup_median": 1.2,
                "state_sharding_active": True,
            },
            "z": {
                "identity_gate_pass": False,
                "engineering_speedup_median": 3.0,
                "state_sharding_active": True,
            },
        }
    )

    assert best == {
        "spec": "kx",
        "engineering_speedup_median": 1.2,
        "state_sharding_active": True,
        "identity_gate_pass": True,
    }


def test_profile_nonlinear_sharding_diagnostic_metrics_compare_rhs_and_phi(monkeypatch) -> None:
    mod = _load_tool_module()

    def fake_rhs(state, cache, params, terms, *, gx_real_fft=True, laguerre_mode="grid"):
        del cache, params, terms, gx_real_fft, laguerre_mode
        arr = jnp.asarray(state)
        return 2.0 * arr, FieldState(phi=jnp.sum(arr, axis=(0, 1)), apar=None, bpar=None)

    monkeypatch.setattr(mod, "nonlinear_rhs_cached", fake_rhs)

    reference = jnp.ones((2, 2, 1, 1, 3), dtype=jnp.complex64)
    candidate = reference.at[0, 0, 0, 0, 0].add(1.0e-3)

    metrics = mod._nonlinear_diagnostic_identity_metrics(
        reference,
        candidate,
        cache=object(),
        params=object(),
        terms=object(),
        gx_real_fft=True,
        laguerre_mode="grid",
    )

    assert metrics["max_abs_rhs_error"] == pytest.approx(2.0e-3, rel=1.0e-4)
    assert metrics["max_abs_phi_error"] == pytest.approx(1.0e-3, rel=1.0e-4)
    assert metrics["max_rel_rhs_error"] > 0.0
    assert metrics["max_rel_phi_error"] > 0.0
