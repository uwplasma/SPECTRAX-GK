from __future__ import annotations

from support.paths import load_profiling_tool
from pathlib import Path

import jax.numpy as jnp
import pytest

from spectraxgk.terms.config import FieldState


def _load_tool_module():
    return load_profiling_tool("profile_nonlinear_sharding")


def test_profile_nonlinear_sharding_parser_defaults_to_tracked_artifact() -> None:
    mod = _load_tool_module()
    args = mod.build_parser().parse_args([])

    assert args.out_json == mod.DEFAULT_OUT
    assert args.sharding == "auto"
    assert args.sharding_options is None
    assert args.method == "rk2"
    assert args.warmups == 1
    assert args.repeats == 3
    assert args.allow_unsafe_cpu_state_sharding is False
    assert (
        mod._artifact_path_for_contract(args.out_json)
        == "docs/_static/nonlinear_sharding_profile.json"
    )


def test_profile_nonlinear_sharding_source_contract_is_machine_readable(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    out_json = tmp_path / "profile.json"
    argv = [
        "--out-json",
        str(out_json),
        "--sharding",
        "kx",
        "--warmups",
        "0",
        "--repeats",
        "2",
    ]
    args = mod.build_parser().parse_args(argv)

    contract = mod._source_contract(args, argv, backend="gpu", device_count=2)

    assert contract["backend"] == "gpu"
    assert contract["source_contract_version"] == 1
    assert contract["device_count"] == 2
    assert contract["sharding_axis"] == "kx"
    assert contract["source_artifact"] == str(out_json.resolve())
    assert contract["timing_warmup_repeat"] == {"warmups": 0, "repeats": 2}
    assert contract["allow_unsafe_cpu_state_sharding"] is False
    assert contract["profile_command_argv"][-len(argv) :] == argv
    assert (
        "tools/profiling/profile_nonlinear_sharding.py" in contract["profile_command"]
    )
    assert {"python", "spectraxgk", "jax", "jaxlib", "numpy"} <= set(
        contract["software_versions"]
    )
    assert all(contract["software_versions"].values())


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


def test_profile_nonlinear_sharding_skips_unsafe_cpu_state_sharding() -> None:
    mod = _load_tool_module()

    assert (
        mod._skip_unsafe_cpu_state_sharding(
            backend="cpu",
            device_count=4,
            state_sharding_active=True,
            allow_unsafe_cpu_state_sharding=False,
        )
        is True
    )
    assert (
        mod._skip_unsafe_cpu_state_sharding(
            backend="cpu",
            device_count=4,
            state_sharding_active=True,
            allow_unsafe_cpu_state_sharding=True,
        )
        is False
    )
    assert (
        mod._skip_unsafe_cpu_state_sharding(
            backend="gpu",
            device_count=2,
            state_sharding_active=True,
            allow_unsafe_cpu_state_sharding=False,
        )
        is False
    )

    row = mod._candidate_failure(
        state_sharding_active=True,
        error=mod.CPU_WHOLE_STATE_SHARDING_SKIP_REASON,
        skip_reason="cpu_whole_state_pjit_sharding_unsafe_for_fft_layout",
    )

    assert row["identity_gate_pass"] is False
    assert row["state_sharding_active"] is True
    assert row["skip_reason"] == "cpu_whole_state_pjit_sharding_unsafe_for_fft_layout"
    assert "unsafe_for_fft_layout" in row["error"]


def test_profile_nonlinear_sharding_diagnostic_metrics_compare_rhs_and_phi(
    monkeypatch,
) -> None:
    mod = _load_tool_module()

    def fake_rhs(
        state, cache, params, terms, *, compressed_real_fft=True, laguerre_mode="grid"
    ):
        del cache, params, terms, compressed_real_fft, laguerre_mode
        arr = jnp.asarray(state)
        return 2.0 * arr, FieldState(
            phi=jnp.sum(arr, axis=(0, 1)), apar=None, bpar=None
        )

    monkeypatch.setattr(mod, "nonlinear_rhs_cached", fake_rhs)

    reference = jnp.ones((2, 2, 1, 1, 3), dtype=jnp.complex64)
    candidate = reference.at[0, 0, 0, 0, 0].add(1.0e-3)

    metrics = mod._nonlinear_diagnostic_identity_metrics(
        reference,
        candidate,
        cache=object(),
        params=object(),
        terms=object(),
        compressed_real_fft=True,
        laguerre_mode="grid",
    )

    assert metrics["max_abs_rhs_error"] == pytest.approx(2.0e-3, rel=1.0e-4)
    assert metrics["max_abs_phi_error"] == pytest.approx(1.0e-3, rel=1.0e-4)
    assert metrics["max_rel_rhs_error"] > 0.0
    assert metrics["max_rel_phi_error"] > 0.0
