from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_tool_module():
    path = Path(__file__).resolve().parents[3] / "tools" / "profile_quasilinear_uq_ensemble_scaling.py"
    spec = importlib.util.spec_from_file_location("profile_quasilinear_uq_ensemble_scaling", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_quasilinear_uq_ensemble_scaling_parser_defaults_to_bounded_solver_case() -> None:
    mod = _load_tool_module()

    args = mod.build_parser().parse_args([])

    assert args.out_prefix == mod.DEFAULT_PREFIX
    assert args.backend == "cpu"
    assert args.devices == [1, 2, 4]
    assert args.gradients == [2.20, 2.40, 2.60, 2.80, 3.00, 3.20]
    assert args.ky == [0.10, 0.20, 0.30, 0.40, 0.50]
    assert args.steps == 2000
    assert args.sample_stride == 10
    assert args.fit_start_fraction == 0.5
    assert args.fit_end_fraction == 0.95


def test_quasilinear_uq_ensemble_scaling_reduced_observable_is_positive_for_unstable_modes() -> None:
    mod = _load_tool_module()

    obs = mod._quasilinear_reduced_observables(
        np.asarray([0.2, 0.4]),
        np.asarray([0.1, -0.1]),
        np.asarray([0.3, 0.1]),
    )

    assert obs["heat_flux_proxy"] > 0.0
    assert obs["weighted_growth"] == 0.1
    assert obs["omega_span"] == 0.19999999999999998


def test_quasilinear_uq_ensemble_scaling_identity_metrics_detect_equal_members() -> None:
    mod = _load_tool_module()
    members = [
        {
            "R_over_LTi": 2.4,
            "heat_flux_proxy": 1.5,
            "gamma": [0.1, 0.2],
        },
        {
            "R_over_LTi": 2.7,
            "heat_flux_proxy": 2.5,
            "gamma": [0.3, 0.4],
        },
    ]

    metrics = mod._identity_metrics(
        {"members": members},
        {"members": list(reversed(members)), "error": None},
        value_rtol=1.0e-12,
        value_atol=1.0e-12,
    )

    assert metrics["identity_gate_pass"] is True
    assert metrics["max_heat_flux_proxy_abs_error"] == 0.0
    assert metrics["max_gamma_abs_error"] == 0.0


def test_quasilinear_uq_ensemble_scaling_worker_env_selects_device() -> None:
    mod = _load_tool_module()

    gpu_env = mod._worker_env({}, backend="gpu", worker_index=1)
    cpu_env = mod._worker_env({}, backend="cpu", worker_index=0)

    assert gpu_env["JAX_PLATFORMS"] == "cuda"
    assert gpu_env["CUDA_VISIBLE_DEVICES"] == "1"
    assert cpu_env["JAX_PLATFORMS"] == "cpu"
    assert cpu_env["OMP_NUM_THREADS"] == "1"
