from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_tool_module():
    path = (
        Path(__file__).resolve().parents[3]
        / "tools"
        / "profiling"
        / "profile_independent_ky_scan_scaling.py"
    )
    spec = importlib.util.spec_from_file_location(
        "profile_independent_ky_scan_scaling", path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_independent_ky_scan_scaling_parser_defaults_to_large_solver_case() -> None:
    mod = _load_tool_module()

    args = mod.build_parser().parse_args([])

    assert args.out_prefix == mod.DEFAULT_PREFIX
    assert args.backend == "cpu"
    assert args.devices == [1, 2, 4]
    assert args.ny == 128
    assert args.nz == 96
    assert args.steps == 240
    assert len(args.ky) == 12


def test_independent_ky_scan_scaling_splits_ky_without_empty_chunks() -> None:
    mod = _load_tool_module()

    chunks = mod._split_ky(np.asarray([0.1, 0.2, 0.3]), 8)

    assert [chunk.tolist() for chunk in chunks] == [[0.1], [0.2], [0.3]]


def test_independent_ky_scan_scaling_worker_env_is_backend_specific() -> None:
    mod = _load_tool_module()

    cpu_env = mod._worker_env({}, backend="cpu", worker_index=3)
    gpu_env = mod._worker_env({}, backend="gpu", worker_index=1)

    assert cpu_env["JAX_PLATFORMS"] == "cpu"
    assert cpu_env["OMP_NUM_THREADS"] == "1"
    assert gpu_env["JAX_PLATFORMS"] == "cuda"
    assert gpu_env["CUDA_VISIBLE_DEVICES"] == "1"
    assert gpu_env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"


def test_independent_ky_scan_scaling_identity_metrics_pass_for_reordered_equal_values() -> (
    None
):
    mod = _load_tool_module()
    reference = {"ky": [0.1, 0.2], "gamma": [0.3, 0.4], "omega": [-0.1, -0.2]}
    row = {"ky": [0.1, 0.2], "gamma": [0.3, 0.4], "omega": [-0.1, -0.2], "error": None}

    metrics = mod._identity_metrics(reference, row)

    assert metrics["identity_gate_pass"] is True
    assert metrics["max_gamma_abs_error"] == 0.0
    assert metrics["max_omega_abs_error"] == 0.0
