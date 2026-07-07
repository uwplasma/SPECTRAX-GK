from __future__ import annotations

import os
from dataclasses import replace
import importlib.util
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from spectraxgk.runtime import run_runtime_nonlinear


def _load_restart_base_cfg():
    path = (
        Path(__file__).resolve().parents[2]
        / "integration"
        / "runtime"
        / "test_restart_gate.py"
    )
    spec = importlib.util.spec_from_file_location("runtime_restart_gate_helpers", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._restart_base_cfg


_restart_base_cfg = _load_restart_base_cfg()


@pytest.mark.gpu
def test_cpu_gpu_short_window_gate_matches_within_tolerance() -> None:
    if os.environ.get("SPECTRAXGK_DEVICE_PARITY", "").strip() not in {
        "1",
        "true",
        "yes",
    }:
        pytest.skip("Set SPECTRAXGK_DEVICE_PARITY=1 to enable CPU/GPU parity gate.")

    try:
        cpu_devices = jax.devices("cpu")
    except Exception:
        cpu_devices = ()
    try:
        gpu_devices = jax.devices("gpu")
    except Exception:
        gpu_devices = ()

    cpu = cpu_devices[0] if cpu_devices else None
    gpu = gpu_devices[0] if gpu_devices else None
    if cpu is None or gpu is None:
        pytest.skip("No GPU backend detected for JAX.")

    cfg = _restart_base_cfg()
    cfg = replace(cfg, time=replace(cfg.time, dt=0.02))

    def _run_on(device):
        with jax.default_device(device):
            out = run_runtime_nonlinear(
                cfg,
                ky_target=0.2,
                kx_target=0.0,
                Nl=4,
                Nm=6,
                dt=0.02,
                steps=6,
                sample_stride=1,
                diagnostics_stride=1,
                return_state=True,
            )
        assert out.state is not None
        return np.asarray(out.state)

    state_cpu = _run_on(cpu)
    state_gpu = _run_on(gpu)

    # GPU FFTs can introduce small roundoff differences; gate on a stable scalar.
    norm_cpu = float(np.linalg.norm(state_cpu.ravel()))
    norm_gpu = float(np.linalg.norm(state_gpu.ravel()))
    assert norm_cpu > 0.0
    assert norm_gpu > 0.0
    assert norm_gpu == pytest.approx(norm_cpu, rel=2.0e-4, abs=1.0e-7)
