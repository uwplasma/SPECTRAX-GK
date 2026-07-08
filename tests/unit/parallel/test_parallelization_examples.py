"""Smoke tests for parallelization examples."""

from __future__ import annotations

from support.paths import REPO_ROOT, load_repo_script
from types import SimpleNamespace

import numpy as np


_REPO = REPO_ROOT
_EXAMPLE = (
    _REPO / "examples" / "parallelization" / "independent_ky_runtime_batch_scan.py"
)
_CONFIG = _REPO / "examples" / "parallelization" / "runtime_batch_ky_scan.toml"


def _load_example_module():
    return load_repo_script(
        _EXAMPLE.relative_to(REPO_ROOT),
        module_name="independent_ky_runtime_batch_scan",
        write_bytecode=False,
    )


def test_runtime_batch_ky_scan_example_uses_independent_workers(
    monkeypatch,
) -> None:
    import spectraxgk.runtime as runtime

    example = _load_example_module()
    calls: list[float] = []

    def _unexpected_combined_batch(*_args, **_kwargs):
        raise AssertionError(
            "strategy='batch' example must not use the combined-ky solver path"
        )

    def _fake_run_runtime_linear(_cfg, **kwargs):
        ky = float(kwargs["ky_target"])
        calls.append(ky)
        return SimpleNamespace(gamma=1.0 + ky, omega=-(2.0 + ky), quasilinear=None)

    monkeypatch.setattr(runtime, "_run_runtime_scan_batch", _unexpected_combined_batch)
    monkeypatch.setattr(runtime, "run_runtime_linear", _fake_run_runtime_linear)

    scan = example.run_example(_CONFIG)

    np.testing.assert_allclose(sorted(calls), [0.1, 0.2, 0.3])
    np.testing.assert_allclose(scan.ky, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(scan.gamma, [1.1, 1.2, 1.3])
    np.testing.assert_allclose(scan.omega, [-2.1, -2.2, -2.3])
    assert scan.parallel is not None
    assert scan.parallel["source"] == "runtime_config"
    assert scan.parallel["requested_workers"] == 2
    assert scan.parallel["effective_workers"] == 2
    assert scan.parallel["executor"] == "thread"
    assert "independent ky workers" in scan.parallel["identity_contract"]
