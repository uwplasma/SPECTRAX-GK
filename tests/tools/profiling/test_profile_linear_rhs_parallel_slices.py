from __future__ import annotations

import json
from pathlib import Path

from tools.profiling import profile_linear_rhs_parallel_slices as profile


def test_profile_linear_rhs_parallel_slices_builds_summary(monkeypatch) -> None:
    class FakeGrid:
        import numpy as np

        ky = np.asarray([0.0, 0.3])
        z = np.asarray([0.0, 1.0, 2.0, 3.0])

    def fake_problem(**_kwargs):  # type: ignore[no-untyped-def]
        import jax.numpy as jnp

        return (
            jnp.ones((1, 4, 2, 1, 4), dtype=jnp.complex64),
            object(),
            object(),
            FakeGrid(),
        )

    def fake_rhs(state, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        import jax.numpy as jnp

        return 2.0 * state, jnp.ones((2, 1, 4), dtype=jnp.complex64)

    monkeypatch.setattr(profile, "build_problem", fake_problem)
    monkeypatch.setattr("jax.devices", lambda _kind=None: [object(), object()])
    monkeypatch.setattr("spectraxgk.linear.linear_rhs_cached", fake_rhs)
    monkeypatch.setattr("spectraxgk.linear.linear_rhs_parallel_cached", fake_rhs)

    summary = profile.profile_linear_rhs_parallel_slices(
        platform="cpu",
        requested_devices=2,
        nx=1,
        ny=2,
        nz=4,
        nl=1,
        nm=4,
        warmups=0,
        repeats=1,
        atol=1.0e-12,
        rtol=1.0e-12,
    )

    assert summary["identity_passed"] is True
    assert summary["max_abs_error"] == 0.0
    assert summary["speedup"] > 0.0
    assert len(summary["rows"]) == 2


def test_profile_linear_rhs_parallel_slices_writes_artifacts(tmp_path: Path) -> None:
    summary = {
        "rows": [
            {"route": "serial", "median_s": 0.02, "samples_s": [0.02]},
            {"route": "sharded", "median_s": 0.03, "samples_s": [0.03]},
        ],
        "atol": 1.0e-8,
        "rtol": 1.0e-8,
        "identity_passed": True,
        "speedup": 0.67,
        "max_abs_error": 0.0,
        "max_rel_error": 0.0,
        "max_phi_abs_error": 0.0,
    }
    out = tmp_path / "linear_rhs_parallel_slices_profile"
    paths = profile.write_artifacts(summary, out)

    assert (
        json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))[
            "identity_passed"
        ]
        is True
    )
    assert "median_s" in out.with_suffix(".csv").read_text(encoding="utf-8")
    assert Path(paths["png"]).exists()
    assert Path(paths["pdf"]).exists()
