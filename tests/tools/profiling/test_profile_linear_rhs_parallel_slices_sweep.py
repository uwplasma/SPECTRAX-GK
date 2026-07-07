from __future__ import annotations

import json
from pathlib import Path

from tools.profiling import profile_linear_rhs_parallel_slices_sweep as sweep


def test_profile_linear_rhs_parallel_slices_sweep_builds_summary(monkeypatch) -> None:
    def fake_profile(**kwargs):  # type: ignore[no-untyped-def]
        devices = int(kwargs["requested_devices"])
        nm = int(kwargs["nm"])
        serial = float(nm) * 1.0e-3
        sharded = serial / max(devices, 1)
        return {
            "state_shape": (4, nm, 8, 1, 16),
            "serial_median_s": serial,
            "sharded_median_s": sharded,
            "speedup": serial / sharded,
            "identity_passed": True,
            "max_abs_error": 0.0,
            "max_rel_error": 0.0,
            "max_phi_abs_error": 0.0,
        }

    monkeypatch.setattr(sweep, "profile_linear_rhs_parallel_slices", fake_profile)

    summary = sweep.run_sweep(
        platform="cpu",
        devices=[1, 2],
        nms=[8, 16],
        nx=1,
        ny=4,
        nz=8,
        nl=2,
        warmups=0,
        repeats=1,
        atol=1.0e-6,
        rtol=1.0e-6,
    )

    assert summary["identity_passed"] is True
    assert len(summary["rows"]) == 4
    assert {row["speedup"] for row in summary["rows"]} == {1.0, 2.0}


def test_profile_linear_rhs_parallel_slices_sweep_writes_artifacts(
    tmp_path: Path,
) -> None:
    summary = {
        "identity_passed": True,
        "rtol": 1.0e-5,
        "rows": [
            {
                "platform": "cpu",
                "requested_devices": 1,
                "nm": 8,
                "state_shape": (2, 8, 4, 1, 8),
                "serial_median_s": 0.02,
                "sharded_median_s": 0.02,
                "speedup": 1.0,
                "identity_passed": True,
                "max_abs_error": 0.0,
                "max_rel_error": 0.0,
                "max_phi_abs_error": 0.0,
            },
            {
                "platform": "cpu",
                "requested_devices": 2,
                "nm": 8,
                "state_shape": (2, 8, 4, 1, 8),
                "serial_median_s": 0.02,
                "sharded_median_s": 0.012,
                "speedup": 1.67,
                "identity_passed": True,
                "max_abs_error": 0.0,
                "max_rel_error": 0.0,
                "max_phi_abs_error": 0.0,
            },
        ],
    }
    out = tmp_path / "linear_rhs_parallel_slices_sweep"
    paths = sweep.write_artifacts(summary, out)

    assert (
        json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))[
            "identity_passed"
        ]
        is True
    )
    assert "requested_devices" in out.with_suffix(".csv").read_text(encoding="utf-8")
    assert Path(paths["png"]).exists()
    assert Path(paths["pdf"]).exists()
