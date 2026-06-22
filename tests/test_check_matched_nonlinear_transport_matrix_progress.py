from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "check_matched_nonlinear_transport_matrix_progress.py"
spec = importlib.util.spec_from_file_location("check_matched_nonlinear_transport_matrix_progress", SCRIPT)
assert spec is not None
assert spec.loader is not None
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def _write_manifest(tmp_path: Path, outputs: list[Path]) -> Path:
    manifest = {
        "kind": "matched_nonlinear_transport_matrix_campaign",
        "config": {"window": {"tmin": 10.0, "tmax": 20.0}},
        "samples": [
            {
                "sample_id": "s0p45_a0_ky0p1",
                "surface_torflux": 0.45,
                "alpha": 0.0,
                "ky": 0.1,
                "states": {
                    "baseline": {"label": "base", "final_outputs": [str(outputs[0])]},
                    "candidate": {"label": "cand", "final_outputs": [str(outputs[1])]},
                },
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _touch_bundle(output: Path) -> None:
    stem = output.name[: -len(".out.nc")] if output.name.endswith(".out.nc") else output.stem
    base = output.with_name(stem)
    for suffix in ("out.nc", "restart.nc", "big.nc"):
        Path(f"{base}.{suffix}").write_text("stub\n", encoding="utf-8")


def test_progress_requires_target_time_even_when_bundle_exists(tmp_path: Path, monkeypatch) -> None:
    base = tmp_path / "base.out.nc"
    cand = tmp_path / "cand.out.nc"
    _touch_bundle(base)
    _touch_bundle(cand)
    manifest = _write_manifest(tmp_path, [base, cand])
    monkeypatch.setattr(mod, "_read_output_tmax", lambda _path: 19.0)

    report = mod.build_report(matrix_manifest=manifest)

    assert report["summary"]["expected_outputs"] == 2
    assert report["summary"]["complete_bundles"] == 2
    assert report["summary"]["target_time_confirmed"] == 0
    assert report["summary"]["ready_for_postprocess"] is False


def test_progress_passes_when_all_bundles_reach_target_time(tmp_path: Path, monkeypatch) -> None:
    base = tmp_path / "base.out.nc"
    cand = tmp_path / "cand.out.nc"
    _touch_bundle(base)
    _touch_bundle(cand)
    manifest = _write_manifest(tmp_path, [base, cand])
    monkeypatch.setattr(mod, "_read_output_tmax", lambda _path: 20.0)

    report = mod.build_report(matrix_manifest=manifest)

    assert report["summary"]["complete_bundles"] == 2
    assert report["summary"]["target_time_confirmed"] == 2
    assert report["summary"]["ready_for_postprocess"] is True
    assert all(row["bundle_complete"] for row in report["rows"])
    assert all(row["target_time_confirmed"] for row in report["rows"])
