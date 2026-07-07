from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT / "tools" / "artifacts" / "build_vmec_boozer_production_holdout_artifact.py"
)
spec = importlib.util.spec_from_file_location(
    "build_vmec_boozer_production_holdout_artifact", SCRIPT
)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _manifest(path: Path) -> Path:
    return _write_json(
        path,
        {
            "kind": "external_vmec_holdout_config_manifest",
            "case": "qh_holdout",
            "transport_sample": {
                "vmec_file": "/tmp/wout_qh.nc",
                "torflux": 0.78,
                "alpha": 1.2,
                "ky": 0.2,
                "npol": 1.0,
            },
        },
    )


def _ensemble(path: Path, *, passed: bool) -> Path:
    return _write_json(
        path,
        {
            "kind": "nonlinear_window_ensemble_report",
            "case": "qh_holdout_replicated_window",
            "passed": passed,
            "gate_report": {"passed": passed},
            "window": {"tmin": 350.0, "tmax": 700.0},
            "statistics": {"ensemble_mean": 1.25, "combined_sem": 0.05},
        },
    )


def test_vmec_boozer_production_holdout_artifact_promotes_only_passed_ensemble(
    tmp_path: Path,
) -> None:
    artifact = mod.build_vmec_boozer_production_holdout_artifact(
        transport_manifest=_manifest(tmp_path / "run_manifest.json"),
        ensemble_json=_ensemble(tmp_path / "ensemble.json", passed=True),
    )

    assert artifact["passed"] is True
    assert artifact["transport_average_gate"] is True
    assert artifact["promotion_gate"]["blockers"] == []
    assert (
        artifact["claim_level"]
        == "production_scope_vmec_boozer_heldout_nonlinear_transport_average"
    )
    sample = artifact["samples"][0]
    assert sample["surface"] == 0.78
    assert sample["torflux"] == 0.78
    assert sample["alpha"] == 1.2
    assert sample["selected_ky_index"] == "ky=0.2"
    assert artifact["holdout_samples"] == artifact["samples"]


def test_vmec_boozer_production_holdout_artifact_fails_closed_for_failed_ensemble(
    tmp_path: Path,
) -> None:
    artifact = mod.build_vmec_boozer_production_holdout_artifact(
        transport_manifest=_manifest(tmp_path / "run_manifest.json"),
        ensemble_json=_ensemble(tmp_path / "ensemble.json", passed=False),
        case="explicit_case",
    )

    assert artifact["case"] == "explicit_case"
    assert artifact["passed"] is False
    assert artifact["transport_average_gate"] is False
    assert artifact["promotion_gate"]["blockers"] == [
        "replicated_nonlinear_window_ensemble_failed"
    ]


def test_vmec_boozer_production_holdout_artifact_main_writes_output(
    tmp_path: Path,
) -> None:
    out = tmp_path / "holdout.json"
    result = mod.main(
        [
            "--transport-manifest",
            str(_manifest(tmp_path / "run_manifest.json")),
            "--ensemble-json",
            str(_ensemble(tmp_path / "ensemble.json", passed=True)),
            "--out",
            str(out),
        ]
    )

    assert result == 0
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert (
        saved["kind"]
        == "vmec_boozer_production_scope_heldout_nonlinear_transport_artifact"
    )
    assert saved["passed"] is True
