from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "build_manuscript_readiness_status.py"
spec = importlib.util.spec_from_file_location("build_manuscript_readiness_status", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def _write_json(root: Path, relative: str, payload: dict[str, object]) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_manuscript_status_closes_negative_ql_and_defers_zonal_tem(tmp_path: Path) -> None:
    _write_json(tmp_path, "docs/_static/quasilinear_validated_calibration_inputs.json", {"passed": True})
    _write_json(
        tmp_path,
        "docs/_static/quasilinear_stellarator_train_holdout_report.json",
        {
            "passed": False,
            "by_split": {"holdout": {"mean_abs_relative_error": 2.5}},
            "points": [
                {"case": "cyclone", "split": "train"},
                {"case": "w7x", "split": "holdout"},
                {"case": "hsx", "split": "holdout"},
            ],
        },
    )
    for name in (
        "quasilinear_saturation_rule_sweep",
        "quasilinear_shape_aware_saturation",
        "quasilinear_candidate_uncertainty",
    ):
        _write_json(
            tmp_path,
            f"docs/_static/{name}.json",
            {"promotion_gate": {"passed": False, "null_training_mean_mean_abs_relative_error": 0.2}},
        )
    _write_json(
        tmp_path,
        "docs/_static/quasilinear_dataset_sufficiency.json",
        {
            "promotion_gate": {"passed": False, "blockers": ["minimum_total_electrostatic_cases"]},
            "requirements": {
                "current_total_cases": 4,
                "min_total_electrostatic_cases": 6,
                "current_explicit_train_geometries": 1,
                "min_explicit_train_geometries": 2,
            },
        },
    )
    _write_json(
        tmp_path,
        "docs/_static/differentiable_geometry_bridge.json",
        {"sensitivity": {"max_abs_ad_fd_error": 1.0e-9}, "uq": {"sensitivity_map_rank": 2}},
    )
    _write_json(
        tmp_path,
        "docs/_static/vmec_boozer_parity_matrix.json",
        {"minimum_boozer_mode_count": 21, "summary": {"all_equal_arc_passed": True, "n_cases": 3}},
    )
    _write_json(
        tmp_path,
        "docs/_static/stellarator_itg_optimization_comparison.json",
        {
            "results": [
                {"initial_objective": 1.0, "final_objective": 0.2, "gradient_gate": {"passed": True}},
                {"initial_objective": 2.0, "final_objective": 0.5, "gradient_gate": {"passed": True}},
            ]
        },
    )
    _write_json(
        tmp_path,
        "docs/_static/stellarator_itg_optimization_uq.json",
        {
            "claim_level": "reduced_objective_uq_and_sensitivity_validation_not_full_vmec_gk_optimization",
            "all_gradient_gates_passed": True,
            "all_sensitivity_maps_full_rank": True,
        },
    )
    _write_json(
        tmp_path,
        "docs/_static/nonlinear_sharding_profile_office_gpu.json",
        {"identity_gate_pass": True, "engineering_speedup": 0.8},
    )

    payload = mod.build_manuscript_readiness_payload(tmp_path)
    lanes = {lane["lane"]: lane for lane in payload["lanes"]}

    assert payload["summary"]["n_deferred"] == 2
    assert lanes["Quasilinear diagnostics and saturation-model selection"]["status"] == "closed"
    assert lanes["Quasilinear diagnostics and saturation-model selection"]["key_metrics"]["absolute_flux_promoted"] is False
    assert (
        lanes["Quasilinear diagnostics and saturation-model selection"]["key_metrics"][
            "dataset_sufficiency_promotion_passed"
        ]
        is False
    )
    assert lanes["VMEC/Boozer differentiable geometry parity"]["status"] == "closed"
    assert lanes["Reduced differentiable stellarator ITG optimization"]["status"] == "closed"
    assert lanes["Production solver-objective geometry gradients"]["status"] == "open"
    assert lanes["W7-X zonal recurrence/damping"]["status"] == "deferred"
    assert lanes["TEM / kinetic-electron stellarator extension"]["status"] == "deferred"


def test_write_manuscript_readiness_artifacts_writes_all_formats(tmp_path: Path) -> None:
    payload = {
        "kind": "manuscript_readiness_status",
        "lanes": [
            {
                "lane": "Production solver-objective geometry gradients",
                "status": "open",
                "claim_level": "required",
                "primary_artifacts": [],
                "key_metrics": {},
                "next_action": "Add gates.",
            }
        ],
        "summary": {"active_fraction_closed": 0.0},
    }

    paths = mod.write_manuscript_readiness_artifacts(payload, out=tmp_path / "status.png")

    for path in paths.values():
        assert Path(path).exists()
    saved = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert saved["kind"] == "manuscript_readiness_status"
