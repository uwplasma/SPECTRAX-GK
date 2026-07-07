from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tools" / "artifacts" / "build_pre_manuscript_closure_runbook.py"
spec = importlib.util.spec_from_file_location(
    "build_pre_manuscript_closure_runbook", SCRIPT
)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def _write_json(root: Path, relative: str, payload: dict[str, object]) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_screen(root: Path) -> Path:
    path = root / "screen.csv"
    path.write_text(
        "case,vmec_file,returncode,best_ky,best_gamma,best_omega,log\n"
        "existing_nc,wout_existing.nc,0,0.3,0.04,0.1,ok\n",
        encoding="utf-8",
    )
    return path


def test_pre_manuscript_runbook_fails_closed_but_lists_actions(tmp_path: Path) -> None:
    inventory = _write_json(
        tmp_path,
        "inventory.json",
        {
            "n_equilibria": 2,
            "rows": [
                {
                    "name": "wout_existing.nc",
                    "path": "wout_existing.nc",
                    "family": "axisymmetric",
                    "reference_scale_valid": True,
                    "candidate_score": 5.0,
                },
                {
                    "name": "wout_new_qh.nc",
                    "path": "wout_new_qh.nc",
                    "family": "quasi-helical",
                    "reference_scale_valid": True,
                    "candidate_score": 4.5,
                },
            ],
        },
    )
    screen = _write_screen(tmp_path)
    external = _write_json(
        tmp_path,
        "external.json",
        {"passed": False, "launch_commands": [], "min_launch_gamma": 0.02},
    )
    optimizer = _write_json(
        tmp_path, "optimizer.json", {"entries": [{"status": "runnable"}]}
    )
    ladder = _write_json(tmp_path, "ladder.json", {"commands": [{"returncode": 0}]})

    payload = mod.build_runbook_payload(
        root=tmp_path,
        inventory_path=inventory,
        screen_path=screen,
        external_runbook_path=external,
        optimizer_manifest_path=optimizer,
        ladder_status_path=ladder,
        office_root=Path("/office/repo"),
        audit_root=Path("tools_out/audits"),
    )

    holdout = payload["external_vmec_holdout_campaign"]
    assert payload["kind"] == "pre_manuscript_closure_runbook"
    assert holdout["status"] == "blocked_on_new_linear_screen"
    assert holdout["unscreened_candidates"][0]["name"] == "wout_new_qh.nc"
    assert payload["vmec_boozer_production_scope_artifacts"]["audit_commands"]
    heldout = payload["vmec_boozer_production_scope_artifacts"][
        "heldout_transport_commands"
    ]
    assert heldout[0]["transport_sample"]["alpha"] == 1.2
    assert "--torflux 0.78 --alpha 1.2" in heldout[0]["generate_configs_command"]
    assert "check_nonlinear_runtime_outputs.py" in heldout[0]["output_gate_command"]
    assert (
        "build_external_vmec_replicate_ensemble.py"
        in heldout[0]["build_ensemble_command"]
    )
    assert (
        "build_vmec_boozer_production_holdout_artifact.py"
        in heldout[0]["build_holdout_artifact_command"]
    )
    assert (
        "check_vmec_boozer_aggregate_holdout_gate.py"
        in heldout[0]["promotion_gate_command"]
    )
    assert "production_scope_vmec_boozer" in heldout[0]["claim_level"]
    assert (
        payload["vmec_boozer_production_scope_artifacts"]["office_seed_queue"][
            "launched"
        ]
        is True
    )
    assert "strict gates" in payload["claim_scope"]


def test_pre_manuscript_runbook_reports_launchable_external_holdout(
    tmp_path: Path,
) -> None:
    inventory = _write_json(tmp_path, "inventory.json", {"n_equilibria": 1, "rows": []})
    screen = _write_screen(tmp_path)
    external = _write_json(
        tmp_path,
        "external.json",
        {
            "passed": True,
            "launch_commands": [
                "python tools/campaigns/write_external_vmec_holdout_configs.py --case solovev"
            ],
            "min_launch_gamma": 0.02,
            "selected_new_family_candidate": {
                "case": "solovev_reference_nc",
                "best_gamma": 0.094,
            },
        },
    )
    optimizer = _write_json(tmp_path, "optimizer.json", {"entries": []})
    ladder = _write_json(tmp_path, "ladder.json", {"commands": []})

    payload = mod.build_runbook_payload(
        root=tmp_path,
        inventory_path=inventory,
        screen_path=screen,
        external_runbook_path=external,
        optimizer_manifest_path=optimizer,
        ladder_status_path=ladder,
        office_root=Path("/office/repo"),
        audit_root=Path("tools_out/audits"),
    )

    holdout = payload["external_vmec_holdout_campaign"]
    assert holdout["status"] == "launchable"
    assert holdout["selected_candidate"]["case"] == "solovev_reference_nc"
    assert holdout["launch_commands"]
    assert "Launch or harvest" in holdout["next_action"]
    assert payload["overall_next_actions"][1] == holdout["next_action"]


def test_pre_manuscript_runbook_marks_selected_external_holdout_harvested(
    tmp_path: Path,
) -> None:
    inventory = _write_json(tmp_path, "inventory.json", {"n_equilibria": 1, "rows": []})
    screen = _write_screen(tmp_path)
    external = _write_json(
        tmp_path,
        "external.json",
        {
            "passed": True,
            "launch_commands": [
                "python tools/campaigns/write_external_vmec_holdout_configs.py --case solovev"
            ],
            "selected_new_family_candidate": {
                "case": "solovev_reference_nc",
                "best_gamma": 0.094,
            },
        },
    )
    holdout_gap = _write_json(
        tmp_path,
        "holdout_gap.json",
        {
            "admitted_holdouts": [
                {
                    "case": "solovev_reference_repair_dt002_amp1em5_n48_t250",
                    "geometry": "solovev_external_vmec",
                    "gate_passed": True,
                    "status": "admitted_holdout",
                }
            ]
        },
    )
    optimizer = _write_json(tmp_path, "optimizer.json", {"entries": []})
    ladder = _write_json(tmp_path, "ladder.json", {"commands": []})

    payload = mod.build_runbook_payload(
        root=tmp_path,
        inventory_path=inventory,
        screen_path=screen,
        external_runbook_path=external,
        holdout_gap_path=holdout_gap,
        optimizer_manifest_path=optimizer,
        ladder_status_path=ladder,
        office_root=Path("/office/repo"),
        audit_root=Path("tools_out/audits"),
    )

    holdout = payload["external_vmec_holdout_campaign"]
    assert holdout["status"] == "harvested_admitted"
    assert holdout["admitted_holdout"]["case"].startswith("solovev_reference_repair")
    assert holdout["launch_commands"] == []
    assert "already harvested" in holdout["next_action"]


def test_write_pre_manuscript_runbook_artifacts(tmp_path: Path) -> None:
    payload = {
        "external_vmec_holdout_campaign": {
            "status": "blocked_on_new_linear_screen",
            "next_action": "screen candidates",
        },
        "vmec_boozer_production_scope_artifacts": {
            "status": "launch_contracts_generated_on_office",
        },
        "nonlinear_optimization_audit_extension": {
            "status": "running_or_launchable",
            "acceptance": "long-window gates",
        },
        "nonlinear_domain_decomposition": {
            "status": "identity_route_extended_no_speedup_claim",
            "next_action": "add distributed routing",
        },
    }

    paths = mod.write_runbook_artifacts(payload, out=tmp_path / "runbook.png")

    for path in paths.values():
        assert Path(path).exists()
    saved = json.loads((tmp_path / "runbook.json").read_text(encoding="utf-8"))
    assert (
        saved["external_vmec_holdout_campaign"]["status"]
        == "blocked_on_new_linear_screen"
    )
