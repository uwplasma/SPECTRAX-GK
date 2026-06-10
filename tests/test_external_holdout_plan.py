from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys

from spectraxgk.external_holdout_plan import (
    ExternalHoldoutScreenRow,
    build_external_holdout_runbook,
    external_vmec_family,
    read_external_holdout_screen,
)


def _gap_report() -> dict:
    return {
        "admitted_holdouts": [
            {"case": "dshape_window", "geometry": "dshape_external_vmec"},
            {"case": "updown_window", "geometry": "updown_asym_external_vmec"},
        ],
        "training_references": [
            {"case": "itermodel_external_vmec_t350_window", "geometry": "itermodel_external_vmec"}
        ],
        "next_actual_nonlinear_holdout_needed": {
            "preferred_family": "itermodel_external_vmec",
            "nearest_tracked_gap": {
                "case": "ITERModel external VMEC nonlinear t250 high-grid convergence",
                "source_artifact": "docs/_static/external_vmec_itermodel_t250_high_grid_convergence_gate.json",
                "failed_gates": ["common slope: 0.002197 > 0.002"],
                "next_best_score": 1.1,
            },
        },
    }


def _gap_report_with_failed_shaped_family() -> dict:
    report = _gap_report()
    report["excluded_candidates"] = [
        {
            "case": "Shaped tokamak external VMEC nonlinear t450 high-grid convergence",
            "geometry": "shaped_tokamak_external_vmec",
            "gate_passed": False,
            "status": "excluded_failed_external_gate",
        }
    ]
    return report


def _gap_report_with_failed_cth_like_family() -> dict:
    report = _gap_report()
    report["excluded_candidates"] = [
        {
            "case": "CTH-like external VMEC nonlinear grid convergence",
            "geometry": "cth_like_external_vmec",
            "gate_passed": False,
            "status": "excluded_failed_external_gate",
        }
    ]
    return report


def _gap_report_with_passed_preferred_audit() -> dict:
    report = _gap_report()
    report["excluded_candidates"] = [
        {
            "case": "ITERModel external VMEC independent audit t450 high-grid convergence",
            "geometry": "itermodel_external_vmec",
            "gate_passed": True,
            "status": "excluded_same_family_training_audit",
        }
    ]
    return report


def test_family_detection_covers_screen_names() -> None:
    assert external_vmec_family("ITERModel_reference_nc") == "itermodel_external_vmec"
    assert external_vmec_family("DSHAPE_nc") == "dshape_external_vmec"
    assert external_vmec_family("circular_tokamak_nc") == "circular_external_vmec"
    assert external_vmec_family("shaped_tokamak_pressure_reference_nc") == "shaped_tokamak_external_vmec"
    assert external_vmec_family("QI_stel_seed_3127_nc", "/vmec/wout_QI_stel_seed_3127.nc") == "qi_external_vmec"
    assert external_vmec_family("li383_low_res_nc", "/vmec/wout_li383_low_res.nc") == "li383_external_vmec"
    assert external_vmec_family("basic_non_stellsym_nc") == "non_stellsym_external_vmec"
    assert external_vmec_family("cth_like_reference_nc", "/vmec/wout_cth_like.nc") == "cth_like_external_vmec"


def test_read_external_holdout_screen_and_rank_runbook(tmp_path: Path) -> None:
    screen = tmp_path / "screen.csv"
    with screen.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case", "vmec_file", "returncode", "best_ky", "best_gamma", "best_omega", "log"])
        writer.writerow(["DSHAPE_nc", "/vmec/wout_DSHAPE.nc", 0, 0.47, 0.096, 0.06, "dshape.log"])
        writer.writerow(["ITERModel_reference_nc", "/vmec/wout_ITER.nc", 0, 0.47, 0.089, 0.40, "iter.log"])
        writer.writerow(["circular_tokamak_nc", "/vmec/wout_circular.nc", 0, 0.47, 0.088, 0.41, "circ.log"])
        writer.writerow(["stable_nc", "/vmec/wout_stable.nc", 0, 0.47, -0.01, 0.1, "stable.log"])

    rows = read_external_holdout_screen(screen)
    assert rows[0].unstable is True
    assert rows[-1].unstable is False

    runbook = build_external_holdout_runbook(
        gap_report=_gap_report(),
        screen_rows=rows,
        out_dir="tools_out/test_holdouts",
        max_candidates=4,
    )

    assert runbook["passed"] is True
    assert runbook["absolute_flux_promoted"] is False
    assert runbook["preferred_family"] == "itermodel_external_vmec"
    assert runbook["recommended_horizons"] == [250.0, 350.0, 450.0]
    assert runbook["selected_preferred_family_audit"]["case"] == "ITERModel_reference_nc"
    assert runbook["selected_new_family_candidate"]["case"] == "circular_tokamak_nc"
    assert runbook["selected_new_family_candidate"]["status"] == "new_family_holdout_candidate"
    assert runbook["acceptance_gate"]["required_split"] == "holdout"
    assert any("write_external_vmec_holdout_configs.py" in command for command in runbook["launch_commands"])


def test_runbook_demotes_recent_failed_external_family() -> None:
    rows = [
        ExternalHoldoutScreenRow(
            case="shaped_tokamak_pressure_reference_nc",
            vmec_file="/vmec/wout_shaped_tokamak_pressure_reference.nc",
            returncode=0,
            best_ky=0.47,
            best_gamma=0.047,
            best_omega=0.28,
        ),
        ExternalHoldoutScreenRow(
            case="ITERModel_reference_nc",
            vmec_file="/vmec/wout_ITER.nc",
            returncode=0,
            best_ky=0.47,
            best_gamma=0.089,
            best_omega=0.40,
        ),
    ]

    runbook = build_external_holdout_runbook(
        gap_report=_gap_report_with_failed_shaped_family(),
        screen_rows=rows,
    )

    shaped = next(row for row in runbook["ranked_candidates"] if row["case"].startswith("shaped_tokamak"))
    assert shaped["status"] == "recent_family_failed_external_gate"
    assert runbook["selected_new_family_candidate"] is None
    assert runbook["selected_preferred_family_audit"]["case"] == "ITERModel_reference_nc"


def test_runbook_allows_failed_family_only_with_explicit_modified_protocol() -> None:
    rows = [
        ExternalHoldoutScreenRow(
            case="cth_like_reference_nc",
            vmec_file="/vmec/wout_cth_like.nc",
            returncode=0,
            best_ky=0.3,
            best_gamma=0.071,
            best_omega=0.22,
        ),
        ExternalHoldoutScreenRow(
            case="ITERModel_reference_nc",
            vmec_file="/vmec/wout_ITER.nc",
            returncode=0,
            best_ky=0.47,
            best_gamma=0.089,
            best_omega=0.40,
        ),
    ]

    default = build_external_holdout_runbook(
        gap_report=_gap_report_with_failed_cth_like_family(),
        screen_rows=rows,
    )
    cth_default = next(row for row in default["ranked_candidates"] if row["family"] == "cth_like_external_vmec")
    assert cth_default["status"] == "recent_family_failed_external_gate"
    assert default["selected_new_family_candidate"] is None

    modified = build_external_holdout_runbook(
        gap_report=_gap_report_with_failed_cth_like_family(),
        screen_rows=rows,
        horizons=(150.0, 250.0, 350.0),
        grids=("n48:48:48:32:32", "n64:64:64:40:40", "n80:80:80:48:48"),
        allow_modified_protocol_families=("cth_like_external_vmec",),
        modified_protocol_note="restart ladder with an added n80 grid and longer post-transient windows",
    )

    selected = modified["selected_new_family_candidate"]
    assert selected["case"] == "cth_like_reference_nc"
    assert selected["status"] == "modified_protocol_failed_family_candidate"
    assert modified["recommended_horizons"] == [150.0, 250.0, 350.0]
    assert modified["allow_modified_protocol_families"] == ["cth_like_external_vmec"]
    assert "n80:80:80:48:48" in modified["launch_commands"][0]
    assert "--horizons 150,250,350" in modified["launch_commands"][0]


def test_runbook_requires_note_for_modified_failed_family() -> None:
    rows = [
        ExternalHoldoutScreenRow(
            case="cth_like_reference_nc",
            vmec_file="/vmec/wout_cth_like.nc",
            returncode=0,
            best_ky=0.3,
            best_gamma=0.071,
            best_omega=0.22,
        )
    ]
    try:
        build_external_holdout_runbook(
            gap_report=_gap_report_with_failed_cth_like_family(),
            screen_rows=rows,
            allow_modified_protocol_families=("cth_like_external_vmec",),
        )
    except ValueError as exc:
        assert "modified_protocol_note" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("modified failed-family reruns must require a protocol note")


def test_runbook_does_not_relaunch_passed_same_family_audit() -> None:
    rows = [
        ExternalHoldoutScreenRow(
            case="ITERModel_reference_nc",
            vmec_file="/vmec/wout_ITER.nc",
            returncode=0,
            best_ky=0.47,
            best_gamma=0.089,
            best_omega=0.40,
        )
    ]

    runbook = build_external_holdout_runbook(
        gap_report=_gap_report_with_passed_preferred_audit(),
        screen_rows=rows,
    )

    assert runbook["passed"] is False
    assert runbook["launch_commands"] == []
    assert runbook["selected_preferred_family_audit"] is None
    assert runbook["ranked_candidates"][0]["status"] == "preferred_family_audit_already_passed"


def test_runbook_fails_closed_when_no_unstable_candidate_exists() -> None:
    rows = [
        ExternalHoldoutScreenRow(
            case="stable_nc",
            vmec_file="/vmec/wout_stable.nc",
            returncode=0,
            best_ky=0.3,
            best_gamma=-0.01,
            best_omega=0.1,
        )
    ]
    runbook = build_external_holdout_runbook(gap_report=_gap_report(), screen_rows=rows)
    assert runbook["passed"] is False
    assert runbook["launch_commands"] == []
    assert runbook["ranked_candidates"][0]["status"] == "screen_rejected_stable_or_failed"


def test_runbook_blocks_marginal_linear_candidate_from_nonlinear_launch() -> None:
    rows = [
        ExternalHoldoutScreenRow(
            case="QI_stel_seed_3127_nc",
            vmec_file="/vmec/wout_QI_stel_seed_3127.nc",
            returncode=0,
            best_ky=0.1429,
            best_gamma=0.0038,
            best_omega=-0.09,
        )
    ]
    runbook = build_external_holdout_runbook(gap_report=_gap_report(), screen_rows=rows)
    assert runbook["passed"] is False
    assert runbook["launch_commands"] == []
    assert runbook["min_launch_gamma"] == 0.02
    assert runbook["ranked_candidates"][0]["status"] == "screen_marginal_needs_linear_refinement"


def _load_tool_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "build_external_vmec_holdout_runbook.py"
    spec = importlib.util.spec_from_file_location("build_external_vmec_holdout_runbook", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_runbook_tool_writes_replayable_artifacts(tmp_path: Path) -> None:
    mod = _load_tool_module()
    gap = tmp_path / "gap.json"
    gap.write_text(json.dumps(_gap_report()), encoding="utf-8")
    screen = tmp_path / "screen.csv"
    screen.write_text(
        "\n".join(
            [
                "case,vmec_file,returncode,best_ky,best_gamma,best_omega,log",
                "circular_tokamak_nc,/vmec/wout_circular.nc,0,0.47,0.088,0.41,circ.log",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "runbook.png"
    assert mod.main(["--gap-report", str(gap), "--screen", str(screen), "--out", str(out), "--no-pdf", "--dpi", "80"]) == 0
    payload = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["png"].endswith("runbook.png")
    assert out.with_suffix(".csv").exists()


def test_runbook_tool_writes_modified_protocol_contract(tmp_path: Path) -> None:
    mod = _load_tool_module()
    gap = tmp_path / "gap.json"
    gap.write_text(json.dumps(_gap_report_with_failed_cth_like_family()), encoding="utf-8")
    screen = tmp_path / "screen.csv"
    screen.write_text(
        "\n".join(
            [
                "case,vmec_file,returncode,best_ky,best_gamma,best_omega,log",
                "cth_like_reference_nc,/vmec/wout_cth_like.nc,0,0.3,0.071,0.22,cth.log",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "runbook.png"
    rc = mod.main(
        [
            "--gap-report",
            str(gap),
            "--screen",
            str(screen),
            "--out",
            str(out),
            "--no-pdf",
            "--dpi",
            "80",
            "--horizons",
            "150,250,350",
            "--grid",
            "n48:48:48:32:32",
            "--grid",
            "n80:80:80:48:48",
            "--allow-modified-protocol-family",
            "cth_like_external_vmec",
            "--modified-protocol-note",
            "n80 grid and longer post-transient window repair",
        ]
    )
    assert rc == 0
    payload = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["selected_new_family_candidate"]["family"] == "cth_like_external_vmec"
    assert payload["selected_new_family_candidate"]["status"] == "modified_protocol_failed_family_candidate"
    assert payload["modified_protocol_note"] == "n80 grid and longer post-transient window repair"
