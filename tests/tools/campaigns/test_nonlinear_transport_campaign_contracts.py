"""Contracts for nonlinear transport campaign orchestration tools."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from support.paths import load_campaign_tool

release_finalizer = load_campaign_tool("finalize_nonlinear_transport_matrix_release")
over_writer = load_campaign_tool("write_overdetermined_nonlinear_gradient_campaign")
over_runner = load_campaign_tool("run_nonlinear_gradient_direct_campaign")
over_post = over_runner
gradient_post = over_runner


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_png_stub(path: Path) -> Path:
    path.write_bytes(b"\x89PNG\r\n\x1a\n")
    return path


def _overdetermined_input_text() -> str:
    return """&INDATA
  RBC(1,1) = 1.0000000000000000E-01
  ZBS(1,0) = -2.0000000000000000E-02
  ZBS(1,1) = 5.0000000000000000E-02
/
"""


def test_finalize_nonlinear_transport_matrix_release_import_artifacts_selected_family_only(
    tmp_path: Path,
) -> None:
    portfolio = _write_json(
        tmp_path / "portfolio.json",
        {
            "kind": "nonlinear_transport_matrix_portfolio_gate",
            "passed": True,
            "selected_family": "projected_0p001",
        },
    )
    portfolio_png = _write_png_stub(tmp_path / "portfolio.png")
    matrix_json = _write_json(
        tmp_path / "projected.json",
        {"kind": "matched_nonlinear_transport_matrix_report", "passed": True},
    )
    matrix_png = _write_png_stub(tmp_path / "projected.png")
    docs_static = tmp_path / "docs_static"

    manifest = release_finalizer.import_artifacts(
        portfolio_json=portfolio,
        portfolio_figure=portfolio_png,
        matrix_report_jsons={"projected_0p001": matrix_json},
        matrix_report_figures={"projected_0p001": matrix_png},
        docs_static=docs_static,
    )

    assert manifest["passed"] is True
    assert manifest["selected_family"] == "projected_0p001"
    assert (docs_static / "nonlinear_transport_matrix_portfolio.json").exists()
    assert (docs_static / "nonlinear_transport_matrix_portfolio.png").exists()
    assert (docs_static / "projected_0p001_matrix_report.json").exists()
    assert (docs_static / "projected_0p001_matrix_report.png").exists()
    import_manifest = json.loads(
        (docs_static / "nonlinear_transport_matrix_portfolio_import.json").read_text(
            encoding="utf-8"
        )
    )
    assert import_manifest["kind"] == "nonlinear_transport_matrix_portfolio_import"


def test_finalize_nonlinear_transport_matrix_release_import_artifacts_fail_closed(
    tmp_path: Path,
) -> None:
    blocked = _write_json(
        tmp_path / "blocked.json",
        {
            "kind": "nonlinear_transport_matrix_portfolio_gate",
            "passed": False,
            "selected_family": None,
        },
    )
    with pytest.raises(ValueError, match="blocked"):
        release_finalizer.import_artifacts(
            portfolio_json=blocked,
            portfolio_figure=None,
            matrix_report_jsons={},
            matrix_report_figures={},
            docs_static=tmp_path / "docs_static",
        )


def test_finalize_nonlinear_transport_matrix_release_imports_passing_portfolio(
    tmp_path: Path,
) -> None:
    portfolio = _write_json(
        tmp_path / "portfolio.json",
        {
            "kind": "nonlinear_transport_matrix_portfolio_gate",
            "passed": True,
            "selected_family": "projected_0p001",
        },
    )
    portfolio_png = _write_png_stub(tmp_path / "portfolio.png")
    matrix_json = _write_json(
        tmp_path / "matrix.json",
        {"kind": "matched_nonlinear_transport_matrix_report", "passed": True},
    )
    matrix_png = _write_png_stub(tmp_path / "matrix.png")
    docs_static = tmp_path / "docs" / "_static"

    manifest = release_finalizer.finalize_release_artifacts(
        portfolio_json=portfolio,
        portfolio_figure=portfolio_png,
        matrix_report_jsons={"projected_0p001": matrix_json},
        matrix_report_figures={"projected_0p001": matrix_png},
        docs_static=docs_static,
        regenerate_dashboards=False,
    )

    assert manifest["passed"] is True
    assert manifest["selected_family"] == "projected_0p001"
    assert manifest["dashboards_regenerated"] is False
    assert (docs_static / "nonlinear_transport_matrix_portfolio.json").exists()
    assert (docs_static / "projected_0p001_matrix_report.json").exists()
    saved = json.loads(
        (
            docs_static / "nonlinear_transport_matrix_release_finalization.json"
        ).read_text(encoding="utf-8")
    )
    assert saved["kind"] == "nonlinear_transport_matrix_release_finalization"
    assert saved["selected_family"] == "projected_0p001"


def test_finalize_nonlinear_transport_matrix_release_refuses_blocked_portfolio(
    tmp_path: Path,
) -> None:
    portfolio = _write_json(
        tmp_path / "portfolio.json",
        {
            "kind": "nonlinear_transport_matrix_portfolio_gate",
            "passed": False,
            "selected_family": None,
        },
    )

    with pytest.raises(ValueError, match="blocked"):
        release_finalizer.finalize_release_artifacts(
            portfolio_json=portfolio,
            portfolio_figure=None,
            matrix_report_jsons={},
            matrix_report_figures={},
            docs_static=tmp_path / "docs" / "_static",
            regenerate_dashboards=False,
        )


def test_finalize_nonlinear_transport_matrix_release_cli_smoke(
    tmp_path: Path,
) -> None:
    portfolio = _write_json(
        tmp_path / "portfolio.json",
        {
            "kind": "nonlinear_transport_matrix_portfolio_gate",
            "passed": True,
            "selected_family": "accepted_qa_ess",
        },
    )
    portfolio_png = _write_png_stub(tmp_path / "portfolio.png")
    matrix_json = _write_json(
        tmp_path / "matrix.json",
        {"kind": "matched_nonlinear_transport_matrix_report", "passed": True},
    )
    matrix_png = _write_png_stub(tmp_path / "matrix.png")
    docs_static = tmp_path / "docs" / "_static"

    rc = release_finalizer.main(
        [
            "--portfolio-json",
            str(portfolio),
            "--portfolio-figure",
            str(portfolio_png),
            "--matrix-report-json",
            f"accepted_qa_ess={matrix_json}",
            "--matrix-report-figure",
            f"accepted_qa_ess={matrix_png}",
            "--docs-static",
            str(docs_static),
            "--skip-dashboard-regeneration",
        ]
    )

    assert rc == 0
    payload = json.loads(
        (
            docs_static / "nonlinear_transport_matrix_release_finalization.json"
        ).read_text(encoding="utf-8")
    )
    assert payload["selected_family"] == "accepted_qa_ess"


def test_overdetermined_writer_creates_multi_control_launch_manifest(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "input.final"
    baseline.write_text(_overdetermined_input_text(), encoding="utf-8")
    previous = tmp_path / "ranking.json"
    previous.write_text(
        json.dumps(
            {
                "passed": False,
                "recommendation": "use an overdetermined least-squares/profile-gradient campaign",
                "best_candidate": {"label": "zbs_1_1"},
            }
        ),
        encoding="utf-8",
    )

    rc = over_writer.main(
        [
            "--baseline-input",
            str(baseline),
            "--out-dir",
            str(tmp_path / "campaign"),
            "--case",
            "qa_profile",
            "--coefficient",
            "ZBS(1,0)",
            "--coefficient",
            "ZBS(1,1)",
            "--relative-delta",
            "0.05",
            "--previous-ranking",
            str(previous),
            "--horizons",
            "5,10",
            "--grid",
            "n8:8:4:4:4",
            "--window-tmin",
            "5",
            "--window-tmax",
            "10",
        ]
    )

    manifest_path = (
        tmp_path
        / "campaign"
        / "overdetermined_nonlinear_gradient_campaign_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert (
        manifest["kind"]
        == "overdetermined_nonlinear_turbulence_gradient_campaign_manifest"
    )
    assert manifest["control_count"] == 2
    assert manifest["previous_ranking"]["passed"] is False
    assert (
        "least-squares/profile-gradient"
        in manifest["previous_ranking"]["recommendation"]
    )
    assert [row["coefficient_slug"] for row in manifest["controls"]] == [
        "zbs_1_0",
        "zbs_1_1",
    ]
    assert all(
        "vmec_jax input." in row["vmec_run_commands"]["plus_delta"]
        for row in manifest["controls"]
    )
    assert all(
        "write_nonlinear_turbulence_gradient_campaign.py"
        in row["nonlinear_campaign_command_after_vmec_runs"]
        for row in manifest["controls"]
    )
    assert all(
        row["expected_nonlinear_campaign_manifest"].endswith(
            f"{row['coefficient_slug']}/gradient_campaign_manifest.json"
        )
        for row in manifest["controls"]
    )
    assert (
        "design_nonlinear_gradient.py rank-candidates"
        in manifest["promotion_contract"]["candidate_ranking_command"]
    )
    assert (
        "--campaign-context overdetermined_followup"
        in manifest["promotion_contract"]["candidate_ranking_command"]
    )
    assert len(manifest["promotion_contract"]["expected_fd_artifacts"]) == 2


def test_overdetermined_writer_requires_multiple_distinct_controls(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "input.final"
    baseline.write_text(_overdetermined_input_text(), encoding="utf-8")

    with pytest.raises(ValueError, match="at least two controls"):
        over_writer.write_overdetermined_campaign(
            baseline_input=baseline,
            out_dir=tmp_path / "campaign",
            case="bad",
            coefficients=(over_writer._parse_coefficient_spec("ZBS(1,0)"),),
            relative_delta=0.05,
        )

    with pytest.raises(ValueError, match="duplicate controls"):
        over_writer.write_overdetermined_campaign(
            baseline_input=baseline,
            out_dir=tmp_path / "campaign",
            case="bad",
            coefficients=(
                over_writer._parse_coefficient_spec("ZBS(1,0)"),
                over_writer._parse_coefficient_spec("ZBS(1,0)"),
            ),
            relative_delta=0.05,
        )


def _nested_overdetermined_manifest(tmp_path: Path, slug: str) -> Path:
    path = tmp_path / slug / "gradient_campaign_manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "kind": "nonlinear_turbulence_gradient_campaign_manifest",
                "state_ensemble_commands": {
                    "baseline": {
                        "direct_full_horizon_launch_commands": [
                            f"python3 -m spectraxgk.cli run-runtime-nonlinear --config tools_out/{slug}/baseline_seed31.toml --steps 2 --no-progress",
                        ],
                    },
                    "plus_delta": {
                        "direct_full_horizon_launch_commands": [
                            f"python3 -m spectraxgk.cli run-runtime-nonlinear --config tools_out/{slug}/plus_seed31.toml --steps 2 --no-progress",
                        ],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _overdetermined_manifest(tmp_path: Path) -> Path:
    zbs10 = _nested_overdetermined_manifest(tmp_path, "zbs_1_0")
    zbs11 = _nested_overdetermined_manifest(tmp_path, "zbs_1_1")
    return _write_json(
        tmp_path / "overdetermined.json",
        {
            "kind": "overdetermined_nonlinear_turbulence_gradient_campaign_manifest",
            "controls": [
                {
                    "coefficient_slug": "zbs_1_0",
                    "expected_nonlinear_campaign_manifest": str(zbs10),
                },
                {
                    "coefficient_slug": "zbs_1_1",
                    "expected_nonlinear_campaign_manifest": str(zbs11),
                },
            ],
        },
    )


def test_overdetermined_runner_preserves_task_order_and_filters_dry_runs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest_path = _overdetermined_manifest(tmp_path)
    manifest = over_runner.load_overdetermined_manifest(manifest_path)

    tasks = over_runner.collect_overdetermined_direct_tasks(manifest)

    assert [(task.state, task.label) for task in tasks] == [
        ("zbs_1_0:baseline", "baseline_seed31.out.nc"),
        ("zbs_1_0:plus_delta", "plus_seed31.out.nc"),
        ("zbs_1_1:baseline", "baseline_seed31.out.nc"),
        ("zbs_1_1:plus_delta", "plus_seed31.out.nc"),
    ]

    rc = over_runner.main(
        [
            "overdetermined",
            str(manifest_path),
            "--control",
            "zbs_1_1",
            "--gpu",
            "0",
            "--gpu",
            "1",
            "--dry-run",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "state=zbs_1_1:baseline" in out
    assert "state=zbs_1_0:baseline" not in out
    assert "gpu=0" in out
    assert "gpu=1" in out

    bad = tmp_path / "bad_runner.json"
    bad.write_text(json.dumps({"kind": "wrong"}), encoding="utf-8")
    with pytest.raises(ValueError, match="overdetermined"):
        over_runner.load_overdetermined_manifest(bad)


def _over_post_manifest(tmp_path: Path) -> Path:
    return _write_json(
        tmp_path / "campaign_plan.json",
        {
            "kind": "overdetermined_nonlinear_turbulence_gradient_campaign_manifest",
            "controls": [
                {
                    "coefficient_slug": "zbs_1_1",
                    "expected_nonlinear_campaign_manifest": (
                        "tools_out/overdetermined/nonlinear_campaigns/zbs_1_1/gradient_campaign_manifest.json"
                    ),
                },
                {
                    "coefficient_slug": "rbc_1_1",
                    "expected_nonlinear_campaign_manifest": (
                        "tools_out/overdetermined/nonlinear_campaigns/rbc_1_1/gradient_campaign_manifest.json"
                    ),
                },
            ],
            "promotion_contract": {
                "candidate_ranking_command": (
                    "python3 tools/campaigns/design_nonlinear_gradient.py rank-candidates "
                    "docs/_static/a.json docs/_static/b.json --json-out docs/_static/rank.json"
                )
            },
        },
    )


def test_overdetermined_postprocess_sequence_and_dry_run(tmp_path: Path) -> None:
    manifest_path = _over_post_manifest(tmp_path)
    manifest = over_post.load_overdetermined_manifest(manifest_path)

    commands = over_post.build_postprocess_commands(
        manifest,
        manifest_path=manifest_path,
        status_json=tmp_path / "status.json",
    )

    assert [command.step for command in commands] == [
        "per-control-postprocess",
        "per-control-postprocess",
        "candidate-ranking",
        "final-status",
    ]
    assert commands[0].label == "zbs_1_1"
    assert "--require-outputs" in commands[0].command
    assert "postprocess_summary.json" in commands[0].command
    assert "design_nonlinear_gradient.py rank-candidates" in commands[2].command
    assert "check_overdetermined_nonlinear_gradient_campaign.py" in commands[3].command
    assert "--fail-on-blocked" in commands[3].command

    summary = tmp_path / "summary.json"
    status = tmp_path / "status.json"
    rc = over_post.main_overdetermined_postprocess(
        [
            str(manifest_path),
            "--status-json",
            str(status),
            "--summary-json",
            str(summary),
            "--dry-run",
        ]
    )
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["kind"] == "overdetermined_nonlinear_gradient_postprocess_summary"
    assert payload["dry_run"] is True
    assert payload["passed"] is True
    assert len(payload["commands"]) == 4
    assert [row["returncode"] for row in payload["results"]] == [0, 0, 0, 0]

    bad = tmp_path / "bad_post.json"
    bad.write_text(
        json.dumps({"kind": "wrong", "controls": [], "promotion_contract": {}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="expected kind"):
        over_post.load_overdetermined_manifest(bad)


def _gradient_manifest() -> dict[str, object]:
    return {
        "kind": "nonlinear_turbulence_gradient_campaign_manifest",
        "state_ensemble_commands": {
            "minus_delta": {
                "expected_outputs": ["tmp/minus_seed31.out.nc"],
                "output_gate_command": "python3 tools/check_minus.py",
                "build_ensemble_command": "python3 tools/build_minus.py",
            },
            "baseline": {
                "expected_outputs": ["tmp/base_seed31.out.nc"],
                "output_gate_command": "python3 tools/check_base.py",
                "build_ensemble_command": "python3 tools/build_base.py",
            },
            "plus_delta": {
                "expected_outputs": ["tmp/plus_seed31.out.nc"],
                "output_gate_command": "python3 tools/check_plus.py",
                "build_ensemble_command": "python3 tools/build_plus.py",
            },
        },
        "promotion_contract": {
            "central_fd_command": "python3 tools/build_fd.py",
            "evidence_check_command": "python3 tools/check_evidence.py",
        },
    }


def test_gradient_manifest_postprocess_order_selection_and_failure_modes(
    tmp_path: Path,
) -> None:
    commands = gradient_post.collect_postprocess_commands(_gradient_manifest())
    assert [(item.step, item.label) for item in commands] == [
        ("output-gates", "baseline"),
        ("output-gates", "plus_delta"),
        ("output-gates", "minus_delta"),
        ("ensembles", "baseline"),
        ("ensembles", "plus_delta"),
        ("ensembles", "minus_delta"),
        ("central-fd", "promotion"),
        ("evidence", "promotion"),
    ]
    ensemble_commands = gradient_post.collect_postprocess_commands(
        _gradient_manifest(), steps={"ensembles"}
    )
    assert [item.command for item in ensemble_commands] == [
        "python3 tools/build_base.py",
        "python3 tools/build_plus.py",
        "python3 tools/build_minus.py",
    ]

    manifest_path = tmp_path / "manifest.json"
    summary_path = tmp_path / "summary.json"
    manifest_path.write_text(json.dumps(_gradient_manifest()), encoding="utf-8")
    rc = gradient_post.main_postprocess(
        [
            str(manifest_path),
            "--step",
            "central-fd",
            "--dry-run",
            "--summary-json",
            str(summary_path),
        ]
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert summary["passed"] is True
    assert summary["dry_run"] is True
    assert summary["results"] == [
        {
            "step": "central-fd",
            "label": "promotion",
            "command": "python3 tools/build_fd.py",
            "returncode": 0,
        }
    ]

    require_summary = tmp_path / "summary_require.json"
    rc = gradient_post.main_postprocess(
        [
            str(manifest_path),
            "--dry-run",
            "--require-outputs",
            "--summary-json",
            str(require_summary),
        ]
    )
    summary = json.loads(require_summary.read_text(encoding="utf-8"))
    assert rc == 2
    assert summary["passed"] is False
    assert len(summary["missing_expected_outputs"]) == 3

    bad = tmp_path / "bad_gradient.json"
    bad.write_text(json.dumps({"kind": "other"}), encoding="utf-8")
    with pytest.raises(
        ValueError, match="nonlinear_turbulence_gradient_campaign_manifest"
    ):
        gradient_post.load_postprocess_manifest(bad)
