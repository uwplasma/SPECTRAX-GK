from __future__ import annotations

import json
import importlib.util
from pathlib import Path

from support.paths import REPO_ROOT, load_campaign_tool
import py_compile
import re
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import spectraxgk
from spectraxgk.objectives.vmec_candidate_admission import (
    build_solved_vmec_candidate_gate,
)


ROOT = REPO_ROOT
EXAMPLES = ROOT / "examples" / "optimization"
DRIVER = ROOT / "tools" / "campaigns" / "vmec_jax_qa_low_turbulence_optimization.py"
EXACT_QA_SCRIPTS = {
    "QA_optimization_linear_ITG.py": ("turbulent_growth_rate", 'JAC = "implicit"'),
    "QA_optimization_quasilinear_ITG.py": ("quasilinear_flux_proxy", "JAC = None"),
    "QA_optimization_nonlinear_ITG.py": ("nonlinear_heat_flux_proxy", "JAC = None"),
}


def _load_driver():
    return load_campaign_tool("vmec_jax_qa_low_turbulence_optimization")


def test_vmec_jax_style_qa_scripts_keep_upstream_iota_tuple_and_append_transport() -> (
    None
):
    for filename, (transport_function, jacobian_policy) in EXACT_QA_SCRIPTS.items():
        script = EXAMPLES / filename
        text = script.read_text(encoding="utf-8")

        py_compile.compile(str(script), doraise=True)
        assert "def main(" not in text
        assert "argparse" not in text
        assert "optimize_stellarator_itg" not in text
        assert "run_stellarator_itg_adam" not in text
        assert "current ``QA_optimization.py`` workflow" in text

        assert "MAX_MODE_SCHEDULE = (1, 2, 3, 4, 5)" in text
        assert "SEED_PERTURBATION = 0.01" in text
        assert "ASPECT_TARGET = 6.0" in text
        assert "IOTA_TARGET = 0.42" in text
        assert "SURFACE_INDEX = 7" in text
        assert "NTHETA = 24" in text
        assert jacobian_policy in text
        assert f"turb.{transport_function}(" in text
        assert "objective_terms = [" in text
        assert "(qs, 0.0, 1.0)," in text
        assert "(opt.aspect_ratio, ASPECT_TARGET, 1.0)," in text
        assert "(opt.mean_iota, IOTA_TARGET, 10.0)," in text
        assert "(transport_objective, 0.0, TRANSPORT_WEIGHT)," in text
        assert "result = opt.least_squares(" in text


def test_docs_do_not_show_exact_qa_scripts_as_argparse_drivers() -> None:
    docs = [
        ROOT / "README.md",
        ROOT / "docs" / "stellarator_optimization.rst",
        EXAMPLES / "README.md",
    ]
    cli_flag_after_exact_script = re.compile(
        r"QA_optimization_(linear_ITG|quasilinear_ITG|nonlinear_ITG)\.py\s*\\\n\s+--"
    )
    for path in docs:
        text = path.read_text(encoding="utf-8")
        assert cli_flag_after_exact_script.search(text) is None, path

    examples_readme = (EXAMPLES / "README.md").read_text(encoding="utf-8")
    assert (
        "python examples/optimization/QA_optimization_linear_ITG.py" in examples_readme
    )
    assert (
        "python examples/optimization/QA_optimization_quasilinear_ITG.py"
        in examples_readme
    )
    assert (
        "python examples/optimization/QA_optimization_nonlinear_ITG.py"
        in examples_readme
    )
    assert (
        "python examples/optimization/QA_nonlinear_ITG_matched_audit.py"
        in examples_readme
    )
    assert (
        "python examples/optimization/QA_nonlinear_ITG_transport_matrix.py"
        in examples_readme
    )
    assert (
        "python tools/campaigns/vmec_jax_qa_low_turbulence_optimization.py"
        in examples_readme
    )
    assert (
        "python examples/optimization/vmec_jax_qa_low_turbulence_optimization.py"
        not in examples_readme
    )


def test_exact_qa_scripts_help_does_not_launch_optimization(tmp_path: Path) -> None:
    for filename in EXACT_QA_SCRIPTS:
        script = EXAMPLES / filename

        completed = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert "Usage:" in completed.stdout
        assert "edit the constants" in completed.stdout.lower()
        assert "itg" in completed.stdout.lower()
        assert "objective" in completed.stdout.lower()
        assert not (tmp_path / "results").exists()


def test_exact_qa_scripts_reject_unexpected_arguments_before_outputs(
    tmp_path: Path,
) -> None:
    script = EXAMPLES / "QA_optimization_linear_ITG.py"

    completed = subprocess.run(
        [sys.executable, str(script), "--max-nfev", "1"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode != 0
    assert "unexpected arguments" in completed.stderr
    assert not (tmp_path / "results").exists()


def test_matched_nonlinear_audit_example_rebuilds_tracked_production_gate(
    tmp_path: Path,
) -> None:
    script = EXAMPLES / "QA_nonlinear_ITG_matched_audit.py"
    py_compile.compile(str(script), doraise=True)
    text = script.read_text(encoding="utf-8")

    assert "argparse" not in text
    assert "BASELINE_ENSEMBLE" in text
    assert "OPTIMIZED_ENSEMBLE" in text
    assert "MIN_RELATIVE_REDUCTION = 0.02" in text

    help_result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert "matched baseline-vs-optimized audit" in re.sub(
        r"\s+", " ", help_result.stdout
    )
    assert not (tmp_path / "results").exists()

    bad_arg = subprocess.run(
        [sys.executable, str(script), "--baseline-ensemble", "x.json"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert bad_arg.returncode != 0
    assert "unexpected arguments" in bad_arg.stderr
    assert not (tmp_path / "results").exists()

    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    payload_path = (
        tmp_path
        / "results/qa_opt/nonlinear_matched_audit/qa_nonlinear_ITG_matched_audit.json"
    )
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    summary = json.loads(completed.stdout)

    assert payload["passed"] is True
    assert payload["comparison"]["relative_reduction"] > 0.18
    assert payload["comparison"]["uncertainty_separation_sigma"] > 7.0
    assert summary["passed"] is True
    assert (
        tmp_path
        / "results/qa_opt/nonlinear_matched_audit/qa_nonlinear_ITG_matched_audit.png"
    ).exists()


def test_matched_nonlinear_matrix_example_writes_broad_campaign(tmp_path: Path) -> None:
    script = EXAMPLES / "QA_nonlinear_ITG_transport_matrix.py"
    py_compile.compile(str(script), doraise=True)
    text = script.read_text(encoding="utf-8")

    assert "argparse" not in text
    assert "def main(" not in text
    assert "BASELINE_VMEC_FILE" in text
    assert "CANDIDATE_VMEC_FILE" in text
    assert 'SURFACES = "0.45,0.64,0.78"' in text
    assert 'KY_VALUES = "0.10,0.30,0.50"' in text

    help_result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert "matched nonlinear ITG transport matrix" in help_result.stdout
    assert not (tmp_path / "results").exists()

    bad_arg = subprocess.run(
        [sys.executable, str(script), "--baseline-vmec-file", "x.nc"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert bad_arg.returncode != 0
    assert "unexpected arguments" in bad_arg.stderr
    assert not (tmp_path / "results").exists()

    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    summary = json.loads(completed.stdout)
    manifest_path = (
        tmp_path
        / "results/qa_opt/nonlinear_transport_matrix/matched_transport_matrix_manifest.json"
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert summary["sample_count"] == 18
    assert summary["coverage_passed"] is True
    assert payload["config"]["sample_count"] == 18
    assert payload["coverage_gate"]["passed"] is True
    assert len(payload["launch_scripts"]["final_horizon_gpu_splits"]) == 2
    assert (
        tmp_path
        / "results/qa_opt/nonlinear_transport_matrix/run_matrix_final_horizon_gpu0.sh"
    ).exists()


def test_docs_scope_vmec_jax_transport_optimizer_claims() -> None:
    docs = [
        ROOT / "README.md",
        ROOT / "docs" / "stellarator_optimization.rst",
        EXAMPLES / "README.md",
    ]
    for path in docs:
        text = path.read_text(encoding="utf-8")
        normalized = re.sub(r"\s+", " ", text)
        assert "transport" in text, path
        assert "nonlinear" in text, path
        assert "replicated" in text or "two-stage" in text, path
        if path.name != "README.md":
            assert "opt.least_squares" in text, path
            assert 'JAC="implicit"' in text, path
            assert "JAC=None" in text, path
            assert "not a nonlinear time average" in normalized, path
            assert (
                "tools/campaigns/finalize_nonlinear_transport_matrix_release.py" in text
            ), path


def test_optimization_examples_document_user_customization_knobs() -> None:
    examples_readme = (EXAMPLES / "README.md").read_text(encoding="utf-8")

    assert "QA Transport Optimizations" in examples_readme
    assert "MAX_MODE_SCHEDULE" not in examples_readme or "max_mode" in examples_readme
    assert "JAC" in examples_readme
    assert "SURFACE_INDEX" in examples_readme
    assert "ALPHA" in examples_readme
    assert "NTHETA" in examples_readme
    assert "SELECTED_KY_INDEX" in examples_readme
    assert "N_LAGUERRE" in examples_readme
    assert "N_HERMITE" in examples_readme
    assert "R_OVER_LT" in examples_readme
    assert "R_OVER_LN" in examples_readme
    assert "BASELINE_VMEC_FILE" in examples_readme
    assert "CANDIDATE_VMEC_FILE" in examples_readme
    assert "objective_terms" in examples_readme
    assert "opt.least_squares" in examples_readme
    assert "t=[1100,1500]" in examples_readme
    assert "Optimizer residuals, startup traces" in examples_readme


def test_readme_uses_solved_vmec_qa_geometry_not_reduced_surface_panel() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    docs = (ROOT / "docs" / "stellarator_optimization.rst").read_text(encoding="utf-8")
    manuscript = (ROOT / "docs" / "manuscript_figures.rst").read_text(encoding="utf-8")
    normalized_readme = re.sub(r"\s+", " ", readme)

    assert "docs/_static/vmec_jax_qa_full_sweep_panel.png" in readme
    assert "docs/_static/vmec_boundary_transport_landscape_rbc11_full.png" in readme
    assert "docs/_static/qa_itg_optimization_summary_panel.png" not in readme
    assert "docs/_static/vmec_jax_qa_solved_boundary_boozer_panel.png" not in readme
    assert "docs/_static/stellarator_itg_optimization_comparison.png" not in readme
    assert "docs/_static/stellarator_itg_optimization_uq.png" not in readme
    assert "not promoted turbulent-flux designs" in normalized_readme
    assert "matched long post-transient nonlinear" in normalized_readme

    assert "_static/vmec_jax_qa_full_sweep_panel.png" in docs
    assert "_static/vmec_boundary_transport_landscape_rbc11_full.png" in docs
    assert ".. figure:: _static/stellarator_itg_optimization_comparison.png" not in docs
    assert "Development-Only Reduced Diagnostics" in docs
    assert (
        "current artifact bases: ``docs/_static/stellarator_itg_optimization_comparison.png``"
        not in manuscript
    )
    assert "is not a solved-geometry optimization figure" in manuscript
    assert (
        "production QA optimization examples are the VMEC-JAX-style scripts"
        in manuscript
    )


def test_reduced_surface_comparison_is_not_current_primary_optimization_figure() -> (
    None
):
    readiness_source = (
        ROOT / "tools" / "artifacts" / "build_research_status.py"
    ).read_text(encoding="utf-8")
    examples_readme = (EXAMPLES / "README.md").read_text(encoding="utf-8")
    docs = (ROOT / "docs" / "stellarator_optimization.rst").read_text(encoding="utf-8")

    reduced_png = '"docs/_static/stellarator_itg_optimization_comparison.png"'
    primary_block = readiness_source.split('"supporting_artifacts"', maxsplit=1)[0]

    assert reduced_png not in primary_block
    assert "Do not use the reduced synthetic surface comparison" in readiness_source
    assert "stellarator_itg_growth_optimization.py" not in examples_readme
    assert "reduced_stellarator_itg" not in examples_readme
    assert "development diagnostics only" in re.sub(r"\s+", " ", docs)


def test_solved_wout_candidate_gate_passes_valid_qa_branch() -> None:
    assert (
        spectraxgk.build_solved_vmec_candidate_gate is build_solved_vmec_candidate_gate
    )
    result = SimpleNamespace(
        history={
            "aspect_final": 5.999233,
            "iota_final": 0.427011,
            "qs_final": 2.604013e-2,
        },
    )

    report = build_solved_vmec_candidate_gate(
        result,
        target_aspect=6.0,
        aspect_atol=5.0e-2,
        min_abs_mean_iota=0.41,
        qs_residual_max=5.0e-2,
        iota_profile_floor=0.41,
        iota_profiles=(
            np.asarray([0.0, 0.410131, 0.414]),
            np.asarray([0.410706, 0.414]),
        ),
    )

    assert report["passed"] is True
    assert report["checks"]["aspect"]["passed"] is True
    assert report["checks"]["mean_iota"]["passed"] is True
    assert report["checks"]["quasisymmetry"]["passed"] is True
    assert report["checks"]["iota_profile"]["passed"] is True
    json.dumps(report, allow_nan=False)


def test_solved_wout_candidate_gate_rejects_transport_branch_that_breaks_constraints() -> (
    None
):
    result = SimpleNamespace(
        history={
            "aspect_final": 5.996817,
            "iota_final": 0.425028,
            "qs_final": 1.091236e-1,
        },
    )

    report = build_solved_vmec_candidate_gate(
        result,
        target_aspect=6.0,
        aspect_atol=5.0e-2,
        min_abs_mean_iota=0.41,
        qs_residual_max=5.0e-2,
        iota_profile_floor=0.41,
        iota_profiles=(
            np.asarray([0.0, 0.402043, 0.414]),
            np.asarray([0.402493, 0.414]),
        ),
    )

    assert report["passed"] is False
    assert report["checks"]["aspect"]["passed"] is True
    assert report["checks"]["mean_iota"]["passed"] is True
    assert report["checks"]["quasisymmetry"]["passed"] is False
    assert report["checks"]["iota_profile"]["passed"] is False
    assert "do not promote" in report["next_action"]
    json.dumps(report, allow_nan=False)


def test_driver_transport_metric_from_result_uses_final_state_context() -> None:
    mod = _load_driver()

    class FakeTransport:
        config = SimpleNamespace(
            kind="growth", objective_transform="log1p", objective_scale=3.0
        )

        def J(self, ctx, state):
            assert state == "final-state"
            assert ctx.indata == "indata"
            assert ctx.signgs == -1
            assert ctx.flux == "flux"
            return np.asarray(0.125)

    result = SimpleNamespace(
        final_state="final-state",
        final_optimizer=SimpleNamespace(
            _static=SimpleNamespace(s=np.asarray([0.0, 1.0])),
            _indata="indata",
            _signgs=-1,
            _flux="flux",
        ),
    )

    metric = mod._transport_metric_from_result(FakeTransport(), result)

    assert metric["transport_objective_final"] == 0.125
    assert metric["spectrax_objective_final"] == 0.125
    assert metric["transport_metric_final"] == 0.125
    assert metric["transport_objective_source"] == "final_vmec_jax_state"
    assert metric["transport_metric_kind"] == "growth"
    json.dumps(metric, allow_nan=False)


def test_driver_defaults_to_multisample_transport_admission_set(monkeypatch) -> None:
    mod = _load_driver()
    monkeypatch.setattr(sys, "argv", [str(DRIVER), "--dry-run"])

    args = mod._parse_args()
    summary = spectraxgk.transport_objective_sample_summary(
        {
            "surfaces": args.surfaces,
            "alphas": args.alphas,
            "ky_values": args.ky_values,
        }
    )

    assert args.surfaces == mod.DEFAULT_TRANSPORT_SURFACES
    assert args.alphas == mod.DEFAULT_TRANSPORT_ALPHAS
    assert args.ky_values == mod.DEFAULT_TRANSPORT_KY_VALUES
    assert summary["passed"] is True
    assert summary["sample_count"] == 18


def test_driver_dry_run_cli_writes_transport_setup_summary(tmp_path: Path) -> None:
    if importlib.util.find_spec("vmec_jax") is None:
        pytest.skip("vmec_jax is optional")

    outdir = tmp_path / "qa_growth_dry_run"
    completed = subprocess.run(
        [
            sys.executable,
            str(DRIVER),
            "--dry-run",
            "--use-simple-seed",
            "--max-mode",
            "1",
            "--min-vmec-mode",
            "3",
            "--mboz",
            "21",
            "--nboz",
            "21",
            "--transport-kind",
            "growth",
            "--surfaces",
            "0.64",
            "--alphas",
            "0.0",
            "--ky-values",
            "0.30",
            "--ntheta",
            "4",
            "--n-laguerre",
            "1",
            "--n-hermite",
            "1",
            "--surface-chunk-size",
            "1",
            "--spectrax-weight",
            "0.001",
            "--outdir",
            str(outdir),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

    summary_path = outdir / "setup_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["constraints_only"] is False
    assert summary["transport_kind"] == "growth"
    assert summary["requested_input"].endswith("input.minimal_seed_nfp2")
    assert summary["objectives"] == [
        "aspect",
        "iota",
        "iota_profile_floor",
        "qs",
        "spectraxgk_transport",
    ]
    assert summary["sample_set"]["n_samples"] == 1
    assert summary["spectrax_config"]["mboz"] == 21
    assert summary["spectrax_config"]["nboz"] == 21
    assert summary["spectrax_config"]["surface_chunk_size"] == 1
    assert summary["optimizer"]["method"] == "scalar_trust"
    assert summary["optimizer_comparison"]["schema_version"] == 1
    assert summary["optimizer_comparison"]["method"] == "scalar_trust"
    assert (
        summary["optimizer_comparison"]["comparison_class"]
        == "spectraxgk_transport_growth"
    )
    assert summary["optimizer_comparison"]["sample_set_fingerprint"]["mboz"] == 21
    assert summary["optimizer_comparison"]["sample_set_fingerprint"]["nboz"] == 21
    assert summary["optimizer_comparison"]["sample_set_fingerprint"]["n_samples"] == 1
    assert (
        summary["optimizer_comparison"]["nonlinear_promotion_policy"][
            "recommended_horizons"
        ]
        == "700,1100,1500"
    )
    assert (
        summary["optimizer_comparison"]["nonlinear_promotion_policy"][
            "recommended_window_tmin"
        ]
        == 1100.0
    )
    assert (
        "production nonlinear flux claims require matched long-window"
        in summary["claim_scope"]
    )
    assert "spectraxgk_transport" in completed.stdout
    assert not (outdir / "history.json").exists()
    assert not (outdir / "solved_wout_gate.json").exists()


def test_driver_strict_upstream_qa_baseline_preset_is_admission_grade(
    tmp_path: Path,
) -> None:
    if importlib.util.find_spec("vmec_jax") is None:
        pytest.skip("vmec_jax is optional")

    outdir = tmp_path / "strict_qa_baseline"
    subprocess.run(
        [
            sys.executable,
            str(DRIVER),
            "--dry-run",
            "--strict-upstream-qa-baseline",
            "--outdir",
            str(outdir),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

    summary = json.loads((outdir / "setup_summary.json").read_text(encoding="utf-8"))

    assert summary["constraints_only"] is True
    assert summary["use_simple_seed"] is True
    assert summary["requested_input"].endswith("input.minimal_seed_nfp2")
    assert summary["max_mode"] == 5
    assert summary["min_vmec_mode"] == 7
    assert summary["target_aspect"] == 5.0
    assert summary["min_iota"] == pytest.approx(0.4102)
    assert summary["strict_upstream_qa_baseline"] is True
    assert summary["strict_iota_admission_buffer"] == pytest.approx(2.0e-4)
    assert summary["iota_objective"] == "target"
    assert summary["iota_profile_floor"] is None
    assert summary["objectives"] == ["aspect", "iota", "qs"]
    assert summary["optimizer"]["method"] == "scipy"
    assert summary["optimizer_comparison"]["method"] == "scipy"
    assert summary["optimizer_comparison"]["comparison_class"] == "constraints_only_qa"
    assert summary["optimizer_comparison"]["transport_kind"] is None
    assert summary["optimizer"]["scipy_tr_solver"] == "exact"
    assert summary["optimizer"]["max_nfev"] >= 80
    assert summary["optimizer"]["inner_max_iter"] >= 180
    assert summary["optimizer"]["trial_max_iter"] >= 180
    assert summary["optimizer"]["ftol"] <= 1.0e-5
    assert summary["optimizer"]["gtol"] <= 1.0e-5
    assert summary["optimizer"]["xtol"] <= 1.0e-8
    assert summary["optimizer"]["use_ess"] is True
    assert summary["optimizer"]["ess_alpha"] == 1.2
    assert summary["optimizer"]["strict_upstream_qa_baseline"] is True
    assert summary["optimizer"]["save_rerun_wouts"] is True
    assert summary["optimizer"]["require_rerun_wout_gate"] is True
    assert summary["optimizer"]["admit_authoritative_rerun_wout"] is False
    assert summary["optimizer"]["wout_repro_mean_iota_atol"] == pytest.approx(5.0e-4)
    assert summary["solved_wout_gate_policy"]["min_abs_mean_iota"] == 0.41


def test_driver_updates_history_with_transport_metric(tmp_path: Path) -> None:
    mod = _load_driver()
    path = tmp_path / "history.json"
    path.write_text(
        json.dumps({"objective_final": 1.0, "history": [{"objective": 1.0}]}),
        encoding="utf-8",
    )

    mod._update_history_with_transport_metric(
        path,
        {
            "transport_objective_final": 0.2,
            "spectrax_objective_final": 0.2,
            "transport_metric_final": 0.2,
            "transport_objective_error": None,
        },
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["transport_objective_final"] == 0.2
    assert payload["spectrax_objective_final"] == 0.2
    assert payload["transport_metric_final"] == 0.2
    assert "transport_objective_error" not in payload
    assert payload["history"][-1]["transport_objective_final"] == 0.2
