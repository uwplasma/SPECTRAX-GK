from __future__ import annotations

import json
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


def test_driver_defaults_to_multisample_current_api() -> None:
    mod = _load_driver()
    args = mod._parse_args(["--input", "input.seed", "--dry-run"])

    assert args.surfaces == mod.DEFAULT_SURFACES
    assert args.alphas == mod.DEFAULT_ALPHAS
    assert args.ky_values == mod.DEFAULT_KY_VALUES
    assert args.mode_schedule == (1, 2, 3, 4, 5)
    assert args.target_aspect == 6.0
    assert args.target_iota == 0.42
    assert args.jacobian == "auto"


def test_driver_dry_run_writes_current_api_summary(tmp_path: Path) -> None:
    outdir = tmp_path / "qa_growth_dry_run"
    completed = subprocess.run(
        [
            sys.executable,
            str(DRIVER),
            "--dry-run",
            "--input",
            "input.seed",
            "--mode-schedule",
            "1,2",
            "--transport-kind",
            "growth",
            "--surfaces",
            "0.64",
            "--alphas",
            "0.0",
            "--ky-values",
            "0.30",
            "--ntheta",
            "16",
            "--n-laguerre",
            "1",
            "--n-hermite",
            "1",
            "--transport-weight",
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

    summary = json.loads((outdir / "setup_summary.json").read_text(encoding="utf-8"))
    assert summary["api"] == "current_vmec_jax_opt_least_squares"
    assert summary["transport_kind"] == "growth"
    assert summary["sample_set"]["n_samples"] == 1
    assert summary["mode_schedule"] == [1, 2]
    assert summary["jacobian"] == "implicit"
    assert summary["targets"] == {"aspect": 6.0, "mean_iota": 0.42}
    assert summary["objectives"] == [
        "quasisymmetry",
        "aspect_ratio",
        "mean_iota",
        "growth",
    ]
    assert "screening evidence" in summary["claim_scope"]
    assert "current_vmec_jax_opt_least_squares" in completed.stdout
    assert not (outdir / "history.json").exists()


def test_driver_constraints_only_and_derivative_policy(tmp_path: Path) -> None:
    mod = _load_driver()
    args = mod._parse_args(
        ["--input", "input.seed", "--constraints-only", "--transport-kind", "growth"]
    )
    summary = mod._summary(args, jacobian="implicit")

    assert summary["transport_kind"] is None
    assert summary["objectives"] == ["quasisymmetry", "aspect_ratio", "mean_iota"]
    with pytest.raises(SystemExit):
        mod._parse_args(
            [
                "--input",
                "input.seed",
                "--transport-kind",
                "quasilinear_flux",
                "--jacobian",
                "implicit",
            ]
        )


def test_driver_surface_index_is_interior_and_validated() -> None:
    mod = _load_driver()
    assert mod._surface_index(0.64, 13) == 8
    assert mod._surface_index(0.99, 13) == 11
    with pytest.raises(ValueError, match="strictly inside"):
        mod._surface_index(1.0, 13)
    with pytest.raises(ValueError, match="five radial"):
        mod._surface_index(0.5, 4)


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--objective-scale", "0"),
        ("--ftol", "nan"),
        ("--max-nfev", "0"),
        ("--seed-perturbation", "inf"),
    ],
)
def test_driver_rejects_invalid_numerical_controls(option: str, value: str) -> None:
    mod = _load_driver()
    with pytest.raises(SystemExit):
        mod._parse_args(["--input", "input.seed", option, value])
