from __future__ import annotations

import json
from pathlib import Path
import py_compile
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "optimization"
EXACT_QA_SCRIPTS = {
    "QA_optimization_linear_ITG.py": "growth",
    "QA_optimization_quasilinear_ITG.py": "quasilinear_flux",
    "QA_optimization_nonlinear_ITG.py": "nonlinear_window_heat_flux",
}


def test_vmec_jax_style_qa_scripts_keep_upstream_iota_tuple_and_append_transport() -> None:
    for filename, kind in EXACT_QA_SCRIPTS.items():
        script = EXAMPLES / filename
        text = script.read_text(encoding="utf-8")

        py_compile.compile(str(script), doraise=True)
        assert "def main(" not in text
        assert "argparse" not in text
        assert "optimize_stellarator_itg" not in text
        assert "run_stellarator_itg_adam" not in text
        assert "examples/optimization/QA_optimization.py" in text

        assert "MAX_MODE = 5" in text
        assert "MIN_VMEC_MODE = MAX_MODE + 2" in text
        assert "USE_SIMPLE_SEED = True" in text
        assert "METHOD = \"scalar_trust\"" in text
        assert "custom VJP" in text
        assert "Pure VMEC-JAX QA only" in text
        assert "SCIPY_TR_SOLVER = \"exact\"" in text
        assert "TARGET_ASPECT = 5.0" in text
        assert "TARGET_IOTA = 0.41" in text
        assert "IOTA_WEIGHT = 10_000.0" in text
        assert "SPECTRAX_MBOZ = 21" in text
        assert "SPECTRAX_NBOZ = 21" in text
        assert f'SPECTRAX_KIND = "{kind}"' in text
        assert 'NONLINEAR_AUDIT_HORIZONS = "700,1100,1500"' in text
        assert "NONLINEAR_AUDIT_WINDOW_TMIN = 1100.0" in text
        assert "NONLINEAR_AUDIT_WINDOW_TMAX = 1500.0" in text
        assert "NONLINEAR_AUDIT_SEED_VARIANTS = (32, 33)" in text

        assert "aspect = vj.AspectRatio()" in text
        assert "iota = vj.MeanIota()" in text
        assert "qs = vj.QuasisymmetryRatioResidual(" in text
        assert "transport = VMECJAXSpectraxTransportObjective(" in text
        assert "VMECJAXTransportObjectiveConfig(" in text
        assert "objective_tuples = [" in text
        assert "(aspect.J, TARGET_ASPECT, ASPECT_WEIGHT)," in text
        assert "(iota.J, TARGET_IOTA, IOTA_WEIGHT)," in text
        assert "(qs.J, 0.0, QS_WEIGHT)," in text
        assert "(transport.J, 0.0, SPECTRAX_WEIGHT)," in text
        assert "problem = vj.LeastSquaresProblem.from_tuples(objective_tuples)" in text
        assert "result = vj.least_squares_solve(" in text


def test_docs_do_not_show_exact_qa_scripts_as_argparse_drivers() -> None:
    docs = [ROOT / "README.md", ROOT / "docs" / "stellarator_optimization.rst", EXAMPLES / "README.md"]
    cli_flag_after_exact_script = re.compile(
        r"QA_optimization_(linear_ITG|quasilinear_ITG|nonlinear_ITG)\.py\s*\\\n\s+--"
    )
    for path in docs:
        text = path.read_text(encoding="utf-8")
        assert cli_flag_after_exact_script.search(text) is None, path

    examples_readme = (EXAMPLES / "README.md").read_text(encoding="utf-8")
    assert "python examples/optimization/QA_optimization_linear_ITG.py" in examples_readme
    assert "python examples/optimization/QA_optimization_quasilinear_ITG.py" in examples_readme
    assert "python examples/optimization/QA_optimization_nonlinear_ITG.py" in examples_readme
    assert "python examples/optimization/QA_nonlinear_ITG_matched_audit.py" in examples_readme
    assert "python examples/optimization/QA_nonlinear_ITG_transport_matrix.py" in examples_readme
    assert "python tools/vmec_jax_qa_low_turbulence_optimization.py" in examples_readme
    assert "python examples/optimization/vmec_jax_qa_low_turbulence_optimization.py" not in examples_readme


def test_exact_qa_scripts_help_does_not_launch_optimization(tmp_path: Path) -> None:
    for filename, kind in EXACT_QA_SCRIPTS.items():
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
        assert "edit the constants" in completed.stdout
        assert ("linear" if kind == "growth" else kind.split("_")[0]) in completed.stdout
        assert not (tmp_path / "results").exists()


def test_exact_qa_scripts_reject_unexpected_arguments_before_outputs(tmp_path: Path) -> None:
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


def test_matched_nonlinear_audit_example_rebuilds_tracked_production_gate(tmp_path: Path) -> None:
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
    assert "matched baseline-vs-optimized audit" in re.sub(r"\s+", " ", help_result.stdout)
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
    assert "SURFACES = \"0.45,0.64,0.78\"" in text
    assert "KY_VALUES = \"0.10,0.30,0.50\"" in text

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
    manifest_path = tmp_path / "results/qa_opt/nonlinear_transport_matrix/matched_transport_matrix_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert summary["sample_count"] == 18
    assert summary["coverage_passed"] is True
    assert payload["config"]["sample_count"] == 18
    assert payload["coverage_gate"]["passed"] is True
    assert len(payload["launch_scripts"]["final_horizon_gpu_splits"]) == 2
    assert (
        tmp_path / "results/qa_opt/nonlinear_transport_matrix/run_matrix_final_horizon_gpu0.sh"
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
            assert "scalar_trust" in text, path
            assert "custom-VJP" in text or "custom VJP" in text, path
            assert "not a transport-optimization success claim" in normalized, path
            assert "tools/finalize_nonlinear_transport_matrix_release.py" in text, path


def test_optimization_examples_document_user_customization_knobs() -> None:
    examples_readme = (EXAMPLES / "README.md").read_text(encoding="utf-8")

    assert "How To Modify The Optimization Examples" in examples_readme
    assert "METHOD" in examples_readme
    assert "SCIPY_TR_SOLVER" in examples_readme
    assert "WARM_START_INPUT_FILE" in examples_readme
    assert "SIMPLE_SEED_INPUT_FILE" in examples_readme
    assert "BASELINE_VMEC_FILE" in examples_readme
    assert "CANDIDATE_VMEC_FILE" in examples_readme
    assert "SPECTRAX_KIND" in examples_readme
    assert "SPECTRAX_SURFACES" in examples_readme
    assert "SPECTRAX_ALPHAS" in examples_readme
    assert "SPECTRAX_KY_VALUES" in examples_readme
    assert "NONLINEAR_AUDIT_*" in examples_readme
    assert "objective_tuples" in examples_readme
    assert "mboz,nboz >= 21" in examples_readme
    assert "long post-transient replicated windows over `t=[1100,1500]`" in examples_readme
    assert "Do not promote optimizer residuals or startup traces" in examples_readme


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
    assert "production QA optimization examples are the VMEC-JAX-style scripts" in manuscript


def test_reduced_surface_comparison_is_not_current_primary_optimization_figure() -> None:
    readiness_source = (ROOT / "tools" / "artifacts" / "build_manuscript_readiness_status.py").read_text(
        encoding="utf-8"
    )
    examples_readme = (EXAMPLES / "README.md").read_text(encoding="utf-8")
    docs = (ROOT / "docs" / "stellarator_optimization.rst").read_text(encoding="utf-8")

    reduced_png = '"docs/_static/stellarator_itg_optimization_comparison.png"'
    primary_block = readiness_source.split('"supporting_artifacts"', maxsplit=1)[0]

    assert reduced_png not in primary_block
    assert "Do not use the reduced synthetic surface comparison" in readiness_source
    assert "stellarator_itg_growth_optimization.py" not in examples_readme
    assert "reduced_stellarator_itg" not in examples_readme
    assert "development diagnostics only" in re.sub(r"\s+", " ", docs)
