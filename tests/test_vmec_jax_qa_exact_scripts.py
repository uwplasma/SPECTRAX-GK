from __future__ import annotations

from pathlib import Path
import py_compile
import re


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "optimization"
EXACT_QA_SCRIPTS = {
    "QA_optimization_with_growth_rate.py": "growth",
    "QA_optimization_with_quasilinear_flux.py": "quasilinear_flux",
    "QA_optimization_with_nonlinear_heat_flux.py": "nonlinear_window_heat_flux",
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
        r"QA_optimization_with_(growth_rate|quasilinear_flux|nonlinear_heat_flux)\.py\s*\\\n\s+--"
    )
    for path in docs:
        text = path.read_text(encoding="utf-8")
        assert cli_flag_after_exact_script.search(text) is None, path

    examples_readme = (EXAMPLES / "README.md").read_text(encoding="utf-8")
    assert "python examples/optimization/QA_optimization_with_growth_rate.py" in examples_readme
    assert "python examples/optimization/QA_optimization_with_quasilinear_flux.py" in examples_readme
    assert "python examples/optimization/QA_optimization_with_nonlinear_heat_flux.py" in examples_readme
    assert "python examples/optimization/vmec_jax_qa_low_turbulence_optimization.py" in examples_readme


def test_docs_scope_vmec_jax_transport_optimizer_claims() -> None:
    docs = [
        ROOT / "README.md",
        ROOT / "docs" / "stellarator_optimization.rst",
        EXAMPLES / "README.md",
    ]
    for path in docs:
        text = path.read_text(encoding="utf-8")
        normalized = re.sub(r"\s+", " ", text)
        assert "scalar_trust" in text, path
        assert "custom-VJP" in text or "custom VJP" in text, path
        assert "two-stage" in text, path
        assert "not a transport-optimization success claim" in normalized, path
