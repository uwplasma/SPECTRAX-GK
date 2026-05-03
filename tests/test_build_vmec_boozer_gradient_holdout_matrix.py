"""Tests for the VMEC/Boozer multi-equilibrium gradient holdout matrix."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "build_vmec_boozer_gradient_holdout_matrix.py"
spec = importlib.util.spec_from_file_location("build_vmec_boozer_gradient_holdout_matrix", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def _write_gate(path: Path, *, case: str, kind: str, passed: bool = True) -> None:
    path.write_text(
        json.dumps(
            {
                "kind": kind,
                "case_name": case,
                "passed": passed,
                "source_scope": "mode21_vmec_boozer_state",
                "mboz": 21,
                "nboz": 21,
                "surface_stencil_width": None,
                "objective_gates": [
                    {
                        "objective": "gamma",
                        "passed": passed,
                        "rel_error": 1.0e-3,
                        "abs_error": 2.0e-4,
                    },
                    {
                        "objective": "omega",
                        "passed": True,
                        "rel_error": 2.0e-3,
                        "abs_error": 3.0e-4,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def test_gradient_holdout_matrix_summarizes_passed_gates(tmp_path: Path) -> None:
    qh_freq = tmp_path / "qh_freq.json"
    qh_ql = tmp_path / "qh_ql.json"
    li_freq = tmp_path / "li_freq.json"
    li_ql = tmp_path / "li_ql.json"
    for path, case, kind in [
        (qh_freq, "nfp4_QH_warm_start", "frequency"),
        (qh_ql, "nfp4_QH_warm_start", "quasilinear"),
        (li_freq, "li383_low_res", "frequency"),
        (li_ql, "li383_low_res", "quasilinear"),
    ]:
        _write_gate(path, case=case, kind=kind)

    payload = mod.build_gradient_holdout_matrix(
        (
            ("QH", "frequency", qh_freq),
            ("QH", "quasilinear", qh_ql),
            ("Li383", "frequency", li_freq),
            ("Li383", "quasilinear", li_ql),
        )
    )

    assert payload["kind"] == "vmec_boozer_gradient_holdout_matrix"
    assert payload["passed"] is True
    assert payload["summary"]["n_cases"] == 2
    assert payload["summary"]["max_relative_error"] == 2.0e-3
    assert payload["rows"][0]["objectives"]["gamma"] is True


def test_gradient_holdout_matrix_writes_artifacts(tmp_path: Path) -> None:
    gate = tmp_path / "gate.json"
    _write_gate(gate, case="case", kind="frequency")
    payload = mod.build_gradient_holdout_matrix((("case", "frequency", gate),))

    paths = mod.write_gradient_holdout_matrix(payload, out=tmp_path / "matrix.png")

    for path in paths.values():
        assert Path(path).exists()
    saved = json.loads((tmp_path / "matrix.json").read_text(encoding="utf-8"))
    assert saved["claim_level"] == (
        "multi_equilibrium_reduced_linear_quasilinear_and_nonlinear_window_estimator_gradient_gate_"
        "not_production_nonlinear_optimization"
    )
