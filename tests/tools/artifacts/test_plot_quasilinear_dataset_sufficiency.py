"""Tests for the quasilinear dataset-sufficiency gate."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


def _load_tool_module():
    tools_dir = Path(__file__).resolve().parents[3] / "tools"
    sys.path.insert(0, str(tools_dir))
    path = tools_dir / "plot_quasilinear_dataset_sufficiency.py"
    spec = importlib.util.spec_from_file_location(
        "plot_quasilinear_dataset_sufficiency", path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_case(
    tmp_path: Path, name: str, *, gate_case: str, split: str, geometry: str
):
    spectrum = tmp_path / f"{name}_spectrum.csv"
    spectrum.write_text(
        "ky,gamma,kperp_eff2,heat_flux_weight_total\n0.1,0.2,1.0,2.0\n",
        encoding="utf-8",
    )
    summary = tmp_path / f"{name}_summary.json"
    summary.write_text(
        json.dumps(
            {
                "case": gate_case,
                "gate_report": {"case": gate_case, "passed": True, "gates": []},
            }
        ),
        encoding="utf-8",
    )
    shape_gate = tmp_path / f"{name}_shape.json"
    shape_gate.write_text(json.dumps({"passed": True}), encoding="utf-8")
    return spectrum, summary, shape_gate, split, geometry


def test_dataset_sufficiency_blocks_under_sampled_quasilinear_promotion(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    cases = []
    for name, gate_case, split, geometry in [
        ("cyclone", "cyclone_nonlinear_long_window", "train", "cyclone"),
        ("miller", "cyclone_miller_nonlinear_window", "holdout", "cyclone_miller"),
        ("hsx", "hsx_nonlinear_window", "holdout", "hsx"),
        ("w7x", "w7x_nonlinear_window", "holdout", "w7x"),
    ]:
        spectrum, summary, shape_gate, split, geometry = _write_case(
            tmp_path,
            name,
            gate_case=gate_case,
            split=split,
            geometry=geometry,
        )
        cases.append(
            mod.SaturationCase(name, split, geometry, spectrum, summary, shape_gate)
        )
    nonlinear_index = tmp_path / "nonlinear_index.json"
    nonlinear_index.write_text(
        json.dumps(
            {
                "cases": [
                    "cyclone_nonlinear_long_window",
                    "cyclone_miller_nonlinear_window",
                    "hsx_nonlinear_window",
                    "w7x_nonlinear_window",
                    "kbm_nonlinear_window",
                ],
                "case_gate_passed": {"kbm_nonlinear_window": True},
                "case_gate_thresholds": {"kbm_nonlinear_window": 0.02},
            }
        ),
        encoding="utf-8",
    )
    candidate_gate = tmp_path / "candidate.json"
    candidate_gate.write_text(
        json.dumps({"kind": "candidate", "promotion_gate": {"passed": False}}),
        encoding="utf-8",
    )
    saturation_gate = tmp_path / "saturation.json"
    saturation_gate.write_text(
        json.dumps({"kind": "saturation", "promotion_gate": {"passed": False}}),
        encoding="utf-8",
    )

    report = mod.build_dataset_sufficiency_report(
        tuple(cases),
        nonlinear_index=nonlinear_index,
        candidate_gate=candidate_gate,
        saturation_gate=saturation_gate,
    )

    assert report["kind"] == "quasilinear_dataset_sufficiency"
    assert report["input_validation"]["passed"] is True
    assert report["promotion_gate"]["passed"] is False
    assert "minimum_total_electrostatic_cases" in report["promotion_gate"]["blockers"]
    assert "minimum_explicit_train_geometries" in report["promotion_gate"]["blockers"]
    assert (
        "downstream_candidate_skill_gates_not_passed"
        in report["promotion_gate"]["blockers"]
    )
    assert report["requirements"]["current_total_cases"] == 4
    assert report["candidate_requirements"][0]["data_volume_passed"] is True
    assert report["candidate_requirements"][-1]["candidate"] == "linear_state_ridge"
    assert report["candidate_requirements"][-1]["data_volume_passed"] is False
    assert (
        report["excluded_validated_nonlinear_cases"][0]["case"]
        == "kbm_nonlinear_window"
    )
    assert (
        "electromagnetic" in report["excluded_validated_nonlinear_cases"][0]["reason"]
    )


def test_dataset_sufficiency_writes_artifacts(tmp_path: Path) -> None:
    mod = _load_tool_module()
    cases = []
    for idx, split in enumerate(
        ["train", "train", "holdout", "holdout", "holdout", "holdout"]
    ):
        spectrum, summary, shape_gate, split, geometry = _write_case(
            tmp_path,
            f"case{idx}",
            gate_case=f"case{idx}_gate",
            split=split,
            geometry=f"geom{idx}",
        )
        cases.append(
            mod.SaturationCase(
                f"case{idx}", split, geometry, spectrum, summary, shape_gate
            )
        )

    report = mod.build_dataset_sufficiency_report(
        tuple(cases),
        nonlinear_index=None,
        candidate_gate=None,
        saturation_gate=None,
    )
    paths = mod.write_dataset_sufficiency_figure(
        report,
        out=tmp_path / "dataset.png",
        title="Dataset sufficiency",
        dpi=80,
        write_pdf=False,
    )

    assert Path(paths["png"]).exists()
    assert Path(paths["json"]).exists()
    assert "pdf" not in paths
    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert (
        payload["claim_level"]
        == "scoped_low_parameter_candidate_promotion_not_runtime_option"
    )
