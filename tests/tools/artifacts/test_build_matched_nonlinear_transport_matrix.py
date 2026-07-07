from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tools" / "build_matched_nonlinear_transport_matrix.py"
spec = importlib.util.spec_from_file_location("build_matched_nonlinear_transport_matrix", SCRIPT)
assert spec is not None
assert spec.loader is not None
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def _write_campaign(tmp_path: Path) -> Path:
    baseline = tmp_path / "baseline_wout.nc"
    candidate = tmp_path / "candidate_wout.nc"
    baseline.write_text("baseline vmec placeholder\n", encoding="utf-8")
    candidate.write_text("candidate vmec placeholder\n", encoding="utf-8")
    rc = mod.main(
        [
            "write",
            "--baseline-vmec-file",
            str(baseline),
            "--candidate-vmec-file",
            str(candidate),
            "--baseline-label",
            "strict_qa",
            "--candidate-label",
            "low_transport",
            "--case-prefix",
            "qa_matrix_test",
            "--out-dir",
            str(tmp_path / "campaign"),
            "--artifact-dir",
            str(tmp_path / "artifacts"),
            "--grid",
            "n8:8:8:8:8",
            "--horizons",
            "10,20",
            "--window-tmin",
            "10",
            "--window-tmax",
            "20",
            "--dt",
            "0.1",
            "--dt-variant",
            "0.08",
            "--seed-variant",
            "7",
            "--min-samples",
            "4",
            "--min-window-samples",
            "2",
            "--gpu-splits",
            "2",
        ]
    )
    assert rc == 0
    return tmp_path / "campaign" / "matched_transport_matrix_manifest.json"


def test_write_campaign_defaults_to_eighteen_point_transport_matrix(tmp_path: Path) -> None:
    manifest_path = _write_campaign(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert payload["kind"] == "matched_nonlinear_transport_matrix_campaign"
    assert payload["config"]["sample_count"] == 18
    assert payload["coverage_gate"] == {"min_alphas": 2, "min_ky_values": 3, "min_surfaces": 3, "passed": True}
    assert payload["config"]["surfaces"] == [0.45, 0.64, 0.78]
    assert payload["config"]["alphas"] == [0.0, 0.7853981633974483]
    assert payload["config"]["ky_values"] == [0.1, 0.3, 0.5]
    assert payload["config"]["seed_variants"] == [7]
    assert payload["config"]["dt_variants"] == [0.08]
    assert payload["config"]["final_horizon_launch_locking"] == "per-output flock with mkdir fallback"
    assert Path(payload["launch_scripts"]["staged_ladder_skip_existing"]).exists()
    assert Path(payload["launch_scripts"]["postprocess"]).exists()
    assert Path(payload["launch_scripts"]["final_horizon_direct_skip_existing"]).exists()
    assert len(payload["launch_scripts"]["final_horizon_gpu_splits"]) == 2
    assert all(Path(path).exists() for path in payload["launch_scripts"]["final_horizon_gpu_splits"])
    assert "build_matched_nonlinear_transport_matrix.py report" in payload["aggregate_report"]["command"]
    final_script = Path(payload["launch_scripts"]["final_horizon_direct_skip_existing"]).read_text(
        encoding="utf-8"
    )
    assert "_nonlinear_t20_" in final_script
    assert "_nonlinear_t10_" not in final_script
    assert "--steps 200" in final_script
    assert "tools/check_nonlinear_output_target.py" in final_script
    assert "flock -n 9" in final_script
    assert "lock_dir=${lock_file}.d" in final_script
    assert "skip-locked" in final_script
    assert "skip-target-confirmed" in final_script
    assert "skip-target-confirmed-after-lock" in final_script
    assert "skip-existing" not in final_script
    gpu1_script = Path(payload["launch_scripts"]["final_horizon_gpu_splits"][1]).read_text(
        encoding="utf-8"
    )
    assert "export DEVICE=1" in gpu1_script

    first = payload["samples"][0]
    assert first["sample_id"] == "s0p45_a0_ky0p1"
    assert first["surface_torflux"] == 0.45
    assert first["alpha"] == 0.0
    assert first["ky"] == 0.1
    assert set(first["states"]) == {"baseline", "candidate"}
    toml = Path(first["states"]["baseline"]["state_manifest"]).parent / (
        "qa_matrix_test_strict_qa_s0p45_a0_ky0p1_nonlinear_t20_n8_seed7.toml"
    )
    text = toml.read_text(encoding="utf-8")
    assert "torflux = 0.45" in text
    assert "alpha = 0" in text
    assert "ky = 0.1" in text
    assert "random_seed = 7" in text


def test_report_passes_when_all_matrix_comparisons_pass(tmp_path: Path) -> None:
    manifest_path = _write_campaign(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for index, sample in enumerate(manifest["samples"]):
        comparison = Path(sample["comparison"]["json"])
        comparison.parent.mkdir(parents=True, exist_ok=True)
        comparison.write_text(
            json.dumps(
                {
                    "kind": "matched_nonlinear_transport_comparison",
                    "passed": True,
                    "baseline": {"ensemble_mean": 10.0 + 0.1 * index},
                    "candidate": {"ensemble_mean": 9.5 + 0.1 * index},
                    "statistics": {"relative_reduction": 0.05, "uncertainty_z_score": 3.5},
                }
            ),
            encoding="utf-8",
        )

    out_json = tmp_path / "matrix_report.json"
    out_png = tmp_path / "matrix_report.png"
    rc = mod.main(
        [
            "report",
            "--matrix-manifest",
            str(manifest_path),
            "--out-json",
            str(out_json),
            "--out-figure",
            str(out_png),
            "--min-pass-fraction",
            "1.0",
            "--min-mean-relative-reduction",
            "0.02",
            "--fail-on-blocked",
        ]
    )
    report = json.loads(out_json.read_text(encoding="utf-8"))

    assert rc == 0
    assert report["passed"] is True
    assert report["summary"]["total_samples"] == 18
    assert report["summary"]["passed_samples"] == 18
    assert report["summary"]["mean_relative_reduction"] == pytest.approx(0.05)
    assert report["blockers"] == []
    assert out_png.exists()
