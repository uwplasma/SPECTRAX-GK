from __future__ import annotations

import json
from pathlib import Path

from support.paths import load_campaign_tool
from tools.campaigns.write_w7x_zonal_closure_sweep import (
    DEFAULT_CASES,
    SweepCase,
    build_manifest as build_w7x_zonal_closure_manifest,
    write_manifest as write_w7x_zonal_closure_manifest,
)


def test_w7x_zonal_closure_sweep_manifest_contract(tmp_path: Path) -> None:
    payload = build_w7x_zonal_closure_manifest(
        config=tmp_path / "runtime_w7x.toml",
        out_dir=tmp_path / "runs",
        kx=0.07,
        dt=0.05,
        steps=2000,
        nl=16,
        nm=64,
        sample_stride=4,
    )

    assert payload["kind"] == "w7x_zonal_closure_sweep_manifest"
    assert payload["kx"] == 0.07
    assert payload["Nl"] == 16
    assert payload["Nm"] == 64
    assert payload["checkpoint_steps"] == 500
    assert len(payload["cases"]) == len(DEFAULT_CASES)
    assert len(payload["launch_commands"]) == len(DEFAULT_CASES)
    assert any(case["family"] == "constant_mixed_lm" for case in payload["cases"])
    assert any(case["family"] == "constant_laguerre" for case in payload["cases"])
    assert any(case["family"] == "constant_isotropic" for case in payload["cases"])
    assert any(
        "--nu-hyper-lm 0.01" in command for command in payload["launch_commands"]
    )
    assert any("--nu-hyper-l 0.03" in command for command in payload["launch_commands"])
    assert any("--nu-hyper 0.01" in command for command in payload["launch_commands"])
    assert any(
        "--hypercollisions-kz 1" in command or "--hypercollisions-kz 1.0" in command
        for command in payload["launch_commands"]
    )
    assert all("--out-png" in command for command in payload["launch_commands"])
    assert all(
        "--checkpoint-steps 500" in command for command in payload["launch_commands"]
    )
    assert "build_w7x_zonal_recurrence_artifacts.py closure-ladder" in payload[
        "plot_command"
    ]
    assert "w7x_zonal_closure_ladder_full.png" in payload["plot_command"]
    assert payload["plot_outputs"]["png"].endswith("w7x_zonal_closure_ladder_full.png")

    single = build_w7x_zonal_closure_manifest(
        config=tmp_path / "runtime_w7x.toml",
        out_dir=tmp_path / "runs",
        cases=(
            SweepCase(
                slug="baseline",
                label="baseline",
                family="baseline",
                hypercollisions_const=0.0,
                hypercollisions_kz=0.0,
            ),
        ),
    )
    path = write_w7x_zonal_closure_manifest(tmp_path / "manifest.json", single)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["kind"] == "w7x_zonal_closure_sweep_manifest"
    assert loaded["cases"][0]["slug"] == "baseline"
    assert loaded["cases"][0]["panel_png"].endswith("baseline/panel.png")
    assert loaded["launch_commands"][0].startswith(
        "python3 tools/artifacts/build_w7x_zonal_validation_artifacts.py response-panel"
    )


def _write_summary_artifacts(root: Path, label: str, *, axis: str, mean: float) -> str:
    summary = root / f"state_{label}_transport_window.json"
    summary.write_text(
        json.dumps(
            {
                "kind": "nonlinear_window_summary",
                "variant_label": label,
                "variant_axis": axis,
            }
        ),
        encoding="utf-8",
    )
    reports = root / "nonlinear_window_convergence_reports"
    reports.mkdir(exist_ok=True)
    (reports / f"{summary.stem}.convergence.json").write_text(
        json.dumps(
            {
                "statistics": {
                    "late_mean": mean,
                    "running_mean_rel_drift": 0.03,
                    "terminal_mean_rel_delta": 0.02,
                    "sem_rel": 0.04,
                    "n_blocks": 8,
                }
            }
        ),
        encoding="utf-8",
    )
    return summary.name


def test_nonlinear_replicate_followup_writes_artifacts(tmp_path: Path) -> None:
    mod = load_campaign_tool("nonlinear_replicate_followup")
    seed31 = _write_summary_artifacts(tmp_path, "seed31", axis="seed", mean=10.0)
    seed32 = _write_summary_artifacts(tmp_path, "seed32", axis="seed", mean=11.5)
    dt0p04 = _write_summary_artifacts(tmp_path, "dt0p04", axis="timestep", mean=8.5)
    ensemble = {
        "case": "qa_ess_nonlinear_gradient_plus_delta_t900_ensemble",
        "passed": False,
        "statistics": {
            "ensemble_mean": 10.0,
            "mean_rel_spread": 0.30,
            "combined_sem_rel": 0.04,
        },
        "config": {"max_mean_rel_spread": 0.15},
        "rows": [
            {
                "index": 0,
                "late_mean": 10.0,
                "sem": 0.2,
                "summary_artifact": seed31,
                "source_artifact": "state_seed31_heat_flux_trace.csv",
                "passed": True,
                "promotion_ready": True,
            },
            {
                "index": 1,
                "late_mean": 11.5,
                "sem": 0.2,
                "summary_artifact": seed32,
                "source_artifact": "state_seed32_heat_flux_trace.csv",
                "passed": True,
                "promotion_ready": True,
            },
            {
                "index": 2,
                "late_mean": 8.5,
                "sem": 0.2,
                "summary_artifact": dt0p04,
                "source_artifact": "state_dt0p04_heat_flux_trace.csv",
                "passed": True,
                "promotion_ready": True,
            },
        ],
    }
    ensemble_path = tmp_path / "ensemble.json"
    ensemble_path.write_text(json.dumps(ensemble), encoding="utf-8")
    out_prefix = tmp_path / "spread"

    rc = mod.main(
        ["spread-summary", str(ensemble_path), "--out-prefix", str(out_prefix)]
    )

    payload = json.loads(out_prefix.with_suffix(".json").read_text(encoding="utf-8"))
    assert rc == 0
    assert out_prefix.with_suffix(".csv").exists()
    assert out_prefix.with_suffix(".png").exists()
    assert payload["state_rows"][0]["classification"] == "mixed_seed_timestep_spread"
    assert payload["replicate_rows"][0]["running_mean_rel_drift"] == 0.03
