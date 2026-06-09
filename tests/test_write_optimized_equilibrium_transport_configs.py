from __future__ import annotations

import json
from pathlib import Path

from tools.write_optimized_equilibrium_transport_configs import main


def test_write_optimized_equilibrium_transport_configs_contract(tmp_path: Path) -> None:
    vmec = tmp_path / "wout_optimized.nc"
    vmec.write_text("placeholder", encoding="utf-8")
    out_dir = tmp_path / "optimized_runs"

    assert (
        main(
            [
                "--vmec-file",
                str(vmec),
                "--case",
                "optimized_equilibrium_test",
                "--out-dir",
                str(out_dir),
                "--horizons",
                "1,2",
                "--grid",
                "n8:8:8:6:6",
                "--dt",
                "0.5",
                "--dt-variant",
                "0.25",
                "--window-tmin",
                "1",
                "--window-tmax",
                "2",
            ]
        )
        == 0
    )

    manifest = json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["kind"] == "external_vmec_holdout_config_manifest"
    assert manifest["claim_level"] == "optimized_equilibrium_transport_launch_plan_not_simulation_claim"
    assert len(manifest["configs"]) == 6
    assert len(manifest["launch_commands"]) == 6
    assert len(manifest["direct_full_horizon_launch_commands"]) == 6
    assert len(manifest["restart_seed_commands"]) == 3

    contract = manifest["promotion_contract"]
    assert contract["claim_level"] == "optimized_equilibrium_replicated_transport_window_launch_contract_not_promotion"
    assert len(contract["expected_outputs"]) == 3
    assert all("optimized_equilibrium_test_nonlinear_t2_n8" in path for path in contract["expected_outputs"])
    assert contract["build_ensemble_command"].startswith(
        "python3 tools/build_external_vmec_replicate_ensemble.py"
    )
    assert "--tmin 1" in contract["build_ensemble_command"]
    assert "--tmax 2" in contract["build_ensemble_command"]
    assert contract["direct_full_horizon_step_counts"] == {
        "dt0p25": 8,
        "seed31": 4,
        "seed32": 4,
    }
    assert any("--steps 8" in command for command in contract["direct_full_horizon_launch_commands"])
    assert contract["output_gate_command"].startswith("python3 tools/check_nonlinear_runtime_outputs.py")
    assert "--tmin 1 --tmax 2" in contract["output_gate_command"]
    assert "--min-window-samples 80" in contract["output_gate_command"]
    assert "restart-ladder segments" in contract["restart_ladder_note"]
    assert contract["run_guard_command"].startswith(
        "python3 tools/check_production_nonlinear_optimization_guard.py"
    )
    assert "--optimized-equilibrium-ensemble" in contract["run_guard_command"]


def test_strict_qa_t1500_contract_exposes_true_full_horizon_commands(tmp_path: Path) -> None:
    """Guard against launching a final restart segment from t=0 as a t1500 audit."""

    vmec = tmp_path / "wout_strict_qa.nc"
    vmec.write_text("placeholder", encoding="utf-8")
    out_dir = tmp_path / "strict_qa_runs"

    assert (
        main(
            [
                "--vmec-file",
                str(vmec),
                "--case",
                "vmec_qa_full_sweep_test",
                "--out-dir",
                str(out_dir),
                "--horizons",
                "700,1100,1500",
                "--grid",
                "n64:64:64:40:40",
                "--dt",
                "0.05",
                "--dt-variant",
                "0.04",
                "--window-tmin",
                "1100",
                "--window-tmax",
                "1500",
                "--seed-variant",
                "32",
                "--seed-variant",
                "33",
            ]
        )
        == 0
    )

    manifest = json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8"))
    t1500_segments = [command for command in manifest["launch_commands"] if "_nonlinear_t1500_" in command]
    t1500_direct = [
        command
        for command in manifest["direct_full_horizon_launch_commands"]
        if "_nonlinear_t1500_" in command
    ]
    assert len(t1500_segments) == 3
    assert len(t1500_direct) == 3
    assert all("--steps 8000" in command for command in t1500_segments[:2])
    assert "--steps 10000" in t1500_segments[-1]
    assert all("--steps 30000" in command for command in t1500_direct[:2])
    assert "--steps 37500" in t1500_direct[-1]

    contract = manifest["promotion_contract"]
    assert contract["direct_full_horizon_step_counts"] == {
        "dt0p04": 37500,
        "seed32": 30000,
        "seed33": 30000,
    }
    assert "check_nonlinear_runtime_outputs.py" in contract["output_gate_command"]
    assert "--tmin 1100 --tmax 1500" in contract["output_gate_command"]
