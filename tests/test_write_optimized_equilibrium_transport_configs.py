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
    assert contract["run_guard_command"].startswith(
        "python3 tools/check_production_nonlinear_optimization_guard.py"
    )
    assert "--optimized-equilibrium-ensemble" in contract["run_guard_command"]
