from __future__ import annotations

import json
from pathlib import Path

import pytest

from support.paths import load_campaign_tool
from tools.campaigns.write_external_vmec_holdout_configs import (
    _parse_grid,
    _parse_horizons,
    _parse_seed_dt_variant,
    write_configs,
    write_manifest,
)
from tools.campaigns.write_optimized_equilibrium_transport_configs import (
    main as optimized_transport_main,
)


def test_external_vmec_restart_ladder_config_contract(tmp_path: Path) -> None:
    vmec_file = tmp_path / "wout_fixture.nc"
    vmec_file.write_text("placeholder", encoding="utf-8")

    written = write_configs(
        case="candidate",
        vmec_file=vmec_file,
        out_dir=tmp_path / "runs",
        grids=[_parse_grid("n8:8:8:6:6"), _parse_grid("n10:10:10:8:8")],
        horizons=(1.0, 1.5, 2.0),
        dt=0.25,
        ky=0.3,
        nl=2,
        nm=3,
    )
    assert len(written) == 6
    assert [item.steps for item in written] == [4, 4, 2, 2, 2, 2]
    assert [item.restart_if_exists for item in written] == [
        False,
        False,
        True,
        True,
        True,
        True,
    ]

    first_config = written[0].path.read_text(encoding="utf-8")
    assert 'vmec_file = "../wout_fixture.nc"' in first_config
    assert "Nx = 8" in first_config
    assert "Nz = 6" in first_config
    assert "ky = 0.3" in first_config
    assert "steps = 4" in first_config
    assert "restart_if_exists = false" in first_config
    assert "nsave = 4" in first_config
    assert 'path = "candidate_nonlinear_t1_n8.out.nc"' in first_config

    continuation_config = written[2].path.read_text(encoding="utf-8")
    assert "t_max = 1.5" in continuation_config
    assert "steps = 2" in continuation_config
    assert "nsave = 2" in continuation_config
    assert "restart_if_exists = true" in continuation_config

    manifest = write_manifest(tmp_path / "runs", written)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["kind"] == "external_vmec_holdout_config_manifest"
    assert len(payload["configs"]) == 6
    assert len(payload["launch_commands"]) == 6
    assert len(payload["direct_full_horizon_launch_commands"]) == 6
    assert len(payload["staged_ladder_commands"]) == 10
    assert len(payload["restart_seed_commands"]) == 4
    assert len(payload["launch_skip_existing_commands"]) == 6
    assert len(payload["direct_full_horizon_skip_existing_launch_commands"]) == 6
    assert len(payload["staged_ladder_skip_existing_commands"]) == 10
    assert len(payload["restart_seed_skip_existing_commands"]) == 4
    assert payload["configs"][0]["dt"] == 0.25
    assert payload["configs"][2]["steps"] == 2
    assert payload["configs"][2]["direct_full_horizon_steps"] == 6
    assert payload["segment_step_counts"]["candidate_nonlinear_t1p5_n8"] == 2
    assert (
        payload["direct_full_horizon_step_counts"]["candidate_nonlinear_t1p5_n8"] == 6
    )
    assert payload["direct_full_horizon_step_counts"]["candidate_nonlinear_t2_n10"] == 8
    assert payload["launch_commands"][0].startswith(
        "PYTHONPATH=src CUDA_VISIBLE_DEVICES=${DEVICE:-0}"
    )
    assert (
        "python3 -m spectraxgk.cli run-runtime-nonlinear"
        in payload["launch_commands"][0]
    )
    assert payload["direct_full_horizon_launch_commands"][0].startswith(
        "PYTHONPATH=src CUDA_VISIBLE_DEVICES=${DEVICE:-0}"
    )
    assert "--steps 4" in payload["launch_commands"][0]
    assert "--steps 2" in payload["launch_commands"][2]
    assert "--steps 6" in payload["direct_full_horizon_launch_commands"][2]
    assert "restart.nc" in payload["restart_seed_commands"][0]
    assert "restart-ladder segments" in payload["restart_ladder_note"]
    assert "candidate_nonlinear_t1p5_n8" in payload["restart_seed_commands"][0]
    assert "candidate_nonlinear_t1p5_n8" in payload["restart_seed_commands"][2]
    assert "candidate_nonlinear_t2_n8" in payload["restart_seed_commands"][2]
    assert payload["staged_ladder_commands"][2] == payload["restart_seed_commands"][0]
    assert payload["staged_ladder_commands"][3] == payload["launch_commands"][2]
    guarded = payload["staged_ladder_skip_existing_commands"][0]
    assert "[skip-existing]" in guarded
    assert "candidate_nonlinear_t1_n8.out.nc" in guarded
    assert "candidate_nonlinear_t1_n8.restart.nc" in guarded
    assert "candidate_nonlinear_t1_n8.big.nc" in guarded
    restart_guarded = payload["restart_seed_skip_existing_commands"][0]
    assert "candidate_nonlinear_t1p5_n8.out.nc" in restart_guarded
    assert "candidate_nonlinear_t1p5_n8.restart.nc" in restart_guarded
    assert "candidate_nonlinear_t1p5_n8.big.nc" in restart_guarded
    assert "skip_existing_note" in payload
    staged_script = tmp_path / "runs" / payload["staged_ladder_skip_existing_script"]
    direct_script = (
        tmp_path / "runs" / payload["direct_full_horizon_skip_existing_script"]
    )
    assert staged_script.exists()
    assert direct_script.exists()
    assert staged_script.stat().st_mode & 0o111
    staged_text = staged_script.read_text(encoding="utf-8")
    assert "external-VMEC staged restart ladder" in staged_text
    assert "cp " in staged_text
    assert "candidate_nonlinear_t1_n8.$ext" in staged_text
    assert "candidate_nonlinear_t1p5_n8.$ext" in staged_text
    assert "[skip-existing]" in staged_text
    assert "candidate_nonlinear_t1p5_n8 bundle already exists" in staged_text
    assert "--steps 2 --no-progress" in staged_text
    direct_text = direct_script.read_text(encoding="utf-8")
    assert "external-VMEC direct full-horizon launches" in direct_text
    assert "cp " not in direct_text
    assert "--steps 6 --no-progress" in direct_text


def test_external_vmec_replicate_variant_config_contract(tmp_path: Path) -> None:
    vmec_file = tmp_path / "wout_fixture.nc"
    vmec_file.write_text("placeholder", encoding="utf-8")

    written = write_configs(
        case="replicate",
        vmec_file=vmec_file,
        out_dir=tmp_path / "runs",
        grids=[_parse_grid("n8:8:8:6:6")],
        horizons=(1.0, 2.0),
        dt=0.25,
        ky=0.3,
        nl=2,
        nm=3,
        baseline_seed=22,
        seed_variants=[31, 32],
        dt_variants=[0.2, 0.125],
        seed_dt_variants=[(31, 0.2)],
    )

    assert len(written) == 10
    labels = [
        item.variant.label if item.variant is not None else "" for item in written
    ]
    assert labels == [
        "seed31",
        "seed31",
        "seed32",
        "seed32",
        "dt0p2",
        "dt0p2",
        "dt0p125",
        "dt0p125",
        "seed31_dt0p2",
        "seed31_dt0p2",
    ]
    assert [item.steps for item in written] == [4, 4, 4, 4, 5, 5, 8, 8, 5, 5]

    seed_config = written[0].path.read_text(encoding="utf-8")
    assert 'path = "replicate_nonlinear_t1_n8_seed31.out.nc"' in seed_config
    assert "random_seed = 31" in seed_config
    assert 'variant_axis = "seed"' in seed_config
    assert 'variant_label = "seed31"' in seed_config
    assert "timestep = 0.25" in seed_config

    dt_config = written[-1].path.read_text(encoding="utf-8")
    assert 'path = "replicate_nonlinear_t2_n8_seed31_dt0p2.out.nc"' in dt_config
    assert "dt = 0.2" in dt_config
    assert "random_seed = 31" in dt_config
    assert 'variant_axis = "seed_timestep"' in dt_config
    assert 'variant_label = "seed31_dt0p2"' in dt_config

    dt_only_config = written[-3].path.read_text(encoding="utf-8")
    assert 'path = "replicate_nonlinear_t2_n8_dt0p125.out.nc"' in dt_only_config
    assert "dt = 0.125" in dt_only_config
    assert "nsave = 8" in dt_only_config
    assert "random_seed = 22" in dt_only_config
    assert 'variant_axis = "timestep"' in dt_only_config
    assert "timestep = 0.125" in dt_only_config

    manifest = write_manifest(tmp_path / "runs", written)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert all(
        "run-runtime-nonlinear" in command for command in payload["launch_commands"]
    )
    assert all("--steps" in command for command in payload["launch_commands"])
    assert all(
        "run-runtime-nonlinear" in command
        for command in payload["direct_full_horizon_launch_commands"]
    )
    assert "--steps 8" in payload["direct_full_horizon_launch_commands"][1]
    assert any(
        "--steps 16" in command
        for command in payload["direct_full_horizon_launch_commands"]
    )
    assert (
        payload["direct_full_horizon_step_counts"]["replicate_nonlinear_t2_n8_dt0p125"]
        == 16
    )
    assert payload["segment_step_counts"]["replicate_nonlinear_t2_n8_dt0p125"] == 8
    assert len(payload["restart_seed_commands"]) == 5
    assert "replicate_nonlinear_t1_n8_seed31" in payload["restart_seed_commands"][0]
    assert "replicate_nonlinear_t2_n8_seed31" in payload["restart_seed_commands"][0]
    assert (
        "replicate_nonlinear_t1_n8_seed31_dt0p2" in payload["restart_seed_commands"][-1]
    )
    assert (
        "replicate_nonlinear_t2_n8_seed31_dt0p2" in payload["restart_seed_commands"][-1]
    )
    variants = [item["variant"] for item in payload["configs"]]
    assert variants[0] == {
        "axis": "seed",
        "label": "seed31",
        "seed": 31,
        "timestep": 0.25,
    }
    assert variants[-1] == {
        "axis": "seed_timestep",
        "label": "seed31_dt0p2",
        "seed": 31,
        "timestep": 0.2,
    }


def test_seed_dt_variant_parser_rejects_bad_inputs() -> None:
    assert _parse_seed_dt_variant("31:0.04") == (31, 0.04)
    with pytest.raises(ValueError, match="SEED:DT"):
        _parse_seed_dt_variant("31")
    with pytest.raises(ValueError, match="non-negative"):
        _parse_seed_dt_variant("-1:0.04")
    with pytest.raises(ValueError, match="positive"):
        _parse_seed_dt_variant("31:0")


def test_external_vmec_timestep_variant_metadata_contract(
    tmp_path: Path,
) -> None:
    vmec_file = tmp_path / "wout_fixture.nc"
    vmec_file.write_text("placeholder", encoding="utf-8")

    written = write_configs(
        case="replicate",
        vmec_file=vmec_file,
        out_dir=tmp_path / "runs",
        grids=[_parse_grid("n8:8:8:6:6")],
        horizons=(1.0, 2.0),
        dt=0.25,
        ky=0.3,
        nl=2,
        nm=3,
        baseline_seed=22,
        dt_variants=[0.125],
    )

    dt_config = written[-1].path.read_text(encoding="utf-8")
    assert 'path = "replicate_nonlinear_t2_n8_dt0p125.out.nc"' in dt_config
    assert "dt = 0.125" in dt_config
    assert "random_seed = 22" in dt_config
    assert 'variant_axis = "timestep"' in dt_config
    assert "timestep = 0.125" in dt_config

    manifest = write_manifest(tmp_path / "runs", written)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert len(payload["restart_seed_commands"]) == 1
    assert "replicate_nonlinear_t1_n8_dt0p125" in payload["restart_seed_commands"][-1]
    assert "replicate_nonlinear_t2_n8_dt0p125" in payload["restart_seed_commands"][-1]
    variants = [item["variant"] for item in payload["configs"]]
    assert variants[-1] == {
        "axis": "timestep",
        "label": "dt0p125",
        "seed": 22,
        "timestep": 0.125,
    }


def test_external_vmec_repair_protocol_knobs_are_written(
    tmp_path: Path,
) -> None:
    vmec_file = tmp_path / "wout_solovev_reference.nc"
    vmec_file.write_text("placeholder", encoding="utf-8")

    written = write_configs(
        case="solovev_repair",
        vmec_file=vmec_file,
        out_dir=tmp_path / "runs",
        grids=[_parse_grid("n48:48:48:32:32")],
        horizons=(50.0,),
        dt=0.02,
        ky=0.2857,
        nl=4,
        nm=8,
        torflux=0.64,
        alpha=1.2,
        npol=2.0,
        tprim=2.5,
        fprim=0.75,
        nu=0.02,
        init_amp=1.0e-5,
        y0=18.0,
        lx=54.0,
        ly=55.0,
        sample_stride=25,
        diagnostics_stride=25,
    )

    config = written[0].path.read_text(encoding="utf-8")
    assert "dt = 0.02" in config
    assert "ky = 0.2857" in config
    assert "init_amp = 1e-05" in config
    assert "alpha = 1.2" in config
    assert "npol = 2" in config
    assert "tprim = 2.5" in config
    assert "fprim = 0.75" in config
    assert "nu = 0.02" in config
    assert "y0 = 18" in config
    assert "Lx = 54" in config
    assert "Ly = 55" in config
    assert "sample_stride = 25" in config
    assert "diagnostics_stride = 25" in config
    assert "steps = 2500" in config


def test_external_vmec_holdout_config_parsers_reject_bad_inputs(tmp_path: Path) -> None:
    assert _parse_horizons("1,2,3") == (1.0, 2.0, 3.0)
    with pytest.raises(ValueError, match="sorted"):
        _parse_horizons("2,1")
    with pytest.raises(ValueError, match="label:Nx:Ny:Nz:ntheta"):
        _parse_grid("bad")
    with pytest.raises(ValueError, match="positive"):
        _parse_grid("n0:0:8:8:8")
    with pytest.raises(ValueError, match="dt variants must be positive"):
        write_configs(
            case="bad",
            vmec_file=Path("wout.nc"),
            out_dir=tmp_path / "unused",
            grids=[_parse_grid("n8:8:8:6:6")],
            dt_variants=[0.0],
        )



def test_optimized_equilibrium_transport_launch_contract(tmp_path: Path) -> None:
    vmec = tmp_path / "wout_optimized.nc"
    vmec.write_text("placeholder", encoding="utf-8")
    out_dir = tmp_path / "optimized_runs"

    assert (
        optimized_transport_main(
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
                "--torflux",
                "0.78",
                "--alpha",
                "0.7",
                "--npol",
                "1.5",
                "--tprim",
                "4.0",
                "--fprim",
                "0.5",
                "--nu",
                "0.02",
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
    assert (
        manifest["claim_level"]
        == "optimized_equilibrium_transport_launch_plan_not_simulation_claim"
    )
    assert len(manifest["configs"]) == 6
    assert len(manifest["launch_commands"]) == 6
    assert len(manifest["direct_full_horizon_launch_commands"]) == 6
    assert len(manifest["restart_seed_commands"]) == 3
    assert manifest["transport_sample"] == {
        "alpha": 0.7,
        "claim_level": "launch_contract_surface_field_line_metadata_not_transport_promotion",
        "fprim": 0.5,
        "ky": 0.47619047619047616,
        "npol": 1.5,
        "nu": 0.02,
        "torflux": 0.78,
        "tprim": 4.0,
        "vmec_file": str(vmec),
    }
    first_config = (
        out_dir / "optimized_equilibrium_test_nonlinear_t1_n8_seed31.toml"
    ).read_text(encoding="utf-8")
    assert "torflux = 0.78" in first_config
    assert "alpha = 0.7" in first_config
    assert "npol = 1.5" in first_config
    assert "tprim = 4" in first_config
    assert "fprim = 0.5" in first_config
    assert "nu = 0.02" in first_config

    contract = manifest["promotion_contract"]
    assert (
        contract["claim_level"]
        == "optimized_equilibrium_replicated_transport_window_launch_contract_not_promotion"
    )
    assert len(contract["expected_outputs"]) == 3
    assert all(
        "optimized_equilibrium_test_nonlinear_t2_n8" in path
        for path in contract["expected_outputs"]
    )
    assert contract["build_ensemble_command"].startswith(
        "python3 tools/artifacts/build_external_vmec_replicate_ensemble.py"
    )
    assert "--tmin 1" in contract["build_ensemble_command"]
    assert "--tmax 2" in contract["build_ensemble_command"]
    assert contract["direct_full_horizon_step_counts"] == {
        "dt0p25": 8,
        "seed31": 4,
        "seed32": 4,
    }
    assert any(
        "--steps 8" in command
        for command in contract["direct_full_horizon_launch_commands"]
    )
    assert contract["output_gate_command"].startswith(
        "python3 tools/release/check_nonlinear_runtime_outputs.py"
    )
    assert "--tmin 1 --tmax 2" in contract["output_gate_command"]
    assert "--min-window-samples 80" in contract["output_gate_command"]
    assert "restart-ladder segments" in contract["restart_ladder_note"]
    assert contract["run_guard_command"].startswith(
        "python3 tools/release/check_production_nonlinear_optimization_guard.py"
    )
    assert "--optimized-equilibrium-ensemble" in contract["run_guard_command"]


def test_strict_qa_t1500_contract_exposes_true_full_horizon_commands(
    tmp_path: Path,
) -> None:
    """Guard against launching a final restart segment from t=0 as a t1500 audit."""

    vmec = tmp_path / "wout_strict_qa.nc"
    vmec.write_text("placeholder", encoding="utf-8")
    out_dir = tmp_path / "strict_qa_runs"

    assert (
        optimized_transport_main(
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
    t1500_segments = [
        command
        for command in manifest["launch_commands"]
        if "_nonlinear_t1500_" in command
    ]
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



def _load_tool_module():
    return load_campaign_tool("write_nonlinear_replicate_followup_campaign")


def _write_config(path: Path, *, label: str, axis: str, seed: int, dt: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""
[[species]]
name = "ion"
tprim = 3
fprim = 1
nu = 0.01

[grid]
Nx = 8
Ny = 8
Nz = 6
Lx = 62.8
Ly = 62.8
y0 = 21
ntheta = 6

[time]
t_max = 10
dt = {dt}
sample_stride = 5
diagnostics_stride = 5
progress_bar = false

[geometry]
model = "vmec"
vmec_file = "../wout_plus.nc"
torflux = 0.64
alpha = 0
npol = 1

[init]
random_seed = {seed}
init_amp = 0.001

[run]
ky = 0.3
Nl = 2
Nm = 3

[output]
path = "case_plus_nonlinear_t10_n8_{label}.out.nc"

[metadata]
case = "case_plus"
variant_axis = "{axis}"
variant_label = "{label}"
seed = {seed}
timestep = {dt}
""".strip()
        + "\n",
        encoding="utf-8",
    )


def test_followup_campaign_writes_cross_variant_configs(tmp_path: Path) -> None:
    mod = _load_tool_module()
    (tmp_path / "wout_plus.nc").write_text("placeholder", encoding="utf-8")
    plus_dir = tmp_path / "campaign" / "plus_delta"
    configs = {
        "seed31": plus_dir / "case_plus_nonlinear_t10_n8_seed31.toml",
        "seed32": plus_dir / "case_plus_nonlinear_t10_n8_seed32.toml",
        "dt0p04": plus_dir / "case_plus_nonlinear_t10_n8_dt0p04.toml",
    }
    _write_config(configs["seed31"], label="seed31", axis="seed", seed=31, dt=0.05)
    _write_config(configs["seed32"], label="seed32", axis="seed", seed=32, dt=0.05)
    _write_config(configs["dt0p04"], label="dt0p04", axis="timestep", seed=22, dt=0.04)

    manifest = {
        "kind": "nonlinear_turbulence_gradient_campaign_manifest",
        "case": "case",
        "parameter_name": "profile_direction",
        "delta_parameter": 0.02,
        "run_contract": {"analysis_window": [5.0, 10.0], "grid": "n8"},
        "state_ensemble_commands": {
            "baseline": {
                "ensemble_json": "docs/_static/case_baseline_replicates/case_baseline_t10_ensemble_gate.json"
            },
            "minus_delta": {
                "ensemble_json": "docs/_static/case_minus_delta_replicates/case_minus_delta_t10_ensemble_gate.json"
            },
            "plus_delta": {
                "ensemble_json": "docs/_static/case_plus_delta_replicates/case_plus_delta_t10_ensemble_gate.json",
                "expected_outputs": [
                    "old_seed31.out.nc",
                    "old_seed32.out.nc",
                    "old_dt0p04.out.nc",
                ],
                "direct_full_horizon_launch_commands": [
                    f"python3 -m spectraxgk.cli run-runtime-nonlinear --config {path} --steps 1 --no-progress"
                    for path in configs.values()
                ],
            },
        },
    }
    spread = {
        "summary": {"failed_states": ["plus_delta"]},
        "state_rows": [
            {
                "state": "plus_delta",
                "classification": "mixed_seed_timestep_spread",
                "high_variant_label": "seed32",
                "low_variant_label": "dt0p04",
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    spread_path = tmp_path / "spread.json"
    out_json = tmp_path / "followup.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    spread_path.write_text(json.dumps(spread), encoding="utf-8")

    payload = mod.build_followup_campaign(
        campaign_manifest_path=manifest_path,
        spread_diagnostic_path=spread_path,
        out_json=out_json,
        case="case_followup",
        include_extra_nominal_seed=True,
        max_runs_per_state=3,
    )

    assert payload["summary"]["planned_run_count"] == 3
    assert out_json.exists()
    written = payload["written_configs_by_state"]["plus_delta"]["configs"]
    assert [row["variant_label"] for row in written] == [
        "seed22_dt0p05",
        "seed32_dt0p04",
        "seed33_dt0p05",
    ]
    assert all(Path(row["path"]).exists() for row in written)
    assert (
        "run-runtime-nonlinear"
        in payload["written_configs_by_state"]["plus_delta"][
            "direct_full_horizon_launch_commands"
        ][0]
    )
    postprocess = payload["postprocess_commands_by_state"]["plus_delta"]
    assert len(postprocess["all_expected_outputs"]) == 6
    assert "check_nonlinear_runtime_outputs.py" in postprocess["output_gate_command"]
    assert (
        "build_external_vmec_replicate_ensemble.py"
        in postprocess["build_ensemble_command"]
    )
    assert (
        "summarize_nonlinear_replicate_spread.py"
        in postprocess["replicate_spread_command"]
    )
    assert (
        "build_nonlinear_turbulence_gradient_fd_gate.py"
        in postprocess["central_fd_command"]
    )

