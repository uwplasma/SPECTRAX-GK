from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tools" / "campaigns" / "write_nonlinear_replicate_followup_campaign.py"


def _load_tool_module():
    spec = importlib.util.spec_from_file_location(
        "write_nonlinear_replicate_followup_campaign", SCRIPT
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
