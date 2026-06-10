from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "write_vmec_jax_optimizer_comparison_manifest.py"
spec = importlib.util.spec_from_file_location("write_vmec_jax_optimizer_comparison_manifest", SCRIPT)
assert spec is not None
assert spec.loader is not None
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def _entry(payload: dict[str, object], case_id: str) -> dict[str, object]:
    entries = payload["entries"]
    assert isinstance(entries, list)
    for entry in entries:
        assert isinstance(entry, dict)
        if entry.get("case_id") == case_id:
            return entry
    raise AssertionError(f"missing entry {case_id}")


def test_optimizer_comparison_manifest_builds_matched_runnable_commands(tmp_path: Path) -> None:
    args = mod.parse_args(
        [
            "--campaign-root",
            str(tmp_path / "campaign"),
            "--out-json",
            str(tmp_path / "manifest.json"),
            "--transport-kinds",
            "growth,quasilinear_flux",
            "--runnable-methods",
            "scipy,scalar_trust",
            "--outer-loop-methods",
            "spsa,bo",
            "--audit-seed-variant",
            "41",
            "--audit-seed-variant",
            "42",
        ]
    )

    payload = mod.build_manifest(args)

    assert payload["kind"] == "vmec_jax_qa_optimizer_comparison_manifest"
    assert payload["comparison_fingerprint"]
    assert payload["comparison_policy"]["sample_policy"]["mboz"] == 21
    assert payload["comparison_policy"]["sample_policy"]["nboz"] == 21
    assert payload["comparison_policy"]["sample_policy"]["surfaces"] == [0.64]
    assert payload["comparison_policy"]["landscape_policy"]["rbc11_points_admit_optimized_candidates"] is False
    assert "simple seed" in payload["comparison_policy"]["landscape_policy"]["optimization_seed_policy"]
    ladder = payload["comparison_policy"]["optimizer_ladder_policy"]
    assert ladder[1]["stage"] == "linear_quasilinear_transport"
    assert ladder[3]["stage"] == "quasilinear_screening_refit"
    assert len(payload["runnable_commands"]) == 5

    baseline = _entry(payload, "qa_baseline_scipy")
    assert baseline["status"] == "runnable"
    baseline_parts = shlex.split(str(baseline["command"]))
    assert "--strict-upstream-qa-baseline" in baseline_parts
    assert "--admit-authoritative-rerun-wout" in baseline_parts
    assert "--allow-failed-solved-wout-gate" not in baseline_parts
    assert baseline["expected_authoritative_wout"].endswith("runs/qa_baseline_scipy/wout_final_rerun.nc")

    growth_scipy = _entry(payload, "growth_scipy_from_strict_baseline")
    growth_scalar = _entry(payload, "growth_scalar_trust_from_strict_baseline")
    for entry in (growth_scipy, growth_scalar):
        command = str(entry["command"])
        parts = shlex.split(command)
        assert parts[0:2] == ["python3", "tools/vmec_jax_qa_low_turbulence_optimization.py"]
        assert parts[parts.index("--input") + 1].endswith("runs/qa_baseline_scipy/input.final")
        assert parts[parts.index("--transport-kind") + 1] == "growth"
        assert parts[parts.index("--surfaces") + 1] == "0.64"
        assert parts[parts.index("--alphas") + 1] == "0"
        assert parts[parts.index("--ky-values") + 1] == "0.3"
        assert parts[parts.index("--mboz") + 1] == "21"
        assert parts[parts.index("--nboz") + 1] == "21"
        assert "--save-rerun-wouts" in parts
        assert "--admit-authoritative-rerun-wout" in parts
        assert "--allow-failed-solved-wout-gate" in parts
        assert "--make-plots" in parts
        audit = str(entry["recommended_nonlinear_audit_command"])
        assert audit.startswith("python3 tools/write_optimized_equilibrium_transport_configs.py")
        assert "--horizons 700,1100,1500" in audit
        assert "--window-tmin 1100 --window-tmax 1500" in audit
        assert "--seed-variant 41 --seed-variant 42" in audit
        assert "--dt-variant 0.04" in audit
        strategy = entry["optimizer_strategy"]
        assert strategy["stage"] == "linear_quasilinear_continuation_multistart"
        assert strategy["uses_transport_weight_continuation"] is True
        assert strategy["uses_multistart_from_simple_seed_qa_solves"] is True
        assert strategy["rbc_landscape_role"] == "conditioning_and_noise_diagnostic_only"

    scipy_parts = shlex.split(str(growth_scipy["command"]))
    assert scipy_parts[scipy_parts.index("--method") + 1] == "scipy"
    assert scipy_parts[scipy_parts.index("--scipy-tr-solver") + 1] == "lsmr"
    assert scipy_parts[scipy_parts.index("--scipy-lsmr-maxiter") + 1] == "200"

    scalar_parts = shlex.split(str(growth_scalar["command"]))
    assert scalar_parts[scalar_parts.index("--method") + 1] == "scalar_trust"


def test_optimizer_comparison_manifest_marks_spsa_cma_bo_as_outer_loop_contracts(
    tmp_path: Path,
) -> None:
    args = mod.parse_args(
        [
            "--campaign-root",
            str(tmp_path / "campaign"),
            "--out-json",
            str(tmp_path / "manifest.json"),
            "--transport-kinds",
            "nonlinear_window_heat_flux",
            "--runnable-methods",
            "lbfgs_adjoint",
            "--outer-loop-methods",
            "spsa,cma_es,bo",
        ]
    )

    payload = mod.build_manifest(args)

    methods = {"spsa", "cma_es", "bo"}
    for method in methods:
        entry = _entry(payload, f"nonlinear_window_heat_flux_{method}_outer_loop")
        assert entry["status"] == "planned_outer_loop"
        assert entry["candidate_generator_required"] is True
        contract = entry["candidate_contract"]
        strategy = entry["optimizer_strategy"]
        assert isinstance(contract, dict)
        assert strategy["requires_common_random_numbers"] is True
        assert strategy["requires_matched_t1500_replicated_audit"] is True
        assert strategy["rbc_landscape_role"] == "conditioning_and_noise_diagnostic_only"
        metric_command = str(contract["metric_eval_command_template"])
        audit_command = str(contract["nonlinear_audit_command_template"])
        spsa_command = contract["spsa_candidate_campaign_command"]
        if method == "spsa":
            assert isinstance(spsa_command, str)
            assert spsa_command.startswith("python3 tools/write_vmec_jax_spsa_candidate_campaign.py")
            assert "--transport-kind nonlinear_window_heat_flux" in spsa_command
            assert "--audit-seed-variant 32 --audit-seed-variant 33" in spsa_command
        else:
            assert spsa_command is None
        assert metric_command.startswith("python3 tools/evaluate_vmec_jax_spectrax_transport_metric.py")
        assert audit_command.startswith("python3 tools/write_optimized_equilibrium_transport_configs.py")
        assert "tools/evaluate_vmec_jax_spectrax_transport_metric.py" in metric_command
        assert "{candidate_input_final}" in metric_command
        assert "{candidate_id}" in metric_command
        assert "--transport-kind nonlinear_window_heat_flux" in metric_command
        assert "tools/write_optimized_equilibrium_transport_configs.py" in audit_command
        assert "{candidate_id}" in audit_command
        assert "--window-tmin 1100 --window-tmax 1500" in audit_command


def test_optimizer_comparison_manifest_cli_writes_json(tmp_path: Path) -> None:
    out_json = tmp_path / "manifest.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--campaign-root",
            str(tmp_path / "campaign"),
            "--out-json",
            str(out_json),
            "--transport-kinds",
            "growth",
            "--runnable-methods",
            "lbfgs_adjoint",
            "--outer-loop-methods",
            "bo",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

    status = json.loads(completed.stdout)
    payload = json.loads(out_json.read_text(encoding="utf-8"))

    assert status["out_json"] == str(out_json)
    assert status["comparison_fingerprint"] == payload["comparison_fingerprint"]
    assert _entry(payload, "growth_lbfgs_adjoint_from_strict_baseline")["status"] == "runnable"
    assert _entry(payload, "growth_bo_outer_loop")["status"] == "planned_outer_loop"
