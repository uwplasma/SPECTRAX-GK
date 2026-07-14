from __future__ import annotations

import json
from pathlib import Path
import shlex
import subprocess
import sys

from support.paths import REPO_ROOT, load_campaign_tool


ROOT = REPO_ROOT
OPTIMIZER_SCRIPT = (
    ROOT / "tools" / "campaigns" / "write_vmec_jax_optimizer_comparison_manifest.py"
)
SPSA_SCRIPT = ROOT / "tools" / "campaigns" / "write_vmec_jax_spsa_candidate_campaign.py"
optimizer_mod = load_campaign_tool("write_vmec_jax_optimizer_comparison_manifest")
spsa_mod = load_campaign_tool("write_vmec_jax_spsa_candidate_campaign")


def _entry(payload: dict[str, object], case_id: str) -> dict[str, object]:
    entries = payload["entries"]
    assert isinstance(entries, list)
    for entry in entries:
        assert isinstance(entry, dict)
        if entry.get("case_id") == case_id:
            return entry
    raise AssertionError(f"missing entry {case_id}")


def test_optimizer_manifest_emits_current_matched_derivative_policies(
    tmp_path: Path,
) -> None:
    args = optimizer_mod.parse_args(
        [
            "--campaign-root",
            str(tmp_path / "campaign"),
            "--out-json",
            str(tmp_path / "manifest.json"),
            "--transport-kinds",
            "growth,quasilinear_flux",
            "--outer-loop-methods",
            "spsa,bo",
            "--audit-seed-variant",
            "41",
            "--audit-seed-variant",
            "42",
        ]
    )

    payload = optimizer_mod.build_manifest(args)

    assert payload["schema_version"] == 2
    assert payload["comparison_policy"]["api"] == "current_vmec_jax_opt_least_squares"
    assert payload["comparison_policy"]["sample_policy"] == {
        "surfaces": [0.45, 0.64, 0.78],
        "alphas": [0.0, 0.7853981633974483],
        "ky_values": [0.1, 0.3, 0.5],
        "ntheta": 24,
        "mboz": 21,
        "nboz": 21,
    }
    assert (
        payload["comparison_policy"]["landscape_policy"][
            "rbc11_points_admit_optimized_candidates"
        ]
        is False
    )
    assert len(payload["runnable_commands"]) == 3

    baseline = _entry(payload, "qa_baseline")
    baseline_parts = shlex.split(str(baseline["command"]))
    assert "--constraints-only" in baseline_parts
    assert baseline_parts[baseline_parts.index("--mode-schedule") + 1] == "1,2,3,4,5"
    assert "--strict-upstream-qa-baseline" not in baseline_parts
    assert baseline["expected_wout"].endswith("runs/qa_baseline/wout_final.nc")

    growth = _entry(payload, "growth_implicit_from_qa_baseline")
    quasilinear = _entry(payload, "quasilinear_finite_difference_from_qa_baseline")
    for entry, expected_kind, expected_jacobian in (
        (growth, "growth", "implicit"),
        (quasilinear, "quasilinear_flux", "finite-difference"),
    ):
        parts = shlex.split(str(entry["command"]))
        assert parts[:2] == [
            "python3",
            "tools/campaigns/vmec_jax_qa_low_turbulence_optimization.py",
        ]
        assert parts[parts.index("--input") + 1].endswith(
            "runs/qa_baseline/input.final"
        )
        assert parts[parts.index("--transport-kind") + 1] == expected_kind
        assert parts[parts.index("--jacobian") + 1] == expected_jacobian
        assert parts[parts.index("--surfaces") + 1] == "0.45,0.64,0.78"
        assert parts[parts.index("--alphas") + 1] == "0,0.7853981633974483"
        assert parts[parts.index("--ky-values") + 1] == "0.1,0.3,0.5"
        assert "--method" not in parts
        assert "--mboz" not in parts
        audit = str(entry["recommended_nonlinear_audit_command"])
        assert "--horizons 700,1100,1500" in audit
        assert "--window-tmin 1100 --window-tmax 1500" in audit
        assert "--seed-variant 41 --seed-variant 42" in audit


def test_optimizer_manifest_keeps_noisy_methods_as_outer_loop_contracts(
    tmp_path: Path,
) -> None:
    args = optimizer_mod.parse_args(
        [
            "--campaign-root",
            str(tmp_path / "campaign"),
            "--transport-kinds",
            "nonlinear_window_heat_flux",
            "--outer-loop-methods",
            "spsa,cma_es,bo",
        ]
    )
    payload = optimizer_mod.build_manifest(args)

    deterministic = _entry(
        payload, "nonlinear_window_finite_difference_from_qa_baseline"
    )
    assert deterministic["derivative_policy"] == "finite-difference outer Jacobian"
    for method in ("spsa", "cma_es", "bo"):
        entry = _entry(payload, f"nonlinear_window_heat_flux_{method}_outer_loop")
        strategy = entry["optimizer_strategy"]
        contract = entry["candidate_contract"]
        assert entry["status"] == "planned_outer_loop"
        assert strategy["requires_common_random_numbers"] is True
        assert strategy["requires_matched_t1500_replicated_audit"] is True
        assert "candidate generation only" in strategy["claim_scope"]
        assert "{candidate_input_final}" in contract["metric_eval_command_template"]
        assert "--mboz 21 --nboz 21" in contract["metric_eval_command_template"]
        assert "{candidate_id}" in contract["nonlinear_audit_command_template"]


def test_optimizer_comparison_manifest_cli_writes_json(tmp_path: Path) -> None:
    out_json = tmp_path / "manifest.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(OPTIMIZER_SCRIPT),
            "--campaign-root",
            str(tmp_path / "campaign"),
            "--out-json",
            str(out_json),
            "--transport-kinds",
            "growth",
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
    payload = json.loads(out_json.read_text())
    assert status["comparison_fingerprint"] == payload["comparison_fingerprint"]
    assert _entry(payload, "growth_implicit_from_qa_baseline")["status"] == "runnable"
    assert _entry(payload, "growth_bo_outer_loop")["status"] == "planned_outer_loop"


def _write_input(path: Path) -> None:
    path.write_text(
        """
&INDATA
  RBC(0,0) = 1.0000000000000000E+00
  ZBS(1,0) = -2.0000000000000000E-02
  ZBS(1,1) = 5.0000000000000000E-02
  RBC(1,1) = 1.0000000000000000E-01
/
""".lstrip(),
        encoding="utf-8",
    )


def test_spsa_candidate_campaign_writes_plus_minus_common_random_number_commands(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "input.final"
    out_dir = tmp_path / "spsa"
    _write_input(baseline)

    args = spsa_mod.parse_args(
        [
            "--baseline-input",
            str(baseline),
            "--out-dir",
            str(out_dir),
            "--controls",
            "ZBS(1,0);ZBS(1,1);RBC(1,1)",
            "--iterations",
            "2",
            "--seed",
            "123",
            "--relative-delta",
            "0.1",
            "--audit-seed-variant",
            "41",
            "--audit-seed-variant",
            "42",
        ]
    )

    payload = spsa_mod.build_campaign(args)

    assert payload["kind"] == "vmec_jax_spsa_transport_candidate_campaign"
    assert payload["controls"] == ["ZBS(1,0)", "ZBS(1,1)", "RBC(1,1)"]
    assert payload["common_random_number_policy"]["audit_seed_variants"] == [41, 42]
    assert len(payload["pairs"]) == 2
    first = payload["pairs"][0]
    assert set(first["states"]) == {"plus", "minus"}
    assert first["states"]["plus"]["input"].endswith("iter_000/plus/input.final")
    plus_text = (out_dir / "iter_000" / "plus" / "input.final").read_text(
        encoding="utf-8"
    )
    minus_text = (out_dir / "iter_000" / "minus" / "input.final").read_text(
        encoding="utf-8"
    )
    assert plus_text != minus_text
    assert "RBC(1,1)" in plus_text
    metric_command = first["states"]["plus"]["metric_eval_command"]
    metric_parts = shlex.split(metric_command)
    assert metric_parts[:2] == [
        "python3",
        "tools/campaigns/evaluate_vmec_jax_spectrax_transport_metric.py",
    ]
    assert (
        metric_parts[metric_parts.index("--transport-kind") + 1]
        == "nonlinear_window_heat_flux"
    )
    assert metric_parts[metric_parts.index("--mboz") + 1] == "21"
    audit_command = first["states"]["plus"]["nonlinear_audit_command"]
    assert "--seed-variant 41 --seed-variant 42" in audit_command
    assert "--window-tmin 1100 --window-tmax 1500" in audit_command
    assert "dJ/dx_i" in first["gradient_estimator"]["RBC(1,1)"]
    manifest = json.loads(
        (out_dir / "vmec_jax_spsa_candidate_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["claim_scope"].startswith(
        "SPSA common-random-number candidate generation"
    )


def test_spsa_candidate_campaign_cli_writes_manifest(tmp_path: Path) -> None:
    baseline = tmp_path / "input.final"
    out_dir = tmp_path / "spsa"
    _write_input(baseline)

    completed = subprocess.run(
        [
            sys.executable,
            str(SPSA_SCRIPT),
            "--baseline-input",
            str(baseline),
            "--out-dir",
            str(out_dir),
            "--controls",
            "ZBS(1,0);RBC(1,1)",
            "--iterations",
            "1",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

    status = json.loads(completed.stdout)
    assert status["pairs"] == 1
    assert status["controls"] == ["ZBS(1,0)", "RBC(1,1)"]
    assert (out_dir / "vmec_jax_spsa_candidate_manifest.json").exists()
