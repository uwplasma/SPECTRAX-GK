from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "run_vmec_jax_guarded_transport_ladder.py"
spec = importlib.util.spec_from_file_location("run_vmec_jax_guarded_transport_ladder", SCRIPT)
assert spec is not None
assert spec.loader is not None
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def _write_candidate(root: Path, *, passed: bool, objective: float, qs: float = 0.02) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "history.json").write_text(
        json.dumps(
            {
                "objective_final": objective,
                "aspect_final": 6.0,
                "iota_final": 0.42,
                "qs_final": qs,
            }
        ),
        encoding="utf-8",
    )
    (root / "solved_wout_gate.json").write_text(
        json.dumps(
            {
                "passed": passed,
                "checks": {
                    "aspect": {"passed": True},
                    "mean_iota": {"passed": True},
                    "quasisymmetry": {"passed": passed},
                    "iota_profile": {"passed": passed},
                },
                "next_action": "candidate may proceed" if passed else "do not promote",
            }
        ),
        encoding="utf-8",
    )


def test_select_promoted_candidate_uses_largest_passing_transport_weight(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    low = tmp_path / "low"
    high = tmp_path / "high"
    failed = tmp_path / "failed"
    _write_candidate(baseline, passed=True, objective=0.04)
    _write_candidate(low, passed=True, objective=0.5)
    _write_candidate(high, passed=True, objective=0.8)
    _write_candidate(failed, passed=False, objective=0.1, qs=0.2)

    summaries = [
        mod.candidate_summary(baseline, label="baseline", baseline=True),
        mod.candidate_summary(low, label="low", weight=0.001),
        mod.candidate_summary(high, label="high", weight=0.005),
        mod.candidate_summary(failed, label="failed", weight=0.01),
    ]

    selected = mod.select_promoted_candidate(summaries)

    assert selected is not None
    assert selected["label"] == "high"
    assert selected["transport_weight"] == 0.005


def test_guarded_ladder_dry_run_writes_commands(tmp_path: Path) -> None:
    constraints = tmp_path / "constraints"
    _write_candidate(constraints, passed=True, objective=0.03)
    (constraints / "input.final").write_text("! vmec restart\n", encoding="utf-8")
    out_json = tmp_path / "ladder.json"

    rc = mod.main(
        [
            "--constraints-dir",
            str(constraints),
            "--outdir",
            str(tmp_path / "ladder"),
            "--weights",
            "0.001,0.005",
            "--driver-args",
            "--max-mode 5 --mboz 21 --nboz 21",
            "--dry-run",
            "--out-json",
            str(out_json),
        ]
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["dry_run"] is True
    assert payload["passed"] is True
    assert len(payload["commands"]) == 2
    assert "--allow-failed-solved-wout-gate" in payload["commands"][0]["command"]
    assert "--disable-mode-continuation" in payload["commands"][0]["command"]
    assert payload["promoted_candidate"]["baseline"] is True


def test_candidate_summary_keeps_reconstructed_gate_advisory_by_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
    constraints = tmp_path / "constraints"
    constraints.mkdir(parents=True)
    (constraints / "history.json").write_text(
        json.dumps(
            {
                "objective_final": 0.03,
                "aspect_final": 6.001,
                "iota_final": 0.427,
                "qs_final": 0.02,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(mod, "_load_wout_iota_profiles", lambda _root: ([0.0, 0.411, 0.414], [0.412, 0.414]))

    summary = mod.candidate_summary(
        constraints,
        label="legacy constraints",
        baseline=True,
        target_aspect=6.0,
        aspect_atol=0.05,
        min_abs_mean_iota=0.41,
        qs_residual_max=0.05,
        iota_profile_floor=0.41,
    )

    assert summary["passed"] is False
    assert summary["gate_reported_passed"] is True
    assert summary["gate_is_authoritative"] is False
    assert summary["gate_path"] is None
    assert summary["gate_source"] == "reconstructed"
    assert summary["gate_checks"]["iota_profile"] is True
    assert "advisory only" in summary["next_action"]


def test_candidate_summary_can_allow_reconstructed_gate_for_exploration(
    tmp_path: Path,
    monkeypatch,
) -> None:
    constraints = tmp_path / "constraints"
    constraints.mkdir(parents=True)
    (constraints / "history.json").write_text(
        json.dumps(
            {
                "objective_final": 0.03,
                "aspect_final": 6.001,
                "iota_final": 0.427,
                "qs_final": 0.02,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(mod, "_load_wout_iota_profiles", lambda _root: ([0.0, 0.411, 0.414], [0.412, 0.414]))

    summary = mod.candidate_summary(
        constraints,
        label="legacy constraints",
        baseline=True,
        target_aspect=6.0,
        aspect_atol=0.05,
        min_abs_mean_iota=0.41,
        qs_residual_max=0.05,
        iota_profile_floor=0.41,
        allow_reconstructed_gate=True,
    )

    assert summary["passed"] is True
    assert summary["gate_reported_passed"] is True
    assert summary["gate_is_authoritative"] is True
    assert summary["gate_path"] is None
    assert summary["gate_source"] == "reconstructed"
