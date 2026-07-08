from __future__ import annotations

import json
from pathlib import Path

from support.paths import load_campaign_tool


mod = load_campaign_tool("run_vmec_jax_guarded_transport_ladder")


def _write_candidate(
    root: Path, *, passed: bool, objective: float, qs: float = 0.02
) -> None:
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


def test_select_promoted_candidate_uses_largest_passing_transport_weight(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    low = tmp_path / "low"
    high = tmp_path / "high"
    failed = tmp_path / "failed"
    _write_candidate(baseline, passed=True, objective=0.04)
    _write_candidate(low, passed=True, objective=0.03)
    _write_candidate(high, passed=True, objective=0.02)
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


def test_select_promoted_candidate_requires_transport_improvement(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    worse = tmp_path / "worse"
    _write_candidate(baseline, passed=True, objective=0.04)
    _write_candidate(worse, passed=True, objective=0.05)

    summaries = [
        mod.candidate_summary(baseline, label="baseline", baseline=True),
        mod.candidate_summary(worse, label="worse", weight=0.001),
    ]

    selected = mod.select_promoted_candidate(summaries)

    assert selected is not None
    assert selected["label"] == "baseline"
    assert selected["baseline"] is True


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
    assert payload["transport_candidate_admitted"] is False
    assert len(payload["commands"]) == 2
    assert "--allow-failed-solved-wout-gate" in payload["commands"][0]["command"]
    assert "--disable-mode-continuation" in payload["commands"][0]["command"]
    assert payload["promoted_candidate"]["baseline"] is True


def test_guarded_ladder_can_disable_profile_floor_for_strict_mean_iota_baseline(
    tmp_path: Path,
) -> None:
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
            "0.0005",
            "--target-aspect",
            "5.0",
            "--min-iota",
            "0.41",
            "--disable-iota-profile-floor",
            "--driver-args",
            "--target-aspect 5.0 --min-iota 0.4102 --mboz 21 --nboz 21",
            "--dry-run",
            "--out-json",
            str(out_json),
        ]
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    command = payload["commands"][0]["command"]
    assert rc == 0
    assert payload["gate_policy"]["iota_profile_floor"] is None
    assert "--disable-iota-profile-floor" in command
    assert command.count("--disable-iota-profile-floor") == 1


def test_guarded_ladder_uses_explicit_baseline_transport_metric_for_constraints_only_history(
    tmp_path: Path,
) -> None:
    constraints = tmp_path / "constraints"
    _write_candidate(constraints, passed=True, objective=99.0)
    (constraints / "input.final").write_text("! vmec restart\n", encoding="utf-8")
    metric_json = tmp_path / "baseline_metric.json"
    metric_json.write_text(
        json.dumps(
            {
                "kind": "vmec_jax_spectrax_transport_metric_eval",
                "transport_metric_kind": "nonlinear_window_heat_flux",
                "transport_objective_final": 0.08,
                "spectrax_objective_final": 0.08,
                "transport_metric_final": 0.08,
                "sample_set": {"n_samples": 18},
            }
        ),
        encoding="utf-8",
    )
    out_json = tmp_path / "ladder.json"

    rc = mod.main(
        [
            "--constraints-dir",
            str(constraints),
            "--baseline-metric-json",
            str(metric_json),
            "--outdir",
            str(tmp_path / "ladder"),
            "--weights",
            "0.0005",
            "--dry-run",
            "--out-json",
            str(out_json),
        ]
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    baseline = payload["candidates"][0]
    admitted_baseline = payload["transport_admission"]["candidates"][0]
    assert rc == 0
    assert payload["baseline_metric_json"] == str(metric_json)
    assert baseline["objective_final"] == 99.0
    assert baseline["transport_metric_final"] == 0.08
    assert baseline["transport_metric_kind"] == "nonlinear_window_heat_flux"
    assert (
        admitted_baseline["transport_metric"]["source"] == "transport_objective_final"
    )
    assert admitted_baseline["transport_metric"]["uses_total_objective_proxy"] is False


def test_guarded_ladder_stops_after_first_failed_transport_gate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    constraints = tmp_path / "constraints"
    _write_candidate(constraints, passed=True, objective=0.03)
    (constraints / "input.final").write_text("! vmec restart\n", encoding="utf-8")
    out_json = tmp_path / "ladder.json"
    launched: list[list[str]] = []

    def fake_run(command, **_kwargs):
        launched.append(list(command))
        outdir = Path(command[command.index("--outdir") + 1])
        _write_candidate(outdir, passed=False, objective=0.04, qs=1.0)
        return None

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    rc = mod.main(
        [
            "--constraints-dir",
            str(constraints),
            "--outdir",
            str(tmp_path / "ladder"),
            "--weights",
            "0.001,0.005",
            "--driver-args",
            "--max-mode 5",
            "--out-json",
            str(out_json),
        ]
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert rc == 0
    assert len(launched) == 1
    assert len(payload["commands"]) == 1
    assert payload["stopped_after_failed_gate"] is True
    assert payload["transport_candidate_admitted"] is False
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

    monkeypatch.setattr(
        mod,
        "_load_wout_iota_profiles",
        lambda _root: ([0.0, 0.411, 0.414], [0.412, 0.414]),
    )

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

    monkeypatch.setattr(
        mod,
        "_load_wout_iota_profiles",
        lambda _root: ([0.0, 0.411, 0.414], [0.412, 0.414]),
    )

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
