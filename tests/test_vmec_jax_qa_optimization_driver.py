from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import spectraxgk
from spectraxgk.vmec_jax_candidate_gate import build_solved_vmec_candidate_gate


ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "tools" / "vmec_jax_qa_low_turbulence_optimization.py"


def _load_driver():
    spec = importlib.util.spec_from_file_location("vmec_jax_qa_low_turbulence_optimization", DRIVER)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_solved_wout_candidate_gate_passes_valid_qa_branch() -> None:
    assert spectraxgk.build_solved_vmec_candidate_gate is build_solved_vmec_candidate_gate
    result = SimpleNamespace(
        history={"aspect_final": 5.999233, "iota_final": 0.427011, "qs_final": 2.604013e-2},
    )

    report = build_solved_vmec_candidate_gate(
        result,
        target_aspect=6.0,
        aspect_atol=5.0e-2,
        min_abs_mean_iota=0.41,
        qs_residual_max=5.0e-2,
        iota_profile_floor=0.41,
        iota_profiles=(
            np.asarray([0.0, 0.410131, 0.414]),
            np.asarray([0.410706, 0.414]),
        ),
    )

    assert report["passed"] is True
    assert report["checks"]["aspect"]["passed"] is True
    assert report["checks"]["mean_iota"]["passed"] is True
    assert report["checks"]["quasisymmetry"]["passed"] is True
    assert report["checks"]["iota_profile"]["passed"] is True
    json.dumps(report, allow_nan=False)


def test_solved_wout_candidate_gate_rejects_transport_branch_that_breaks_constraints() -> None:
    result = SimpleNamespace(
        history={"aspect_final": 5.996817, "iota_final": 0.425028, "qs_final": 1.091236e-1},
    )

    report = build_solved_vmec_candidate_gate(
        result,
        target_aspect=6.0,
        aspect_atol=5.0e-2,
        min_abs_mean_iota=0.41,
        qs_residual_max=5.0e-2,
        iota_profile_floor=0.41,
        iota_profiles=(
            np.asarray([0.0, 0.402043, 0.414]),
            np.asarray([0.402493, 0.414]),
        ),
    )

    assert report["passed"] is False
    assert report["checks"]["aspect"]["passed"] is True
    assert report["checks"]["mean_iota"]["passed"] is True
    assert report["checks"]["quasisymmetry"]["passed"] is False
    assert report["checks"]["iota_profile"]["passed"] is False
    assert "do not promote" in report["next_action"]
    json.dumps(report, allow_nan=False)


def test_driver_transport_metric_from_result_uses_final_state_context() -> None:
    mod = _load_driver()

    class FakeTransport:
        config = SimpleNamespace(kind="growth", objective_transform="log1p", objective_scale=3.0)

        def J(self, ctx, state):
            assert state == "final-state"
            assert ctx.indata == "indata"
            assert ctx.signgs == -1
            assert ctx.flux == "flux"
            return np.asarray(0.125)

    result = SimpleNamespace(
        final_state="final-state",
        final_optimizer=SimpleNamespace(
            _static=SimpleNamespace(s=np.asarray([0.0, 1.0])),
            _indata="indata",
            _signgs=-1,
            _flux="flux",
        ),
    )

    metric = mod._transport_metric_from_result(FakeTransport(), result)

    assert metric["transport_objective_final"] == 0.125
    assert metric["spectrax_objective_final"] == 0.125
    assert metric["transport_metric_final"] == 0.125
    assert metric["transport_objective_source"] == "final_vmec_jax_state"
    assert metric["transport_metric_kind"] == "growth"
    json.dumps(metric, allow_nan=False)


def test_driver_defaults_to_multisample_transport_admission_set(monkeypatch) -> None:
    mod = _load_driver()
    monkeypatch.setattr(sys, "argv", [str(DRIVER), "--dry-run"])

    args = mod._parse_args()
    summary = spectraxgk.transport_objective_sample_summary(
        {
            "surfaces": args.surfaces,
            "alphas": args.alphas,
            "ky_values": args.ky_values,
        }
    )

    assert args.surfaces == mod.DEFAULT_TRANSPORT_SURFACES
    assert args.alphas == mod.DEFAULT_TRANSPORT_ALPHAS
    assert args.ky_values == mod.DEFAULT_TRANSPORT_KY_VALUES
    assert summary["passed"] is True
    assert summary["sample_count"] == 18


def test_driver_dry_run_cli_writes_transport_setup_summary(tmp_path: Path) -> None:
    if importlib.util.find_spec("vmec_jax") is None:
        pytest.skip("vmec_jax is optional")

    outdir = tmp_path / "qa_growth_dry_run"
    completed = subprocess.run(
        [
            sys.executable,
            str(DRIVER),
            "--dry-run",
            "--use-simple-seed",
            "--max-mode",
            "1",
            "--min-vmec-mode",
            "3",
            "--mboz",
            "21",
            "--nboz",
            "21",
            "--transport-kind",
            "growth",
            "--surfaces",
            "0.64",
            "--alphas",
            "0.0",
            "--ky-values",
            "0.30",
            "--ntheta",
            "4",
            "--n-laguerre",
            "1",
            "--n-hermite",
            "1",
            "--surface-chunk-size",
            "1",
            "--spectrax-weight",
            "0.001",
            "--outdir",
            str(outdir),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

    summary_path = outdir / "setup_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["constraints_only"] is False
    assert summary["transport_kind"] == "growth"
    assert summary["requested_input"].endswith("input.minimal_seed_nfp2")
    assert summary["objectives"] == [
        "aspect",
        "iota",
        "iota_profile_floor",
        "qs",
        "spectraxgk_transport",
    ]
    assert summary["sample_set"]["n_samples"] == 1
    assert summary["spectrax_config"]["mboz"] == 21
    assert summary["spectrax_config"]["nboz"] == 21
    assert summary["spectrax_config"]["surface_chunk_size"] == 1
    assert summary["optimizer"]["method"] == "scalar_trust"
    assert "production nonlinear flux claims require matched long-window" in summary["claim_scope"]
    assert "spectraxgk_transport" in completed.stdout
    assert not (outdir / "history.json").exists()
    assert not (outdir / "solved_wout_gate.json").exists()


def test_driver_strict_upstream_qa_baseline_preset_is_admission_grade(tmp_path: Path) -> None:
    if importlib.util.find_spec("vmec_jax") is None:
        pytest.skip("vmec_jax is optional")

    outdir = tmp_path / "strict_qa_baseline"
    subprocess.run(
        [
            sys.executable,
            str(DRIVER),
            "--dry-run",
            "--strict-upstream-qa-baseline",
            "--outdir",
            str(outdir),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

    summary = json.loads((outdir / "setup_summary.json").read_text(encoding="utf-8"))

    assert summary["constraints_only"] is True
    assert summary["use_simple_seed"] is True
    assert summary["requested_input"].endswith("input.minimal_seed_nfp2")
    assert summary["max_mode"] == 5
    assert summary["min_vmec_mode"] == 7
    assert summary["target_aspect"] == 5.0
    assert summary["min_iota"] == pytest.approx(0.4102)
    assert summary["strict_upstream_qa_baseline"] is True
    assert summary["strict_iota_admission_buffer"] == pytest.approx(2.0e-4)
    assert summary["iota_objective"] == "target"
    assert summary["iota_profile_floor"] is None
    assert summary["objectives"] == ["aspect", "iota", "qs"]
    assert summary["optimizer"]["method"] == "scipy"
    assert summary["optimizer"]["scipy_tr_solver"] == "exact"
    assert summary["optimizer"]["max_nfev"] >= 80
    assert summary["optimizer"]["inner_max_iter"] >= 180
    assert summary["optimizer"]["trial_max_iter"] >= 180
    assert summary["optimizer"]["ftol"] <= 1.0e-5
    assert summary["optimizer"]["gtol"] <= 1.0e-5
    assert summary["optimizer"]["xtol"] <= 1.0e-8
    assert summary["optimizer"]["use_ess"] is True
    assert summary["optimizer"]["ess_alpha"] == 1.2
    assert summary["optimizer"]["strict_upstream_qa_baseline"] is True
    assert summary["optimizer"]["save_rerun_wouts"] is True
    assert summary["optimizer"]["require_rerun_wout_gate"] is True
    assert summary["optimizer"]["admit_authoritative_rerun_wout"] is False
    assert summary["optimizer"]["wout_repro_mean_iota_atol"] == pytest.approx(5.0e-4)
    assert summary["solved_wout_gate_policy"]["min_abs_mean_iota"] == 0.41


def test_driver_updates_history_with_transport_metric(tmp_path: Path) -> None:
    mod = _load_driver()
    path = tmp_path / "history.json"
    path.write_text(
        json.dumps({"objective_final": 1.0, "history": [{"objective": 1.0}]}),
        encoding="utf-8",
    )

    mod._update_history_with_transport_metric(
        path,
        {
            "transport_objective_final": 0.2,
            "spectrax_objective_final": 0.2,
            "transport_metric_final": 0.2,
            "transport_objective_error": None,
        },
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["transport_objective_final"] == 0.2
    assert payload["spectrax_objective_final"] == 0.2
    assert payload["transport_metric_final"] == 0.2
    assert "transport_objective_error" not in payload
    assert payload["history"][-1]["transport_objective_final"] == 0.2
