from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import jax.numpy as jnp


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "benchmark_mapped_velocity_rhs.py"
spec = importlib.util.spec_from_file_location("benchmark_mapped_velocity_rhs", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def test_parse_map_specs_accepts_label_and_three_parameters() -> None:
    specs = mod._parse_map_specs(["shifted:0.1:-0.2:0.3"])

    label, cfg = specs[0]

    assert label == "shifted"
    assert float(cfg.parallel_shift) == 0.1
    assert float(cfg.parallel_log_scale) == -0.2
    assert float(cfg.perpendicular_log_scale) == 0.3


def test_rayleigh_frequency_proxy_uses_rhs_eigenvalue_convention() -> None:
    state = jnp.asarray([1.0 + 0.0j, 0.0 + 1.0j])
    rhs = (0.25 - 1.5j) * state

    gamma, omega = mod._rayleigh_frequency_proxy(state, rhs)

    assert abs(gamma - 0.25) < 1.0e-12
    assert abs(omega - 1.5) < 1.0e-12


def test_dense_dominant_frequency_returns_tiny_operator_eigenvalue() -> None:
    template = jnp.zeros((2,), dtype=jnp.complex64)
    diag = jnp.asarray([0.1 - 0.5j, 0.4 - 0.2j], dtype=jnp.complex64)

    def rhs_fn(state):
        return diag * state, None

    dense = mod._dense_dominant_frequency(rhs_fn, template, max_size=4)

    assert dense is not None
    gamma, omega = dense
    assert abs(gamma - 0.4) < 1.0e-6
    assert abs(omega - 0.2) < 1.0e-6


def test_tiny_eigen_scorecard_validates_identity_operator() -> None:
    scorecard = mod._build_tiny_eigen_scorecard(
        [
            ("identity", mod.VelocityMapConfig()),
            ("parallel_shift", mod.VelocityMapConfig(parallel_shift=0.15)),
        ],
        max_size=128,
        identity_tolerance=1.0e-10,
    )

    assert scorecard is not None
    assert scorecard["kind"] == "mapped_velocity_rhs_tiny_dense_eigen_scorecard"
    assert scorecard["matrix_size"] == 24
    assert scorecard["readiness"]["identity_dense_operator_matches_unmapped"] is True
    assert scorecard["readiness"]["all_dense_metrics_finite"] is True
    assert scorecard["max_identity_matrix_rel_error"] == 0.0
    labels = {row["map_label"] for row in scorecard["rows"]}
    assert {"unmapped", "identity", "parallel_shift"} <= labels


def test_build_summary_reports_identity_and_dense_eigen_readiness() -> None:
    rows = [
        {
            "map_label": "unmapped",
            "Nl": 4,
            "Nm": 6,
            "warm_seconds": 0.10,
            "warm_over_baseline": 1.0,
            "rhs_rel_error_vs_unmapped": 0.0,
            "gamma_proxy": 0.2,
            "omega_proxy": 1.0,
            "gamma_proxy_abs_error_vs_unmapped": 0.0,
            "omega_proxy_abs_error_vs_unmapped": 0.0,
        },
        {
            "map_label": "identity",
            "Nl": 4,
            "Nm": 6,
            "warm_seconds": 0.11,
            "warm_over_baseline": 1.1,
            "rhs_rel_error_vs_unmapped": 1.0e-13,
            "gamma_proxy": 0.2,
            "omega_proxy": 1.0,
            "gamma_proxy_abs_error_vs_unmapped": 0.0,
            "omega_proxy_abs_error_vs_unmapped": 0.0,
        },
    ]
    scorecard = {
        "kind": "mapped_velocity_rhs_tiny_dense_eigen_scorecard",
        "readiness": {
            "identity_dense_operator_matches_unmapped": True,
            "all_dense_metrics_finite": True,
        },
        "rows": [],
    }

    payload = mod._build_summary(
        rows,
        config="examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear_miller.toml",
        backend="cpu",
        repeats=3,
        state="z_wave",
        z_variation_norm=0.5,
        nonlinear_weight=1.0,
        identity_tolerance=1.0e-10,
        dense_eigen_max_size=128,
        eigen_scorecard=scorecard,
    )

    assert payload["kind"] == "mapped_velocity_rhs_readiness"
    assert payload["case"] == "runtime_cyclone_nonlinear_miller"
    assert payload["nonlinear_terms_forced_linear"] is True
    assert payload["readiness"]["identity_map_matches_unmapped"] is True
    assert payload["readiness"]["tiny_dense_eigen_scorecard_available"] is True
    assert payload["readiness"]["identity_dense_operator_matches_unmapped"] is True
    assert payload["eigen_scorecard"] is scorecard
    assert "Nonlinear real-space mapped-basis support is not asserted" in payload["claim_scope"]


def test_write_json_and_csv_roundtrip_with_lf(tmp_path: Path) -> None:
    rows = [
        {
            "case": "case",
            "backend": "cpu",
            "state": "z_wave",
            "Nl": 1,
            "Nm": 2,
            "vector_size": 2,
            "map_label": "identity",
            "parallel_shift": 0.0,
            "parallel_log_scale": 0.0,
            "perpendicular_log_scale": 0.0,
            "parallel_scale": 1.0,
            "perpendicular_scale": 1.0,
            "compile_execute_seconds": 0.1,
            "warm_seconds": 0.01,
            "baseline_warm_seconds": 0.01,
            "warm_over_baseline": 1.0,
            "rhs_norm": 1.0,
            "phi_norm": 2.0,
            "rhs_rel_error_vs_unmapped": 0.0,
            "gamma_proxy": 0.0,
            "omega_proxy": 0.0,
            "gamma_proxy_abs_error_vs_unmapped": 0.0,
            "omega_proxy_abs_error_vs_unmapped": 0.0,
            "dense_gamma": None,
            "dense_omega": None,
            "dense_gamma_abs_error_vs_unmapped": None,
            "dense_omega_abs_error_vs_unmapped": None,
        }
    ]
    json_path = tmp_path / "readiness.json"
    csv_path = tmp_path / "readiness.csv"

    mod._write_json(json_path, {"kind": "mapped"})
    mod._write_csv(csv_path, rows)

    assert json.loads(json_path.read_text(encoding="utf-8")) == {"kind": "mapped"}
    raw = csv_path.read_bytes()
    assert b"\r" not in raw
    assert b"map_label" in raw


def test_tracked_mapped_velocity_rhs_artifact_is_scoped_and_replayable() -> None:
    artifact = ROOT / "docs" / "_static" / "mapped_velocity_rhs_readiness.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))

    assert payload["kind"] == "mapped_velocity_rhs_readiness"
    assert payload["case"] == "runtime_cyclone_nonlinear_miller"
    assert payload["state"] == "z_wave"
    assert payload["readiness"]["identity_map_matches_unmapped"] is True
    assert payload["readiness"]["all_proxy_metrics_finite"] is True
    assert payload["readiness"]["tiny_dense_eigen_scorecard_available"] is True
    assert payload["readiness"]["identity_dense_operator_matches_unmapped"] is True
    assert payload["eigen_scorecard"]["matrix_size"] == 24
    assert payload["eigen_scorecard"]["readiness"]["identity_dense_operator_matches_unmapped"] is True
    assert payload["max_identity_rhs_rel_error"] == 0.0
    assert payload["row_count"] >= 8
    assert payload["max_mapped_warm_over_unmapped"] < 2.5
    assert "Nonlinear real-space mapped-basis support is not asserted" in payload["claim_scope"]
