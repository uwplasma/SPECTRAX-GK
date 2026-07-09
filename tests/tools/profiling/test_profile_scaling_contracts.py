"""Contract tests for profiling and scaling helper tools."""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from support.paths import REPO_ROOT, load_profiling_tool
from tools.profiling import profile_linear_rhs_parallel_slices as linear_slices
from tools.profiling import profile_linear_rhs_parallel_slices_sweep as linear_sweep

independent_ky = load_profiling_tool("profile_independent_ky_scan_scaling")
linear_terms = load_profiling_tool("profile_linear_rhs_terms")
spectral_domain = load_profiling_tool("profile_nonlinear_spectral_domain_routing")
quasilinear_uq = load_profiling_tool("profile_quasilinear_uq_ensemble_scaling")


def test_independent_ky_scan_scaling_parser_defaults_to_large_solver_case() -> None:
    args = independent_ky.build_parser().parse_args([])

    assert args.out_prefix == independent_ky.DEFAULT_PREFIX
    assert args.backend == "cpu"
    assert args.devices == [1, 2, 4]
    assert args.ny == 128
    assert args.nz == 96
    assert args.steps == 240
    assert len(args.ky) == 12


def test_independent_ky_scan_scaling_splits_ky_without_empty_chunks() -> None:
    chunks = independent_ky._split_ky(np.asarray([0.1, 0.2, 0.3]), 8)

    assert [chunk.tolist() for chunk in chunks] == [[0.1], [0.2], [0.3]]


def test_independent_ky_scan_scaling_worker_env_is_backend_specific() -> None:
    cpu_env = independent_ky._worker_env({}, backend="cpu", worker_index=3)
    gpu_env = independent_ky._worker_env({}, backend="gpu", worker_index=1)

    assert cpu_env["JAX_PLATFORMS"] == "cpu"
    assert cpu_env["OMP_NUM_THREADS"] == "1"
    assert gpu_env["JAX_PLATFORMS"] == "cuda"
    assert gpu_env["CUDA_VISIBLE_DEVICES"] == "1"
    assert gpu_env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"


def test_independent_ky_scan_scaling_identity_metrics_pass_equal_values() -> None:
    reference = {"ky": [0.1, 0.2], "gamma": [0.3, 0.4], "omega": [-0.1, -0.2]}
    row = {"ky": [0.1, 0.2], "gamma": [0.3, 0.4], "omega": [-0.1, -0.2], "error": None}

    metrics = independent_ky._identity_metrics(reference, row)

    assert metrics["identity_gate_pass"] is True
    assert metrics["max_gamma_abs_error"] == 0.0
    assert metrics["max_omega_abs_error"] == 0.0


def test_profile_linear_rhs_parallel_slices_builds_summary(monkeypatch) -> None:
    class FakeGrid:
        ky = np.asarray([0.0, 0.3])
        z = np.asarray([0.0, 1.0, 2.0, 3.0])

    def fake_problem(**_kwargs):  # type: ignore[no-untyped-def]
        return (
            jnp.ones((1, 4, 2, 1, 4), dtype=jnp.complex64),
            object(),
            object(),
            FakeGrid(),
        )

    def fake_rhs(state, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        return 2.0 * state, jnp.ones((2, 1, 4), dtype=jnp.complex64)

    monkeypatch.setattr(linear_slices, "build_problem", fake_problem)
    monkeypatch.setattr("jax.devices", lambda _kind=None: [object(), object()])
    monkeypatch.setattr("spectraxgk.linear.linear_rhs_cached", fake_rhs)
    monkeypatch.setattr("spectraxgk.linear.linear_rhs_parallel_cached", fake_rhs)

    summary = linear_slices.profile_linear_rhs_parallel_slices(
        platform="cpu",
        requested_devices=2,
        nx=1,
        ny=2,
        nz=4,
        nl=1,
        nm=4,
        warmups=0,
        repeats=1,
        atol=1.0e-12,
        rtol=1.0e-12,
    )

    assert summary["identity_passed"] is True
    assert summary["max_abs_error"] == 0.0
    assert summary["speedup"] > 0.0
    assert len(summary["rows"]) == 2


def test_profile_linear_rhs_parallel_slices_writes_artifacts(tmp_path: Path) -> None:
    summary = {
        "rows": [
            {"route": "serial", "median_s": 0.02, "samples_s": [0.02]},
            {"route": "sharded", "median_s": 0.03, "samples_s": [0.03]},
        ],
        "atol": 1.0e-8,
        "rtol": 1.0e-8,
        "identity_passed": True,
        "speedup": 0.67,
        "max_abs_error": 0.0,
        "max_rel_error": 0.0,
        "max_phi_abs_error": 0.0,
    }
    out = tmp_path / "linear_rhs_parallel_slices_profile"
    paths = linear_slices.write_artifacts(summary, out)

    assert json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))["identity_passed"] is True
    assert "median_s" in out.with_suffix(".csv").read_text(encoding="utf-8")
    assert Path(paths["png"]).exists()
    assert Path(paths["pdf"]).exists()


def test_profile_linear_rhs_parallel_slices_sweep_builds_summary(monkeypatch) -> None:
    def fake_profile(**kwargs):  # type: ignore[no-untyped-def]
        devices = int(kwargs["requested_devices"])
        nm = int(kwargs["nm"])
        serial = float(nm) * 1.0e-3
        sharded = serial / max(devices, 1)
        return {
            "state_shape": (4, nm, 8, 1, 16),
            "serial_median_s": serial,
            "sharded_median_s": sharded,
            "speedup": serial / sharded,
            "identity_passed": True,
            "max_abs_error": 0.0,
            "max_rel_error": 0.0,
            "max_phi_abs_error": 0.0,
        }

    monkeypatch.setattr(linear_sweep, "profile_linear_rhs_parallel_slices", fake_profile)

    summary = linear_sweep.run_sweep(
        platform="cpu",
        devices=[1, 2],
        nms=[8, 16],
        nx=1,
        ny=4,
        nz=8,
        nl=2,
        warmups=0,
        repeats=1,
        atol=1.0e-6,
        rtol=1.0e-6,
    )

    assert summary["identity_passed"] is True
    assert len(summary["rows"]) == 4
    assert {row["speedup"] for row in summary["rows"]} == {1.0, 2.0}


def test_profile_linear_rhs_parallel_slices_sweep_writes_artifacts(tmp_path: Path) -> None:
    summary = {
        "identity_passed": True,
        "rtol": 1.0e-5,
        "rows": [
            {
                "platform": "cpu",
                "requested_devices": 1,
                "nm": 8,
                "state_shape": (2, 8, 4, 1, 8),
                "serial_median_s": 0.02,
                "sharded_median_s": 0.02,
                "speedup": 1.0,
                "identity_passed": True,
                "max_abs_error": 0.0,
                "max_rel_error": 0.0,
                "max_phi_abs_error": 0.0,
            },
            {
                "platform": "cpu",
                "requested_devices": 2,
                "nm": 8,
                "state_shape": (2, 8, 4, 1, 8),
                "serial_median_s": 0.02,
                "sharded_median_s": 0.012,
                "speedup": 1.67,
                "identity_passed": True,
                "max_abs_error": 0.0,
                "max_rel_error": 0.0,
                "max_phi_abs_error": 0.0,
            },
        ],
    }
    out = tmp_path / "linear_rhs_parallel_slices_sweep"
    paths = linear_sweep.write_artifacts(summary, out)

    assert json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))["identity_passed"] is True
    assert "requested_devices" in out.with_suffix(".csv").read_text(encoding="utf-8")
    assert Path(paths["png"]).exists()
    assert Path(paths["pdf"]).exists()


def test_linear_rhs_terms_summary_reports_dominant_and_zero_norm_terms() -> None:
    payload = linear_terms._build_summary(
        [
            {"term": "field_solve", "seconds": 0.1, "norm": 1.0},
            {"term": "streaming", "seconds": 0.2, "norm": 2.0},
            {"term": "collisions", "seconds": 0.4, "norm": 0.0},
            {"term": "linked_abs_kz", "seconds": 0.3, "norm": 1.0e-16},
            {"term": "full_linear_rhs", "seconds": 1.2, "norm": 3.0},
        ],
        config="examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear.toml",
        ky=0.3,
        kx=None,
        nl=4,
        nm=8,
        repeats=5,
        backend="cpu",
        state="z_wave",
        z_variation_norm=0.25,
    )

    assert payload["kind"] == "linear_rhs_terms_profile_summary"
    assert payload["case"] == "runtime_cyclone_nonlinear"
    assert payload["state"] == "z_wave"
    assert payload["z_variation_norm"] == 0.25
    assert payload["dominant_measured_term"] == "collisions"
    assert payload["dominant_nonzero_norm_term"] == "streaming"
    assert payload["zero_norm_terms_by_time"][0]["term"] == "collisions"
    assert payload["full_over_sum_independently_measured_components"] == 1.2
    assert "initial state only" in payload["claim_scope"]


def test_linear_rhs_terms_write_summary_json_roundtrips(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    linear_terms._write_summary_json({"kind": "linear", "value": 2.0}, path)

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "kind": "linear",
        "value": 2.0,
    }


def test_linear_rhs_terms_inject_z_wave_adds_parallel_variation() -> None:
    state = jnp.zeros((1, 4, 3, 2, 1, 5), dtype=jnp.complex64)

    out = linear_terms._inject_z_wave(state, ky_index=1, kx_index=0, amplitude=0.2, z_mode=1)

    assert linear_terms._z_variation_norm(out) > 0.0
    assert jnp.linalg.norm(out[0, 0, 2, 1, 0]) > 0.0


def test_linear_rhs_terms_hypercollision_kz_source_uses_nu_hyper_m_path() -> None:
    state = jnp.zeros((1, 2, 4, 1, 1, 4), dtype=jnp.complex64)
    state = state.at[0, 0, 3, 0, 0, :].set(
        jnp.asarray([0.0, 1.0, 0.0, -1.0], dtype=jnp.complex64)
    )
    mask_kz = jnp.ones((2, 4, 1, 1, 1), dtype=bool)
    m_pow = jnp.ones((2, 4, 1, 1, 1), dtype=jnp.float32)

    source = linear_terms._hypercollision_kz_source(
        state,
        weight=jnp.asarray(1.0, dtype=jnp.float32),
        hypercollisions_kz=jnp.asarray(1.0, dtype=jnp.float32),
        nu_hyper_m=jnp.asarray(1.0, dtype=jnp.float32),
        m_norm_kz_factor=jnp.asarray(0.5, dtype=jnp.float32),
        vth=jnp.asarray([2.0], dtype=jnp.float32),
        kpar_scale=jnp.asarray(1.0, dtype=jnp.float32),
        mask_kz=mask_kz,
        m_pow=m_pow,
    )

    assert jnp.linalg.norm(source) > 0.0
    assert jnp.allclose(source[0, 0, 3, 0, 0, 1], -2.3 + 0.0j)


def test_linear_rhs_terms_tracked_miller_profile_is_active_artifact() -> None:
    path = REPO_ROOT / "docs" / "_static" / "linear_rhs_terms_profile_miller_cpu.json"
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["kind"] == "linear_rhs_terms_profile_summary"
    assert payload["case"] == "runtime_cyclone_nonlinear_miller"
    assert payload["state"] == "z_wave_linear_kick"
    assert payload["full_linear_rhs_seconds"] > 0.0
    assert payload["rows"]["streaming"]["norm"] > 0.0
    assert payload["dominant_nonzero_norm_term"] == "streaming"


def test_nonlinear_spectral_domain_routing_parser_defaults() -> None:
    args = spectral_domain.build_parser().parse_args([])

    assert args.out_prefix == spectral_domain.DEFAULT_OUT_PREFIX
    assert (args.nl, args.nm, args.ny, args.nx, args.nz) == (2, 4, 32, 32, 4)
    assert args.y_chunks == (16, 16)
    assert args.x_chunks == (16, 16)
    assert args.min_speedup == 1.5


def test_nonlinear_spectral_domain_routing_builds_identity_payload() -> None:
    payload = spectral_domain.build_profile(
        shape=(2, 2, 4, 4, 1),
        y_chunks=(2, 2),
        x_chunks=(2, 2),
        steps=1,
        dt=0.001,
        warmups=0,
        repeats=1,
        min_speedup=1.5,
        atol=5.0e-6,
        rtol=5.0e-6,
    )

    assert payload["kind"] == "nonlinear_spectral_domain_routing_profile"
    assert payload["identity_passed"] is True
    assert payload["decomposed_path_enabled"] is True
    assert payload["timing_identity_max_abs_error"] <= payload["atol"]
    assert payload["timing_identity_max_rel_error"] <= payload["rtol"]
    assert payload["production_speedup_claim_allowed"] is False
    assert payload["work_model"]["num_tiles"] == 4
    assert payload["work_model"]["production_speedup_feasible"] is False
    assert payload["communication_to_owned_work_ratio"] > 1.0
    assert payload["parallel_efficiency_ceiling"] < 0.5
    assert payload["serial_stats_s"]["median"] > 0.0
    assert payload["logical_domain_stats_s"]["median"] > 0.0
    assert payload["strong_speedup_vs_serial"] is not None


def test_nonlinear_spectral_domain_routing_writes_artifacts(tmp_path: Path) -> None:
    payload = spectral_domain.build_profile(
        shape=(2, 2, 4, 4, 1),
        y_chunks=(2, 2),
        x_chunks=(2, 2),
        steps=1,
        dt=0.001,
        warmups=0,
        repeats=1,
        min_speedup=1.5,
        atol=5.0e-6,
        rtol=5.0e-6,
    )

    paths = spectral_domain.write_artifacts(payload, tmp_path / "domain_profile")

    for path in paths.values():
        assert Path(path).exists()
    saved = json.loads((tmp_path / "domain_profile.json").read_text(encoding="utf-8"))
    assert saved["identity_passed"] is True
    assert saved["work_model_speedup_feasible"] is False


def test_quasilinear_uq_ensemble_scaling_parser_defaults_to_bounded_solver_case() -> None:
    args = quasilinear_uq.build_parser().parse_args([])

    assert args.out_prefix == quasilinear_uq.DEFAULT_PREFIX
    assert args.backend == "cpu"
    assert args.devices == [1, 2, 4]
    assert args.gradients == [2.20, 2.40, 2.60, 2.80, 3.00, 3.20]
    assert args.ky == [0.10, 0.20, 0.30, 0.40, 0.50]
    assert args.steps == 2000
    assert args.sample_stride == 10
    assert args.fit_start_fraction == 0.5
    assert args.fit_end_fraction == 0.95


def test_quasilinear_uq_ensemble_scaling_reduced_observable_is_positive() -> None:
    obs = quasilinear_uq._quasilinear_reduced_observables(
        np.asarray([0.2, 0.4]),
        np.asarray([0.1, -0.1]),
        np.asarray([0.3, 0.1]),
    )

    assert obs["heat_flux_proxy"] > 0.0
    assert obs["weighted_growth"] == 0.1
    assert obs["omega_span"] == 0.19999999999999998


def test_quasilinear_uq_ensemble_scaling_identity_metrics_detect_equal_members() -> None:
    members = [
        {"R_over_LTi": 2.4, "heat_flux_proxy": 1.5, "gamma": [0.1, 0.2]},
        {"R_over_LTi": 2.7, "heat_flux_proxy": 2.5, "gamma": [0.3, 0.4]},
    ]

    metrics = quasilinear_uq._identity_metrics(
        {"members": members},
        {"members": list(reversed(members)), "error": None},
        value_rtol=1.0e-12,
        value_atol=1.0e-12,
    )

    assert metrics["identity_gate_pass"] is True
    assert metrics["max_heat_flux_proxy_abs_error"] == 0.0
    assert metrics["max_gamma_abs_error"] == 0.0


def test_quasilinear_uq_ensemble_scaling_worker_env_selects_device() -> None:
    gpu_env = quasilinear_uq._worker_env({}, backend="gpu", worker_index=1)
    cpu_env = quasilinear_uq._worker_env({}, backend="cpu", worker_index=0)

    assert gpu_env["JAX_PLATFORMS"] == "cuda"
    assert gpu_env["CUDA_VISIBLE_DEVICES"] == "1"
    assert cpu_env["JAX_PLATFORMS"] == "cpu"
    assert cpu_env["OMP_NUM_THREADS"] == "1"
