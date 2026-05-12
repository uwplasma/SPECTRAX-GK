from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from spectraxgk.nonlinear_parallel import (
    NonlinearDomainDecompositionPlan,
    build_nonlinear_domain_decomposition_plan,
    deterministic_nonlinear_domain_state,
    nonlinear_domain_identity_report,
    nonlinear_domain_parallel_identity_gate,
    prototype_nonlinear_domain_decomposed_step,
    prototype_nonlinear_domain_serial_step,
)


def test_nonlinear_domain_plan_uses_static_halo_chunks() -> None:
    plan = build_nonlinear_domain_decomposition_plan(
        (7, 3),
        axis=-2,
        num_domains=3,
    )

    assert plan == NonlinearDomainDecompositionPlan(
        state_shape=(7, 3),
        axis=0,
        chunk_sizes=(3, 2, 2),
        halo=1,
    )
    assert plan.num_domains == 3
    assert plan.domain_size == 7
    assert plan.offsets == (0, 3, 5)
    assert plan.to_dict() == {
        "state_shape": (7, 3),
        "axis": 0,
        "chunk_sizes": (3, 2, 2),
        "halo": 1,
    }


def test_nonlinear_domain_identity_gate_enables_only_matching_decomposition() -> None:
    state = deterministic_nonlinear_domain_state((6, 4))
    plan = build_nonlinear_domain_decomposition_plan(state.shape, num_domains=2)

    gated_state, report = nonlinear_domain_parallel_identity_gate(
        state,
        plan,
        dt=0.025,
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    serial = prototype_nonlinear_domain_serial_step(state, axis=plan.axis, dt=0.025)
    decomposed = prototype_nonlinear_domain_decomposed_step(state, plan, dt=0.025)

    assert report.identity_passed is True
    assert report.decomposed_path_enabled is True
    assert report.max_abs_error <= report.atol
    assert report.max_rel_error <= report.rtol
    assert "no production routing or speedup claim" in report.claim_scope
    assert jnp.allclose(decomposed, serial, atol=1.0e-6, rtol=1.0e-6)
    assert jnp.allclose(gated_state, decomposed, atol=1.0e-6, rtol=1.0e-6)


def test_nonlinear_domain_identity_report_fails_closed_on_mismatch() -> None:
    state = deterministic_nonlinear_domain_state((5, 3))
    plan = build_nonlinear_domain_decomposition_plan(state.shape, num_domains=2)
    serial = prototype_nonlinear_domain_serial_step(state, axis=plan.axis, dt=0.05)
    perturbed = serial.at[0, 0].add(1.0e-3)

    report = nonlinear_domain_identity_report(
        serial,
        perturbed,
        plan,
        atol=1.0e-6,
        rtol=1.0e-6,
    )

    assert report.identity_passed is False
    assert report.decomposed_path_enabled is False
    assert report.max_abs_error > report.atol
    assert report.to_dict()["identity_passed"] is False


def test_nonlinear_domain_decomposed_step_is_jax_jittable_with_static_plan() -> None:
    state = deterministic_nonlinear_domain_state((6, 4))
    plan = build_nonlinear_domain_decomposition_plan(state.shape, axis=0, num_domains=3)

    serial = prototype_nonlinear_domain_serial_step(state, axis=plan.axis, dt=0.05)
    jit_step = jax.jit(
        lambda item: prototype_nonlinear_domain_decomposed_step(item, plan, dt=0.05)
    )
    decomposed = jit_step(state)

    assert jnp.allclose(decomposed, serial, atol=1.0e-6, rtol=1.0e-6)


def test_nonlinear_domain_plan_rejects_unsupported_or_invalid_domains() -> None:
    with pytest.raises(ValueError, match="num_domains cannot exceed"):
        build_nonlinear_domain_decomposition_plan((2, 4), num_domains=3)
    with pytest.raises(ValueError, match="one-cell halo"):
        build_nonlinear_domain_decomposition_plan((4, 4), halo=2)
    with pytest.raises(ValueError, match="non-empty"):
        build_nonlinear_domain_decomposition_plan((0, 4))
