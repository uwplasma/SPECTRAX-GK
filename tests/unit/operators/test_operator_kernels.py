"""Focused operator-kernel and reduced full-operator regression checks."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import jax.scipy.linalg
import numpy as np
import pytest

from spectraxgk.config import GridConfig
from spectraxgk.core.grid import build_spectral_grid
from spectraxgk.core.velocity import J_l_all, gamma0, sum_Jl2
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.linear import LinearParams, build_linear_cache
from spectraxgk.linear import LinearTerms, linear_rhs_cached
from spectraxgk.operators import hermite_streaming
from spectraxgk.operators.linear import (
    apply_collision_moment_matrix,
    apply_multispecies_collision_moment_matrix,
    assemble_drift_kinetic_sugama_matrix,
    DriftKineticSugamaOperator,
    drift_kinetic_sugama_pair_matrices,
    interpolate_collision_moment_matrix,
    load_collision_moment_matrix,
)
from spectraxgk.operators.linear.dissipation import (
    collision_quadratic_rate,
    collisions_contribution,
    multispecies_collision_invariant_rates,
)
from spectraxgk.terms.assembly import assemble_rhs_cached
from spectraxgk.terms.config import TermConfig
from spectraxgk.terms.linear_terms import (
    drift_kinetic_coulomb_six_moment_contribution,
    drift_kinetic_sugama_six_moment_contribution,
)
from spectraxgk.runtime import run_runtime_scan
from spectraxgk.workflows.runtime.toml import load_runtime_from_toml


def test_streaming_zero_kpar() -> None:
    """Hermite streaming should vanish at k_parallel = 0."""
    state = jnp.ones((2, 4))

    derivative = hermite_streaming(state, kpar=0.0, vth=1.0)

    assert jnp.allclose(derivative, 0.0)


def test_streaming_energy_conservation() -> None:
    """Hermite streaming is skew-adjoint for constant k_parallel."""
    state = jnp.sin(jnp.arange(12, dtype=jnp.float32)).reshape(3, 4)

    derivative = hermite_streaming(state, kpar=0.7, vth=1.2)
    energy_rate = jnp.real(jnp.sum(jnp.conj(state) * derivative))

    assert jnp.isclose(energy_rate, 0.0, atol=1e-6)


def test_streaming_known_case() -> None:
    """Simple Hermite ladder behavior should match a hand calculation."""
    state = jnp.array([[1.0, 2.0, 0.0]])

    derivative = hermite_streaming(state, kpar=1.0, vth=1.0)

    assert jnp.isclose(derivative[0, 0], -1j * 2.0)
    assert jnp.isclose(derivative[0, 1], -1j * 1.0)
    assert jnp.isclose(derivative[0, 2], -1j * (jnp.sqrt(2.0) * 2.0))


def test_streaming_invalid_shape() -> None:
    """Hermite axis length must be positive."""
    with pytest.raises(ValueError):
        hermite_streaming(jnp.zeros((2, 0)), kpar=1.0, vth=1.0)


def test_linear_operator_package_reexports_streaming_kernel() -> None:
    from spectraxgk.operators.linear import hermite_streaming as package_streaming
    from spectraxgk.operators.linear.moments import hermite_streaming as streaming_impl

    assert hermite_streaming is package_streaming
    assert package_streaming is streaming_impl


@pytest.mark.parametrize(
    ("model", "direct"),
    [
        ("sugama", drift_kinetic_sugama_six_moment_contribution),
        ("coulomb", drift_kinetic_coulomb_six_moment_contribution),
    ],
)
def test_generated_collision_matrix_matches_direct_published_equations(
    model, direct
) -> None:
    state = (
        jnp.arange(2 * 2 * 4 * 2, dtype=jnp.float32).reshape(2, 2, 4, 1, 1, 2) + 0.17j
    ).astype(jnp.complex64)
    frequency = jnp.asarray([0.2, 0.35], dtype=jnp.float32)
    table_result = apply_collision_moment_matrix(
        state,
        load_collision_moment_matrix(model),
        nu=frequency,
        weight=jnp.asarray(0.7),
    )
    direct_result = direct(state, nu=frequency, weight=jnp.asarray(0.7))
    np.testing.assert_allclose(
        np.asarray(table_result), np.asarray(direct_result), rtol=2.0e-6, atol=2.0e-6
    )


def test_collision_matrix_application_is_differentiable_and_fail_closed() -> None:
    state = (
        jnp.arange(8, dtype=jnp.float32).reshape(2, 4, 1, 1, 1).astype(jnp.complex64)
    )
    matrix = jnp.asarray(load_collision_moment_matrix("coulomb"))
    frequency = jnp.asarray([0.3], dtype=jnp.float32)
    tangent = jax.jvp(
        lambda scale, nu: apply_collision_moment_matrix(scale * state, matrix, nu=nu),
        (jnp.asarray(1.0), frequency),
        (jnp.asarray(0.4), jnp.asarray([0.2])),
    )[1]
    step = jnp.asarray(1.0e-3)
    plus = apply_collision_moment_matrix(
        (1.0 + 0.4 * step) * state, matrix, nu=frequency + 0.2 * step
    )
    minus = apply_collision_moment_matrix(
        (1.0 - 0.4 * step) * state, matrix, nu=frequency - 0.2 * step
    )
    np.testing.assert_allclose(
        np.asarray(tangent),
        np.asarray((plus - minus) / (2.0 * step)),
        rtol=3.0e-4,
        atol=3.0e-4,
    )

    with pytest.raises(ValueError, match="collision model must be one of"):
        load_collision_moment_matrix("unknown")
    with pytest.raises(ValueError, match="collision matrix must have shape"):
        apply_collision_moment_matrix(state, jnp.eye(7), nu=frequency)
    with pytest.raises(ValueError, match="nu must have length 2"):
        apply_collision_moment_matrix(
            jnp.broadcast_to(state, (2,) + state.shape), matrix, nu=jnp.ones(3)
        )


def test_collision_kperp_interpolation_matches_nodes_and_direct_operator() -> None:
    grid = jnp.asarray([0.0, 1.0, 2.0], dtype=jnp.float32)
    base = jnp.asarray(load_collision_moment_matrix("coulomb"), dtype=jnp.float32)
    table = jnp.stack([(1.0 + 0.5 * value) * base for value in grid])
    kperp = jnp.asarray([[[-0.3, 0.5]], [[1.0, 2.7]]], dtype=jnp.float32)
    interpolated = jax.jit(
        lambda target: interpolate_collision_moment_matrix(grid, table, target)
    )(kperp)
    expected = base[:, :, None, None, None] * (
        1.0 + 0.5 * jnp.clip(kperp, grid[0], grid[-1])
    )
    np.testing.assert_allclose(interpolated, expected, rtol=2.0e-6, atol=2.0e-6)

    state = (
        jnp.arange(2 * 4 * 4, dtype=jnp.float32).reshape(2, 4, 2, 1, 2) + 0.13j
    ).astype(jnp.complex64)
    result = apply_collision_moment_matrix(state, interpolated, nu=jnp.asarray(0.3))
    direct = drift_kinetic_coulomb_six_moment_contribution(
        state, nu=jnp.asarray(0.3)
    ) * (1.0 + 0.5 * jnp.clip(kperp, grid[0], grid[-1]))
    np.testing.assert_allclose(result, direct, rtol=3.0e-6, atol=3.0e-6)


def test_collision_matrix_kperp_interpolation_species_jvp_and_validation() -> None:
    grid = jnp.asarray([0.0, 1.0, 2.0], dtype=jnp.float32)
    base = jnp.asarray(load_collision_moment_matrix("sugama"), dtype=jnp.float32)
    shared_table = jnp.stack([(1.0 + value) * base for value in grid])
    species_table = jnp.stack([shared_table, 2.0 * shared_table])
    target = jnp.asarray([[[[0.25]]], [[[1.25]]]], dtype=jnp.float32)
    result = interpolate_collision_moment_matrix(grid, species_table, target)
    np.testing.assert_allclose(result[0], 1.25 * base[:, :, None, None, None])
    np.testing.assert_allclose(result[1], 4.5 * base[:, :, None, None, None])
    state = (
        jnp.arange(2 * 2 * 4, dtype=jnp.float32).reshape(2, 2, 4, 1, 1, 1) + 0.2j
    ).astype(jnp.complex64)
    frequency = jnp.asarray([0.2, 0.3], dtype=jnp.float32)
    applied = jax.jit(apply_collision_moment_matrix)(state, result, nu=frequency)
    direct = drift_kinetic_sugama_six_moment_contribution(state, nu=frequency)
    expected = direct * jnp.asarray([1.25, 4.5])[:, None, None, None, None, None]
    np.testing.assert_allclose(applied, expected, rtol=3.0e-6, atol=3.0e-6)

    scalar = jnp.asarray(0.6, dtype=jnp.float32)
    tangent = jax.jvp(
        lambda value: interpolate_collision_moment_matrix(grid, shared_table, value),
        (scalar,),
        (jnp.asarray(0.4, dtype=jnp.float32),),
    )[1]
    step = jnp.asarray(1.0e-3, dtype=jnp.float32)
    finite_difference = (
        interpolate_collision_moment_matrix(grid, shared_table, scalar + 0.4 * step)
        - interpolate_collision_moment_matrix(grid, shared_table, scalar - 0.4 * step)
    ) / (2.0 * step)
    np.testing.assert_allclose(tangent, finite_difference, rtol=4.0e-4, atol=4.0e-4)

    with pytest.raises(ValueError, match="at least two points"):
        interpolate_collision_moment_matrix(
            jnp.asarray([0.0]), shared_table[:1], scalar
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        interpolate_collision_moment_matrix(
            jnp.asarray([0.0, 2.0, 1.0]), shared_table, scalar
        )
    with pytest.raises(ValueError, match="must be square"):
        interpolate_collision_moment_matrix(grid, jnp.ones((3, 8, 7)), scalar)
    with pytest.raises(ValueError, match="species-leading"):
        interpolate_collision_moment_matrix(grid, species_table, jnp.ones((3, 1)))


def test_collision_table_converges_for_physical_finite_larmor_operator() -> None:
    """Held-out Mandell finite-b matrices converge at interpolation design order."""

    nl, nm = 2, 3
    mode_count = nl * nm
    eigenvalues = jnp.asarray(
        [[2 * ell + hermite for hermite in range(nm)] for ell in range(nl)],
        dtype=jnp.float32,
    )

    def collision_matrix(kperp: float) -> jnp.ndarray:
        b = jnp.asarray([[[[kperp**2]]]], dtype=jnp.float32)
        gyroaverage = jnp.moveaxis(J_l_all(b, nl - 1), 0, 1)
        lower = jnp.pad(gyroaverage[:, :-1], ((0, 0), (1, 0), (0, 0), (0, 0), (0, 0)))
        columns = []
        for column in range(mode_count):
            ell, hermite = column % nl, column // nl
            basis = jnp.zeros((1, nl, nm, 1, 1, 1), dtype=jnp.complex64)
            basis = basis.at[0, ell, hermite, 0, 0, 0].set(1.0)
            contribution = collisions_contribution(
                basis,
                G=basis,
                Jl=gyroaverage,
                JlB=gyroaverage + lower,
                b=b,
                nu=jnp.asarray([1.0]),
                lb_lam=eigenvalues,
                weight=jnp.asarray(1.0),
            )
            packed = jnp.swapaxes(contribution, 1, 2).reshape(1, mode_count)
            columns.append(packed[0])
        return jnp.stack(columns, axis=1).real

    targets = (0.17, 0.43, 0.79, 1.11)
    state = (
        jnp.arange(mode_count * 4, dtype=jnp.float32).reshape(nl, nm, 2, 1, 2) + 0.2j
    ).astype(jnp.complex64)
    spacings = np.asarray([0.4, 0.2, 0.1])
    errors = []
    for spacing in spacings:
        grid = jnp.arange(0.0, 1.6 + spacing / 2.0, spacing, dtype=jnp.float32)
        table = jnp.stack([collision_matrix(float(value)) for value in grid])
        interpolated = []
        direct = []
        for target in targets:
            matrix = interpolate_collision_moment_matrix(
                grid, table, jnp.asarray(target)
            )
            interpolated.append(
                apply_collision_moment_matrix(state, matrix, nu=jnp.asarray(1.0))
            )
            direct.append(
                apply_collision_moment_matrix(
                    state, collision_matrix(target), nu=jnp.asarray(1.0)
                )
            )
        interpolated_stack = jnp.stack(interpolated)
        direct_stack = jnp.stack(direct)
        errors.append(
            float(
                jnp.linalg.norm(interpolated_stack - direct_stack)
                / jnp.linalg.norm(direct_stack)
            )
        )

    observed_order = float(np.polyfit(np.log(spacings), np.log(errors), 1)[0])
    assert all(fine < coarse for coarse, fine in zip(errors, errors[1:]))
    assert errors[-1] < 1.0e-3
    assert 1.8 <= observed_order <= 2.3


def test_sugama_pair_matrices_recover_published_like_species_operator() -> None:
    test_matrix, field_matrix = drift_kinetic_sugama_pair_matrices(
        jnp.asarray(1.0), jnp.asarray(1.0)
    )
    np.testing.assert_allclose(
        np.asarray(test_matrix + field_matrix),
        load_collision_moment_matrix("sugama"),
        rtol=3.0e-6,
        atol=3.0e-6,
    )

    unequal_test, unequal_field = drift_kinetic_sugama_pair_matrices(
        jnp.asarray(4.0), jnp.asarray(2.0)
    )
    np.testing.assert_allclose(
        np.asarray([unequal_test[2, 6], unequal_test[4, 1]]),
        np.asarray([0.2746838629245758, -0.2001875340938568]),
        rtol=2.0e-6,
    )
    np.testing.assert_allclose(
        np.asarray([unequal_field[2, 6], unequal_field[3, 2]]),
        np.asarray([-1.0987355709075928, -0.3171776235103607]),
        rtol=2.0e-6,
    )


def test_multispecies_sugama_pair_operator_conserves_and_differentiates() -> None:
    density = jnp.asarray([1.3, 0.7], dtype=jnp.float32)
    mass = jnp.asarray([2.0, 1.0], dtype=jnp.float32)
    temperature = jnp.asarray([1.2, 0.8], dtype=jnp.float32)
    pair_matrix = jax.jit(assemble_drift_kinetic_sugama_matrix)(
        density, mass, temperature
    )

    state = (
        jnp.arange(2 * 2 * 4, dtype=jnp.float32).reshape(2, 2, 4, 1, 1, 1) + 0.13j
    ).astype(jnp.complex64)
    contribution = jax.jit(apply_multispecies_collision_moment_matrix)(
        state, pair_matrix
    )
    rates = multispecies_collision_invariant_rates(
        contribution, density=density, mass=mass, temperature=temperature
    )
    np.testing.assert_allclose(rates.particle_density, 0.0, atol=2.0e-6)
    np.testing.assert_allclose(rates.total_parallel_momentum, 0.0, atol=3.0e-6)
    np.testing.assert_allclose(rates.total_thermal_energy, 0.0, atol=7.0e-6)
    free_energy_rate = collision_quadratic_rate(
        state,
        contribution,
        weights=(density * temperature)[:, None, None, None, None, None],
    )
    assert float(free_energy_rate) < 0.0

    ratios = jnp.asarray([4.0, 2.0], dtype=jnp.float32)
    direction = jnp.asarray([0.3, -0.2], dtype=jnp.float32)

    def pair_function(values):
        return jnp.stack(drift_kinetic_sugama_pair_matrices(values[0], values[1]))

    tangent = jax.jvp(pair_function, (ratios,), (direction,))[1]
    step = jnp.asarray(1.0e-3, dtype=jnp.float32)
    finite_difference = (
        pair_function(ratios + step * direction)
        - pair_function(ratios - step * direction)
    ) / (2.0 * step)
    np.testing.assert_allclose(tangent, finite_difference, rtol=8.0e-4, atol=8.0e-4)

    with pytest.raises(ValueError, match="must be scalars"):
        drift_kinetic_sugama_pair_matrices(jnp.ones(2), jnp.asarray(1.0))
    with pytest.raises(ValueError, match="mass_ratio must be positive"):
        drift_kinetic_sugama_pair_matrices(jnp.asarray(0.0), jnp.asarray(1.0))
    with pytest.raises(ValueError, match="target/source species"):
        apply_multispecies_collision_moment_matrix(state, jnp.zeros((2, 8, 8)))


def test_multispecies_sugama_pair_relaxation_preserves_invariants() -> None:
    density = jnp.asarray([1.3, 0.7], dtype=jnp.float32)
    mass = jnp.asarray([2.0, 1.0], dtype=jnp.float32)
    temperature = jnp.asarray([1.2, 0.8], dtype=jnp.float32)
    matrix = assemble_drift_kinetic_sugama_matrix(density, mass, temperature)
    generator = jnp.transpose(matrix, (0, 2, 1, 3)).reshape(16, 16)
    initial_modes = jnp.asarray(
        [
            [0.0, 0.3, -0.2, 0.15, 0.4, -0.1, 0.2, -0.3],
            [0.0, -0.4, 0.1, -0.2, -0.3, 0.2, -0.1, 0.25],
        ],
        dtype=jnp.float32,
    )

    times = jnp.linspace(0.0, 30.0, 31)
    trajectory_modes = jax.vmap(
        lambda time: jax.scipy.linalg.expm(time * generator) @ initial_modes.reshape(-1)
    )(times).reshape(-1, 2, 8)
    trajectory = jnp.transpose(
        trajectory_modes.reshape(-1, 2, 4, 2, 1, 1, 1), (0, 1, 3, 2, 4, 5, 6)
    )

    invariants = jax.vmap(
        lambda state: multispecies_collision_invariant_rates(
            state, density=density, mass=mass, temperature=temperature
        )
    )(trajectory)
    assert (
        float(
            jnp.max(
                jnp.abs(invariants.particle_density - invariants.particle_density[0])
            )
        )
        < 1.0e-6
    )
    assert (
        float(
            jnp.max(
                jnp.abs(
                    invariants.total_parallel_momentum
                    - invariants.total_parallel_momentum[0]
                )
            )
        )
        < 5.0e-6
    )
    assert (
        float(
            jnp.max(
                jnp.abs(
                    invariants.total_thermal_energy - invariants.total_thermal_energy[0]
                )
            )
        )
        < 8.0e-6
    )

    collision_rhs = jax.vmap(
        lambda state: apply_multispecies_collision_moment_matrix(state, matrix)
    )(trajectory)
    residual_norm = jnp.linalg.norm(collision_rhs.reshape(times.size, -1), axis=1)
    # Monotone decay is required until the float32 matrix-exponential floor.
    assert bool(jnp.all(jnp.diff(residual_norm[:11]) < 0.0))
    assert float(residual_norm[-1] / residual_norm[0]) < 1.0e-5

    with pytest.raises(ValueError, match="equal length"):
        assemble_drift_kinetic_sugama_matrix(jnp.ones(2), jnp.ones(3), jnp.ones(2))
    with pytest.raises(ValueError, match="temperature must"):
        assemble_drift_kinetic_sugama_matrix(jnp.ones(2), jnp.ones(2), jnp.ones(3))
    with pytest.raises(ValueError, match="density must be positive"):
        assemble_drift_kinetic_sugama_matrix(
            jnp.asarray([1.0, 0.0]), jnp.ones(2), jnp.ones(2)
        )


def test_multispecies_sugama_operator_runs_through_linear_rhs() -> None:
    density = jnp.asarray([1.3, 0.7], dtype=jnp.float32)
    mass = jnp.asarray([2.0, 1.0], dtype=jnp.float32)
    temperature = jnp.asarray([1.2, 0.8], dtype=jnp.float32)
    params = LinearParams(
        charge_sign=jnp.asarray([1.0, -1.0]),
        density=density,
        mass=mass,
        temp=temperature,
        vth=jnp.ones(2),
        rho=jnp.asarray([1.0, 0.5]),
        tz=jnp.asarray([1.0, -1.0]),
    )
    grid = build_spectral_grid(GridConfig(Nx=2, Ny=2, Nz=4, Lx=6.0, Ly=6.0))
    geometry = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.18)
    state = (
        1.0e-3
        * jnp.arange(2 * 2 * 4 * 2 * 2 * 4, dtype=jnp.float32).reshape(2, 2, 4, 2, 2, 4)
    ).astype(jnp.complex64)
    cache = build_linear_cache(grid, geometry, params, Nl=2, Nm=4)
    operator = DriftKineticSugamaOperator.from_species(density, mass, temperature)
    terms = LinearTerms(
        streaming=0.0,
        mirror=0.0,
        curvature=0.0,
        gradb=0.0,
        diamagnetic=0.0,
        collisions=1.0,
        hypercollisions=0.0,
        hyperdiffusion=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )

    contribution, _ = linear_rhs_cached(
        state,
        cache,
        params,
        terms=terms,
        collision_operator=operator,
    )
    rates = multispecies_collision_invariant_rates(
        contribution, density=density, mass=mass, temperature=temperature
    )
    assert float(jnp.linalg.norm(contribution)) > 1.0
    np.testing.assert_allclose(rates.particle_density, 0.0, atol=2.0e-6)
    np.testing.assert_allclose(rates.total_parallel_momentum, 0.0, atol=4.0e-6)
    np.testing.assert_allclose(rates.total_thermal_energy, 0.0, atol=1.0e-5)


def test_gamma0_basic_properties() -> None:
    """Gamma0 should be 1 at b=0 and decrease with b."""
    b = jnp.array([0.0, 1.0, 4.0])

    values = gamma0(b)

    assert jnp.isclose(values[0], 1.0)
    assert values[1] < values[0]
    assert values[2] < values[1]


def test_gamma0_small_b_series() -> None:
    """Small-b series should be accurate to O(b^2)."""
    b = jnp.array(1.0e-3)

    value = gamma0(b)
    approx = 1.0 - b + 0.75 * b * b

    assert jnp.isclose(value, approx, rtol=1e-6, atol=1e-12)


def test_jl_shape_and_j0() -> None:
    """J_l should return the correct array shape and J0 factor."""
    b = jnp.array([0.0, 0.5])

    values = J_l_all(b, l_max=3)

    assert values.shape == (4, 2)
    assert jnp.allclose(values[0], jnp.exp(-0.5 * b))


def test_jl_zero_b_is_one() -> None:
    """At b=0 only the l=0 coefficient is nonzero."""
    b = jnp.array(0.0)

    values = J_l_all(b, l_max=4)

    assert jnp.isclose(values[0], 1.0)
    assert jnp.allclose(values[1:], 0.0)


def test_jl_large_b_decay() -> None:
    """Large b should suppress J0 through the exponential factor."""
    j0 = J_l_all(jnp.array(10.0), l_max=0)[0]

    assert j0 < 1.0e-2


def test_jl_large_b_stays_finite() -> None:
    """Large-b ETG coefficients should underflow cleanly."""
    b = jnp.array([100.0, 500.0, 2000.0], dtype=jnp.float32)

    values = J_l_all(b, l_max=23)

    assert jnp.all(jnp.isfinite(values))


def test_jl_first_order_coeff() -> None:
    """J1 should match the analytic (-b/2) * exp(-b/2) coefficient."""
    b = jnp.array(0.6)

    values = J_l_all(b, l_max=1)
    expected = -0.5 * b * jnp.exp(-0.5 * b)

    assert jnp.isclose(values[1], expected)


def test_sum_jl2_monotone_in_lmax() -> None:
    """Truncated sum of J_l^2 should increase with l_max."""
    b = jnp.array([0.2, 1.5])

    s2 = sum_Jl2(b, l_max=2)
    s5 = sum_Jl2(b, l_max=5)

    assert jnp.all(s5 >= s2)


def test_jl_invalid_lmax() -> None:
    """Negative l_max should raise a ValueError."""
    with pytest.raises(ValueError):
        J_l_all(jnp.array(0.0), l_max=-1)


def test_hyperdiffusion_damps_high_k() -> None:
    """k_perp hyperdiffusion should damp high-k modes more strongly."""
    grid_cfg = GridConfig(Nx=6, Ny=6, Nz=4, Lx=6.0, Ly=6.0)
    grid = build_spectral_grid(grid_cfg)
    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778, B0=1.0)
    params = LinearParams(D_hyper=0.5, p_hyper_kperp=2.0)
    cache = build_linear_cache(grid, geom, params, Nl=1, Nm=1)
    state = jnp.ones(
        (1, 1, grid.ky.size, grid.kx.size, grid.z.size),
        dtype=jnp.complex64,
    )
    terms = TermConfig(
        streaming=0.0,
        mirror=0.0,
        curvature=0.0,
        gradb=0.0,
        diamagnetic=0.0,
        collisions=0.0,
        hypercollisions=0.0,
        hyperdiffusion=1.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
        nonlinear=0.0,
    )

    derivative, _ = assemble_rhs_cached(state, cache, params, terms=terms)
    derivative = derivative[0, 0]
    kperp2 = (
        jnp.asarray(grid.ky)[:, None] ** 2 + jnp.asarray(grid.kx)[None, :] ** 2
    ).astype(float)
    mask = np.asarray(grid.dealias_mask, dtype=bool)
    kperp2_np = np.asarray(kperp2).copy()
    kperp2_np[~mask] = np.nan
    low_idx = np.unravel_index(np.nanargmin(kperp2_np), kperp2_np.shape)
    high_idx = np.unravel_index(np.nanargmax(kperp2_np), kperp2_np.shape)

    low = jnp.abs(derivative[low_idx[0], low_idx[1], 0])
    high = jnp.abs(derivative[high_idx[0], high_idx[1], 0])

    assert high > low


def test_full_operator_scan_relaxed() -> None:
    """Full operator should produce finite scans on a field-aligned grid."""
    grid = GridConfig(
        Nx=1,
        Ny=24,
        Nz=96,
        Lx=62.8,
        Ly=62.8,
        y0=20.0,
        ntheta=32,
        nperiod=2,
    )
    base_cfg, _ = load_runtime_from_toml("examples/linear/axisymmetric/cyclone.toml")
    cfg = replace(base_cfg, grid=grid)
    ky_values = np.array([0.2, 0.3, 0.4])

    scan = run_runtime_scan(
        cfg,
        ky_values,
        Nl=4,
        Nm=8,
        steps=200,
        dt=0.002,
        method="imex2",
        solver="time",
        batch_ky=True,
        sample_stride=2,
        auto_window=False,
        tmin=0.1,
        tmax=0.4,
        min_points=10,
        require_positive=False,
    )

    for gamma, omega in zip(scan.gamma, scan.omega, strict=True):
        assert np.isfinite(gamma)
        assert np.isfinite(omega)
        assert abs(gamma) < 50.0
        assert abs(omega) < 100.0
