import pytest
import jax
import jax.numpy as jnp

from spectraxgk._initialization_multispecies import initialize_simulation_parameters_multispecies
from spectraxgk._model_multispecies import (
    rhs_gk_multispecies,
    solve_phi_quasineutrality_multispecies,
    collision_lenard_bernstein_conserving_multispecies,
    enforce_conjugate_symmetry_fftshifted,
)


def _symmetry_error_fftshifted(Ak, params):
    iy, ix, iz = params["conj_y"], params["conj_x"], params["conj_z"]
    Aneg = jnp.take(Ak, iy, axis=-3)
    Aneg = jnp.take(Aneg, ix, axis=-2)
    Aneg = jnp.take(Aneg, iz, axis=-1)
    denom = jnp.linalg.norm(Ak.ravel()) + 1e-30
    return jnp.linalg.norm((Ak - jnp.conj(Aneg)).ravel()) / denom


@pytest.fixture(scope="module")
def params_linear():
    # Nonlinear off to isolate linear physics; small sizes.
    Nx, Ny, Nz = 5, 5, 9
    Nl, Nh = 2, 10
    species = [
        dict(name="ion", q=+1.0, T=1.0, n0=1.0, rho=1.0, vth=1.0, nu=0.0, Upar=0.0),
        dict(name="e",   q=-1.0, T=1.0, n0=1.0, rho=1.0, vth=1.0, nu=0.0, Upar=0.0),
    ]
    p = initialize_simulation_parameters_multispecies(
        dict(species=species, enable_nonlinear=False, enable_collisions=False, enforce_reality=True),
        Nx=Nx, Ny=Ny, Nz=Nz, Nl=Nl, Nh=Nh, timesteps=10, dt=1e-2
    )
    return p


def test_rhs_preserves_conjugate_symmetry(params_linear):
    p = params_linear
    key = jax.random.PRNGKey(0)
    G = jax.random.normal(key, (p["Ns"], p["Nl"], p["Nh"], p["Ny"], p["Nx"], p["Nz"])) \
        + 1j * jax.random.normal(key, (p["Ns"], p["Nl"], p["Nh"], p["Ny"], p["Nx"], p["Nz"]))
    G = G.astype(p["Gk_0"].dtype)

    # Symmetrize input and check output stays symmetric
    G = enforce_conjugate_symmetry_fftshifted(G, p)
    dG, phi = rhs_gk_multispecies(G, p, Nh=p["Nh"], Nl=p["Nl"])

    assert float(_symmetry_error_fftshifted(dG, p)) < 1e-6
    assert float(_symmetry_error_fftshifted(phi[None, None, None, ...], p)) < 1e-6  # embed to reuse helper logic


def test_streaming_is_energy_conserving_linear(params_linear):
    """
    With collisions=off and nonlinear=off, the streaming operator should be skew-adjoint:
      d/dt (||H||^2) = 2 Re <H, dH/dt> ~ 0
    Here we check Re(vdot(H, RHS)) ~ 0 (after building H internally).
    """
    p = params_linear
    key = jax.random.PRNGKey(1)

    G = jax.random.normal(key, (p["Ns"], p["Nl"], p["Nh"], p["Ny"], p["Nx"], p["Nz"])) \
        + 1j * jax.random.normal(key, (p["Ns"], p["Nl"], p["Nh"], p["Ny"], p["Nx"], p["Nz"]))
    G = G.astype(p["Gk_0"].dtype)

    # Force quasineutrality numerator to cancel so phi ~ 0:
    # with q=±1 and equal n0, setting identical g_{m=0} in both species cancels.
    G = G.at[1, :, 0, ...].set(G[0, :, 0, ...])

    G = enforce_conjugate_symmetry_fftshifted(G, p)

    dG, _phi = rhs_gk_multispecies(G, p, Nh=p["Nh"], Nl=p["Nl"])

    phi = solve_phi_quasineutrality_multispecies(G, p)
    assert float(jnp.max(jnp.abs(phi))) < 1e-5

    # In this linear-no-collisions-no-nonlinear config, dG is purely streaming.
    # Check "energy derivative" proxy is ~0: Re <G, dG> is not exactly the invariant,
    # but it should still be very small for a skew-adjoint linear operator in this basis.
    val = jnp.real(jnp.vdot(G, dG))
    assert float(jnp.abs(val)) < 1e-4


def test_collisions_are_dissipative():
    """
    For random H, the conserving Lenard–Bernstein operator should satisfy
      -Re <H, C(H)> >= 0
    """
    Nx, Ny, Nz = 3, 3, 5
    Nl, Nh = 2, 8
    species = [
        dict(name="ion", q=+1.0, T=1.0, n0=1.0, rho=1.0, vth=1.0, nu=0.2, Upar=0.0),
        dict(name="e",   q=-1.0, T=1.0, n0=1.0, rho=1.0, vth=1.0, nu=0.2, Upar=0.0),
    ]
    p = initialize_simulation_parameters_multispecies(
        dict(species=species, enable_nonlinear=False, enforce_reality=False),
        Nx=Nx, Ny=Ny, Nz=Nz, Nl=Nl, Nh=Nh, timesteps=10, dt=1e-2
    )

    key = jax.random.PRNGKey(2)
    H = jax.random.normal(key, (p["Ns"], p["Nl"], p["Nh"], p["Ny"], p["Nx"], p["Nz"])) \
        + 1j * jax.random.normal(key, (p["Ns"], p["Nl"], p["Nh"], p["Ny"], p["Nx"], p["Nz"]))
    H = H.astype(p["Gk_0"].dtype)

    C = collision_lenard_bernstein_conserving_multispecies(H, p)
    diss = -jnp.real(jnp.vdot(H, C))
    assert float(diss) >= -1e-6
