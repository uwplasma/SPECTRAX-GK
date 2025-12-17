# tests/test_conjugate_symmetry.py
import jax
import jax.numpy as jnp

from spectraxgk._model import enforce_conjugate_symmetry_fftshifted


def test_enforce_conjugate_symmetry_property(tiny_params, tiny_state):
    G = enforce_conjugate_symmetry_fftshifted(tiny_state, tiny_params)

    iy = tiny_params["conj_y"]
    ix = tiny_params["conj_x"]
    iz = tiny_params["conj_z"]

    # Check: G(k) == conj(G(-k))
    G_conj = jnp.conj(G[..., iy][:, :, :, ix][:, :, :, :, iz])
    assert jnp.allclose(G, G_conj, rtol=1e-12, atol=1e-12)


def test_enforce_conjugate_symmetry_makes_ifft_more_real(tiny_params, tiny_state):
    # If the spectrum is conjugate-symmetric, inverse FFT should be (nearly) real.
    from spectraxgk._model import ifft_shifted

    # pick a single component to reduce memory
    A = tiny_state[0, 0]  # (Ny,Nx,Nz)
    A2 = enforce_conjugate_symmetry_fftshifted(A[None, None, ...], tiny_params)[0, 0]
    a_phys = ifft_shifted(A2)
    assert jnp.max(jnp.abs(jnp.imag(a_phys))) < 1e-10
