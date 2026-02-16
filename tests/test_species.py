"""Tests for species parameter helpers."""

import numpy as np

from spectraxgk.species import Species, build_linear_params


def test_build_linear_params_shapes():
    """Species helper should create per-species arrays."""
    ion = Species(charge=1.0, mass=1.0, density=1.0, temperature=1.0, tprim=2.0, fprim=1.0)
    ele = Species(charge=-1.0, mass=0.001, density=1.0, temperature=1.0, tprim=2.0, fprim=1.0)
    params = build_linear_params([ion, ele], beta=1.0e-4, fapar=1.0)
    assert params.charge_sign.shape == (2,)
    assert params.vth.shape == (2,)
    assert np.isclose(params.vth[0], 1.0)
    assert np.isclose(params.tz[1], -1.0)
    assert params.beta == 1.0e-4
