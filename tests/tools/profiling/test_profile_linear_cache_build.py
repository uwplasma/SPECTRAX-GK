from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp

from tools.profile_linear_cache_build import build_low_rank_moment_cache


def test_profile_linear_cache_uses_low_rank_moment_factors() -> None:
    params = SimpleNamespace(
        nu_hermite=0.5,
        nu_laguerre=0.25,
        p_hyper=4,
        p_hyper_l=3,
        p_hyper_m=5,
        p_hyper_lm=2,
    )

    cache = build_low_rank_moment_cache(nl=3, nm=4, params=params, real_dtype=jnp.float32)

    assert cache["lb_lam"].shape == (3, 4)
    assert cache["collision_lam"].shape == (0,)
    assert cache["hyper_ratio"].shape == (3, 4, 1, 1, 1)
    assert cache["sqrt_p"].shape == (1, 1, 4, 1, 1, 1)
    assert cache["mask_const"].dtype == jnp.bool_
