from __future__ import annotations

import spectraxgk.linear_cache as legacy_cache
import spectraxgk.linear_krylov as legacy_krylov
import spectraxgk.linear_linked as legacy_linked
import spectraxgk.linear_moments as legacy_moments
import spectraxgk.linear_parallel as legacy_parallel
import spectraxgk.linear_params as legacy_params
from spectraxgk.operators import hermite_streaming
from spectraxgk.operators.linear import hermite_streaming as package_streaming
from spectraxgk.operators.linear.cache import LinearCache, build_linear_cache
from spectraxgk.operators.linear.linked import _build_linked_fft_maps
from spectraxgk.operators.linear.moments import build_H, quasineutrality_phi
from spectraxgk.operators.linear.params import LinearParams, LinearTerms
from spectraxgk.operators.linear.streaming import hermite_streaming as streaming_impl
from spectraxgk.solvers.linear.krylov import KrylovConfig, dominant_eigenpair
from spectraxgk.solvers.linear.parallel import linear_rhs_parallel_cached


def test_linear_operator_package_reexports_streaming_kernel() -> None:
    assert hermite_streaming is package_streaming
    assert package_streaming is streaming_impl


def test_linear_root_facades_reexport_operator_implementations() -> None:
    assert legacy_cache.LinearCache is LinearCache
    assert legacy_cache.build_linear_cache is build_linear_cache
    assert legacy_linked._build_linked_fft_maps is _build_linked_fft_maps
    assert legacy_moments.build_H is build_H
    assert legacy_moments.quasineutrality_phi is quasineutrality_phi
    assert legacy_params.LinearParams is LinearParams
    assert legacy_params.LinearTerms is LinearTerms


def test_linear_root_facades_reexport_solver_implementations() -> None:
    assert legacy_krylov.KrylovConfig is KrylovConfig
    assert legacy_krylov.dominant_eigenpair is dominant_eigenpair
    assert legacy_parallel.linear_rhs_parallel_cached is linear_rhs_parallel_cached
