#!/usr/bin/env python3
"""Generate linear-validation figures and gate reports.

Subcommands:
  collision-table Generate checked high-precision collision coefficient data.
  collision-verification
                  Build the Coulomb algebra/convergence verification panel.
  collision-response
                  Build the drift-kinetic driven-response convergence panel.
  figures         Build Cyclone, ETG, and KBM comparison figures.
  observed-order  Build a convergence observed-order JSON/plot gate.
  kbm-branch      Build a KBM branch-continuity JSON gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from spectraxgk.benchmarking.shared import LinearScanResult

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OBSERVED_CSV = REPO_ROOT / "docs" / "_static" / "cyclone_resolution_subset.csv"
DEFAULT_OBSERVED_JSON = (
    REPO_ROOT / "docs" / "_static" / "cyclone_resolution_observed_order.json"
)
DEFAULT_OBSERVED_PNG = (
    REPO_ROOT / "docs" / "_static" / "cyclone_resolution_observed_order.png"
)
DEFAULT_KBM_CANDIDATES = (
    REPO_ROOT / "docs" / "_static" / "comparison" / "kbm_reference_candidates.csv"
)
DEFAULT_KBM_BRANCH_OUT = REPO_ROOT / "docs" / "_static" / "kbm_branch_gate_summary.json"
DEFAULT_COLLISION_TABLE = (
    REPO_ROOT / "src" / "spectraxgk" / "data" / "advanced_collision_six_moment.npy"
)
DEFAULT_COLLISION_METADATA = DEFAULT_COLLISION_TABLE.with_suffix(".json")
DEFAULT_COLLISION_VERIFICATION_JSON = (
    REPO_ROOT / "docs" / "_static" / "collision_operator_verification.json"
)
DEFAULT_COLLISION_VERIFICATION_PNG = (
    REPO_ROOT / "docs" / "_static" / "collision_operator_verification.png"
)
DEFAULT_COLLISION_RESPONSE_JSON = (
    REPO_ROOT / "docs" / "_static" / "collision_response_convergence.json"
)
DEFAULT_COLLISION_RESPONSE_CSV = DEFAULT_COLLISION_RESPONSE_JSON.with_suffix(".csv")
DEFAULT_COLLISION_RESPONSE_PNG = DEFAULT_COLLISION_RESPONSE_JSON.with_suffix(".png")


def _coulomb_e_mp(order: int, chi: Any, mp: Any) -> Any:
    half = mp.mpf("0.5")
    return mp.gamma(order + half) / (mp.gamma(half) * (1 + chi * chi) ** (order + half))


def _coulomb_E_mp(order: int, chi: Any, mp: Any) -> Any:
    return (
        chi
        / 2
        * sum(
            mp.factorial(order) / mp.factorial(index) * _coulomb_e_mp(index, chi, mp)
            for index in range(order + 1)
        )
    )


def _cached_coulomb_integrals_mp(
    chi: Any, mp: Any
) -> tuple[Callable[[int], Any], Callable[[int], Any]]:
    """Return call-local Appendix-A integrals with the exact E-order recurrence."""

    @cache
    def coulomb_e(order: int) -> Any:
        return _coulomb_e_mp(order, chi, mp)

    @cache
    def coulomb_E(order: int) -> Any:
        if order == 0:
            return chi * coulomb_e(0) / 2
        return order * coulomb_E(order - 1) + chi * coulomb_e(order) / 2

    return coulomb_e, coulomb_E


def coulomb_speed_integrals(
    max_order: int,
    chi: float,
    *,
    digits: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Return the Appendix-A Coulomb speed integrals ``e_k`` and ``E_k``.

    ``chi`` is the thermal-speed ratio.  The recurrence-free multiprecision
    formulas implement Frei et al. (2021), equations (A8a)--(A8b), and are
    intended for offline coefficient generation rather than runtime kernels.
    """

    if max_order < 0:
        raise ValueError("max_order must be >= 0")
    if chi <= 0.0 or not math.isfinite(chi):
        raise ValueError("chi must be finite and > 0")
    if digits < 16:
        raise ValueError("digits must be >= 16")

    import mpmath as mp

    with mp.workdps(digits):
        chi_mp = mp.mpf(chi)
        e_values = [_coulomb_e_mp(order, chi_mp, mp) for order in range(max_order + 1)]
        E_values = [_coulomb_E_mp(order, chi_mp, mp) for order in range(max_order + 1)]
    return (
        np.asarray([float(value) for value in e_values]),
        np.asarray([float(value) for value in E_values]),
    )


def coulomb_speed_moments(
    spherical_order: int,
    spherical_radial_order: int,
    speed_power_order: int,
    mass_ratio: float,
    temperature_ratio: float,
    *,
    collision_frequency: float = 1.0,
    digits: int = 80,
) -> tuple[float, float]:
    r"""Return velocity-integrated Coulomb test and field speed functions.

    The result is ``(nu_T, nu_F)`` from equations (A5) and (A13) of Frei et
    al. (2021) for ``sigma=mass_ratio`` and ``tau=temperature_ratio``.
    """

    if any(
        order < 0
        for order in (spherical_order, spherical_radial_order, speed_power_order)
    ):
        raise ValueError("basis orders must be >= 0")
    if mass_ratio <= 0.0 or not math.isfinite(mass_ratio):
        raise ValueError("mass_ratio must be finite and > 0")
    if temperature_ratio <= 0.0 or not math.isfinite(temperature_ratio):
        raise ValueError("temperature_ratio must be finite and > 0")
    if collision_frequency < 0.0 or not math.isfinite(collision_frequency):
        raise ValueError("collision_frequency must be finite and >= 0")
    if digits < 16:
        raise ValueError("digits must be >= 16")

    import mpmath as mp

    with mp.workdps(digits):
        test_moment, field_moment = _coulomb_speed_moments_mp(
            spherical_order,
            spherical_radial_order,
            speed_power_order,
            mass_ratio,
            temperature_ratio,
            collision_frequency,
            mp,
        )
    return float(test_moment), float(field_moment)


def _coulomb_speed_moments_mp(
    spherical_order: int,
    spherical_radial_order: int,
    speed_power_order: int,
    mass_ratio: Any,
    temperature_ratio: Any,
    collision_frequency: Any,
    mp: Any,
    *,
    coulomb_e: Callable[[int], Any] | None = None,
    coulomb_E: Callable[[int], Any] | None = None,
) -> tuple[Any, Any]:
    p = spherical_order
    j = spherical_radial_order
    d = speed_power_order
    sigma = mp.mpf(mass_ratio)
    tau = mp.mpf(temperature_ratio)
    nu = mp.mpf(collision_frequency)
    chi = mp.sqrt(tau / sigma)
    e_integral = (
        (lambda order: _coulomb_e_mp(order, chi, mp))
        if coulomb_e is None
        else coulomb_e
    )
    E_integral = (
        (lambda order: _coulomb_E_mp(order, chi, mp))
        if coulomb_E is None
        else coulomb_E
    )
    inverse_sqrt_pi = 1 / mp.sqrt(mp.pi)
    test_moment = mp.mpf(0)
    field_moment = mp.mpf(0)

    for monomial_order in range(j + 1):
        laguerre_coefficient = _associated_laguerre_monomial_coefficient_mp(
            j,
            p,
            monomial_order,
            mp,
        )
        common_polynomial = (
            4 * monomial_order**2
            + 4 * monomial_order * (p - 1)
            + mp.mpf("1.5") * p * (p - 1)
        )
        test_erf_coefficients = (
            -4 * sigma * (tau - 1) / tau,
            4 * sigma / tau * monomial_order * (tau - 2)
            - p * (p - 2 * sigma + 4 * sigma / tau + 1),
            sigma / tau * common_polynomial,
        )
        test_gaussian_coefficients = (
            4 * (tau - 1) * (sigma + tau) / tau,
            -2 * (p + 2 * monomial_order) * (sigma / tau * (tau - 2) - 1),
            -sigma / tau * common_polynomial,
        )
        for summation_order in range(3):
            erf_coefficient = test_erf_coefficients[summation_order]
            gaussian_coefficient = test_gaussian_coefficients[summation_order]
            if erf_coefficient != 0:
                integral_order = d + monomial_order + p - summation_order
                test_moment += (
                    laguerre_coefficient
                    * erf_coefficient
                    * 4
                    * nu
                    * inverse_sqrt_pi
                    * E_integral(integral_order)
                )
            if gaussian_coefficient != 0:
                integral_order = d + monomial_order + p - summation_order + 1
                test_moment += (
                    laguerre_coefficient
                    * gaussian_coefficient
                    * 4
                    * chi
                    * nu
                    * inverse_sqrt_pi
                    * e_integral(integral_order)
                )

        field_prime_coefficients = (
            4 * tau**2 / sigma,
            -8 * (p * p + p - 1) / ((2 * p - 1) * (2 * p + 3)),
        )
        field_plus_coefficients = (
            -8 * p * (p - 1) * (monomial_order + 1) / ((2 * p + 1) * (2 * p - 1))
            - 8 * tau * (1 + p - sigma * p) / (sigma * (2 * p + 1)),
            8 * (p + 1) * (p + 2) / ((2 * p + 1) * (2 * p + 3)),
        )
        field_minus_coefficients = (
            4
            * (p + 1)
            * (p + 2)
            * (2 * p + 2 * monomial_order + 3)
            / ((2 * p + 1) * (2 * p + 3))
            + 8 * chi**2 * (p - sigma * p - sigma) / (2 * p + 1),
            -8 * p * (p - 1) / ((2 * p + 1) * (2 * p - 1)),
        )
        for summation_order in range(2):
            beta_e = (
                4
                * nu
                * inverse_sqrt_pi
                * chi ** (p + 2 * monomial_order + 2 * summation_order - 1)
                * e_integral(p + 1 + d + monomial_order + summation_order)
            )
            beta_plus = (
                4
                * nu
                * inverse_sqrt_pi
                * chi ** (p + 2 * summation_order - 1)
                * sum(
                    chi ** (2 * inner_order)
                    * mp.factorial(monomial_order)
                    / mp.factorial(inner_order)
                    * e_integral(p + summation_order + 1 + d + inner_order)
                    / 2
                    for inner_order in range(monomial_order + 1)
                )
            )
            gamma_factor = mp.gamma(p + monomial_order + mp.mpf("1.5"))
            beta_minus = (
                4
                * nu
                * inverse_sqrt_pi
                * chi ** (2 * summation_order - p - 2)
                * (
                    gamma_factor
                    / mp.gamma(mp.mpf("1.5"))
                    * E_integral(d + summation_order)
                    / 2
                    - sum(
                        chi ** (2 * inner_order + 1)
                        * gamma_factor
                        / mp.gamma(inner_order + mp.mpf("1.5"))
                        * e_integral(d + summation_order + inner_order + 1)
                        / 2
                        for inner_order in range(p + monomial_order + 1)
                    )
                )
            )
            field_moment += laguerre_coefficient * (
                field_prime_coefficients[summation_order] * beta_e
                + field_plus_coefficients[summation_order] * beta_plus
                + field_minus_coefficients[summation_order] * beta_minus
            )
    return test_moment, field_moment


def associated_laguerre_monomial_coefficients(
    polynomial_order: int,
    tensor_order: int,
    *,
    digits: int = 80,
) -> np.ndarray:
    r"""Return monomial coefficients of ``L_j^(p+1/2)``.

    This is equation (3.10) of Frei et al. (2021), evaluated with
    multiprecision arithmetic for the offline collision-table generator.
    Coefficient index ``ell`` multiplies ``x**ell``.
    """

    if polynomial_order < 0:
        raise ValueError("polynomial_order must be >= 0")
    if tensor_order < 0:
        raise ValueError("tensor_order must be >= 0")
    if digits < 16:
        raise ValueError("digits must be >= 16")

    import mpmath as mp

    with mp.workdps(digits):
        coefficients = [
            _associated_laguerre_monomial_coefficient_mp(
                polynomial_order,
                tensor_order,
                ell,
                mp,
            )
            for ell in range(polynomial_order + 1)
        ]
    return np.asarray([float(value) for value in coefficients])


def _associated_laguerre_monomial_coefficient_mp(
    polynomial_order: int,
    tensor_order: Any,
    monomial_order: int,
    mp: Any,
) -> Any:
    half = mp.mpf("0.5")
    return (
        (-1) ** monomial_order
        * mp.gamma(tensor_order + polynomial_order + 1 + half)
        / (
            mp.factorial(polynomial_order - monomial_order)
            * mp.gamma(monomial_order + tensor_order + 1 + half)
            * mp.factorial(monomial_order)
        )
    )


def _legendre_monomial_coefficient_mp(order: int, power: int, mp: Any) -> Any:
    if power < 0 or power > order or (order - power) % 2:
        return mp.mpf(0)
    half_difference = (order - power) // 2
    return (
        (-1) ** half_difference
        * mp.factorial(order + power)
        / (
            mp.power(2, order)
            * mp.factorial(half_difference)
            * mp.factorial((order + power) // 2)
            * mp.factorial(power)
        )
    )


def _nonnegative_binomial(n: int, k: int, mp: Any) -> Any:
    if k < 0 or k > n:
        return mp.mpf(0)
    # Every caller supplies integer polynomial indices. ``math.comb`` is exact
    # and avoids mpmath's gamma-based generalized-binomial path in the deepest
    # basis-transform loop.
    return math.comb(n, k)


def _legendre_to_hermite_laguerre_mp(
    legendre_order: int,
    radial_order: int,
    hermite_order: int,
    laguerre_order: int,
    mp: Any,
) -> Any:
    total = mp.mpf(0)
    half = mp.mpf("0.5")
    for q in range(legendre_order // 2 + 1):
        for v in range(hermite_order // 2 + 1):
            for i in range(radial_order + 1):
                for r in range(q + 1):
                    for s in range(min(laguerre_order, i) + 1):
                        for m in range(radial_order - i + 1):
                            combinatoric = (
                                _nonnegative_binomial(legendre_order, q, mp)
                                * _nonnegative_binomial(
                                    2 * (legendre_order - q),
                                    legendre_order,
                                    mp,
                                )
                                * _nonnegative_binomial(q, r, mp)
                                * _nonnegative_binomial(r, laguerre_order - s, mp)
                                * _nonnegative_binomial(r, i - s, mp)
                                * _nonnegative_binomial(s + r, s, mp)
                                * mp.factorial(r)
                            )
                            if combinatoric == 0:
                                continue
                            exponent = (
                                mp.mpf(3 * legendre_order + hermite_order) / 2
                                + m
                                + v
                                - r
                            )
                            numerator = mp.gamma(
                                radial_order - i + legendre_order + half
                            ) * mp.fac2(
                                legendre_order + hermite_order + 2 * (m - r - v) - 1
                            )
                            denominator = (
                                mp.factorial(hermite_order - 2 * v)
                                * mp.factorial(radial_order - i - m)
                                * mp.gamma(legendre_order + m + half)
                                * mp.factorial(v)
                                * mp.factorial(m)
                                * mp.power(2, exponent)
                            )
                            total += (
                                (-1) ** (q + i + laguerre_order + v + m)
                                * combinatoric
                                * numerator
                                / denominator
                            )
    return total


def legendre_to_hermite_laguerre_coefficient(
    legendre_order: int,
    radial_order: int,
    hermite_order: int,
    laguerre_order: int,
    *,
    digits: int = 80,
) -> float:
    r"""Return the isotropic Legendre-to-Hermite--Laguerre coefficient.

    This implements equation (A4) of Jorge, Ricci & Loureiro (2017).  The
    coefficient maps ``c**l P_l(xi) L_k^(l+1/2)(c**2)`` onto
    ``H_p(s_parallel) L_j(s_perp**2)``.  Coefficients on different total-degree
    shells are exactly zero.
    """

    indices = (legendre_order, radial_order, hermite_order, laguerre_order)
    if any(index < 0 for index in indices):
        raise ValueError("basis orders must be >= 0")
    if digits < 16:
        raise ValueError("digits must be >= 16")
    if hermite_order + 2 * laguerre_order != legendre_order + 2 * radial_order:
        return 0.0

    import mpmath as mp

    with mp.workdps(digits):
        total = _legendre_to_hermite_laguerre_mp(
            legendre_order,
            radial_order,
            hermite_order,
            laguerre_order,
            mp,
        )
    return float(total)


def hermite_laguerre_to_legendre_coefficient(
    hermite_order: int,
    laguerre_order: int,
    legendre_order: int,
    radial_order: int,
    *,
    digits: int = 80,
) -> float:
    r"""Return the inverse isotropic basis-transform coefficient.

    This applies equation (A3) of Jorge, Ricci & Loureiro (2017) to the
    multiprecision forward coefficient.
    """

    indices = (legendre_order, radial_order, hermite_order, laguerre_order)
    if any(index < 0 for index in indices):
        raise ValueError("basis orders must be >= 0")
    if digits < 16:
        raise ValueError("digits must be >= 16")
    if hermite_order + 2 * laguerre_order != legendre_order + 2 * radial_order:
        return 0.0

    import mpmath as mp

    with mp.workdps(digits):
        forward = _legendre_to_hermite_laguerre_mp(
            legendre_order,
            radial_order,
            hermite_order,
            laguerre_order,
            mp,
        )
        factor = (
            mp.sqrt(mp.pi)
            * mp.power(2, hermite_order)
            * mp.factorial(hermite_order)
            * (legendre_order + mp.mpf("0.5"))
            * mp.factorial(radial_order)
            / mp.gamma(radial_order + legendre_order + mp.mpf("1.5"))
        )
        inverse = factor * forward
    return float(inverse)


def _associated_legendre_to_hermite_laguerre_mp(
    legendre_order: int,
    radial_order: int,
    bessel_order: int,
    hermite_order: int,
    laguerre_order: int,
    mp: Any,
    *,
    base_transform: Callable[[int, int, int, int], Any] | None = None,
    associated_laguerre: Callable[[int, Any, int], Any] | None = None,
    legendre_monomial: Callable[[int, int], Any] | None = None,
) -> Any:
    left_degree = legendre_order + 2 * radial_order - bessel_order
    right_degree = hermite_order + 2 * laguerre_order
    if right_degree > left_degree or (left_degree - right_degree) % 2:
        return mp.mpf(0)

    half = mp.mpf("0.5")
    total = mp.mpf(0)
    if base_transform is None:

        def base_transform(p: int, j: int, g: int, s: int) -> Any:
            return _legendre_to_hermite_laguerre_mp(p, j, g, s, mp)

    if associated_laguerre is None:

        def associated_laguerre(n: int, alpha: Any, power: int) -> Any:
            return _associated_laguerre_monomial_coefficient_mp(n, alpha, power, mp)

    if legendre_monomial is None:

        def legendre_monomial(n: int, power: int) -> Any:
            return _legendre_monomial_coefficient_mp(n, power, mp)

    for auxiliary_radial_order in range(right_degree // 2 + 1):
        auxiliary_legendre_order = right_degree - 2 * auxiliary_radial_order
        transform = base_transform(
            auxiliary_legendre_order,
            auxiliary_radial_order,
            hermite_order,
            laguerre_order,
        )
        prefactor = (
            transform
            * (auxiliary_legendre_order + half)
            * mp.factorial(auxiliary_radial_order)
            / mp.gamma(
                auxiliary_legendre_order + auxiliary_radial_order + mp.mpf("1.5")
            )
        )
        inner = mp.mpf(0)
        for left_monomial in range(radial_order + 1):
            left_laguerre = associated_laguerre(
                radial_order,
                legendre_order,
                left_monomial,
            )
            for auxiliary_monomial in range(auxiliary_radial_order + 1):
                auxiliary_laguerre = associated_laguerre(
                    auxiliary_radial_order,
                    auxiliary_legendre_order,
                    auxiliary_monomial,
                )
                for left_power in range(bessel_order, legendre_order + 1):
                    left_legendre = legendre_monomial(
                        legendre_order,
                        left_power,
                    )
                    if left_legendre == 0:
                        continue
                    derivative_factor = mp.factorial(left_power) / mp.factorial(
                        left_power - bessel_order
                    )
                    for auxiliary_power in range(auxiliary_legendre_order + 1):
                        auxiliary_legendre = legendre_monomial(
                            auxiliary_legendre_order,
                            auxiliary_power,
                        )
                        parity_factor = 1 + (-1) ** (
                            left_power + auxiliary_power - bessel_order
                        )
                        if auxiliary_legendre == 0 or parity_factor == 0:
                            continue
                        velocity_power_integral = mp.gamma(
                            left_monomial
                            + auxiliary_monomial
                            + mp.mpf(
                                legendre_order
                                + auxiliary_legendre_order
                                - bessel_order
                                + 3
                            )
                            / 2
                        )
                        inner += (
                            left_laguerre
                            * auxiliary_laguerre
                            * left_legendre
                            * auxiliary_legendre
                            * derivative_factor
                            * parity_factor
                            * velocity_power_integral
                            / (2 * (left_power + auxiliary_power + 1 - bessel_order))
                        )
        total += prefactor * inner

    # Equation (B5) is half-normalized at m=0 and has the opposite odd-m
    # phase under scipy's associated-Legendre convention.  This factor is
    # fixed by the m=0 endpoint and independent velocity-space projection.
    return 2 * (-1) ** bessel_order * total


def associated_legendre_to_hermite_laguerre_coefficient(
    legendre_order: int,
    radial_order: int,
    bessel_order: int,
    hermite_order: int,
    laguerre_order: int,
    *,
    digits: int = 80,
) -> float:
    r"""Return a finite-Bessel-order collision-basis coefficient.

    This is the convention-corrected forward transform from equation (B5) of
    Jorge, Frei & Ricci (2019).  Unlike the isotropic transform, finite
    ``bessel_order`` couples every lower reduced-degree shell of the same
    parity.
    """

    indices = (
        legendre_order,
        radial_order,
        bessel_order,
        hermite_order,
        laguerre_order,
    )
    if any(index < 0 for index in indices):
        raise ValueError("basis orders must be >= 0")
    if bessel_order > legendre_order:
        raise ValueError("bessel_order must be <= legendre_order")
    if digits < 16:
        raise ValueError("digits must be >= 16")

    import mpmath as mp

    with mp.workdps(digits):
        coefficient = _associated_legendre_to_hermite_laguerre_mp(
            legendre_order,
            radial_order,
            bessel_order,
            hermite_order,
            laguerre_order,
            mp,
        )
    return float(coefficient)


def associated_basis_transform_matrices(
    bessel_order: int,
    maximum_reduced_degree: int,
    *,
    digits: int = 80,
) -> tuple[
    tuple[tuple[int, int], ...],
    tuple[tuple[int, int], ...],
    np.ndarray,
    np.ndarray,
]:
    r"""Generate and invert one finite-``m`` parity block.

    The block contains all nonnegative reduced-degree shells with the parity
    of ``maximum_reduced_degree``.  Rows are Hermite--Laguerre ``(p, j)``
    orders, columns are associated Legendre--Laguerre ``(l, k)`` orders, and
    both order tuples are sorted by increasing reduced degree.  The inverse is
    formed from the complete multiprecision block rather than equation (B6),
    whose literal finite-``m`` normalization fails the inverse identity.
    """

    if bessel_order < 0:
        raise ValueError("bessel_order must be >= 0")
    if maximum_reduced_degree < 0:
        raise ValueError("maximum_reduced_degree must be >= 0")
    if digits < 16:
        raise ValueError("digits must be >= 16")

    parity = maximum_reduced_degree % 2
    right_orders = tuple(
        (degree - 2 * radial_order, radial_order)
        for degree in range(parity, maximum_reduced_degree + 1, 2)
        for radial_order in range(degree // 2 + 1)
    )
    left_orders = tuple(
        (bessel_order + degree - 2 * radial_order, radial_order)
        for degree in range(parity, maximum_reduced_degree + 1, 2)
        for radial_order in range(degree // 2 + 1)
    )

    import mpmath as mp

    with mp.workdps(digits):
        forward_mp = mp.matrix(
            [
                [
                    _associated_legendre_to_hermite_laguerre_mp(
                        legendre_order,
                        radial_order,
                        bessel_order,
                        hermite_order,
                        laguerre_order,
                        mp,
                    )
                    for legendre_order, radial_order in left_orders
                ]
                for hermite_order, laguerre_order in right_orders
            ]
        )
        inverse_mp = forward_mp**-1
        forward = np.asarray(forward_mp.tolist(), dtype=np.float64)
        inverse = np.asarray(inverse_mp.tolist(), dtype=np.float64)
    return right_orders, left_orders, forward, inverse


def _laguerre_product_expansion_coefficient_mp(
    associated_order: int,
    associated_polynomial_order: int,
    laguerre_order: int,
    output_order: int,
    radial_power: int,
    mp: Any,
) -> Any:
    if output_order > associated_polynomial_order + laguerre_order + radial_power:
        return mp.mpf(0)

    half = mp.mpf("0.5")
    total = mp.mpf(0)
    for associated_power in range(associated_polynomial_order + 1):
        associated_coefficient = _associated_laguerre_monomial_coefficient_mp(
            associated_polynomial_order,
            associated_order - half,
            associated_power,
            mp,
        )
        for laguerre_power in range(laguerre_order + 1):
            laguerre_coefficient = _associated_laguerre_monomial_coefficient_mp(
                laguerre_order,
                -half,
                laguerre_power,
                mp,
            )
            for output_power in range(output_order + 1):
                output_coefficient = _associated_laguerre_monomial_coefficient_mp(
                    output_order,
                    -half,
                    output_power,
                    mp,
                )
                total += (
                    associated_coefficient
                    * laguerre_coefficient
                    * output_coefficient
                    * mp.factorial(
                        associated_power + laguerre_power + output_power + radial_power
                    )
                )
    return total


def laguerre_product_expansion_coefficient(
    associated_order: int,
    associated_polynomial_order: int,
    laguerre_order: int,
    output_order: int,
    *,
    radial_power: int = 0,
    digits: int = 80,
) -> float:
    r"""Expand ``x**r L_n^m(x) L_k(x)`` in ordinary Laguerre modes.

    ``radial_power=associated_order`` implements equations (3.36)--(3.37) of
    Frei et al. (2021); ``radial_power=0`` implements equations
    (3.44)--(3.45).  The returned coefficient multiplies ``L_output_order``.
    """

    indices = (
        associated_order,
        associated_polynomial_order,
        laguerre_order,
        output_order,
        radial_power,
    )
    if any(index < 0 for index in indices):
        raise ValueError("Laguerre orders and radial_power must be >= 0")
    if digits < 16:
        raise ValueError("digits must be >= 16")

    import mpmath as mp

    with mp.workdps(digits):
        coefficient = _laguerre_product_expansion_coefficient_mp(
            associated_order,
            associated_polynomial_order,
            laguerre_order,
            output_order,
            radial_power,
            mp,
        )
    return float(coefficient)


def _hermite_laguerre_to_associated_legendre_mp(
    hermite_order: int,
    laguerre_order: int,
    spherical_order: int,
    spherical_radial_order: int,
    bessel_order: int,
    mp: Any,
    *,
    associated_transform: Callable[[int, int, int, int, int], Any] | None = None,
    laguerre_product: Callable[[int, int, int, int, int], Any] | None = None,
) -> Any:
    if hermite_order > spherical_order + bessel_order + 2 * spherical_radial_order:
        return mp.mpf(0)

    prefactor = (
        mp.sqrt(mp.pi)
        * mp.power(2, hermite_order)
        * mp.factorial(hermite_order)
        * mp.factorial(spherical_radial_order)
        * (spherical_order + mp.mpf("0.5"))
        / mp.gamma(spherical_radial_order + spherical_order + mp.mpf("1.5"))
        * mp.factorial(spherical_order - bessel_order)
        / mp.factorial(spherical_order + bessel_order)
    )
    maximum_auxiliary_order = min(
        spherical_radial_order + (spherical_order + bessel_order) // 2,
        bessel_order + laguerre_order,
    )
    if associated_transform is None:

        def associated_transform(p: int, j: int, m: int, g: int, s: int) -> Any:
            return _associated_legendre_to_hermite_laguerre_mp(p, j, m, g, s, mp)

    if laguerre_product is None:

        def laguerre_product(m: int, n: int, k: int, output: int, radial: int) -> Any:
            return _laguerre_product_expansion_coefficient_mp(
                m, n, k, output, radial, mp
            )

    contraction = sum(
        associated_transform(
            spherical_order,
            spherical_radial_order,
            bessel_order,
            hermite_order,
            auxiliary_order,
        )
        * laguerre_product(
            bessel_order,
            0,
            laguerre_order,
            auxiliary_order,
            bessel_order,
        )
        for auxiliary_order in range(maximum_auxiliary_order + 1)
    )
    return prefactor * contraction


def hermite_laguerre_to_associated_legendre_coefficient(
    hermite_order: int,
    laguerre_order: int,
    spherical_order: int,
    spherical_radial_order: int,
    bessel_order: int,
    *,
    digits: int = 80,
) -> float:
    r"""Return the finite-``m`` inverse collision-basis coefficient.

    This implements equation (3.33) of Frei et al. (2021), including its
    weighted Laguerre-product contraction.  It maps ``H_l L_k x**(m/2)`` onto
    ``s**p P_p^m L_j^(p+1/2)``.
    """

    indices = (
        hermite_order,
        laguerre_order,
        spherical_order,
        spherical_radial_order,
        bessel_order,
    )
    if any(index < 0 for index in indices):
        raise ValueError("basis orders must be >= 0")
    if bessel_order > spherical_order:
        raise ValueError("bessel_order must be <= spherical_order")
    if digits < 16:
        raise ValueError("digits must be >= 16")

    import mpmath as mp

    with mp.workdps(digits):
        coefficient = _hermite_laguerre_to_associated_legendre_mp(
            hermite_order,
            laguerre_order,
            spherical_order,
            spherical_radial_order,
            bessel_order,
            mp,
        )
    return float(coefficient)


def gyroaveraged_spherical_moment_coefficient(
    spherical_order: int,
    spherical_radial_order: int,
    bessel_order: int,
    hermite_order: int,
    laguerre_order: int,
    kperp_rho: float,
    *,
    maximum_bessel_laguerre_order: int = 24,
    digits: int = 80,
) -> float:
    r"""Map one gyro-moment into a finite-``m`` spherical particle moment.

    This evaluates one ``N^(g,s)`` coefficient in equation (3.35) of Frei et
    al. (2021).  The Bessel--Laguerre sum is explicitly truncated at
    ``maximum_bessel_laguerre_order``; convergence must be checked when tables
    are generated.
    """

    indices = (
        spherical_order,
        spherical_radial_order,
        bessel_order,
        hermite_order,
        laguerre_order,
        maximum_bessel_laguerre_order,
    )
    if any(index < 0 for index in indices):
        raise ValueError("basis and truncation orders must be >= 0")
    if bessel_order > spherical_order:
        raise ValueError("bessel_order must be <= spherical_order")
    if not math.isfinite(kperp_rho) or kperp_rho < 0.0:
        raise ValueError("kperp_rho must be finite and >= 0")
    if digits < 16:
        raise ValueError("digits must be >= 16")

    import mpmath as mp

    with mp.workdps(digits):
        coefficient = _gyroaveraged_spherical_moment_coefficient_mp(
            spherical_order,
            spherical_radial_order,
            bessel_order,
            hermite_order,
            laguerre_order,
            kperp_rho,
            maximum_bessel_laguerre_order,
            mp,
        )
    return float(coefficient)


def _gyroaveraged_spherical_moment_coefficient_mp(
    spherical_order: int,
    spherical_radial_order: int,
    bessel_order: int,
    hermite_order: int,
    laguerre_order: int,
    kperp_rho: Any,
    maximum_bessel_laguerre_order: int,
    mp: Any,
    *,
    associated_transform: Callable[[int, int, int, int, int], Any] | None = None,
    laguerre_product: Callable[[int, int, int, int, int], Any] | None = None,
) -> Any:
    b = mp.mpf(kperp_rho)
    half_b = b / 2
    argument = half_b * half_b
    coefficient = mp.mpf(0)
    maximum_auxiliary_laguerre = (
        spherical_radial_order + (spherical_order + bessel_order) // 2
    )
    if associated_transform is None:

        def associated_transform(p: int, j: int, m: int, g: int, s: int) -> Any:
            return _associated_legendre_to_hermite_laguerre_mp(p, j, m, g, s, mp)

    if laguerre_product is None:

        def laguerre_product(m: int, n: int, k: int, output: int, radial: int) -> Any:
            return _laguerre_product_expansion_coefficient_mp(
                m, n, k, output, radial, mp
            )

    for auxiliary_laguerre_order in range(maximum_auxiliary_laguerre + 1):
        transform = associated_transform(
            spherical_order,
            spherical_radial_order,
            bessel_order,
            hermite_order,
            auxiliary_laguerre_order,
        )
        if transform == 0:
            continue
        for bessel_laguerre_order in range(maximum_bessel_laguerre_order + 1):
            product = laguerre_product(
                bessel_order,
                bessel_laguerre_order,
                auxiliary_laguerre_order,
                laguerre_order,
                bessel_order,
            )
            if product == 0:
                continue
            kernel = (
                mp.exp(-argument)
                * argument**bessel_laguerre_order
                / mp.factorial(bessel_laguerre_order)
            )
            coefficient += (
                transform
                * product
                * mp.factorial(bessel_laguerre_order)
                * kernel
                * half_b**bessel_order
                / mp.factorial(bessel_laguerre_order + bessel_order)
            )
    coefficient *= mp.sqrt(mp.power(2, hermite_order) * mp.factorial(hermite_order))
    return coefficient


def _gyroaveraged_polarization_coefficient_mp(
    spherical_order: int,
    spherical_radial_order: int,
    bessel_order: int,
    kperp_rho: Any,
    maximum_bessel_laguerre_order: int,
    mp: Any,
) -> Any:
    b = mp.mpf(kperp_rho)
    half_b = b / 2
    argument = half_b * half_b
    total = mp.mpf(0)
    maximum_auxiliary_laguerre = (
        spherical_radial_order + (spherical_order + bessel_order) // 2
    )
    kernels = [
        mp.exp(-argument) * argument**order / mp.factorial(order)
        for order in range(
            2 * maximum_bessel_laguerre_order
            + bessel_order
            + maximum_auxiliary_laguerre
            + 1
        )
    ]
    for auxiliary_laguerre_order in range(maximum_auxiliary_laguerre + 1):
        transform = _associated_legendre_to_hermite_laguerre_mp(
            spherical_order,
            spherical_radial_order,
            bessel_order,
            0,
            auxiliary_laguerre_order,
            mp,
        )
        if transform == 0:
            continue
        for bessel_laguerre_order in range(maximum_bessel_laguerre_order + 1):
            leading = (
                transform
                * half_b**bessel_order
                * mp.factorial(bessel_laguerre_order)
                * kernels[bessel_laguerre_order]
                / mp.factorial(bessel_laguerre_order + bessel_order)
            )
            if leading == 0:
                continue
            for output_order in range(
                bessel_laguerre_order + bessel_order + auxiliary_laguerre_order + 1
            ):
                product = _laguerre_product_expansion_coefficient_mp(
                    bessel_order,
                    bessel_laguerre_order,
                    auxiliary_laguerre_order,
                    output_order,
                    bessel_order,
                    mp,
                )
                total += leading * kernels[output_order] * product
    return total


def gyroaveraged_polarization_coefficient(
    spherical_order: int,
    spherical_radial_order: int,
    bessel_order: int,
    kperp_rho: float,
    *,
    maximum_bessel_laguerre_order: int = 24,
    digits: int = 80,
) -> float:
    r"""Return equation (3.41)'s finite-``b`` polarization coefficient."""

    if any(
        order < 0
        for order in (
            spherical_order,
            spherical_radial_order,
            bessel_order,
            maximum_bessel_laguerre_order,
        )
    ):
        raise ValueError("basis and truncation orders must be >= 0")
    if bessel_order > spherical_order:
        raise ValueError("bessel_order must be <= spherical_order")
    if not math.isfinite(kperp_rho) or kperp_rho < 0.0:
        raise ValueError("kperp_rho must be finite and >= 0")
    if digits < 16:
        raise ValueError("digits must be >= 16")

    import mpmath as mp

    with mp.workdps(digits):
        coefficient = _gyroaveraged_polarization_coefficient_mp(
            spherical_order,
            spherical_radial_order,
            bessel_order,
            kperp_rho,
            maximum_bessel_laguerre_order,
            mp,
        )
    return float(coefficient)


def _coulomb_coefficient_functions(
    mp: Any,
    sigma: Any,
    tau: Any,
) -> tuple[Callable[..., Any], ...]:
    """Cache the kperp-independent basis algebra for one species pair."""

    coulomb_e, coulomb_E = _cached_coulomb_integrals_mp(mp.sqrt(tau / sigma), mp)

    @cache
    def base_transform(
        legendre_order: int,
        radial_order: int,
        hermite_order: int,
        laguerre_order: int,
    ) -> Any:
        return _legendre_to_hermite_laguerre_mp(
            legendre_order,
            radial_order,
            hermite_order,
            laguerre_order,
            mp,
        )

    @cache
    def associated_laguerre(
        polynomial_order: int,
        tensor_order: Any,
        monomial_order: int,
    ) -> Any:
        return _associated_laguerre_monomial_coefficient_mp(
            polynomial_order,
            tensor_order,
            monomial_order,
            mp,
        )

    @cache
    def legendre_monomial(order: int, power: int) -> Any:
        return _legendre_monomial_coefficient_mp(order, power, mp)

    @cache
    def inverse_associated(
        spherical_order: int,
        spherical_radial_order: int,
        bessel_order: int,
        hermite_order: int,
        laguerre_order: int,
    ) -> Any:
        return _associated_legendre_to_hermite_laguerre_mp(
            spherical_order,
            spherical_radial_order,
            bessel_order,
            hermite_order,
            laguerre_order,
            mp,
            base_transform=base_transform,
            associated_laguerre=associated_laguerre,
            legendre_monomial=legendre_monomial,
        )

    @cache
    def inverse_product(
        associated_order: int,
        associated_polynomial_order: int,
        laguerre_order: int,
        output_order: int,
        radial_power: int,
    ) -> Any:
        return _laguerre_product_expansion_coefficient_mp(
            associated_order,
            associated_polynomial_order,
            laguerre_order,
            output_order,
            radial_power,
            mp,
        )

    @cache
    def speed_moment(p: int, j: int, speed_power: int) -> tuple[Any, Any]:
        return _coulomb_speed_moments_mp(
            p,
            j,
            speed_power,
            sigma,
            tau,
            1,
            mp,
            coulomb_e=coulomb_e,
            coulomb_E=coulomb_E,
        )

    return (
        base_transform,
        associated_laguerre,
        legendre_monomial,
        inverse_associated,
        inverse_product,
        speed_moment,
    )


def coulomb_nonpolarized_moment_matrices(
    maximum_hermite_order: int,
    maximum_laguerre_order: int,
    target_kperp_rho: float,
    mass_ratio: float,
    temperature_ratio: float,
    *,
    source_kperp_rho: float | None = None,
    maximum_spherical_order: int | None = None,
    maximum_spherical_radial_order: int | None = None,
    maximum_bessel_laguerre_order: int = 24,
    digits: int = 80,
    _coefficient_functions: tuple[Callable[..., Any], ...] | None = None,
    _assembly_cache: dict[str, dict[tuple[Any, ...], Any]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Generate finite-``b`` Coulomb test and field moment matrices.

    This contracts equations (3.48)--(3.49) of Frei et al. (2021), excluding
    the electrostatic polarization terms in equation (3.50).  Rows and columns
    use Hermite-major ``p * Nl + j`` ordering and the paper's Laguerre
    convention. ``target_kperp_rho`` enters the outer gyroaverage and the test
    moments; ``source_kperp_rho`` enters the field-particle source moments and
    defaults to the target value for like species. Every finite truncation is
    explicit so convergence can be assessed before a table is promoted.
    """

    if maximum_hermite_order < 0:
        raise ValueError("maximum_hermite_order must be >= 0")
    if maximum_laguerre_order < 0:
        raise ValueError("maximum_laguerre_order must be >= 0")
    if not math.isfinite(target_kperp_rho) or target_kperp_rho < 0.0:
        raise ValueError("target_kperp_rho must be finite and >= 0")
    source_kperp_rho = (
        target_kperp_rho if source_kperp_rho is None else source_kperp_rho
    )
    if not math.isfinite(source_kperp_rho) or source_kperp_rho < 0.0:
        raise ValueError("source_kperp_rho must be finite and >= 0")
    if mass_ratio <= 0.0 or not math.isfinite(mass_ratio):
        raise ValueError("mass_ratio must be finite and > 0")
    if temperature_ratio <= 0.0 or not math.isfinite(temperature_ratio):
        raise ValueError("temperature_ratio must be finite and > 0")
    if maximum_bessel_laguerre_order < 0:
        raise ValueError("maximum_bessel_laguerre_order must be >= 0")
    if digits < 16:
        raise ValueError("digits must be >= 16")

    maximum_degree = maximum_hermite_order + 2 * maximum_laguerre_order
    spherical_limit = (
        maximum_degree if maximum_spherical_order is None else maximum_spherical_order
    )
    radial_limit = (
        maximum_degree // 2
        if maximum_spherical_radial_order is None
        else maximum_spherical_radial_order
    )
    if spherical_limit < 0:
        raise ValueError("maximum_spherical_order must be >= 0")
    if radial_limit < 0:
        raise ValueError("maximum_spherical_radial_order must be >= 0")

    import mpmath as mp

    n_laguerre = maximum_laguerre_order + 1
    n_modes = (maximum_hermite_order + 1) * n_laguerre
    with mp.workdps(digits):
        target_b = mp.mpf(target_kperp_rho)
        source_b = mp.mpf(source_kperp_rho)
        sigma = mp.mpf(mass_ratio)
        tau = mp.mpf(temperature_ratio)
        half_b = target_b / 2
        bessel_argument = half_b * half_b
        drift_kinetic = target_b == 0
        bessel_orders = (0,) if drift_kinetic else range(maximum_bessel_laguerre_order + 1)
        test_matrix = mp.matrix(n_modes, n_modes)
        field_matrix = mp.matrix(n_modes, n_modes)
        assembly_cache = {} if _assembly_cache is None else _assembly_cache
        moment_cache = assembly_cache.setdefault("spherical_moment", {})
        speed_cache = assembly_cache.setdefault("integrated_speed", {})
        product_cache = assembly_cache.setdefault("laguerre_product", {})
        inverse_cache = assembly_cache.setdefault("inverse_transform", {})

        coefficient_functions = (
            _coulomb_coefficient_functions(mp, sigma, tau)
            if _coefficient_functions is None
            else _coefficient_functions
        )
        (
            _base_transform,
            associated_laguerre,
            _legendre_monomial,
            inverse_associated,
            inverse_product,
            speed_moment,
        ) = coefficient_functions

        def spherical_moment(
            p: int,
            j: int,
            m: int,
            g: int,
            s: int,
            *,
            source: bool,
        ) -> Any:
            wavelength = source_b if source else target_b
            key = (float(wavelength), p, j, m, g, s)
            if key not in moment_cache:
                moment_cache[key] = _gyroaveraged_spherical_moment_coefficient_mp(
                    p,
                    j,
                    m,
                    g,
                    s,
                    source_b if source else target_b,
                    maximum_bessel_laguerre_order,
                    mp,
                    associated_transform=inverse_associated,
                    laguerre_product=inverse_product,
                )
            return moment_cache[key]

        def integrated_speed(p: int, j: int, t: int) -> tuple[Any, Any]:
            key = (p, j, t)
            if key not in speed_cache:
                test_speed = mp.mpf(0)
                field_speed = mp.mpf(0)
                for speed_power in range(t + 1):
                    laguerre_coefficient = associated_laguerre(
                        t,
                        p,
                        speed_power,
                    )
                    test_term, field_term = speed_moment(p, j, speed_power)
                    test_speed += laguerre_coefficient * test_term
                    field_speed += laguerre_coefficient * field_term
                speed_cache[key] = (test_speed, field_speed)
            return speed_cache[key]

        def laguerre_product(
            m: int,
            n: int,
            output_laguerre: int,
            product_order: int,
        ) -> Any:
            key = (m, n, output_laguerre, product_order)
            if key not in product_cache:
                product_cache[key] = _laguerre_product_expansion_coefficient_mp(
                    m,
                    n,
                    output_laguerre,
                    product_order,
                    0,
                    mp,
                )
            return product_cache[key]

        def inverse_transform(
            output_hermite: int,
            product_order: int,
            p: int,
            speed_order: int,
            m: int,
        ) -> Any:
            key = (output_hermite, product_order, p, speed_order, m)
            if key not in inverse_cache:
                inverse_cache[key] = _hermite_laguerre_to_associated_legendre_mp(
                    output_hermite,
                    product_order,
                    p,
                    speed_order,
                    m,
                    mp,
                    associated_transform=inverse_associated,
                    laguerre_product=inverse_product,
                )
            return inverse_cache[key]

        moment_vectors = {
            (p, j, m): (
                tuple(
                    spherical_moment(
                        p,
                        j,
                        m,
                        input_hermite,
                        input_laguerre,
                        source=False,
                    )
                    for input_hermite in range(maximum_hermite_order + 1)
                    for input_laguerre in range(n_laguerre)
                ),
                tuple(
                    spherical_moment(
                        p,
                        j,
                        m,
                        input_hermite,
                        input_laguerre,
                        source=True,
                    )
                    for input_hermite in range(maximum_hermite_order + 1)
                    for input_laguerre in range(n_laguerre)
                ),
            )
            for p in range(spherical_limit + 1)
            for j in range(radial_limit + 1)
            for m in ((0,) if drift_kinetic else range(p + 1))
        }

        for output_hermite in range(maximum_hermite_order + 1):
            output_normalization = mp.sqrt(
                mp.power(2, output_hermite) * mp.factorial(output_hermite)
            )
            for output_laguerre in range(n_laguerre):
                row = output_hermite * n_laguerre + output_laguerre
                for p in range(spherical_limit + 1):
                    for j in range(radial_limit + 1):
                        sigma_pj = (
                            mp.factorial(p)
                            * mp.gamma(p + j + mp.mpf("1.5"))
                            / (
                                mp.power(2, p)
                                * mp.gamma(p + mp.mpf("1.5"))
                                * mp.factorial(j)
                            )
                        )
                        angular_base = (
                            mp.power(2, p)
                            * mp.factorial(p) ** 2
                            / (sigma_pj * mp.factorial(2 * p) * (2 * p + 1))
                        )
                        for m in ((0,) if drift_kinetic else range(p + 1)):
                            test_moment_vector, field_moment_vector = moment_vectors[
                                (p, j, m)
                            ]
                            if not any(
                                value != 0
                                for value in test_moment_vector + field_moment_vector
                            ):
                                continue
                            angular = (1 if m == 0 else 2) * angular_base
                            test_output_coefficient = mp.mpf(0)
                            field_output_coefficient = mp.mpf(0)
                            for n in bessel_orders:
                                kernel = (
                                    mp.exp(-bessel_argument)
                                    * bessel_argument**n
                                    / mp.factorial(n)
                                )
                                bessel_factor = (
                                    mp.factorial(n)
                                    * kernel
                                    * half_b**m
                                    / mp.factorial(n + m)
                                )
                                if bessel_factor == 0:
                                    continue
                                for product_order in range(output_laguerre + n + 1):
                                    product = laguerre_product(
                                        m,
                                        n,
                                        output_laguerre,
                                        product_order,
                                    )
                                    if product == 0:
                                        continue
                                    if p > output_hermite + m + 2 * product_order:
                                        continue
                                    maximum_speed_order = (
                                        product_order + (output_hermite + m) // 2
                                    )
                                    for speed_order in range(maximum_speed_order + 1):
                                        inverse = inverse_transform(
                                            output_hermite,
                                            product_order,
                                            p,
                                            speed_order,
                                            m,
                                        )
                                        if inverse == 0:
                                            continue
                                        test_speed, field_speed = integrated_speed(
                                            p,
                                            j,
                                            speed_order,
                                        )
                                        common = (
                                            angular
                                            * product
                                            * inverse
                                            * bessel_factor
                                            / output_normalization
                                        )
                                        test_output_coefficient += common * test_speed
                                        field_output_coefficient += common * field_speed
                            if test_output_coefficient != 0:
                                for column, moment in enumerate(test_moment_vector):
                                    test_matrix[row, column] += (
                                        test_output_coefficient * moment
                                    )
                            if field_output_coefficient != 0:
                                for column, moment in enumerate(field_moment_vector):
                                    field_matrix[row, column] += (
                                        field_output_coefficient * moment
                                    )

        test = np.asarray(test_matrix.tolist(), dtype=np.float64)
        field = np.asarray(field_matrix.tolist(), dtype=np.float64)
    return test, field


def coulomb_drift_kinetic_moment_matrices(
    maximum_hermite_order: int,
    maximum_laguerre_order: int,
    mass_ratio: float,
    temperature_ratio: float,
    *,
    maximum_spherical_order: int | None = None,
    maximum_spherical_radial_order: int | None = None,
    digits: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Generate drift-kinetic Coulomb test and field moment matrices.

    This evaluates equations (3.53)--(3.56) of Frei et al. (2021) directly.
    At zero Larmor radius only the azimuthal harmonic ``m=0`` remains, so the
    Bessel and Laguerre-product sums used by the finite-wavelength generator
    collapse exactly. Rows and columns use Hermite-major ``p * Nl + j``
    ordering and the paper's Laguerre convention.

    The spherical and radial limits remain explicit because collision-table
    promotion requires a resolved hierarchy, not only a resolved runtime
    Hermite--Laguerre state.
    """

    if maximum_hermite_order < 0:
        raise ValueError("maximum_hermite_order must be >= 0")
    if maximum_laguerre_order < 0:
        raise ValueError("maximum_laguerre_order must be >= 0")
    if mass_ratio <= 0.0 or not math.isfinite(mass_ratio):
        raise ValueError("mass_ratio must be finite and > 0")
    if temperature_ratio <= 0.0 or not math.isfinite(temperature_ratio):
        raise ValueError("temperature_ratio must be finite and > 0")
    if digits < 16:
        raise ValueError("digits must be >= 16")

    maximum_degree = maximum_hermite_order + 2 * maximum_laguerre_order
    spherical_limit = (
        maximum_degree if maximum_spherical_order is None else maximum_spherical_order
    )
    radial_limit = (
        maximum_degree // 2
        if maximum_spherical_radial_order is None
        else maximum_spherical_radial_order
    )
    if spherical_limit < 0:
        raise ValueError("maximum_spherical_order must be >= 0")
    if radial_limit < 0:
        raise ValueError("maximum_spherical_radial_order must be >= 0")

    import mpmath as mp

    n_laguerre = maximum_laguerre_order + 1
    n_modes = (maximum_hermite_order + 1) * n_laguerre
    moment_orders = tuple(
        (p, j)
        for p in range(spherical_limit + 1)
        for j in range(radial_limit + 1)
    )

    with mp.workdps(digits):
        sigma = mp.mpf(mass_ratio)
        tau = mp.mpf(temperature_ratio)
        chi = mp.sqrt(tau / sigma)

        coulomb_e, coulomb_E = _cached_coulomb_integrals_mp(chi, mp)

        @cache
        def spherical_polynomial(p: int, j: int) -> tuple[tuple[int, int, Any], ...]:
            coefficients: dict[tuple[int, int], Any] = {}
            for parallel_power in range(p + 1):
                legendre = _legendre_monomial_coefficient_mp(
                    p, parallel_power, mp
                )
                if legendre == 0:
                    continue
                for radial_power in range(j + 1):
                    laguerre = _associated_laguerre_monomial_coefficient_mp(
                        j, p, radial_power, mp
                    )
                    total_radial_power = (p - parallel_power) // 2 + radial_power
                    for parallel_radial_power in range(total_radial_power + 1):
                        key = (
                            parallel_power + 2 * parallel_radial_power,
                            total_radial_power - parallel_radial_power,
                        )
                        coefficients[key] = coefficients.get(key, mp.mpf(0)) + (
                            legendre
                            * laguerre
                            * math.comb(total_radial_power, parallel_radial_power)
                        )
            return tuple(
                (parallel_power, perpendicular_power, coefficient)
                for (parallel_power, perpendicular_power), coefficient in sorted(
                    coefficients.items()
                )
                if coefficient != 0
            )

        @cache
        def hermite_gaussian_moment(g: int, power: int) -> Any:
            total = mp.mpf(0)
            for pair_order in range(g // 2 + 1):
                combined_power = power + g - 2 * pair_order
                if combined_power % 2:
                    continue
                coefficient = (
                    (-1) ** pair_order
                    * mp.factorial(g)
                    * mp.power(2, g - 2 * pair_order)
                    / (
                        mp.factorial(pair_order)
                        * mp.factorial(g - 2 * pair_order)
                    )
                )
                total += coefficient * mp.gamma((combined_power + 1) / 2)
            return total

        @cache
        def laguerre_exponential_moment(h: int, power: int) -> Any:
            return sum(
                _associated_laguerre_monomial_coefficient_mp(
                    h, -mp.mpf("0.5"), monomial_order, mp
                )
                * mp.factorial(power + monomial_order)
                for monomial_order in range(h + 1)
            )

        @cache
        def transform(p: int, j: int, g: int, h: int) -> Any:
            left_degree = p + 2 * j
            right_degree = g + 2 * h
            if right_degree > left_degree or (left_degree - right_degree) % 2:
                return mp.mpf(0)
            projection = sum(
                coefficient
                * hermite_gaussian_moment(g, parallel_power)
                * laguerre_exponential_moment(h, perpendicular_power)
                for parallel_power, perpendicular_power, coefficient in spherical_polynomial(
                    p, j
                )
            )
            return projection / (
                mp.sqrt(mp.pi) * mp.power(2, g) * mp.factorial(g)
            )

        @cache
        def speed_moment(p: int, j: int, speed_power: int) -> tuple[Any, Any]:
            return _coulomb_speed_moments_mp(
                p,
                j,
                speed_power,
                sigma,
                tau,
                1,
                mp,
                coulomb_e=coulomb_e,
                coulomb_E=coulomb_E,
            )

        @cache
        def integrated_speed(p: int, j: int, t: int) -> tuple[Any, Any]:
            test_speed = mp.mpf(0)
            field_speed = mp.mpf(0)
            for speed_power in range(t + 1):
                laguerre_coefficient = _associated_laguerre_monomial_coefficient_mp(
                    t, p, speed_power, mp
                )
                test_term, field_term = speed_moment(p, j, speed_power)
                test_speed += laguerre_coefficient * test_term
                field_speed += laguerre_coefficient * field_term
            return test_speed, field_speed

        # Equation (3.54): map runtime Hermite--Laguerre coefficients to the
        # particle spherical moments used by both collision components.
        moment_map = mp.matrix(len(moment_orders), n_modes)
        for moment_index, (p, j) in enumerate(moment_orders):
            for g in range(maximum_hermite_order + 1):
                normalization = mp.sqrt(mp.power(2, g) * mp.factorial(g))
                for h in range(n_laguerre):
                    moment_map[moment_index, g * n_laguerre + h] = (
                        transform(p, j, g, h) * normalization
                    )

        test_projection = mp.matrix(n_modes, len(moment_orders))
        field_projection = mp.matrix(n_modes, len(moment_orders))
        for output_hermite in range(maximum_hermite_order + 1):
            output_normalization = mp.sqrt(
                mp.power(2, output_hermite) * mp.factorial(output_hermite)
            )
            for output_laguerre in range(n_laguerre):
                row = output_hermite * n_laguerre + output_laguerre
                maximum_speed_order = output_laguerre + output_hermite // 2
                for moment_index, (p, j) in enumerate(moment_orders):
                    if p > output_hermite + 2 * output_laguerre:
                        continue
                    sigma_pj = (
                        mp.factorial(p)
                        * mp.gamma(p + j + mp.mpf("1.5"))
                        / (
                            mp.power(2, p)
                            * mp.gamma(p + mp.mpf("1.5"))
                            * mp.factorial(j)
                        )
                    )
                    angular = (
                        mp.power(2, p)
                        * mp.factorial(p) ** 2
                        / (sigma_pj * mp.factorial(2 * p) * (2 * p + 1))
                    )
                    test_coefficient = mp.mpf(0)
                    field_coefficient = mp.mpf(0)
                    for speed_order in range(maximum_speed_order + 1):
                        if output_laguerre > speed_order + p // 2:
                            continue
                        inverse_prefactor = (
                            mp.sqrt(mp.pi)
                            * mp.power(2, output_hermite)
                            * mp.factorial(output_hermite)
                            * mp.factorial(speed_order)
                            * (p + mp.mpf("0.5"))
                            / mp.gamma(speed_order + p + mp.mpf("1.5"))
                        )
                        inverse = inverse_prefactor * transform(
                            p,
                            speed_order,
                            output_hermite,
                            output_laguerre,
                        )
                        if inverse == 0:
                            continue
                        test_speed, field_speed = integrated_speed(p, j, speed_order)
                        common = angular * inverse / output_normalization
                        test_coefficient += common * test_speed
                        field_coefficient += common * field_speed
                    test_projection[row, moment_index] = test_coefficient
                    field_projection[row, moment_index] = field_coefficient

        test = np.asarray((test_projection * moment_map).tolist(), dtype=np.float64)
        field = np.asarray((field_projection * moment_map).tolist(), dtype=np.float64)
    return test, field


def original_sugama_like_species_moment_matrices(
    coulomb_test_matrix: np.ndarray,
    maximum_hermite_order: int,
    maximum_laguerre_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Build the equal-temperature like-species original-Sugama matrices.

    At equal temperature the original-Sugama test component is the Coulomb
    test component. Its field component is the self-adjoint, rank-one
    restoration of parallel momentum and thermal energy described by Sugama,
    Watanabe & Nunami (2009). In an orthonormal Hermite--Laguerre basis each
    invariant ``u`` contributes ``-(T u)(T u)^T / (u^T T u)``. The input and
    output use the paper's Laguerre convention, matching the offline Coulomb
    generator.
    """

    if maximum_hermite_order < 2:
        raise ValueError("maximum_hermite_order must be >= 2")
    if maximum_laguerre_order < 1:
        raise ValueError("maximum_laguerre_order must be >= 1")
    n_laguerre = maximum_laguerre_order + 1
    mode_count = (maximum_hermite_order + 1) * n_laguerre
    test_paper = np.asarray(coulomb_test_matrix, dtype=float)
    if test_paper.shape != (mode_count, mode_count):
        raise ValueError("coulomb_test_matrix shape does not match the resolution")
    if not np.all(np.isfinite(test_paper)):
        raise ValueError("coulomb_test_matrix must contain only finite values")

    convention_sign = np.asarray(
        [
            (-1.0) ** laguerre
            for _hermite in range(maximum_hermite_order + 1)
            for laguerre in range(n_laguerre)
        ]
    )
    convention = convention_sign[:, None] * convention_sign[None, :]
    test = convention * test_paper
    momentum = np.zeros(mode_count)
    momentum[n_laguerre] = 1.0
    energy = np.zeros(mode_count)
    energy[1] = 1.0
    energy[2 * n_laguerre] = 1 / np.sqrt(2.0)
    field = np.zeros_like(test)
    for invariant in (momentum, energy):
        image = test @ invariant
        denominator = float(invariant @ image)
        if denominator >= 0.0 or not math.isfinite(denominator):
            raise ValueError(
                "coulomb_test_matrix must dissipate momentum and thermal energy"
            )
        field -= np.outer(image, image) / denominator
    return test_paper.copy(), convention * field


def improved_sugama_equal_temperature_moment_matrices(
    coulomb_test_matrix: np.ndarray,
    maximum_hermite_order: int,
    maximum_laguerre_order: int,
    *,
    correction_order: int = 3,
    digits: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Build the equal-species improved-Sugama matrices through order ``K``.

    The original-Sugama matrices are augmented by the drift-kinetic field
    correction in Frei, Ernst & Ricci (2022), equation (79).  At equal mass
    and temperature the test correction vanishes.  The field correction uses
    the exact Coulomb and original-Sugama Braginskii ``N`` matrices from their
    equations (45), (60), and (80), followed by the basis transform in
    equations (79) and (81).  Rows and columns retain the paper's Laguerre
    convention.

    ``correction_order`` must fit completely inside the retained basis:
    ``P >= 2 K + 1`` and ``J >= K``.  This prevents a nominal high-order
    correction from silently using an incomplete total-degree shell.
    """

    if correction_order < 0:
        raise ValueError("correction_order must be >= 0")
    if maximum_hermite_order < 2 * correction_order + 1:
        raise ValueError("maximum_hermite_order must be >= 2 * correction_order + 1")
    if maximum_laguerre_order < correction_order:
        raise ValueError("maximum_laguerre_order must be >= correction_order")
    if digits < 16:
        raise ValueError("digits must be >= 16")

    original_test, original_field = original_sugama_like_species_moment_matrices(
        coulomb_test_matrix,
        maximum_hermite_order,
        maximum_laguerre_order,
    )
    import mpmath as mp

    n_laguerre = maximum_laguerre_order + 1
    mode_count = (maximum_hermite_order + 1) * n_laguerre
    with mp.workdps(digits):
        root_pi = mp.sqrt(mp.pi)
        collision_time = 3 * root_pi / 4
        coulomb_e, coulomb_E = _cached_coulomb_integrals_mp(mp.mpf(1), mp)

        @cache
        def field_speed_moment(k: int, speed_power: int) -> Any:
            return _coulomb_speed_moments_mp(
                1,
                k,
                speed_power,
                mp.mpf(1),
                mp.mpf(1),
                mp.mpf(1),
                mp,
                coulomb_e=coulomb_e,
                coulomb_E=coulomb_E,
            )[1]

        @cache
        def laguerre(order: int, monomial: int) -> Any:
            return _associated_laguerre_monomial_coefficient_mp(
                order, 1, monomial, mp
            )

        def flow_weight(order: int) -> Any:
            double_factorial = mp.fac2(2 * order + 3)
            return 3 * mp.power(2, order) * mp.factorial(order) / double_factorial

        coulomb_n = mp.matrix(correction_order + 1, correction_order + 1)
        for ell in range(correction_order + 1):
            for k in range(correction_order + 1):
                coulomb_n[ell, k] = sum(
                    mp.mpf(2)
                    / 3
                    * collision_time
                    * laguerre(ell, power)
                    * field_speed_moment(k, power)
                    for power in range(ell + 1)
                )
        delta_n = mp.matrix(correction_order + 1, correction_order + 1)
        for ell in range(correction_order + 1):
            for k in range(correction_order + 1):
                delta_n[ell, k] = (
                    coulomb_n[ell, k]
                    - coulomb_n[ell, 0]
                    * coulomb_n[0, k]
                    / coulomb_n[0, 0]
                )
        for order in range(correction_order + 1):
            delta_n[0, order] = mp.mpf(0)
            delta_n[order, 0] = mp.mpf(0)

        correction = mp.matrix(mode_count, mode_count)
        for ell in range(correction_order + 1):
            output = mp.matrix(mode_count, 1)
            shell_degree = 1 + 2 * ell
            for p in range(maximum_hermite_order + 1):
                remainder = shell_degree - p
                if remainder < 0 or remainder % 2:
                    continue
                j = remainder // 2
                if j > maximum_laguerre_order:
                    continue
                forward = _legendre_to_hermite_laguerre_mp(1, ell, p, j, mp)
                inverse_factor = (
                    root_pi
                    * mp.power(2, p)
                    * mp.factorial(p)
                    * mp.mpf("1.5")
                    * mp.factorial(ell)
                    / mp.gamma(ell + mp.mpf("2.5"))
                )
                inverse = inverse_factor * forward
                output[p * n_laguerre + j] = (
                    inverse
                    / mp.sqrt(mp.power(2, p) * mp.factorial(p))
                    * mp.gamma(ell + mp.mpf("2.5"))
                    / mp.factorial(ell)
                )

            for k in range(correction_order + 1):
                source = mp.matrix(mode_count, 1)
                source_degree = 1 + 2 * k
                for g in range(maximum_hermite_order + 1):
                    remainder = source_degree - g
                    if remainder < 0 or remainder % 2:
                        continue
                    h = remainder // 2
                    if h > maximum_laguerre_order:
                        continue
                    source[g * n_laguerre + h] = (
                        _legendre_to_hermite_laguerre_mp(1, k, g, h, mp)
                        * mp.sqrt(mp.power(2, g) * mp.factorial(g))
                    )
                prefactor = (
                    4
                    * flow_weight(ell)
                    * flow_weight(k)
                    / (3 * root_pi * collision_time)
                    * delta_n[ell, k]
                )
                correction += prefactor * output * source.T

    return original_test, original_field + np.asarray(
        correction.tolist(), dtype=np.float64
    )


def coulomb_polarization_vectors(
    maximum_hermite_order: int,
    maximum_laguerre_order: int,
    target_kperp_rho: float,
    source_kperp_rho: float,
    mass_ratio: float,
    temperature_ratio: float,
    *,
    maximum_spherical_order: int | None = None,
    maximum_spherical_radial_order: int | None = None,
    maximum_bessel_laguerre_order: int = 24,
    digits: int = 80,
    _coefficient_functions: tuple[Callable[..., Any], ...] | None = None,
    _assembly_cache: dict[str, dict[tuple[Any, ...], Any]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Generate the four Coulomb polarization vectors in equation (3.50).

    The result is ``(test_phi1, field_phi1, test_phi2, field_phi2)`` in the
    paper's Laguerre convention.  Test vectors multiply ``q_a phi / T_a``;
    field vectors multiply ``q_b phi / T_b``.  Target and source gyroradii are
    separate because unlike-species polarization uses both.
    """

    if maximum_hermite_order < 0:
        raise ValueError("maximum_hermite_order must be >= 0")
    if maximum_laguerre_order < 0:
        raise ValueError("maximum_laguerre_order must be >= 0")
    for value, name in (
        (target_kperp_rho, "target_kperp_rho"),
        (source_kperp_rho, "source_kperp_rho"),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and >= 0")
    if mass_ratio <= 0.0 or not math.isfinite(mass_ratio):
        raise ValueError("mass_ratio must be finite and > 0")
    if temperature_ratio <= 0.0 or not math.isfinite(temperature_ratio):
        raise ValueError("temperature_ratio must be finite and > 0")
    if maximum_bessel_laguerre_order < 0:
        raise ValueError("maximum_bessel_laguerre_order must be >= 0")
    if digits < 16:
        raise ValueError("digits must be >= 16")

    maximum_degree = maximum_hermite_order + 2 * maximum_laguerre_order
    spherical_limit = (
        maximum_degree if maximum_spherical_order is None else maximum_spherical_order
    )
    radial_limit = (
        maximum_degree // 2
        if maximum_spherical_radial_order is None
        else maximum_spherical_radial_order
    )
    if spherical_limit < 0:
        raise ValueError("maximum_spherical_order must be >= 0")
    if radial_limit < 0:
        raise ValueError("maximum_spherical_radial_order must be >= 0")

    import mpmath as mp

    n_laguerre = maximum_laguerre_order + 1
    n_modes = (maximum_hermite_order + 1) * n_laguerre
    with mp.workdps(digits):
        target_b = mp.mpf(target_kperp_rho)
        source_b = mp.mpf(source_kperp_rho)
        sigma = mp.mpf(mass_ratio)
        tau = mp.mpf(temperature_ratio)
        half_target_b = target_b / 2
        target_argument = half_target_b * half_target_b
        test_phi1 = mp.matrix(n_modes, 1)
        field_phi1 = mp.matrix(n_modes, 1)
        test_phi2 = mp.matrix(n_modes, 1)
        field_phi2 = mp.matrix(n_modes, 1)
        assembly_cache = {} if _assembly_cache is None else _assembly_cache
        speed_cache = assembly_cache.setdefault("integrated_speed", {})
        polarization_cache = assembly_cache.setdefault("polarization", {})
        product_cache = assembly_cache.setdefault("laguerre_product", {})
        inverse_cache = assembly_cache.setdefault("inverse_transform", {})
        coefficient_functions = (
            _coulomb_coefficient_functions(mp, sigma, tau)
            if _coefficient_functions is None
            else _coefficient_functions
        )
        (
            _base_transform,
            associated_laguerre,
            _legendre_monomial,
            inverse_associated,
            inverse_product,
            speed_moment,
        ) = coefficient_functions

        def integrated_speed(p: int, j: int, t: int) -> tuple[Any, Any]:
            key = (p, j, t)
            if key not in speed_cache:
                test_speed = mp.mpf(0)
                field_speed = mp.mpf(0)
                for speed_power in range(t + 1):
                    coefficient = associated_laguerre(
                        t,
                        p,
                        speed_power,
                    )
                    test_term, field_term = speed_moment(p, j, speed_power)
                    test_speed += coefficient * test_term
                    field_speed += coefficient * field_term
                speed_cache[key] = (test_speed, field_speed)
            return speed_cache[key]

        def polarization(p: int, j: int, m: int, *, source: bool) -> Any:
            wavelength = source_b if source else target_b
            key = (float(wavelength), p, j, m)
            if key not in polarization_cache:
                polarization_cache[key] = _gyroaveraged_polarization_coefficient_mp(
                    p,
                    j,
                    m,
                    wavelength,
                    maximum_bessel_laguerre_order,
                    mp,
                )
            return polarization_cache[key]

        def laguerre_product(
            m: int,
            n: int,
            output_laguerre: int,
            product_order: int,
        ) -> Any:
            key = (m, n, output_laguerre, product_order)
            if key not in product_cache:
                product_cache[key] = _laguerre_product_expansion_coefficient_mp(
                    m, n, output_laguerre, product_order, 0, mp
                )
            return product_cache[key]

        def inverse_transform(
            output_hermite: int,
            product_order: int,
            p: int,
            speed_order: int,
            m: int,
        ) -> Any:
            key = (output_hermite, product_order, p, speed_order, m)
            if key not in inverse_cache:
                inverse_cache[key] = _hermite_laguerre_to_associated_legendre_mp(
                    output_hermite,
                    product_order,
                    p,
                    speed_order,
                    m,
                    mp,
                    associated_transform=inverse_associated,
                    laguerre_product=inverse_product,
                )
            return inverse_cache[key]

        for output_hermite in range(maximum_hermite_order + 1):
            output_normalization = mp.sqrt(
                mp.power(2, output_hermite) * mp.factorial(output_hermite)
            )
            for output_laguerre in range(n_laguerre):
                row = output_hermite * n_laguerre + output_laguerre
                for n in range(maximum_bessel_laguerre_order + 1):
                    kernel = (
                        mp.exp(-target_argument) * target_argument**n / mp.factorial(n)
                    )
                    if kernel == 0:
                        continue
                    for product_order in range(output_laguerre + n + 1):
                        product = laguerre_product(0, n, output_laguerre, product_order)
                        for speed_order in range(
                            product_order + output_hermite // 2 + 1
                        ):
                            inverse = inverse_transform(
                                output_hermite,
                                product_order,
                                0,
                                speed_order,
                                0,
                            )
                            if product == 0 or inverse == 0:
                                continue
                            test_speed, field_speed = integrated_speed(
                                0,
                                0,
                                speed_order,
                            )
                            common = product * kernel * inverse / output_normalization
                            test_phi1[row] -= common * test_speed
                            field_phi1[row] -= common * field_speed

                for p in range(spherical_limit + 1):
                    for j in range(radial_limit + 1):
                        sigma_pj = (
                            mp.factorial(p)
                            * mp.gamma(p + j + mp.mpf("1.5"))
                            / (
                                mp.power(2, p)
                                * mp.gamma(p + mp.mpf("1.5"))
                                * mp.factorial(j)
                            )
                        )
                        angular_base = (
                            mp.power(2, p)
                            * mp.factorial(p) ** 2
                            / (sigma_pj * mp.factorial(2 * p) * (2 * p + 1))
                        )
                        for m in range(p + 1):
                            target_polarization = polarization(p, j, m, source=False)
                            source_polarization = polarization(p, j, m, source=True)
                            if target_polarization == 0 and source_polarization == 0:
                                continue
                            angular = (1 if m == 0 else 2) * angular_base
                            for n in range(maximum_bessel_laguerre_order + 1):
                                kernel = (
                                    mp.exp(-target_argument)
                                    * target_argument**n
                                    / mp.factorial(n)
                                )
                                bessel_factor = (
                                    mp.factorial(n)
                                    * kernel
                                    * half_target_b**m
                                    / mp.factorial(n + m)
                                )
                                if bessel_factor == 0:
                                    continue
                                for product_order in range(output_laguerre + n + 1):
                                    product = laguerre_product(
                                        m,
                                        n,
                                        output_laguerre,
                                        product_order,
                                    )
                                    if (
                                        product == 0
                                        or p > output_hermite + m + 2 * product_order
                                    ):
                                        continue
                                    maximum_speed_order = (
                                        product_order + (output_hermite + m) // 2
                                    )
                                    for speed_order in range(maximum_speed_order + 1):
                                        inverse = inverse_transform(
                                            output_hermite,
                                            product_order,
                                            p,
                                            speed_order,
                                            m,
                                        )
                                        if inverse == 0:
                                            continue
                                        test_speed, field_speed = integrated_speed(
                                            p,
                                            j,
                                            speed_order,
                                        )
                                        common = (
                                            angular
                                            * product
                                            * inverse
                                            * bessel_factor
                                            / output_normalization
                                        )
                                        test_phi2[row] += (
                                            common * target_polarization * test_speed
                                        )
                                        field_phi2[row] += (
                                            common * source_polarization * field_speed
                                        )

        vectors = tuple(
            np.asarray(vector.tolist(), dtype=np.float64).reshape(n_modes)
            for vector in (test_phi1, field_phi1, test_phi2, field_phi2)
        )
    return vectors


def build_finite_wavelength_coulomb_pair_tables(
    kperp_rho: tuple[float, ...],
    maximum_hermite_order: int,
    maximum_laguerre_order: int,
    mass_ratio: float,
    temperature_ratio: float,
    *,
    maximum_spherical_order: int | None = None,
    maximum_spherical_radial_order: int | None = None,
    maximum_bessel_laguerre_order: int = 24,
    digits: int = 80,
) -> tuple[np.ndarray, ...]:
    r"""Build one ordered-pair table for the JAX finite-wavelength operator.

    The returned test/field matrices and four polarization vectors have
    independent target/source ``kperp*rho`` axes.  All kperp-independent
    multiprecision basis algebra is shared across the scan.  Unlike the two
    equation-level generators, these tables use the runtime's signed Laguerre
    convention and can therefore be inserted directly below target/source
    species axes in :class:`FiniteWavelengthCoulombOperator`.
    """

    grid = np.asarray(kperp_rho, dtype=float)
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError("kperp_rho must contain at least two points")
    if not np.all(np.isfinite(grid)) or np.any(grid < 0.0):
        raise ValueError("kperp_rho values must be finite and >= 0")
    if np.any(np.diff(grid) <= 0.0):
        raise ValueError("kperp_rho must be strictly increasing")

    import mpmath as mp

    mode_count = (maximum_hermite_order + 1) * (maximum_laguerre_order + 1)
    matrices = [np.empty((grid.size, grid.size, mode_count, mode_count)) for _ in range(2)]
    vectors = [np.empty((grid.size, grid.size, mode_count)) for _ in range(4)]
    laguerre_sign = np.asarray(
        [
            (-1.0) ** laguerre_order
            for _hermite_order in range(maximum_hermite_order + 1)
            for laguerre_order in range(maximum_laguerre_order + 1)
        ]
    )
    matrix_convention = laguerre_sign[:, None] * laguerre_sign[None, :]

    with mp.workdps(digits):
        coefficient_functions = _coulomb_coefficient_functions(
            mp,
            mp.mpf(mass_ratio),
            mp.mpf(temperature_ratio),
        )
        assembly_cache: dict[str, dict[tuple[Any, ...], Any]] = {}
        for target_index, target_kperp in enumerate(grid):
            for source_index, source_kperp in enumerate(grid):
                pair_matrices = coulomb_nonpolarized_moment_matrices(
                    maximum_hermite_order,
                    maximum_laguerre_order,
                    float(target_kperp),
                    mass_ratio,
                    temperature_ratio,
                    source_kperp_rho=float(source_kperp),
                    maximum_spherical_order=maximum_spherical_order,
                    maximum_spherical_radial_order=maximum_spherical_radial_order,
                    maximum_bessel_laguerre_order=maximum_bessel_laguerre_order,
                    digits=digits,
                    _coefficient_functions=coefficient_functions,
                    _assembly_cache=assembly_cache,
                )
                pair_vectors = coulomb_polarization_vectors(
                    maximum_hermite_order,
                    maximum_laguerre_order,
                    float(target_kperp),
                    float(source_kperp),
                    mass_ratio,
                    temperature_ratio,
                    maximum_spherical_order=maximum_spherical_order,
                    maximum_spherical_radial_order=maximum_spherical_radial_order,
                    maximum_bessel_laguerre_order=maximum_bessel_laguerre_order,
                    digits=digits,
                    _coefficient_functions=coefficient_functions,
                    _assembly_cache=assembly_cache,
                )
                for table, values in zip(matrices, pair_matrices, strict=True):
                    table[target_index, source_index] = matrix_convention * values
                for table, values in zip(vectors, pair_vectors, strict=True):
                    table[target_index, source_index] = laguerre_sign * values
    return (*matrices, *vectors)


def build_coulomb_operator_verification_summary(*, digits: int = 80) -> dict[str, Any]:
    r"""Build equation, convergence, conservation, and entropy gates.

    This is an offline verification artifact for Frei et al. (2021), equations
    (3.41) and (3.48)--(3.50).  It deliberately stops short of transport claims:
    those require the generated blocks to be coupled through the runtime field
    solve and quasineutrality relation.
    """

    from scipy.special import eval_genlaguerre, jv, lpmv

    truncation_orders = np.asarray((0, 1, 2, 3, 4, 6, 8, 12), dtype=int)
    reference_order = 24
    truncation_values = np.asarray(
        [
            gyroaveraged_polarization_coefficient(
                3,
                1,
                1,
                1.8,
                maximum_bessel_laguerre_order=int(order),
                digits=digits,
            )
            for order in truncation_orders
        ]
    )
    reference_value = gyroaveraged_polarization_coefficient(
        3,
        1,
        1,
        1.8,
        maximum_bessel_laguerre_order=reference_order,
        digits=digits,
    )
    truncation_errors = np.abs(truncation_values - reference_value) / max(
        abs(reference_value), np.finfo(float).tiny
    )

    parallel, parallel_weights = np.polynomial.hermite.hermgauss(80)
    perpendicular, perpendicular_weights = np.polynomial.laguerre.laggauss(80)
    x_parallel = parallel[:, None]
    x_perpendicular = perpendicular[None, :]
    speed = np.sqrt(x_parallel**2 + x_perpendicular)
    pitch = x_parallel / speed
    projection_cases = (
        (0, 0, 0, 0.7),
        (1, 0, 1, 0.7),
        (2, 0, 0, 0.7),
        (2, 0, 2, 0.7),
        (3, 1, 1, 1.2),
    )
    projection_labels: list[str] = []
    projection_errors: list[float] = []
    for spherical_order, radial_order, bessel_order, b_value in projection_cases:
        generated = gyroaveraged_polarization_coefficient(
            spherical_order,
            radial_order,
            bessel_order,
            b_value,
            maximum_bessel_laguerre_order=14,
            digits=digits,
        )
        spherical_basis = (
            speed**spherical_order
            * lpmv(bessel_order, spherical_order, pitch)
            * eval_genlaguerre(
                radial_order,
                spherical_order + 0.5,
                speed**2,
            )
        )
        projected = np.sum(
            parallel_weights[:, None]
            * perpendicular_weights[None, :]
            * spherical_basis
            * jv(0, b_value * np.sqrt(x_perpendicular))
            * jv(bessel_order, b_value * np.sqrt(x_perpendicular))
        ) / np.sqrt(np.pi)
        projection_labels.append(
            rf"$\Pi^{{{spherical_order},{radial_order},{bessel_order}}}$"
        )
        projection_errors.append(
            float(abs(generated - projected) / max(abs(projected), 1.0e-14))
        )

    matrix_cache: dict[tuple[int, int, int], np.ndarray] = {}

    def finite_b_matrix(
        spherical_order: int,
        spherical_radial_order: int,
        bessel_laguerre_order: int,
    ) -> np.ndarray:
        key = (spherical_order, spherical_radial_order, bessel_laguerre_order)
        if key not in matrix_cache:
            matrix_test, matrix_field = coulomb_nonpolarized_moment_matrices(
                1,
                1,
                0.8,
                1.0,
                1.0,
                maximum_spherical_order=spherical_order,
                maximum_spherical_radial_order=spherical_radial_order,
                maximum_bessel_laguerre_order=bessel_laguerre_order,
                digits=max(32, min(digits, 40)),
            )
            matrix_cache[key] = matrix_test + matrix_field
        return matrix_cache[key]

    matrix_truncation_orders = np.asarray((2, 4, 6), dtype=int)
    matrix_truncations = [
        finite_b_matrix(8, 4, int(order)) for order in matrix_truncation_orders
    ]
    matrix_truncation_reference = matrix_truncations[-1]
    matrix_reference_norm = np.linalg.norm(matrix_truncation_reference)
    matrix_truncation_errors = np.asarray(
        [
            np.linalg.norm(matrix - matrix_truncation_reference) / matrix_reference_norm
            for matrix in matrix_truncations
        ]
    )

    spherical_truncation_cutoffs = ((3, 1), (8, 4), (9, 4))
    spherical_truncations = [
        finite_b_matrix(spherical_order, spherical_radial_order, 6)
        for spherical_order, spherical_radial_order in spherical_truncation_cutoffs
    ]
    spherical_truncation_reference = spherical_truncations[-1]
    spherical_reference_norm = np.linalg.norm(spherical_truncation_reference)
    spherical_truncation_errors = np.asarray(
        [
            np.linalg.norm(matrix - spherical_truncation_reference)
            / spherical_reference_norm
            for matrix in spherical_truncations
        ]
    )

    test_matrix, field_matrix = coulomb_nonpolarized_moment_matrices(
        3,
        1,
        0.0,
        1.0,
        1.0,
        maximum_spherical_order=5,
        maximum_spherical_radial_order=2,
        maximum_bessel_laguerre_order=0,
        digits=digits,
    )
    laguerre_sign = np.asarray(
        [(-1.0) ** laguerre for _hermite in range(4) for laguerre in range(2)]
    )
    collision_matrix = (
        laguerre_sign[:, None] * laguerre_sign[None, :] * (test_matrix + field_matrix)
    )
    symmetric_matrix = 0.5 * (collision_matrix + collision_matrix.T)
    eigenvalues = np.linalg.eigvalsh(symmetric_matrix)
    published = build_collision_table(digits=digits)[2]
    published_mask = published != 0.0

    invariants = {
        "density": np.eye(8)[0],
        "parallel_momentum": np.eye(8)[2],
        "thermal_energy": np.asarray(
            [0.0, 1.0, 0.0, 0.0, 1.0 / np.sqrt(2), 0.0, 0.0, 0.0]
        ),
    }
    invariant_residuals = {
        name: float(np.linalg.norm(collision_matrix @ vector, ord=np.inf))
        for name, vector in invariants.items()
    }

    # Finite wavelength mixes particle position and gyrocenter velocity. The
    # resulting gyrocenter-density row is classical gyro-diffusion, not a
    # violation of the particle-space conservation law (Frei et al. 2021,
    # discussion following Eq. 3.5).
    gyrocenter_b = np.asarray((0.0, 0.1, 0.2, 0.4), dtype=float)
    gyrocenter_test_density_rows = []
    gyrocenter_field_density_rows = []
    gyrocenter_density_rows = []
    for b_value in gyrocenter_b:
        gyro_test, gyro_field = coulomb_nonpolarized_moment_matrices(
            1,
            1,
            float(b_value),
            1.0,
            1.0,
            maximum_spherical_order=3,
            maximum_spherical_radial_order=1,
            maximum_bessel_laguerre_order=4,
            digits=max(32, min(digits, 48)),
        )
        gyrocenter_test_density_rows.append(
            float(np.linalg.norm(gyro_test[0], ord=np.inf))
        )
        gyrocenter_field_density_rows.append(
            float(np.linalg.norm(gyro_field[0], ord=np.inf))
        )
        gyrocenter_density_rows.append(
            float(np.linalg.norm((gyro_test + gyro_field)[0], ord=np.inf))
        )
    gyrocenter_test_density_rows_array = np.asarray(gyrocenter_test_density_rows)
    gyrocenter_field_density_rows_array = np.asarray(gyrocenter_field_density_rows)
    gyrocenter_density_rows_array = np.asarray(gyrocenter_density_rows)

    def small_b_observed_order(values: np.ndarray) -> float:
        return float(np.polyfit(np.log(gyrocenter_b[1:]), np.log(values[1:]), 1)[0])

    test_observed_order = small_b_observed_order(gyrocenter_test_density_rows_array)
    field_observed_order = small_b_observed_order(gyrocenter_field_density_rows_array)
    observed_order = small_b_observed_order(gyrocenter_density_rows_array)
    metrics = {
        "final_bessel_relative_error": float(truncation_errors[-1]),
        "maximum_projection_relative_error": float(max(projection_errors)),
        "published_coefficient_maximum_absolute_error": float(
            np.max(np.abs(collision_matrix[published_mask] - published[published_mask]))
        ),
        "symmetry_maximum_absolute_error": float(
            np.max(np.abs(collision_matrix - collision_matrix.T))
        ),
        "maximum_eigenvalue": float(np.max(eigenvalues)),
        "maximum_invariant_residual": float(max(invariant_residuals.values())),
        "drift_kinetic_gyrocenter_density_row": float(gyrocenter_density_rows_array[0]),
        "finite_b_gyrocenter_density_row": float(gyrocenter_density_rows_array[-1]),
        "small_b_test_gyrocenter_diffusion_observed_order": test_observed_order,
        "small_b_field_gyrocenter_diffusion_observed_order": field_observed_order,
        "small_b_gyrocenter_diffusion_observed_order": observed_order,
        "finite_b_matrix_relative_error_at_order_4": float(
            matrix_truncation_errors[-2]
        ),
        "finite_b_spherical_relative_error_at_order_8": float(
            spherical_truncation_errors[1]
        ),
    }
    thresholds = {
        "final_bessel_relative_error": 5.0e-9,
        "maximum_projection_relative_error": 5.0e-9,
        "published_coefficient_maximum_absolute_error": 5.0e-12,
        "symmetry_maximum_absolute_error": 5.0e-12,
        "maximum_eigenvalue": 1.0e-12,
        "maximum_invariant_residual": 5.0e-12,
        "drift_kinetic_gyrocenter_density_row": 5.0e-12,
        "finite_b_gyrocenter_density_row_minimum": 1.0e-4,
        "small_b_gyrocenter_diffusion_order_minimum": 1.7,
        "small_b_gyrocenter_diffusion_order_maximum": 2.3,
        "finite_b_matrix_relative_error_at_order_4": 5.0e-6,
        "finite_b_spherical_relative_error_at_order_8": 2.0e-6,
    }
    gates = {
        name: bool(metrics[name] <= thresholds[name])
        for name in (
            "final_bessel_relative_error",
            "maximum_projection_relative_error",
            "published_coefficient_maximum_absolute_error",
            "symmetry_maximum_absolute_error",
            "maximum_eigenvalue",
            "maximum_invariant_residual",
            "drift_kinetic_gyrocenter_density_row",
            "finite_b_matrix_relative_error_at_order_4",
            "finite_b_spherical_relative_error_at_order_8",
        )
    }
    gates["finite_b_gyrocenter_diffusion_nonzero"] = bool(
        metrics["finite_b_gyrocenter_density_row"]
        >= thresholds["finite_b_gyrocenter_density_row_minimum"]
    )
    gates["small_b_gyrocenter_diffusion_quadratic"] = bool(
        thresholds["small_b_gyrocenter_diffusion_order_minimum"]
        <= observed_order
        <= thresholds["small_b_gyrocenter_diffusion_order_maximum"]
    )
    gates["small_b_test_gyrocenter_diffusion_quadratic"] = bool(
        thresholds["small_b_gyrocenter_diffusion_order_minimum"]
        <= test_observed_order
        <= thresholds["small_b_gyrocenter_diffusion_order_maximum"]
    )
    gates["small_b_field_gyrocenter_diffusion_quadratic"] = bool(
        thresholds["small_b_gyrocenter_diffusion_order_minimum"]
        <= field_observed_order
        <= thresholds["small_b_gyrocenter_diffusion_order_maximum"]
    )
    return _json_clean(
        {
            "case": "finite_b_coulomb_operator_algebra",
            "claim_scope": "offline_operator_algebra_not_runtime_transport",
            "precision_decimal_digits": int(digits),
            "basis": {"maximum_hermite_order": 3, "maximum_laguerre_order": 1},
            "truncation": {
                "coefficient": "Pi^(3,1,1)(b=1.8)",
                "orders": truncation_orders,
                "values": truncation_values,
                "relative_errors": truncation_errors,
                "reference_order": reference_order,
                "reference_value": reference_value,
            },
            "direct_projection": {
                "quadrature": "80-point Gauss-Hermite x 80-point Gauss-Laguerre",
                "labels": projection_labels,
                "relative_errors": projection_errors,
            },
            "matrix_truncation": {
                "basis": {"maximum_hermite_order": 1, "maximum_laguerre_order": 1},
                "kperp_rho": 0.8,
                "spherical_cutoff": {
                    "maximum_order": 8,
                    "maximum_radial_order": 4,
                },
                "orders": matrix_truncation_orders,
                "relative_errors": matrix_truncation_errors,
                "reference_order": int(matrix_truncation_orders[-1]),
            },
            "spherical_truncation": {
                "basis": {"maximum_hermite_order": 1, "maximum_laguerre_order": 1},
                "kperp_rho": 0.8,
                "bessel_laguerre_order": 6,
                "maximum_spherical_orders": [
                    cutoff[0] for cutoff in spherical_truncation_cutoffs
                ],
                "maximum_spherical_radial_orders": [
                    cutoff[1] for cutoff in spherical_truncation_cutoffs
                ],
                "relative_errors": spherical_truncation_errors,
                "reference_cutoff": spherical_truncation_cutoffs[-1],
            },
            "matrix": collision_matrix,
            "eigenvalues": eigenvalues,
            "invariant_residuals": invariant_residuals,
            "gyrocenter_diffusion": {
                "kperp_rho": gyrocenter_b,
                "test_density_row_infinity_norm": gyrocenter_test_density_rows_array,
                "field_density_row_infinity_norm": gyrocenter_field_density_rows_array,
                "density_row_infinity_norm": gyrocenter_density_rows_array,
                "test_small_b_observed_order": test_observed_order,
                "field_small_b_observed_order": field_observed_order,
                "small_b_observed_order": observed_order,
                "interpretation": (
                    "equation-(3.5) finite-b classical gyro-diffusion in the "
                    "test, field, and combined gyrocenter moments; particle-space "
                    "local conservation cannot be reconstructed from the "
                    "gyrophase-averaged matrix alone"
                ),
            },
            "metrics": metrics,
            "thresholds": thresholds,
            "gates": gates,
            "gate_passed": all(gates.values()),
            "references": [
                {
                    "title": "Frei et al. (2021), advanced gyrokinetic collision operators",
                    "url": "https://arxiv.org/abs/2104.11480",
                    "equations": ["3.2", "3.5", "3.41", "3.48", "3.49", "3.50"],
                },
                {
                    "title": "Abel et al. (2008), collision-operator physical constraints",
                    "url": "https://arxiv.org/abs/0808.1300",
                    "tests": ["conservation", "H-theorem", "Maxwellian null space"],
                },
            ],
        }
    )


def write_coulomb_operator_verification_figure(
    summary: dict[str, Any], out_png: Path
) -> None:
    """Write the compact paper-facing collision algebra verification panel."""

    from matplotlib.colors import TwoSlopeNorm

    colors = {
        "blue": "#16697A",
        "orange": "#E56B1F",
        "red": "#A23B3B",
        "ink": "#18232E",
    }
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.6), constrained_layout=True)
    truncation = summary["truncation"]
    errors = np.maximum(np.asarray(truncation["relative_errors"], dtype=float), 1.0e-16)
    axes[0, 0].semilogy(
        truncation["orders"],
        errors,
        "o-",
        color=colors["blue"],
        lw=2.2,
        ms=5.5,
        label=r"polarization coefficient ($N_b$)",
    )
    matrix_truncation = summary["matrix_truncation"]
    matrix_errors = np.maximum(
        np.asarray(matrix_truncation["relative_errors"], dtype=float), 1.0e-16
    )
    axes[0, 0].semilogy(
        matrix_truncation["orders"],
        matrix_errors,
        "s--",
        color=colors["orange"],
        lw=1.7,
        ms=4.8,
        label=r"assembled block ($N_b$)",
    )
    spherical_truncation = summary["spherical_truncation"]
    spherical_errors = np.maximum(
        np.asarray(spherical_truncation["relative_errors"], dtype=float), 1.0e-16
    )
    axes[0, 0].semilogy(
        spherical_truncation["maximum_spherical_orders"],
        spherical_errors,
        "D-.",
        color=colors["red"],
        lw=1.5,
        ms=4.4,
        label=r"assembled block ($p_{\max}$)",
    )
    axes[0, 0].set(
        xlabel="spectral truncation order",
        ylabel="relative truncation error",
        title="(a) Finite-$b$ spectral convergence",
    )
    axes[0, 0].legend(frameon=False, fontsize=8)

    projection = summary["direct_projection"]
    projection_errors = np.maximum(
        np.asarray(projection["relative_errors"], dtype=float), 1.0e-16
    )
    axes[0, 1].bar(
        np.arange(len(projection_errors)),
        projection_errors,
        color=colors["orange"],
        edgecolor="white",
        linewidth=0.8,
    )
    axes[0, 1].axhline(
        summary["thresholds"]["maximum_projection_relative_error"],
        color=colors["red"],
        ls="--",
        lw=1.5,
    )
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xticks(
        np.arange(len(projection_errors)), projection["labels"], fontsize=8
    )
    axes[0, 1].set(
        ylabel="relative projection error",
        title="(b) Independent velocity-space projection",
    )

    eigenvalues = np.asarray(summary["eigenvalues"], dtype=float)
    dissipation = np.maximum(-np.sort(eigenvalues), 1.0e-17)
    axes[1, 0].semilogy(
        np.arange(1, dissipation.size + 1),
        dissipation,
        "o",
        color=colors["blue"],
        ms=6,
    )
    axes[1, 0].axhspan(1.0e-17, 5.0e-13, color="#DDE9E7", alpha=0.9)
    axes[1, 0].set(
        xlabel="ordered moment-space eigenmode",
        ylabel=r"dissipation rate $-\lambda(C)$",
        title="(c) H-theorem and three invariant null modes",
    )
    axes[1, 0].text(
        0.04,
        0.06,
        "shaded: numerical null space\n"
        + rf"max invariant residual $={summary['metrics']['maximum_invariant_residual']:.1e}$",
        transform=axes[1, 0].transAxes,
        fontsize=8,
        color=colors["ink"],
    )
    gyrocenter = summary["gyrocenter_diffusion"]
    gyrocenter_b = np.asarray(gyrocenter["kperp_rho"], dtype=float)[1:]
    test_density_row = np.asarray(
        gyrocenter["test_density_row_infinity_norm"], dtype=float
    )[1:]
    field_density_row = np.asarray(
        gyrocenter["field_density_row_infinity_norm"], dtype=float
    )[1:]
    density_row = np.asarray(gyrocenter["density_row_infinity_norm"], dtype=float)[1:]
    inset = axes[1, 0].inset_axes([0.54, 0.52, 0.42, 0.40])
    inset.semilogy(
        gyrocenter_b,
        test_density_row,
        "^-.",
        color=colors["red"],
        lw=1.0,
        ms=3.2,
        label="test",
    )
    inset.semilogy(
        gyrocenter_b,
        field_density_row,
        "v:",
        color=colors["blue"],
        lw=1.0,
        ms=3.2,
        label="field",
    )
    inset.semilogy(
        gyrocenter_b,
        density_row,
        "o-",
        color=colors["orange"],
        lw=1.4,
        ms=3.8,
        label="combined",
    )
    inset.semilogy(
        gyrocenter_b,
        density_row[0] * (gyrocenter_b / gyrocenter_b[0]) ** 2,
        "--",
        color=colors["ink"],
        lw=1.0,
        label=r"$b^2$",
    )
    inset.set_title("finite-$b$ gyro-diffusion", fontsize=7)
    inset.set_xticks(gyrocenter_b, ["0.1", "0.2", "0.4"])
    inset.set_xlabel(r"$b=k_\perp\rho$", fontsize=7)
    inset.set_ylabel(r"$\|C_{0,:}\|_\infty$", fontsize=7)
    inset.tick_params(labelsize=6)
    inset.legend(frameon=False, fontsize=5.2, loc="lower right", ncol=2)
    inset.grid(alpha=0.2, lw=0.4)

    matrix = np.asarray(summary["matrix"], dtype=float)
    matrix_limit = float(np.max(np.abs(matrix)))
    image = axes[1, 1].imshow(
        matrix,
        origin="upper",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-matrix_limit, vcenter=0.0, vmax=matrix_limit),
        interpolation="nearest",
    )
    mode_labels = [rf"$({p},{j})$" for p in range(4) for j in range(2)]
    axes[1, 1].set_xticks(range(8), mode_labels, rotation=45, ha="right", fontsize=7)
    axes[1, 1].set_yticks(range(8), mode_labels, fontsize=7)
    axes[1, 1].set(
        xlabel="input moment $(p,j)$",
        ylabel="output moment $(p,j)$",
        title="(d) Drift-kinetic Coulomb moment block",
    )
    fig.colorbar(image, ax=axes[1, 1], shrink=0.82, label="normalized collision rate")

    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.2, lw=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Linearized Coulomb operator: algebraic and numerical closure",
        fontsize=15,
        color=colors["ink"],
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, metadata={"Software": "SPECTRAX-GK"})
    plt.close(fig)


def write_coulomb_operator_verification_artifacts(
    out_json: Path,
    out_png: Path,
    *,
    digits: int = 80,
) -> dict[str, Any]:
    """Generate and persist the collision verification report and figure."""

    summary = build_coulomb_operator_verification_summary(digits=digits)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    write_coulomb_operator_verification_figure(summary, out_png)
    return summary


def build_drift_kinetic_response_convergence_summary(
    *,
    resolutions: tuple[tuple[int, int], ...] = (
        (3, 1),
        (5, 2),
        (7, 3),
        (9, 4),
        (11, 5),
        (15, 5),
        (20, 5),
    ),
    ion_charges: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0, 100.0),
    required_resolution: tuple[int, int] = (20, 5),
    nested_current_rtol: float = 5.0e-3,
    algebra_atol: float = 2.0e-12,
    original_sugama_low_charge_gap_min: float = 8.0e-2,
    original_sugama_high_charge_gap_max: float = 2.0e-2,
    improved_sugama_coulomb_gap_max: float = 1.0e-2,
    improved_sugama_correction_order: int = 5,
    paper_normalized_field: float = 1.0e-3,
    saturation_times: tuple[float, ...] = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0),
    saturation_charge: float = 1.0,
    saturation_rtol: float = 1.0e-3,
    field_linearity_rtol: float = 2.0e-12,
    spitzer_high_charge_minimum: float = 100.0,
    spitzer_high_charge_rtol: float = 8.0e-2,
    digits: int = 50,
) -> dict[str, Any]:
    r"""Build the converged driven Coulomb-response hierarchy.

    Electron--electron and electron--ion blocks use equations (3.53)--(3.56)
    of Frei et al. (2021). The electric-field source follows equation (81) of
    Frei, Ernst & Ricci (2022). The report verifies the matrix algebra and
    velocity-space convergence of the resulting current; it is deliberately
    not labelled electrical conductivity because that comparison also needs
    the paper's collision-frequency normalization and full original/improved
    Sugama hierarchies.
    """

    if not resolutions:
        raise ValueError("resolutions must contain at least one (P, J) pair")
    if any(p < 2 or j < 1 for p, j in resolutions):
        raise ValueError("response resolutions require P >= 2 and J >= 1")
    if required_resolution[0] < 2 or required_resolution[1] < 1:
        raise ValueError("required_resolution must satisfy P >= 2 and J >= 1")
    if any(not math.isfinite(z) or z <= 0.0 for z in ion_charges):
        raise ValueError("ion_charges must be finite and > 0")
    if any(right <= left for left, right in zip(ion_charges, ion_charges[1:])):
        raise ValueError("ion_charges must increase strictly")
    if nested_current_rtol <= 0.0 or not math.isfinite(nested_current_rtol):
        raise ValueError("nested_current_rtol must be finite and > 0")
    if algebra_atol <= 0.0 or not math.isfinite(algebra_atol):
        raise ValueError("algebra_atol must be finite and > 0")
    if not 0.0 < original_sugama_low_charge_gap_min < 1.0:
        raise ValueError("original_sugama_low_charge_gap_min must lie in (0, 1)")
    if not 0.0 < original_sugama_high_charge_gap_max < 1.0:
        raise ValueError("original_sugama_high_charge_gap_max must lie in (0, 1)")
    if not 0.0 < improved_sugama_coulomb_gap_max < 1.0:
        raise ValueError("improved_sugama_coulomb_gap_max must lie in (0, 1)")
    if improved_sugama_correction_order < 1:
        raise ValueError("improved_sugama_correction_order must be >= 1")
    if paper_normalized_field <= 0.0 or not math.isfinite(paper_normalized_field):
        raise ValueError("paper_normalized_field must be finite and > 0")
    if (
        not saturation_times
        or saturation_times[0] != 0.0
        or any(
            not math.isfinite(value) or value < 0.0
            for value in saturation_times
        )
        or any(
            right <= left
            for left, right in zip(saturation_times, saturation_times[1:])
        )
    ):
        raise ValueError("saturation_times must increase strictly from zero")
    if saturation_charge not in ion_charges:
        raise ValueError("saturation_charge must be present in ion_charges")
    for name, value in (
        ("saturation_rtol", saturation_rtol),
        ("field_linearity_rtol", field_linearity_rtol),
        ("spitzer_high_charge_rtol", spitzer_high_charge_rtol),
    ):
        if value <= 0.0 or not math.isfinite(value):
            raise ValueError(f"{name} must be finite and > 0")
    if spitzer_high_charge_minimum <= 0.0 or not math.isfinite(
        spitzer_high_charge_minimum
    ):
        raise ValueError("spitzer_high_charge_minimum must be finite and > 0")

    charge_values = np.asarray(ion_charges, dtype=float)
    solver_normalized_field = paper_normalized_field / np.sqrt(2.0)
    spitzer_conductivity = 64.0 / (
        3.0 * 2.0**1.5 * np.pi * charge_values
    )
    rows: list[dict[str, Any]] = []
    previous_current: np.ndarray | None = None
    previous_improved_current: np.ndarray | None = None
    resolution_reports: list[dict[str, Any]] = []
    saturation_report: dict[str, Any] = {}
    for maximum_hermite, maximum_laguerre in resolutions:
        maximum_degree = maximum_hermite + 2 * maximum_laguerre
        radial_limit = maximum_degree // 2
        pair_blocks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        pair_times: dict[str, float] = {}
        for label, mass_ratio in (("electron_electron", 1.0), ("electron_ion", 1 / 1836)):
            start = time.perf_counter()
            pair_blocks[label] = coulomb_drift_kinetic_moment_matrices(
                maximum_hermite,
                maximum_laguerre,
                mass_ratio,
                1.0,
                maximum_spherical_order=maximum_degree,
                maximum_spherical_radial_order=radial_limit,
                digits=digits,
            )
            pair_times[label] = time.perf_counter() - start

        n_laguerre = maximum_laguerre + 1
        mode_count = (maximum_hermite + 1) * n_laguerre
        convention_sign = np.asarray(
            [
                (-1.0) ** laguerre
                for _hermite in range(maximum_hermite + 1)
                for laguerre in range(n_laguerre)
            ]
        )

        electron_test, electron_field = pair_blocks["electron_electron"]
        ion_test, _ion_field = pair_blocks["electron_ion"]
        _original_test, original_field = (
            original_sugama_like_species_moment_matrices(
                electron_test,
                maximum_hermite,
                maximum_laguerre,
            )
        )
        effective_correction_order = min(
            improved_sugama_correction_order,
            maximum_laguerre,
            (maximum_hermite - 1) // 2,
        )
        _improved_test, improved_field = (
            improved_sugama_equal_temperature_moment_matrices(
                electron_test,
                maximum_hermite,
                maximum_laguerre,
                correction_order=effective_correction_order,
                digits=digits,
            )
        )
        convention = convention_sign[:, None] * convention_sign[None, :]
        electron_collision = (
            convention * (electron_test + electron_field)
        )
        original_electron_collision = convention * (electron_test + original_field)
        improved_electron_collision = convention * (electron_test + improved_field)
        previous_order_improved_collision: np.ndarray | None = None
        if (
            (maximum_hermite, maximum_laguerre) == resolutions[-1]
            and effective_correction_order >= 2
        ):
            _previous_test, previous_field = (
                improved_sugama_equal_temperature_moment_matrices(
                    electron_test,
                    maximum_hermite,
                    maximum_laguerre,
                    correction_order=effective_correction_order - 1,
                    digits=digits,
                )
            )
            previous_order_improved_collision = convention * (
                electron_test + previous_field
            )
        ion_collision = convention * ion_test
        active_modes = np.asarray(
            [
                hermite * n_laguerre + laguerre
                for hermite in range(1, maximum_hermite + 1, 2)
                for laguerre in range(n_laguerre)
            ],
            dtype=int,
        )
        source = np.zeros(mode_count)
        source[n_laguerre] = -np.sqrt(2.0) * solver_normalized_field
        currents: list[float] = []
        original_currents: list[float] = []
        improved_currents: list[float] = []
        previous_order_improved_currents: list[float] = []
        solve_residuals: list[float] = []
        original_solve_residuals: list[float] = []
        improved_solve_residuals: list[float] = []
        for charge in charge_values:
            collision = electron_collision + charge * ion_collision
            original_collision = original_electron_collision + charge * ion_collision
            improved_collision = improved_electron_collision + charge * ion_collision
            previous_order_collision = (
                None
                if previous_order_improved_collision is None
                else previous_order_improved_collision + charge * ion_collision
            )
            response = np.zeros(mode_count)
            original_response = np.zeros(mode_count)
            improved_response = np.zeros(mode_count)
            response[active_modes] = np.linalg.solve(
                collision[np.ix_(active_modes, active_modes)],
                -source[active_modes],
            )
            original_response[active_modes] = np.linalg.solve(
                original_collision[np.ix_(active_modes, active_modes)],
                -source[active_modes],
            )
            improved_response[active_modes] = np.linalg.solve(
                improved_collision[np.ix_(active_modes, active_modes)],
                -source[active_modes],
            )
            currents.append(float(abs(response[n_laguerre] / np.sqrt(2.0))))
            original_currents.append(
                float(abs(original_response[n_laguerre] / np.sqrt(2.0)))
            )
            improved_currents.append(
                float(abs(improved_response[n_laguerre] / np.sqrt(2.0)))
            )
            if previous_order_collision is not None:
                previous_order_response = np.zeros(mode_count)
                previous_order_response[active_modes] = np.linalg.solve(
                    previous_order_collision[np.ix_(active_modes, active_modes)],
                    -source[active_modes],
                )
                previous_order_improved_currents.append(
                    float(abs(previous_order_response[n_laguerre] / np.sqrt(2.0)))
                )
            solve_residuals.append(
                float(np.max(np.abs((collision @ response + source)[active_modes])))
            )
            original_solve_residuals.append(
                float(
                    np.max(
                        np.abs(
                            (original_collision @ original_response + source)[
                                active_modes
                            ]
                        )
                    )
                )
            )
            improved_solve_residuals.append(
                float(
                    np.max(
                        np.abs(
                            (improved_collision @ improved_response + source)[
                                active_modes
                            ]
                        )
                    )
                )
            )

        current_array = np.asarray(currents)
        original_current_array = np.asarray(original_currents)
        improved_current_array = np.asarray(improved_currents)
        correction_order_relative_change = (
            None
            if not previous_order_improved_currents
            else np.abs(
                improved_current_array - np.asarray(previous_order_improved_currents)
            )
            / np.maximum(np.abs(improved_current_array), np.finfo(float).tiny)
        )
        relative_change = (
            None
            if previous_current is None
            else np.abs(current_array - previous_current)
            / np.maximum(np.abs(current_array), np.finfo(float).tiny)
        )
        previous_current = current_array
        improved_relative_change = (
            None
            if previous_improved_current is None
            else np.abs(improved_current_array - previous_improved_current)
            / np.maximum(np.abs(improved_current_array), np.finfo(float).tiny)
        )
        previous_improved_current = improved_current_array

        density = np.zeros(mode_count)
        density[0] = 1.0
        momentum = np.zeros(mode_count)
        momentum[n_laguerre] = 1.0
        energy = np.zeros(mode_count)
        if maximum_laguerre >= 1 and maximum_hermite >= 2:
            energy[1] = 1.0
            energy[2 * n_laguerre] = 1 / np.sqrt(2.0)
        invariant_residuals = {
            name: float(np.max(np.abs(electron_collision @ vector)))
            for name, vector in (
                ("density", density),
                ("parallel_momentum", momentum),
                ("thermal_energy", energy),
            )
        }
        original_invariant_residuals = {
            name: float(np.max(np.abs(original_electron_collision @ vector)))
            for name, vector in (
                ("density", density),
                ("parallel_momentum", momentum),
                ("thermal_energy", energy),
            )
        }
        improved_invariant_residuals = {
            name: float(np.max(np.abs(improved_electron_collision @ vector)))
            for name, vector in (
                ("density", density),
                ("parallel_momentum", momentum),
                ("thermal_energy", energy),
            )
        }
        collision_models = (
            electron_collision,
            original_electron_collision,
            improved_electron_collision,
        )
        symmetry_error = max(
            float(np.max(np.abs(matrix - matrix.T)))
            for matrix in collision_models
        )
        maximum_eigenvalue = max(
            float(np.linalg.eigvalsh(0.5 * (matrix + matrix.T)).max())
            for matrix in collision_models
        )
        maximum_change = (
            None if relative_change is None else float(np.max(relative_change))
        )
        if (maximum_hermite, maximum_laguerre) == resolutions[-1]:
            charge = float(saturation_charge)
            time_values = np.asarray(saturation_times, dtype=float)
            paper_field_values = paper_normalized_field * np.asarray(
                (0.1, 1.0, 10.0), dtype=float
            )
            current_mode = int(np.flatnonzero(active_modes == n_laguerre)[0])
            model_matrices = {
                "coulomb": electron_collision + charge * ion_collision,
                "original_sugama": (
                    original_electron_collision + charge * ion_collision
                ),
                "improved_sugama": (
                    improved_electron_collision + charge * ion_collision
                ),
            }
            saturation_models: dict[str, Any] = {}
            for model, collision_matrix in model_matrices.items():
                reduced = collision_matrix[np.ix_(active_modes, active_modes)]
                reduced = 0.5 * (reduced + reduced.T)
                steady = np.linalg.solve(reduced, -source[active_modes])
                eigenvalues, eigenvectors = np.linalg.eigh(reduced)
                transient_coefficients = eigenvectors.T @ steady
                transient = (
                    np.exp(time_values[:, None] * eigenvalues[None, :])
                    * transient_coefficients[None, :]
                ) @ eigenvectors.T
                trajectory = steady[None, :] - transient
                current_trace = trajectory[:, current_mode] / np.sqrt(2.0)
                stationary_current = float(steady[current_mode] / np.sqrt(2.0))

                conductivity_values = []
                for paper_field in paper_field_values:
                    solver_field = paper_field / np.sqrt(2.0)
                    drive = np.zeros_like(source[active_modes])
                    drive[current_mode] = -np.sqrt(2.0) * solver_field
                    field_response = np.linalg.solve(reduced, -drive)
                    conductivity_values.append(
                        float(
                            abs(field_response[current_mode] / np.sqrt(2.0))
                            / solver_field
                        )
                    )
                conductivity_array = np.asarray(conductivity_values)
                reference_conductivity = conductivity_array[1]
                saturation_models[model] = {
                    "current_over_vte": current_trace.tolist(),
                    "stationary_current_over_vte": stationary_current,
                    "saturation_relative_error": float(
                        abs(current_trace[-1] - stationary_current)
                        / max(abs(stationary_current), np.finfo(float).tiny)
                    ),
                    "conductivity_over_ne2_mnu": conductivity_array.tolist(),
                    "field_linearity_relative_error": float(
                        np.max(
                            np.abs(
                                conductivity_array / reference_conductivity - 1.0
                            )
                        )
                    ),
                }
            saturation_report = {
                "ion_charge": charge,
                "time_nu_ee": time_values.tolist(),
                "paper_normalized_field": paper_normalized_field,
                "solver_normalized_field": solver_normalized_field,
                "paper_field_scan": paper_field_values.tolist(),
                "models": saturation_models,
                "maximum_saturation_relative_error": max(
                    row["saturation_relative_error"]
                    for row in saturation_models.values()
                ),
                "maximum_field_linearity_relative_error": max(
                    row["field_linearity_relative_error"]
                    for row in saturation_models.values()
                ),
            }
        report = {
            "maximum_hermite_order": maximum_hermite,
            "maximum_laguerre_order": maximum_laguerre,
            "mode_count": mode_count,
            "maximum_spherical_order": maximum_degree,
            "maximum_spherical_radial_order": radial_limit,
            "improved_sugama_correction_order": effective_correction_order,
            "current": current_array.tolist(),
            "original_sugama_current": original_current_array.tolist(),
            "original_sugama_relative_gap": (
                (current_array - original_current_array)
                / np.maximum(np.abs(current_array), np.finfo(float).tiny)
            ).tolist(),
            "improved_sugama_current": improved_current_array.tolist(),
            "improved_sugama_relative_gap": (
                (improved_current_array - current_array)
                / np.maximum(np.abs(current_array), np.finfo(float).tiny)
            ).tolist(),
            "conductivity_over_ne2_mnu": (
                current_array / solver_normalized_field
            ).tolist(),
            "original_sugama_conductivity_over_ne2_mnu": (
                original_current_array / solver_normalized_field
            ).tolist(),
            "improved_sugama_conductivity_over_ne2_mnu": (
                improved_current_array / solver_normalized_field
            ).tolist(),
            "spitzer_relative_error": (
                np.abs(current_array / solver_normalized_field - spitzer_conductivity)
                / spitzer_conductivity
            ).tolist(),
            "relative_change_from_previous": (
                None if relative_change is None else relative_change.tolist()
            ),
            "maximum_relative_change": maximum_change,
            "improved_relative_change_from_previous": (
                None
                if improved_relative_change is None
                else improved_relative_change.tolist()
            ),
            "improved_maximum_relative_change": (
                None
                if improved_relative_change is None
                else float(np.max(improved_relative_change))
            ),
            "improved_correction_order_relative_change": (
                None
                if correction_order_relative_change is None
                else correction_order_relative_change.tolist()
            ),
            "improved_correction_order_maximum_change": (
                None
                if correction_order_relative_change is None
                else float(np.max(correction_order_relative_change))
            ),
            "invariant_residuals": invariant_residuals,
            "original_sugama_invariant_residuals": original_invariant_residuals,
            "improved_sugama_invariant_residuals": improved_invariant_residuals,
            "symmetry_max_abs": symmetry_error,
            "maximum_eigenvalue": maximum_eigenvalue,
            "solve_residual_max": float(
                max(
                    (
                        *solve_residuals,
                        *original_solve_residuals,
                        *improved_solve_residuals,
                    )
                )
            ),
            "generation_seconds": pair_times,
        }
        resolution_reports.append(report)
        for charge_index, charge in enumerate(charge_values):
            rows.append(
                {
                    "maximum_hermite_order": maximum_hermite,
                    "maximum_laguerre_order": maximum_laguerre,
                    "mode_count": mode_count,
                    "ion_charge": float(charge),
                    "normalized_current": float(current_array[charge_index]),
                    "original_sugama_normalized_current": float(
                        original_current_array[charge_index]
                    ),
                    "improved_sugama_normalized_current": float(
                        improved_current_array[charge_index]
                    ),
                    "conductivity_over_ne2_mnu": float(
                        current_array[charge_index] / solver_normalized_field
                    ),
                    "spitzer_conductivity_over_ne2_mnu": float(
                        spitzer_conductivity[charge_index]
                    ),
                    "relative_change_from_previous": (
                        None
                        if relative_change is None
                        else float(relative_change[charge_index])
                    ),
                }
            )

    final = resolution_reports[-1]
    resolution_reached = (
        final["maximum_hermite_order"] >= required_resolution[0]
        and final["maximum_laguerre_order"] >= required_resolution[1]
    )
    high_charge_index = int(np.argmax(charge_values))
    gates = {
        "required_resolution_reached": resolution_reached,
        "nested_current_converged": (
            final["maximum_relative_change"] is not None
            and final["maximum_relative_change"] <= nested_current_rtol
        ),
        "collision_invariants": max(final["invariant_residuals"].values())
        <= algebra_atol,
        "original_sugama_invariants": max(
            final["original_sugama_invariant_residuals"].values()
        )
        <= algebra_atol,
        "improved_sugama_invariants": max(
            final["improved_sugama_invariant_residuals"].values()
        )
        <= algebra_atol,
        "self_adjoint_symmetry": final["symmetry_max_abs"] <= algebra_atol,
        "nonpositive_spectrum": final["maximum_eigenvalue"] <= algebra_atol,
        "driven_solve_residual": final["solve_residual_max"] <= algebra_atol,
        "original_sugama_low_charge_underprediction": final[
            "original_sugama_relative_gap"
        ][0]
        >= original_sugama_low_charge_gap_min,
        "original_sugama_high_charge_convergence": final[
            "original_sugama_relative_gap"
        ][-1]
        <= original_sugama_high_charge_gap_max,
        "improved_sugama_order_reached": final[
            "improved_sugama_correction_order"
        ]
        >= improved_sugama_correction_order,
        "improved_sugama_nested_current_converged": final[
            "improved_maximum_relative_change"
        ]
        is not None
        and final["improved_maximum_relative_change"] <= nested_current_rtol,
        "improved_sugama_correction_hierarchy_converged": final[
            "improved_correction_order_maximum_change"
        ]
        is not None
        and final["improved_correction_order_maximum_change"]
        <= nested_current_rtol,
        "improved_sugama_matches_coulomb": max(
            abs(value) for value in final["improved_sugama_relative_gap"]
        )
        <= improved_sugama_coulomb_gap_max,
        "spitzer_high_charge_asymptote": (
            charge_values[high_charge_index] >= spitzer_high_charge_minimum
            and final["spitzer_relative_error"][high_charge_index]
            <= spitzer_high_charge_rtol
        ),
        "stationary_current_saturated": saturation_report[
            "maximum_saturation_relative_error"
        ]
        <= saturation_rtol,
        "electric_field_linearity": saturation_report[
            "maximum_field_linearity_relative_error"
        ]
        <= field_linearity_rtol,
    }
    return {
        "schema_version": 4,
        "title": "Drift-kinetic Coulomb and Sugama conductivity convergence",
        "literature_equations": {
            "collision": "Frei et al. (2021), equations (3.53)--(3.56)",
            "original_sugama": "Sugama, Watanabe & Nunami (2009)",
            "improved_sugama": (
                "Frei, Ernst & Ricci (2022), equations (45), (60), (79)--(81)"
            ),
            "drive": "Frei, Ernst & Ricci (2022), equation (81)",
            "conductivity": "Frei, Ernst & Ricci (2022), equations (83)--(84) and Figures 15--16",
        },
        "scope": (
            "Converged Coulomb and arbitrary-order, equal-temperature original- and "
            "improved-Sugama conductivity, including dimensional normalization, "
            "electric-field linearity, stationary saturation, and the high-charge "
            "Spitzer asymptote."
        ),
        "ion_charge": charge_values.tolist(),
        "normalized_field": solver_normalized_field,
        "field_normalization": {
            "paper": "e E / (sqrt(m_e T_e) nu_ee)",
            "paper_value": paper_normalized_field,
            "solver": "e E / (m_e v_Te nu_ee)",
            "solver_value": solver_normalized_field,
            "thermal_speed": "v_Te = sqrt(2 T_e / m_e)",
        },
        "conductivity_normalization": {
            "computed": "sigma_parallel / (n_e e^2 / (m_e nu_ee))",
            "computed_value": (
                np.asarray(final["conductivity_over_ne2_mnu"], dtype=float)
            ).tolist(),
            "spitzer_high_charge": (
                "64 / (3 * 2^(3/2) * pi * Z)"
            ),
            "spitzer_value": spitzer_conductivity.tolist(),
            "high_charge_relative_error": final["spitzer_relative_error"][
                high_charge_index
            ],
        },
        "saturation": saturation_report,
        "required_resolution": list(required_resolution),
        "thresholds": {
            "nested_current_rtol": nested_current_rtol,
            "algebra_atol": algebra_atol,
            "original_sugama_low_charge_gap_min": (
                original_sugama_low_charge_gap_min
            ),
            "original_sugama_high_charge_gap_max": (
                original_sugama_high_charge_gap_max
            ),
            "improved_sugama_coulomb_gap_max": improved_sugama_coulomb_gap_max,
            "improved_sugama_correction_order": improved_sugama_correction_order,
            "saturation_rtol": saturation_rtol,
            "field_linearity_rtol": field_linearity_rtol,
            "spitzer_high_charge_minimum": spitzer_high_charge_minimum,
            "spitzer_high_charge_rtol": spitzer_high_charge_rtol,
        },
        "resolutions": resolution_reports,
        "rows": rows,
        "gates": gates,
        "gate_passed": all(gates.values()),
    }


def write_drift_kinetic_response_convergence_figure(
    summary: dict[str, Any], out_png: Path
) -> None:
    """Write the publication panel for the driven-response hierarchy."""

    colors = {
        "navy": "#17324D",
        "blue": "#2878A8",
        "teal": "#16817A",
        "orange": "#D97732",
        "red": "#B33A3A",
        "ink": "#20272E",
    }
    reports = summary["resolutions"]
    final_report = reports[-1]
    charges = np.asarray(summary["ion_charge"], dtype=float)
    mode_counts = np.asarray([row["mode_count"] for row in reports], dtype=float)
    labels = [
        f"({row['maximum_hermite_order']}, {row['maximum_laguerre_order']})"
        for row in reports
    ]
    palette = plt.get_cmap("viridis")(np.linspace(0.15, 0.88, len(reports)))
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.0), constrained_layout=True)

    for color, label, row in zip(palette, labels, reports, strict=True):
        axes[0, 0].semilogx(
            charges,
            np.asarray(row["current"], dtype=float) / summary["normalized_field"],
            "o-",
            color=color,
            lw=1.7,
            ms=4.2,
            label=label,
        )
    axes[0, 0].semilogx(
        charges,
        np.asarray(final_report["original_sugama_current"], dtype=float)
        / summary["normalized_field"],
        "^--",
        color=colors["ink"],
        markerfacecolor="white",
        lw=1.8,
        ms=5.0,
        label=(
            "original Sugama "
            f"({final_report['maximum_hermite_order']}, "
            f"{final_report['maximum_laguerre_order']})"
        ),
    )
    axes[0, 0].semilogx(
        charges,
        np.asarray(final_report["improved_sugama_current"], dtype=float)
        / summary["normalized_field"],
        "x-.",
        color=colors["red"],
        lw=1.8,
        ms=5.0,
        markeredgewidth=1.5,
        label=(
            "improved Sugama "
            f"({final_report['maximum_hermite_order']}, "
            f"{final_report['maximum_laguerre_order']}; "
            f"K={final_report['improved_sugama_correction_order']})"
        ),
    )
    axes[0, 0].semilogx(
        charges,
        np.asarray(
            summary["conductivity_normalization"]["spitzer_value"], dtype=float
        ),
        "o:",
        color=colors["ink"],
        markerfacecolor="white",
        lw=1.5,
        ms=4.4,
        label="Spitzer high-$Z$ asymptote",
    )
    axes[0, 0].set(
        xlabel="ion charge $Z$",
        ylabel=r"$\sigma_\parallel/[n_e e^2/(m_e\nu_{ee})]$",
        title="(a) Conductivity hierarchy",
    )
    axes[0, 0].legend(
        title=r"$(P,J)$", frameon=False, fontsize=6.8, title_fontsize=8, ncol=3
    )

    for charge_index, charge in enumerate(charges):
        x_values: list[float] = []
        y_values: list[float] = []
        for row in reports[1:]:
            change = row["relative_change_from_previous"]
            if change is not None:
                x_values.append(float(row["mode_count"]))
                y_values.append(max(float(change[charge_index]), 1.0e-16))
        axes[0, 1].semilogy(
            x_values,
            y_values,
            "o-",
            lw=1.5,
            ms=4.0,
            label=rf"$Z={charge:g}$",
        )
    axes[0, 1].axhline(
        summary["thresholds"]["nested_current_rtol"],
        color=colors["red"],
        ls="--",
        lw=1.4,
        label="acceptance",
    )
    axes[0, 1].set(
        xlabel="retained Hermite--Laguerre modes",
        ylabel="nested relative current change",
        title="(b) Velocity-space convergence",
    )
    axes[0, 1].legend(frameon=False, fontsize=7, ncol=2)

    invariant = np.asarray(
        [
            max(
                *row["invariant_residuals"].values(),
                *row["original_sugama_invariant_residuals"].values(),
                *row["improved_sugama_invariant_residuals"].values(),
            )
            for row in reports
        ]
    )
    symmetry = np.asarray([row["symmetry_max_abs"] for row in reports])
    positive_eigenvalue = np.maximum(
        np.asarray([row["maximum_eigenvalue"] for row in reports]), 0.0
    )
    solve_residual = np.asarray([row["solve_residual_max"] for row in reports])
    for values, marker, label, color in (
        (invariant, "o", "invariants", colors["blue"]),
        (symmetry, "s", "self-adjoint symmetry", colors["orange"]),
        (positive_eigenvalue, "^", "positive spectrum", colors["red"]),
        (solve_residual, "D", "forced solve", colors["teal"]),
    ):
        axes[1, 0].semilogy(
            mode_counts,
            np.maximum(values, 1.0e-18),
            marker + "-",
            lw=1.5,
            ms=4.2,
            color=color,
            label=label,
        )
    axes[1, 0].axhline(
        summary["thresholds"]["algebra_atol"],
        color=colors["ink"],
        ls="--",
        lw=1.2,
        label="acceptance",
    )
    axes[1, 0].set(
        xlabel="retained Hermite--Laguerre modes",
        ylabel="absolute residual",
        title="(c) Conservation and dissipation gates",
    )
    axes[1, 0].legend(frameon=False, fontsize=7, ncol=2)

    saturation = summary["saturation"]
    time_values = np.asarray(saturation["time_nu_ee"], dtype=float)
    for model, label, style, color in (
        ("coulomb", "Coulomb", "-", colors["blue"]),
        ("original_sugama", "original Sugama", "--", colors["ink"]),
        ("improved_sugama", "improved Sugama $K=5$", "-.", colors["red"]),
    ):
        axes[1, 1].plot(
            time_values,
            1.0e3
            * np.asarray(saturation["models"][model]["current_over_vte"]),
            style,
            color=color,
            lw=1.8,
            label=label,
        )
    axes[1, 1].set(
        xlabel=r"$t\nu_{ee}$",
        ylabel=r"$10^3 u_e/v_{Te}$",
        title="(d) Stationary-current saturation ($Z=1$)",
    )
    axes[1, 1].legend(frameon=False, fontsize=8)

    for axis in axes.flat:
        axis.grid(alpha=0.22, lw=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    status = "PASS" if summary["gate_passed"] else "OPEN"
    fig.suptitle(
        f"Drift-kinetic Coulomb/Sugama conductivity: resolved hierarchies ({status})",
        fontsize=15,
        color=colors["ink"],
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, metadata={"Software": "SPECTRAX-GK"})
    fig.savefig(out_png.with_suffix(".pdf"), metadata={"Creator": "SPECTRAX-GK"})
    plt.close(fig)


def write_drift_kinetic_response_convergence_artifacts(
    out_json: Path,
    out_csv: Path,
    out_png: Path,
    **summary_kwargs: Any,
) -> dict[str, Any]:
    """Generate compact machine-readable and publication response artifacts."""

    summary = build_drift_kinetic_response_convergence_summary(**summary_kwargs)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary["rows"]).to_csv(out_csv, index=False)
    write_drift_kinetic_response_convergence_figure(summary, out_png)
    return summary


def build_collision_table(*, digits: int = 80) -> np.ndarray:
    """Generate the published C6/C9/103 matrices with multiprecision arithmetic."""

    import mpmath as mp

    with mp.workdps(digits):
        inverse_sqrt_pi = 1 / mp.sqrt(mp.pi)
        sqrt_two = mp.sqrt(2)
        sqrt_three = mp.sqrt(3)
        blocks = (
            (
                (-64 * sqrt_two / 45, 64 / 45, -32 * sqrt_two / 45),
                (
                    -361 * sqrt_two / 175,
                    208 / (175 * sqrt_three),
                    -1187 * sqrt_two / 525,
                ),
            ),
            (
                (-16 * sqrt_two / 15, 16 / 15, -8 * sqrt_two / 15),
                (-8 * sqrt_two / 5, 8 / (5 * sqrt_three), -28 * sqrt_two / 15),
            ),
        )
        matrices = np.zeros((3, 8, 8), dtype=np.float64)
        temperature_modes = (4, 1)
        heat_modes = (6, 3)
        for model, (thermal, heat) in zip((0, 2), blocks):
            for modes, coefficients in (
                (temperature_modes, thermal),
                (heat_modes, heat),
            ):
                row0, row1 = modes
                diagonal0, coupling, diagonal1 = coefficients
                matrices[model, row0, row0] = float(diagonal0 * inverse_sqrt_pi)
                matrices[model, row0, row1] = float(coupling * inverse_sqrt_pi)
                matrices[model, row1, row0] = float(coupling * inverse_sqrt_pi)
                matrices[model, row1, row1] = float(diagonal1 * inverse_sqrt_pi)
        matrices[1] = matrices[0]
        matrices[1, 6, 6] += float(mp.mpf(9) / 25 * mp.sqrt(2 / mp.pi))
        improved_heat_coupling = float(mp.mpf(6) / 25 * mp.sqrt(3 / mp.pi))
        matrices[1, 6, 3] += improved_heat_coupling
        matrices[1, 3, 6] += improved_heat_coupling
        matrices[1, 3, 3] += float(mp.mpf(6) / 25 * mp.sqrt(2 / mp.pi))
    return matrices


def write_collision_table(
    out: Path, metadata_out: Path, *, digits: int = 80
) -> dict[str, Any]:
    matrices = build_collision_table(digits=digits)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as stream:
        np.save(stream, matrices, allow_pickle=False)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    metadata = {
        "kind": "spectraxgk_collision_moment_coefficients",
        "models": ["sugama", "improved_sugama", "coulomb"],
        "shape": list(matrices.shape),
        "dtype": str(matrices.dtype),
        "sha256": digest,
        "precision_decimal_digits": int(digits),
        "moment_order": "hermite_major_index=p*Nl+j",
        "Nl": 2,
        "Nm": 4,
        "laguerre_convention": "spectraxgk_opposite_to_paper",
        "source": "Frei, Ernst & Ricci (2022), arXiv:2202.06293",
        "equations": {
            "sugama": "C6a-C6f",
            "improved_sugama": "C6a-C6f plus C103a-C103c",
            "coulomb": "C9a-C9f",
        },
        "claim_scope": "validated_drift_kinetic_like_species_low_order_vertical_slice",
    }
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def build_collision_table_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate collision coefficient tables."
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_COLLISION_TABLE)
    parser.add_argument("--metadata-out", type=Path, default=DEFAULT_COLLISION_METADATA)
    parser.add_argument("--digits", type=int, default=80)
    return parser


def build_collision_verification_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the finite-b Coulomb operator verification artifacts."
    )
    parser.add_argument(
        "--out-json", type=Path, default=DEFAULT_COLLISION_VERIFICATION_JSON
    )
    parser.add_argument(
        "--out-png", type=Path, default=DEFAULT_COLLISION_VERIFICATION_PNG
    )
    parser.add_argument("--digits", type=int, default=80)
    return parser


def build_collision_response_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the drift-kinetic Coulomb response convergence artifacts."
    )
    parser.add_argument("--out-json", type=Path, default=DEFAULT_COLLISION_RESPONSE_JSON)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_COLLISION_RESPONSE_CSV)
    parser.add_argument("--out-png", type=Path, default=DEFAULT_COLLISION_RESPONSE_PNG)
    parser.add_argument("--digits", type=int, default=50)
    return parser


def build_figures_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate linear validation figures.")
    parser.add_argument(
        "--case",
        choices=["all", "cyclone", "etg"],
        default="all",
        help="Limit figure generation to a specific case.",
    )
    parser.add_argument(
        "--no-progress", action="store_true", help="Disable progress bars."
    )
    return parser


def _load_spectrax_scan_from_mismatch(
    csv_path: Path, *, x_col: str = "ky"
) -> LinearScanResult:
    from spectraxgk.benchmarking.shared import LinearScanResult

    table = pd.read_csv(csv_path).sort_values(x_col)
    return LinearScanResult(
        ky=table[x_col].to_numpy(dtype=float),
        gamma=table["gamma_spectrax"].to_numpy(dtype=float),
        omega=table["omega_spectrax"].to_numpy(dtype=float),
    )


def _cyclone_refresh_reference(ref: LinearScanResult) -> LinearScanResult:
    from spectraxgk.benchmarking.shared import LinearScanResult

    keep = np.asarray(ref.ky) <= 0.45 + 1.0e-12
    return LinearScanResult(
        ky=np.asarray(ref.ky)[keep],
        gamma=np.asarray(ref.gamma)[keep],
        omega=np.asarray(ref.omega)[keep],
    )


def _run_etg_figures(*, outdir: Path, progress: bool) -> None:
    from spectraxgk.artifacts.plotting import scan_comparison_figure
    from spectraxgk.benchmarking.shared import load_etg_reference
    from spectraxgk.runtime import run_runtime_scan
    from spectraxgk.workflows.runtime.toml import load_runtime_from_toml

    reference = load_etg_reference()
    mismatch_csv = outdir / "etg_mismatch_table.csv"
    if mismatch_csv.exists():
        scan = _load_spectrax_scan_from_mismatch(mismatch_csv)
    else:
        config, _ = load_runtime_from_toml(
            REPO_ROOT / "examples/linear/axisymmetric/etg.toml"
        )
        scan = run_runtime_scan(
            config,
            np.asarray(reference.ky),
            Nl=24,
            Nm=8,
            solver="time",
            batch_ky=True,
            method=config.time.method,
            dt=config.time.dt,
            steps=int(round(config.time.t_max / config.time.dt)),
            sample_stride=config.time.sample_stride,
            auto_window=False,
            tmin=1.0,
            tmax=config.time.t_max,
            fit_signal="phi",
            mode_method="z_index",
            show_progress=progress,
        )
    fig, _axes = scan_comparison_figure(
        scan.ky,
        scan.gamma,
        scan.omega,
        r"$k_y \rho_i$",
        "ETG Benchmark Scan",
        x_ref=reference.ky,
        gamma_ref=reference.gamma,
        omega_ref=reference.omega,
        label="SPECTRAX-GK",
        ref_label="Reference",
        log_x=True,
    )
    fig.savefig(outdir / "etg_comparison.png", dpi=200)
    fig.savefig(outdir / "etg_comparison.pdf")


def _run_kbm_figures(*, outdir: Path) -> None:
    from spectraxgk.artifacts.plotting import scan_comparison_figure
    from spectraxgk.benchmarking.shared import load_kbm_reference

    reference = load_kbm_reference()
    mismatch_csv = outdir / "kbm_mismatch_table.csv"
    if not mismatch_csv.exists():
        raise FileNotFoundError(
            f"missing {mismatch_csv}; generate the KBM mismatch table first"
        )
    scan = _load_spectrax_scan_from_mismatch(mismatch_csv)
    fig, _axes = scan_comparison_figure(
        scan.ky,
        scan.gamma,
        scan.omega,
        r"$\beta$",
        "KBM Benchmark Scan",
        x_ref=reference.ky,
        gamma_ref=reference.gamma,
        omega_ref=reference.omega,
        label="SPECTRAX-GK",
        ref_label="Reference",
        log_x=False,
    )
    fig.savefig(outdir / "kbm_comparison.png", dpi=200)
    fig.savefig(outdir / "kbm_comparison.pdf")


def main_figures(argv: list[str] | None = None) -> int:
    from spectraxgk.artifacts.plotting import (
        cyclone_comparison_figure,
        cyclone_reference_figure,
    )
    from spectraxgk.benchmarking.shared import load_cyclone_reference

    args = build_figures_parser().parse_args(argv)
    outdir = REPO_ROOT / "docs" / "_static"
    outdir.mkdir(parents=True, exist_ok=True)
    progress = not args.no_progress
    if args.case == "etg":
        _run_etg_figures(outdir=outdir, progress=progress)
        return 0

    reference = _cyclone_refresh_reference(load_cyclone_reference())
    fig, _axes = cyclone_reference_figure(reference)
    fig.savefig(outdir / "cyclone_reference.png", dpi=200)
    fig.savefig(outdir / "cyclone_reference.pdf")
    scan = _load_spectrax_scan_from_mismatch(outdir / "cyclone_mismatch_table.csv")
    fig, _axes = cyclone_comparison_figure(reference, scan)
    fig.savefig(outdir / "cyclone_comparison.png", dpi=200)
    fig.savefig(outdir / "cyclone_comparison.pdf")
    if args.case == "cyclone":
        return 0

    _run_etg_figures(outdir=outdir, progress=progress)
    _run_kbm_figures(outdir=outdir)
    return 0


def _json_clean(value: Any) -> Any:
    """Return a strict-JSON-compatible copy with nonfinite numbers set to null."""

    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [_json_clean(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, np.generic):
        return _json_clean(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def load_convergence_series(
    csv_path: Path,
    *,
    step_column: str | None,
    resolution_column: str | None,
    error_column: str,
    absolute_error: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    """Load effective step sizes and errors from a convergence table."""

    if (step_column is None) == (resolution_column is None):
        raise ValueError("Specify exactly one of step_column or resolution_column.")
    table = pd.read_csv(csv_path)
    required = {error_column, step_column or resolution_column}
    missing = sorted(col for col in required if col not in table.columns)
    if missing:
        raise ValueError(
            f"{csv_path} is missing required columns: {', '.join(missing)}"
        )

    if step_column is not None:
        h = np.asarray(table[step_column], dtype=float)
        step_source = step_column
    else:
        resolution = np.asarray(table[resolution_column], dtype=float)
        if np.any(resolution <= 0.0):
            raise ValueError("resolution values must be positive")
        h = 1.0 / resolution
        step_source = f"1/{resolution_column}"

    err = np.asarray(table[error_column], dtype=float)
    if absolute_error:
        err = np.abs(err)

    order = np.argsort(h)[::-1]
    h = h[order]
    err = err[order]
    selected = table.iloc[order].copy()
    selected["effective_step"] = h
    selected["error_for_gate"] = err
    selected["step_source"] = step_source
    rows = _json_clean(selected.to_dict(orient="records"))
    return h, err, rows


def build_summary(
    csv_path: Path,
    *,
    step_column: str | None,
    resolution_column: str | None,
    error_column: str,
    case: str,
    source: str,
    min_order: float,
    min_pairwise_order: float | None,
    max_final_error: float | None,
    absolute_error: bool = True,
) -> dict[str, object]:
    """Build the JSON payload for an observed-order convergence gate."""

    from spectraxgk.diagnostics.analysis import estimate_observed_order
    from spectraxgk.diagnostics.validation_gates import (
        gate_report_to_dict,
        observed_order_gate_report,
    )

    h, err, rows = load_convergence_series(
        csv_path,
        step_column=step_column,
        resolution_column=resolution_column,
        error_column=error_column,
        absolute_error=absolute_error,
    )
    metrics = estimate_observed_order(h, err)
    report = observed_order_gate_report(
        metrics,
        case=case,
        source=source,
        min_asymptotic_order=min_order,
        min_pairwise_order=min_pairwise_order,
        max_final_error=max_final_error,
    )
    payload = {
        "case": case,
        "source": source,
        "csv": str(csv_path),
        "error_column": error_column,
        "absolute_error": bool(absolute_error),
        "step_sizes": metrics.step_sizes.tolist(),
        "errors": metrics.errors.tolist(),
        "pairwise_orders": metrics.orders.tolist(),
        "asymptotic_order": metrics.asymptotic_order,
        "min_pairwise_order": float(np.min(metrics.orders)),
        "final_error": float(metrics.errors[-1]),
        "gate_report": gate_report_to_dict(report),
        "gate_passed": bool(report.passed),
        "rows": rows,
    }
    return _json_clean(payload)


def write_observed_order_plot(
    summary: dict[str, object],
    out_png: Path,
    *,
    title: str,
    min_order: float,
) -> None:
    """Write a log-log convergence panel for an observed-order gate."""

    h = np.asarray(summary["step_sizes"], dtype=float)
    err = np.asarray(summary["errors"], dtype=float)
    asymptotic_order = float(summary["asymptotic_order"])
    min_pairwise_order = float(summary["min_pairwise_order"])
    final_error = float(summary["final_error"])
    gate_status = "passed" if bool(summary["gate_passed"]) else "open"

    fig, ax = plt.subplots(figsize=(5.2, 3.7), constrained_layout=True)
    ax.loglog(h, err, "o-", color="#1b6ca8", lw=2.0, ms=6, label="measured error")
    ref = err[-1] * (h / h[-1]) ** float(min_order)
    ax.loglog(h, ref, "--", color="#8f2d2d", lw=1.8, label=f"order {min_order:g} guide")
    ax.invert_xaxis()
    ax.set_xlabel("effective step size h (coarse to fine)")
    ax.set_ylabel("absolute error")
    ax.set_title(title, fontsize=13)
    ax.grid(True, which="both", alpha=0.28)
    ax.legend(frameon=False, loc="best")
    ax.text(
        0.03,
        0.04,
        (
            f"final-pair order = {asymptotic_order:.2f}\n"
            f"min pairwise order = {min_pairwise_order:.2f}\n"
            f"final error = {final_error:.3g}\n"
            f"gate: {gate_status}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "0.85", "alpha": 0.92},
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)


def _branch_gate_report_from_selected_rows(
    rows: list[dict[str, object]],
    *,
    max_rel_gamma_jump: float,
    max_rel_omega_jump: float,
    min_successive_overlap: float | None,
) -> dict[str, object] | None:
    from spectraxgk.diagnostics.analysis import branch_continuity_metrics
    from spectraxgk.diagnostics.validation_gates import (
        branch_continuity_gate_report,
        gate_report_to_dict,
    )

    if len(rows) < 2:
        return None
    table = pd.DataFrame(rows).sort_values("ky")
    overlap = None
    if "eig_overlap_prev" in table.columns:
        overlap_values = np.asarray(table["eig_overlap_prev"], dtype=float)[1:]
        if overlap_values.size == table.shape[0] - 1 and np.all(
            np.isfinite(overlap_values)
        ):
            overlap = overlap_values
    metrics = branch_continuity_metrics(
        np.asarray(table["ky"], dtype=float),
        np.asarray(table["gamma"], dtype=float),
        np.asarray(table["omega"], dtype=float),
        successive_overlap=overlap,
    )
    report = branch_continuity_gate_report(
        metrics,
        case="kbm_linear_branch_continuity",
        source="selected KBM comparison rows",
        max_rel_gamma_jump=max_rel_gamma_jump,
        max_rel_omega_jump=max_rel_omega_jump,
        min_successive_overlap=(
            min_successive_overlap if overlap is not None else None
        ),
    )
    return gate_report_to_dict(report)


def _coerce_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def selected_candidate_rows(path: Path) -> list[dict[str, object]]:
    """Load selected branch rows from a KBM candidate table."""

    table = pd.read_csv(path)
    required = {"ky", "gamma", "omega", "selected"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    selected = table["selected"].map(_coerce_bool)
    rows = table.loc[selected].sort_values("ky").to_dict(orient="records")
    return [_json_clean(row) for row in rows]


def build_kbm_branch_summary(
    candidate_csv: Path,
    *,
    max_rel_gamma_jump: float,
    max_rel_omega_jump: float,
    min_successive_overlap: float | None,
) -> dict[str, object]:
    """Build the JSON payload for the selected KBM branch-continuity gate."""

    rows = selected_candidate_rows(candidate_csv)
    try:
        candidate_label = str(candidate_csv.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        candidate_label = str(candidate_csv)
    report = _branch_gate_report_from_selected_rows(
        rows,
        max_rel_gamma_jump=max_rel_gamma_jump,
        max_rel_omega_jump=max_rel_omega_jump,
        min_successive_overlap=min_successive_overlap,
    )
    payload: dict[str, object] = {
        "case": "kbm_linear_branch_continuity",
        "candidate_csv": candidate_label,
        "selected_count": len(rows),
        "thresholds": {
            "max_rel_gamma_jump": float(max_rel_gamma_jump),
            "max_rel_omega_jump": float(max_rel_omega_jump),
            "min_successive_overlap": min_successive_overlap,
        },
        "rows": rows,
        "gate_report": report,
        "gate_passed": None if report is None else bool(report["passed"]),
        "notes": (
            "Selected rows are taken from the KBM comparison candidate table. "
            "The gate is intentionally a branch-identity check: adjacent gamma "
            "and omega jumps should stay smooth, and successive eigenfunction "
            "overlaps should remain high when those overlaps are available."
        ),
    }
    return _json_clean(payload)


def build_observed_order_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write an observed-order gate report from a convergence CSV."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_OBSERVED_CSV)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--step-column", default=None)
    group.add_argument("--resolution-column", default=None)
    parser.add_argument("--error-column", default="rel_gamma")
    parser.add_argument("--case", default="cyclone_resolution_observed_order")
    parser.add_argument("--source", default="tracked Cyclone resolution subset")
    parser.add_argument("--min-order", type=float, default=1.0)
    parser.add_argument(
        "--min-pairwise-order",
        type=float,
        default=0.0,
        help="Optional floor for every pairwise observed order; set negative to disable.",
    )
    parser.add_argument("--max-final-error", type=float, default=0.05)
    parser.add_argument(
        "--signed-error",
        action="store_true",
        help="Use signed errors instead of absolute values.",
    )
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OBSERVED_JSON)
    parser.add_argument("--out-png", type=Path, default=DEFAULT_OBSERVED_PNG)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--title", default="Cyclone Resolution Convergence")
    return parser


def build_kbm_branch_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write a KBM branch-continuity gate summary from selected candidate rows."
    )
    parser.add_argument("--candidates", type=Path, default=DEFAULT_KBM_CANDIDATES)
    parser.add_argument("--out", type=Path, default=DEFAULT_KBM_BRANCH_OUT)
    parser.add_argument("--max-rel-gamma-jump", type=float, default=0.5)
    parser.add_argument("--max-rel-omega-jump", type=float, default=0.5)
    parser.add_argument("--min-successive-overlap", type=float, default=0.95)
    return parser


def main_observed_order(argv: list[str] | None = None) -> int:
    args = build_observed_order_parser().parse_args(argv)
    resolution_column = (
        args.resolution_column if args.resolution_column is not None else "Nm"
    )
    min_pairwise_order = (
        None
        if args.min_pairwise_order is None or float(args.min_pairwise_order) < 0.0
        else float(args.min_pairwise_order)
    )
    summary = build_summary(
        args.csv,
        step_column=args.step_column,
        resolution_column=resolution_column if args.step_column is None else None,
        error_column=args.error_column,
        case=args.case,
        source=args.source,
        min_order=float(args.min_order),
        min_pairwise_order=min_pairwise_order,
        max_final_error=args.max_final_error,
        absolute_error=not bool(args.signed_error),
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    if not args.no_plot:
        write_observed_order_plot(
            summary, args.out_png, title=args.title, min_order=float(args.min_order)
        )
    print(f"Wrote {args.out_json}")
    if not args.no_plot:
        print(f"Wrote {args.out_png}")
    return 0


def main_kbm_branch(argv: list[str] | None = None) -> int:
    args = build_kbm_branch_parser().parse_args(argv)
    summary = build_kbm_branch_summary(
        args.candidates,
        max_rel_gamma_jump=float(args.max_rel_gamma_jump),
        max_rel_omega_jump=float(args.max_rel_omega_jump),
        min_successive_overlap=float(args.min_successive_overlap),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(f"Wrote {args.out}")
    return 0


def main_collision_table(argv: list[str] | None = None) -> int:
    args = build_collision_table_parser().parse_args(argv)
    metadata = write_collision_table(
        args.out, args.metadata_out, digits=int(args.digits)
    )
    print(f"Wrote {args.out} ({metadata['sha256']})")
    print(f"Wrote {args.metadata_out}")
    return 0


def main_collision_verification(argv: list[str] | None = None) -> int:
    args = build_collision_verification_parser().parse_args(argv)
    summary = write_coulomb_operator_verification_artifacts(
        args.out_json,
        args.out_png,
        digits=int(args.digits),
    )
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_png}")
    print(
        f"Collision verification gate: {'PASS' if summary['gate_passed'] else 'FAIL'}"
    )
    return 0 if summary["gate_passed"] else 1


def main_collision_response(argv: list[str] | None = None) -> int:
    args = build_collision_response_parser().parse_args(argv)
    summary = write_drift_kinetic_response_convergence_artifacts(
        args.out_json,
        args.out_csv,
        args.out_png,
        digits=int(args.digits),
    )
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_png}")
    print(f"Collision response gate: {'PASS' if summary['gate_passed'] else 'FAIL'}")
    return 0 if summary["gate_passed"] else 1


def main(argv: list[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    if not tokens:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            "command",
            choices=(
                "figures",
                "observed-order",
                "kbm-branch",
                "collision-table",
                "collision-verification",
                "collision-response",
            ),
        )
        parser.print_help()
        return 2
    command, rest = tokens[0], tokens[1:]
    if command == "figures":
        return main_figures(rest)
    if command == "observed-order":
        return main_observed_order(rest)
    if command == "kbm-branch":
        return main_kbm_branch(rest)
    if command == "collision-table":
        return main_collision_table(rest)
    if command == "collision-verification":
        return main_collision_verification(rest)
    if command == "collision-response":
        return main_collision_response(rest)
    raise SystemExit(f"unknown command: {command}")


if __name__ == "__main__":
    raise SystemExit(main())
