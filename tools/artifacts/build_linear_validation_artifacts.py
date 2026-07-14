#!/usr/bin/env python3
"""Generate linear-validation figures and gate reports.

Subcommands:
  collision-table Generate checked high-precision collision coefficient data.
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
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from spectraxgk.diagnostics.analysis import (
    branch_continuity_metrics,
    estimate_observed_order,
)
from spectraxgk.diagnostics.validation_gates import (
    branch_continuity_gate_report,
    gate_report_to_dict,
    observed_order_gate_report,
)
from spectraxgk.runtime import run_runtime_scan
from spectraxgk.workflows.runtime.toml import load_runtime_from_toml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from spectraxgk.artifacts.plotting import (  # noqa: E402
    cyclone_comparison_figure,
    cyclone_reference_figure,
    scan_comparison_figure,
)
from spectraxgk.benchmarking.shared import (  # noqa: E402
    LinearScanResult,
    load_cyclone_reference,
    load_etg_reference,
    load_kbm_reference,
)

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
        denominator = 1 + chi_mp * chi_mp
        e_values = [
            mp.gamma(k + mp.mpf("0.5"))
            / (mp.gamma(mp.mpf("0.5")) * denominator ** (k + mp.mpf("0.5")))
            for k in range(max_order + 1)
        ]
        E_values = [
            chi_mp
            / 2
            * sum(mp.factorial(k) / mp.factorial(j) * e_values[j] for j in range(k + 1))
            for k in range(max_order + 1)
        ]
    return (
        np.asarray([float(value) for value in e_values]),
        np.asarray([float(value) for value in E_values]),
    )


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
    tensor_order: int,
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
    return mp.binomial(n, k)


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
) -> Any:
    left_degree = legendre_order + 2 * radial_order - bessel_order
    right_degree = hermite_order + 2 * laguerre_order
    if right_degree > left_degree or (left_degree - right_degree) % 2:
        return mp.mpf(0)

    half = mp.mpf("0.5")
    total = mp.mpf(0)
    for auxiliary_radial_order in range(right_degree // 2 + 1):
        auxiliary_legendre_order = right_degree - 2 * auxiliary_radial_order
        base_transform = _legendre_to_hermite_laguerre_mp(
            auxiliary_legendre_order,
            auxiliary_radial_order,
            hermite_order,
            laguerre_order,
            mp,
        )
        prefactor = (
            base_transform
            * (auxiliary_legendre_order + half)
            * mp.factorial(auxiliary_radial_order)
            / mp.gamma(
                auxiliary_legendre_order + auxiliary_radial_order + mp.mpf("1.5")
            )
        )
        inner = mp.mpf(0)
        for left_monomial in range(radial_order + 1):
            left_laguerre = _associated_laguerre_monomial_coefficient_mp(
                radial_order,
                legendre_order,
                left_monomial,
                mp,
            )
            for auxiliary_monomial in range(auxiliary_radial_order + 1):
                auxiliary_laguerre = _associated_laguerre_monomial_coefficient_mp(
                    auxiliary_radial_order,
                    auxiliary_legendre_order,
                    auxiliary_monomial,
                    mp,
                )
                for left_power in range(bessel_order, legendre_order + 1):
                    left_legendre = _legendre_monomial_coefficient_mp(
                        legendre_order,
                        left_power,
                        mp,
                    )
                    if left_legendre == 0:
                        continue
                    derivative_factor = mp.factorial(left_power) / mp.factorial(
                        left_power - bessel_order
                    )
                    for auxiliary_power in range(auxiliary_legendre_order + 1):
                        auxiliary_legendre = _legendre_monomial_coefficient_mp(
                            auxiliary_legendre_order,
                            auxiliary_power,
                            mp,
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
    table = pd.read_csv(csv_path).sort_values(x_col)
    return LinearScanResult(
        ky=table[x_col].to_numpy(dtype=float),
        gamma=table["gamma_spectrax"].to_numpy(dtype=float),
        omega=table["omega_spectrax"].to_numpy(dtype=float),
    )


def _cyclone_refresh_reference(ref: LinearScanResult) -> LinearScanResult:
    keep = np.asarray(ref.ky) <= 0.45 + 1.0e-12
    return LinearScanResult(
        ky=np.asarray(ref.ky)[keep],
        gamma=np.asarray(ref.gamma)[keep],
        omega=np.asarray(ref.omega)[keep],
    )


def _run_etg_figures(*, outdir: Path, progress: bool) -> None:
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


def main(argv: list[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    if not tokens:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            "command",
            choices=("figures", "observed-order", "kbm-branch", "collision-table"),
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
    raise SystemExit(f"unknown command: {command}")


if __name__ == "__main__":
    raise SystemExit(main())
