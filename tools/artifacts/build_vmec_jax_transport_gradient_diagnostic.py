#!/usr/bin/env python3
"""Build a VMEC-JAX/SPECTRAX-GK boundary transport-gradient diagnostic.

This tool is meant to be run on a solved VMEC input deck, typically the
``input.final`` from a constraints-only QA optimization.  It rebuilds the same
active boundary basis, assembles a transport-only SPECTRAX-GK objective, and
reports whether the local boundary gradient is measurable before launching more
expensive transport-weight ladders or long-window nonlinear audits.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import re
from typing import cast

import numpy as np

from spectraxgk.objectives.stellarator import StellaratorITGSampleSet
from spectraxgk.validation.stellarator.transport_policies import (
    VMECJAXNonlinearAuditPolicy,
)
from spectraxgk.validation.stellarator.transport_samples import (
    transport_objective_sample_summary,
)
from spectraxgk.objectives.vmec_transport_gradient import (
    boundary_spec_record,
    build_boundary_transport_gradient_report,
    write_boundary_transport_gradient_report,
)
from spectraxgk.objectives.vmec_transport import (
    VMECJAXSpectraxTransportObjective,
    VMECJAXTransportObjectiveConfig,
    VMECJAXTransportObjectiveKind,
    VMECJAXTransportObjectiveTransform,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUDIT_POLICY = VMECJAXNonlinearAuditPolicy()
DEFAULT_TRANSPORT_SURFACES = DEFAULT_AUDIT_POLICY.recommended_surfaces
DEFAULT_TRANSPORT_ALPHAS = DEFAULT_AUDIT_POLICY.recommended_alphas
DEFAULT_TRANSPORT_KY_VALUES = DEFAULT_AUDIT_POLICY.recommended_ky_values
_VMEC_COEFFICIENT_RE = re.compile(
    r"(?P<family>RBC|RBS|ZBC|ZBS)\(\s*(?P<n>[+-]?\d+)\s*,\s*(?P<m>[+-]?\d+)\s*\)"
    r"\s*=\s*(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"
)
_SPEC_KIND_TO_INPUT_FAMILY = {
    "rc": "RBC",
    "rs": "RBS",
    "zc": "ZBC",
    "zs": "ZBS",
}


def _float_tuple(raw: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in str(raw).split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return values


def _int_tuple(raw: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in str(raw).split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError(
            "expected at least one comma-separated integer"
        )
    return values


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Solved VMEC input deck, e.g. input.final",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=ROOT
        / "docs"
        / "_static"
        / "vmec_jax_transport_gradient_diagnostic.json",
        help="Output JSON diagnostic path",
    )
    parser.add_argument(
        "--max-mode", type=int, default=5, help="Active boundary max mode"
    )
    parser.add_argument(
        "--min-vmec-mode", type=int, default=7, help="VMEC resolution floor"
    )
    parser.add_argument(
        "--surfaces", type=_float_tuple, default=DEFAULT_TRANSPORT_SURFACES
    )
    parser.add_argument("--alphas", type=_float_tuple, default=DEFAULT_TRANSPORT_ALPHAS)
    parser.add_argument(
        "--ky-values", type=_float_tuple, default=DEFAULT_TRANSPORT_KY_VALUES
    )
    parser.add_argument(
        "--allow-underresolved-sample-set",
        action="store_true",
        help="Permit exploratory single-point gradients; production gradient admission fails closed by default",
    )
    parser.add_argument(
        "--transport-kind",
        choices=("growth", "quasilinear_flux", "nonlinear_window_heat_flux"),
        default="nonlinear_window_heat_flux",
    )
    parser.add_argument("--transport-weight", type=float, default=1.0)
    parser.add_argument("--ntheta", type=int, default=24)
    parser.add_argument("--mboz", type=int, default=21)
    parser.add_argument("--nboz", type=int, default=21)
    parser.add_argument("--n-laguerre", type=int, default=2)
    parser.add_argument("--n-hermite", type=int, default=3)
    parser.add_argument(
        "--spectrax-objective-transform",
        choices=("raw", "scaled", "log1p"),
        default="log1p",
    )
    parser.add_argument("--spectrax-objective-scale", type=float, default=1.0)
    parser.add_argument("--inner-max-iter", type=int, default=120)
    parser.add_argument("--inner-ftol", type=float, default=1.0e-9)
    parser.add_argument("--trial-max-iter", type=int, default=120)
    parser.add_argument("--trial-ftol", type=float, default=1.0e-9)
    parser.add_argument("--solver-device", choices=("cpu", "gpu"), default=None)
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--sensitivity-atol", type=float, default=1.0e-12)
    parser.add_argument(
        "--fd-check-indices",
        type=_int_tuple,
        default=(),
        help=(
            "Comma-separated active-boundary parameter indices for central finite-difference "
            "checks against the reverse scalar gradient"
        ),
    )
    parser.add_argument(
        "--fd-check-step",
        type=float,
        default=1.0e-4,
        help="Central finite-difference step for --fd-check-indices",
    )
    parser.add_argument(
        "--fd-check-relative-step",
        type=float,
        default=0.0,
        help=(
            "If positive, cap each finite-difference step at this fraction of the "
            "matching VMEC input coefficient magnitude when that coefficient is present"
        ),
    )
    parser.add_argument(
        "--fd-check-rtol",
        type=float,
        default=5.0e-2,
        help="Relative tolerance for AD-vs-finite-difference cost-gradient checks",
    )
    parser.add_argument(
        "--fd-check-atol",
        type=float,
        default=1.0e-6,
        help="Absolute tolerance for AD-vs-finite-difference cost-gradient checks",
    )
    parser.add_argument(
        "--surface-gradient-chunk-size",
        type=int,
        default=0,
        help=(
            "Compute exact weighted-mean transport gradients in surface chunks to lower peak memory; "
            "0 evaluates the full sample set in one reverse pass"
        ),
    )
    parser.add_argument(
        "--include-jacobian",
        action="store_true",
        help="Also materialize dense residual-Jacobian norms; slower than the scalar-gradient path",
    )
    parser.add_argument(
        "--require-sensitive",
        action="store_true",
        help="Exit 2 if the local transport-gradient norm is below --sensitivity-atol",
    )
    parser.add_argument(
        "--require-fd-consistency",
        action="store_true",
        help="Exit 3 if --fd-check-indices are absent or any AD-vs-finite-difference check fails",
    )
    return parser.parse_args(argv)


def _copy_args(args: argparse.Namespace, **updates: object) -> argparse.Namespace:
    payload = vars(args).copy()
    payload.update(updates)
    return argparse.Namespace(**payload)


def _vmec_input_coefficients(input_path: Path) -> dict[str, float]:
    """Return VMEC boundary coefficients from an input deck.

    VMEC input files frequently place multiple coefficients on one line and may
    use Fortran ``D`` exponents.  Comments are stripped before matching so
    inactive examples do not enter the finite-difference conditioning report.
    """

    coefficients: dict[str, float] = {}
    if not Path(input_path).exists():
        return coefficients
    for raw_line in Path(input_path).read_text(encoding="utf-8").splitlines():
        line = raw_line.split("!", 1)[0]
        for match in _VMEC_COEFFICIENT_RE.finditer(line):
            label = (
                f"{match.group('family')}({int(match.group('n'))},"
                f"{int(match.group('m'))})"
            )
            coefficients[label] = float(
                match.group("value").replace("D", "E").replace("d", "e")
            )
    return coefficients


def _coefficient_label_from_spec_record(row: dict[str, object]) -> str | None:
    kind = row.get("kind")
    family = (
        _SPEC_KIND_TO_INPUT_FAMILY.get(str(kind).lower()) if kind is not None else None
    )
    if family is None:
        return None
    m = row.get("m")
    n = row.get("n")
    if m is None or n is None:
        return None
    return f"{family}({int(n)},{int(m)})"


def _effective_fd_step(
    requested_step: float,
    coefficient_value: float | None,
    *,
    relative_step: float,
) -> float:
    """Return the finite-difference step after optional coefficient-relative capping."""

    step = float(requested_step)
    rel = float(relative_step)
    if rel <= 0.0 or coefficient_value is None:
        return step
    coefficient_abs = abs(float(coefficient_value))
    if coefficient_abs <= 0.0:
        return step
    return min(step, coefficient_abs * rel)


def _sample_set_from_args(args: argparse.Namespace) -> StellaratorITGSampleSet:
    return StellaratorITGSampleSet(
        surfaces=tuple(float(x) for x in args.surfaces),
        alphas=tuple(float(x) for x in args.alphas),
        ky_values=tuple(float(x) for x in args.ky_values),
    )


def _normalized_axis_weights(values: Sequence[float] | None, size: int) -> np.ndarray:
    if values is None:
        return np.full((int(size),), 1.0 / float(size), dtype=float)
    arr = np.asarray(values, dtype=float)
    return arr / float(np.sum(arr))


def _surface_chunk_sample_sets(
    sample_set: StellaratorITGSampleSet,
    *,
    chunk_size: int,
) -> Iterable[tuple[StellaratorITGSampleSet, float, list[int]]]:
    """Yield surface chunks and their aggregate objective weights."""

    if int(chunk_size) <= 0:
        yield sample_set, 1.0, list(range(len(sample_set.surfaces)))
        return
    if sample_set.reduction not in ("weighted_mean", "mean"):
        raise ValueError(
            "surface-gradient chunking currently supports only mean or weighted_mean reductions"
        )
    surfaces = tuple(float(x) for x in sample_set.surfaces)
    surface_weights = _normalized_axis_weights(
        sample_set.surface_weights, len(surfaces)
    )
    for start in range(0, len(surfaces), int(chunk_size)):
        indices = list(range(start, min(start + int(chunk_size), len(surfaces))))
        chunk_surfaces = tuple(surfaces[i] for i in indices)
        if sample_set.reduction == "mean":
            chunk_weight = float(len(indices) / len(surfaces))
            chunk_surface_weights = None
        else:
            chunk_weight = float(np.sum(surface_weights[indices]))
            chunk_surface_weights = (
                None
                if sample_set.surface_weights is None
                else tuple(float(sample_set.surface_weights[i]) for i in indices)
            )
        yield (
            StellaratorITGSampleSet(
                surfaces=chunk_surfaces,
                alphas=sample_set.alphas,
                ky_values=sample_set.ky_values,
                surface_weights=chunk_surface_weights,
                alpha_weights=sample_set.alpha_weights,
                ky_weights=sample_set.ky_weights,
                reduction=sample_set.reduction,
            ),
            chunk_weight,
            indices,
        )


def _residual_gradient_from_optimizer(
    optimizer: object, params: np.ndarray
) -> tuple[float, np.ndarray]:
    residual = np.asarray(
        getattr(optimizer, "residual_fun")(params), dtype=float
    ).reshape(-1)
    if residual.size != 1:
        raise ValueError(
            "chunked transport-gradient aggregation requires a scalar residual"
        )
    residual_value = float(residual[0])
    _cost, cost_gradient = getattr(optimizer, "objective_and_gradient_fun")(params)
    cost_gradient_array = np.asarray(cost_gradient, dtype=float).reshape(-1)
    if abs(residual_value) > 1.0e-300:
        return residual_value, cost_gradient_array / residual_value
    jacobian_fun = getattr(optimizer, "jacobian_fun", None)
    if callable(jacobian_fun):
        jacobian = np.asarray(jacobian_fun(params), dtype=float)
        if jacobian.ndim == 1:
            jacobian = jacobian.reshape(1, -1)
        if jacobian.shape[0] == 1:
            return residual_value, np.asarray(jacobian[0], dtype=float).reshape(-1)
    raise ValueError(
        "cannot recover residual gradient for a zero-residual chunk without jacobian_fun"
    )


def _transform_residual_and_gradient(
    raw_value: float,
    raw_gradient: np.ndarray,
    args: argparse.Namespace,
) -> tuple[float, np.ndarray]:
    scale = float(args.spectrax_objective_scale)
    if str(args.spectrax_objective_transform) == "raw":
        transformed = float(raw_value)
        derivative = 1.0
    elif str(args.spectrax_objective_transform) == "scaled":
        transformed = float(raw_value) / scale
        derivative = 1.0 / scale
    elif str(args.spectrax_objective_transform) == "log1p":
        scaled = float(raw_value) / scale
        transformed = float(np.sign(scaled) * np.log1p(abs(scaled)))
        derivative = 1.0 / (scale * (1.0 + abs(scaled)))
    else:  # pragma: no cover - argparse constrains this path.
        raise ValueError(
            f"unknown objective transform {args.spectrax_objective_transform!r}"
        )
    weight = float(args.transport_weight)
    return weight * transformed, weight * derivative * np.asarray(
        raw_gradient, dtype=float
    )


def _transformed_residual_value(
    raw_value: float,
    args: argparse.Namespace,
) -> float:
    transformed, _gradient = _transform_residual_and_gradient(
        raw_value,
        np.zeros(1, dtype=float),
        args,
    )
    return float(transformed)


@dataclass(frozen=True)
class _SyntheticScalarOptimizer:
    _specs: tuple[object, ...]
    residual: float
    residual_gradient: np.ndarray

    def residual_fun(self, params: Sequence[float] | np.ndarray) -> np.ndarray:
        params_array = np.asarray(params, dtype=float).reshape(-1)
        if params_array.size != self.residual_gradient.size:
            raise ValueError(
                "parameter vector size changed during synthetic gradient report"
            )
        return np.asarray([self.residual], dtype=float)

    def objective_and_gradient_fun(
        self, params: Sequence[float] | np.ndarray
    ) -> tuple[float, np.ndarray]:
        _ = self.residual_fun(params)
        gradient = float(self.residual) * np.asarray(
            self.residual_gradient, dtype=float
        )
        return 0.5 * float(self.residual) ** 2, gradient

    def jacobian_fun(self, params: Sequence[float] | np.ndarray) -> np.ndarray:
        _ = self.residual_fun(params)
        return np.asarray([self.residual_gradient], dtype=float)


def _scalar_residual_from_optimizer(optimizer: object, params: np.ndarray) -> float:
    residual = np.asarray(
        getattr(optimizer, "residual_fun")(params), dtype=float
    ).reshape(-1)
    if residual.size != 1:
        raise ValueError(
            "finite-difference transport-gradient checks require a scalar residual"
        )
    return float(residual[0])


def _chunked_transformed_residual_value(
    args: argparse.Namespace,
    sample_set: StellaratorITGSampleSet,
    params: np.ndarray,
) -> float:
    raw_value = 0.0
    expected_size: int | None = None
    for chunk_sample_set, chunk_weight, _surface_indices in _surface_chunk_sample_sets(
        sample_set,
        chunk_size=int(args.surface_gradient_chunk_size),
    ):
        chunk_args = _copy_args(
            args,
            surfaces=chunk_sample_set.surfaces,
            alphas=chunk_sample_set.alphas,
            ky_values=chunk_sample_set.ky_values,
            spectrax_objective_transform="raw",
            spectrax_objective_scale=1.0,
            transport_weight=1.0,
        )
        stage, _setup = _build_stage(chunk_args)
        chunk_size = len(tuple(stage.specs))
        if expected_size is None:
            expected_size = chunk_size
        elif chunk_size != expected_size:
            raise ValueError("surface chunks produced inconsistent parameter counts")
        if params.size != chunk_size:
            raise ValueError(
                f"finite-difference parameter size {params.size} does not match chunk size {chunk_size}"
            )
        raw_value += float(chunk_weight) * _scalar_residual_from_optimizer(
            stage.optimizer, params
        )
    return _transformed_residual_value(raw_value, args)


def _finite_difference_consistency_report(
    optimizer: object,
    params: Sequence[float] | np.ndarray,
    *,
    indices: Sequence[int],
    step: float,
    relative_step: float = 0.0,
    rtol: float,
    atol: float,
    residual_fun: object | None = None,
    input_coefficients: dict[str, float] | None = None,
) -> dict[str, object]:
    """Compare reverse scalar-gradient components with central finite differences."""

    params_array = np.asarray(params, dtype=float).reshape(-1)
    if float(step) <= 0.0:
        raise ValueError("--fd-check-step must be positive")
    if float(relative_step) < 0.0:
        raise ValueError("--fd-check-relative-step must be non-negative")
    if float(rtol) < 0.0 or float(atol) < 0.0:
        raise ValueError("--fd-check-rtol and --fd-check-atol must be non-negative")
    unique_indices = tuple(dict.fromkeys(int(i) for i in indices))
    if not unique_indices:
        return {
            "enabled": False,
            "passed": False,
            "blockers": ["no_fd_check_indices"],
            "classification": "finite_difference_check_not_requested",
        }
    invalid = [
        int(i) for i in unique_indices if int(i) < 0 or int(i) >= params_array.size
    ]
    if invalid:
        raise ValueError(
            f"--fd-check-indices out of range for {params_array.size} parameters: {invalid}"
        )

    specs = tuple(getattr(optimizer, "_specs", ()))
    evaluate_residual = (
        residual_fun if callable(residual_fun) else getattr(optimizer, "residual_fun")
    )
    base_residual = np.asarray(evaluate_residual(params_array), dtype=float).reshape(-1)
    base_cost_raw, reverse_gradient_raw = getattr(
        optimizer, "objective_and_gradient_fun"
    )(params_array)
    reverse_gradient = np.asarray(reverse_gradient_raw, dtype=float).reshape(-1)
    if reverse_gradient.size != params_array.size:
        raise ValueError(
            f"reverse gradient size {reverse_gradient.size} does not match parameter size {params_array.size}"
        )
    base_cost = float(base_cost_raw)
    base_residual_scalar = float(base_residual[0]) if base_residual.size == 1 else None
    rows: list[dict[str, object]] = []
    finite = bool(
        np.isfinite(base_cost)
        and np.all(np.isfinite(base_residual))
        and np.all(np.isfinite(reverse_gradient))
    )
    max_abs_error = 0.0
    max_relative_error = 0.0
    max_abs_fd_cost_gradient = 0.0
    step_conditioning_warnings: list[str] = []
    coefficients = input_coefficients or {}

    for index in unique_indices:
        spec = specs[index] if index < len(specs) else object()
        spec_row = boundary_spec_record(spec, fallback_index=index)
        coefficient_label = _coefficient_label_from_spec_record(spec_row)
        coefficient_value = (
            None if coefficient_label is None else coefficients.get(coefficient_label)
        )
        effective_step = _effective_fd_step(
            float(step),
            coefficient_value,
            relative_step=float(relative_step),
        )
        plus = params_array.copy()
        minus = params_array.copy()
        plus[index] += float(effective_step)
        minus[index] -= float(effective_step)
        residual_plus = np.asarray(evaluate_residual(plus), dtype=float).reshape(-1)
        residual_minus = np.asarray(evaluate_residual(minus), dtype=float).reshape(-1)
        cost_plus = 0.5 * float(np.vdot(residual_plus, residual_plus))
        cost_minus = 0.5 * float(np.vdot(residual_minus, residual_minus))
        fd_cost_gradient = (cost_plus - cost_minus) / (2.0 * float(effective_step))
        reverse_cost_gradient = float(reverse_gradient[index])
        abs_error = abs(reverse_cost_gradient - fd_cost_gradient)
        denominator = max(
            abs(reverse_cost_gradient), abs(fd_cost_gradient), float(atol), 1.0e-300
        )
        relative_error = abs_error / denominator
        passed = bool(
            np.isfinite(fd_cost_gradient)
            and np.isfinite(reverse_cost_gradient)
            and np.isclose(
                reverse_cost_gradient,
                fd_cost_gradient,
                rtol=float(rtol),
                atol=float(atol),
            )
        )
        finite = bool(
            finite
            and np.all(np.isfinite(residual_plus))
            and np.all(np.isfinite(residual_minus))
            and np.isfinite(cost_plus)
            and np.isfinite(cost_minus)
            and np.isfinite(fd_cost_gradient)
        )
        coefficient_step_ratio = (
            None
            if coefficient_value is None or coefficient_value == 0.0
            else abs(float(effective_step)) / abs(float(coefficient_value))
        )
        requested_step_ratio = (
            None
            if coefficient_value is None or coefficient_value == 0.0
            else abs(float(step)) / abs(float(coefficient_value))
        )
        if requested_step_ratio is not None and requested_step_ratio > 1.0:
            step_conditioning_warnings.append(
                f"fd_step_exceeds_input_coefficient:{coefficient_label}"
            )
        row = {
            **spec_row,
            "parameter_index": int(index),
            "requested_step": float(step),
            "step": float(effective_step),
            "coefficient_label": coefficient_label,
            "input_coefficient_value": (
                None if coefficient_value is None else float(coefficient_value)
            ),
            "requested_step_to_input_coefficient_abs": requested_step_ratio,
            "step_to_input_coefficient_abs": coefficient_step_ratio,
            "residual_plus": [float(x) for x in residual_plus],
            "residual_minus": [float(x) for x in residual_minus],
            "cost_plus": cost_plus,
            "cost_minus": cost_minus,
            "fd_cost_gradient": float(fd_cost_gradient),
            "reverse_cost_gradient": reverse_cost_gradient,
            "abs_cost_gradient_error": float(abs_error),
            "relative_cost_gradient_error": float(relative_error),
            "passed": passed,
        }
        if (
            base_residual_scalar is not None
            and residual_plus.size == 1
            and residual_minus.size == 1
        ):
            fd_residual_gradient = (
                float(residual_plus[0]) - float(residual_minus[0])
            ) / (2.0 * float(effective_step))
            reverse_residual_gradient = (
                reverse_cost_gradient / base_residual_scalar
                if abs(base_residual_scalar) > 1.0e-300
                else None
            )
            residual_abs_error = (
                None
                if reverse_residual_gradient is None
                else abs(float(reverse_residual_gradient) - fd_residual_gradient)
            )
            row.update(
                {
                    "fd_residual_gradient": float(fd_residual_gradient),
                    "reverse_residual_gradient": (
                        None
                        if reverse_residual_gradient is None
                        else float(reverse_residual_gradient)
                    ),
                    "abs_residual_gradient_error": (
                        None
                        if residual_abs_error is None
                        else float(residual_abs_error)
                    ),
                }
            )
        rows.append(row)
        max_abs_error = max(max_abs_error, float(abs_error))
        max_relative_error = max(max_relative_error, float(relative_error))
        max_abs_fd_cost_gradient = max(
            max_abs_fd_cost_gradient, abs(float(fd_cost_gradient))
        )

    mismatches = [row for row in rows if not bool(row["passed"])]
    blockers: list[str] = []
    if not finite:
        blockers.append("nonfinite_ad_fd_check")
    if mismatches:
        blockers.append("ad_fd_mismatch")
    if max_abs_fd_cost_gradient <= float(atol):
        blockers.append("finite_difference_signal_below_atol")
    passed = bool(
        finite
        and not mismatches
        and "finite_difference_signal_below_atol" not in blockers
    )
    return {
        "enabled": True,
        "indices": [int(i) for i in unique_indices],
        "step": float(step),
        "relative_step": float(relative_step),
        "rtol": float(rtol),
        "atol": float(atol),
        "base_residual": [float(x) for x in base_residual],
        "base_cost": base_cost,
        "row_count": len(rows),
        "max_abs_cost_gradient_error": float(max_abs_error),
        "max_relative_cost_gradient_error": float(max_relative_error),
        "max_abs_fd_cost_gradient": float(max_abs_fd_cost_gradient),
        "rows": rows,
        "finite": finite,
        "passed": passed,
        "blockers": blockers,
        "conditioning_warnings": sorted(set(step_conditioning_warnings)),
        "classification": "ad_fd_consistent"
        if passed
        else "ad_fd_inconsistent_or_underresolved",
        "next_action": (
            "reverse-mode transport gradients can be used for local projected updates"
            if passed
            else (
                "do not use this reverse gradient for VMEC-JAX optimization; repair the differentiable "
                "VMEC/Boozer/SPECTRAX path or use finite differences only as diagnostics"
            )
        ),
    }


def _chunked_boundary_transport_gradient_report(
    args: argparse.Namespace,
    sample_set: StellaratorITGSampleSet,
) -> tuple[dict[str, object], _SyntheticScalarOptimizer, np.ndarray]:
    """Build the same weighted-mean scalar gradient using smaller surface chunks."""

    params: np.ndarray | None = None
    specs: tuple[object, ...] | None = None
    raw_value = 0.0
    raw_gradient: np.ndarray | None = None
    chunks: list[dict[str, object]] = []
    for chunk_sample_set, chunk_weight, surface_indices in _surface_chunk_sample_sets(
        sample_set,
        chunk_size=int(args.surface_gradient_chunk_size),
    ):
        chunk_args = _copy_args(
            args,
            surfaces=chunk_sample_set.surfaces,
            alphas=chunk_sample_set.alphas,
            ky_values=chunk_sample_set.ky_values,
            spectrax_objective_transform="raw",
            spectrax_objective_scale=1.0,
            transport_weight=1.0,
        )
        stage, setup = _build_stage(chunk_args)
        chunk_specs = tuple(stage.specs)
        chunk_params = np.zeros(len(chunk_specs), dtype=float)
        if params is None:
            params = chunk_params
            specs = chunk_specs
            raw_gradient = np.zeros_like(chunk_params)
        elif chunk_params.size != params.size:
            raise ValueError("surface chunks produced inconsistent parameter counts")
        chunk_residual, chunk_residual_gradient = _residual_gradient_from_optimizer(
            stage.optimizer,
            chunk_params,
        )
        raw_value += float(chunk_weight) * float(chunk_residual)
        raw_gradient = (
            np.asarray(raw_gradient, dtype=float)
            + float(chunk_weight) * chunk_residual_gradient
        )
        chunks.append(
            {
                "surface_indices": [int(i) for i in surface_indices],
                "surfaces": [float(x) for x in chunk_sample_set.surfaces],
                "sample_count": int(chunk_sample_set.n_samples),
                "weight": float(chunk_weight),
                "raw_residual": float(chunk_residual),
                "raw_gradient_norm_l2": float(np.linalg.norm(chunk_residual_gradient)),
                "setup": setup,
            }
        )
    if params is None or specs is None or raw_gradient is None:
        raise RuntimeError("surface-gradient chunking produced no chunks")
    residual, residual_gradient = _transform_residual_and_gradient(
        raw_value, raw_gradient, args
    )
    synthetic = _SyntheticScalarOptimizer(
        _specs=specs,
        residual=float(residual),
        residual_gradient=np.asarray(residual_gradient, dtype=float),
    )
    report = build_boundary_transport_gradient_report(
        synthetic,
        params=params,
        label="vmec_jax_transport_gradient",
        top_n=int(args.top_n),
        sensitivity_atol=float(args.sensitivity_atol),
        include_jacobian=bool(args.include_jacobian),
    )
    report["chunked_gradient"] = {
        "enabled": True,
        "axis": "surface",
        "surface_gradient_chunk_size": int(args.surface_gradient_chunk_size),
        "chunk_count": len(chunks),
        "raw_weighted_residual": float(raw_value),
        "raw_weighted_gradient_norm_l2": float(np.linalg.norm(raw_gradient)),
        "chunks": chunks,
        "aggregation": (
            "raw per-surface-chunk residual gradients are combined with the original "
            "sample weights, then the scalar objective transform is applied once"
        ),
    }
    return report, synthetic, params


def _build_stage(args: argparse.Namespace):
    vj = importlib.import_module("vmec_jax")
    build_stage = getattr(vj, "build_fixed_boundary_objective_stage", None)
    if build_stage is None:  # vmec_jax releases may not re-export workflow helpers.
        workflow = importlib.import_module("vmec_jax.optimization_workflow")
        build_stage = getattr(workflow, "build_fixed_boundary_objective_stage")

    sample_set = _sample_set_from_args(args)
    config = VMECJAXTransportObjectiveConfig(
        kind=cast(VMECJAXTransportObjectiveKind, str(args.transport_kind)),
        sample_set=sample_set,
        ntheta=int(args.ntheta),
        mboz=int(args.mboz),
        nboz=int(args.nboz),
        n_laguerre=int(args.n_laguerre),
        n_hermite=int(args.n_hermite),
        objective_transform=cast(
            VMECJAXTransportObjectiveTransform, str(args.spectrax_objective_transform)
        ),
        objective_scale=float(args.spectrax_objective_scale),
    )
    transport = VMECJAXSpectraxTransportObjective(config=config)
    vmec = vj.FixedBoundaryVMEC.from_input(
        args.input,
        max_mode=int(args.max_mode),
        min_vmec_mode=int(args.min_vmec_mode),
        output_dir=Path(args.out_json).parent,
    )
    problem = vj.LeastSquaresProblem.from_tuples(
        [(transport.J, 0.0, float(args.transport_weight))]
    )
    stage = build_stage(
        vmec.cfg,
        vmec.indata,
        stage_mode=int(args.max_mode),
        objectives=problem.objective_terms,
        include=vmec.include,
        fix=vmec.fix,
        project_input_boundary_to_max_mode=bool(
            vmec.project_input_boundary_to_max_mode
        ),
        inner_max_iter=int(args.inner_max_iter),
        inner_ftol=float(args.inner_ftol),
        trial_max_iter=int(args.trial_max_iter),
        trial_ftol=float(args.trial_ftol),
        solver_device=args.solver_device,
    )
    setup = {
        "input": str(args.input),
        "max_mode": int(args.max_mode),
        "min_vmec_mode": int(args.min_vmec_mode),
        "transport_kind": str(args.transport_kind),
        "transport_weight": float(args.transport_weight),
        "sample_set": sample_set.to_dict(),
        "spectrax_config": {
            "ntheta": int(args.ntheta),
            "mboz": int(args.mboz),
            "nboz": int(args.nboz),
            "n_laguerre": int(args.n_laguerre),
            "n_hermite": int(args.n_hermite),
            "objective_transform": str(args.spectrax_objective_transform),
            "objective_scale": float(args.spectrax_objective_scale),
            "gradient_scope": config.gradient_scope,
        },
        "solver_device": args.solver_device,
    }
    return stage, setup


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    sample_set = _sample_set_from_args(args)
    sample_summary = transport_objective_sample_summary(sample_set)
    if not bool(sample_summary["passed"]) and not bool(
        args.allow_underresolved_sample_set
    ):
        raise ValueError(
            "under-resolved transport-gradient sample set; use the default multi-sample set "
            "or pass --allow-underresolved-sample-set for exploratory diagnostics"
        )
    if int(args.surface_gradient_chunk_size) > 0:
        report, gradient_optimizer, params = (
            _chunked_boundary_transport_gradient_report(args, sample_set)
        )

        def residual_fun(params_array: Sequence[float] | np.ndarray) -> np.ndarray:
            return np.asarray(
                [
                    _chunked_transformed_residual_value(
                        args,
                        sample_set,
                        np.asarray(params_array, dtype=float).reshape(-1),
                    )
                ],
                dtype=float,
            )

        report["setup"] = {
            "input": str(args.input),
            "max_mode": int(args.max_mode),
            "min_vmec_mode": int(args.min_vmec_mode),
            "transport_kind": str(args.transport_kind),
            "transport_weight": float(args.transport_weight),
            "sample_set": sample_set.to_dict(),
            "spectrax_config": {
                "ntheta": int(args.ntheta),
                "mboz": int(args.mboz),
                "nboz": int(args.nboz),
                "n_laguerre": int(args.n_laguerre),
                "n_hermite": int(args.n_hermite),
                "objective_transform": str(args.spectrax_objective_transform),
                "objective_scale": float(args.spectrax_objective_scale),
                "gradient_scope": VMECJAXTransportObjectiveConfig(
                    kind=cast(VMECJAXTransportObjectiveKind, str(args.transport_kind)),
                    sample_set=sample_set,
                ).gradient_scope,
            },
            "solver_device": args.solver_device,
        }
    else:
        stage, setup = _build_stage(args)
        params = np.zeros(len(stage.specs), dtype=float)
        gradient_optimizer = stage.optimizer
        residual_fun = getattr(stage.optimizer, "residual_fun", None)
        report = build_boundary_transport_gradient_report(
            stage.optimizer,
            params=params,
            label="vmec_jax_transport_gradient",
            top_n=int(args.top_n),
            sensitivity_atol=float(args.sensitivity_atol),
            include_jacobian=bool(args.include_jacobian),
        )
        report["setup"] = setup
        report["chunked_gradient"] = {"enabled": False}
    if tuple(args.fd_check_indices):
        report["finite_difference_consistency"] = _finite_difference_consistency_report(
            gradient_optimizer,
            params,
            indices=tuple(args.fd_check_indices),
            step=float(args.fd_check_step),
            relative_step=float(args.fd_check_relative_step),
            rtol=float(args.fd_check_rtol),
            atol=float(args.fd_check_atol),
            residual_fun=residual_fun,
            input_coefficients=_vmec_input_coefficients(args.input),
        )
    else:
        report["finite_difference_consistency"] = {
            "enabled": False,
            "passed": False,
            "blockers": ["no_fd_check_indices"],
            "classification": "finite_difference_check_not_requested",
        }
    report["objective_sample_summary"] = sample_summary
    report["nonlinear_audit_policy"] = DEFAULT_AUDIT_POLICY.to_dict()
    write_boundary_transport_gradient_report(report, args.out_json)
    print(json.dumps(report, indent=2, allow_nan=False))
    if bool(args.require_sensitive) and not bool(
        report["transport_sensitivity_detected"]
    ):
        return 2
    if bool(args.require_fd_consistency) and not bool(
        cast(dict[str, object], report["finite_difference_consistency"])["passed"]
    ):
        return 3
    return 0 if bool(report["finite"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
