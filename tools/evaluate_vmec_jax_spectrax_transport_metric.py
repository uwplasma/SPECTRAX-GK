#!/usr/bin/env python3
"""Evaluate a SPECTRAX-GK VMEC-JAX transport metric without optimizing.

This tool is intentionally separate from the optimization driver.  It builds a
solved VMEC-JAX state from an input deck, evaluates one SPECTRAX-GK transport
objective on that state, and writes a history-compatible JSON sidecar.  Use it
for baseline/candidate admission bookkeeping before launching expensive matched
nonlinear audits.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any, cast

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spectraxgk import (  # noqa: E402
    StellaratorITGSampleSet,
    VMECJAXSpectraxTransportObjective,
    VMECJAXTransportObjectiveConfig,
)
from spectraxgk.solver_objective_gradients import SOLVER_OBJECTIVE_NAMES  # noqa: E402
from spectraxgk.stellarator_objective_portfolio import (  # noqa: E402
    aggregate_objective_portfolio,
    portfolio_objective_weight_vector,
    portfolio_sample_weight_tensor,
)
from spectraxgk.vmec_jax_transport_objective import VMECJAXTransportObjectiveTransform  # noqa: E402
from spectraxgk.vmec_jax_transport_objective import (  # noqa: E402
    _apply_objective_transform,
    _reference_wout_from_context,
    _solver_table_to_nonlinear_window_proxy,
    _static_grid_options_from_ky_values,
    _transport_feature_table_from_state,
)


DEFAULT_SURFACES = (0.45, 0.64, 0.78)
DEFAULT_ALPHAS = (0.0, 0.7853981633974483)
DEFAULT_KY_VALUES = (0.10, 0.30, 0.50)
DEFAULT_TRANSPORT_KINDS = ("growth", "quasilinear_flux", "nonlinear_window_heat_flux")


def _float_tuple(raw: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    if not all(np.isfinite(value) for value in values):
        raise argparse.ArgumentTypeError("values must be finite")
    return values


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_report(
    *,
    input_path: Path,
    max_mode: int,
    min_vmec_mode: int,
    transport_kind: str,
    sample_set: StellaratorITGSampleSet,
    config: VMECJAXTransportObjectiveConfig,
    metric: float,
    solver_device: str | None,
    inner_max_iter: int,
    inner_ftol: float,
    trial_max_iter: int,
    trial_ftol: float,
    sample_statistics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a JSON-safe metric report."""

    return {
        "kind": "vmec_jax_spectrax_transport_metric_eval",
        "claim_scope": (
            "SPECTRAX-GK transport metric evaluated on a supplied VMEC-JAX input; "
            "this is not an optimization and does not imply nonlinear turbulent-flux promotion"
        ),
        "input": str(input_path),
        "max_mode": int(max_mode),
        "min_vmec_mode": int(min_vmec_mode),
        "transport_kind": str(transport_kind),
        "transport_metric_kind": str(transport_kind),
        "transport_objective_final": float(metric),
        "spectrax_objective_final": float(metric),
        "transport_metric_final": float(metric),
        "transport_objective_source": "vmec_jax_input_final_state_eval_only",
        "sample_set": sample_set.to_dict(),
        "spectrax_config": {
            "ntheta": int(config.ntheta),
            "mboz": int(config.mboz),
            "nboz": int(config.nboz),
            "n_laguerre": int(config.n_laguerre),
            "n_hermite": int(config.n_hermite),
            "objective_transform": str(config.objective_transform),
            "objective_scale": float(config.objective_scale),
            "surface_chunk_size": int(config.surface_chunk_size),
            "gradient_scope": config.gradient_scope,
        },
        "solver": {
            "device": solver_device,
            "inner_max_iter": int(inner_max_iter),
            "inner_ftol": float(inner_ftol),
            "trial_max_iter": int(trial_max_iter),
            "trial_ftol": float(trial_ftol),
        },
        "sample_statistics": sample_statistics,
        "next_action": (
            "use this metric only for reduced-objective admission; matched long-window "
            "nonlinear audits remain required for turbulent-flux claims"
        ),
    }


def _objective_table_from_feature_table(
    table: Any,
    config: VMECJAXTransportObjectiveConfig,
) -> Any:
    if config.kind == "nonlinear_window_heat_flux":
        return _solver_table_to_nonlinear_window_proxy(table, config)[..., None]
    if config.kind == "growth":
        return table[..., SOLVER_OBJECTIVE_NAMES.index("gamma")][..., None]
    return table[..., SOLVER_OBJECTIVE_NAMES.index("mixing_length_heat_flux_proxy")][..., None]


def _metric_from_objective_table(
    objective_table: Any,
    config: VMECJAXTransportObjectiveConfig,
) -> float:
    samples = config.sample_set
    weights = (1.0,) if config.objective_weights is None else config.objective_weights
    raw = aggregate_objective_portfolio(
        objective_table,
        surface_weights=samples.surface_weights,
        alpha_weights=samples.alpha_weights,
        ky_weights=samples.ky_weights,
        objective_weights=weights,
        reduction=samples.reduction,
    )
    return float(np.asarray(_apply_objective_transform(raw, config)))


def sample_statistics_from_objective_table(
    *,
    objective_table: Any,
    config: VMECJAXTransportObjectiveConfig,
    include_rows: bool = False,
) -> dict[str, Any]:
    """Return deterministic sample-spread diagnostics for the reduced objective.

    The returned standard error is the weighted sample dispersion across the
    configured surface/field-line/``k_y`` grid. It is not a stochastic error bar
    and must not be presented as nonlinear heat-flux uncertainty.
    """

    samples = config.sample_set
    if samples.reduction not in ("weighted_mean", "mean"):
        return {
            "claim_scope": "sample_spread_not_defined_for_non_mean_reduction",
            "reduction": samples.reduction,
        }
    if samples.reduction == "mean":
        weights = np.full(objective_table.shape[:-1], 1.0 / float(np.prod(objective_table.shape[:-1])), dtype=float)
        objective_weights = np.full((int(objective_table.shape[-1]),), 1.0 / float(objective_table.shape[-1]), dtype=float)
    else:
        weights = np.asarray(
            portfolio_sample_weight_tensor(
                objective_table,
                surface_weights=samples.surface_weights,
                alpha_weights=samples.alpha_weights,
                ky_weights=samples.ky_weights,
            ),
            dtype=float,
        )
        objective_weights = np.asarray(
            portfolio_objective_weight_vector(
                objective_table,
                objective_weights=config.objective_weights,
            ),
            dtype=float,
        )
    per_sample = np.sum(np.asarray(objective_table, dtype=float) * objective_weights, axis=-1)
    if not np.all(np.isfinite(per_sample)):
        raise RuntimeError("non-finite reduced objective sample value")
    mean = float(np.sum(weights * per_sample))
    variance = float(np.sum(weights * (per_sample - mean) ** 2))
    std = float(np.sqrt(max(0.0, variance)))
    n_samples = int(per_sample.size)
    sem = float(std / np.sqrt(max(1, n_samples)))
    payload: dict[str, Any] = {
        "claim_scope": (
            "deterministic cross-sample dispersion over the configured surface/field-line/ky grid; "
            "not stochastic uncertainty and not a nonlinear heat-flux SEM"
        ),
        "reduction": samples.reduction,
        "n_samples": n_samples,
        "weighted_mean": mean,
        "weighted_std": std,
        "weighted_standard_error": sem,
        "min": float(np.min(per_sample)),
        "max": float(np.max(per_sample)),
        "rows_included": bool(include_rows),
    }
    if include_rows:
        rows: list[dict[str, Any]] = []
        for i_surface, surface in enumerate(samples.surfaces):
            for i_alpha, alpha in enumerate(samples.alphas):
                for i_ky, ky in enumerate(samples.ky_values):
                    rows.append(
                        {
                            "surface": float(surface),
                            "alpha": float(alpha),
                            "ky": float(ky),
                            "value": float(per_sample[i_surface, i_alpha, i_ky]),
                            "weight": float(weights[i_surface, i_alpha, i_ky]),
                        }
                    )
        payload["rows"] = rows
    return payload


def sample_statistics_from_state(
    *,
    ctx: Any,
    state: Any,
    config: VMECJAXTransportObjectiveConfig,
    wout_reference: Any | None = None,
    include_rows: bool = False,
) -> dict[str, Any]:
    """Return deterministic sample-spread diagnostics for the reduced objective."""

    samples = config.sample_set
    grid_options = _static_grid_options_from_ky_values(
        samples.ky_values,
        min_ny=int(config.ny),
    )
    table = _transport_feature_table_from_state(
        state,
        ctx.static,
        ctx.indata,
        wout_reference if wout_reference is not None else _reference_wout_from_context(ctx),
        config,
        grid_options,
    )
    return sample_statistics_from_objective_table(
        objective_table=_objective_table_from_feature_table(table, config),
        config=config,
        include_rows=include_rows,
    )


def _sample_set_from_args(args: argparse.Namespace) -> StellaratorITGSampleSet:
    return StellaratorITGSampleSet(
        surfaces=tuple(float(x) for x in args.surfaces),
        alphas=tuple(float(x) for x in args.alphas),
        ky_values=tuple(float(x) for x in args.ky_values),
    )


def _config_from_args(
    args: argparse.Namespace,
    *,
    transport_kind: str,
    sample_set: StellaratorITGSampleSet,
) -> VMECJAXTransportObjectiveConfig:
    return VMECJAXTransportObjectiveConfig(
        kind=transport_kind,
        sample_set=sample_set,
        ntheta=int(args.ntheta),
        mboz=int(args.mboz),
        nboz=int(args.nboz),
        n_laguerre=int(args.n_laguerre),
        n_hermite=int(args.n_hermite),
        objective_transform=cast(VMECJAXTransportObjectiveTransform, str(args.spectrax_objective_transform)),
        objective_scale=float(args.spectrax_objective_scale),
        surface_chunk_size=int(args.surface_chunk_size),
    )


def evaluate_metrics(args: argparse.Namespace, kinds: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    """Evaluate one VMEC state and return reports for several transport metrics."""

    import vmec_jax as vj
    import vmec_jax.optimization_workflow as workflow

    sample_set = _sample_set_from_args(args)
    base_kind = "quasilinear_flux" if any(kind != "growth" for kind in kinds) else "growth"
    base_config = _config_from_args(args, transport_kind=base_kind, sample_set=sample_set)
    objective = VMECJAXSpectraxTransportObjective(config=base_config)
    vmec = vj.FixedBoundaryVMEC.from_input(
        args.input,
        max_mode=int(args.max_mode),
        min_vmec_mode=int(args.min_vmec_mode),
        output_dir=args.outdir,
    )
    problem = vj.LeastSquaresProblem.from_tuples([(objective.J, 0.0, 1.0)])
    stage = workflow.build_fixed_boundary_objective_stage(
        vmec.cfg,
        vmec.indata,
        stage_mode=int(args.max_mode),
        objectives=problem.objective_terms,
        include=vmec.include,
        fix=vmec.fix,
        inner_max_iter=int(args.inner_max_iter),
        inner_ftol=float(args.inner_ftol),
        trial_max_iter=int(args.trial_max_iter),
        trial_ftol=float(args.trial_ftol),
        solver_device=args.solver_device,
    )
    params0 = np.zeros(len(stage.specs), dtype=float)
    # The public VMEC-JAX workflow currently exposes final states through the
    # optimizer path.  For eval-only admission bookkeeping, solve the supplied
    # boundary once and call the SPECTRAX objective directly; no boundary update
    # or least-squares step is taken.
    state = stage.optimizer._solve_forward(params0, trial=False)  # noqa: SLF001
    grid_options = _static_grid_options_from_ky_values(
        sample_set.ky_values,
        min_ny=int(base_config.ny),
    )
    table = _transport_feature_table_from_state(
        state=state,
        static=stage.ctx.static,
        indata=stage.ctx.indata,
        wout_reference=objective.wout_reference
        if objective.wout_reference is not None
        else _reference_wout_from_context(stage.ctx),
        config=base_config,
        grid_options=grid_options,
    )
    reports: dict[str, dict[str, Any]] = {}
    for kind in kinds:
        config = replace(base_config, kind=kind)
        objective_table = _objective_table_from_feature_table(table, config)
        metric = _metric_from_objective_table(objective_table, config)
        if not np.isfinite(metric):
            raise RuntimeError(f"non-finite SPECTRAX transport metric for {kind}: {metric!r}")
        sample_statistics = sample_statistics_from_objective_table(
            objective_table=objective_table,
            config=config,
            include_rows=bool(args.include_sample_rows),
        )
        reports[kind] = build_report(
            input_path=Path(args.input),
            max_mode=int(args.max_mode),
            min_vmec_mode=int(args.min_vmec_mode),
            transport_kind=str(kind),
            sample_set=sample_set,
            config=config,
            metric=metric,
            solver_device=args.solver_device,
            inner_max_iter=int(args.inner_max_iter),
            inner_ftol=float(args.inner_ftol),
            trial_max_iter=int(args.trial_max_iter),
            trial_ftol=float(args.trial_ftol),
            sample_statistics=sample_statistics,
        )
    return reports


def evaluate_metric(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate the configured transport metric on the supplied VMEC input."""

    return evaluate_metrics(args, (str(args.transport_kind),))[str(args.transport_kind)]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Solved VMEC input deck, typically input.final")
    parser.add_argument("--out-json", required=True, type=Path, help="JSON sidecar to write")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "tools_out" / "vmec_jax_transport_metric_eval",
        help="Scratch VMEC-JAX output directory used while building the state",
    )
    parser.add_argument("--max-mode", type=int, default=5)
    parser.add_argument("--min-vmec-mode", type=int, default=7)
    parser.add_argument(
        "--transport-kind",
        choices=(*DEFAULT_TRANSPORT_KINDS, "all"),
        default="growth",
    )
    parser.add_argument(
        "--out-json-dir",
        type=Path,
        default=None,
        help="When --transport-kind=all, also write one JSON report per metric kind into this directory.",
    )
    parser.add_argument("--surfaces", type=_float_tuple, default=DEFAULT_SURFACES)
    parser.add_argument("--alphas", type=_float_tuple, default=DEFAULT_ALPHAS)
    parser.add_argument("--ky-values", type=_float_tuple, default=DEFAULT_KY_VALUES)
    parser.add_argument("--ntheta", type=int, default=24)
    parser.add_argument("--mboz", type=int, default=21)
    parser.add_argument("--nboz", type=int, default=21)
    parser.add_argument("--n-laguerre", type=int, default=2)
    parser.add_argument("--n-hermite", type=int, default=3)
    parser.add_argument(
        "--surface-chunk-size",
        type=int,
        default=0,
        help="Evaluate the transport objective in surface chunks before applying the scalar transform",
    )
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
    parser.add_argument(
        "--include-sample-rows",
        action="store_true",
        help="Store every deterministic surface/alpha/ky sample value, not only summary spread statistics.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.transport_kind == "all":
        reports = evaluate_metrics(args, DEFAULT_TRANSPORT_KINDS)
        if args.out_json_dir is not None:
            args.out_json_dir.mkdir(parents=True, exist_ok=True)
            for kind, report in reports.items():
                (args.out_json_dir / f"{kind}.json").write_text(
                    json.dumps(_json_safe(report), indent=2, allow_nan=False),
                    encoding="utf-8",
                )
        report = {
            "kind": "vmec_jax_spectrax_transport_metric_eval_batch",
            "input": str(args.input),
            "transport_kinds": list(DEFAULT_TRANSPORT_KINDS),
            "reports": reports,
        }
    else:
        report = evaluate_metric(args)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(_json_safe(report), indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(_json_safe(report), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
