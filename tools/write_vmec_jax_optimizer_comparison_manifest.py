#!/usr/bin/env python3
"""Write a reproducible VMEC-JAX QA optimizer-comparison manifest.

The manifest keeps optimizer comparisons honest by generating matched launch
commands from one policy block.  Runnable commands are emitted for the
currently supported deterministic VMEC-JAX/SPECTRAX-GK driver methods.  Noisy
outer-loop methods such as SPSA, CMA-ES, and Bayesian optimization are emitted
as explicit campaign contracts with deterministic evaluation/audit command
templates, because their proposal generators are intentionally external to the
VMEC-JAX least-squares driver.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DRIVER = Path("tools/vmec_jax_qa_low_turbulence_optimization.py")
METRIC_EVAL = Path("tools/evaluate_vmec_jax_spectrax_transport_metric.py")
AUDIT_WRITER = Path("tools/write_optimized_equilibrium_transport_configs.py")

TRANSPORT_KINDS = ("growth", "quasilinear_flux", "nonlinear_window_heat_flux")
RUNNABLE_METHODS = ("scipy", "scalar_trust", "lbfgs_adjoint")
OUTER_LOOP_METHODS = ("spsa", "cma_es", "bo")


def _csv(values: tuple[float, ...]) -> str:
    return ",".join(f"{value:.16g}" for value in values)


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(ROOT.resolve(strict=False)).as_posix()
    except ValueError:
        return path.as_posix()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return _repo_relative(value)
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def _fingerprint(payload: dict[str, Any]) -> str:
    raw = json.dumps(_json_ready(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _float_tuple(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return values


def _method_args(args: argparse.Namespace, method: str) -> list[str]:
    if method == "scipy":
        out = ["--method", "scipy", "--scipy-tr-solver", str(args.scipy_tr_solver)]
        if args.scipy_lsmr_maxiter is not None:
            out += ["--scipy-lsmr-maxiter", str(args.scipy_lsmr_maxiter)]
        return out
    if method in {"scalar_trust", "lbfgs_adjoint"}:
        return ["--method", method]
    raise ValueError(f"unknown runnable optimizer method {method!r}")


def _baseline_command(args: argparse.Namespace, outdir: Path) -> str:
    cmd = [
        "python",
        _repo_relative(ROOT / DRIVER),
        "--strict-upstream-qa-baseline",
        "--outdir",
        _repo_relative(outdir),
        "--solver-device",
        str(args.solver_device),
        "--max-nfev",
        str(args.baseline_max_nfev),
        "--continuation-nfev",
        str(args.continuation_nfev),
        "--inner-max-iter",
        str(args.inner_max_iter),
        "--trial-max-iter",
        str(args.trial_max_iter),
        "--make-plots",
    ]
    return shlex.join(cmd)


def _transport_command(
    args: argparse.Namespace,
    *,
    method: str,
    transport_kind: str,
    input_path: Path,
    outdir: Path,
) -> str:
    cmd = [
        "python",
        _repo_relative(ROOT / DRIVER),
        "--input",
        _repo_relative(input_path),
        "--outdir",
        _repo_relative(outdir),
        "--max-mode",
        str(args.max_mode),
        "--min-vmec-mode",
        str(args.min_vmec_mode),
        "--disable-mode-continuation",
        "--target-aspect",
        f"{args.target_aspect:.16g}",
        "--min-iota",
        f"{args.min_iota:.16g}",
        "--iota-objective",
        "target",
        "--disable-iota-profile-floor",
        "--aspect-weight",
        f"{args.aspect_weight:.16g}",
        "--iota-floor-weight",
        f"{args.iota_weight:.16g}",
        "--qs-weight",
        f"{args.qs_weight:.16g}",
        "--spectrax-weight",
        f"{args.spectrax_weight:.16g}",
        "--transport-kind",
        transport_kind,
        "--surfaces",
        _csv(args.surfaces),
        "--alphas",
        _csv(args.alphas),
        "--ky-values",
        _csv(args.ky_values),
        "--ntheta",
        str(args.ntheta),
        "--mboz",
        str(args.mboz),
        "--nboz",
        str(args.nboz),
        "--n-laguerre",
        str(args.n_laguerre),
        "--n-hermite",
        str(args.n_hermite),
        "--surface-chunk-size",
        str(args.surface_chunk_size),
        "--spectrax-objective-transform",
        str(args.spectrax_objective_transform),
        "--spectrax-objective-scale",
        f"{args.spectrax_objective_scale:.16g}",
        "--max-nfev",
        str(args.max_nfev),
        "--continuation-nfev",
        str(args.continuation_nfev),
        "--inner-max-iter",
        str(args.inner_max_iter),
        "--trial-max-iter",
        str(args.trial_max_iter),
        "--inner-ftol",
        f"{args.inner_ftol:.16g}",
        "--trial-ftol",
        f"{args.trial_ftol:.16g}",
        "--ftol",
        f"{args.ftol:.16g}",
        "--gtol",
        f"{args.gtol:.16g}",
        "--xtol",
        f"{args.xtol:.16g}",
        "--solver-device",
        str(args.solver_device),
        "--save-rerun-wouts",
        "--admit-authoritative-rerun-wout",
        "--allow-failed-solved-wout-gate",
        "--make-plots",
    ]
    cmd += _method_args(args, method)
    return shlex.join(cmd)


def _audit_command(args: argparse.Namespace, *, case_id: str, wout_path: Path, outdir: Path) -> str:
    cmd = [
        "python",
        _repo_relative(ROOT / AUDIT_WRITER),
        "--vmec-file",
        _repo_relative(wout_path),
        "--case",
        f"vmec_qa_optimizer_comparison_{case_id}",
        "--out-dir",
        _repo_relative(outdir),
        "--horizons",
        str(args.audit_horizons),
        "--grid",
        str(args.audit_grid),
        "--window-tmin",
        f"{args.audit_window_tmin:.16g}",
        "--window-tmax",
        f"{args.audit_window_tmax:.16g}",
        "--dt-variant",
        f"{args.audit_dt_variant:.16g}",
    ]
    for seed in args.audit_seed_variants:
        cmd += ["--seed-variant", str(seed)]
    return shlex.join(cmd)


def _metric_eval_template(args: argparse.Namespace, *, outdir: Path, transport_kind: str) -> str:
    cmd = [
        "python",
        _repo_relative(ROOT / METRIC_EVAL),
        "--input",
        "{candidate_input_final}",
        "--out-json",
        str(outdir / "{candidate_id}" / f"{transport_kind}_metric.json"),
        "--outdir",
        str(outdir / "{candidate_id}" / "metric_eval_scratch"),
        "--out-wout",
        str(outdir / "{candidate_id}" / "wout_final_rerun.nc"),
        "--max-mode",
        str(args.max_mode),
        "--min-vmec-mode",
        str(args.min_vmec_mode),
        "--transport-kind",
        transport_kind,
        "--surfaces",
        _csv(args.surfaces),
        "--alphas",
        _csv(args.alphas),
        "--ky-values",
        _csv(args.ky_values),
        "--ntheta",
        str(args.ntheta),
        "--mboz",
        str(args.mboz),
        "--nboz",
        str(args.nboz),
        "--n-laguerre",
        str(args.n_laguerre),
        "--n-hermite",
        str(args.n_hermite),
        "--surface-chunk-size",
        str(args.surface_chunk_size),
        "--spectrax-objective-transform",
        str(args.spectrax_objective_transform),
        "--spectrax-objective-scale",
        f"{args.spectrax_objective_scale:.16g}",
        "--solver-device",
        str(args.solver_device),
        "--include-sample-rows",
    ]
    return shlex.join(cmd)


def _case_id(transport_kind: str, method: str) -> str:
    return f"{transport_kind}_{method}_from_strict_baseline".replace("quasilinear_flux", "quasilinear").replace(
        "nonlinear_window_heat_flux", "nonlinear_window"
    )


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    campaign_root = args.campaign_root
    runs_root = campaign_root / "runs"
    baseline_outdir = runs_root / "qa_baseline_scipy"
    baseline_input = baseline_outdir / "input.final"
    baseline_wout = baseline_outdir / "wout_final_rerun.nc"
    sample_policy = {
        "max_mode": args.max_mode,
        "min_vmec_mode": args.min_vmec_mode,
        "target_aspect": args.target_aspect,
        "min_iota": args.min_iota,
        "surfaces": args.surfaces,
        "alphas": args.alphas,
        "ky_values": args.ky_values,
        "ntheta": args.ntheta,
        "mboz": args.mboz,
        "nboz": args.nboz,
        "n_laguerre": args.n_laguerre,
        "n_hermite": args.n_hermite,
        "surface_chunk_size": args.surface_chunk_size,
        "spectrax_objective_transform": args.spectrax_objective_transform,
        "spectrax_objective_scale": args.spectrax_objective_scale,
    }
    optimizer_budget = {
        "baseline_max_nfev": args.baseline_max_nfev,
        "max_nfev": args.max_nfev,
        "continuation_nfev": args.continuation_nfev,
        "inner_max_iter": args.inner_max_iter,
        "trial_max_iter": args.trial_max_iter,
        "inner_ftol": args.inner_ftol,
        "trial_ftol": args.trial_ftol,
        "ftol": args.ftol,
        "gtol": args.gtol,
        "xtol": args.xtol,
        "solver_device": args.solver_device,
    }
    comparison_policy = {
        "sample_policy": sample_policy,
        "optimizer_budget": optimizer_budget,
        "transport_weight": args.spectrax_weight,
        "claim_boundary": (
            "optimizer outputs are candidate-generation evidence only; promote turbulent-flux "
            "reduction only after matched long post-transient nonlinear audits and convergence gates"
        ),
    }
    entries: list[dict[str, Any]] = []
    baseline_entry = {
        "case_id": "qa_baseline_scipy",
        "kind": "strict_constraints_baseline",
        "method": "scipy",
        "transport_kind": None,
        "status": "runnable",
        "command": _baseline_command(args, baseline_outdir),
        "outdir": baseline_outdir,
        "expected_input_final": baseline_input,
        "expected_authoritative_wout": baseline_wout,
    }
    entries.append(baseline_entry)
    for transport_kind in args.transport_kinds:
        for method in args.runnable_methods:
            case_id = _case_id(transport_kind, method)
            outdir = runs_root / case_id
            audit_outdir = campaign_root / "nonlinear_audits" / case_id
            wout_path = outdir / "wout_final_rerun.nc"
            entries.append(
                {
                    "case_id": case_id,
                    "kind": "deterministic_transport_optimizer",
                    "method": method,
                    "transport_kind": transport_kind,
                    "status": "runnable",
                    "depends_on": "qa_baseline_scipy",
                    "command": _transport_command(
                        args,
                        method=method,
                        transport_kind=transport_kind,
                        input_path=baseline_input,
                        outdir=outdir,
                    ),
                    "outdir": outdir,
                    "expected_authoritative_wout": wout_path,
                    "recommended_nonlinear_audit_command": _audit_command(
                        args,
                        case_id=case_id,
                        wout_path=wout_path,
                        outdir=audit_outdir,
                    ),
                }
            )
        derivative_free_root = campaign_root / "outer_loop_candidates" / transport_kind
        for method in args.outer_loop_methods:
            entries.append(
                {
                    "case_id": f"{transport_kind}_{method}_outer_loop",
                    "kind": "derivative_free_outer_loop_contract",
                    "method": method,
                    "transport_kind": transport_kind,
                    "status": "planned_outer_loop",
                    "depends_on": "qa_baseline_scipy",
                    "candidate_generator_required": True,
                    "candidate_contract": {
                        "candidate_input_placeholder": "{candidate_input_final}",
                        "candidate_id_placeholder": "{candidate_id}",
                        "metric_eval_command_template": _metric_eval_template(
                            args,
                            outdir=derivative_free_root,
                            transport_kind=transport_kind,
                        ),
                        "nonlinear_audit_command_template": _audit_command(
                            args,
                            case_id=f"{transport_kind}_{method}" + "_{candidate_id}",
                            wout_path=derivative_free_root / "{candidate_id}" / "wout_final_rerun.nc",
                            outdir=derivative_free_root / "{candidate_id}" / "nonlinear_audit",
                        ),
                    },
                    "notes": (
                        "Use this entry for SPSA/CMA/BO candidate proposal loops. "
                        "Each proposed input.final must be evaluated with the metric command, "
                        "then admitted candidates receive the same strict nonlinear audit command."
                    ),
                }
            )
    manifest = {
        "schema_version": 1,
        "kind": "vmec_jax_qa_optimizer_comparison_manifest",
        "campaign_root": campaign_root,
        "driver": DRIVER,
        "metric_eval_tool": METRIC_EVAL,
        "audit_writer": AUDIT_WRITER,
        "comparison_policy": comparison_policy,
        "comparison_fingerprint": _fingerprint(comparison_policy),
        "entries": entries,
        "runnable_commands": [entry["command"] for entry in entries if entry.get("status") == "runnable"],
        "next_actions": [
            "Run qa_baseline_scipy first and verify wout_final_rerun.nc passes admission gates.",
            "Run deterministic transport optimizer entries only after the strict baseline input.final exists.",
            "Build matched nonlinear audit configs from the emitted audit commands before making transport claims.",
            "Use planned_outer_loop entries only with an explicit proposal generator and the same metric/audit policy.",
        ],
    }
    return _json_ready(manifest)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-root",
        type=Path,
        default=ROOT / "tools_out" / "vmec_jax_qa_optimizer_comparison_campaign",
        help="Campaign root where run commands will write outputs",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=ROOT / "tools_out" / "vmec_jax_qa_optimizer_comparison_manifest.json",
        help="Manifest JSON path",
    )
    parser.add_argument("--max-mode", type=int, default=5)
    parser.add_argument("--min-vmec-mode", type=int, default=7)
    parser.add_argument("--target-aspect", type=float, default=5.0)
    parser.add_argument("--min-iota", type=float, default=0.41)
    parser.add_argument("--aspect-weight", type=float, default=1.0)
    parser.add_argument("--iota-weight", type=float, default=10_000.0)
    parser.add_argument("--qs-weight", type=float, default=1.0)
    parser.add_argument("--spectrax-weight", type=float, default=0.1)
    parser.add_argument("--transport-kinds", type=str, default=",".join(TRANSPORT_KINDS))
    parser.add_argument("--runnable-methods", type=str, default=",".join(RUNNABLE_METHODS))
    parser.add_argument("--outer-loop-methods", type=str, default=",".join(OUTER_LOOP_METHODS))
    parser.add_argument("--surfaces", type=_float_tuple, default=(0.64,))
    parser.add_argument("--alphas", type=_float_tuple, default=(0.0,))
    parser.add_argument("--ky-values", type=_float_tuple, default=(0.3,))
    parser.add_argument("--ntheta", type=int, default=24)
    parser.add_argument("--mboz", type=int, default=21)
    parser.add_argument("--nboz", type=int, default=21)
    parser.add_argument("--n-laguerre", type=int, default=2)
    parser.add_argument("--n-hermite", type=int, default=3)
    parser.add_argument("--surface-chunk-size", type=int, default=1)
    parser.add_argument("--spectrax-objective-transform", choices=("raw", "scaled", "log1p"), default="log1p")
    parser.add_argument("--spectrax-objective-scale", type=float, default=1.0)
    parser.add_argument("--baseline-max-nfev", type=int, default=80)
    parser.add_argument("--max-nfev", type=int, default=60)
    parser.add_argument("--continuation-nfev", type=int, default=25)
    parser.add_argument("--inner-max-iter", type=int, default=140)
    parser.add_argument("--trial-max-iter", type=int, default=140)
    parser.add_argument("--inner-ftol", type=float, default=1.0e-9)
    parser.add_argument("--trial-ftol", type=float, default=1.0e-9)
    parser.add_argument("--ftol", type=float, default=1.0e-7)
    parser.add_argument("--gtol", type=float, default=1.0e-7)
    parser.add_argument("--xtol", type=float, default=1.0e-8)
    parser.add_argument("--solver-device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--scipy-tr-solver", choices=("exact", "lsmr"), default="lsmr")
    parser.add_argument("--scipy-lsmr-maxiter", type=int, default=200)
    parser.add_argument("--audit-horizons", default="700,1100,1500")
    parser.add_argument("--audit-grid", default="n64:64:64:40:40")
    parser.add_argument("--audit-window-tmin", type=float, default=1100.0)
    parser.add_argument("--audit-window-tmax", type=float, default=1500.0)
    parser.add_argument("--audit-dt-variant", type=float, default=0.04)
    parser.add_argument("--audit-seed-variant", dest="audit_seed_variants", type=int, action="append")
    args = parser.parse_args(argv)
    args.transport_kinds = tuple(part.strip() for part in args.transport_kinds.split(",") if part.strip())
    args.runnable_methods = tuple(part.strip() for part in args.runnable_methods.split(",") if part.strip())
    args.outer_loop_methods = tuple(part.strip() for part in args.outer_loop_methods.split(",") if part.strip())
    unknown_kinds = set(args.transport_kinds) - set(TRANSPORT_KINDS)
    unknown_methods = set(args.runnable_methods) - set(RUNNABLE_METHODS)
    unknown_outer = set(args.outer_loop_methods) - set(OUTER_LOOP_METHODS)
    if unknown_kinds:
        parser.error(f"unknown transport kinds: {sorted(unknown_kinds)}")
    if unknown_methods:
        parser.error(f"unknown runnable methods: {sorted(unknown_methods)}")
    if unknown_outer:
        parser.error(f"unknown outer-loop methods: {sorted(unknown_outer)}")
    if args.audit_seed_variants is None:
        args.audit_seed_variants = [32, 33]
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_manifest(args)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out_json": _repo_relative(args.out_json), "comparison_fingerprint": manifest["comparison_fingerprint"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
