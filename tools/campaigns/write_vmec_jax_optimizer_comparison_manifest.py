#!/usr/bin/env python3
"""Write matched current-VMEC-JAX QA transport campaign commands.

Deterministic candidates use the optimizer owned by current VMEC-JAX. Growth
uses its implicit equilibrium Jacobian; eigenvector-weighted quasilinear and
reduced nonlinear objectives use finite-difference outer Jacobians. SPSA,
CMA-ES, and Bayesian optimization remain explicit outer-loop contracts for
noisy long-window transport rather than pretending to be VMEC-JAX methods.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DRIVER = Path("tools/campaigns/vmec_jax_qa_low_turbulence_optimization.py")
METRIC_EVAL = Path("tools/campaigns/evaluate_vmec_jax_spectrax_transport_metric.py")
AUDIT_WRITER = Path("tools/campaigns/write_optimized_equilibrium_transport_configs.py")
SPSA_WRITER = Path("tools/campaigns/write_vmec_jax_spsa_candidate_campaign.py")
TRANSPORT_KINDS = ("growth", "quasilinear_flux", "nonlinear_window_heat_flux")
OUTER_LOOP_METHODS = ("spsa", "cma_es", "bo")


def _csv(values: tuple[float, ...]) -> str:
    return ",".join(f"{value:.16g}" for value in values)


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(ROOT).as_posix()
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
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _float_tuple(raw: str) -> tuple[float, ...]:
    values = tuple(float(item) for item in raw.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated values")
    return values


def _case_id(kind: str) -> str:
    return {
        "growth": "growth_implicit_from_qa_baseline",
        "quasilinear_flux": "quasilinear_finite_difference_from_qa_baseline",
        "nonlinear_window_heat_flux": "nonlinear_window_finite_difference_from_qa_baseline",
    }[kind]


def _baseline_command(args: argparse.Namespace, outdir: Path) -> str:
    return shlex.join(
        [
            "python3",
            str(DRIVER),
            "--constraints-only",
            "--outdir",
            _repo_relative(outdir),
            "--target-aspect",
            f"{args.target_aspect:.16g}",
            "--target-iota",
            f"{args.target_iota:.16g}",
            "--mode-schedule",
            str(args.baseline_mode_schedule),
            "--max-nfev",
            str(args.baseline_max_nfev),
            "--ftol",
            f"{args.ftol:.16g}",
            "--solver-device",
            args.solver_device,
            "--make-plots",
        ]
    )


def _transport_command(
    args: argparse.Namespace, *, kind: str, input_path: Path, outdir: Path
) -> str:
    jacobian = "implicit" if kind == "growth" else "finite-difference"
    return shlex.join(
        [
            "python3",
            str(DRIVER),
            "--input",
            _repo_relative(input_path),
            "--outdir",
            _repo_relative(outdir),
            "--transport-kind",
            kind,
            "--transport-weight",
            f"{args.transport_weight:.16g}",
            "--jacobian",
            jacobian,
            "--target-aspect",
            f"{args.target_aspect:.16g}",
            "--target-iota",
            f"{args.target_iota:.16g}",
            "--mode-schedule",
            str(args.candidate_mode_schedule),
            "--surfaces",
            _csv(args.surfaces),
            "--alphas",
            _csv(args.alphas),
            "--ky-values",
            _csv(args.ky_values),
            "--ntheta",
            str(args.ntheta),
            "--n-laguerre",
            str(args.n_laguerre),
            "--n-hermite",
            str(args.n_hermite),
            "--objective-transform",
            args.objective_transform,
            "--objective-scale",
            f"{args.objective_scale:.16g}",
            "--max-nfev",
            str(args.max_nfev),
            "--ftol",
            f"{args.ftol:.16g}",
            "--solver-device",
            args.solver_device,
            "--make-plots",
        ]
    )


def _audit_command(
    args: argparse.Namespace, *, case_id: str, wout_path: Path, outdir: Path
) -> str:
    command = [
        "python3",
        str(AUDIT_WRITER),
        "--vmec-file",
        _repo_relative(wout_path),
        "--case",
        f"vmec_qa_optimizer_comparison_{case_id}",
        "--out-dir",
        _repo_relative(outdir),
        "--horizons",
        args.audit_horizons,
        "--grid",
        args.audit_grid,
        "--window-tmin",
        f"{args.audit_window_tmin:.16g}",
        "--window-tmax",
        f"{args.audit_window_tmax:.16g}",
        "--dt-variant",
        f"{args.audit_dt_variant:.16g}",
    ]
    for seed in args.audit_seed_variants:
        command.extend(("--seed-variant", str(seed)))
    return shlex.join(command)


def _metric_template(args: argparse.Namespace, *, outdir: Path, kind: str) -> str:
    return shlex.join(
        [
            "python3",
            str(METRIC_EVAL),
            "--input",
            "{candidate_input_final}",
            "--out-json",
            str(outdir / "{candidate_id}" / f"{kind}_metric.json"),
            "--out-wout",
            str(outdir / "{candidate_id}" / "wout_final.nc"),
            "--transport-kind",
            kind,
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
            "--spectrax-objective-transform",
            args.objective_transform,
            "--spectrax-objective-scale",
            f"{args.objective_scale:.16g}",
            "--solver-device",
            args.solver_device,
            "--include-sample-rows",
        ]
    )


def _outer_loop_strategy(method: str, kind: str) -> dict[str, Any]:
    return {
        "stage": "noisy_long_window_transport",
        "method_family": (
            "stochastic_common_random_numbers"
            if method == "spsa"
            else "derivative_free_low_dimensional_search"
        ),
        "transport_kind": kind,
        "requires_common_random_numbers": True,
        "requires_matched_t1500_replicated_audit": True,
        "claim_scope": "candidate generation only until matched nonlinear gates pass",
    }


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    runs = args.campaign_root / "runs"
    baseline_dir = runs / "qa_baseline"
    baseline_input = baseline_dir / "input.final"
    policy = {
        "api": "current_vmec_jax_opt_least_squares",
        "sample_policy": {
            "surfaces": args.surfaces,
            "alphas": args.alphas,
            "ky_values": args.ky_values,
            "ntheta": args.ntheta,
            "mboz": args.mboz,
            "nboz": args.nboz,
        },
        "derivative_policy": {
            "growth": "implicit equilibrium Jacobian",
            "quasilinear_flux": "finite-difference outer Jacobian",
            "nonlinear_window_heat_flux": "finite-difference outer Jacobian",
        },
        "landscape_policy": {
            "rbc11_points_admit_optimized_candidates": False,
            "role": "conditioning, noise, and convergence-window diagnosis",
        },
        "claim_boundary": (
            "optimizer outputs are candidates; nonlinear reduction requires matched "
            "replicated post-transient transport windows"
        ),
    }
    entries: list[dict[str, Any]] = [
        {
            "case_id": "qa_baseline",
            "kind": "constraints_baseline",
            "status": "runnable",
            "command": _baseline_command(args, baseline_dir),
            "outdir": baseline_dir,
            "expected_input_final": baseline_input,
            "expected_wout": baseline_dir / "wout_final.nc",
        }
    ]
    for kind in args.transport_kinds:
        case_id = _case_id(kind)
        outdir = runs / case_id
        wout = outdir / "wout_final.nc"
        entries.append(
            {
                "case_id": case_id,
                "kind": "deterministic_transport_candidate",
                "transport_kind": kind,
                "derivative_policy": policy["derivative_policy"][kind],
                "status": "runnable",
                "depends_on": "qa_baseline",
                "command": _transport_command(
                    args, kind=kind, input_path=baseline_input, outdir=outdir
                ),
                "outdir": outdir,
                "expected_wout": wout,
                "recommended_nonlinear_audit_command": _audit_command(
                    args,
                    case_id=case_id,
                    wout_path=wout,
                    outdir=args.campaign_root / "nonlinear_audits" / case_id,
                ),
            }
        )
        outer_root = args.campaign_root / "outer_loop_candidates" / kind
        for method in args.outer_loop_methods:
            entries.append(
                {
                    "case_id": f"{kind}_{method}_outer_loop",
                    "kind": "derivative_free_outer_loop_contract",
                    "transport_kind": kind,
                    "method": method,
                    "status": "planned_outer_loop",
                    "depends_on": "qa_baseline",
                    "optimizer_strategy": _outer_loop_strategy(method, kind),
                    "candidate_contract": {
                        "candidate_input_placeholder": "{candidate_input_final}",
                        "metric_eval_command_template": _metric_template(
                            args, outdir=outer_root, kind=kind
                        ),
                        "nonlinear_audit_command_template": _audit_command(
                            args,
                            case_id=f"{kind}_{method}_{{candidate_id}}",
                            wout_path=outer_root / "{candidate_id}" / "wout_final.nc",
                            outdir=outer_root / "{candidate_id}" / "nonlinear_audit",
                        ),
                    },
                }
            )
    manifest = {
        "schema_version": 2,
        "kind": "vmec_jax_qa_optimizer_comparison_manifest",
        "campaign_root": args.campaign_root,
        "driver": DRIVER,
        "comparison_policy": policy,
        "comparison_fingerprint": _fingerprint(policy),
        "entries": entries,
        "runnable_commands": [
            entry["command"] for entry in entries if entry["status"] == "runnable"
        ],
        "next_actions": [
            "Run and replay the QA baseline before transport candidates.",
            "Run deterministic candidates with the recorded derivative policy.",
            "Promote only candidates passing matched long-window nonlinear audits.",
        ],
    }
    return _json_ready(manifest)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-root", type=Path, default=ROOT / "tools_out/qa_transport_campaign"
    )
    parser.add_argument(
        "--out-json", type=Path, default=ROOT / "tools_out/qa_transport_manifest.json"
    )
    parser.add_argument("--transport-kinds", default=",".join(TRANSPORT_KINDS))
    parser.add_argument("--outer-loop-methods", default=",".join(OUTER_LOOP_METHODS))
    parser.add_argument("--target-aspect", type=float, default=6.0)
    parser.add_argument("--target-iota", type=float, default=0.42)
    parser.add_argument("--transport-weight", type=float, default=0.01)
    parser.add_argument("--baseline-mode-schedule", default="1,2,3,4,5")
    parser.add_argument("--candidate-mode-schedule", default="5")
    parser.add_argument("--surfaces", type=_float_tuple, default=(0.45, 0.64, 0.78))
    parser.add_argument(
        "--alphas", type=_float_tuple, default=(0.0, 0.7853981633974483)
    )
    parser.add_argument("--ky-values", type=_float_tuple, default=(0.1, 0.3, 0.5))
    parser.add_argument("--ntheta", type=int, default=24)
    parser.add_argument("--mboz", type=int, default=21)
    parser.add_argument("--nboz", type=int, default=21)
    parser.add_argument("--n-laguerre", type=int, default=2)
    parser.add_argument("--n-hermite", type=int, default=3)
    parser.add_argument(
        "--objective-transform", choices=("raw", "scaled", "log1p"), default="log1p"
    )
    parser.add_argument("--objective-scale", type=float, default=1.0)
    parser.add_argument("--baseline-max-nfev", type=int, default=2000)
    parser.add_argument("--max-nfev", type=int, default=200)
    parser.add_argument("--ftol", type=float, default=1.0e-6)
    parser.add_argument("--solver-device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--audit-horizons", default="700,1100,1500")
    parser.add_argument("--audit-grid", default="n64:64:64:40:40")
    parser.add_argument("--audit-window-tmin", type=float, default=1100.0)
    parser.add_argument("--audit-window-tmax", type=float, default=1500.0)
    parser.add_argument("--audit-dt-variant", type=float, default=0.04)
    parser.add_argument(
        "--audit-seed-variant", dest="audit_seed_variants", type=int, action="append"
    )
    args = parser.parse_args(argv)
    args.transport_kinds = tuple(
        item.strip() for item in args.transport_kinds.split(",") if item.strip()
    )
    args.outer_loop_methods = tuple(
        item.strip() for item in args.outer_loop_methods.split(",") if item.strip()
    )
    unknown_kinds = set(args.transport_kinds) - set(TRANSPORT_KINDS)
    unknown_methods = set(args.outer_loop_methods) - set(OUTER_LOOP_METHODS)
    if unknown_kinds:
        parser.error(f"unknown transport kinds: {sorted(unknown_kinds)}")
    if unknown_methods:
        parser.error(f"unknown outer-loop methods: {sorted(unknown_methods)}")
    if args.audit_seed_variants is None:
        args.audit_seed_variants = [32, 33]
    if args.mboz < 21 or args.nboz < 21:
        parser.error("--mboz and --nboz must be at least 21")
    if args.objective_scale <= 0.0 or args.max_nfev < 1 or args.baseline_max_nfev < 1:
        parser.error("objective scale and solve budgets must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_manifest(args)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "out_json": _repo_relative(args.out_json),
                "comparison_fingerprint": manifest["comparison_fingerprint"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
