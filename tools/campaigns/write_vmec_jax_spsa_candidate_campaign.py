#!/usr/bin/env python3
"""Write SPSA plus/minus VMEC-JAX transport-candidate commands.

This is a reproducible candidate launcher, not a promoted optimization result.
It starts from a solved QA ``input.final``, writes simultaneous plus/minus
boundary perturbations, emits SPECTRAX-GK reduced-metric evaluation commands,
and emits matched nonlinear-audit config commands that use the same seed and
timestep variants for each plus/minus pair.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import re
import shlex
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.campaigns.write_nonlinear_turbulence_gradient_campaign import _repo_relative  # noqa: E402
from tools.campaigns.write_vmec_boundary_campaigns import _coefficient_value  # noqa: E402
from tools.campaigns.write_vmec_boundary_campaigns import (
    _parse_coefficient_spec,
)  # noqa: E402
from tools.campaigns.write_vmec_boundary_campaigns import _patch_coefficient  # noqa: E402


METRIC_EVAL = Path("tools/campaigns/evaluate_vmec_jax_spectrax_transport_metric.py")
AUDIT_WRITER = Path("tools/campaigns/write_optimized_equilibrium_transport_configs.py")
DEFAULT_CONTROLS = ("ZBS(1,0)", "ZBS(1,1)", "RBC(1,1)")
CONTROL_TOKEN_RE = re.compile(r"(?:RBC|RBS|ZBC|ZBS)\(\s*[+-]?\d+\s*,\s*[+-]?\d+\s*\)")


def _float_tuple(raw: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    if not all(math.isfinite(value) for value in values):
        raise argparse.ArgumentTypeError("values must be finite")
    return values


def _csv(values: tuple[float, ...]) -> str:
    return ",".join(f"{value:.16g}" for value in values)


def _control_tuple(raw: str) -> tuple[str, ...]:
    values = tuple(match.group(0) for match in CONTROL_TOKEN_RE.finditer(raw))
    if not values:
        raise argparse.ArgumentTypeError(
            "expected one or more VMEC boundary controls such as ZBS(1,0);RBC(1,1)"
        )
    return values


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


def _metric_command(
    args: argparse.Namespace,
    *,
    input_path: Path,
    out_json: Path,
    outdir: Path,
    out_wout: Path,
) -> str:
    cmd = [
        "python3",
        _repo_relative(ROOT / METRIC_EVAL),
        "--input",
        _repo_relative(input_path),
        "--out-json",
        _repo_relative(out_json),
        "--outdir",
        _repo_relative(outdir),
        "--out-wout",
        _repo_relative(out_wout),
        "--max-mode",
        str(args.max_mode),
        "--min-vmec-mode",
        str(args.min_vmec_mode),
        "--transport-kind",
        str(args.transport_kind),
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
        "--inner-max-iter",
        str(args.inner_max_iter),
        "--trial-max-iter",
        str(args.trial_max_iter),
        "--solver-device",
        str(args.solver_device),
        "--include-sample-rows",
    ]
    return shlex.join(cmd)


def _audit_command(
    args: argparse.Namespace, *, case_id: str, wout_path: Path, outdir: Path
) -> str:
    cmd = [
        "python3",
        _repo_relative(ROOT / AUDIT_WRITER),
        "--vmec-file",
        _repo_relative(wout_path),
        "--case",
        case_id,
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


def build_campaign(args: argparse.Namespace) -> dict[str, Any]:
    baseline_input = Path(args.baseline_input)
    text = baseline_input.read_text(encoding="utf-8")
    controls = tuple(_parse_coefficient_spec(raw) for raw in args.controls)
    base_values = {spec.label: _coefficient_value(text, spec) for spec in controls}
    rng = random.Random(int(args.seed))
    pairs: list[dict[str, Any]] = []
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for iteration in range(int(args.iterations)):
        direction = {spec.label: 1 if rng.random() >= 0.5 else -1 for spec in controls}
        pair_dir = out_dir / f"iter_{iteration:03d}"
        pair_dir.mkdir(parents=True, exist_ok=True)
        pair: dict[str, Any] = {
            "iteration": iteration,
            "direction": direction,
            "states": {},
            "gradient_estimator": {},
        }
        for label, sign in (("plus", 1.0), ("minus", -1.0)):
            patched = text
            deltas: dict[str, float] = {}
            values: dict[str, float] = {}
            for spec in controls:
                base = float(base_values[spec.label])
                scale = max(abs(base), float(args.zero_scale))
                delta = float(args.relative_delta) * scale
                step = sign * float(direction[spec.label]) * delta
                value = base + step
                patched = _patch_coefficient(patched, spec, value)
                deltas[spec.label] = delta
                values[spec.label] = value
            state_dir = pair_dir / label
            state_dir.mkdir(parents=True, exist_ok=True)
            input_path = state_dir / "input.final"
            out_json = state_dir / f"{args.transport_kind}_metric.json"
            scratch = state_dir / "metric_eval_scratch"
            out_wout = state_dir / "wout_final_rerun.nc"
            input_path.write_text(patched, encoding="utf-8")
            case_id = f"{args.case_prefix}_iter{iteration:03d}_{label}"
            pair["states"][label] = {
                "input": input_path,
                "coefficient_values": values,
                "metric_json": out_json,
                "wout": out_wout,
                "metric_eval_command": _metric_command(
                    args,
                    input_path=input_path,
                    out_json=out_json,
                    outdir=scratch,
                    out_wout=out_wout,
                ),
                "nonlinear_audit_command": _audit_command(
                    args,
                    case_id=case_id,
                    wout_path=out_wout,
                    outdir=state_dir / "nonlinear_audit",
                ),
            }
            pair["gradient_estimator"] = {
                spec.label: (
                    "after plus/minus metrics complete, estimate dJ/dx_i = "
                    f"(J_plus - J_minus) / (2 * {deltas[spec.label]:.16e} * {direction[spec.label]:+d})"
                )
                for spec in controls
            }
        pairs.append(pair)

    manifest = {
        "kind": "vmec_jax_spsa_transport_candidate_campaign",
        "claim_scope": (
            "SPSA common-random-number candidate generation for noisy nonlinear-Q optimization; "
            "not a promoted nonlinear turbulent-flux optimization result."
        ),
        "baseline_input": baseline_input,
        "out_dir": out_dir,
        "seed": int(args.seed),
        "iterations": int(args.iterations),
        "relative_delta": float(args.relative_delta),
        "zero_scale": float(args.zero_scale),
        "controls": [spec.label for spec in controls],
        "baseline_values": base_values,
        "transport_kind": str(args.transport_kind),
        "sample_policy": {
            "surfaces": args.surfaces,
            "alphas": args.alphas,
            "ky_values": args.ky_values,
            "ntheta": int(args.ntheta),
            "mboz": int(args.mboz),
            "nboz": int(args.nboz),
            "n_laguerre": int(args.n_laguerre),
            "n_hermite": int(args.n_hermite),
        },
        "common_random_number_policy": {
            "audit_seed_variants": list(args.audit_seed_variants),
            "audit_dt_variant": float(args.audit_dt_variant),
            "audit_window": [
                float(args.audit_window_tmin),
                float(args.audit_window_tmax),
            ],
            "audit_horizons": str(args.audit_horizons),
        },
        "pairs": pairs,
        "next_actions": [
            "Run every metric_eval_command.",
            "Rank pairs by the plus/minus reduced metric and SPSA gradient sign.",
            "Write nonlinear audit configs only for reduced-metric candidates worth long-window auditing.",
            "Promote no nonlinear-Q claim until matched t=1500 replicated audits pass.",
        ],
    }
    manifest_path = out_dir / "vmec_jax_spsa_candidate_manifest.json"
    manifest_path.write_text(
        json.dumps(_json_ready(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest["manifest"] = manifest_path
    return _json_ready(manifest)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-input", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--case-prefix", default="vmec_qa_spsa_nonlinear_window")
    parser.add_argument("--controls", type=_control_tuple, default=DEFAULT_CONTROLS)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--relative-delta", type=float, default=0.03)
    parser.add_argument("--zero-scale", type=float, default=1.0e-2)
    parser.add_argument("--max-mode", type=int, default=5)
    parser.add_argument("--min-vmec-mode", type=int, default=7)
    parser.add_argument("--transport-kind", default="nonlinear_window_heat_flux")
    parser.add_argument("--surfaces", type=_float_tuple, default=(0.64,))
    parser.add_argument("--alphas", type=_float_tuple, default=(0.0,))
    parser.add_argument("--ky-values", type=_float_tuple, default=(0.3,))
    parser.add_argument("--ntheta", type=int, default=24)
    parser.add_argument("--mboz", type=int, default=21)
    parser.add_argument("--nboz", type=int, default=21)
    parser.add_argument("--n-laguerre", type=int, default=2)
    parser.add_argument("--n-hermite", type=int, default=3)
    parser.add_argument("--surface-chunk-size", type=int, default=1)
    parser.add_argument(
        "--spectrax-objective-transform",
        choices=("raw", "scaled", "log1p"),
        default="log1p",
    )
    parser.add_argument("--spectrax-objective-scale", type=float, default=1.0)
    parser.add_argument("--inner-max-iter", type=int, default=140)
    parser.add_argument("--trial-max-iter", type=int, default=140)
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
    if not args.controls:
        parser.error("--controls must include at least one coefficient")
    if int(args.iterations) <= 0:
        parser.error("--iterations must be positive")
    if float(args.relative_delta) <= 0.0:
        parser.error("--relative-delta must be positive")
    if float(args.zero_scale) <= 0.0:
        parser.error("--zero-scale must be positive")
    if args.audit_seed_variants is None:
        args.audit_seed_variants = [32, 33]
    return args


def main(argv: list[str] | None = None) -> int:
    manifest = build_campaign(parse_args(argv))
    print(
        json.dumps(
            {
                "manifest": manifest["manifest"],
                "pairs": len(manifest["pairs"]),
                "controls": manifest["controls"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
