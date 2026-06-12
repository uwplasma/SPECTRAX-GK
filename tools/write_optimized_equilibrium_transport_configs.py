#!/usr/bin/env python3
"""Write production-scope optimized-equilibrium nonlinear transport configs.

This is a thin, explicit wrapper around ``write_external_vmec_holdout_configs``.
It encodes the promotion contract for production nonlinear stellarator
optimization: long post-transient, replicated nonlinear windows for a concrete
post-optimization VMEC equilibrium. It writes configs and commands only; it does
not run simulations or promote the claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.write_external_vmec_holdout_configs import (
    _parse_grid,
    _parse_horizons,
    write_configs,
    write_manifest,
)


DEFAULT_CASE = "optimized_equilibrium_post_optimization"
DEFAULT_OUT_DIR = ROOT / "tools_out" / "optimized_equilibrium_replicates"
DEFAULT_HORIZONS = "250,350,450,700"
DEFAULT_GRID = "n64:64:64:40:40"
DEFAULT_SEEDS = (31, 32)
DEFAULT_DT_VARIANT = 0.04
DEFAULT_WINDOW = (350.0, 700.0)


def _repo_relative(path: Path | str) -> str:
    raw = Path(path)
    try:
        return raw.resolve().relative_to(ROOT.resolve()).as_posix()
    except (OSError, ValueError):
        return str(path)


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, Path):
        return _repo_relative(value)
    return value


def _expected_output(out_dir: Path, case: str, horizon: float, grid_label: str, variant: str) -> Path:
    horizon_label = str(int(horizon)) if abs(horizon - int(horizon)) < 1e-12 else str(horizon).replace(".", "p")
    return out_dir / f"{case}_nonlinear_t{horizon_label}_{grid_label}_{variant}.out.nc"


def _promotion_commands(
    *,
    out_dir: Path,
    case: str,
    grid_label: str,
    tmin: float,
    tmax: float,
    baseline_dt: float,
    seed_variants: tuple[int, ...],
    dt_variant: float,
    dt_variant_label: str,
) -> dict[str, Any]:
    ensemble_dir = ROOT / "docs" / "_static" / "optimized_equilibrium_replicates"
    inputs = [
        _expected_output(out_dir, case, tmax, grid_label, f"seed{seed}")
        for seed in seed_variants
    ]
    inputs.append(_expected_output(out_dir, case, tmax, grid_label, dt_variant_label))
    variants = [(f"seed{seed}", float(baseline_dt)) for seed in seed_variants]
    variants.append((dt_variant_label, float(dt_variant)))
    direct_full_horizon_step_counts = {
        label: int(round(float(tmax) / dt)) for label, dt in variants
    }
    direct_full_horizon_launch_commands = [
        (
            "CUDA_VISIBLE_DEVICES=${DEVICE:-0} python3 -m spectraxgk.cli run-runtime-nonlinear "
            f"--config {_repo_relative(_expected_output(out_dir, case, tmax, grid_label, label).with_suffix('').with_suffix('.toml'))} "
            f"--steps {steps} --no-progress"
        )
        for label, steps in direct_full_horizon_step_counts.items()
    ]
    ensemble_json = f"{case}_ensemble_gate.json"
    readiness_json = f"{case}_readiness.json"
    ensemble_png = f"{case}_ensemble_gate.png"
    build_ensemble = (
        "python3 tools/build_external_vmec_replicate_ensemble.py "
        + " ".join(_repo_relative(path) for path in inputs)
        + f" --out-dir {_repo_relative(ensemble_dir)}"
        + f" --case {case}_replicated_nonlinear_window"
        + f" --tmin {tmin:.12g} --tmax {tmax:.12g}"
        + f" --artifact-prefix {_repo_relative(ensemble_dir)}"
        + f" --readiness-json {readiness_json}"
        + f" --ensemble-json {ensemble_json}"
        + f" --out-png {ensemble_png}"
    )
    output_gate_json = f"{case}_output_gate.json"
    output_gate_command = (
        "python3 tools/check_nonlinear_runtime_outputs.py "
        + " ".join(_repo_relative(path) for path in inputs)
        + f" --min-samples 200 --tmin {tmin:.12g} --tmax {tmax:.12g}"
        + " --min-window-samples 80 --min-abs-window-mean 0.0001"
        + f" --json-out {_repo_relative(ensemble_dir / output_gate_json)}"
    )
    guard_json = ROOT / "docs" / "_static" / "production_nonlinear_optimization_guard.json"
    guard_png = ROOT / "docs" / "_static" / "production_nonlinear_optimization_guard.png"
    run_guard = (
        "python3 tools/check_production_nonlinear_optimization_guard.py "
        f"--optimized-equilibrium-ensemble {_repo_relative(ensemble_dir / ensemble_json)} "
        f"--out-json {_repo_relative(guard_json)} "
        f"--out-png {_repo_relative(guard_png)} "
        "--fail-on-unpromoted"
    )
    return {
        "claim_level": "optimized_equilibrium_replicated_transport_window_launch_contract_not_promotion",
        "expected_outputs": [_repo_relative(path) for path in inputs],
        "ensemble_json": _repo_relative(ensemble_dir / ensemble_json),
        "readiness_json": _repo_relative(ensemble_dir / readiness_json),
        "ensemble_png": _repo_relative(ensemble_dir / ensemble_png),
        "build_ensemble_command": build_ensemble,
        "output_gate_json": _repo_relative(ensemble_dir / output_gate_json),
        "output_gate_command": output_gate_command,
        "run_guard_command": run_guard,
        "direct_full_horizon_step_counts": direct_full_horizon_step_counts,
        "direct_full_horizon_launch_commands": direct_full_horizon_launch_commands,
        "restart_ladder_note": (
            "The generated TOMLs are restart-ladder segments. If launching only "
            "the final tmax TOMLs from t=0, use direct_full_horizon_launch_commands "
            "so --steps equals tmax/dt. Otherwise run the staged ladder commands "
            "from run_manifest.json in order and seed restart bundles between horizons."
        ),
    }


def _transport_sample_metadata(args: argparse.Namespace) -> dict[str, Any]:
    """Return explicit surface/field-line metadata encoded in the TOMLs."""

    return {
        "vmec_file": _repo_relative(args.vmec_file),
        "torflux": float(args.torflux),
        "alpha": float(args.alpha),
        "npol": float(args.npol),
        "ky": float(args.ky),
        "tprim": float(args.tprim),
        "fprim": float(args.fprim),
        "nu": float(args.nu),
        "claim_level": (
            "launch_contract_surface_field_line_metadata_not_transport_promotion"
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vmec-file", required=True, type=Path, help="Concrete optimized-equilibrium VMEC wout file.")
    parser.add_argument("--case", default=DEFAULT_CASE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ky", type=float, default=0.47619047619047616)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--dt-variant", type=float, default=DEFAULT_DT_VARIANT)
    parser.add_argument("--baseline-seed", type=int, default=22)
    parser.add_argument("--seed-variant", action="append", type=int, default=None)
    parser.add_argument("--horizons", default=DEFAULT_HORIZONS)
    parser.add_argument("--grid", action="append", default=None, help="Grid spec label:Nx:Ny:Nz:ntheta")
    parser.add_argument("--Nl", type=int, default=4)
    parser.add_argument("--Nm", type=int, default=8)
    parser.add_argument("--torflux", type=float, default=0.64, help="VMEC toroidal-flux label for the flux tube.")
    parser.add_argument("--alpha", type=float, default=0.0, help="VMEC/Boozer field-line label for the flux tube.")
    parser.add_argument("--npol", type=float, default=1.0, help="Number of poloidal turns in the VMEC flux tube.")
    parser.add_argument("--tprim", type=float, default=3.0, help="Ion temperature-gradient drive.")
    parser.add_argument("--fprim", type=float, default=1.0, help="Ion density-gradient drive.")
    parser.add_argument("--nu", type=float, default=0.01, help="Collision frequency used by the nonlinear transport audit.")
    parser.add_argument("--window-tmin", type=float, default=DEFAULT_WINDOW[0])
    parser.add_argument("--window-tmax", type=float, default=DEFAULT_WINDOW[1])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    grids = tuple(_parse_grid(raw) for raw in (args.grid or (DEFAULT_GRID,)))
    if len(grids) != 1:
        raise ValueError("optimized-equilibrium promotion configs currently require exactly one grid")
    seed_variants = tuple(args.seed_variant or DEFAULT_SEEDS)
    written = write_configs(
        case=str(args.case),
        vmec_file=args.vmec_file,
        out_dir=args.out_dir,
        grids=grids,
        horizons=_parse_horizons(str(args.horizons)),
        dt=float(args.dt),
        ky=float(args.ky),
        nl=int(args.Nl),
        nm=int(args.Nm),
        torflux=float(args.torflux),
        alpha=float(args.alpha),
        npol=float(args.npol),
        tprim=float(args.tprim),
        fprim=float(args.fprim),
        nu=float(args.nu),
        baseline_seed=int(args.baseline_seed),
        seed_variants=seed_variants,
        dt_variants=(float(args.dt_variant),),
    )
    manifest_path = write_manifest(args.out_dir, written)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dt_variant_label = f"dt{float(args.dt_variant):.12g}".replace(".", "p").replace("-", "m")
    manifest["claim_level"] = "optimized_equilibrium_transport_launch_plan_not_simulation_claim"
    manifest["optimized_equilibrium_vmec_file"] = _repo_relative(args.vmec_file)
    manifest["transport_sample"] = _transport_sample_metadata(args)
    manifest["promotion_contract"] = _promotion_commands(
        out_dir=args.out_dir,
        case=str(args.case),
        grid_label=grids[0].label,
        tmin=float(args.window_tmin),
        tmax=float(args.window_tmax),
        baseline_dt=float(args.dt),
        seed_variants=seed_variants,
        dt_variant=float(args.dt_variant),
        dt_variant_label=dt_variant_label,
    )
    manifest_path.write_text(json.dumps(_json_clean(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"configs": len(written), "manifest": _repo_relative(manifest_path)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
