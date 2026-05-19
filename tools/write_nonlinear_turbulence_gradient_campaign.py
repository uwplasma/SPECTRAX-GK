#!/usr/bin/env python3
"""Write matched nonlinear campaigns for a production turbulence-gradient gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.write_external_vmec_holdout_configs import (  # noqa: E402
    _parse_grid,
    _parse_horizons,
    _parse_seed_dt_variant,
    write_configs,
    write_manifest,
)


DEFAULT_CASE = "optimized_equilibrium_turbulence_gradient"
DEFAULT_OUT_DIR = ROOT / "tools_out" / "nonlinear_turbulence_gradient_campaign"
DEFAULT_HORIZONS = "250,350,450,700"
DEFAULT_GRID = "n64:64:64:40:40"
DEFAULT_SEEDS = (31, 32)
DEFAULT_DT_VARIANT = 0.04
DEFAULT_WINDOW = (350.0, 700.0)
PYTHON_CMD = "python3"


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_state_vmec_files(
    state_vmec: dict[str, Path],
    *,
    allow_identical_vmec_content: bool = False,
) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    resolved_paths: dict[str, Path] = {}
    for state, raw_path in state_vmec.items():
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"{state} VMEC file does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"{state} VMEC path is not a file: {path}")
        resolved_paths[state] = path
        stat = path.stat()
        metadata[state] = {
            "path": path,
            "size_bytes": int(stat.st_size),
            "sha256": _sha256(path),
        }

    unique_paths = {path for path in resolved_paths.values()}
    if len(unique_paths) != len(resolved_paths):
        duplicates = {
            state: _repo_relative(path)
            for state, path in resolved_paths.items()
            if list(resolved_paths.values()).count(path) > 1
        }
        raise ValueError(
            "baseline/plus/minus VMEC files must be distinct paths for a matched "
            f"gradient campaign; duplicates: {duplicates}"
        )

    hashes = [str(row["sha256"]) for row in metadata.values()]
    if len(set(hashes)) != len(hashes) and not allow_identical_vmec_content:
        duplicate_hash_states: dict[str, list[str]] = {}
        for state, row in metadata.items():
            duplicate_hash_states.setdefault(str(row["sha256"]), []).append(state)
        duplicate_hash_states = {
            digest: states
            for digest, states in duplicate_hash_states.items()
            if len(states) > 1
        }
        raise ValueError(
            "baseline/plus/minus VMEC files must not have identical contents for "
            "production evidence; use --allow-identical-vmec-content only for "
            f"plumbing smoke tests. Duplicate SHA256 groups: {duplicate_hash_states}"
        )

    return metadata


def _horizon_label(value: float) -> str:
    rounded = int(round(value))
    if abs(value - rounded) < 1.0e-12:
        return str(rounded)
    return f"{value:.12g}".replace(".", "p").replace("-", "m")


def _float_label(value: float) -> str:
    return f"{float(value):.12g}".replace(".", "p").replace("-", "m")


def _expected_output(out_dir: Path, case: str, horizon: float, grid_label: str, variant: str) -> Path:
    return out_dir / f"{case}_nonlinear_t{_horizon_label(horizon)}_{grid_label}_{variant}.out.nc"


def _state_ensemble_command(
    *,
    case: str,
    state: str,
    state_out_dir: Path,
    grid_label: str,
    tmin: float,
    tmax: float,
    baseline_dt: float,
    seed_variants: tuple[int, ...],
    dt_variant: float,
    dt_variant_label: str,
    seed_dt_variants: tuple[tuple[int, float], ...] = (),
) -> dict[str, Any]:
    state_case = f"{case}_{state}"
    ensemble_dir = ROOT / "docs" / "_static" / f"{case}_{state}_replicates"
    inputs = [
        _expected_output(state_out_dir, state_case, tmax, grid_label, f"seed{seed}")
        for seed in seed_variants
    ]
    inputs.append(_expected_output(state_out_dir, state_case, tmax, grid_label, dt_variant_label))
    variants = [(f"seed{seed}", float(baseline_dt)) for seed in seed_variants]
    variants.append((dt_variant_label, float(dt_variant)))
    for seed, seed_dt in seed_dt_variants:
        label = f"seed{int(seed)}_dt{_float_label(seed_dt)}"
        inputs.append(_expected_output(state_out_dir, state_case, tmax, grid_label, label))
        variants.append((label, float(seed_dt)))
    direct_full_horizon_step_counts = {
        label: int(round(float(tmax) / dt)) for label, dt in variants
    }
    direct_full_horizon_launch_commands = [
        (
            f"{PYTHON_CMD} -m spectraxgk.cli run-runtime-nonlinear "
            f"--config {_repo_relative(_expected_output(state_out_dir, state_case, tmax, grid_label, label).with_suffix('').with_suffix('.toml'))} "
            f"--steps {steps} --no-progress"
        )
        for label, steps in direct_full_horizon_step_counts.items()
    ]
    ensemble_json = f"{case}_{state}_t{_horizon_label(tmax)}_ensemble_gate.json"
    readiness_json = f"{case}_{state}_readiness.json"
    ensemble_png = f"{case}_{state}_t{_horizon_label(tmax)}_ensemble_gate.png"
    command = (
        f"{PYTHON_CMD} tools/build_external_vmec_replicate_ensemble.py "
        + " ".join(_repo_relative(path) for path in inputs)
        + f" --out-dir {_repo_relative(ensemble_dir)}"
        + f" --case {case}_{state}_replicated_nonlinear_window"
        + f" --tmin {tmin:.12g} --tmax {tmax:.12g}"
        + f" --artifact-prefix {_repo_relative(ensemble_dir)}"
        + f" --readiness-json {readiness_json}"
        + f" --ensemble-json {ensemble_json}"
        + f" --out-png {ensemble_png}"
    )
    output_gate_json = f"{case}_{state}_t{_horizon_label(tmax)}_output_gate.json"
    output_gate_command = (
        f"{PYTHON_CMD} tools/check_nonlinear_runtime_outputs.py "
        + " ".join(_repo_relative(path) for path in inputs)
        + f" --min-samples 200 --tmin {tmin:.12g} --tmax {tmax:.12g}"
        + " --min-window-samples 80 --min-abs-window-mean 1e-4"
        + f" --json-out {_repo_relative(ensemble_dir / output_gate_json)}"
    )
    return {
        "state": state,
        "expected_outputs": [_repo_relative(path) for path in inputs],
        "ensemble_json": _repo_relative(ensemble_dir / ensemble_json),
        "readiness_json": _repo_relative(ensemble_dir / readiness_json),
        "ensemble_png": _repo_relative(ensemble_dir / ensemble_png),
        "build_ensemble_command": command,
        "output_gate_json": _repo_relative(ensemble_dir / output_gate_json),
        "output_gate_command": output_gate_command,
        "direct_full_horizon_step_counts": direct_full_horizon_step_counts,
        "direct_full_horizon_launch_commands": direct_full_horizon_launch_commands,
        "restart_ladder_note": (
            "The generated TOMLs are restart-ladder segments. To run only the final "
            "tmax TOMLs without first executing and seeding the earlier horizons, "
            "use the direct_full_horizon_launch_commands so the CLI overrides "
            "[run].steps with tmax/dt."
        ),
    }


def _promotion_contract(
    *,
    case: str,
    parameter_name: str,
    delta_parameter: float,
    state_commands: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    fd_prefix = ROOT / "docs" / "_static" / f"{case}_{parameter_name}_central_fd_gradient_gate"
    fd_json = fd_prefix.with_suffix(".json")
    evidence_json = ROOT / "docs" / "_static" / "nonlinear_turbulence_gradient_evidence_status.json"
    gap_json = ROOT / "docs" / "_static" / "nonlinear_turbulence_gradient_evidence_gap_report.json"
    fd_command = (
        f"{PYTHON_CMD} tools/build_nonlinear_turbulence_gradient_fd_gate.py "
        f"--baseline {state_commands['baseline']['ensemble_json']} "
        f"--plus {state_commands['plus_delta']['ensemble_json']} "
        f"--minus {state_commands['minus_delta']['ensemble_json']} "
        f"--delta-parameter {float(delta_parameter):.12g} "
        f"--parameter-name {parameter_name} "
        f"--out-prefix {_repo_relative(fd_prefix)} "
        "--fail-on-blocked"
    )
    evidence_command = (
        f"{PYTHON_CMD} tools/check_nonlinear_turbulence_gradient_evidence.py "
        f"--gradient-artifact {_repo_relative(fd_json)} "
        f"--window-artifact {state_commands['baseline']['ensemble_json']} "
        f"--window-artifact {state_commands['plus_delta']['ensemble_json']} "
        f"--window-artifact {state_commands['minus_delta']['ensemble_json']} "
        f"--json-out {_repo_relative(evidence_json)} "
        f"--gap-json-out {_repo_relative(gap_json)} "
        "--fail-on-blocked"
    )
    return {
        "claim_level": "production_long_window_turbulence_gradient_launch_contract_not_promotion",
        "central_fd_json": _repo_relative(fd_json),
        "central_fd_command": fd_command,
        "evidence_status_json": _repo_relative(evidence_json),
        "evidence_gap_json": _repo_relative(gap_json),
        "evidence_check_command": evidence_command,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-vmec-file", required=True, type=Path)
    parser.add_argument("--plus-vmec-file", required=True, type=Path)
    parser.add_argument("--minus-vmec-file", required=True, type=Path)
    parser.add_argument("--case", default=DEFAULT_CASE)
    parser.add_argument("--parameter-name", default="vmec_state_control_or_profile_gradient")
    parser.add_argument("--delta-parameter", type=float, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ky", type=float, default=0.47619047619047616)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--dt-variant", type=float, default=DEFAULT_DT_VARIANT)
    parser.add_argument("--baseline-seed", type=int, default=22)
    parser.add_argument("--seed-variant", action="append", type=int, default=None)
    parser.add_argument(
        "--seed-dt-variant",
        action="append",
        default=None,
        help="Joint seed/timestep replicate encoded as SEED:DT. Repeat for cross-check variants.",
    )
    parser.add_argument("--horizons", default=DEFAULT_HORIZONS)
    parser.add_argument("--grid", action="append", default=None, help="Grid spec label:Nx:Ny:Nz:ntheta")
    parser.add_argument("--Nl", type=int, default=4)
    parser.add_argument("--Nm", type=int, default=8)
    parser.add_argument("--window-tmin", type=float, default=DEFAULT_WINDOW[0])
    parser.add_argument("--window-tmax", type=float, default=DEFAULT_WINDOW[1])
    parser.add_argument(
        "--allow-identical-vmec-content",
        action="store_true",
        help=(
            "Allow byte-identical baseline/plus/minus VMEC files. This is only "
            "for plumbing smoke tests and is not production turbulence-gradient evidence."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if float(args.delta_parameter) <= 0.0:
        raise ValueError("delta-parameter must be positive")
    grids = tuple(_parse_grid(raw) for raw in (args.grid or (DEFAULT_GRID,)))
    if len(grids) != 1:
        raise ValueError("gradient campaign configs currently require exactly one grid")
    seed_variants = tuple(args.seed_variant or DEFAULT_SEEDS)
    seed_dt_variants = tuple(_parse_seed_dt_variant(raw) for raw in (args.seed_dt_variant or ()))
    dt_variant_label = f"dt{_float_label(float(args.dt_variant))}"
    state_vmec = {
        "minus_delta": args.minus_vmec_file,
        "baseline": args.baseline_vmec_file,
        "plus_delta": args.plus_vmec_file,
    }
    state_file_metadata = _validate_state_vmec_files(
        state_vmec,
        allow_identical_vmec_content=bool(args.allow_identical_vmec_content),
    )
    state_manifests: dict[str, str] = {}
    state_commands: dict[str, dict[str, Any]] = {}
    total_configs = 0
    for state, vmec_file in state_vmec.items():
        state_case = f"{args.case}_{state}"
        state_out_dir = Path(args.out_dir) / state
        written = write_configs(
            case=state_case,
            vmec_file=Path(vmec_file),
            out_dir=state_out_dir,
            grids=grids,
            horizons=_parse_horizons(str(args.horizons)),
            dt=float(args.dt),
            ky=float(args.ky),
            nl=int(args.Nl),
            nm=int(args.Nm),
            baseline_seed=int(args.baseline_seed),
            seed_variants=seed_variants,
            dt_variants=(float(args.dt_variant),),
            seed_dt_variants=seed_dt_variants,
        )
        total_configs += len(written)
        state_manifests[state] = _repo_relative(write_manifest(state_out_dir, written))
        state_commands[state] = _state_ensemble_command(
            case=str(args.case),
            state=state,
            state_out_dir=state_out_dir,
            grid_label=grids[0].label,
            tmin=float(args.window_tmin),
            tmax=float(args.window_tmax),
            baseline_dt=float(args.dt),
            seed_variants=seed_variants,
            dt_variant=float(args.dt_variant),
            dt_variant_label=dt_variant_label,
            seed_dt_variants=seed_dt_variants,
        )

    manifest = {
        "kind": "nonlinear_turbulence_gradient_campaign_manifest",
        "claim_level": "matched_baseline_plus_minus_launch_plan_not_simulation_claim",
        "case": str(args.case),
        "parameter_name": str(args.parameter_name),
        "delta_parameter": float(args.delta_parameter),
        "state_vmec_files": {state: _repo_relative(path) for state, path in state_vmec.items()},
        "vmec_file_preflight": {
            "vmec_files_exist": True,
            "vmec_paths_distinct": True,
            "vmec_contents_distinct": len({row["sha256"] for row in state_file_metadata.values()})
            == len(state_file_metadata),
            "allow_identical_vmec_content": bool(args.allow_identical_vmec_content),
            "claim_boundary": (
                "Production nonlinear turbulence-gradient evidence requires real "
                "baseline/plus/minus re-equilibrated VMEC files. Identical file "
                "content is accepted only for explicit plumbing smoke tests."
            ),
            "files": state_file_metadata,
        },
        "state_manifests": state_manifests,
        "state_ensemble_commands": state_commands,
        "promotion_contract": _promotion_contract(
            case=str(args.case),
            parameter_name=str(args.parameter_name),
            delta_parameter=float(args.delta_parameter),
            state_commands=state_commands,
        ),
        "run_contract": {
            "same_numerics_except_parameter": True,
            "minimum_tmax": float(args.window_tmax),
            "analysis_window": [float(args.window_tmin), float(args.window_tmax)],
            "grid": grids[0].label,
            "replicates": [f"seed{seed}" for seed in seed_variants] + [dt_variant_label],
            "joint_seed_timestep_replicates": [
                f"seed{seed}_dt{_float_label(seed_dt)}" for seed, seed_dt in seed_dt_variants
            ],
        },
        "configs_written": total_configs,
    }
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.out_dir) / "gradient_campaign_manifest.json"
    manifest_path.write_text(json.dumps(_json_clean(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"configs": total_configs, "manifest": _repo_relative(manifest_path)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
