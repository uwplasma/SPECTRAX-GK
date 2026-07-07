#!/usr/bin/env python3
"""Write targeted nonlinear replicate follow-up configs from a spread diagnostic."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import shlex
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

tomllib: Any = importlib.import_module("tomllib" if sys.version_info >= (3, 11) else "tomli")

from spectraxgk.validation.nonlinear_transport.replicate_followup import (  # noqa: E402
    NonlinearReplicateFollowupConfig,
    nonlinear_replicate_followup_plan,
)
from tools.write_external_vmec_holdout_configs import (  # noqa: E402
    _parse_grid,
    write_configs,
    write_manifest,
)


STATE_ORDER = ("baseline", "plus_delta", "minus_delta")


def _repo_relative(path: Path | str) -> str:
    raw = Path(path)
    try:
        return raw.resolve().relative_to(ROOT.resolve()).as_posix()
    except (OSError, ValueError):
        return str(path)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _config_from_direct_command(command: str) -> Path:
    parts = shlex.split(command)
    try:
        raw = parts[parts.index("--config") + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"direct command is missing '--config': {command}") from exc
    return ROOT / raw


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _metadata_from_config(path: Path, *, state: str) -> dict[str, Any]:
    payload = _read_toml(path)
    metadata = payload.get("metadata", {})
    init = payload.get("init", {})
    time = payload.get("time", {})
    output = payload.get("output", {})
    if not isinstance(metadata, dict):
        metadata = {}
    return {
        "state": state,
        "variant_label": str(metadata.get("variant_label") or path.stem),
        "variant_axis": str(metadata.get("variant_axis") or "unknown"),
        "seed": int(metadata.get("seed", init.get("random_seed", 0))),
        "timestep": float(metadata.get("timestep", time.get("dt", 0.0))),
        "source_config": _repo_relative(path),
        "source_output": _repo_relative(path.parent / str(output.get("path", path.with_suffix(".out.nc").name))),
    }


def collect_variant_metadata(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Read variant metadata from all direct full-horizon TOMLs in a manifest."""

    state_commands = manifest.get("state_ensemble_commands")
    if not isinstance(state_commands, dict):
        raise ValueError("manifest is missing state_ensemble_commands")
    rows: list[dict[str, Any]] = []
    ordered_states = [state for state in STATE_ORDER if state in state_commands]
    ordered_states.extend(sorted(state for state in state_commands if state not in ordered_states))
    for state in ordered_states:
        raw_row = state_commands[state]
        if not isinstance(raw_row, dict):
            continue
        for command in raw_row.get("direct_full_horizon_launch_commands", []):
            config = _config_from_direct_command(str(command))
            rows.append(_metadata_from_config(config, state=state))
    return rows


def _state_reference_config(manifest: dict[str, Any], state: str) -> tuple[Path, dict[str, Any]]:
    row = manifest["state_ensemble_commands"][state]
    commands = row.get("direct_full_horizon_launch_commands", [])
    if not commands:
        raise ValueError(f"state {state!r} has no direct full-horizon launch commands")
    config_path = _config_from_direct_command(str(commands[0]))
    return config_path, _read_toml(config_path)


def _grid_spec_from_toml(payload: dict[str, Any], *, fallback_label: str) -> str:
    grid = payload.get("grid")
    if not isinstance(grid, dict):
        raise ValueError("reference config is missing [grid]")
    return "{label}:{nx}:{ny}:{nz}:{ntheta}".format(
        label=fallback_label,
        nx=int(grid["Nx"]),
        ny=int(grid["Ny"]),
        nz=int(grid["Nz"]),
        ntheta=int(grid["ntheta"]),
    )


def _value(payload: dict[str, Any], section: str, key: str, default: Any) -> Any:
    raw = payload.get(section)
    if isinstance(raw, dict) and key in raw:
        return raw[key]
    return default


def _write_followup_configs(
    *,
    manifest: dict[str, Any],
    plan: dict[str, Any],
) -> dict[str, Any]:
    run_contract = manifest.get("run_contract")
    if not isinstance(run_contract, dict):
        raise ValueError("manifest is missing run_contract")
    tmax = float(run_contract.get("analysis_window", [0.0, run_contract.get("minimum_tmax", 0.0)])[1])
    grid_label = str(run_contract.get("grid", "n64"))
    by_state: dict[str, list[dict[str, Any]]] = {}
    for row in plan.get("planned_runs", []):
        if isinstance(row, dict):
            by_state.setdefault(str(row["state"]), []).append(row)

    written_by_state: dict[str, Any] = {}
    for state, rows in by_state.items():
        reference_path, reference = _state_reference_config(manifest, state)
        geometry = reference.get("geometry")
        if not isinstance(geometry, dict):
            raise ValueError(f"reference config {reference_path} is missing [geometry]")
        vmec_file = (reference_path.parent / str(geometry["vmec_file"])).resolve()
        state_case = str(_value(reference, "metadata", "case", f"{manifest['case']}_{state}"))
        out_dir = reference_path.parent
        joint = tuple((int(row["seed"]), float(row["timestep"])) for row in rows)
        written = write_configs(
            case=state_case,
            vmec_file=vmec_file,
            out_dir=out_dir,
            grids=[_parse_grid(_grid_spec_from_toml(reference, fallback_label=grid_label))],
            horizons=(tmax,),
            dt=float(_value(reference, "time", "dt", 0.05)),
            ky=float(_value(reference, "run", "ky", 0.47619047619047616)),
            nl=int(_value(reference, "run", "Nl", 4)),
            nm=int(_value(reference, "run", "Nm", 8)),
            torflux=float(_value(reference, "geometry", "torflux", 0.64)),
            alpha=float(_value(reference, "geometry", "alpha", 0.0)),
            npol=float(_value(reference, "geometry", "npol", 1.0)),
            tprim=float(reference.get("species", [{}])[0].get("tprim", 3.0)),
            fprim=float(reference.get("species", [{}])[0].get("fprim", 1.0)),
            nu=float(reference.get("species", [{}])[0].get("nu", 0.01)),
            init_amp=float(_value(reference, "init", "init_amp", 1.0e-3)),
            y0=float(_value(reference, "grid", "y0", 21.0)),
            lx=float(_value(reference, "grid", "Lx", 62.8)),
            ly=float(_value(reference, "grid", "Ly", 62.8)),
            sample_stride=int(_value(reference, "time", "sample_stride", 50)),
            diagnostics_stride=int(_value(reference, "time", "diagnostics_stride", 50)),
            progress_bar=bool(_value(reference, "time", "progress_bar", False)),
            baseline_seed=int(_value(reference, "init", "random_seed", 22)),
            seed_dt_variants=joint,
        )
        manifest_path = write_manifest(out_dir, written)
        direct_commands = [
            (
                "python3 -m spectraxgk.cli run-runtime-nonlinear "
                f"--config {_repo_relative(item.path)} "
                f"--steps {int(round(tmax / float(item.variant.dt if item.variant else _value(reference, 'time', 'dt', 0.05))))} "
                "--no-progress"
            )
            for item in written
        ]
        written_by_state[state] = {
            "state": state,
            "reference_config": _repo_relative(reference_path),
            "run_manifest": _repo_relative(manifest_path),
            "configs": [
                {
                    "path": _repo_relative(item.path),
                    "output": _repo_relative(item.output_path),
                    "variant_label": item.variant.label if item.variant else None,
                    "seed": item.variant.random_seed if item.variant else None,
                    "timestep": item.variant.dt if item.variant else None,
                    "steps": int(round(tmax / float(item.variant.dt if item.variant else _value(reference, "time", "dt", 0.05)))),
                }
                for item in written
            ],
            "direct_full_horizon_launch_commands": direct_commands,
        }
    return written_by_state


def _planned_outputs_for_state(written_by_state: dict[str, Any], state: str) -> list[str]:
    state_payload = written_by_state.get(state)
    if not isinstance(state_payload, dict):
        return []
    return [
        str(row["output"])
        for row in state_payload.get("configs", [])
        if isinstance(row, dict) and row.get("output")
    ]


def _postprocess_commands(
    *,
    manifest: dict[str, Any],
    written_by_state: dict[str, Any],
) -> dict[str, Any]:
    run_contract = manifest.get("run_contract")
    if not isinstance(run_contract, dict):
        return {}
    analysis_window = run_contract.get("analysis_window", [0.0, 0.0])
    tmin = float(analysis_window[0])
    tmax = float(analysis_window[1])
    t_label = str(int(round(tmax))) if abs(tmax - round(tmax)) < 1.0e-12 else f"{tmax:.12g}".replace(".", "p")
    commands: dict[str, Any] = {}
    state_commands = manifest.get("state_ensemble_commands")
    if not isinstance(state_commands, dict):
        return commands
    for state in sorted(written_by_state):
        original = state_commands.get(state)
        if not isinstance(original, dict):
            continue
        ensemble_json = Path(str(original.get("ensemble_json", "")))
        if not ensemble_json.name:
            continue
        ensemble_dir = ensemble_json.parent
        existing_outputs = [str(path) for path in original.get("expected_outputs", [])]
        planned_outputs = _planned_outputs_for_state(written_by_state, state)
        all_outputs = existing_outputs + planned_outputs
        prefix = f"{manifest['case']}_{state}_t{t_label}_followup"
        output_gate_json = ensemble_dir / f"{prefix}_output_gate.json"
        ensemble_gate_json = f"{prefix}_ensemble_gate.json"
        readiness_json = f"{prefix}_ensemble_readiness.json"
        ensemble_png = f"{prefix}_ensemble_gate.png"
        commands[state] = {
            "all_expected_outputs": all_outputs,
            "output_gate_json": _repo_relative(output_gate_json),
            "output_gate_command": (
                "python3 tools/release/check_nonlinear_runtime_outputs.py "
                + " ".join(all_outputs)
                + f" --min-samples 200 --tmin {tmin:.12g} --tmax {tmax:.12g}"
                + " --min-window-samples 80 --min-abs-window-mean 1e-4"
                + f" --json-out {_repo_relative(output_gate_json)}"
            ),
            "ensemble_json": _repo_relative(ensemble_dir / ensemble_gate_json),
            "readiness_json": _repo_relative(ensemble_dir / readiness_json),
            "ensemble_png": _repo_relative(ensemble_dir / ensemble_png),
            "build_ensemble_command": (
                "python3 tools/artifacts/build_external_vmec_replicate_ensemble.py "
                + " ".join(all_outputs)
                + f" --out-dir {_repo_relative(ensemble_dir)}"
                + f" --case {manifest['case']}_{state}_replicated_nonlinear_window_followup"
                + f" --tmin {tmin:.12g} --tmax {tmax:.12g}"
                + f" --artifact-prefix {_repo_relative(ensemble_dir)}"
                + f" --readiness-json {readiness_json}"
                + f" --ensemble-json {ensemble_gate_json}"
                + f" --out-png {ensemble_png}"
            ),
        }

    baseline_json = state_commands.get("baseline", {}).get("ensemble_json")
    minus_json = state_commands.get("minus_delta", {}).get("ensemble_json")
    if baseline_json and minus_json:
        for state, row in commands.items():
            if state != "plus_delta":
                continue
            spread_prefix = ROOT / "docs" / "_static" / f"{manifest['case']}_{state}_followup_replicate_spread_diagnostic"
            fd_prefix = ROOT / "docs" / "_static" / f"{manifest['case']}_{state}_followup_central_fd_gradient_gate"
            evidence_json = ROOT / "docs" / "_static" / f"{manifest['case']}_{state}_followup_evidence_status.json"
            gap_json = ROOT / "docs" / "_static" / f"{manifest['case']}_{state}_followup_evidence_gap_report.json"
            row["replicate_spread_command"] = (
                f"python3 tools/summarize_nonlinear_replicate_spread.py {baseline_json} "
                f"{row['ensemble_json']} {minus_json} --out-prefix {_repo_relative(spread_prefix)} "
                f"--case {manifest['case']}_{state}_followup_replicate_spread"
            )
            row["central_fd_command"] = (
                "python3 tools/artifacts/build_nonlinear_turbulence_gradient_fd_gate.py "
                f"--baseline {baseline_json} --plus {row['ensemble_json']} --minus {minus_json} "
                f"--delta-parameter {float(manifest['delta_parameter']):.12g} "
                f"--parameter-name {manifest['parameter_name']} "
                f"--out-prefix {_repo_relative(fd_prefix)} --fail-on-blocked"
            )
            row["evidence_check_command"] = (
                "python3 tools/release/check_nonlinear_turbulence_gradient_evidence.py "
                f"--gradient-artifact {_repo_relative(fd_prefix.with_suffix('.json'))} "
                f"--window-artifact {baseline_json} --window-artifact {row['ensemble_json']} "
                f"--window-artifact {minus_json} --json-out {_repo_relative(evidence_json)} "
                f"--gap-json-out {_repo_relative(gap_json)} --fail-on-blocked"
            )
    return commands


def build_followup_campaign(
    *,
    campaign_manifest_path: Path,
    spread_diagnostic_path: Path,
    out_json: Path,
    case: str,
    include_extra_nominal_seed: bool,
    max_runs_per_state: int,
    dry_run: bool = False,
) -> dict[str, Any]:
    campaign_manifest = _load_json(campaign_manifest_path)
    spread_report = _load_json(spread_diagnostic_path)
    metadata = collect_variant_metadata(campaign_manifest)
    plan = nonlinear_replicate_followup_plan(
        spread_report,
        variant_metadata=metadata,
        case=case,
        config=NonlinearReplicateFollowupConfig(
            include_extra_nominal_seed=include_extra_nominal_seed,
            max_runs_per_state=max_runs_per_state,
        ),
    )
    written = {} if dry_run else _write_followup_configs(manifest=campaign_manifest, plan=plan)
    postprocess = {} if dry_run else _postprocess_commands(manifest=campaign_manifest, written_by_state=written)
    payload = {
        **plan,
        "campaign_manifest": _repo_relative(campaign_manifest_path),
        "spread_diagnostic": _repo_relative(spread_diagnostic_path),
        "variant_metadata": metadata,
        "written_configs_by_state": written,
        "postprocess_commands_by_state": postprocess,
        "dry_run": bool(dry_run),
        "next_action": (
            "Run the direct_full_horizon_launch_commands for each written state, rebuild the failed "
            "ensemble with the added outputs, then rerun the replicate-spread and central-FD gates."
        ),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-manifest", required=True, type=Path)
    parser.add_argument("--spread-diagnostic", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--case", default="nonlinear_replicate_followup")
    parser.add_argument("--no-extra-nominal-seed", action="store_true")
    parser.add_argument("--max-runs-per-state", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_followup_campaign(
        campaign_manifest_path=Path(args.campaign_manifest),
        spread_diagnostic_path=Path(args.spread_diagnostic),
        out_json=Path(args.out_json),
        case=str(args.case),
        include_extra_nominal_seed=not bool(args.no_extra_nominal_seed),
        max_runs_per_state=int(args.max_runs_per_state),
        dry_run=bool(args.dry_run),
    )
    print(
        json.dumps(
            {
                "planned_run_count": payload["summary"]["planned_run_count"],
                "states": sorted(payload["written_configs_by_state"]),
                "out_json": _repo_relative(args.out_json),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
