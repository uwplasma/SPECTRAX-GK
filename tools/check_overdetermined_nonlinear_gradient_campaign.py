#!/usr/bin/env python3
"""Check readiness of a multi-control nonlinear-gradient campaign.

The overdetermined campaign manifest is a launch plan.  This checker turns it
into a fail-closed status artifact by verifying, for each control, that the
re-equilibrated VMEC files exist, the nested nonlinear campaign manifest has
been written, the expected runtime outputs exist, and the central finite-
difference artifact has passed before any ranking/promotion claim is allowed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _repo_path(path: Path | str) -> str:
    raw = Path(path)
    try:
        return raw.resolve().relative_to(ROOT.resolve()).as_posix()
    except (OSError, ValueError):
        return str(path)


def _resolve_repo_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return ROOT / path


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return payload


def _extract_out_dir_from_command(command: str) -> Path | None:
    parts = shlex.split(command)
    try:
        raw = parts[parts.index("--out-dir") + 1]
    except (ValueError, IndexError):
        return None
    return _resolve_repo_path(raw)


def _expected_nested_manifest(control: dict[str, Any]) -> Path | None:
    explicit = control.get("expected_nonlinear_campaign_manifest")
    if explicit:
        return _resolve_repo_path(str(explicit))
    out_dir = _extract_out_dir_from_command(str(control.get("nonlinear_campaign_command_after_vmec_runs", "")))
    return None if out_dir is None else out_dir / "gradient_campaign_manifest.json"


def _expected_runtime_outputs(nested_manifest: dict[str, Any] | None) -> list[Path]:
    if nested_manifest is None:
        return []
    state_commands = nested_manifest.get("state_ensemble_commands")
    if not isinstance(state_commands, dict):
        return []
    outputs: list[Path] = []
    for row in state_commands.values():
        if not isinstance(row, dict):
            continue
        for raw_path in row.get("expected_outputs", []):
            outputs.append(_resolve_repo_path(str(raw_path)))
    return outputs


def _required_runtime_tmax(manifest: dict[str, Any]) -> float | None:
    contract = manifest.get("run_contract")
    if not isinstance(contract, dict):
        return None
    window = contract.get("analysis_window")
    if isinstance(window, (list, tuple)) and len(window) >= 2:
        try:
            return float(window[1])
        except (TypeError, ValueError):
            return None
    horizons = contract.get("horizons")
    if isinstance(horizons, str):
        values: list[float] = []
        for raw in horizons.split(","):
            try:
                values.append(float(raw.strip()))
            except ValueError:
                continue
        return max(values) if values else None
    return None


def _read_runtime_time_max(path: Path) -> float | None:
    try:
        import netCDF4
        import numpy as np
    except ImportError:
        return None

    candidates: list[float] = []
    try:
        with netCDF4.Dataset(path) as root:
            arrays = []
            if "time" in root.variables:
                arrays.append(root.variables["time"][:])
            grids = root.groups.get("Grids")
            if grids is not None and "time" in grids.variables:
                arrays.append(grids.variables["time"][:])
            for array in arrays:
                values = np.asarray(array, dtype=float)
                finite = values[np.isfinite(values)]
                if finite.size:
                    candidates.append(float(finite.max()))
    except Exception:
        return None
    return max(candidates) if candidates else None


def _runtime_output_status(paths: list[Path], *, required_tmax: float | None) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        exists = path.exists()
        size_bytes = int(path.stat().st_size) if exists else 0
        time_max = _read_runtime_time_max(path) if exists and size_bytes > 0 and required_tmax is not None else None
        complete = bool(exists and size_bytes > 0)
        if complete and required_tmax is not None:
            complete = bool(time_max is not None and time_max >= required_tmax - 1.0e-6)
        rows.append(
            {
                "path": _repo_path(path),
                "exists": exists,
                "size_bytes": size_bytes,
                "time_max": time_max,
                "complete": complete,
            }
        )
    missing = [row for row in rows if not row["exists"] or int(row["size_bytes"]) <= 0]
    incomplete = [row for row in rows if row["exists"] and int(row["size_bytes"]) > 0 and not row["complete"]]
    return {
        "expected_count": len(rows),
        "complete_count": sum(1 for row in rows if row["complete"]),
        "missing_count": len(missing),
        "incomplete_count": len(incomplete),
        "required_tmax": required_tmax,
        "missing_outputs": [row["path"] for row in missing[:20]],
        "incomplete_outputs": [
            {
                "path": row["path"],
                "time_max": row["time_max"],
                "size_bytes": row["size_bytes"],
            }
            for row in incomplete[:20]
        ],
    }


def _state_file_status(paths: dict[str, Any] | None) -> dict[str, Any]:
    paths = paths or {}
    rows: dict[str, dict[str, Any]] = {}
    for state, raw in sorted(paths.items()):
        path = _resolve_repo_path(str(raw))
        rows[str(state)] = {
            "path": _repo_path(path),
            "exists": path.exists(),
            "size_bytes": int(path.stat().st_size) if path.exists() else 0,
        }
    return {
        "passed": bool(rows) and all(row["exists"] and row["size_bytes"] > 0 for row in rows.values()),
        "rows": rows,
    }


def _fd_status(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if payload is None:
        return {"path": _repo_path(path), "exists": False, "passed": False, "blockers": ["missing_fd_artifact"]}
    return {
        "path": _repo_path(path),
        "exists": True,
        "passed": bool(payload.get("passed", False)),
        "blockers": list(payload.get("blockers", [])) if isinstance(payload.get("blockers", []), list) else [],
        "metrics": payload.get("metrics", {}),
    }


def _ranking_status(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if payload is None:
        return {
            "path": _repo_path(path),
            "exists": False,
            "passed": False,
            "recommendation": "ranking artifact is missing because not all central-FD artifacts exist yet",
        }
    return {
        "path": _repo_path(path),
        "exists": True,
        "passed": bool(payload.get("passed", False)),
        "recommendation": str(payload.get("recommendation", "")),
        "best_candidate": payload.get("best_candidate"),
    }


def _control_status(control: dict[str, Any], *, required_tmax: float | None) -> dict[str, Any]:
    slug = str(control.get("coefficient_slug", "unknown"))
    vmec_inputs = _state_file_status(control.get("state_input_files") if isinstance(control.get("state_input_files"), dict) else {})
    vmec_wouts = _state_file_status(control.get("expected_wout_files") if isinstance(control.get("expected_wout_files"), dict) else {})
    nested_manifest_path = _expected_nested_manifest(control)
    nested_manifest = _load_json(nested_manifest_path) if nested_manifest_path is not None else None
    runtime_outputs = _expected_runtime_outputs(nested_manifest)
    runtime_status = _runtime_output_status(runtime_outputs, required_tmax=required_tmax)
    fd_artifact = _resolve_repo_path(str(control.get("expected_fd_artifact", "")))
    fd = _fd_status(fd_artifact)
    nonlinear_manifest_exists = nested_manifest is not None
    ready_for_runtime = bool(vmec_wouts["passed"] and nonlinear_manifest_exists)
    runtime_outputs_complete = bool(runtime_outputs) and runtime_status["complete_count"] == len(runtime_outputs)
    passed = bool(ready_for_runtime and runtime_outputs_complete and fd["passed"])
    blockers: list[str] = []
    if not vmec_inputs["passed"]:
        blockers.append("missing_vmec_inputs")
    if not vmec_wouts["passed"]:
        blockers.append("missing_vmec_wouts")
    if not nonlinear_manifest_exists:
        blockers.append("missing_nested_nonlinear_campaign_manifest")
    if nonlinear_manifest_exists and runtime_status["missing_count"]:
        blockers.append("missing_runtime_outputs")
    if nonlinear_manifest_exists and runtime_status["incomplete_count"]:
        blockers.append("incomplete_runtime_outputs")
    if not fd["passed"]:
        blockers.append("central_fd_not_promoted")
    return {
        "coefficient": str(control.get("coefficient", slug)),
        "coefficient_slug": slug,
        "case": str(control.get("case", "")),
        "passed": passed,
        "ready_for_runtime": ready_for_runtime,
        "runtime_outputs_complete": runtime_outputs_complete,
        "blockers": blockers,
        "vmec_input_status": vmec_inputs,
        "vmec_wout_status": vmec_wouts,
        "nested_nonlinear_campaign_manifest": None
        if nested_manifest_path is None
        else {
            "path": _repo_path(nested_manifest_path),
            "exists": nonlinear_manifest_exists,
        },
        "runtime_output_status": runtime_status,
        "central_fd_status": fd,
        "vmec_run_commands": control.get("vmec_run_commands", {}),
        "write_nonlinear_campaign_command": control.get("nonlinear_campaign_command_after_vmec_runs", ""),
    }


def overdetermined_campaign_status_report(manifest: dict[str, Any], *, manifest_path: Path | None = None) -> dict[str, Any]:
    """Return a fail-closed status report for an overdetermined campaign."""

    if manifest.get("kind") != "overdetermined_nonlinear_turbulence_gradient_campaign_manifest":
        raise ValueError(
            "expected kind='overdetermined_nonlinear_turbulence_gradient_campaign_manifest', "
            f"got {manifest.get('kind')!r}"
        )
    controls_raw = manifest.get("controls")
    if not isinstance(controls_raw, list) or not controls_raw:
        raise ValueError("manifest must contain a non-empty controls list")
    controls = [row for row in controls_raw if isinstance(row, dict)]
    if len(controls) != len(controls_raw):
        raise ValueError("all controls must be JSON objects")

    required_tmax = _required_runtime_tmax(manifest)
    control_rows = [_control_status(control, required_tmax=required_tmax) for control in controls]
    contract = manifest.get("promotion_contract")
    contract_map = contract if isinstance(contract, dict) else {}
    ranking_json = contract_map.get("candidate_ranking_json")
    ranking = (
        _ranking_status(_resolve_repo_path(str(ranking_json)))
        if ranking_json
        else {
            "path": "",
            "exists": False,
            "passed": False,
            "recommendation": "manifest is missing candidate_ranking_json",
        }
    )
    ready_controls = [row for row in control_rows if bool(row["ready_for_runtime"])]
    completed_controls = [row for row in control_rows if bool(row["runtime_outputs_complete"])]
    promoted_controls = [row for row in control_rows if bool(row["central_fd_status"]["passed"])]
    passed = bool(
        len(promoted_controls) >= 1
        and all(bool(row["passed"]) for row in control_rows)
        and bool(ranking["passed"])
    )
    next_actions: list[str] = []
    if not all(row["vmec_wout_status"]["passed"] for row in control_rows):
        next_actions.append("run the per-control VMEC re-equilibration commands")
    if any(
        row["vmec_wout_status"]["passed"]
        and not row["nested_nonlinear_campaign_manifest"]["exists"]
        for row in control_rows
    ):
        next_actions.append("run nonlinear_campaign_command_after_vmec_runs for each VMEC-complete control")
    if any(row["ready_for_runtime"] and not row["runtime_outputs_complete"] for row in control_rows):
        next_actions.append("run direct full-horizon nonlinear tasks for each nested campaign manifest")
    if all(row["runtime_outputs_complete"] for row in control_rows) and not promoted_controls:
        next_actions.append("run output gates, ensemble gates, central-FD gates, then candidate ranking")
    if not next_actions and not passed:
        next_actions.append("inspect failed central-FD/ranking blockers before any release promotion")
    return {
        "kind": "overdetermined_nonlinear_gradient_campaign_status",
        "claim_level": "multi_control_profile_gradient_status_not_simulation_claim",
        "manifest": "" if manifest_path is None else _repo_path(manifest_path),
        "case": str(manifest.get("case", "")),
        "passed": passed,
        "summary": {
            "control_count": len(control_rows),
            "ready_for_runtime_count": len(ready_controls),
            "runtime_complete_count": len(completed_controls),
            "central_fd_promoted_count": len(promoted_controls),
            "ranking_passed": bool(ranking["passed"]),
        },
        "controls": control_rows,
        "ranking_status": ranking,
        "next_actions": next_actions,
        "claim_boundary": (
            "This status can only pass after at least one real control has a passing "
            "long-window central-FD nonlinear turbulence-gradient artifact and the "
            "candidate ranking promotes it. Missing VMEC, runtime, or FD artifacts "
            "remain release blockers for the broader gradient claim."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--fail-on-blocked", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"{manifest_path} does not contain a JSON object")
    report = overdetermined_campaign_status_report(manifest, manifest_path=manifest_path)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if args.fail_on_blocked and not bool(report["passed"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
