#!/usr/bin/env python3
"""Report progress for a matched nonlinear transport matrix campaign.

This checker consumes the manifest written by
``build_matched_nonlinear_transport_matrix.py write``.  It is intentionally
stricter than a file-count check: checkpointed nonlinear runs can create
``.out.nc/.restart.nc/.big.nc`` bundles before the final horizon is reached, so
this report separately records bundle presence and confirmed output time.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def _repo_relative(path: Path | str) -> str:
    raw = Path(path)
    try:
        return raw.resolve().relative_to(ROOT.resolve()).as_posix()
    except (OSError, ValueError):
        return raw.as_posix()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _resolve_path(raw: str, *, manifest_path: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    repo_candidate = ROOT / path
    if repo_candidate.exists():
        return repo_candidate
    manifest_candidate = manifest_path.parent / path
    if manifest_candidate.exists():
        return manifest_candidate
    return repo_candidate


def _bundle_base(path: Path) -> Path:
    name = path.name
    for suffix in (".out.nc", ".big.nc", ".restart.nc"):
        if name.endswith(suffix):
            return path.with_name(name[: -len(suffix)])
    return path.with_suffix("") if path.suffix == ".nc" else path


def _bundle_paths(output_path: Path) -> dict[str, Path]:
    base = _bundle_base(output_path)
    return {
        "out": Path(f"{base}.out.nc"),
        "restart": Path(f"{base}.restart.nc"),
        "big": Path(f"{base}.big.nc"),
    }


def _last_finite_time(values: Any) -> float | None:
    import numpy as np  # noqa: PLC0415

    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return None
    value = float(arr.reshape(-1)[-1])
    return value if math.isfinite(value) else None


def _read_output_tmax(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        import netCDF4  # type: ignore[import-not-found]  # noqa: PLC0415

        with netCDF4.Dataset(str(path)) as dataset:
            for group_name, variable_name in (
                ("Grids", "time"),
                ("Diagnostics", "time"),
                ("", "time"),
                ("", "t"),
            ):
                group = dataset if not group_name else dataset.groups.get(group_name)
                if group is None or variable_name not in group.variables:
                    continue
                value = _last_finite_time(group.variables[variable_name][:])
                if value is not None:
                    return value
    except Exception:
        pass
    try:
        from spectraxgk.artifacts.nonlinear_netcdf_diagnostics import (  # noqa: PLC0415
            load_nonlinear_netcdf_diagnostics,
        )

        diagnostics = load_nonlinear_netcdf_diagnostics(path)
        return _last_finite_time(getattr(diagnostics, "t", []))
    except Exception:
        return None


def _iter_expected_outputs(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    samples = manifest.get("samples")
    if not isinstance(samples, list):
        raise ValueError("matrix manifest is missing a samples list")
    for sample in samples:
        if not isinstance(sample, Mapping):
            continue
        states = sample.get("states")
        if not isinstance(states, Mapping):
            continue
        for role, state in states.items():
            if not isinstance(state, Mapping):
                continue
            for output in state.get("final_outputs", []):
                rows.append(
                    {
                        "sample_id": sample.get("sample_id"),
                        "surface_torflux": sample.get("surface_torflux"),
                        "alpha": sample.get("alpha"),
                        "ky": sample.get("ky"),
                        "role": str(role),
                        "state_label": state.get("label"),
                        "output": str(output),
                    }
                )
    return rows


def _default_time_tolerance(cfg: Mapping[str, Any]) -> float:
    """Allow fixed-step output grids to stop just shy of the nominal horizon."""

    dt_values: list[float] = []
    for key in ("dt",):
        try:
            value = float(cfg.get(key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0.0:
            dt_values.append(value)
    raw_variants = cfg.get("dt_variants", ())
    if isinstance(raw_variants, (list, tuple)):
        for raw in raw_variants:
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and value > 0.0:
                dt_values.append(value)
    if not dt_values:
        return 1.0e-9
    return max(1.0e-9, 2.0 * max(dt_values))


def build_report(
    *,
    matrix_manifest: Path,
    target_time: float | None = None,
    time_tolerance: float | None = None,
    skip_time_check: bool = False,
) -> dict[str, Any]:
    manifest = _load_json(matrix_manifest)
    cfg = manifest.get("config") if isinstance(manifest.get("config"), Mapping) else {}
    effective_time_tolerance = (
        _default_time_tolerance(cfg) if time_tolerance is None else float(time_tolerance)
    )
    if target_time is None:
        window = cfg.get("window") if isinstance(cfg.get("window"), Mapping) else {}
        target_time = float(window.get("tmax", 0.0) or 0.0)
    if target_time <= 0.0:
        raise ValueError("target time must be positive; pass --target-time if the manifest has no window.tmax")
    expected = _iter_expected_outputs(manifest)
    rows: list[dict[str, Any]] = []
    complete_bundles = 0
    confirmed_targets = 0
    for item in expected:
        output_path = _resolve_path(str(item["output"]), manifest_path=matrix_manifest)
        bundle = _bundle_paths(output_path)
        present = {key: path.exists() for key, path in bundle.items()}
        bundle_complete = all(present.values())
        complete_bundles += int(bundle_complete)
        tmax = _read_output_tmax(bundle["out"])
        target_confirmed = bool(
            bundle_complete
            and (
                skip_time_check
                or (
                    tmax is not None
                    and tmax >= float(target_time) - float(effective_time_tolerance)
                )
            )
        )
        confirmed_targets += int(target_confirmed)
        rows.append(
            {
                **item,
                "output_path": _repo_relative(output_path),
                "bundle": {key: _repo_relative(path) for key, path in bundle.items()},
                "present": present,
                "bundle_complete": bundle_complete,
                "output_tmax": tmax,
                "target_time_confirmed": target_confirmed,
            }
        )
    expected_count = len(expected)
    return {
        "kind": "matched_nonlinear_transport_matrix_progress_report",
        "matrix_manifest": _repo_relative(matrix_manifest),
        "target_time": float(target_time),
        "time_tolerance": float(effective_time_tolerance),
        "skip_time_check": bool(skip_time_check),
        "summary": {
            "expected_outputs": expected_count,
            "complete_bundles": complete_bundles,
            "target_time_confirmed": confirmed_targets,
            "missing_or_incomplete_bundles": expected_count - complete_bundles,
            "not_confirmed_at_target_time": expected_count - confirmed_targets,
            "ready_for_postprocess": bool(expected_count and confirmed_targets == expected_count),
        },
        "rows": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-manifest", required=True, type=Path)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--target-time", type=float)
    parser.add_argument(
        "--time-tolerance",
        type=float,
        help=(
            "Absolute tolerance for confirming the final time. Defaults to two "
            "matrix time steps so fixed-step output grids that stop just shy of "
            "the nominal horizon are not misclassified as incomplete."
        ),
    )
    parser.add_argument(
        "--skip-time-check",
        action="store_true",
        help="Count complete bundles as target-confirmed without reading NetCDF time. Intended only for manifest unit tests.",
    )
    parser.add_argument("--fail-on-incomplete", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_report(
        matrix_manifest=args.matrix_manifest,
        target_time=args.target_time,
        time_tolerance=(
            None if args.time_tolerance is None else float(args.time_tolerance)
        ),
        skip_time_check=bool(args.skip_time_check),
    )
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    if args.fail_on_incomplete and not bool(report["summary"]["ready_for_postprocess"]):
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
