#!/usr/bin/env python3
"""Compact replicated nonlinear-window ensemble artifacts for git tracking.

``build_external_vmec_replicate_ensemble.py`` writes per-trace CSV and
convergence-report intermediates. Those are useful on the workstation, but the
release repository tracks only the ensemble JSON, output-gate JSON, and plot.
This helper rewrites ensemble-row provenance so the compact JSON points back to
the authoritative NetCDF outputs and the tracked output-gate summary instead of
to intentionally untracked intermediate CSV files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any


_CASE_RE = re.compile(
    r"(?P<case>.+)_nonlinear_t[^_]+_[^_]+_(?:seed[0-9]+|dt[0-9pm]+)\.out\.nc$"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble-json", type=Path, required=True)
    parser.add_argument("--output-gate-json", required=True)
    parser.add_argument(
        "--netcdf-root",
        required=True,
        help="Prefix containing per-case NetCDF output directories, e.g. office:/.../tools_out/audits",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        help="Output path. Defaults to in-place rewrite of --ensemble-json.",
    )
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _output_stem(row: dict[str, Any]) -> str:
    source = str(row.get("source_artifact", ""))
    name = Path(source).name
    if name.endswith("_heat_flux_trace.csv"):
        return name[: -len("_heat_flux_trace.csv")] + ".out.nc"
    if name.endswith(".out.nc"):
        return name
    generated = str(row.get("generated_trace_artifact", ""))
    generated_name = Path(generated).name
    if generated_name.endswith("_heat_flux_trace.csv"):
        return generated_name[: -len("_heat_flux_trace.csv")] + ".out.nc"
    raise ValueError(f"cannot infer NetCDF output from row source_artifact={source!r}")


def _case_dir(output_name: str) -> str:
    match = _CASE_RE.match(output_name)
    if match is None:
        raise ValueError(
            f"cannot infer case directory from output name {output_name!r}"
        )
    return match.group("case")


def compact_ensemble_payload(
    payload: dict[str, Any],
    *,
    output_gate_json: str,
    netcdf_root: str,
) -> dict[str, Any]:
    """Return a compact-provenance copy of a replicated ensemble payload."""

    out = json.loads(json.dumps(payload))
    rows = out.get("rows")
    if not isinstance(rows, list):
        raise ValueError("ensemble payload must contain a rows list")
    root = netcdf_root.rstrip("/")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"row {index} must be a JSON object")
        output_name = _output_stem(row)
        previous_source = str(row.get("source_artifact", ""))
        row.setdefault("generated_trace_artifact", previous_source)
        row["source_artifact"] = f"{root}/{_case_dir(output_name)}/{output_name}"
        row["summary_artifact"] = f"{output_gate_json}#rows[{row.get('index', index)}]"
    out["compact_bundle_policy"] = {
        "tracked_bundle": "ensemble JSON, output-gate JSON, and PNG only",
        "reason": (
            "avoid committing regenerable per-trace CSV and convergence "
            "intermediates under repository-size policy"
        ),
        "regeneration": (
            "run tools/artifacts/build_external_vmec_replicate_ensemble.py against the "
            "NetCDF outputs listed in rows[].source_artifact"
        ),
        "output_gate_json": output_gate_json,
    }
    return out


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = compact_ensemble_payload(
        _load_json(args.ensemble_json),
        output_gate_json=str(args.output_gate_json),
        netcdf_root=str(args.netcdf_root),
    )
    out_path = args.out_json or args.ensemble_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"out_json": out_path.as_posix(), "rows": len(payload["rows"])},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
