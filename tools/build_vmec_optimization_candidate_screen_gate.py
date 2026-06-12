#!/usr/bin/env python3
"""Build a fail-closed screen for solved VMEC optimization-result WOUTs.

This gate is intentionally cheaper than a nonlinear holdout audit.  It reads
SPECTRAX-GK runtime linear-scan quasilinear spectra and decides whether any
case is eligible for an expensive nonlinear launch.  Positive growth alone is
not enough: the effective ``k_perp^2`` metric and linear heat-flux weights must
also be finite and physically admissible.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_OUT = ROOT / "docs" / "_static" / "vmec_optimization_candidate_screen_gate.json"


def _repo_relative(path: str | Path) -> str:
    target = Path(path)
    try:
        return target.resolve().relative_to(ROOT.resolve()).as_posix()
    except (OSError, ValueError):
        return str(path)


def _finite_float(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _read_spectrum(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"ky", "gamma", "omega", "kperp_eff2", "heat_flux_weight_total"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        for raw in reader:
            row = {name: _finite_float(raw.get(name)) for name in required}
            if any(value is None for value in row.values()):
                continue
            rows.append({name: float(value) for name, value in row.items() if value is not None})
    if not rows:
        raise ValueError(f"{path} has no finite candidate-screen rows")
    rows.sort(key=lambda item: item["ky"])
    return rows


def _case_status(
    *,
    max_gamma: float,
    min_kperp_eff2: float,
    finite_heat_weights: bool,
    n_rows: int,
    min_launch_gamma: float,
    min_points: int,
) -> tuple[str, list[str], bool]:
    blockers: list[str] = []
    if n_rows < int(min_points):
        blockers.append("too_few_ky_points")
    if max_gamma < float(min_launch_gamma):
        blockers.append("below_nonlinear_launch_growth")
    if min_kperp_eff2 <= 0.0 or not math.isfinite(min_kperp_eff2):
        blockers.append("nonpositive_effective_kperp2")
    if not finite_heat_weights:
        blockers.append("nonfinite_heat_flux_weight")
    if blockers:
        if "nonpositive_effective_kperp2" in blockers:
            return "invalid_metric_nonpositive_kperp2", blockers, False
        if max_gamma <= 0.0:
            return "stable_or_damped", blockers, False
        return "marginal_or_incomplete_screen", blockers, False
    return "nonlinear_launch_candidate", blockers, True


def summarize_spectrum(
    *,
    label: str,
    spectrum_path: str | Path,
    min_launch_gamma: float = 0.02,
    min_points: int = 3,
) -> dict[str, Any]:
    """Return one fail-closed candidate-screen row."""

    path = Path(spectrum_path)
    rows = _read_spectrum(path)
    gamma = np.asarray([row["gamma"] for row in rows], dtype=float)
    ky = np.asarray([row["ky"] for row in rows], dtype=float)
    omega = np.asarray([row["omega"] for row in rows], dtype=float)
    kperp = np.asarray([row["kperp_eff2"] for row in rows], dtype=float)
    heat = np.asarray([row["heat_flux_weight_total"] for row in rows], dtype=float)
    imax = int(np.nanargmax(gamma))
    status, blockers, passed = _case_status(
        max_gamma=float(gamma[imax]),
        min_kperp_eff2=float(np.nanmin(kperp)),
        finite_heat_weights=bool(np.all(np.isfinite(heat))),
        n_rows=int(gamma.size),
        min_launch_gamma=float(min_launch_gamma),
        min_points=int(min_points),
    )
    return {
        "label": str(label),
        "source": _repo_relative(path),
        "status": status,
        "passed": passed,
        "blockers": blockers,
        "n_ky": int(gamma.size),
        "max_gamma": float(gamma[imax]),
        "max_gamma_ky": float(ky[imax]),
        "omega_at_max_gamma": float(omega[imax]),
        "min_kperp_eff2": float(np.nanmin(kperp)),
        "max_heat_flux_weight_total": float(np.nanmax(heat)),
        "min_launch_gamma": float(min_launch_gamma),
    }


def build_report(
    spectra: list[tuple[str, Path]],
    *,
    min_launch_gamma: float = 0.02,
    min_points: int = 3,
) -> dict[str, Any]:
    """Build a JSON-ready optimization-result candidate-screen gate."""

    rows = [
        summarize_spectrum(
            label=label,
            spectrum_path=path,
            min_launch_gamma=min_launch_gamma,
            min_points=min_points,
        )
        for label, path in spectra
    ]
    candidates = [row for row in rows if row["passed"]]
    return {
        "kind": "vmec_optimization_candidate_screen_gate",
        "claim_level": "linear_candidate_screen_not_nonlinear_transport_validation",
        "passed": bool(candidates),
        "absolute_flux_promoted": False,
        "n_cases": len(rows),
        "n_launch_candidates": len(candidates),
        "min_launch_gamma": float(min_launch_gamma),
        "min_points": int(min_points),
        "rows": rows,
        "launch_candidates": candidates,
        "notes": (
            "This gate screens solved VMEC optimization-result WOUTs before any nonlinear holdout launch. "
            "A case must have finite growth, max gamma above the nonlinear-launch threshold, finite heat-flux "
            "weights, and positive effective k_perp^2 on every sampled ky. It is not a nonlinear transport "
            "validation or quasilinear absolute-flux promotion."
        ),
    }


def _parse_spectrum_spec(raw: str) -> tuple[str, Path]:
    if ":" not in raw:
        raise argparse.ArgumentTypeError("--spectrum must have form LABEL:PATH")
    label, path = raw.split(":", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("--spectrum label cannot be empty")
    return label, Path(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "label",
        "status",
        "passed",
        "max_gamma",
        "max_gamma_ky",
        "omega_at_max_gamma",
        "min_kperp_eff2",
        "max_heat_flux_weight_total",
        "blockers",
        "source",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            out = {field: row.get(field, "") for field in fields}
            out["blockers"] = ";".join(str(item) for item in row.get("blockers", []))
            writer.writerow(out)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spectrum", action="append", type=_parse_spectrum_spec, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-launch-gamma", type=float, default=0.02)
    parser.add_argument("--min-points", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_report(
        list(args.spectrum),
        min_launch_gamma=float(args.min_launch_gamma),
        min_points=int(args.min_points),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out.with_suffix(".csv")
    report["csv"] = _repo_relative(csv_path)
    out.write_text(json.dumps(report, separators=(",", ":"), sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(csv_path, list(report["rows"]))
    print(json.dumps({"passed": report["passed"], "json": str(out), "csv": str(csv_path)}, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
