#!/usr/bin/env python3
"""Write a KBM branch-continuity gate summary from an existing candidate CSV."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
DEFAULT_CANDIDATES = REPO_ROOT / "docs" / "_static" / "kbm_gx_candidates.csv"
DEFAULT_OUT = REPO_ROOT / "docs" / "_static" / "kbm_branch_gate_summary.json"


def _branch_gate_report_from_selected_rows(
    rows: list[dict[str, object]],
    *,
    max_rel_gamma_jump: float,
    max_rel_omega_jump: float,
    min_successive_overlap: float | None,
) -> dict[str, object] | None:
    if str(TOOLS_DIR) not in sys.path:
        sys.path.insert(0, str(TOOLS_DIR))
    from compare_gx_kbm import _branch_gate_report_from_rows

    return _branch_gate_report_from_rows(
        rows,
        max_rel_gamma_jump=max_rel_gamma_jump,
        max_rel_omega_jump=max_rel_omega_jump,
        min_successive_overlap=min_successive_overlap,
    )


def _coerce_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def _json_clean(value: Any) -> Any:
    """Return a strict-JSON-compatible copy with nonfinite numbers set to null."""

    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, np.generic):
        return _json_clean(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def selected_candidate_rows(path: Path) -> list[dict[str, object]]:
    """Load selected branch rows from a KBM candidate table."""

    table = pd.read_csv(path)
    required = {"ky", "gamma", "omega", "selected"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    selected = table["selected"].map(_coerce_bool)
    rows = table.loc[selected].sort_values("ky").to_dict(orient="records")
    return [_json_clean(row) for row in rows]


def build_summary(
    candidate_csv: Path,
    *,
    max_rel_gamma_jump: float,
    max_rel_omega_jump: float,
    min_successive_overlap: float | None,
) -> dict[str, object]:
    """Build the JSON payload for the selected KBM branch."""

    rows = selected_candidate_rows(candidate_csv)
    report = _branch_gate_report_from_selected_rows(
        rows,
        max_rel_gamma_jump=max_rel_gamma_jump,
        max_rel_omega_jump=max_rel_omega_jump,
        min_successive_overlap=min_successive_overlap,
    )
    payload: dict[str, object] = {
        "case": "kbm_linear_branch_continuity",
        "candidate_csv": str(candidate_csv),
        "selected_count": len(rows),
        "thresholds": {
            "max_rel_gamma_jump": float(max_rel_gamma_jump),
            "max_rel_omega_jump": float(max_rel_omega_jump),
            "min_successive_overlap": min_successive_overlap,
        },
        "rows": rows,
        "gate_report": report,
        "gate_passed": None if report is None else bool(report["passed"]),
        "notes": (
            "Selected rows are taken from the KBM GX comparison candidate table. "
            "The gate is intentionally a branch-identity check: adjacent gamma "
            "and omega jumps should stay smooth, and successive eigenfunction "
            "overlaps should remain high when those overlaps are available."
        ),
    }
    return _json_clean(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write a JSON KBM branch-continuity gate summary from selected candidate rows."
    )
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-rel-gamma-jump", type=float, default=0.5)
    parser.add_argument("--max-rel-omega-jump", type=float, default=0.5)
    parser.add_argument("--min-successive-overlap", type=float, default=0.95)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_summary(
        args.candidates,
        max_rel_gamma_jump=float(args.max_rel_gamma_jump),
        max_rel_omega_jump=float(args.max_rel_omega_jump),
        min_successive_overlap=float(args.min_successive_overlap),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
