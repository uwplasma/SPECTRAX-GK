#!/usr/bin/env python3
"""Build a validation-gate index from JSON artifact metadata."""

from __future__ import annotations

import argparse
import glob
import json
import math
import textwrap
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GLOB = str(REPO_ROOT / "docs" / "_static" / "*.json")
DEFAULT_JSON = REPO_ROOT / "docs" / "_static" / "validation_gate_index.json"
DEFAULT_CSV = REPO_ROOT / "docs" / "_static" / "validation_gate_index.csv"
DEFAULT_PNG = REPO_ROOT / "docs" / "_static" / "validation_gate_index.png"


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, np.generic):
        return _json_clean(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _load_json(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return data


def _report_entries(path: Path, data: dict[str, object]) -> list[dict[str, object]]:
    reports: list[dict[str, object]] = []
    report = data.get("gate_report")
    if isinstance(report, dict):
        reports.append(report)
    extra = data.get("gate_reports")
    if isinstance(extra, list):
        reports.extend(item for item in extra if isinstance(item, dict))

    entries: list[dict[str, object]] = []
    for report in reports:
        gates = report.get("gates", [])
        gate_list = gates if isinstance(gates, list) else []
        failed = [
            str(gate.get("metric", "unknown"))
            for gate in gate_list
            if isinstance(gate, dict) and not bool(gate.get("passed", False))
        ]
        entries.append(
            {
                "artifact": str(path),
                "case": str(report.get("case", data.get("case", path.stem))),
                "source": str(report.get("source", data.get("source", ""))),
                "passed": bool(report.get("passed", False)),
                "n_gates": len(gate_list),
                "n_failed": len(failed),
                "failed_metrics": ",".join(failed),
                "max_abs_error": report.get("max_abs_error"),
                "max_rel_error": report.get("max_rel_error"),
            }
        )
    return entries


def collect_gate_entries(patterns: list[str]) -> list[dict[str, object]]:
    """Collect gate-report rows from one or more glob patterns."""

    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path(item) for item in glob.glob(pattern))
    entries: list[dict[str, object]] = []
    for path in sorted(set(paths)):
        data = _load_json(path)
        entries.extend(_report_entries(path, data))
    return sorted(entries, key=lambda row: str(row["case"]))


def build_index(patterns: list[str]) -> dict[str, object]:
    entries = collect_gate_entries(patterns)
    n_passed = sum(1 for row in entries if bool(row["passed"]))
    payload = {
        "patterns": patterns,
        "n_reports": len(entries),
        "n_passed": n_passed,
        "n_open": len(entries) - n_passed,
        "reports": entries,
    }
    return _json_clean(payload)


def write_index_plot(rows: list[dict[str, object]], out_png: Path) -> None:
    if not rows:
        return
    labels = [str(row["case"]).replace("_", " ") for row in rows]
    y = np.arange(len(rows))
    colors = ["#2a9d55" if bool(row["passed"]) else "#c2410c" for row in rows]
    fig_height = max(2.6, 0.45 * len(rows) + 1.2)
    fig, ax = plt.subplots(figsize=(7.0, fig_height), constrained_layout=True)
    ax.barh(y, np.ones(len(rows)), color=colors, alpha=0.88)
    ax.set_yticks(y, labels)
    ax.set_xticks([])
    ax.set_xlim(0.0, 1.0)
    ax.invert_yaxis()
    ax.set_title("Validation Gate Index", fontsize=13, fontweight="bold")
    for idx, row in enumerate(rows):
        failed = str(row["failed_metrics"]).replace("_", " ")
        status = "passed" if bool(row["passed"]) else f"open: {failed}"
        status = textwrap.shorten(status, width=64, placeholder="...")
        ax.text(0.02, idx, status, va="center", ha="left", color="white", fontsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate JSON validation gate reports.")
    parser.add_argument("--glob", action="append", dest="patterns", default=None, help="JSON glob to scan.")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-png", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--no-plot", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    patterns = args.patterns or [DEFAULT_GLOB]
    index = build_index(patterns)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(index, indent=2, sort_keys=True, allow_nan=False) + "\n")
    rows = list(index["reports"])
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    if not args.no_plot:
        write_index_plot(rows, args.out_png)
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_csv}")
    if not args.no_plot:
        print(f"Wrote {args.out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
