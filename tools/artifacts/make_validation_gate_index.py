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


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GLOB = str(REPO_ROOT / "docs" / "_static" / "**" / "*.json")
DEFAULT_JSON = REPO_ROOT / "docs" / "_static" / "validation_gate_index.json"
DEFAULT_CSV = REPO_ROOT / "docs" / "_static" / "validation_gate_index.csv"
DEFAULT_PNG = REPO_ROOT / "docs" / "_static" / "validation_gate_index.png"

CURATED_RELEASE_GATE_ARTIFACTS = {
    "docs/_static/external_vmec_dshape_t250_n64_transport_window.json",
    "docs/_static/external_vmec_itermodel_t350_n64_transport_window.json",
    "docs/_static/external_vmec_circular_t450_n64_transport_window.json",
    "docs/_static/nonlinear_cyclone_miller_gate_summary.json",
    "docs/_static/nonlinear_cyclone_gate_summary.json",
    "docs/_static/cyclone_resolution_observed_order.json",
    "docs/_static/nonlinear_hsx_gate_summary.json",
    "docs/_static/kbm_branch_gate_summary.json",
    "docs/_static/reference_modes/kbm_eigenfunction_reference_overlay_ky0p3000.json",
    "docs/_static/nonlinear_kbm_gate_summary.json",
    "docs/_static/miller_zonal_response_pilot.json",
    "docs/_static/quasilinear_promotion_guardrails.json",
    "docs/_static/quasilinear_model_selection_status.json",
    "docs/_static/external_vmec_updown_asym_t450_n64_transport_window.json",
    "docs/_static/reference_modes/w7x_eigenfunction_reference_overlay_ky0p3000.json",
    "docs/_static/nonlinear_w7x_gate_summary.json",
}


def _repo_relative_path(path: Path) -> str:
    """Return a stable repo-relative path when ``path`` is inside the checkout."""

    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _repo_relative_pattern(pattern: str) -> str:
    """Return a stable repo-relative glob pattern when possible."""

    repo = str(REPO_ROOT.resolve())
    raw = str(pattern)
    return raw[len(repo) + 1 :] if raw.startswith(repo + "/") else raw


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
    if data.get("gate_index_include") is False:
        return []
    artifact = _repo_relative_path(path)
    if (
        artifact.startswith("docs/_static/")
        and data.get("gate_index_include") is not True
        and artifact not in CURATED_RELEASE_GATE_ARTIFACTS
    ):
        return []

    reports: list[dict[str, object]] = []
    report = data.get("gate_report")
    if isinstance(report, dict):
        reports.append(report)
    extra = data.get("gate_reports")
    if isinstance(extra, list):
        reports.extend(item for item in extra if isinstance(item, dict))
    promotion_gate = data.get("promotion_gate")
    if (
        data.get("gate_index_include") is True
        and not reports
        and isinstance(promotion_gate, dict)
    ):
        reports.append(
            {
                "case": data.get("case", path.stem),
                "source": data.get("source", data.get("kind", "")),
                "passed": bool(promotion_gate.get("passed", False)),
                "gates": promotion_gate.get("gates", []),
                "max_abs_error": promotion_gate.get("max_abs_error"),
                "max_rel_error": promotion_gate.get("max_rel_error"),
            }
        )

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
                "artifact": _repo_relative_path(path),
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
        paths.extend(Path(item) for item in glob.glob(pattern, recursive=True))
    entries: list[dict[str, object]] = []
    for path in sorted(set(paths)):
        try:
            data = _load_json(path)
        except ValueError:
            continue
        entries.extend(_report_entries(path, data))
    return sorted(entries, key=lambda row: str(row["case"]))


def build_index(patterns: list[str]) -> dict[str, object]:
    entries = collect_gate_entries(patterns)
    n_passed = sum(1 for row in entries if bool(row["passed"]))
    payload = {
        "patterns": [_repo_relative_pattern(pattern) for pattern in patterns],
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
    fig, ax = plt.subplots(figsize=(9.5, fig_height))
    ax.barh(y, np.ones(len(rows)), color=colors, alpha=0.88)
    ax.set_yticks(y, labels, fontsize=8.8)
    ax.set_xticks([])
    ax.set_xlim(0.0, 1.0)
    ax.invert_yaxis()
    fig.suptitle("Validation Gate Index", fontsize=13, fontweight="bold")
    for idx, row in enumerate(rows):
        failed = str(row["failed_metrics"]).replace("_", " ")
        status = "passed" if bool(row["passed"]) else f"open: {failed}"
        status = textwrap.shorten(status, width=64, placeholder="...")
        ax.text(0.02, idx, status, va="center", ha="left", color="white", fontsize=8.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.55, right=0.97, top=0.94, bottom=0.03)
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate JSON validation gate reports."
    )
    parser.add_argument(
        "--glob",
        action="append",
        dest="patterns",
        default=None,
        help="JSON glob to scan.",
    )
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
    args.out_json.write_text(
        json.dumps(index, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
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
