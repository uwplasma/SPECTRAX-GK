#!/usr/bin/env python3
"""Plot the W7-X nonlinear exact-state convention audit against GX.

The audit consumes the output of ``tools/run_exact_state_audit.py`` for the
``w7x_vmec`` lane. It is intentionally a no-rerun plotting/reporting step: the
state dumps already contain the GX startup and late diagnostic states, and this
script turns their scalar/array agreement into a tracked publication artifact.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from typing import Any

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from spectraxgk.plotting import set_plot_style  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT_DIR = ROOT / "tools_out" / "exact_state_audit_w7x_20260424" / "w7x_vmec"
DEFAULT_OUT = ROOT / "docs" / "_static" / "w7x_exact_state_audit.png"
REL_FLOOR = 1.0e-12
PASS_THRESHOLD = 1.0e-4

SUMMARY_RE = re.compile(
    r"^(?P<metric>\S+)\s+"
    r"max\|ref\|=(?P<max_ref>\S+)\s+"
    r"max\|test\|=(?P<max_test>\S+)\s+"
    r"max\|diff\|=(?P<max_diff>\S+)\s+"
    r"max\|rel\|=(?P<max_rel>\S+)\s+"
    r"rms_rel=(?P<rms_rel>\S+)"
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--out-png", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--pass-threshold", type=float, default=PASS_THRESHOLD)
    return parser.parse_args(argv)


def _repo_relative(path: Path | str) -> str:
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _as_float(value: str | float | int) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


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


def parse_array_summary_log(path: Path, *, phase: str) -> list[dict[str, object]]:
    """Parse ``_summary`` lines from an exact-state audit log."""

    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = SUMMARY_RE.match(line.strip())
        if match is None:
            continue
        data = match.groupdict()
        rows.append(
            {
                "phase": phase,
                "kind": "array",
                "metric": data["metric"],
                "quantity": "max_rel",
                "value": _as_float(data["max_rel"]),
                "max_ref": _as_float(data["max_ref"]),
                "max_test": _as_float(data["max_test"]),
                "max_diff": _as_float(data["max_diff"]),
                "rms_rel": _as_float(data["rms_rel"]),
                "source_path": _repo_relative(path),
            }
        )
    return rows


def parse_diagnostic_csv(path: Path) -> list[dict[str, object]]:
    """Load scalar diagnostic relative errors from ``diag_state.csv``."""

    table = pd.read_csv(path)
    required = {"metric", "gx_out", "spectrax_dump", "rel_dump", "spectrax_solve", "rel_solve"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for item in table.to_dict(orient="records"):
        for quantity in ("rel_dump", "rel_solve"):
            rows.append(
                {
                    "phase": "late diagnostics",
                    "kind": "diagnostic",
                    "metric": str(item["metric"]),
                    "quantity": quantity,
                    "value": float(item[quantity]),
                    "max_ref": float(item["gx_out"]),
                    "max_test": float(item["spectrax_dump" if quantity == "rel_dump" else "spectrax_solve"]),
                    "max_diff": abs(
                        float(item["gx_out"])
                        - float(item["spectrax_dump" if quantity == "rel_dump" else "spectrax_solve"])
                    ),
                    "rms_rel": float("nan"),
                    "source_path": _repo_relative(path),
                }
            )
    return rows


def build_rows(audit_dir: Path) -> list[dict[str, object]]:
    """Build the long-form exact-state audit table from an audit directory."""

    startup_log = audit_dir / "startup.log"
    diag_log = audit_dir / "diag_state.log"
    diag_csv = audit_dir / "diag_state.csv"
    missing = [path for path in (startup_log, diag_log, diag_csv) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing exact-state audit files: {', '.join(str(path) for path in missing)}")
    rows = []
    rows.extend(parse_array_summary_log(startup_log, phase="startup"))
    rows.extend(parse_array_summary_log(diag_log, phase="late arrays"))
    rows.extend(parse_diagnostic_csv(diag_csv))
    return rows


def _finite_plot_value(value: object) -> float | None:
    value_f = float(value)
    if not np.isfinite(value_f):
        return None
    return max(value_f, REL_FLOOR)


def _bar_panel(
    ax: plt.Axes,
    rows: list[dict[str, object]],
    *,
    title: str,
    threshold: float,
    color: str,
    quantity: str = "max_rel",
) -> None:
    filtered = [row for row in rows if row["quantity"] == quantity and _finite_plot_value(row["value"]) is not None]
    labels = [str(row["metric"]) for row in filtered]
    values = [_finite_plot_value(row["value"]) or REL_FLOOR for row in filtered]
    if not filtered:
        ax.text(0.5, 0.5, "no finite rows", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    xpos = np.arange(len(filtered))
    ax.bar(xpos, values, color=color, alpha=0.86)
    ax.axhline(threshold, color="#991b1b", linestyle="--", linewidth=1.2, label=f"{threshold:.0e} gate")
    ax.set_yscale("log")
    ax.set_ylim(REL_FLOOR * 0.8, max(max(values) * 6.0, threshold * 4.0))
    ax.set_xticks(xpos, labels, rotation=35, ha="right")
    ax.set_ylabel("relative error")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    for xval, row, yval in zip(xpos, filtered, values, strict=True):
        if float(row["value"]) == 0.0:
            ax.text(xval, REL_FLOOR * 1.8, "0", ha="center", va="bottom", fontsize=8)


def exact_state_figure(rows: list[dict[str, object]], *, threshold: float) -> plt.Figure:
    """Create the W7-X exact-state convention audit figure."""

    set_plot_style()
    startup = [row for row in rows if row["phase"] == "startup" and row["kind"] == "array"]
    late_arrays = [
        row
        for row in rows
        if row["phase"] == "late arrays" and row["kind"] == "array" and str(row["metric"]) not in {"apar", "bpar"}
    ]
    diagnostics = [row for row in rows if row["kind"] == "diagnostic"]

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.9), constrained_layout=True)
    _bar_panel(axes[0], startup, title="Startup state", threshold=threshold, color="#0f4c81")
    _bar_panel(axes[1], late_arrays, title="Late dumped arrays", threshold=threshold, color="#2a9d8f")

    diag_table = pd.DataFrame(diagnostics)
    diag_metrics = list(dict.fromkeys(diag_table["metric"].astype(str))) if not diag_table.empty else []
    xpos = np.arange(len(diag_metrics))
    width = 0.36
    for offset, quantity, color, label in (
        (-0.5 * width, "rel_dump", "#7b2cbf", "dumped fields"),
        (0.5 * width, "rel_solve", "#c2410c", "re-solved fields"),
    ):
        values = []
        raw_values = []
        for metric in diag_metrics:
            subset = diag_table[(diag_table["metric"] == metric) & (diag_table["quantity"] == quantity)]
            value = float(subset["value"].iloc[0]) if not subset.empty else float("nan")
            raw_values.append(value)
            values.append(max(value, REL_FLOOR) if np.isfinite(value) else REL_FLOOR)
        axes[2].bar(xpos + offset, values, width=width, color=color, alpha=0.86, label=label)
        for xval, raw, plotted in zip(xpos + offset, raw_values, values, strict=True):
            if np.isfinite(raw) and raw == 0.0:
                axes[2].text(xval, REL_FLOOR * 1.8, "0", ha="center", va="bottom", fontsize=8)
    axes[2].axhline(threshold, color="#991b1b", linestyle="--", linewidth=1.2, label=f"{threshold:.0e} gate")
    axes[2].set_yscale("log")
    axes[2].set_ylim(REL_FLOOR * 0.8, threshold * 4.0)
    axes[2].set_xticks(xpos, diag_metrics, rotation=35, ha="right")
    axes[2].set_ylabel("relative error")
    axes[2].set_title("Late scalar diagnostics")
    axes[2].grid(True, axis="y", alpha=0.25)
    axes[2].legend(frameon=False, fontsize=8)
    fig.suptitle("W7-X nonlinear exact-state convention audit", y=1.03, fontsize=14, fontweight="bold")
    return fig


def write_outputs(
    rows: list[dict[str, object]],
    *,
    audit_dir: Path,
    out_png: Path,
    out_csv: Path,
    out_json: Path,
    threshold: float,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    finite = [float(row["value"]) for row in rows if np.isfinite(float(row["value"]))]
    max_rel = float(max(finite)) if finite else float("nan")
    payload = {
        "case": "w7x_nonlinear_exact_state_audit",
        "validation_status": "closed" if np.isfinite(max_rel) and max_rel <= threshold else "open",
        "gate_index_include": False,
        "reference": "GX W7-X nonlinear VMEC exact-state startup and late diagnostic dumps",
        "audit_dir": _repo_relative(audit_dir),
        "audit_csv": _repo_relative(out_csv),
        "audit_png": _repo_relative(out_png),
        "pass_threshold": float(threshold),
        "max_finite_relative_error": max_rel,
        "rows": rows,
        "notes": (
            "This audit checks state/geometry/fieldsolve/diagnostic conventions on exact GX W7-X nonlinear states. "
            "It closes those convention layers but does not close the separate W7-X zonal-response recurrence "
            "and damping-envelope literature lane."
        ),
    }
    out_json.write_text(json.dumps(_json_clean(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rows = build_rows(args.audit_dir)
    out_csv = args.out_csv or args.out_png.with_suffix(".csv")
    out_json = args.out_json or args.out_png.with_suffix(".json")
    fig = exact_state_figure(rows, threshold=float(args.pass_threshold))
    fig.savefig(args.out_png, dpi=240, bbox_inches="tight")
    fig.savefig(args.out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    write_outputs(
        rows,
        audit_dir=args.audit_dir,
        out_png=args.out_png,
        out_csv=out_csv,
        out_json=out_json,
        threshold=float(args.pass_threshold),
    )
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_json}")
    print(f"Wrote {args.out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
