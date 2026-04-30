#!/usr/bin/env python3
"""Build an executable status panel for W7-X fluctuation/TEM validation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import textwrap
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from spectraxgk.plotting import set_plot_style  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "docs" / "_static" / "w7x_tem_extension_status.png"
DEFAULT_W7X_SPECTRUM = ROOT / "docs" / "_static" / "w7x_fluctuation_spectrum_panel.json"
DEFAULT_TEM_TABLE = ROOT / "docs" / "_static" / "tem_mismatch_table.csv"

STATUS_ORDER = {"closed": 0, "partial": 1, "open": 2}
STATUS_COLORS = {"closed": "#2a9d8f", "partial": "#e9c46a", "open": "#f4a261"}


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _finite_float(value: object) -> float | None:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


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


def _tem_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "available": False,
            "n_rows": 0,
            "max_abs_rel_gamma": None,
            "max_abs_rel_omega": None,
            "worst_gamma_ky": None,
            "worst_omega_ky": None,
        }
    rows: list[dict[str, str]] = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    gamma_values: list[tuple[float, float]] = []
    omega_values: list[tuple[float, float]] = []
    for row in rows:
        ky = _finite_float(row.get("ky"))
        rel_gamma = _finite_float(row.get("rel_gamma"))
        rel_omega = _finite_float(row.get("rel_omega"))
        if ky is not None and rel_gamma is not None:
            gamma_values.append((ky, abs(rel_gamma)))
        if ky is not None and rel_omega is not None:
            omega_values.append((ky, abs(rel_omega)))
    worst_gamma = max(gamma_values, key=lambda item: item[1]) if gamma_values else (None, None)
    worst_omega = max(omega_values, key=lambda item: item[1]) if omega_values else (None, None)
    return {
        "available": True,
        "n_rows": len(rows),
        "max_abs_rel_gamma": worst_gamma[1],
        "max_abs_rel_omega": worst_omega[1],
        "worst_gamma_ky": worst_gamma[0],
        "worst_omega_ky": worst_omega[0],
    }


def build_status_payload(
    *,
    w7x_spectrum: Path = DEFAULT_W7X_SPECTRUM,
    tem_table: Path = DEFAULT_TEM_TABLE,
) -> dict[str, Any]:
    """Return a JSON-ready W7-X/TEM validation extension status payload."""

    spectrum = _read_json(Path(w7x_spectrum))
    tem = _tem_metrics(Path(tem_table))
    spectrum_closed = bool(spectrum and spectrum.get("source_gate_passed") is True)
    tem_gamma = _finite_float(tem.get("max_abs_rel_gamma"))
    tem_omega = _finite_float(tem.get("max_abs_rel_omega"))
    tem_open = tem_gamma is None or tem_omega is None or tem_gamma > 0.2 or tem_omega > 0.2
    rows = [
        {
            "lane": "W7-X nonlinear fluctuation spectrum",
            "status": "closed" if spectrum_closed else "open",
            "claim_level": "validated_simulation_spectrum_not_experimental_validation",
            "primary_artifact": "docs/_static/w7x_fluctuation_spectrum_panel.json",
            "key_metrics": {
                "time_samples": None if spectrum is None else spectrum.get("time_samples"),
                "dominant_phi_ky": None if spectrum is None else spectrum.get("dominant_phi_ky"),
                "dominant_heat_flux_ky": None if spectrum is None else spectrum.get("dominant_heat_flux_ky"),
            },
            "next_action": (
                "Keep as the released simulation-spectrum diagnostic; do not use as an experimental density-spectrum claim."
            ),
        },
        {
            "lane": "TEM / kinetic-electron linear parity",
            "status": "open" if tem_open else "closed",
            "claim_level": "open_linear_mismatch_blocks_tem_extension",
            "primary_artifact": "docs/_static/tem_mismatch_table.csv",
            "key_metrics": tem,
            "next_action": (
                "Fix the TEM branch/frequency mismatch before using kinetic-electron W7-X scans for validation or optimization."
            ),
        },
        {
            "lane": "W7-X multi-flux-tube and multi-surface scan",
            "status": "open",
            "claim_level": "untracked_required_manuscript_extension",
            "primary_artifact": None,
            "key_metrics": {
                "required_alpha_values": [0.0, "field-period displaced", "bad-curvature tube"],
                "required_torflux_values": [0.25, 0.50, 0.64],
                "requires_nonlinear_window_gates": True,
            },
            "next_action": (
                "Run alpha/surface-resolved W7-X ITG scans and promote only after linear branch, nonlinear window, "
                "and resolved-spectrum gates pass."
            ),
        },
        {
            "lane": "W7-X kinetic-electron/TEM nonlinear window",
            "status": "open",
            "claim_level": "not_started",
            "primary_artifact": None,
            "key_metrics": {
                "requires_kinetic_electrons": True,
                "requires_density_gradient_scan": True,
                "requires_independent_reference_or_literature_window": True,
            },
            "next_action": (
                "After TEM linear parity closes, run bounded kinetic-electron W7-X pilots and add matched nonlinear gates."
            ),
        },
    ]
    return {
        "kind": "w7x_tem_extension_status",
        "claim_scope": "w7x_fluctuation_tem_multi_flux_extension_tracking",
        "rows": rows,
        "summary": {
            "n_rows": len(rows),
            "n_closed": sum(1 for row in rows if row["status"] == "closed"),
            "n_partial": sum(1 for row in rows if row["status"] == "partial"),
            "n_open": sum(1 for row in rows if row["status"] == "open"),
        },
        "notes": (
            "This artifact is a claim-scope gate. It closes the reproducible W7-X simulation-spectrum "
            "estimator only; TEM, multi-flux-tube, and kinetic-electron W7-X validation remain open."
        ),
    }


def write_artifacts(payload: dict[str, Any], *, out_png: Path = DEFAULT_OUT) -> dict[str, str]:
    """Write PNG/PDF/JSON/CSV artifacts for the extension-status payload."""

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_json = out_png.with_suffix(".json")
    out_csv = out_png.with_suffix(".csv")
    out_pdf = out_png.with_suffix(".pdf")
    out_json.write_text(json.dumps(_json_clean(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    fieldnames = ["lane", "status", "claim_level", "primary_artifact", "next_action"]
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in payload["rows"]:
            writer.writerow({key: row.get(key) for key in fieldnames})

    set_plot_style()
    rows = payload["rows"]
    y = np.arange(len(rows))
    values = [STATUS_ORDER[str(row["status"])] for row in rows]
    labels = [textwrap.fill(str(row["lane"]), width=34) for row in rows]
    colors = [STATUS_COLORS[str(row["status"])] for row in rows]
    fig, ax = plt.subplots(figsize=(11.2, 5.0))
    ax.barh(y, values, color=colors, edgecolor="#333333", alpha=0.95)
    ax.set_yticks(y, labels)
    ax.set_xlim(0.0, 2.25)
    ax.set_xticks([0, 1, 2], ["closed", "partial", "open"])
    ax.invert_yaxis()
    ax.set_title("W7-X fluctuation, multi-flux, and TEM extension status")
    ax.grid(axis="x", alpha=0.25)
    for yi, row, value in zip(y, rows, values, strict=True):
        metrics = row.get("key_metrics", {})
        if row["lane"].startswith("W7-X nonlinear"):
            text = f"samples={metrics.get('time_samples')}, ky_phi={metrics.get('dominant_phi_ky'):.3g}"
        elif row["lane"].startswith("TEM"):
            text = f"TEM max |rel gamma|={metrics.get('max_abs_rel_gamma'):.2g}"
        elif row["lane"].startswith("W7-X multi"):
            text = "alpha/surface missing"
        else:
            text = "kinetic-e window missing"
        if str(row["status"]) == "open":
            xpos = 1.95
            ha = "right"
        else:
            xpos = min(float(value) + 0.05, 2.05)
            ha = "left"
        ax.text(xpos, yi, text, va="center", ha=ha, fontsize=8.1)
    fig.text(
        0.5,
        0.02,
        "Only the simulation-spectrum estimator is closed; TEM and multi-flux W7-X validation remain paper-level work.",
        ha="center",
        fontsize=8.2,
        color="#333333",
    )
    fig.subplots_adjust(left=0.36, right=0.98, top=0.88, bottom=0.16)
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_pdf)
    plt.close(fig)
    return {"png": str(out_png), "pdf": str(out_pdf), "json": str(out_json), "csv": str(out_csv)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--w7x-spectrum", type=Path, default=DEFAULT_W7X_SPECTRUM)
    parser.add_argument("--tem-table", type=Path, default=DEFAULT_TEM_TABLE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--json-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_status_payload(w7x_spectrum=args.w7x_spectrum, tem_table=args.tem_table)
    if args.json_only:
        print(json.dumps(_json_clean(payload), indent=2, sort_keys=True))
        return 0
    print(json.dumps(write_artifacts(payload, out_png=args.out), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
