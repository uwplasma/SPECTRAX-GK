#!/usr/bin/env python3
"""Build a branch-level audit for the provisional TEM parity lane."""

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

from spectraxgk.artifacts.plotting import set_plot_style  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TABLE = ROOT / "docs" / "_static" / "tem_mismatch_table.csv"
DEFAULT_REFERENCE = ROOT / "src" / "spectraxgk" / "data" / "tem_reference.csv"
DEFAULT_OUT = ROOT / "docs" / "_static" / "tem_branch_parity_audit.png"

REFERENCE_PROVENANCE = (
    "Digitized low-ky TEM branch from the literature figure tracked in "
    "src/spectraxgk/data/tem_reference.csv; this is not a direct code-to-code dump."
)

REQUIRED_COLUMNS = (
    "ky",
    "gamma_ref",
    "omega_ref",
    "gamma_spectrax",
    "omega_spectrax",
    "rel_gamma",
    "rel_omega",
)


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


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    ranks[order] = np.arange(values.size, dtype=float)
    unique = np.unique(values)
    for val in unique:
        mask = values == val
        if np.count_nonzero(mask) > 1:
            ranks[mask] = float(np.mean(ranks[mask]))
    return ranks


def _corrcoef(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or y.size < 2:
        return None
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    return _corrcoef(_rankdata(x), _rankdata(y))


def _load_table(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(set(REQUIRED_COLUMNS) - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        for row in reader:
            parsed: dict[str, float] = {}
            for column in REQUIRED_COLUMNS:
                value = _finite_float(row.get(column))
                if value is None:
                    raise ValueError(
                        f"{path} contains non-finite {column!r}: {row.get(column)!r}"
                    )
                parsed[column] = value
            rows.append(parsed)
    if not rows:
        raise ValueError(f"{path} contains no TEM mismatch rows")
    return sorted(rows, key=lambda row: row["ky"])


def _read_reference_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "path": str(path), "n_rows": 0}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        "available": True,
        "path": str(path),
        "n_rows": len(rows),
        "columns": list(rows[0].keys()) if rows else [],
        "provenance": REFERENCE_PROVENANCE,
    }


def build_audit_payload(
    *,
    table: Path = DEFAULT_TABLE,
    reference: Path = DEFAULT_REFERENCE,
) -> dict[str, Any]:
    """Return a JSON-ready TEM branch parity audit payload."""

    rows = _load_table(Path(table))
    ky = np.array([row["ky"] for row in rows], dtype=float)
    gamma_ref = np.array([row["gamma_ref"] for row in rows], dtype=float)
    omega_ref = np.array([row["omega_ref"] for row in rows], dtype=float)
    gamma_spectrax = np.array([row["gamma_spectrax"] for row in rows], dtype=float)
    omega_spectrax = np.array([row["omega_spectrax"] for row in rows], dtype=float)
    rel_gamma = np.array([row["rel_gamma"] for row in rows], dtype=float)
    rel_omega = np.array([row["rel_omega"] for row in rows], dtype=float)

    gamma_error = gamma_spectrax - gamma_ref
    omega_error = omega_spectrax - omega_ref
    gamma_abs_error = np.abs(gamma_error)
    omega_abs_error = np.abs(omega_error)
    gamma_sign_mismatch = np.signbit(gamma_ref) != np.signbit(gamma_spectrax)
    omega_sign_mismatch = np.signbit(omega_ref) != np.signbit(omega_spectrax)
    safe_omega = np.abs(omega_ref) >= 0.2
    if np.any(safe_omega):
        max_safe_rel_omega = float(np.max(np.abs(rel_omega[safe_omega])))
        worst_safe_rel_omega_ky = float(
            ky[safe_omega][int(np.argmax(np.abs(rel_omega[safe_omega])))]
        )
    else:
        max_safe_rel_omega = None
        worst_safe_rel_omega_ky = None

    omega_spearman = _spearman(omega_ref, omega_spectrax)
    gamma_spearman = _spearman(gamma_ref, gamma_spectrax)
    branch_inversion = omega_spearman is not None and omega_spearman < 0.0

    worst_gamma_idx = int(np.argmax(gamma_abs_error))
    worst_omega_idx = int(np.argmax(omega_abs_error))
    raw_worst_gamma_rel_idx = int(np.argmax(np.abs(rel_gamma)))
    raw_worst_omega_rel_idx = int(np.argmax(np.abs(rel_omega)))
    status = "open"
    rows_out = []
    for i, row in enumerate(rows):
        rows_out.append(
            {
                **row,
                "gamma_error": float(gamma_error[i]),
                "omega_error": float(omega_error[i]),
                "abs_gamma_error": float(gamma_abs_error[i]),
                "abs_omega_error": float(omega_abs_error[i]),
                "gamma_sign_mismatch": bool(gamma_sign_mismatch[i]),
                "omega_sign_mismatch": bool(omega_sign_mismatch[i]),
            }
        )

    return {
        "kind": "tem_branch_parity_audit",
        "status": status,
        "claim_level": "provisional_literature_digitization_not_closed_parity",
        "source_table": str(table),
        "reference": _read_reference_metadata(Path(reference)),
        "metrics": {
            "n_ky": int(ky.size),
            "ky_min": float(np.min(ky)),
            "ky_max": float(np.max(ky)),
            "max_abs_rel_gamma": float(np.max(np.abs(rel_gamma))),
            "max_abs_rel_gamma_ky": float(ky[raw_worst_gamma_rel_idx]),
            "max_abs_rel_omega_raw": float(np.max(np.abs(rel_omega))),
            "max_abs_rel_omega_raw_ky": float(ky[raw_worst_omega_rel_idx]),
            "max_abs_rel_omega_ref_ge_0p2": max_safe_rel_omega,
            "max_abs_rel_omega_ref_ge_0p2_ky": worst_safe_rel_omega_ky,
            "max_abs_gamma_error": float(gamma_abs_error[worst_gamma_idx]),
            "max_abs_gamma_error_ky": float(ky[worst_gamma_idx]),
            "max_abs_omega_error": float(omega_abs_error[worst_omega_idx]),
            "max_abs_omega_error_ky": float(ky[worst_omega_idx]),
            "gamma_rmse": float(np.sqrt(np.mean(gamma_error**2))),
            "omega_rmse": float(np.sqrt(np.mean(omega_error**2))),
            "gamma_sign_mismatch_count": int(np.count_nonzero(gamma_sign_mismatch)),
            "omega_sign_mismatch_count": int(np.count_nonzero(omega_sign_mismatch)),
            "gamma_spearman": gamma_spearman,
            "omega_spearman": omega_spearman,
            "omega_branch_inversion": bool(branch_inversion),
        },
        "rows": rows_out,
        "interpretation": (
            "The tracked SPECTRAX-GK low-ky TEM branch does not match the digitized reference: "
            "growth-rate amplitude/sign and frequency branch orientation fail simultaneously. "
            "Because the reference is a reconstructed literature digitization rather than a direct "
            "case dump, this artifact blocks broad TEM validation claims but should not be used by "
            "itself to tune solver physics."
        ),
        "next_action": (
            "Reconstruct the full TEM case definition or obtain an independent reference dump, then "
            "rerun a solver/signal/moment-resolution branch audit before promoting W7-X kinetic-electron claims."
        ),
    }


def write_artifacts(
    payload: dict[str, Any], *, out_png: Path = DEFAULT_OUT
) -> dict[str, str]:
    """Write PNG/PDF/JSON/CSV artifacts for a TEM audit payload."""

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_pdf = out_png.with_suffix(".pdf")
    out_json = out_png.with_suffix(".json")
    out_csv = out_png.with_suffix(".csv")
    out_json.write_text(
        json.dumps(_json_clean(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    rows = payload["rows"]
    fieldnames = [
        "ky",
        "gamma_ref",
        "omega_ref",
        "gamma_spectrax",
        "omega_spectrax",
        "rel_gamma",
        "rel_omega",
        "gamma_error",
        "omega_error",
        "gamma_sign_mismatch",
        "omega_sign_mismatch",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    set_plot_style()
    ky = np.array([row["ky"] for row in rows], dtype=float)
    gamma_ref = np.array([row["gamma_ref"] for row in rows], dtype=float)
    omega_ref = np.array([row["omega_ref"] for row in rows], dtype=float)
    gamma_spectrax = np.array([row["gamma_spectrax"] for row in rows], dtype=float)
    omega_spectrax = np.array([row["omega_spectrax"] for row in rows], dtype=float)
    gamma_error = np.abs(gamma_spectrax - gamma_ref)
    omega_error = np.abs(omega_spectrax - omega_ref)
    metrics = payload["metrics"]

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
    ax = axes[0, 0]
    ax.plot(ky, gamma_ref, "o-", label="digitized reference", color="#264653")
    ax.plot(ky, gamma_spectrax, "s--", label="SPECTRAX-GK tracked", color="#e76f51")
    ax.axhline(0.0, color="#444444", linewidth=0.8, alpha=0.5)
    ax.set_xlabel(r"$k_y \rho_i$")
    ax.set_ylabel(r"$\gamma$")
    ax.set_title("Growth-rate branch")
    ax.legend(frameon=False, fontsize=8.0)

    ax = axes[0, 1]
    ax.plot(ky, omega_ref, "o-", label="digitized reference", color="#264653")
    ax.plot(ky, omega_spectrax, "s--", label="SPECTRAX-GK tracked", color="#e76f51")
    ax.axhline(0.0, color="#444444", linewidth=0.8, alpha=0.5)
    ax.set_xlabel(r"$k_y \rho_i$")
    ax.set_ylabel(r"$\omega$")
    ax.set_title("Frequency branch")
    ax.legend(frameon=False, fontsize=8.0)

    ax = axes[1, 0]
    width = 0.017
    ax.bar(
        ky - width / 2,
        gamma_error,
        width=width,
        label=r"$|\Delta\gamma|$",
        color="#2a9d8f",
    )
    ax.bar(
        ky + width / 2,
        omega_error,
        width=width,
        label=r"$|\Delta\omega|$",
        color="#e9c46a",
    )
    ax.set_xlabel(r"$k_y \rho_i$")
    ax.set_ylabel("absolute error")
    ax.set_title("Pointwise mismatch")
    ax.legend(frameon=False, fontsize=8.0)

    ax = axes[1, 1]
    ax.axis("off")
    text = "\n".join(
        [
            "TEM lane status: OPEN",
            f"max |rel gamma| = {metrics['max_abs_rel_gamma']:.3g}",
            f"max |rel omega|, |omega_ref|>=0.2 = {metrics['max_abs_rel_omega_ref_ge_0p2']:.3g}",
            f"max |Delta gamma| = {metrics['max_abs_gamma_error']:.3g}",
            f"max |Delta omega| = {metrics['max_abs_omega_error']:.3g}",
            f"gamma sign mismatches = {metrics['gamma_sign_mismatch_count']}",
            f"omega sign mismatches = {metrics['omega_sign_mismatch_count']}",
            f"omega Spearman = {metrics['omega_spearman']:.3g}",
            "",
            textwrap.fill(
                "Reference provenance: provisional literature digitization, not a direct case dump. "
                "This blocks TEM/W7-X kinetic-electron validation claims without being a tuning target.",
                width=48,
            ),
        ]
    )
    ax.text(0.0, 1.0, text, transform=ax.transAxes, va="top", ha="left", fontsize=9.0)

    fig.suptitle("Tracked TEM branch parity audit", y=0.98)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_pdf)
    plt.close(fig)
    return {
        "png": str(out_png),
        "pdf": str(out_pdf),
        "json": str(out_json),
        "csv": str(out_csv),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--json-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_audit_payload(table=args.table, reference=args.reference)
    if args.json_only:
        print(json.dumps(_json_clean(payload), indent=2, sort_keys=True))
        return 0
    print(
        json.dumps(write_artifacts(payload, out_png=args.out), indent=2, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
