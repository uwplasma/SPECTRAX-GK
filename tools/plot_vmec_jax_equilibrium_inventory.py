#!/usr/bin/env python3
"""Inventory local vmec_jax VMEC equilibria for future validation lanes."""

from __future__ import annotations

import argparse
import hashlib
from itertools import cycle
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from netCDF4 import Dataset


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from spectraxgk.plotting import set_plot_style  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = Path(os.environ.get("VMEC_JAX_DATA_DIR", "/Users/rogeriojorge/local/vmec_jax/examples/data"))
DEFAULT_OUT = ROOT / "docs/_static/vmec_jax_equilibrium_inventory.png"


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar(ds: Dataset, *names: str, default: float | int | bool | None = None) -> float | int | bool | None:
    for name in names:
        if name not in ds.variables:
            continue
        arr = np.asarray(ds.variables[name][:])
        if arr.shape == ():
            value = arr.item()
        elif arr.size:
            value = arr.flat[0]
        else:
            continue
        if isinstance(value, bytes):
            return value.decode(errors="replace")
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        return value
    return default


def _profile_edge(ds: Dataset, *names: str) -> tuple[float | None, float | None]:
    for name in names:
        if name not in ds.variables:
            continue
        arr = np.asarray(ds.variables[name][:], dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            return float(finite[0]), float(finite[-1])
    return None, None


def _family(path: Path, nfp: int | None, ntor: int | None, betatotal: float | None) -> str:
    name = path.name.lower()
    if "tokamak" in name or int(ntor or 0) == 0:
        return "axisymmetric"
    if "qh" in name:
        return "quasi-helical"
    if "qa" in name:
        return "quasi-axisymmetric"
    if "qi" in name:
        return "quasi-isodynamic"
    if int(nfp or 1) > 1:
        return "stellarator"
    if betatotal is not None and abs(float(betatotal)) > 0.0:
        return "finite-beta"
    return "general"


def _positive_finite(value: Any) -> bool:
    if value is None:
        return False
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric) and numeric > 0.0


def _priority_score(row: dict[str, Any]) -> float:
    if not row.get("reference_scale_valid", False):
        return 0.0
    score = 0.0
    size = float(row["size_bytes"])
    if size <= 1_000_000:
        score += 2.0
    if row["family"] in {"quasi-helical", "quasi-axisymmetric", "quasi-isodynamic", "stellarator"}:
        score += 2.0
    if row["family"] == "axisymmetric":
        score += 1.0
    if row.get("betatotal") not in (None, 0.0):
        score += 0.5
    ns = int(row.get("ns") or 0)
    if 12 <= ns <= 80:
        score += 1.0
    if "reference" not in str(row["path"]):
        score += 0.25
    return score


def read_vmec_equilibrium_metadata(path: str | Path, *, root: str | Path | None = None) -> dict[str, Any]:
    """Read a compact metadata row from one VMEC ``wout`` NetCDF file."""

    target = Path(path)
    with Dataset(target) as ds:
        nfp = _scalar(ds, "nfp")
        ns = _scalar(ds, "ns")
        mpol = _scalar(ds, "mpol")
        ntor = _scalar(ds, "ntor")
        lasym = _scalar(ds, "lasym", "lasym__logical__", default=False)
        aminor = _scalar(ds, "Aminor_p")
        rmajor = _scalar(ds, "Rmajor_p")
        aspect = _scalar(ds, "aspect")
        volume = _scalar(ds, "volume_p")
        betatotal = _scalar(ds, "betatotal")
        iota_axis, iota_edge = _profile_edge(ds, "iotaf", "iota_full")
        pres_axis, pres_edge = _profile_edge(ds, "presf", "pres")

    if root is not None:
        try:
            rel = str(target.resolve().relative_to(Path(root).resolve()))
        except ValueError:
            rel = str(target)
    else:
        rel = str(target)
    row = {
        "path": rel,
        "name": target.name,
        "size_bytes": target.stat().st_size,
        "sha256": _sha256(target),
        "nfp": None if nfp is None else int(nfp),
        "ns": None if ns is None else int(ns),
        "mpol": None if mpol is None else int(mpol),
        "ntor": None if ntor is None else int(ntor),
        "lasym": bool(lasym),
        "aminor": None if aminor is None else float(aminor),
        "rmajor": None if rmajor is None else float(rmajor),
        "aspect": None if aspect is None else float(aspect),
        "volume": None if volume is None else float(volume),
        "betatotal": None if betatotal is None else float(betatotal),
        "iota_axis": iota_axis,
        "iota_edge": iota_edge,
        "pressure_axis": pres_axis,
        "pressure_edge": pres_edge,
    }
    row["family"] = _family(target, row["nfp"], row["ntor"], row["betatotal"])
    row["reference_scale_valid"] = all(
        _positive_finite(row.get(name)) for name in ("aminor", "rmajor", "aspect", "volume")
    )
    row["geometry_contract_status"] = (
        "ready_for_vmec_eik_smoke"
        if row["reference_scale_valid"]
        else "deferred_degenerate_vmec_reference_scale"
    )
    row["candidate_score"] = _priority_score(row)
    row["validation_role"] = (
        "external_vmec_fixture_for_linear_geometry_and_future_nonlinear_holdout; "
        "not an accepted quasilinear transport calibration point without a matched nonlinear window"
    )
    return row


def build_inventory(data_dir: str | Path = DEFAULT_DATA_DIR, *, max_files: int | None = None) -> dict[str, Any]:
    """Build a JSON-ready inventory for local vmec_jax example equilibria."""

    root = Path(data_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"VMEC data directory does not exist: {root}")
    paths = sorted(root.glob("wout*.nc"))
    if max_files is not None:
        paths = paths[: int(max_files)]
    rows = [read_vmec_equilibrium_metadata(path, root=root) for path in paths]
    rows = sorted(rows, key=lambda item: (-float(item["candidate_score"]), str(item["name"])))
    family_counts: dict[str, int] = {}
    for row in rows:
        family_counts[str(row["family"])] = family_counts.get(str(row["family"]), 0) + 1
    recommended = [
        row["name"]
        for row in rows
        if row["reference_scale_valid"] and float(row["candidate_score"]) >= 3.0 and int(row["size_bytes"]) <= 1_100_000
    ][:8]
    return {
        "kind": "vmec_jax_equilibrium_inventory",
        "claim_level": "equilibrium_selection_not_transport_validation",
        "source_root": str(root),
        "n_equilibria": len(rows),
        "family_counts": family_counts,
        "recommended_next_linear_portfolio": recommended,
        "rows": rows,
        "notes": (
            "These VMEC files are external fixtures from vmec_jax examples/data. "
            "They can broaden linear geometry and quasilinear-feature scans, but "
            "they must not be used as quasilinear transport calibration holdouts "
            "until matched nonlinear heat-flux windows are generated and gated."
        ),
    }


def write_inventory_figure(report: dict[str, Any], *, out: str | Path = DEFAULT_OUT) -> dict[str, str]:
    """Write a compact VMEC equilibrium inventory figure plus JSON/PDF companions."""

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(report["rows"])
    if not rows:
        raise ValueError("inventory has no equilibria to plot")
    set_plot_style()
    families = sorted({str(row["family"]) for row in rows})
    palette = (
        "#0f4c81",
        "#2a9d8f",
        "#b45309",
        "#6b7280",
        "#7c3aed",
        "#c2410c",
        "#0e7490",
        "#7f1d1d",
        "#3f6212",
        "#4c1d95",
    )
    color_map = {
        family: color
        for family, color in zip(families, cycle(palette), strict=False)
    }
    aspect = np.asarray([np.nan if row["aspect"] in (None, 0.0) else row["aspect"] for row in rows], dtype=float)
    iota_edge = np.asarray([np.nan if row["iota_edge"] is None else abs(float(row["iota_edge"])) for row in rows])
    sizes = np.asarray([row["size_bytes"] for row in rows], dtype=float)
    marker_size = 45.0 + 65.0 * np.clip(np.log10(np.maximum(sizes, 1.0)) - 4.0, 0.0, 2.0) / 2.0

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.3), constrained_layout=True)
    ax0, ax1 = axes
    for family in families:
        mask = np.asarray([row["family"] == family for row in rows], dtype=bool)
        ax0.scatter(
            aspect[mask],
            iota_edge[mask],
            s=marker_size[mask],
            color=color_map[family],
            alpha=0.82,
            edgecolor="white",
            linewidth=0.8,
            label=family,
        )
    for idx, row in enumerate(rows):
        if row["name"] in report["recommended_next_linear_portfolio"][:6]:
            ax0.annotate(row["name"].removeprefix("wout_").removesuffix(".nc"), (aspect[idx], iota_edge[idx]), fontsize=7)
    ax0.set_xlabel("VMEC aspect ratio")
    ax0.set_ylabel(r"$|\iota|$ at edge")
    ax0.set_title("Local vmec_jax equilibrium portfolio")
    ax0.grid(True, alpha=0.25)
    ax0.legend(frameon=True, fontsize=8)

    top = rows[: min(10, len(rows))]
    labels = [row["name"].removeprefix("wout_").removesuffix(".nc") for row in top]
    y = np.arange(len(top))
    scores = np.asarray([row["candidate_score"] for row in top], dtype=float)
    bars = ax1.barh(y, scores, color=[color_map[str(row["family"])] for row in top])
    for bar, row in zip(bars, top, strict=True):
        text = f"nfp={row['nfp']} ns={row['ns']} {row['size_bytes'] / 1e6:.2f} MB"
        if not row.get("reference_scale_valid", False):
            text += " deferred"
        ax1.text(bar.get_width() + 0.03, bar.get_y() + bar.get_height() / 2.0, text, va="center", fontsize=8)
    ax1.set_yticks(y, labels)
    ax1.invert_yaxis()
    ax1.set_xlabel("selection score")
    ax1.set_title("Best candidates for next linear/nonlinear holdouts")
    ax1.grid(True, axis="x", alpha=0.25)
    ax1.set_xlim(0.0, max(5.5, float(np.max(scores)) + 1.2))

    fig.suptitle("External VMEC equilibria for future validation, not transport calibration", fontsize=14)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(_json_clean(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"png": str(out_path), "pdf": str(pdf_path), "json": str(json_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Directory containing vmec_jax wout*.nc files.")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output PNG path.")
    parser.add_argument("--max-files", type=int, default=None, help="Optional limit for debugging.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_inventory(args.data_dir, max_files=args.max_files)
    paths = write_inventory_figure(report, out=args.out)
    print(f"saved {paths['png']}")
    print(f"saved {paths['pdf']}")
    print(f"saved {paths['json']}")
    print(
        "equilibria={n} recommended={items}".format(
            n=report["n_equilibria"],
            items=",".join(report["recommended_next_linear_portfolio"]) or "none",
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
