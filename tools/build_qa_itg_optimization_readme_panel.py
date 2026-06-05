#!/usr/bin/env python3
"""Build the compact README QA ITG optimization evidence panel.

The panel intentionally composes tracked real artifacts instead of rerunning the
multi-hour VMEC-JAX/SPECTRAX-GK campaign. It redraws scalar and transport axes
from JSON/CSV sidecars and embeds the solved-boundary VMEC-JAX geometry panel so
README claims stay tied to the same audited sources used in the docs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from spectraxgk.plotting import set_plot_style  # noqa: E402

DEFAULT_GEOMETRY_PNG = ROOT / "docs/_static/vmec_jax_qa_solved_boundary_boozer_panel.png"
DEFAULT_SWEEP_JSON = ROOT / "docs/_static/vmec_jax_qa_full_sweep_panel.json"
DEFAULT_LANDSCAPE_JSON = ROOT / "docs/_static/vmec_boundary_transport_landscape_rbc01.json"
DEFAULT_LANDSCAPE_CSV = ROOT / "docs/_static/vmec_boundary_transport_landscape_rbc01.csv"
DEFAULT_ADMISSION_JSON = ROOT / "docs/_static/vmec_boundary_transport_landscape_admission.json"
DEFAULT_MATCHED_JSON = ROOT / "docs/_static/vmec_jax_qa_projected_weight_0p001_matched_comparison.json"
DEFAULT_OUT = ROOT / "docs/_static/qa_itg_optimization_summary_panel.png"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _read_landscape_rows(path: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(
                {
                    "label": str(row["label"]),
                    "relative_fraction": float(row["relative_fraction"]),
                    "coefficient_value": float(row["coefficient_value"]),
                    "growth": float(row["growth"]),
                    "quasilinear_flux": float(row["quasilinear_flux"]),
                    "nonlinear_window_heat_flux": float(row["nonlinear_window_heat_flux"]),
                }
            )
    return rows


def _standard_errors_from_landscape(payload: dict[str, Any], kind: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in payload.get("rows", []):
        if not isinstance(row, dict):
            continue
        label = str(row.get("label"))
        report = row.get("reduced_metric_reports", {}).get(kind, {})
        if not isinstance(report, dict):
            continue
        stats = report.get("payload", {}).get("sample_statistics", {})
        if isinstance(stats, dict) and stats.get("weighted_standard_error") is not None:
            out[label] = float(stats["weighted_standard_error"])
    return out


def _normalize(values: np.ndarray, errors: np.ndarray, baseline_index: int) -> tuple[np.ndarray, np.ndarray]:
    scale = float(values[baseline_index]) if np.isfinite(values[baseline_index]) else float("nan")
    if not np.isfinite(scale) or abs(scale) <= 1.0e-30:
        scale = float(np.nanmax(np.abs(values))) if np.any(np.isfinite(values)) else 1.0
    scale = max(abs(scale), 1.0e-30)
    return values / scale, errors / scale


def _read_trace(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    return np.asarray(data["t"], dtype=float), np.asarray(data["heat_flux"], dtype=float)


def _case_by_id(payload: dict[str, Any], case_id: str) -> dict[str, Any]:
    for case in payload.get("cases", []):
        if isinstance(case, dict) and str(case.get("case_id")) == case_id:
            return case
    raise KeyError(case_id)


def _wout_path_from_case(sweep: dict[str, Any], case_id: str) -> Path:
    case = _case_by_id(sweep, case_id)
    raw = case.get("wout_final")
    if raw is None:
        raise FileNotFoundError(f"{case_id} has no wout_final entry")
    path = ROOT / str(raw)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _plot_wout_surface(ax: plt.Axes, wout_path: Path, title: str) -> None:
    import vmec_jax as vj  # type: ignore[import-not-found]

    wout = vj.load_wout(wout_path)
    _theta, phi, radius, height, bmag = vj.vmecplot2_lcfs_3d_grid(
        wout,
        s_index=-1,
        ntheta=84,
        nzeta=84,
    )
    x = radius * np.cos(phi)
    y = radius * np.sin(phi)
    norm = plt.Normalize(float(np.nanmin(bmag)), float(np.nanmax(bmag)))
    cmap = plt.get_cmap("jet")
    ax.plot_surface(
        x,
        y,
        height,
        facecolors=cmap(norm(bmag)),
        rstride=1,
        cstride=1,
        linewidth=0.0,
        antialiased=False,
        shade=False,
    )
    ax.set_title(title, fontsize=9.2, pad=2)
    ax.set_xlabel("x", labelpad=-8, fontsize=8)
    ax.set_ylabel("y", labelpad=-8, fontsize=8)
    ax.set_zlabel("z", labelpad=-8, fontsize=8)
    ax.tick_params(labelsize=7, pad=-2)
    ax.view_init(elev=22, azim=-52)
    span = max(np.ptp(x), np.ptp(y), np.ptp(height)) / 2.0
    cx, cy, cz = float(np.mean(x)), float(np.mean(y)), float(np.mean(height))
    ax.set_xlim(cx - span, cx + span)
    ax.set_ylim(cy - span, cy + span)
    ax.set_zlim(cz - span, cz + span)


def _plot_wout_boozer(ax: plt.Axes, wout_path: Path, title: str) -> None:
    import vmec_jax as vj  # type: ignore[import-not-found]

    wout = vj.load_wout(wout_path)
    theta, zeta, bmag = vj.vmecplot2_bmag_grid(
        wout,
        s_index=-1,
        ntheta=96,
        nzeta=96,
        zeta_max=2.0 * np.pi / float(wout.nfp),
    )
    zeta_grid, theta_grid = np.meshgrid(zeta, theta, indexing="ij")
    contour = ax.contourf(zeta_grid, theta_grid, bmag, levels=34, cmap="jet")
    ax.contour(zeta_grid, theta_grid, bmag, levels=14, colors="k", alpha=0.22, linewidths=0.35)
    ax.set_title(title, fontsize=9.2, pad=2)
    ax.set_xlabel(r"$\phi_B$", fontsize=8)
    ax.set_ylabel(r"$\theta_B$", fontsize=8)
    ax.tick_params(labelsize=7)
    cbar = plt.colorbar(contour, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label(r"$|B|$", fontsize=7)
    cbar.ax.tick_params(labelsize=6)


def _plot_geometry_fallback(ax: plt.Axes, image_path: Path) -> None:
    image = plt.imread(image_path)
    ax.imshow(image)
    ax.set_axis_off()
    ax.set_title("Solved QA geometry: LCFS |B|", loc="left", fontweight="bold")


def _plot_iota(ax: plt.Axes, sweep: dict[str, Any]) -> None:
    cases = [
        ("qa_baseline_scipy", "QA baseline", "#334155", "-"),
        ("growth_scalar_trust", "growth opt.", "#0f766e", "--"),
        ("quasilinear_scalar_trust", "QL opt.", "#1d4ed8", "-."),
        ("projected_guarded_ladder/transport_weight_0p001", "projected opt.", "#c2410c", "-"),
    ]
    for case_id, label, color, linestyle in cases:
        try:
            case = _case_by_id(sweep, case_id)
        except KeyError:
            continue
        prof = case.get("iota_profile", {})
        s = np.asarray(prof.get("s_iotaf", prof.get("s", [])), dtype=float)
        y = np.asarray(prof.get("iotaf", prof.get("iotas", [])), dtype=float)
        if s.size and y.size:
            n = min(s.size, y.size)
            ax.plot(s[:n], y[:n], color=color, linestyle=linestyle, lw=2.0, label=label)
    ax.axhline(0.41, color="0.25", ls=":", lw=1.4, label=r"$\iota=0.41$")
    ax.set_xlabel("normalized toroidal flux")
    ax.set_ylabel(r"rotational transform $\iota$")
    ax.set_title("Solved-WOUT iota gate", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8, ncols=1)


def _plot_landscape(ax: plt.Axes, rows: list[dict[str, float | str]], payload: dict[str, Any]) -> None:
    x = np.asarray([float(row["relative_fraction"]) for row in rows]) * 100.0
    baseline_index = int(np.argmin(np.abs(x)))
    styles = {
        "growth": ("linear growth", "#0f766e", "o"),
        "quasilinear_flux": ("quasilinear flux", "#1d4ed8", "s"),
        "nonlinear_window_heat_flux": ("NL-window", "#c2410c", "^"),
    }
    for kind, (label, color, marker) in styles.items():
        y = np.asarray([float(row[kind]) for row in rows], dtype=float)
        err_lookup = _standard_errors_from_landscape(payload, kind)
        err = np.asarray([err_lookup.get(str(row["label"]), np.nan) for row in rows], dtype=float)
        if not np.any(np.isfinite(err)):
            err = np.zeros_like(y)
        yn, en = _normalize(y, err, baseline_index)
        ax.errorbar(x, yn, yerr=en, marker=marker, color=color, lw=2.0, capsize=3, label=label)
    ax.axhline(1.0, color="0.5", lw=1.0, ls=":")
    ax.set_xlabel(r"relative $RBC(0,1)$ perturbation [%]")
    ax.set_ylabel("objective / baseline")
    ax.set_title("Noisy transport-objective landscape", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8, ncols=1)


def _plot_replicated_landscape(ax: plt.Axes, admission: dict[str, Any]) -> None:
    baseline = admission["baseline"]
    candidates = admission.get("candidates", [])
    labels = ["baseline"] + [str(row["label"]) for row in candidates]
    means = [float(baseline["ensemble_mean"])] + [float(row["ensemble_mean"]) for row in candidates]
    sems = [float(baseline["combined_sem"])] + [float(row["combined_sem"]) for row in candidates]
    colors = ["#334155"] + ["#0f766e" if bool(row.get("admitted", False)) else "#b91c1c" for row in candidates]
    xpos = np.arange(len(labels))
    ax.bar(xpos, means, yerr=sems, color=colors, edgecolor="0.15", linewidth=0.5, capsize=4)
    ax.set_xticks(xpos, labels, rotation=18, ha="right")
    ax.set_ylabel(r"late-window $\langle Q_i\rangle$")
    selected = admission.get("selected_candidate", {})
    rel = 100.0 * float(selected.get("relative_reduction", float("nan")))
    z = float(selected.get("uncertainty_z_score", float("nan")))
    ax.set_title("Replicated nonlinear landscape gate", loc="left", fontweight="bold")
    if math.isfinite(rel):
        ax.text(
            0.98,
            0.96,
            f"selected: {rel:.1f}% lower Q\nz = {z:.1f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.28", "fc": "white", "ec": "0.75", "alpha": 0.9},
        )


def _plot_q_traces(ax: plt.Axes, sweep: dict[str, Any], matched: dict[str, Any]) -> None:
    cases = [
        ("qa_baseline_scipy", "QA baseline", "#64748b"),
        ("projected_guarded_ladder/transport_weight_0p001", "projected opt.", "#0f766e"),
    ]
    for case_id, label, color in cases:
        case = _case_by_id(sweep, case_id)
        for idx, trace in enumerate(case.get("q_traces", [])):
            path = ROOT / trace["path"]
            if not path.exists():
                continue
            t, q = _read_trace(path)
            ax.plot(t, q, color=color, lw=1.0, alpha=0.35 if idx else 0.85, label=label if idx == 0 else None)
    ax.axvspan(350.0, 700.0, color="#e0f2fe", alpha=0.35, lw=0.0)
    stats = matched.get("statistics", {})
    rel = 100.0 * float(stats.get("relative_reduction", float("nan")))
    z = float(stats.get("uncertainty_z_score", float("nan")))
    ax.set_xlabel(r"$t v_{ti}/a$")
    ax.set_ylabel(r"$Q_i/Q_{gB}$")
    ax.set_title("Matched long-window nonlinear audit", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    if math.isfinite(rel):
        ax.text(
            0.98,
            0.96,
            f"{rel:.2f}% lower Q\nz = {z:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.28", "fc": "white", "ec": "0.75", "alpha": 0.9},
        )


def _optimize_png(path: Path) -> None:
    try:
        from PIL import Image
    except Exception:
        return
    with Image.open(path) as image:
        image.save(path, optimize=True)


def build_panel(
    *,
    geometry_png: Path,
    sweep_json: Path,
    landscape_json: Path,
    landscape_csv: Path,
    admission_json: Path,
    matched_json: Path,
    out: Path,
) -> dict[str, Any]:
    sweep = _read_json(sweep_json)
    landscape = _read_json(landscape_json)
    admission = _read_json(admission_json)
    matched = _read_json(matched_json)
    rows = _read_landscape_rows(landscape_csv)

    set_plot_style()
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.titlesize": 11,
            "axes.labelsize": 10.5,
            "legend.fontsize": 8.2,
            "xtick.labelsize": 9.2,
            "ytick.labelsize": 9.2,
        }
    )
    fig = plt.figure(figsize=(16.0, 9.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=(1.45, 1.0, 1.0), height_ratios=(1.0, 1.0))
    geom_gs = gs[:, 0].subgridspec(2, 2, wspace=0.02, hspace=0.16)
    ax_geom_title = fig.add_subplot(gs[:, 0], frame_on=False)
    ax_geom_title.set_axis_off()
    ax_geom_title.set_title("Solved QA geometry and LCFS |B|", loc="left", fontweight="bold", pad=4)
    ax_iota = fig.add_subplot(gs[0, 1])
    ax_q = fig.add_subplot(gs[0, 2])
    ax_land = fig.add_subplot(gs[1, 1])
    ax_repl = fig.add_subplot(gs[1, 2])

    try:
        baseline_wout = _wout_path_from_case(sweep, "qa_baseline_scipy")
        candidate_wout = _wout_path_from_case(sweep, "projected_guarded_ladder/transport_weight_0p001")
        ax_surf0 = fig.add_subplot(geom_gs[0, 0], projection="3d")
        ax_surf1 = fig.add_subplot(geom_gs[0, 1], projection="3d")
        ax_booz0 = fig.add_subplot(geom_gs[1, 0])
        ax_booz1 = fig.add_subplot(geom_gs[1, 1])
        _plot_wout_surface(ax_surf0, baseline_wout, "QA baseline")
        _plot_wout_surface(ax_surf1, candidate_wout, "projected candidate")
        _plot_wout_boozer(ax_booz0, baseline_wout, "baseline Boozer |B|")
        _plot_wout_boozer(ax_booz1, candidate_wout, "candidate Boozer |B|")
    except Exception:
        ax_fallback = fig.add_subplot(gs[:, 0])
        _plot_geometry_fallback(ax_fallback, geometry_png)

    _plot_iota(ax_iota, sweep)
    _plot_q_traces(ax_q, sweep, matched)
    _plot_landscape(ax_land, rows, landscape)
    _plot_replicated_landscape(ax_repl, admission)

    fig.suptitle(
        "Differentiable QA stellarator optimization with SPECTRAX-GK ITG transport objectives",
        fontsize=15,
        fontweight="bold",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    _optimize_png(out)

    sidecar = {
        "kind": "qa_itg_optimization_readme_panel",
        "claim_scope": (
            "real VMEC-JAX/SPECTRAX-GK tracked optimization evidence; reduced objectives are "
            "diagnostic/model-development quantities unless matched long-window nonlinear audits pass"
        ),
        "sources": {
            "geometry_png": _repo_relative(geometry_png),
            "sweep_json": _repo_relative(sweep_json),
            "landscape_json": _repo_relative(landscape_json),
            "landscape_csv": _repo_relative(landscape_csv),
            "landscape_admission_json": _repo_relative(admission_json),
            "matched_nonlinear_json": _repo_relative(matched_json),
        },
        "selected_landscape_candidate": admission.get("selected_candidate"),
        "matched_projected_candidate": matched.get("statistics"),
        "output": _repo_relative(out),
    }
    sidecar_path = out.with_suffix(".json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sidecar


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-png", type=Path, default=DEFAULT_GEOMETRY_PNG)
    parser.add_argument("--sweep-json", type=Path, default=DEFAULT_SWEEP_JSON)
    parser.add_argument("--landscape-json", type=Path, default=DEFAULT_LANDSCAPE_JSON)
    parser.add_argument("--landscape-csv", type=Path, default=DEFAULT_LANDSCAPE_CSV)
    parser.add_argument("--admission-json", type=Path, default=DEFAULT_ADMISSION_JSON)
    parser.add_argument("--matched-json", type=Path, default=DEFAULT_MATCHED_JSON)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sidecar = build_panel(
        geometry_png=args.geometry_png,
        sweep_json=args.sweep_json,
        landscape_json=args.landscape_json,
        landscape_csv=args.landscape_csv,
        admission_json=args.admission_json,
        matched_json=args.matched_json,
        out=args.out,
    )
    print(json.dumps({"output": sidecar["output"], "sources": sidecar["sources"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
