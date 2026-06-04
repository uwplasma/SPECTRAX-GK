#!/usr/bin/env python3
"""Build a full VMEC-JAX QA transport optimizer sweep comparison panel.

This tool consumes real optimizer outputs from a campaign directory, typically
copied from an ``office`` run.  It intentionally does not synthesize nonlinear
heat-flux traces: ``Q(t)`` is plotted only when long-window SPECTRAX-GK audit
CSV files are present beside a candidate.  Otherwise the panel labels the
nonlinear audit as pending, so reduced optimizer objectives cannot be mistaken
for saturated turbulent-transport evidence.
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
from matplotlib.colors import Normalize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spectraxgk.plotting import set_plot_style  # noqa: E402


DEFAULT_RUN_ROOT = ROOT / "tools_out" / "vmec_jax_qa_full_sweep_20260603"
DEFAULT_OUT = ROOT / "docs" / "_static" / "vmec_jax_qa_full_sweep_panel.png"

CASE_LABELS = {
    "qa_baseline_scipy": "QA baseline\nscipy",
    "qa_baseline_scalar_trust": "QA baseline\nscalar trust",
    "growth_scalar_trust": "growth\nscalar trust",
    "growth_lbfgs_adjoint": "growth\nL-BFGS adjoint",
    "quasilinear_scalar_trust": "QL flux\nscalar trust",
    "nonlinear_window_scalar_trust": "NL-window\nscalar trust",
}
PREFERRED_ORDER = (
    "qa_baseline_scipy",
    "qa_baseline_scalar_trust",
    "growth_scalar_trust",
    "growth_lbfgs_adjoint",
    "quasilinear_scalar_trust",
    "nonlinear_window_scalar_trust",
)
PREFERRED_SURFACE_IDS = (
    "qa_baseline_scipy",
    "growth_scalar_trust",
    "quasilinear_scalar_trust",
    "projected_guarded_ladder/transport_weight_0p0005",
)


def _optimize_png_if_possible(path: Path) -> None:
    """Keep tracked docs PNGs small without changing the plotted data."""

    if path.suffix.lower() != ".png" or not path.exists():
        return
    try:
        from PIL import Image
    except Exception:
        return
    try:
        with Image.open(path) as image:
            optimized = image.convert("P", palette=Image.Palette.ADAPTIVE, colors=256)
            optimized.save(path, optimize=True)
    except Exception:
        return


def _repo_relative(path: Path | str) -> str:
    raw = Path(path)
    try:
        return raw.resolve(strict=False).relative_to(ROOT.resolve(strict=False)).as_posix()
    except ValueError:
        return str(path)


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _finite_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(np.asarray(value))
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.floating, float)):
        item = float(value)
        return item if math.isfinite(item) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _case_sort_key(row: dict[str, Any]) -> tuple[int, str]:
    case_id = str(row["case_id"])
    try:
        rank = PREFERRED_ORDER.index(case_id)
    except ValueError:
        rank = len(PREFERRED_ORDER)
    return rank, case_id


def _campaign_root(run_root: Path) -> Path:
    if (run_root / "runs").is_dir():
        return run_root
    if run_root.name in {"runs", "runs_onepoint"}:
        return run_root.parent
    return run_root


def _runs_root(run_root: Path) -> Path:
    return run_root / "runs" if (run_root / "runs").is_dir() else run_root


def discover_run_dirs(run_root: Path) -> list[Path]:
    """Return optimizer run directories containing ``history.json``."""

    base = _runs_root(run_root)
    histories = set(base.glob("*/history.json"))
    histories.update(base.glob("*/*/history.json"))
    return sorted((path.parent for path in histories), key=lambda item: item.as_posix())


def _status_text(campaign_root: Path, case_id: str) -> str:
    status = campaign_root / "logs" / f"{case_id}.status"
    if not status.exists():
        status = campaign_root / "logs_onepoint" / f"{case_id}.status"
    if not status.exists():
        return "unknown"
    lines = [line.strip() for line in status.read_text(encoding="utf-8").splitlines() if line.strip()]
    return lines[-1] if lines else "unknown"


def _status_completed(status: str) -> bool:
    return " END " in f" {status} " and "rc=0" in status


def _objective_series(history: dict[str, Any]) -> list[float]:
    entries = history.get("history", ())
    values: list[float] = []
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            for key in ("objective", "cost", "objective_value"):
                if key in entry:
                    value = _finite_float(entry[key])
                    if math.isfinite(value):
                        values.append(value)
                    break
    if values:
        return values
    initial = _finite_float(history.get("objective_initial"))
    final = _finite_float(history.get("objective_final"))
    if math.isfinite(initial) and math.isfinite(final):
        return [initial, final]
    return []


def _load_iota_profile_from_wout(wout_path: Path) -> dict[str, Any] | None:
    if not wout_path.exists():
        return None
    try:
        import vmec_jax as vj  # type: ignore[import-not-found]

        wout = vj.load_wout(wout_path)
        profiles = vj.profiles_from_wout(wout)
        iotas = np.asarray(profiles["iotas"], dtype=float)
        iotaf = np.asarray(profiles["iotaf"], dtype=float)
        s_iotas = np.linspace(0.0, 1.0, int(iotas.size)) if iotas.size else np.asarray([])
        s_iotaf = np.linspace(0.0, 1.0, int(iotaf.size)) if iotaf.size else np.asarray([])
        return {
            "source": _repo_relative(wout_path),
            "s": s_iotas.tolist(),
            "iotas": iotas.tolist(),
            "s_iotaf": s_iotaf.tolist(),
            "iotaf": iotaf.tolist(),
            "mean_iotas": float(np.nanmean(iotas)) if iotas.size else None,
            "min_iotas_excluding_axis": float(np.nanmin(iotas[1:]))
            if iotas.size > 1
            else (float(np.nanmin(iotas)) if iotas.size else None),
            "edge_iota": float(iotas[-1]) if iotas.size else None,
            "nfp": int(np.asarray(getattr(wout, "nfp", 1))),
            "ns": int(np.asarray(getattr(wout, "ns", iotas.size))),
        }
    except Exception as exc:
        return {
            "source": _repo_relative(wout_path),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _load_gate(root: Path) -> dict[str, Any] | None:
    path = root / "solved_wout_gate.json"
    if not path.exists():
        return None
    return _read_json(path)


def _gate_blockers(gate: dict[str, Any] | None) -> list[str]:
    if gate is None:
        return ["missing_solved_wout_gate"]
    checks = gate.get("checks", {})
    if not isinstance(checks, dict):
        return ["malformed_solved_wout_gate"]
    blockers = [
        str(name)
        for name, check in checks.items()
        if isinstance(check, dict) and not bool(check.get("passed", False))
    ]
    if not bool(gate.get("passed", False)) and not blockers:
        blockers.append("gate_reported_failed")
    return blockers


def _load_q_trace(path: Path) -> dict[str, Any] | None:
    times: list[float] = []
    fluxes: list[float] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            t = _finite_float(row.get("t", row.get("time")))
            q = _finite_float(row.get("heat_flux", row.get("Q", row.get("q"))))
            if math.isfinite(t) and math.isfinite(q):
                times.append(t)
                fluxes.append(q)
    if not times:
        return None
    t_arr = np.asarray(times, dtype=float)
    q_arr = np.asarray(fluxes, dtype=float)
    tail = max(1, int(math.ceil(0.25 * q_arr.size)))
    return {
        "path": _repo_relative(path),
        "t": t_arr.tolist(),
        "heat_flux": q_arr.tolist(),
        "late_window_mean": float(np.nanmean(q_arr[-tail:])),
        "late_window_tmin": float(t_arr[-tail]),
        "late_window_tmax": float(t_arr[-1]),
    }


def _q_traces(root: Path) -> list[dict[str, Any]]:
    traces = []
    for path in sorted(root.rglob("*heat_flux_trace.csv")):
        loaded = _load_q_trace(path)
        if loaded is not None:
            traces.append(loaded)
    return traces


def _audit_case_token(case_id: str) -> str:
    return (
        "vmec_qa_full_sweep_"
        + case_id.replace("/", "_")
        .replace(".", "p")
        .replace("-", "m")
        .replace("+", "p")
    )


def _nonlinear_audit_command(wout_path: Path | None, case_id: str, *, run_completed: bool) -> str | None:
    if not run_completed or wout_path is None or not wout_path.exists():
        return None
    token = _audit_case_token(case_id)
    out_dir = ROOT / "tools_out" / "vmec_qa_full_sweep_nonlinear_audits" / token
    return (
        "python tools/write_optimized_equilibrium_transport_configs.py "
        f"--vmec-file {_repo_relative(wout_path)} "
        f"--case {token} "
        f"--out-dir {_repo_relative(out_dir)} "
        "--horizons 700 "
        "--grid n64:64:64:40:40 "
        "--window-tmin 350 --window-tmax 700"
    )


def _case_label(case_id: str) -> str:
    if case_id in CASE_LABELS:
        return CASE_LABELS[case_id]
    if case_id.startswith("projected_guarded_ladder/transport_weight_"):
        return "projected\nweight " + case_id.rsplit("_", maxsplit=1)[-1]
    if case_id.startswith("transport_weight_"):
        return "projected\n" + case_id.removeprefix("transport_weight_")
    return case_id.replace("_", "\n")


def _case_id(root: Path, runs_root: Path) -> str:
    try:
        return root.relative_to(runs_root).as_posix()
    except ValueError:
        return root.name


def _row_from_run(root: Path, *, campaign_root: Path, runs_root: Path) -> dict[str, Any]:
    case_id = _case_id(root, runs_root)
    history = _read_json(root / "history.json")
    setup_path = root / "setup_summary.json"
    setup = _read_json(setup_path) if setup_path.exists() else {}
    gate = _load_gate(root)
    wout_path = root / "wout_final.nc"
    iota_profile = _load_iota_profile_from_wout(wout_path)
    transport_kind = history.get("transport_metric_kind", setup.get("transport_kind"))
    status = _status_text(campaign_root, case_id.split("/")[-1])
    artifact_completed = (
        status == "unknown"
        and gate is not None
        and wout_path.exists()
        and math.isfinite(_finite_float(history.get("objective_final")))
    )
    run_completed = _status_completed(status) or artifact_completed
    return {
        "case_id": case_id,
        "label": _case_label(case_id),
        "root": _repo_relative(root),
        "status": status,
        "run_completed": run_completed,
        "setup": setup,
        "has_wout_final": wout_path.exists(),
        "wout_final": _repo_relative(wout_path) if wout_path.exists() else None,
        "history": {
            "objective_initial": _finite_float(history.get("objective_initial")),
            "objective_final": _finite_float(history.get("objective_final")),
            "aspect_initial": _finite_float(history.get("aspect_initial")),
            "aspect_final": _finite_float(history.get("aspect_final")),
            "iota_initial": _finite_float(history.get("iota_initial")),
            "iota_final": _finite_float(history.get("iota_final")),
            "qs_initial": _finite_float(history.get("qs_initial")),
            "qs_final": _finite_float(history.get("qs_final")),
            "transport_metric_final": _finite_float(
                history.get("transport_metric_final", history.get("transport_objective_final"))
            ),
            "transport_metric_kind": transport_kind,
            "success": history.get("success"),
            "message": history.get("message"),
            "nfev": history.get("nfev"),
            "total_wall_time_s": _finite_float(history.get("total_wall_time_s")),
        },
        "objective_history": _objective_series(history),
        "gate": gate,
        "gate_passed": None if gate is None else bool(gate.get("passed", False)),
        "gate_blockers": _gate_blockers(gate),
        "iota_profile": iota_profile,
        "q_traces": _q_traces(root),
        "recommended_nonlinear_audit_command": _nonlinear_audit_command(
            wout_path if wout_path.exists() else None,
            case_id,
            run_completed=run_completed,
        ),
    }


def build_payload(run_root: Path = DEFAULT_RUN_ROOT) -> dict[str, Any]:
    """Return a JSON-ready summary of a full optimizer sweep."""

    run_root = Path(run_root)
    campaign = _campaign_root(run_root)
    runs = _runs_root(run_root)
    rows = [
        _row_from_run(root, campaign_root=campaign, runs_root=runs)
        for root in discover_run_dirs(run_root)
    ]
    rows.sort(key=_case_sort_key)
    completed = [
        row
        for row in rows
        if bool(row["run_completed"]) and bool(row["has_wout_final"])
    ]
    with_q = [row for row in rows if row["q_traces"]]
    return {
        "kind": "vmec_jax_qa_full_max_mode5_sweep",
        "claim_scope": (
            "real VMEC-JAX max_mode=5 optimizer-output comparison; reduced "
            "growth/quasilinear/nonlinear-window objectives are optimizer "
            "diagnostics; production nonlinear Q claims require matched "
            "long post-transient SPECTRAX-GK audit traces"
        ),
        "run_root": _repo_relative(run_root),
        "campaign_root": _repo_relative(campaign),
        "runs_root": _repo_relative(runs),
        "cases": rows,
        "summary": {
            "n_cases": len(rows),
            "n_completed_wouts": len(completed),
            "n_cases_with_nonlinear_q_traces": len(with_q),
            "completed_case_ids": [row["case_id"] for row in completed],
            "cases_with_nonlinear_q_traces": [row["case_id"] for row in with_q],
            "nonlinear_transport_audit_status": (
                "available_for_some_candidates" if with_q else "pending_for_this_sweep"
            ),
        },
    }


def _selected_rows(rows: list[dict[str, Any]], *, max_count: int = 4) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    by_id = {str(row["case_id"]): row for row in rows}
    for case_id in PREFERRED_SURFACE_IDS:
        row = by_id.get(case_id)
        if row is not None and bool(row.get("run_completed")) and bool(row.get("has_wout_final")):
            selected.append(row)
    for row in rows:
        if len(selected) >= max_count:
            break
        if bool(row.get("run_completed")) and bool(row.get("has_wout_final")) and row not in selected:
            selected.append(row)
    return selected[:max_count]


def _surface_arrays(wout_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    try:
        import vmec_jax as vj  # type: ignore[import-not-found]

        wout = vj.load_wout(wout_path)
        ns = int(np.asarray(getattr(wout, "ns")))
        _theta, phi, r_grid, z_grid, b_grid = vj.vmecplot2_lcfs_3d_grid(
            wout,
            s_index=ns - 1,
            ntheta=58,
            nzeta=96,
        )
        x_grid = np.asarray(r_grid) * np.cos(np.asarray(phi)[None, :])
        y_grid = np.asarray(r_grid) * np.sin(np.asarray(phi)[None, :])
        return np.asarray(x_grid), np.asarray(y_grid), np.asarray(z_grid), np.asarray(b_grid)
    except Exception:
        return None


def _bmag_arrays(wout_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    try:
        import vmec_jax as vj  # type: ignore[import-not-found]

        wout = vj.load_wout(wout_path)
        ns = int(np.asarray(getattr(wout, "ns")))
        nfp = int(np.asarray(getattr(wout, "nfp", 1)))
        theta, zeta, b_grid = vj.vmecplot2_bmag_grid(
            wout,
            s_index=ns - 1,
            ntheta=96,
            nzeta=96,
            zeta_max=2.0 * np.pi / max(1, nfp),
        )
        return np.asarray(theta), np.asarray(zeta), np.asarray(b_grid)
    except Exception:
        return None


def _resolved_path(raw: str | None) -> Path | None:
    if raw is None:
        return None
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def _plot_summary_table(ax: plt.Axes, rows: list[dict[str, Any]]) -> None:
    ax.axis("off")
    lines = ["case                         A      iota      QS        transport       status"]
    for row in rows[:8]:
        hist = row["history"]
        if not row.get("run_completed"):
            status = "running"
        else:
            status = "pass" if row["gate_passed"] is True else ("fail" if row["gate_passed"] is False else "n/a")
        transport = hist["transport_metric_final"]
        transport_text = f"{transport:9.2e}" if math.isfinite(transport) else "    n/a  "
        label = str(row["label"]).replace("\n", " ")
        lines.append(
            f"{label[:26]:26s} "
            f"{hist['aspect_final']:6.3g} {hist['iota_final']:8.3g} "
            f"{hist['qs_final']:8.2e} {transport_text:>13s} {status:>8s}"
        )
    ax.text(
        0.01,
        0.98,
        "\n".join(lines),
        ha="left",
        va="top",
        family="monospace",
        fontsize=8.0,
    )
    ax.set_title("Completed full max-mode-5 optimizer outputs", loc="left")


def _plot_objective_history(ax: plt.Axes, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        series = np.asarray(row.get("objective_history", []), dtype=float)
        series = series[np.isfinite(series)]
        if series.size == 0:
            continue
        y0 = series[0] if series[0] != 0.0 else 1.0
        ax.semilogy(np.arange(series.size), np.maximum(series / abs(y0), 1.0e-16), lw=1.9, label=row["label"])
    ax.set_xlabel("optimizer callback")
    ax.set_ylabel(r"$||r||^2 / ||r_0||^2$")
    ax.set_title("Objective histories")
    ax.grid(alpha=0.25, which="both")
    if ax.lines:
        ax.legend(frameon=False, fontsize=7, ncols=2)


def _plot_metric_bars(ax: plt.Axes, rows: list[dict[str, Any]]) -> None:
    labels = [row["label"] for row in rows]
    x = np.arange(len(rows), dtype=float)
    aspect = np.asarray([row["history"]["aspect_final"] for row in rows], dtype=float)
    iota = np.asarray([row["history"]["iota_final"] for row in rows], dtype=float)
    qs = np.asarray([row["history"]["qs_final"] for row in rows], dtype=float)
    transport = np.asarray([row["history"]["transport_metric_final"] for row in rows], dtype=float)
    ax.bar(x - 0.27, np.nan_to_num(aspect, nan=0.0), width=0.18, label="aspect")
    ax.bar(x - 0.09, np.nan_to_num(iota, nan=0.0), width=0.18, label="mean iota")
    ax.bar(x + 0.09, np.nan_to_num(qs, nan=0.0), width=0.18, label="QS")
    finite_transport = np.isfinite(transport)
    if np.any(finite_transport):
        scaled = np.zeros_like(transport)
        max_t = float(np.nanmax(transport[finite_transport]))
        if max_t > 0.0:
            scaled[finite_transport] = transport[finite_transport] / max_t
        ax.bar(x + 0.27, scaled, width=0.18, label="transport / max")
    ax.axhline(0.41, color="#111827", lw=1.0, ls=":", alpha=0.6, label=r"$\iota=0.41$")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.set_title("Final scalar diagnostics")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=7, ncols=2)


def _plot_iota_profiles(ax: plt.Axes, rows: list[dict[str, Any]]) -> None:
    plotted = False
    for row in rows:
        profile = row.get("iota_profile")
        if not isinstance(profile, dict) or "iotas" not in profile:
            continue
        s = np.asarray(profile.get("s", []), dtype=float)
        iotas = np.asarray(profile.get("iotas", []), dtype=float)
        if s.size != iotas.size or s.size == 0:
            continue
        ax.plot(s, iotas, lw=2.0, label=row["label"])
        plotted = True
    ax.axhline(0.41, color="#111827", lw=1.2, ls=":", alpha=0.7)
    ax.set_xlabel("normalized toroidal flux")
    ax.set_ylabel(r"$\iota$")
    ax.set_title("Solved WOUT iota profiles")
    ax.grid(alpha=0.25)
    if plotted:
        ax.legend(frameon=False, fontsize=7, ncols=2)
    else:
        ax.text(0.5, 0.5, "No WOUT iota profiles available yet", ha="center", va="center")


def _plot_transport_bars(ax: plt.Axes, rows: list[dict[str, Any]]) -> None:
    transport_rows = [
        row
        for row in rows
        if math.isfinite(float(row["history"].get("transport_metric_final", math.nan)))
    ]
    if not transport_rows:
        ax.axis("off")
        ax.text(0.5, 0.5, "No reduced transport metric completed yet", ha="center", va="center")
        ax.set_title("Reduced optimizer transport metric")
        return
    values = np.asarray([row["history"]["transport_metric_final"] for row in transport_rows], dtype=float)
    x = np.arange(values.size)
    ax.bar(x, values, color="#b45309", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([row["label"] for row in transport_rows], rotation=35, ha="right", fontsize=7)
    ax.set_yscale("log" if np.nanmax(values) / max(np.nanmin(values), 1.0e-300) > 100.0 else "linear")
    ax.set_ylabel("lower is better")
    ax.set_title("Reduced growth/QL/NL-window objective")
    ax.grid(axis="y", alpha=0.25, which="both")


def _plot_q_traces(ax: plt.Axes, rows: list[dict[str, Any]]) -> None:
    plotted = False
    case_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    color_index = 0
    for row in rows:
        traces = row.get("q_traces", [])
        if not traces:
            continue
        color = case_colors[color_index % len(case_colors)] if case_colors else None
        color_index += 1
        late_tmins = []
        late_tmaxs = []
        for trace_index, trace in enumerate(traces):
            t = np.asarray(trace.get("t", []), dtype=float)
            q = np.asarray(trace.get("heat_flux", []), dtype=float)
            if t.size == 0 or t.size != q.size:
                continue
            label = f"{row['label']} (n={len(traces)})" if trace_index == 0 else None
            ax.plot(t, q, lw=1.15, alpha=0.65, color=color, label=label)
            tmin = _finite_float(trace.get("late_window_tmin"))
            tmax = _finite_float(trace.get("late_window_tmax"))
            if math.isfinite(tmin):
                late_tmins.append(tmin)
            if math.isfinite(tmax):
                late_tmaxs.append(tmax)
            plotted = True
        if late_tmins and late_tmaxs:
            ax.axvspan(
                float(min(late_tmins)),
                float(max(late_tmaxs)),
                color=color,
                alpha=0.06,
                lw=0,
            )
    ax.set_xlabel(r"$t v_{ti}/a$")
    ax.set_ylabel(r"$Q_i$")
    ax.set_title("Matched nonlinear heat-flux audits")
    ax.grid(alpha=0.25)
    if plotted:
        ax.legend(frameon=False, fontsize=7, ncols=2)
    else:
        ax.clear()
        ax.axis("off")
        ax.set_title("Matched nonlinear heat-flux audits")
        ax.text(
            0.5,
            0.5,
            "Pending for this sweep:\nrun long post-transient SPECTRAX-GK audits\n"
            "from each completed wout_final.nc before comparing Q(t).",
            ha="center",
            va="center",
            fontsize=9,
        )


def _plot_surface(ax: plt.Axes, row: dict[str, Any]) -> None:
    wout_path = _resolved_path(row.get("wout_final"))
    arrays = _surface_arrays(wout_path) if wout_path is not None else None
    if arrays is None:
        ax.text2D(0.08, 0.5, "surface\npending", transform=ax.transAxes)
        ax.set_axis_off()
        return
    x_grid, y_grid, z_grid, b_grid = arrays
    norm = Normalize(vmin=float(np.nanmin(b_grid)), vmax=float(np.nanmax(b_grid)))
    ax.plot_surface(
        x_grid,
        y_grid,
        z_grid,
        facecolors=plt.cm.jet(norm(b_grid)),
        linewidth=0,
        antialiased=False,
        shade=False,
        rstride=1,
        cstride=1,
    )
    ax.set_title(row["label"], fontsize=9)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect((1, 1, 0.35))
    ax.view_init(elev=20.0, azim=-48.0)


def _plot_bmag(ax: plt.Axes, row: dict[str, Any]) -> None:
    wout_path = _resolved_path(row.get("wout_final"))
    arrays = _bmag_arrays(wout_path) if wout_path is not None else None
    if arrays is None:
        ax.axis("off")
        ax.text(0.5, 0.5, "LCFS |B|\npending", ha="center", va="center")
        return
    theta, zeta, b_grid = arrays
    mesh = ax.contourf(zeta, theta, b_grid, levels=32, cmap="jet")
    ax.set_title(row["label"], fontsize=9)
    ax.set_xlabel(r"$\phi$ over one field period")
    ax.set_ylabel(r"$\theta$")
    plt.colorbar(mesh, ax=ax, fraction=0.046, pad=0.02, label=r"$|B|$")


def plot_payload(payload: dict[str, Any], out: Path) -> None:
    """Render the optimizer sweep panel."""

    rows = list(payload.get("cases", []))
    rows.sort(key=_case_sort_key)
    set_plot_style()
    fig = plt.figure(figsize=(18.0, 16.0), constrained_layout=True)
    gs = fig.add_gridspec(4, 4, height_ratios=[1.0, 1.05, 1.35, 1.15])

    _plot_summary_table(fig.add_subplot(gs[0, 0:2]), rows)
    _plot_objective_history(fig.add_subplot(gs[0, 2]), rows)
    _plot_metric_bars(fig.add_subplot(gs[0, 3]), rows)
    _plot_iota_profiles(fig.add_subplot(gs[1, 0:2]), rows)
    _plot_transport_bars(fig.add_subplot(gs[1, 2]), rows)
    _plot_q_traces(fig.add_subplot(gs[1, 3]), rows)

    selected = _selected_rows(rows, max_count=4)
    for col in range(4):
        if col < len(selected):
            surf_ax = fig.add_subplot(gs[2, col], projection="3d")
            _plot_surface(surf_ax, selected[col])
            _plot_bmag(fig.add_subplot(gs[3, col]), selected[col])
        else:
            surf_ax = fig.add_subplot(gs[2, col], projection="3d")
            surf_ax.set_axis_off()
            b_ax = fig.add_subplot(gs[3, col])
            b_ax.axis("off")

    status = payload["summary"]["nonlinear_transport_audit_status"]
    fig.suptitle(
        "VMEC-JAX QA max-mode-5 optimizer sweep with SPECTRAX-GK transport objectives\n"
        f"Q(t) status: {status}; reduced objectives are diagnostics until matched nonlinear audits exist",
        fontsize=15,
        fontweight="bold",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    _optimize_png_if_possible(out)


def _write_csv(payload: dict[str, Any], path: Path) -> None:
    rows = []
    for row in payload.get("cases", []):
        hist = row["history"]
        rows.append(
            {
                "case_id": row["case_id"],
                "root": row["root"],
                "has_wout_final": row["has_wout_final"],
                "gate_passed": row["gate_passed"],
                "gate_blockers": ";".join(str(item) for item in row["gate_blockers"]),
                "aspect_final": hist["aspect_final"],
                "iota_final": hist["iota_final"],
                "qs_final": hist["qs_final"],
                "transport_metric_kind": hist["transport_metric_kind"],
                "transport_metric_final": hist["transport_metric_final"],
                "q_trace_count": len(row["q_traces"]),
                "total_wall_time_s": hist["total_wall_time_s"],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = list(rows[0]) if rows else ["case_id"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--pdf", action="store_true", help="also write a PDF companion")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = build_payload(args.run_root)
    if not payload["cases"]:
        raise FileNotFoundError(f"no optimizer history.json files found under {args.run_root}")
    base = args.out.with_suffix("")
    base.with_suffix(".json").write_text(
        json.dumps(_json_ready(payload), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    _write_csv(payload, base.with_suffix(".csv"))
    plot_payload(payload, args.out)
    if args.pdf:
        plot_payload(payload, base.with_suffix(".pdf"))
    print(
        json.dumps(
            {
                "cases": payload["summary"]["n_cases"],
                "completed_wouts": payload["summary"]["n_completed_wouts"],
                "cases_with_q_traces": payload["summary"]["n_cases_with_nonlinear_q_traces"],
                "nonlinear_transport_audit_status": payload["summary"]["nonlinear_transport_audit_status"],
                "paths": {
                    "png": _repo_relative(args.out),
                    "json": _repo_relative(base.with_suffix(".json")),
                    "csv": _repo_relative(base.with_suffix(".csv")),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
