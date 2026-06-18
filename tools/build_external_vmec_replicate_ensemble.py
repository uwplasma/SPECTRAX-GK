#!/usr/bin/env python3
"""Build replicated nonlinear-window evidence from external-VMEC outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from spectraxgk.validation.quasilinear.window_config import (  # noqa: E402
    NonlinearWindowConvergenceConfig,
    NonlinearWindowEnsembleConfig,
    NonlinearWindowEnsembleManifestConfig,
)
from spectraxgk.validation.quasilinear.window_io import (  # noqa: E402
    nonlinear_window_convergence_from_summary,
)
from spectraxgk.validation.quasilinear.window_ensemble import (  # noqa: E402
    nonlinear_window_ensemble_artifact_manifest,
    nonlinear_window_ensemble_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outputs", nargs="+", type=Path, help="SPECTRAX-GK .out.nc files.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--tmin", type=float, required=True)
    parser.add_argument("--tmax", type=float, required=True)
    parser.add_argument("--baseline-seed", type=int, default=22)
    parser.add_argument("--baseline-dt", type=float, default=0.05)
    parser.add_argument("--heat-flux-variable", default="Diagnostics/HeatFlux_st")
    parser.add_argument("--artifact-prefix", help="Repo/documentation prefix used in JSON provenance.")
    parser.add_argument("--figure-title")
    parser.add_argument("--readiness-json", default="replicate_ensemble_readiness.json")
    parser.add_argument("--ensemble-json", default="replicate_ensemble_gate.json")
    parser.add_argument("--out-png", default="replicate_ensemble_gate.png")
    parser.add_argument("--min-samples", type=int, default=24)
    parser.add_argument("--min-blocks", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=256)
    parser.add_argument("--max-running-mean-rel-drift", type=float, default=0.20)
    parser.add_argument("--max-terminal-mean-rel-delta", type=float, default=0.15)
    parser.add_argument("--max-sem-rel", type=float, default=0.30)
    parser.add_argument("--max-mean-rel-spread", type=float, default=0.15)
    parser.add_argument("--max-combined-sem-rel", type=float, default=0.25)
    parser.add_argument(
        "--allow-failed-gates",
        action="store_true",
        help=(
            "Write all readiness/ensemble artifacts and return success even when "
            "the physics gates fail. Use only for diagnostic landscapes that plot "
            "failed points explicitly; promotion/release gates should not use this."
        ),
    )
    return parser


def _base_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".out.nc"):
        return name[: -len(".out.nc")]
    return path.stem


def _float_from_label(raw: str) -> float:
    return float(raw.replace("p", ".").replace("m", "-"))


def _variant_from_path(
    path: Path, *, baseline_seed: int, baseline_dt: float
) -> dict[str, Any]:
    stem = _base_stem(path)
    tokens = stem.split("_")
    seed_tokens = [
        (index, match)
        for index, token in enumerate(tokens)
        if (match := re.fullmatch(r"seed([0-9]+)", token)) is not None
    ]
    dt_tokens = [
        (index, match)
        for index, token in enumerate(tokens)
        if (match := re.fullmatch(r"dt([0-9]+(?:p[0-9]+)?)", token)) is not None
    ]

    # Case slugs may contain protocol labels such as ``repair_dt002``.  Treat
    # only suffix-style ``seedNN``/``dtNN`` tokens as replicate variants.
    seed_item = seed_tokens[-1] if seed_tokens else None
    dt_item = None
    if seed_item is not None:
        seed_index = seed_item[0]
        dt_after_seed = [item for item in dt_tokens if item[0] > seed_index]
        dt_item = dt_after_seed[-1] if dt_after_seed else None
    elif dt_tokens:
        dt_item = dt_tokens[-1]

    if seed_item is not None and dt_item is not None:
        seed = int(seed_item[1].group(1))
        dt = _float_from_label(dt_item[1].group(1))
        return {
            "variant_axis": "seed_timestep",
            "variant_label": f"seed{seed}_dt{dt_item[1].group(1)}",
            "seed": seed,
            "dt": dt,
            "variant": {"seed": seed, "timestep": dt},
        }
    if seed_item is not None:
        seed = int(seed_item[1].group(1))
        return {
            "variant_axis": "seed",
            "variant_label": f"seed{seed}",
            "seed": seed,
            "dt": float(baseline_dt),
            "variant": {"seed": seed, "timestep": float(baseline_dt)},
        }
    if dt_item is not None:
        dt = _float_from_label(dt_item[1].group(1))
        return {
            "variant_axis": "timestep",
            "variant_label": f"dt{dt_item[1].group(1)}",
            "seed": int(baseline_seed),
            "dt": dt,
            "variant": {"seed": int(baseline_seed), "timestep": dt},
        }
    raise ValueError(
        f"{path} does not contain a seedNN or dtNN variant label; "
        "use files generated by write_external_vmec_holdout_configs.py"
    )


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _artifact_path(path: Path, *, out_dir: Path, artifact_prefix: str | None) -> str:
    if artifact_prefix:
        rel = path.resolve().relative_to(out_dir.resolve())
        return f"{artifact_prefix.rstrip('/')}/{rel.as_posix()}"
    return _repo_relative(path)


def _output_artifact(path: Path, *, out_dir: Path, artifact_prefix: str | None) -> str:
    if path.resolve().is_relative_to(out_dir.resolve()):
        return _artifact_path(path, out_dir=out_dir, artifact_prefix=artifact_prefix)
    return path.name if artifact_prefix else path.as_posix()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _finite_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _format_gate_metric(value: Any) -> str:
    out = _finite_or_nan(value)
    return "n/a" if not math.isfinite(out) else f"{out:.3f}"


def _netcdf_variable(root: Any, variable_path: str) -> np.ndarray:
    group = root
    parts = variable_path.split("/")
    for part in parts[:-1]:
        group = group.groups[part]
    return np.asarray(group.variables[parts[-1]][:], dtype=float)


def _read_heat_flux_trace(path: Path, *, heat_flux_variable: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        import netCDF4
    except Exception as exc:  # pragma: no cover - dependency is in pyproject
        raise RuntimeError("netCDF4 is required to read SPECTRAX-GK output files") from exc

    with netCDF4.Dataset(path) as root:
        time = _netcdf_variable(root, "Grids/time").reshape(-1)
        heat = _netcdf_variable(root, heat_flux_variable)
    if heat.ndim == 1:
        heat_total = heat
    elif heat.shape[0] == time.size:
        heat_total = heat.reshape((time.size, -1)).sum(axis=1)
    else:
        raise ValueError(f"{path} heat-flux variable has incompatible shape {heat.shape}")
    return np.asarray(time, dtype=float), np.asarray(heat_total, dtype=float)


def _write_trace(path: Path, time: np.ndarray, heat_flux: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("t,heat_flux\n")
        for t, q in zip(time, heat_flux):
            fh.write(f"{float(t):.12g},{float(q):.12g}\n")


def _normalize_report_paths(
    report: dict[str, Any],
    *,
    source_artifact: str,
    summary_artifact: str,
) -> dict[str, Any]:
    out = json.loads(json.dumps(report))
    out.setdefault("provenance", {})["source_artifact"] = source_artifact
    out["provenance"]["summary_artifact"] = summary_artifact
    if isinstance(out.get("gate_report"), dict):
        out["gate_report"]["source"] = source_artifact
    return out


def _record_from_report(
    *,
    case: str,
    summary_artifact: str,
    source_artifact: str,
    convergence_report_artifact: str,
    variant: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, Any]:
    return {
        "case": case,
        "summary_artifact": summary_artifact,
        "source_artifact": source_artifact,
        "convergence_report_artifact": convergence_report_artifact,
        "variant": variant["variant"],
        "report": report,
    }


def _write_png(
    *,
    traces: list[dict[str, Any]],
    ensemble: dict[str, Any],
    tmin: float,
    tmax: float,
    title: str,
    out_png: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spectraxgk.artifacts.plotting import set_plot_style

    set_plot_style()
    colors = ["#11756f", "#b8612a", "#4569b2", "#7c3aed", "#b91c1c"]
    fig, (ax0, ax1) = plt.subplots(
        1,
        2,
        figsize=(10.5, 4.25),
        gridspec_kw={"width_ratios": [1.35, 1.0]},
    )
    rows = list(ensemble["rows"])
    labels: list[str] = []
    for idx, item in enumerate(traces):
        variant = item["variant"]
        label = f"seed {variant['seed']}, dt={variant['timestep']:.6g}"
        labels.append(f"seed {variant['seed']}\ndt={variant['timestep']:.6g}")
        ax0.plot(item["time"], item["heat_flux"], lw=1.65, color=colors[idx % len(colors)], label=label)
    ax0.axvspan(float(tmin), float(tmax), color="#d9a441", alpha=0.18, label="accepted window")
    ax0.set_xlabel(r"$t\,v_{ti}/a$")
    ax0.set_ylabel(r"$Q_i/Q_{gB}$")
    ax0.set_title("Nonlinear heat-flux traces")
    ax0.grid(alpha=0.25)
    ax0.legend(frameon=False, loc="upper left")

    means = np.asarray([_finite_or_nan(row.get("late_mean")) for row in rows], dtype=float)
    sems = np.asarray([_finite_or_nan(row.get("sem")) for row in rows], dtype=float)
    x = np.arange(len(means))
    finite_means = np.isfinite(means)
    if np.any(finite_means):
        ax1.bar(
            x,
            means,
            yerr=np.where(np.isfinite(sems), sems, 0.0),
            capsize=4,
            color=[colors[idx % len(colors)] for idx in range(len(means))],
            alpha=0.88,
            edgecolor="0.2",
            linewidth=0.5,
        )
    else:
        ax1.text(
            0.5,
            0.55,
            "No finite samples in the\nrequested late-time window",
            transform=ax1.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="#7f1d1d",
            bbox={"boxstyle": "round,pad=0.35", "fc": "#fff7ed", "ec": "#fed7aa"},
        )
    ensemble_mean = _finite_or_nan(ensemble["statistics"].get("ensemble_mean"))
    if math.isfinite(ensemble_mean):
        ax1.axhline(
            ensemble_mean,
            ls="--",
            lw=1.3,
            color="0.15",
        )
    ax1.set_xticks(x, labels)
    ax1.set_ylabel(r"late-window $\langle Q_i\rangle/Q_{gB}$")
    mean_label = "n/a" if not math.isfinite(ensemble_mean) else f"{ensemble_mean:.2f}"
    ax1.set_title(f"Seed/timestep robustness\nmean = {mean_label}")
    ax1.grid(axis="y", alpha=0.25)

    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.22, top=0.80, wspace=0.16)
    fig.suptitle(title, fontsize=12.5, y=0.96)
    stats = ensemble["statistics"]
    caption = (
        f"Gate {'passed' if ensemble['passed'] else 'failed'}: "
        f"mean relative spread = {_format_gate_metric(stats.get('mean_rel_spread'))} "
        f"(limit {ensemble['config']['max_mean_rel_spread']}), "
        f"combined SEM/mean = {_format_gate_metric(stats.get('combined_sem_rel'))} "
        f"(limit {ensemble['config']['max_combined_sem_rel']})."
    )
    fig.text(0.5, 0.055, caption, ha="center", va="center", fontsize=8.8, color="0.25")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", dpi=220)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = out_dir / "nonlinear_window_convergence_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    cfg = NonlinearWindowConvergenceConfig(
        tmin=float(args.tmin),
        tmax=float(args.tmax),
        min_samples=int(args.min_samples),
        min_blocks=int(args.min_blocks),
        bootstrap_samples=int(args.bootstrap_samples),
        max_running_mean_rel_drift=float(args.max_running_mean_rel_drift),
        max_terminal_mean_rel_delta=float(args.max_terminal_mean_rel_delta),
        max_sem_rel=float(args.max_sem_rel),
    )
    records: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for output in args.outputs:
        variant = _variant_from_path(
            output,
            baseline_seed=int(args.baseline_seed),
            baseline_dt=float(args.baseline_dt),
        )
        stem = _base_stem(output)
        trace_path = out_dir / f"{stem}_heat_flux_trace.csv"
        summary_path = out_dir / f"{stem}_transport_window.json"
        report_path = reports_dir / f"{stem}_transport_window.convergence.json"
        time, heat_flux = _read_heat_flux_trace(output, heat_flux_variable=str(args.heat_flux_variable))
        _write_trace(trace_path, time, heat_flux)
        summary_artifact = _artifact_path(summary_path, out_dir=out_dir, artifact_prefix=args.artifact_prefix)
        trace_artifact = _artifact_path(trace_path, out_dir=out_dir, artifact_prefix=args.artifact_prefix)
        output_artifact = _output_artifact(
            output,
            out_dir=out_dir,
            artifact_prefix=args.artifact_prefix,
        )
        summary = {
            "kind": "nonlinear_window_summary",
            "case": str(args.case),
            "spectrax": trace_path.name,
            "nonlinear_artifact": output_artifact,
            "tmin": float(args.tmin),
            "tmax": float(args.tmax),
            "promotion_gate": {
                "passed": True,
                "source": "replicated nonlinear-window extraction; convergence gate is stored separately",
            },
            "variant_axis": variant["variant_axis"],
            "variant_label": variant["variant_label"],
            "seed": variant["seed"],
            "dt": variant["dt"],
            "variant": variant["variant"],
        }
        _write_json(summary_path, summary)
        report = nonlinear_window_convergence_from_summary(
            summary_path,
            case=str(args.case),
            config=cfg,
        )
        report = _normalize_report_paths(
            report,
            source_artifact=trace_artifact,
            summary_artifact=summary_artifact,
        )
        _write_json(report_path, report)
        report_artifact = _artifact_path(report_path, out_dir=out_dir, artifact_prefix=args.artifact_prefix)
        records.append(
            _record_from_report(
                case=str(args.case),
                summary_artifact=summary_artifact,
                source_artifact=trace_artifact,
                convergence_report_artifact=report_artifact,
                variant=variant,
                report=report,
            )
        )
        traces.append({"time": time, "heat_flux": heat_flux, "variant": variant["variant"]})

    readiness = nonlinear_window_ensemble_artifact_manifest(
        records,
        case=f"{args.case}_readiness",
        config=NonlinearWindowEnsembleManifestConfig(),
    )
    readiness_path = out_dir / str(args.readiness_json)
    _write_json(readiness_path, readiness)
    ensemble = nonlinear_window_ensemble_report(
        [record["report"] for record in records],
        case=f"{args.case}_ensemble_gate",
        comparison=f"{args.case}_seed_timestep_replicates",
        config=NonlinearWindowEnsembleConfig(
            max_mean_rel_spread=float(args.max_mean_rel_spread),
            max_combined_sem_rel=float(args.max_combined_sem_rel),
        ),
    )
    ensemble_path = out_dir / str(args.ensemble_json)
    _write_json(ensemble_path, ensemble)
    png_path = out_dir / str(args.out_png)
    _write_png(
        traces=traces,
        ensemble=ensemble,
        tmin=float(args.tmin),
        tmax=float(args.tmax),
        title=str(args.figure_title or f"{args.case} nonlinear replicate gate, t=[{args.tmin:g},{args.tmax:g}]"),
        out_png=png_path,
    )
    payload = {
        "readiness_json": _artifact_path(readiness_path, out_dir=out_dir, artifact_prefix=args.artifact_prefix),
        "ensemble_json": _artifact_path(ensemble_path, out_dir=out_dir, artifact_prefix=args.artifact_prefix),
        "png": _artifact_path(png_path, out_dir=out_dir, artifact_prefix=args.artifact_prefix),
        "allow_failed_gates": bool(args.allow_failed_gates),
        "readiness_passed": bool(readiness["passed"]),
        "ensemble_passed": bool(ensemble["passed"]),
        "statistics": ensemble["statistics"],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    passed = bool(readiness["passed"]) and bool(ensemble["passed"])
    return 0 if passed or bool(args.allow_failed_gates) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
