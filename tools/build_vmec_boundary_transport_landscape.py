#!/usr/bin/env python3
"""Scan one VMEC boundary coefficient against SPECTRAX-GK transport objectives.

The first nonlinear turbulence-optimization studies in stellarators show that
late-time nonlinear heat-flux objectives can be noisy in boundary-parameter
space. This tool makes that diagnostic explicit for SPECTRAX-GK: perturb one
VMEC input coefficient, evaluate deterministic reduced linear/quasilinear
objectives, and optionally overlay replicated nonlinear heat-flux ensemble
points with error bars when those expensive simulations have finished.

The deterministic reduced curves are not nonlinear turbulent-flux claims. The
nonlinear points are accepted only from supplied ensemble JSON files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectraxgk.plotting import set_plot_style  # noqa: E402
from tools.write_vmec_boundary_perturbation_inputs import (  # noqa: E402
    CoefficientSpec,
    _coefficient_value,
    _parse_coefficient_spec,
    _patch_coefficient,
)


DEFAULT_BASELINE_INPUT = ROOT / "tools_out/latest_vmec_stack/authoritative_qa_baseline/input.final"
DEFAULT_OUT_DIR = ROOT / "tools_out/vmec_boundary_transport_landscape"
DEFAULT_DOCS_PREFIX = ROOT / "docs/_static/vmec_boundary_transport_landscape_rbc01"
DEFAULT_FRACTIONS = (-0.06, -0.03, 0.0, 0.03, 0.06)
DEFAULT_KINDS = ("growth", "quasilinear_flux", "nonlinear_window_heat_flux")
DEFAULT_SURFACES = "0.64"
DEFAULT_ALPHAS = "0.0"
DEFAULT_KY_VALUES = "0.3"


def _repo_relative(path: Path | str) -> str:
    raw = Path(path)
    try:
        return raw.resolve().relative_to(ROOT.resolve()).as_posix()
    except (OSError, ValueError):
        return str(path)


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, Path):
        return _repo_relative(value)
    if isinstance(value, np.ndarray):
        return _json_clean(value.tolist())
    if isinstance(value, np.generic):
        return _json_clean(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _parse_float_list(raw: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    if not all(math.isfinite(value) for value in values):
        raise argparse.ArgumentTypeError("all values must be finite")
    return values


def _label_float(value: float) -> str:
    if abs(value) < 5.0e-15:
        return "0"
    return f"{float(value):+.6g}".replace("+", "p").replace("-", "m").replace(".", "p")


def _coefficient_slug(spec: CoefficientSpec) -> str:
    return (
        f"{spec.family}_{spec.m}_{spec.n}"
        .replace("-", "m")
        .replace("+", "")
        .lower()
    )


def _write_scan_inputs(
    *,
    baseline_input: Path,
    coefficient: CoefficientSpec,
    fractions: tuple[float, ...],
    out_dir: Path,
) -> tuple[float, list[dict[str, Any]]]:
    text = baseline_input.read_text(encoding="utf-8")
    base_value = _coefficient_value(text, coefficient)
    if abs(base_value) <= 0.0:
        raise ValueError("relative landscape scan requires a nonzero baseline coefficient")
    rows: list[dict[str, Any]] = []
    input_dir = out_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    seen: set[float] = set()
    for fraction in fractions:
        fraction_f = float(fraction)
        value = float(base_value * (1.0 + fraction_f))
        rounded = round(value, 15)
        if rounded in seen:
            continue
        seen.add(rounded)
        label = _label_float(fraction_f)
        input_path = input_dir / f"input.{_coefficient_slug(coefficient)}_{label}"
        input_path.write_text(_patch_coefficient(text, coefficient, value), encoding="utf-8")
        rows.append(
            {
                "label": label,
                "relative_fraction": fraction_f,
                "coefficient_value": value,
                "input_path": input_path,
                "expected_wout": input_dir / f"wout_{_coefficient_slug(coefficient)}_{label}.nc",
            }
        )
    return float(base_value), rows


def _run_reduced_metric(
    *,
    row: dict[str, Any],
    kind: str,
    args: argparse.Namespace,
    metric_dir: Path,
) -> dict[str, Any]:
    out_json = metric_dir / str(row["label"]) / f"{kind}.json"
    cmd = [
        sys.executable,
        str(ROOT / "tools/evaluate_vmec_jax_spectrax_transport_metric.py"),
        "--input",
        str(row["input_path"]),
        "--out-json",
        str(out_json),
        "--outdir",
        str(metric_dir / str(row["label"]) / f"{kind}_vmec"),
        "--transport-kind",
        str(kind),
        "--surfaces",
        str(args.surfaces),
        "--alphas",
        str(args.alphas),
        "--ky-values",
        str(args.ky_values),
        "--ntheta",
        str(int(args.ntheta)),
        "--mboz",
        str(int(args.mboz)),
        "--nboz",
        str(int(args.nboz)),
        "--n-laguerre",
        str(int(args.n_laguerre)),
        "--n-hermite",
        str(int(args.n_hermite)),
        "--inner-max-iter",
        str(int(args.inner_max_iter)),
        "--trial-max-iter",
        str(int(args.trial_max_iter)),
        "--inner-ftol",
        f"{float(args.inner_ftol):.16g}",
        "--trial-ftol",
        f"{float(args.trial_ftol):.16g}",
        "--spectrax-objective-transform",
        str(args.spectrax_objective_transform),
        "--spectrax-objective-scale",
        f"{float(args.spectrax_objective_scale):.16g}",
    ]
    if int(args.surface_chunk_size) > 0:
        cmd.extend(["--surface-chunk-size", str(int(args.surface_chunk_size))])
    if args.solver_device is not None:
        cmd.extend(["--solver-device", str(args.solver_device)])
    try:
        result = subprocess.run(
            cmd,
            cwd=ROOT,
            timeout=float(args.metric_timeout),
            check=False,
            text=True,
            capture_output=True,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "kind": kind,
            "metric_json": out_json,
            "command": cmd,
            "returncode": 124,
            "error": f"timed out after {float(args.metric_timeout):.1f}s",
            "stdout_tail": (exc.stdout or "")[-1200:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-1200:] if isinstance(exc.stderr, str) else "",
            "value": None,
        }
    payload: dict[str, Any] | None = None
    value: float | None = None
    if result.returncode == 0 and out_json.exists():
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        value = float(payload["transport_objective_final"])
    return {
        "kind": kind,
        "metric_json": out_json,
        "command": cmd,
        "returncode": int(result.returncode),
        "stdout_tail": result.stdout[-1200:],
        "stderr_tail": result.stderr[-1200:],
        "value": value,
        "payload": payload,
    }


def _load_nonlinear_ensemble(raw: str) -> dict[str, Any]:
    try:
        value_raw, path_raw = raw.split(":", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--nonlinear-ensemble entries must have form coefficient_value:path"
        ) from exc
    value = float(value_raw)
    path = Path(path_raw)
    payload = json.loads(path.read_text(encoding="utf-8"))
    stats = payload.get("statistics", {})
    if not isinstance(stats, dict):
        raise argparse.ArgumentTypeError(f"{path} has no statistics object")
    mean = float(stats.get("ensemble_mean"))
    sem = float(stats.get("combined_sem", stats.get("sample_sem", 0.0)))
    return {
        "coefficient_value": value,
        "ensemble_path": path,
        "mean": mean,
        "sem": max(0.0, sem),
        "passed": bool(payload.get("passed", False)),
        "case": payload.get("case", path.stem),
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "relative_fraction",
        "coefficient_value",
        "input_path",
        "growth",
        "quasilinear_flux",
        "nonlinear_window_heat_flux",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            metrics = row.get("reduced_metrics", {})
            writer.writerow(
                {
                    "label": row["label"],
                    "relative_fraction": row["relative_fraction"],
                    "coefficient_value": row["coefficient_value"],
                    "input_path": _repo_relative(row["input_path"]),
                    "growth": metrics.get("growth"),
                    "quasilinear_flux": metrics.get("quasilinear_flux"),
                    "nonlinear_window_heat_flux": metrics.get("nonlinear_window_heat_flux"),
                }
            )


def _normalize(values: np.ndarray, *, baseline_index: int) -> np.ndarray:
    finite = np.isfinite(values)
    if not np.any(finite):
        return values
    baseline_index = max(0, min(int(baseline_index), int(values.size) - 1))
    scale = float(values[baseline_index]) if np.isfinite(values[baseline_index]) else float("nan")
    if abs(scale) < 1.0e-14:
        scale = float(np.nanmax(np.abs(values[finite])))
    if abs(scale) <= 0.0 or not np.isfinite(scale):
        scale = 1.0
    return values / scale


def _write_plot(
    *,
    report: dict[str, Any],
    rows: list[dict[str, Any]],
    nonlinear_points: list[dict[str, Any]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    set_plot_style()
    x = np.asarray([float(row["relative_fraction"]) for row in rows], dtype=float) * 100.0
    baseline_index = int(np.argmin(np.abs(x)))
    metrics = {kind: np.asarray([row.get("reduced_metrics", {}).get(kind, np.nan) for row in rows], dtype=float) for kind in DEFAULT_KINDS}

    fig, axes = plt.subplots(3, 1, figsize=(7.8, 8.2), sharex=True, constrained_layout=True)
    colors = {
        "growth": "#0f766e",
        "quasilinear_flux": "#1d4ed8",
        "nonlinear_window_heat_flux": "#c2410c",
    }
    labels = {
        "growth": "linear growth objective",
        "quasilinear_flux": "quasilinear-flux objective",
        "nonlinear_window_heat_flux": "reduced nonlinear-window objective",
    }
    ax0 = axes[0]
    for kind in ("growth", "quasilinear_flux"):
        y = metrics[kind]
        if np.any(np.isfinite(y)):
            ax0.plot(x, _normalize(y, baseline_index=baseline_index), marker="o", lw=1.8, color=colors[kind], label=labels[kind])
    ax0.set_ylabel("normalized reduced objective")
    ax0.set_title(f"{report['coefficient']} transport-objective landscape")
    ax0.legend(frameon=False, fontsize=8)
    ax0.grid(True, alpha=0.25)

    ax1 = axes[1]
    y = metrics["nonlinear_window_heat_flux"]
    if np.any(np.isfinite(y)):
        ax1.plot(x, _normalize(y, baseline_index=baseline_index), marker="o", lw=1.8, color=colors["nonlinear_window_heat_flux"], label=labels["nonlinear_window_heat_flux"])
    ax1.set_ylabel("normalized reduced objective")
    ax1.legend(frameon=False, fontsize=8)
    ax1.grid(True, alpha=0.25)

    ax2 = axes[2]
    if nonlinear_points:
        base = float(report["baseline_coefficient_value"])
        xs = np.asarray([(float(point["coefficient_value"]) / base - 1.0) * 100.0 for point in nonlinear_points])
        means = np.asarray([float(point["mean"]) for point in nonlinear_points])
        sem = np.asarray([float(point["sem"]) for point in nonlinear_points])
        colors_nl = ["#0f766e" if bool(point.get("passed", False)) else "#b91c1c" for point in nonlinear_points]
        ax2.errorbar(xs, means, yerr=sem, fmt="none", ecolor="0.25", elinewidth=1.1, capsize=4)
        ax2.scatter(xs, means, c=colors_nl, s=48, edgecolors="0.15", linewidths=0.5, zorder=3)
        ax2.set_ylabel(r"replicated $\langle Q_i\rangle$")
    else:
        ax2.text(
            0.5,
            0.55,
            "No replicated nonlinear heat-flux ensembles supplied yet.\n"
            "Use the launch manifest, then rerun this plot with --nonlinear-ensemble value:path.",
            transform=ax2.transAxes,
            ha="center",
            va="center",
            fontsize=9,
        )
        ax2.set_ylabel(r"replicated $\langle Q_i\rangle$")
    ax2.grid(True, alpha=0.25)
    ax2.set_xlabel(f"relative {report['coefficient']} perturbation [%]")
    ax2_secondary = ax2.secondary_xaxis(
        "top",
        functions=(
            lambda frac_percent: float(report["baseline_coefficient_value"]) * (1.0 + frac_percent / 100.0),
            lambda value: (value / float(report["baseline_coefficient_value"]) - 1.0) * 100.0,
        ),
    )
    ax2_secondary.set_xlabel(f"{report['coefficient']} value")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _nonlinear_launch_manifest(
    *,
    rows: list[dict[str, Any]],
    out_dir: Path,
    case_prefix: str,
    nonlinear_tmax: float,
    nonlinear_window_tmin: float,
    nonlinear_grid: str,
) -> dict[str, Any]:
    commands: list[dict[str, Any]] = []
    for row in rows:
        point = f"{case_prefix}_{row['label']}"
        input_path = Path(row["input_path"])
        expected_wout = Path(row["expected_wout"])
        config_dir = out_dir / "nonlinear_configs" / str(row["label"])
        commands.append(
            {
                "label": row["label"],
                "coefficient_value": float(row["coefficient_value"]),
                "vmec_command": f"cd {_repo_relative(input_path.parent)} && vmec_jax {input_path.name}",
                "expected_wout": _repo_relative(expected_wout),
                "write_nonlinear_configs_command": (
                    "python tools/write_optimized_equilibrium_transport_configs.py "
                    f"--vmec-file {_repo_relative(expected_wout)} "
                    f"--case {point} "
                    f"--out-dir {_repo_relative(config_dir)} "
                    f"--horizons {float(nonlinear_tmax):.12g} "
                    f"--window-tmin {float(nonlinear_window_tmin):.12g} "
                    f"--window-tmax {float(nonlinear_tmax):.12g} "
                    f"--grid {nonlinear_grid}"
                ),
            }
        )
    return {
        "kind": "vmec_boundary_transport_landscape_nonlinear_launch_manifest",
        "claim_level": "launch_plan_for_replicated_nonlinear_heat_flux_error_bars_not_simulation_claim",
        "commands": commands,
        "postprocess_instruction": (
            "After running the generated nonlinear configs, build one ensemble JSON per coefficient "
            "and rerun this landscape with --nonlinear-ensemble coefficient_value:path."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-input", type=Path, default=DEFAULT_BASELINE_INPUT)
    parser.add_argument("--coefficient", default="RBC(0,1)")
    parser.add_argument("--fractions", type=_parse_float_list, default=DEFAULT_FRACTIONS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_DOCS_PREFIX)
    parser.add_argument("--evaluate-reduced", action="store_true")
    parser.add_argument("--transport-kind", action="append", choices=DEFAULT_KINDS, default=None)
    parser.add_argument("--metric-timeout", type=float, default=300.0)
    parser.add_argument("--surfaces", default=DEFAULT_SURFACES)
    parser.add_argument("--alphas", default=DEFAULT_ALPHAS)
    parser.add_argument("--ky-values", default=DEFAULT_KY_VALUES)
    parser.add_argument("--ntheta", type=int, default=16)
    parser.add_argument("--mboz", type=int, default=21)
    parser.add_argument("--nboz", type=int, default=21)
    parser.add_argument("--n-laguerre", type=int, default=1)
    parser.add_argument("--n-hermite", type=int, default=2)
    parser.add_argument("--surface-chunk-size", type=int, default=0)
    parser.add_argument("--inner-max-iter", type=int, default=40)
    parser.add_argument("--trial-max-iter", type=int, default=40)
    parser.add_argument("--inner-ftol", type=float, default=1.0e-8)
    parser.add_argument("--trial-ftol", type=float, default=1.0e-8)
    parser.add_argument("--spectrax-objective-transform", choices=("raw", "scaled", "log1p"), default="log1p")
    parser.add_argument("--spectrax-objective-scale", type=float, default=1.0)
    parser.add_argument("--solver-device", choices=("cpu", "gpu"), default=None)
    parser.add_argument(
        "--nonlinear-ensemble",
        action="append",
        default=[],
        help="Optional replicated nonlinear ensemble, formatted as coefficient_value:path/to/ensemble.json",
    )
    parser.add_argument("--nonlinear-tmax", type=float, default=700.0)
    parser.add_argument("--nonlinear-window-tmin", type=float, default=350.0)
    parser.add_argument("--nonlinear-grid", default="n64:64:64:40:40")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = _parse_coefficient_spec(str(args.coefficient))
    kinds = tuple(args.transport_kind or DEFAULT_KINDS)
    base_value, rows = _write_scan_inputs(
        baseline_input=args.baseline_input,
        coefficient=spec,
        fractions=tuple(float(value) for value in args.fractions),
        out_dir=args.out_dir,
    )
    metric_dir = args.out_dir / "reduced_metrics"
    for row in rows:
        row["reduced_metrics"] = {}
        row["reduced_metric_reports"] = {}
        if bool(args.evaluate_reduced):
            for kind in kinds:
                result = _run_reduced_metric(row=row, kind=kind, args=args, metric_dir=metric_dir)
                row["reduced_metric_reports"][kind] = result
                if result.get("value") is not None:
                    row["reduced_metrics"][kind] = float(result["value"])
    nonlinear_points = [_load_nonlinear_ensemble(raw) for raw in args.nonlinear_ensemble]
    out_prefix = args.out_prefix
    report = {
        "kind": "vmec_boundary_transport_objective_landscape",
        "claim_level": (
            "coefficient_landscape_diagnostic; deterministic reduced metrics are not nonlinear "
            "turbulent-flux claims, and nonlinear heat-flux points require replicated ensemble JSON"
        ),
        "baseline_input": args.baseline_input,
        "coefficient": spec.label,
        "baseline_coefficient_value": base_value,
        "fractions": [float(row["relative_fraction"]) for row in rows],
        "sample_set": {
            "surfaces": list(_parse_float_list(str(args.surfaces))),
            "alphas": list(_parse_float_list(str(args.alphas))),
            "ky_values": list(_parse_float_list(str(args.ky_values))),
        },
        "reduced_metric_kinds": list(kinds),
        "rows": rows,
        "nonlinear_ensemble_points": nonlinear_points,
        "nonlinear_launch_manifest": _nonlinear_launch_manifest(
            rows=rows,
            out_dir=args.out_dir,
            case_prefix=f"landscape_{_coefficient_slug(spec)}",
            nonlinear_tmax=float(args.nonlinear_tmax),
            nonlinear_window_tmin=float(args.nonlinear_window_tmin),
            nonlinear_grid=str(args.nonlinear_grid),
        ),
        "literature_context": {
            "optimization_of_nonlinear_turbulence_in_stellarators": (
                "Published boundary-mode scans show noisy time-averaged nonlinear heat fluxes; "
                "this artifact is the analogous pre-optimizer landscape diagnostic."
            )
        },
    }
    json_path = out_prefix.with_suffix(".json")
    csv_path = out_prefix.with_suffix(".csv")
    png_path = out_prefix.with_suffix(".png")
    launch_path = args.out_dir / "nonlinear_landscape_launch_manifest.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(_json_clean(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(rows, csv_path)
    _write_plot(report=_json_clean(report), rows=rows, nonlinear_points=nonlinear_points, path=png_path)
    launch_path.parent.mkdir(parents=True, exist_ok=True)
    launch_path.write_text(
        json.dumps(_json_clean(report["nonlinear_launch_manifest"]), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "json": _repo_relative(json_path),
                "csv": _repo_relative(csv_path),
                "png": _repo_relative(png_path),
                "launch_manifest": _repo_relative(launch_path),
                "n_points": len(rows),
                "evaluated_reduced": bool(args.evaluate_reduced),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
