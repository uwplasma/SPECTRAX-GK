#!/usr/bin/env python3
"""Scan one VMEC boundary coefficient against SPECTRAX-GK transport objectives.

The first nonlinear turbulence-optimization studies in stellarators show that
late-time nonlinear heat-flux objectives can be noisy in boundary-parameter
space. This tool makes that diagnostic explicit for SPECTRAX-GK: perturb one
VMEC input coefficient, evaluate deterministic reduced linear/quasilinear
objectives, and optionally overlay true post-transient nonlinear heat-flux
points with uncertainty bars when those expensive simulations have finished.

The deterministic reduced curves are not nonlinear turbulent-flux claims. The
nonlinear points are accepted only from supplied post-transient nonlinear-output
ensemble JSON files.
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

from spectraxgk.artifacts.plotting import set_plot_style  # noqa: E402
from tools.write_vmec_boundary_perturbation_inputs import (  # noqa: E402
    CoefficientSpec,
    _coefficient_value,
    _parse_coefficient_spec,
    _patch_coefficient,
)


DEFAULT_BASELINE_INPUT = ROOT / "tools_out/latest_vmec_stack/authoritative_qa_baseline/input.final"
DEFAULT_OUT_DIR = ROOT / "tools_out/vmec_boundary_transport_landscape"
DEFAULT_DOCS_PREFIX = ROOT / "docs/_static/vmec_boundary_transport_landscape_rbc11_full"
DEFAULT_FRACTIONS = tuple(float(round(value, 2)) for value in np.linspace(-0.75, 0.75, 31))
DEFAULT_KINDS = (
    "growth",
    "quasilinear_flux_linear_weight",
    "quasilinear_flux_mixing_length",
    "quasilinear_flux_lapillonne_2011",
    "quasilinear_flux_absolute_growth_mixing_length",
    "quasilinear_flux_shape_aware_power_law",
)
DEFAULT_SURFACES = "0.45,0.64,0.78"
DEFAULT_ALPHAS = "0.0,0.7853981633974483"
DEFAULT_KY_VALUES = "0.10,0.30,0.50"


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
    zero_reference_coefficients: tuple[CoefficientSpec, ...] = (),
) -> tuple[float, float, str, list[dict[str, Any]]]:
    text = baseline_input.read_text(encoding="utf-8")
    base_value = _coefficient_value(text, coefficient)
    scan_amplitude = float(base_value)
    scan_reference = coefficient.label
    if abs(base_value) <= 0.0:
        reference_values = [
            (_coefficient_value(text, ref), ref.label)
            for ref in zero_reference_coefficients
        ]
        reference_values = [
            (float(value), label)
            for value, label in reference_values
            if math.isfinite(float(value)) and abs(float(value)) > 0.0
        ]
        if not reference_values:
            raise ValueError(
                "zero-baseline landscape scan requires at least one nonzero "
                "--zero-reference-coefficient"
            )
        scan_amplitude, scan_reference = max(reference_values, key=lambda item: abs(item[0]))
        scan_amplitude = abs(float(scan_amplitude))
    rows: list[dict[str, Any]] = []
    input_dir = out_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    seen: set[float] = set()
    for fraction in fractions:
        fraction_f = float(fraction)
        value = (
            float(base_value * (1.0 + fraction_f))
            if abs(base_value) > 0.0
            else float(base_value + scan_amplitude * fraction_f)
        )
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
                "scan_amplitude": scan_amplitude,
                "scan_reference": scan_reference,
                "input_path": input_path,
                "expected_wout": input_dir / f"wout_{_coefficient_slug(coefficient)}_{label}.nc",
            }
        )
    return float(base_value), float(scan_amplitude), str(scan_reference), rows


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
        "--out-wout",
        str(row["expected_wout"]),
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


def _run_reduced_metric_batch(
    *,
    row: dict[str, Any],
    kinds: tuple[str, ...],
    args: argparse.Namespace,
    metric_dir: Path,
) -> dict[str, dict[str, Any]]:
    """Evaluate all missing reduced metrics for one coefficient in one VMEC solve."""

    out_dir = metric_dir / str(row["label"])
    out_json = out_dir / "batch.json"
    cmd = [
        sys.executable,
        str(ROOT / "tools/evaluate_vmec_jax_spectrax_transport_metric.py"),
        "--input",
        str(row["input_path"]),
        "--out-json",
        str(out_json),
        "--out-json-dir",
        str(out_dir),
        "--outdir",
        str(out_dir / "batch_vmec"),
        "--out-wout",
        str(row["expected_wout"]),
        "--transport-kind",
        "all",
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
            kind: {
                "kind": kind,
                "metric_json": out_dir / f"{kind}.json",
                "command": cmd,
                "returncode": 124,
                "error": f"batched metric eval timed out after {float(args.metric_timeout):.1f}s",
                "stdout_tail": (exc.stdout or "")[-1200:] if isinstance(exc.stdout, str) else "",
                "stderr_tail": (exc.stderr or "")[-1200:] if isinstance(exc.stderr, str) else "",
                "value": None,
            }
            for kind in kinds
        }
    reports: dict[str, dict[str, Any]] = {}
    for kind in kinds:
        kind_json = out_dir / f"{kind}.json"
        payload = None
        value = None
        if result.returncode == 0 and kind_json.exists():
            payload = json.loads(kind_json.read_text(encoding="utf-8"))
            value = float(payload["transport_objective_final"])
        reports[kind] = {
            "kind": kind,
            "metric_json": kind_json,
            "command": cmd,
            "returncode": int(result.returncode),
            "stdout_tail": result.stdout[-1200:],
            "stderr_tail": result.stderr[-1200:],
            "value": value,
            "payload": payload,
        }
    return reports


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


def _matching_float_lists(left: tuple[float, ...], right: list[Any], *, rtol: float = 1.0e-10) -> bool:
    if len(left) != len(right):
        return False
    return all(math.isclose(float(a), float(b), rel_tol=rtol, abs_tol=rtol) for a, b in zip(left, right))


def _compact_sample_statistics(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    compact = dict(value)
    compact.pop("rows", None)
    compact["rows_included"] = False
    return compact


def _reuse_reduced_metrics_from_report(
    *,
    rows: list[dict[str, Any]],
    kinds: tuple[str, ...],
    path: Path,
    coefficient_label: str,
    baseline_value: float,
    args: argparse.Namespace,
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if str(payload.get("coefficient")) != coefficient_label:
        raise ValueError(
            f"{path} stores coefficient {payload.get('coefficient')!r}, expected {coefficient_label!r}"
        )
    old_base = float(payload.get("baseline_coefficient_value"))
    if not math.isclose(old_base, float(baseline_value), rel_tol=1.0e-10, abs_tol=1.0e-12):
        raise ValueError(
            f"{path} baseline coefficient {old_base:.16g} does not match current {baseline_value:.16g}"
        )
    sample_set = payload.get("sample_set", {})
    if not isinstance(sample_set, dict):
        raise ValueError(f"{path} does not contain a sample_set object")
    expected_samples = {
        "surfaces": _parse_float_list(str(args.surfaces)),
        "alphas": _parse_float_list(str(args.alphas)),
        "ky_values": _parse_float_list(str(args.ky_values)),
    }
    for key, expected in expected_samples.items():
        if not _matching_float_lists(expected, list(sample_set.get(key, []))):
            raise ValueError(
                f"{path} sample_set.{key}={sample_set.get(key)!r} does not match current {list(expected)!r}"
            )

    reusable_rows = payload.get("rows", [])
    if not isinstance(reusable_rows, list):
        raise ValueError(f"{path} rows must be a list")
    by_label = {str(row.get("label")): row for row in reusable_rows if isinstance(row, dict)}
    for row in rows:
        stored = by_label.get(str(row["label"]))
        if stored is None:
            raise ValueError(f"{path} has no reusable reduced metrics for label {row['label']!r}")
        old_value = float(stored.get("coefficient_value"))
        if not math.isclose(old_value, float(row["coefficient_value"]), rel_tol=1.0e-10, abs_tol=1.0e-12):
            raise ValueError(
                f"{path} coefficient for label {row['label']!r} is {old_value:.16g}, "
                f"expected {float(row['coefficient_value']):.16g}"
            )
        metrics = stored.get("reduced_metrics", {})
        reports = stored.get("reduced_metric_reports", {})
        if not isinstance(metrics, dict):
            continue
        if not isinstance(reports, dict):
            reports = {}
        for kind in kinds:
            value = metrics.get(kind)
            if value is None:
                continue
            stored_report = reports.get(kind, {})
            payload = stored_report.get("payload", {}) if isinstance(stored_report, dict) else {}
            sample_statistics = payload.get("sample_statistics") if isinstance(payload, dict) else None
            reused_report: dict[str, Any] = {
                "kind": kind,
                "value": float(value),
                "returncode": 0,
                "reused_from": path,
            }
            if sample_statistics is not None:
                reused_report["payload"] = {
                    "sample_statistics": _compact_sample_statistics(sample_statistics)
                }
            row["reduced_metrics"][kind] = float(value)
            row["reduced_metric_reports"][kind] = reused_report


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_names = sorted(
        {str(key) for row in rows for key in row.get("reduced_metrics", {}).keys()}
        | set(DEFAULT_KINDS)
    )
    fieldnames = [
        "label",
        "relative_fraction",
        "coefficient_value",
        "input_path",
        *metric_names,
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
                    **{name: metrics.get(name) for name in metric_names},
                }
            )


def _normalization_scale(values: np.ndarray, *, baseline_index: int) -> float:
    finite = np.isfinite(values)
    if not np.any(finite):
        return 1.0
    baseline_index = max(0, min(int(baseline_index), int(values.size) - 1))
    scale = float(values[baseline_index]) if np.isfinite(values[baseline_index]) else float("nan")
    if abs(scale) < 1.0e-14:
        scale = float(np.nanmax(np.abs(values[finite])))
    if abs(scale) <= 0.0 or not np.isfinite(scale):
        scale = 1.0
    return scale


def _normalize(values: np.ndarray, *, baseline_index: int) -> np.ndarray:
    finite = np.isfinite(values)
    if not np.any(finite):
        return values
    scale = _normalization_scale(values, baseline_index=baseline_index)
    return values / scale


def _sample_standard_error(row: dict[str, Any], kind: str) -> float:
    report = row.get("reduced_metric_reports", {}).get(kind, {})
    payload = report.get("payload", {}) if isinstance(report, dict) else {}
    stats = payload.get("sample_statistics", {}) if isinstance(payload, dict) else {}
    if not isinstance(stats, dict):
        return float("nan")
    value = stats.get("weighted_standard_error")
    if value is None:
        return float("nan")
    return float(value)


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
    baseline_value = float(report["baseline_coefficient_value"])
    scan_amplitude = float(report.get("scan_amplitude", baseline_value))
    metric_kinds = tuple(str(kind) for kind in report.get("reduced_metric_kinds", DEFAULT_KINDS))
    metrics = {
        kind: np.asarray([row.get("reduced_metrics", {}).get(kind, np.nan) for row in rows], dtype=float)
        for kind in metric_kinds
    }

    fig, axes = plt.subplots(2, 1, figsize=(7.8, 6.6), sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    colors = {
        "growth": "#111827",
        "quasilinear_flux": "#1d4ed8",
        "quasilinear_flux_linear_weight": "#0891b2",
        "quasilinear_flux_mixing_length": "#2563eb",
        "quasilinear_flux_lapillonne_2011": "#7c3aed",
        "quasilinear_flux_absolute_growth_mixing_length": "#db2777",
        "quasilinear_flux_shape_aware_power_law": "#ea580c",
    }
    labels = {
        "growth": r"linear growth $\gamma$",
        "quasilinear_flux": "QL mixing-length",
        "quasilinear_flux_linear_weight": "QL linear weight",
        "quasilinear_flux_mixing_length": "QL mixing-length",
        "quasilinear_flux_lapillonne_2011": "QL Lapillonne 2011",
        "quasilinear_flux_absolute_growth_mixing_length": r"QL $|\gamma|/k_\perp^2$",
        "quasilinear_flux_shape_aware_power_law": "QL shape-aware",
    }
    ax0 = axes[0]
    top_kinds = tuple(kind for kind in metric_kinds if kind == "growth" or kind.startswith("quasilinear_flux"))
    for kind in top_kinds:
        y = metrics[kind]
        if np.any(np.isfinite(y)):
            y_plot = y
            ax0.errorbar(
                x,
                y_plot,
                yerr=None,
                marker="o",
                lw=1.6,
                capsize=3,
                color=colors.get(kind),
                label=labels.get(kind, kind),
            )
    ax0.set_ylabel("linear / QL metric")
    positive = np.concatenate([metrics[kind] for kind in top_kinds if kind in metrics])
    positive = positive[np.isfinite(positive) & (positive > 0.0)]
    if positive.size and float(np.nanmax(positive) / max(np.nanmin(positive), 1.0e-300)) > 100.0:
        ax0.set_yscale("log")
    ax0.set_title(f"{report['coefficient']} linear and quasilinear landscape")
    if ax0.has_data():
        ax0.legend(frameon=False, fontsize=7, ncols=2)
    ax0.grid(True, alpha=0.25)

    ax1 = axes[1]
    if nonlinear_points:
        base = float(report["baseline_coefficient_value"])
        xs = np.asarray([(float(point["coefficient_value"]) / base - 1.0) * 100.0 for point in nonlinear_points])
        if abs(base) <= 0.0:
            xs = np.asarray(
                [
                    (float(point["coefficient_value"]) - baseline_value)
                    / max(abs(scan_amplitude), 1.0e-300)
                    * 100.0
                    for point in nonlinear_points
                ]
            )
        means = np.asarray([float(point["mean"]) for point in nonlinear_points])
        sem = np.asarray([float(point["sem"]) for point in nonlinear_points])
        colors_nl = ["#0f766e" if bool(point.get("passed", False)) else "#b91c1c" for point in nonlinear_points]
        order = np.argsort(xs)
        ax1.errorbar(xs[order], means[order], yerr=sem[order], fmt="none", ecolor="0.25", elinewidth=1.1, capsize=4)
        ax1.plot(xs[order], means[order], color="#475569", lw=1.0, alpha=0.45)
        ax1.scatter(xs, means, c=colors_nl, s=48, edgecolors="0.15", linewidths=0.5, zorder=3)
    else:
        ax1.text(
            0.5,
            0.5,
            "Long-window nonlinear heat-flux landscape pending.\n"
            "Generate t=[1100,1500] or longer nonlinear simulations for each coefficient.",
            transform=ax1.transAxes,
            ha="center",
            va="center",
        )
    ax1.set_ylabel(r"post-transient nonlinear heat flux $Q_i$")
    ax1.grid(True, alpha=0.25)
    ax1.set_xlabel(f"relative {report['coefficient']} perturbation [%]")
    ax_secondary = ax1.secondary_xaxis(
        "top",
        functions=(
            lambda frac_percent: baseline_value + scan_amplitude * (frac_percent / 100.0),
            lambda value: (value - baseline_value) / scan_amplitude * 100.0,
        ),
    )
    ax_secondary.set_xlabel(f"{report['coefficient']} value")
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
            "and rerun this landscape with --nonlinear-ensemble coefficient_value:path. For diagnostic "
            "landscapes, build those ensemble JSON files with "
            "tools/build_external_vmec_replicate_ensemble.py --allow-failed-gates so failed "
            "post-transient convergence points remain visible instead of aborting the full scan."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-input", type=Path, default=DEFAULT_BASELINE_INPUT)
    parser.add_argument("--coefficient", default="RBC(1,1)")
    parser.add_argument(
        "--zero-reference-coefficient",
        action="append",
        default=["RBC(1,0)", "RBC(0,1)"],
        help=(
            "Reference coefficient used to set the absolute scan amplitude when "
            "--coefficient has zero baseline value; may be supplied multiple times."
        ),
    )
    parser.add_argument("--fractions", type=_parse_float_list, default=DEFAULT_FRACTIONS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_DOCS_PREFIX)
    parser.add_argument("--evaluate-reduced", action="store_true")
    parser.add_argument(
        "--reuse-reduced-json",
        type=Path,
        default=None,
        help=(
            "Reuse deterministic reduced metrics from a previous landscape JSON after validating "
            "coefficient values and the surface/alpha/ky sample set. Missing kinds are recomputed "
            "only when --evaluate-reduced is also supplied."
        ),
    )
    parser.add_argument(
        "--transport-kind",
        action="append",
        choices=(*DEFAULT_KINDS, "quasilinear_flux", "nonlinear_window_heat_flux"),
        default=None,
    )
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
    parser.add_argument("--nonlinear-tmax", type=float, default=1500.0)
    parser.add_argument("--nonlinear-window-tmin", type=float, default=1100.0)
    parser.add_argument("--nonlinear-grid", default="n64:64:64:40:40")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = _parse_coefficient_spec(str(args.coefficient))
    kinds = tuple(args.transport_kind or DEFAULT_KINDS)
    zero_reference_coefficients = tuple(
        _parse_coefficient_spec(str(item)) for item in (args.zero_reference_coefficient or [])
    )
    base_value, scan_amplitude, scan_reference, rows = _write_scan_inputs(
        baseline_input=args.baseline_input,
        coefficient=spec,
        fractions=tuple(float(value) for value in args.fractions),
        out_dir=args.out_dir,
        zero_reference_coefficients=zero_reference_coefficients,
    )
    metric_dir = args.out_dir / "reduced_metrics"
    for row in rows:
        row["reduced_metrics"] = {}
        row["reduced_metric_reports"] = {}
    if args.reuse_reduced_json is not None:
        _reuse_reduced_metrics_from_report(
            rows=rows,
            kinds=kinds,
            path=Path(args.reuse_reduced_json),
            coefficient_label=spec.label,
            baseline_value=base_value,
            args=args,
        )
    for row in rows:
        if bool(args.evaluate_reduced):
            missing_kinds = tuple(kind for kind in kinds if kind not in row["reduced_metrics"])
            if len(missing_kinds) > 1:
                batch_results = _run_reduced_metric_batch(
                    row=row,
                    kinds=missing_kinds,
                    args=args,
                    metric_dir=metric_dir,
                )
                for kind, result in batch_results.items():
                    row["reduced_metric_reports"][kind] = result
                    if result.get("value") is not None:
                        row["reduced_metrics"][kind] = float(result["value"])
                continue
            for kind in missing_kinds:
                result = _run_reduced_metric(row=row, kind=kind, args=args, metric_dir=metric_dir)
                row["reduced_metric_reports"][kind] = result
                if result.get("value") is not None:
                    row["reduced_metrics"][kind] = float(result["value"])
    nonlinear_points = [_load_nonlinear_ensemble(raw) for raw in args.nonlinear_ensemble]
    out_prefix = args.out_prefix
    report = {
        "kind": "vmec_boundary_transport_objective_landscape",
        "claim_level": (
            "coefficient_landscape_diagnostic; top-panel linear/quasilinear metrics are deterministic "
            "linear-state diagnostics, and bottom-panel nonlinear heat-flux points require long-window "
            "post-transient nonlinear-output ensemble JSON"
        ),
        "baseline_input": args.baseline_input,
        "coefficient": spec.label,
        "baseline_coefficient_value": base_value,
        "scan_amplitude": scan_amplitude,
        "scan_reference": scan_reference,
        "scan_mode": "relative_to_baseline" if abs(base_value) > 0.0 else "absolute_from_reference",
        "fractions": [float(row["relative_fraction"]) for row in rows],
        "sample_set": {
            "surfaces": list(_parse_float_list(str(args.surfaces))),
            "alphas": list(_parse_float_list(str(args.alphas))),
            "ky_values": list(_parse_float_list(str(args.ky_values))),
        },
        "reduced_metric_kinds": list(kinds),
        "reused_reduced_metrics_json": args.reuse_reduced_json,
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
