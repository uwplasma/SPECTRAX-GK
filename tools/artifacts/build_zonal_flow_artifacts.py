#!/usr/bin/env python3
"""Build zonal-response figures and optimization-row artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import netCDF4 as nc  # noqa: E402
import numpy as np  # noqa: E402

from spectraxgk.artifacts.nonlinear_diagnostics import (  # noqa: E402
    load_diagnostic_time_series,
)
from spectraxgk.artifacts.plotting import (  # noqa: E402
    set_plot_style,
    zonal_flow_response_figure,
)
from spectraxgk.diagnostics.zonal_validation import (  # noqa: E402
    zonal_flow_response_metrics,
)
from spectraxgk.diagnostics.validation_gates import (  # noqa: E402
    evaluate_scalar_gate,
    gate_report,
    gate_report_to_dict,
)
from spectraxgk.objectives.zonal import (  # noqa: E402
    ZonalFlowObjectiveConfig,
    zonal_flow_objective_artifact_from_records,
)
from spectraxgk.workflows.runtime.artifacts import (  # noqa: E402
    run_runtime_nonlinear_with_artifacts,
)
from spectraxgk.workflows.runtime.toml import load_runtime_from_toml  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY = ROOT / "docs" / "_static" / "w7x_zonal_response_panel.csv"
DEFAULT_COMPARISON = ROOT / "docs" / "_static" / "w7x_zonal_reference_compare.csv"
DEFAULT_OUT_JSON = ROOT / "docs" / "_static" / "zonal_flow_objective_gate.json"
DEFAULT_OUT_CSV = ROOT / "docs" / "_static" / "zonal_flow_objective_gate.csv"
DEFAULT_OUT_PNG = ROOT / "docs" / "_static" / "zonal_flow_objective_gate.png"

MERLO_CASE_III_REFERENCE = {
    "paper": "Merlo et al., Phys. Plasmas 23, 032104 (2016)",
    "case": "III",
    "q_s": 1.389,
    "s_hat": 0.751,
    "epsilon": 0.18,
    "kappa": 1.4723,
    "delta": -0.0070,
    "D": -0.0139,
    "a_MHD": 0.5425,
    "dRgeom_dr": -0.1569,
    "dkappa_dr": -0.0728,
    "ddelta_dr": -0.0140,
    "kx_rhoi": 0.05,
    "ky": 0.0,
    "tmax_R0_over_vi": 150.0,
    # Figure read-offs are paper-scale gates, not frozen trace replacements.
    "residual_phi_over_phi0": 0.190,
    "omega_gam_R0_over_vi": 2.24,
    "gamma_gam_R0_over_vi": -0.17,
}

MERLO_CASE_III_GATE_TOLERANCES = {
    "residual_atol": 0.015,
    "omega_atol_R0_over_vi": 0.10,
    "gamma_atol_R0_over_vi": 0.03,
}


def _add_response_metric_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tail-fraction", type=float, default=0.3)
    parser.add_argument("--initial-fraction", type=float, default=0.1)
    parser.add_argument(
        "--initial-policy",
        choices=("window_abs_mean", "first_abs"),
        default="window_abs_mean",
    )
    parser.add_argument("--peak-fit-max-peaks", type=int, default=None)
    parser.add_argument(
        "--damping-fit-mode",
        choices=("combined_envelope", "branchwise_extrema"),
        default="combined_envelope",
    )
    parser.add_argument(
        "--frequency-fit-mode",
        choices=("peak_spacing", "hilbert_phase"),
        default="peak_spacing",
    )
    parser.add_argument("--fit-window-tmin", type=float, default=None)
    parser.add_argument("--fit-window-tmax", type=float, default=None)
    parser.add_argument("--hilbert-trim-fraction", type=float, default=0.2)


def _response_metrics(args: argparse.Namespace, t: np.ndarray, response: np.ndarray):
    return zonal_flow_response_metrics(
        t,
        response,
        tail_fraction=float(args.tail_fraction),
        initial_fraction=float(args.initial_fraction),
        initial_policy=str(args.initial_policy),
        peak_fit_max_peaks=args.peak_fit_max_peaks,
        damping_fit_mode=str(args.damping_fit_mode),
        frequency_fit_mode=str(args.frequency_fit_mode),
        fit_window_tmin=args.fit_window_tmin,
        fit_window_tmax=args.fit_window_tmax,
        hilbert_trim_fraction=float(args.hilbert_trim_fraction),
    )


def _response_payload(metrics) -> dict[str, object]:
    return {
        "initial_level": metrics.initial_level,
        "initial_policy": metrics.initial_policy,
        "residual_level": metrics.residual_level,
        "residual_std": metrics.residual_std,
        "response_rms": metrics.response_rms,
        "gam_frequency": metrics.gam_frequency,
        "gam_damping_rate": metrics.gam_damping_rate,
        "damping_method": metrics.damping_method,
        "frequency_method": metrics.frequency_method,
        "peak_count": metrics.peak_count,
        "peak_fit_count": metrics.peak_fit_count,
        "tmin": metrics.tmin,
        "tmax": metrics.tmax,
        "fit_tmin": metrics.fit_tmin,
        "fit_tmax": metrics.fit_tmax,
    }


def _write_response_panel(
    *, t: np.ndarray, response: np.ndarray, out: Path, title: str, metrics
) -> None:
    fig, _axes = zonal_flow_response_figure(
        t, response, metrics=metrics, title=title
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    if out.suffix.lower() != ".pdf":
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _build_response_csv_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot a t,response CSV artifact.")
    parser.add_argument("csv", type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "docs" / "_static" / "zonal_flow_response.png",
    )
    parser.add_argument("--title", default="Zonal-flow response")
    _add_response_metric_args(parser)
    return parser


def _build_response_output_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot a saved zonal diagnostic.")
    parser.add_argument("output", type=Path)
    parser.add_argument("--var", default="Phi2_zonal_t")
    parser.add_argument("--kx-index", type=int, default=None)
    parser.add_argument(
        "--component", choices=("real", "imag", "abs", "complex"), default="real"
    )
    parser.add_argument("--align-phase", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "docs" / "_static" / "zonal_flow_response_from_output.png",
    )
    parser.add_argument("--csv-out", type=Path, default=None)
    parser.add_argument("--title", default=None)
    _add_response_metric_args(parser)
    return parser


def _main_response_csv(argv: list[str]) -> int:
    args = _build_response_csv_parser().parse_args(argv)
    data = np.genfromtxt(args.csv, delimiter=",", names=True, dtype=float)
    if {"t", "response"} - set(data.dtype.names or ()):
        raise ValueError("CSV must contain columns t,response")
    t = np.asarray(data["t"], dtype=float)
    response = np.asarray(data["response"], dtype=float)
    metrics = _response_metrics(args, t, response)
    _write_response_panel(
        t=t, response=response, out=args.out, title=args.title, metrics=metrics
    )
    _write_json(args.out.with_suffix(".json"), _response_payload(metrics))
    return 0


def _main_response_output(argv: list[str]) -> int:
    args = _build_response_output_parser().parse_args(argv)
    series = load_diagnostic_time_series(
        args.output,
        variable=args.var,
        kx_index=args.kx_index,
        component=args.component,
        align_phase=bool(args.align_phase),
    )
    if np.iscomplexobj(series.values):
        raise ValueError(
            "zonal-response plotting requires a real extracted component"
        )
    metrics = _response_metrics(args, series.t, series.values)
    _write_response_panel(
        t=series.t,
        response=series.values,
        out=args.out,
        title=args.title or f"{args.var} response",
        metrics=metrics,
    )
    csv_out = args.csv_out or args.out.with_suffix(".csv")
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        csv_out,
        np.column_stack([series.t, series.values]),
        delimiter=",",
        header="t,response",
        comments="",
    )
    payload = {
        "source_path": series.source_path,
        "variable": series.variable,
        **_response_payload(metrics),
        "notes": (
            "Phi2_zonal_t is a zonal-energy proxy. Prefer Phi_zonal_mode_kxt "
            "with a selected kx and phase alignment for signed response studies."
        ),
    }
    _write_json(args.out.with_suffix(".json"), payload)
    return 0


def _repo_relative(path: Path | str) -> str:
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in {"", "nan", "none", "null"}:
            return None
        value = stripped
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(scalar):
        return None
    return scalar


def _kx_key(value: Any) -> float:
    scalar = _optional_float(value)
    if scalar is None:
        raise ValueError(f"missing finite kx value: {value!r}")
    return round(float(scalar), 10)


def _comparison_by_kx(path: Path | None) -> dict[float, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    table = _read_csv(path)
    rows: dict[float, dict[str, str]] = {}
    for row in table:
        if "kx" in row:
            rows[_kx_key(row["kx"])] = row
        elif "kx_target" in row:
            rows[_kx_key(row["kx_target"])] = row
    return rows


def _tail_std_ratio(row: dict[str, str] | None) -> float | None:
    if row is None:
        return None
    direct = _optional_float(row.get("tail_std_ratio"))
    if direct is not None:
        return direct
    tail_std = _optional_float(row.get("tail_std"))
    reference_tail_std = _optional_float(row.get("reference_tail_std"))
    if tail_std is None or reference_tail_std is None or reference_tail_std <= 0.0:
        return None
    return tail_std / reference_tail_std


def _recurrence_value(
    *,
    summary_row: dict[str, str],
    comparison_row: dict[str, str] | None,
    source: str,
) -> float | None:
    if source == "residual_std":
        return _optional_float(summary_row.get("residual_std"))
    if source == "tail_std":
        return (
            None
            if comparison_row is None
            else _optional_float(comparison_row.get("tail_std"))
        )
    if source == "tail_std_ratio":
        return _tail_std_ratio(comparison_row)
    if source != "auto":
        raise ValueError(f"unknown recurrence source: {source}")
    ratio = _tail_std_ratio(comparison_row)
    if ratio is not None:
        return ratio
    return _optional_float(summary_row.get("residual_std"))


def records_from_w7x_summary(
    summary_csv: Path,
    *,
    comparison_csv: Path | None = None,
    recurrence_source: str = "auto",
) -> list[dict[str, object]]:
    """Return normalized zonal-objective records from the W7-X summary CSV."""

    summary = _read_csv(summary_csv)
    comparison = _comparison_by_kx(comparison_csv)
    records: list[dict[str, object]] = []
    for row in summary:
        kx = _kx_key(row.get("kx_target", row.get("kx")))
        comparison_row = comparison.get(kx)
        recurrence = _recurrence_value(
            summary_row=row,
            comparison_row=comparison_row,
            source=recurrence_source,
        )
        records.append(
            {
                "surface": _optional_float(row.get("surface")) or 0.0,
                "alpha": _optional_float(row.get("alpha")) or 0.0,
                "kx": float(kx),
                "residual_level": row.get("residual_level"),
                "damping_rate": row.get("gam_damping_rate", row.get("damping_rate")),
                "linear_growth_rate": row.get("linear_growth_rate", 0.0),
                "recurrence_amplitude": recurrence,
            }
        )
    return records


def _write_row_csv(path: Path, payload: dict[str, object]) -> None:
    rows = list(payload["row_table"])
    if not rows:
        raise ValueError("cannot write an empty zonal-flow objective table")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "surface",
        "alpha",
        "kx",
        "residual_level",
        "damping_rate",
        "linear_growth_rate",
        "recurrence_amplitude",
        "inverse_residual",
        "growth_over_residual",
        "sample_objective",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})


def _plot_payload(path: Path, payload: dict[str, object]) -> None:
    set_plot_style()
    table = list(payload["row_table"])
    kx = np.asarray([float(row["kx"]) for row in table], dtype=float)
    order = np.argsort(kx)
    labels = [f"{kx[index]:.2f}" for index in order]
    x = np.arange(order.size)
    metrics = {
        "Residual response\n(higher is better)": [
            float(table[index]["residual_level"]) for index in order
        ],
        "Damping penalty\n(lower is better)": [
            float(table[index]["damping_rate"]) for index in order
        ],
        "Recurrence/tail penalty\n(lower is better)": [
            float(table[index]["recurrence_amplitude"]) for index in order
        ],
        "Weighted sample objective\n(lower is better)": [
            float(table[index]["sample_objective"]) for index in order
        ],
    }
    colors = ["#1f77b4", "#d62728", "#ff7f0e", "#2ca02c"]

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.6), constrained_layout=True)
    for ax, (title, values), color in zip(
        axes.ravel(), metrics.items(), colors, strict=True
    ):
        ax.bar(x, values, color=color, alpha=0.86, edgecolor="black", linewidth=0.8)
        ax.set_xticks(x, labels)
        ax.set_xlabel(r"$k_x \rho_i$")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Zonal-flow objective row gate", fontsize=15)
    status = "promotion-ready" if payload["promotion_ready"] else "diagnostic only"
    fig.text(
        0.5,
        0.01,
        (
            f"Status: {status}; missing damping rows: {payload['missing_damping_count']}; "
            f"claim: {payload['claim_level']}"
        ),
        ha="center",
        va="bottom",
        fontsize=9,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _parse_objective_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--comparison-csv", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument(
        "--recurrence-source",
        choices=("auto", "residual_std", "tail_std", "tail_std_ratio"),
        default="auto",
        help="Late-envelope recurrence metric used for the fourth objective column.",
    )
    parser.add_argument(
        "--missing-damping-policy",
        choices=("zero", "fail"),
        default="zero",
        help=(
            "Use 'fail' for promoted physics gates. The default 'zero' writes a "
            "diagnostic W7-X row artifact while preserving promotion_ready=false."
        ),
    )
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--out-png", type=Path, default=DEFAULT_OUT_PNG)
    parser.add_argument("--residual-weight", type=float, default=1.0)
    parser.add_argument("--damping-weight", type=float, default=1.0)
    parser.add_argument("--growth-over-residual-weight", type=float, default=0.0)
    parser.add_argument("--recurrence-weight", type=float, default=0.25)
    parser.add_argument("--residual-floor", type=float, default=1.0e-6)
    parser.add_argument(
        "--claim-level",
        default="diagnostic_zonal_objective_row_producer_not_promoted_w7x_optimization_claim",
    )
    return parser.parse_args(argv)


def _main_objective_gate(argv: list[str]) -> int:
    args = _parse_objective_args(argv)
    records = records_from_w7x_summary(
        args.summary_csv,
        comparison_csv=args.comparison_csv,
        recurrence_source=args.recurrence_source,
    )
    config = ZonalFlowObjectiveConfig(
        residual_weight=args.residual_weight,
        damping_weight=args.damping_weight,
        growth_over_residual_weight=args.growth_over_residual_weight,
        recurrence_weight=args.recurrence_weight,
        residual_floor=args.residual_floor,
    )
    payload = zonal_flow_objective_artifact_from_records(
        records,
        config=config,
        missing_damping_policy=args.missing_damping_policy,
        claim_level=args.claim_level,
        source_paths=[
            _repo_relative(args.summary_csv),
            _repo_relative(args.comparison_csv),
        ],
    )
    payload["input_summary_csv"] = _repo_relative(args.summary_csv)
    payload["input_comparison_csv"] = _repo_relative(args.comparison_csv)
    payload["recurrence_source"] = args.recurrence_source
    payload["validation_status"] = (
        "closed" if payload["promotion_ready"] else "diagnostic"
    )
    payload["gate_index_include"] = False
    payload["notes"] = [
        "This artifact verifies the row-production contract for zonal-flow optimization objectives.",
        "W7-X rows with missing GAM damping remain diagnostic and are not promoted to an optimization claim.",
        "Use --missing-damping-policy=fail for closed QA/QH/Miller-style promotion gates.",
    ]
    json.dumps(payload, allow_nan=False)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_row_csv(args.out_csv, payload)
    _plot_payload(args.out_png, payload)
    print(
        "wrote zonal-flow objective gate "
        f"samples={payload['sample_count']} promotion_ready={payload['promotion_ready']} "
        f"json={_repo_relative(args.out_json)}"
    )
    return 0


def _build_miller_panel_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "benchmarks" / "runtime_miller_zonal_response.toml",
        help="Runtime TOML for the shaped-Miller zonal-response panel.",
    )
    parser.add_argument(
        "--out-bundle",
        type=Path,
        default=ROOT
        / "tools_out"
        / "zonal_response"
        / "miller_caseIII_initial_density_Nl4_Nm24_Nz32_dt0005_t60.out.nc",
        help="Runtime output bundle path.",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=ROOT / "docs" / "_static" / "miller_zonal_response_pilot.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--tail-fraction",
        type=float,
        default=0.3,
        help="Late-time fraction used for the residual window.",
    )
    parser.add_argument(
        "--initial-fraction",
        type=float,
        default=0.1,
        help="Leading fraction used for the initial normalization window.",
    )
    parser.add_argument(
        "--initial-policy",
        choices=("first_abs", "window_abs_mean"),
        default="first_abs",
        help=(
            "Initial normalization convention. Merlo/Rosenbluth-Hinton residuals "
            "are quoted as phi(t->infinity)/phi(0), so this tool defaults to first_abs."
        ),
    )
    parser.add_argument(
        "--peak-fit-max-peaks",
        type=int,
        default=4,
        help=(
            "Maximum number of positive and negative extrema used per branch for the "
            "Merlo-style damping fit."
        ),
    )
    parser.add_argument(
        "--fit-window-tmax",
        type=float,
        default=30.0,
        help="Upper time bound for the common pre-recurrence GAM fit window.",
    )
    parser.add_argument(
        "--reuse-output",
        action="store_true",
        help="Reuse an existing out.nc bundle instead of rerunning the panel source simulation.",
    )
    return parser


def _nearest_kx_index(path: Path, target_kx: float) -> tuple[int, float]:
    with nc.Dataset(path, "r") as ds:
        grids = ds.groups.get("Grids")
        if grids is None or "kx" not in grids.variables:
            raise ValueError(f"missing Grids/kx in {path}")
        kx = np.asarray(grids.variables["kx"][:], dtype=float)
    if kx.ndim != 1 or kx.size == 0:
        raise ValueError(f"invalid kx grid in {path}")
    idx = int(np.argmin(np.abs(kx - float(target_kx))))
    return idx, float(kx[idx])


def _setup_note(cfg) -> str:
    source = str(getattr(cfg.expert, "source", "default")).strip().lower()
    if source == "phiext_full":
        return "external phiext_full source"
    return f"initial {cfg.init.init_field} perturbation"


def _main_miller_panel(argv: list[str]) -> int:
    args = _build_miller_panel_parser().parse_args(argv)
    cfg, raw = load_runtime_from_toml(args.config)
    run_cfg = dict(raw.get("run", {}))
    ky_target = float(run_cfg.get("ky", 0.0))
    kx_target = float(run_cfg.get("kx", 0.1))
    nl = int(run_cfg.get("Nl", 2))
    nm = int(run_cfg.get("Nm", 2))
    dt = float(run_cfg.get("dt", cfg.time.dt))
    steps = int(run_cfg.get("steps", max(int(round(float(cfg.time.t_max) / dt)), 1)))
    sample_stride = int(run_cfg.get("sample_stride", cfg.time.sample_stride))
    diagnostics = bool(run_cfg.get("diagnostics", cfg.time.diagnostics))

    out_bundle = Path(args.out_bundle)
    out_bundle.parent.mkdir(parents=True, exist_ok=True)
    if not args.reuse_output or not out_bundle.exists():
        run_runtime_nonlinear_with_artifacts(
            cfg,
            out=out_bundle,
            ky_target=ky_target,
            kx_target=kx_target,
            Nl=nl,
            Nm=nm,
            dt=dt,
            steps=steps,
            sample_stride=sample_stride,
            diagnostics=diagnostics,
            show_progress=False,
        )

    kx_index, kx_selected = _nearest_kx_index(out_bundle, kx_target)
    series = load_diagnostic_time_series(
        out_bundle,
        variable="Phi_zonal_mode_kxt",
        kx_index=kx_index,
        component="real",
        align_phase=True,
    )
    if np.iscomplexobj(series.values):
        raise ValueError(
            "signed zonal-response plotting requires a real-valued phase-aligned trace"
        )

    metrics = zonal_flow_response_metrics(
        series.t,
        np.asarray(series.values, dtype=float),
        tail_fraction=float(args.tail_fraction),
        initial_fraction=float(args.initial_fraction),
        initial_policy=str(args.initial_policy),
        peak_fit_max_peaks=int(args.peak_fit_max_peaks)
        if args.peak_fit_max_peaks is not None
        else None,
        damping_fit_mode="branchwise_extrema",
        frequency_fit_mode="hilbert_phase",
        fit_window_tmax=float(args.fit_window_tmax),
        hilbert_trim_fraction=0.2,
    )
    setup_note = _setup_note(cfg)
    ref_residual = float(MERLO_CASE_III_REFERENCE["residual_phi_over_phi0"])
    ref_omega = float(MERLO_CASE_III_REFERENCE["omega_gam_R0_over_vi"])
    ref_gamma = float(MERLO_CASE_III_REFERENCE["gamma_gam_R0_over_vi"])
    r0 = float(getattr(cfg.geometry, "R0", 1.0))
    omega_r0_over_vi = float(metrics.gam_frequency) * r0
    damping_r0_over_vi = float(metrics.gam_damping_rate) * r0
    gamma_r0_over_vi = -damping_r0_over_vi
    residual_abs_error = abs(float(metrics.residual_level) - ref_residual)
    omega_abs_error = abs(omega_r0_over_vi - ref_omega)
    gamma_abs_error = abs(gamma_r0_over_vi - ref_gamma)
    validation_gate_report = gate_report(
        "merlo_case_iii_zonal_response",
        "Merlo et al. paper-scale read-off",
        (
            evaluate_scalar_gate(
                "residual_level",
                metrics.residual_level,
                ref_residual,
                atol=float(MERLO_CASE_III_GATE_TOLERANCES["residual_atol"]),
                rtol=0.0,
            ),
            evaluate_scalar_gate(
                "gam_frequency_R0_over_vi",
                omega_r0_over_vi,
                ref_omega,
                atol=float(MERLO_CASE_III_GATE_TOLERANCES["omega_atol_R0_over_vi"]),
                rtol=0.0,
                units="R0/vi",
            ),
            evaluate_scalar_gate(
                "gam_growth_rate_R0_over_vi",
                gamma_r0_over_vi,
                ref_gamma,
                atol=float(MERLO_CASE_III_GATE_TOLERANCES["gamma_atol_R0_over_vi"]),
                rtol=0.0,
                units="R0/vi",
                notes="Signed growth-rate convention; negative values correspond to damping.",
            ),
        ),
    )
    title = f"Merlo Case III zonal-response (ky={ky_target:.3f}, kx={kx_selected:.3f})"
    fig, _axes = zonal_flow_response_figure(
        series.t,
        np.asarray(series.values, dtype=float),
        metrics=metrics,
        title=title,
        y_label=r"$\phi_\mathrm{zonal}/|\phi_\mathrm{zonal}(0)|$",
    )
    ax0 = _axes[0]
    ax0.axhline(
        ref_residual,
        color="#7b2cbf",
        linestyle=":",
        linewidth=2.1,
        label="Merlo Case III residual",
    )
    ax0.legend(loc="best", frameon=False)

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=220, bbox_inches="tight")
    fig.savefig(args.out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    csv_out = args.out_png.with_suffix(".csv")
    np.savetxt(
        csv_out,
        np.column_stack([series.t, np.asarray(series.values, dtype=float)]),
        delimiter=",",
        header="t,phi_zonal_real",
        comments="",
    )

    meta_out = args.out_png.with_suffix(".json")
    meta_out.write_text(
        json.dumps(
            {
                "config": _repo_relative(args.config),
                "source_path": _repo_relative(series.source_path),
                "variable": "Phi_zonal_mode_kxt",
                "kx_index": int(kx_index),
                "kx_selected": float(kx_selected),
                "ky_target": float(ky_target),
                "initial_level": float(metrics.initial_level),
                "initial_policy": str(metrics.initial_policy),
                "residual_level": float(metrics.residual_level),
                "residual_std": float(metrics.residual_std),
                "response_rms": float(metrics.response_rms),
                "gam_frequency": float(metrics.gam_frequency),
                "gam_damping_rate": float(metrics.gam_damping_rate),
                "gam_frequency_R0_over_vi": float(omega_r0_over_vi),
                "gam_damping_rate_R0_over_vi": float(damping_r0_over_vi),
                "gam_growth_rate_R0_over_vi": float(gamma_r0_over_vi),
                "damping_method": str(metrics.damping_method),
                "frequency_method": str(metrics.frequency_method),
                "peak_count": int(metrics.peak_count),
                "peak_fit_count": int(metrics.peak_fit_count),
                "tmin": float(metrics.tmin),
                "tmax": float(metrics.tmax),
                "fit_tmin": float(metrics.fit_tmin),
                "fit_tmax": float(metrics.fit_tmax),
                "literature_reference": dict(MERLO_CASE_III_REFERENCE),
                "gate_tolerances": dict(MERLO_CASE_III_GATE_TOLERANCES),
                "gate_report": gate_report_to_dict(validation_gate_report),
                "paper_scale_gate_passed": bool(validation_gate_report.passed),
                "residual_abs_error_vs_literature": float(residual_abs_error),
                "omega_abs_error_vs_literature_R0_over_vi": float(omega_abs_error),
                "gamma_abs_error_vs_literature_R0_over_vi": float(gamma_abs_error),
                "setup": setup_note,
                "validation_status": "open",
                "notes": (
                    "This is a Merlo Case-III shaped-Miller zonal-relaxation run "
                    f"built from the signed zonal observable Phi_zonal_mode_kxt with zero gradients, "
                    f"adiabatic electrons, and an {setup_note}. "
                    "The literature reference values are read from Merlo et al. Figs. 12, 14, and 16; "
                    "the residual is normalized with the Rosenbluth-Hinton first-sample convention. "
                    f"The GAM damping follows the paper convention by fitting positive and negative extrema separately "
                    f"over the common pre-recurrence window t in [{metrics.fit_tmin:.1f}, {metrics.fit_tmax:.1f}] "
                    f"using up to {args.peak_fit_max_peaks} extrema per branch, while the frequency is obtained from "
                    "the instantaneous phase of the same window via a Hilbert-transform analytic signal. "
                    "The residual, damping, and GAM frequency are now close to the paper-scale read-off; the long-time recurrence "
                    "behavior still remains an explicit numerical follow-up item."
                ),
                "references": [
                    "Merlo et al. 2016 shaped-tokamak collisionless GAM benchmark, Case III",
                    "W7-X stella/GENE benchmark 2022 for zonal-flow observable conventions",
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0

def main(argv: list[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    if not tokens:
        print(
            "usage: build_zonal_flow_artifacts.py "
            "{response-csv,response-output,objective-gate,miller-panel} ..."
        )
        return 2
    command, rest = tokens[0], tokens[1:]
    if command == "response-csv":
        return _main_response_csv(rest)
    if command == "response-output":
        return _main_response_output(rest)
    if command == "objective-gate":
        return _main_objective_gate(rest)
    if command == "miller-panel":
        return _main_miller_panel(rest)
    raise SystemExit(f"unknown command: {command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
