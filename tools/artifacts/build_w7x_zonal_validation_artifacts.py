#!/usr/bin/env python3
"""Build W7-X zonal contract and state-convention validation artifacts.

This script does not run simulations. It combines the tracked SPECTRAX-GK
W7-X zonal response artifacts with digitized stella/GENE Fig. 11 data from the
González-Jerez et al. W7-X benchmark paper. The output is deliberately
paper-facing but marked as an open audit until the residual and late-envelope
gates close under the literature normalization convention.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import matplotlib
import jax.numpy as jnp
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from spectraxgk.artifacts.plotting import set_plot_style  # noqa: E402
from spectraxgk.core.grid import SpectralGrid, build_spectral_grid  # noqa: E402
from spectraxgk.diagnostics import (  # noqa: E402
    fieldline_quadrature_weights,
    zonal_phi_line_kxt,
    zonal_phi_mode_kxt,
)
from spectraxgk.diagnostics.zonal_validation import (
    load_w7x_combined_trace_csv,
    reference_mean_trace,
)  # noqa: E402
from spectraxgk.geometry import (  # noqa: E402
    apply_geometry_grid_defaults,
    ensure_flux_tube_geometry_data,
)
from spectraxgk.linear import build_linear_cache  # noqa: E402
from spectraxgk.runtime import (  # noqa: E402
    _build_initial_condition,
    build_runtime_geometry,
    build_runtime_linear_params,
    build_runtime_term_config,
)
from spectraxgk.terms.assembly import compute_fields_cached  # noqa: E402
from spectraxgk.workflows.runtime.config import RuntimeConfig  # noqa: E402
from spectraxgk.workflows.runtime.toml import load_runtime_from_toml  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT_OUT = ROOT / "docs" / "_static" / "w7x_zonal_contract_audit.png"
TRACE_OVERLAY_KX = (0.07, 0.30)


def _parse_contract_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-traces",
        type=Path,
        default=ROOT / "docs" / "_static" / "w7x_zonal_reference_digitized.csv",
        help="Digitized stella/GENE Fig. 11 trace CSV.",
    )
    parser.add_argument(
        "--reference-residuals",
        type=Path,
        default=ROOT
        / "docs"
        / "_static"
        / "w7x_zonal_reference_digitized_residuals.csv",
        help="Digitized stella/GENE Fig. 11 inset residual CSV.",
    )
    parser.add_argument(
        "--spectrax-summary",
        type=Path,
        default=ROOT / "docs" / "_static" / "w7x_zonal_response_panel.csv",
        help="SPECTRAX-GK W7-X zonal summary CSV.",
    )
    parser.add_argument(
        "--spectrax-traces",
        type=Path,
        default=ROOT / "docs" / "_static" / "w7x_zonal_response_panel.traces.csv",
        help="Combined SPECTRAX-GK W7-X zonal trace CSV from generate_w7x_zonal_response_panel.py.",
    )
    parser.add_argument(
        "--compare-csv",
        type=Path,
        default=ROOT / "docs" / "_static" / "w7x_zonal_reference_compare.csv",
        help=(
            "Residual/time/envelope CSV from "
            "build_w7x_zonal_reference_artifacts.py compare."
        ),
    )
    parser.add_argument(
        "--out-png", type=Path, default=DEFAULT_CONTRACT_OUT, help="Output PNG path."
    )
    parser.add_argument(
        "--out-csv", type=Path, default=None, help="Output audit CSV path."
    )
    parser.add_argument(
        "--out-json", type=Path, default=None, help="Output audit JSON path."
    )
    parser.add_argument("--residual-rtol", type=float, default=0.10)
    parser.add_argument("--envelope-atol", type=float, default=0.03)
    return parser.parse_args(argv)


def _repo_relative(path: Path | str) -> str:
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, np.generic):
        return _json_clean(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def load_audit_rows(
    compare_csv: Path,
    *,
    residual_rtol: float = 0.10,
    envelope_atol: float = 0.03,
) -> list[dict[str, object]]:
    """Build a compact audit table from the tracked W7-X comparison CSV."""

    table = pd.read_csv(compare_csv)
    required = {
        "kx",
        "spectrax_residual",
        "reference_residual",
        "reference_min",
        "reference_max",
        "coverage_ratio",
        "residual_abs_error",
        "residual_atol_effective",
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{compare_csv} missing columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for _, item in table.sort_values("kx").iterrows():
        ref = float(item["reference_residual"])
        residual_tol = float(item["residual_atol_effective"]) + float(
            residual_rtol
        ) * abs(ref)
        tail_std = (
            float(item["tail_std"])
            if "tail_std" in table.columns and pd.notna(item["tail_std"])
            else np.nan
        )
        reference_tail_std = (
            float(item["reference_tail_std"])
            if "reference_tail_std" in table.columns
            and pd.notna(item["reference_tail_std"])
            else np.nan
        )
        tail_ratio = (
            tail_std / reference_tail_std
            if np.isfinite(tail_std) and reference_tail_std > 0.0
            else np.nan
        )
        rows.append(
            {
                "kx": float(item["kx"]),
                "spectrax_residual": float(item["spectrax_residual"]),
                "reference_residual": ref,
                "reference_min": float(item["reference_min"]),
                "reference_max": float(item["reference_max"]),
                "residual_abs_error": float(item["residual_abs_error"]),
                "residual_tolerance": residual_tol,
                "tail_std": tail_std,
                "reference_tail_std": reference_tail_std,
                "tail_std_ratio": tail_ratio,
                "coverage_ratio": float(item["coverage_ratio"]),
                "residual_gate_passed": bool(
                    float(item["residual_abs_error"]) <= residual_tol
                ),
                "tail_gate_passed": bool(
                    np.isfinite(tail_std)
                    and np.isfinite(reference_tail_std)
                    and abs(tail_std - reference_tail_std) <= float(envelope_atol)
                ),
            }
        )
    return rows


def contract_audit_figure(
    rows: list[dict[str, object]],
    reference_traces: pd.DataFrame,
    spectrax_traces: Path,
    *,
    overlay_kx: tuple[float, ...] = TRACE_OVERLAY_KX,
) -> plt.Figure:
    """Create the four-panel W7-X zonal-response contract audit figure."""

    if not rows:
        raise ValueError("no W7-X audit rows to plot")
    set_plot_style()
    kx = np.asarray([float(row["kx"]) for row in rows])
    x = np.arange(len(rows))
    xlabels = [f"{value:.2f}" for value in kx]
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.2), constrained_layout=True)

    ax = axes[0, 0]
    ax.fill_between(
        x,
        np.asarray([float(row["reference_min"]) for row in rows]),
        np.asarray([float(row["reference_max"]) for row in rows]),
        color="#8ecae6",
        alpha=0.42,
        label="digitized stella/GENE band",
    )
    ax.plot(
        x,
        [float(row["reference_residual"]) for row in rows],
        marker="o",
        linewidth=2.2,
        color="#1d4e89",
        label="reference mean",
    )
    ax.plot(
        x,
        [float(row["spectrax_residual"]) for row in rows],
        marker="s",
        linewidth=2.2,
        color="#c2410c",
        label="SPECTRAX-GK",
    )
    ax.set_xticks(x, xlabels)
    ax.set_xlabel(r"$k_x \rho_i$")
    ax.set_ylabel("late residual")
    ax.set_title("Residual level")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[0, 1]
    width = 0.34
    ref_tail = np.asarray(
        [float(row["reference_tail_std"]) for row in rows], dtype=float
    )
    obs_tail = np.asarray([float(row["tail_std"]) for row in rows], dtype=float)
    ax.bar(
        x - width / 2,
        ref_tail,
        width=width,
        color="#1d4e89",
        alpha=0.78,
        label="reference",
    )
    ax.bar(
        x + width / 2,
        obs_tail,
        width=width,
        color="#c2410c",
        alpha=0.78,
        label="SPECTRAX-GK",
    )
    ax.set_yscale("log")
    ax.set_xticks(x, xlabels)
    ax.set_xlabel(r"$k_x \rho_i$")
    ax.set_ylabel("late-window standard deviation")
    ax.set_title("Late envelope")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    for ax, this_kx in zip(axes[1, :], overlay_kx, strict=True):
        ref_t, ref_y = reference_mean_trace(reference_traces, this_kx)
        obs_t, obs_y = load_w7x_combined_trace_csv(
            spectrax_traces, this_kx, normalized=True
        )
        ax.plot(ref_t, ref_y, color="#1d4e89", linewidth=2.0, label="digitized mean")
        ax.plot(
            obs_t,
            obs_y,
            color="#c2410c",
            linewidth=1.7,
            alpha=0.92,
            label="SPECTRAX-GK",
        )
        ax.set_xlabel(r"$t v_{ti}/a$")
        ax.set_ylabel(r"$\phi_z/\phi_z(0)$")
        ax.set_title(rf"Trace overlay, $k_x \rho_i={this_kx:.2f}$")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle(
        "W7-X test-4 zonal-response contract audit",
        y=1.02,
        fontsize=14,
        fontweight="bold",
    )
    return fig


def write_contract_rows(rows: list[dict[str, object]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_contract_metadata(
    *,
    rows: list[dict[str, object]],
    out_json: Path,
    out_csv: Path,
    out_png: Path,
    args: argparse.Namespace,
) -> None:
    payload = {
        "case": "w7x_zonal_response_contract_audit",
        "validation_status": "open",
        "gate_index_include": False,
        "reference": "Gonzalez-Jerez et al., J. Plasma Phys. 88, 905880310 (2022), W7-X test 4",
        "reference_contract": {
            "observable": "unweighted line-averaged electrostatic potential",
            "normalization": "line-averaged potential normalized to its t=0 line-average value",
            "kx_rhoi": [float(row["kx"]) for row in rows],
        },
        "all_residual_gates_pass": all(
            bool(row["residual_gate_passed"]) for row in rows
        ),
        "all_tail_gates_pass": all(bool(row["tail_gate_passed"]) for row in rows),
        "audit_csv": _repo_relative(out_csv),
        "audit_png": _repo_relative(out_png),
        "reference_traces": _repo_relative(args.reference_traces),
        "reference_residuals": _repo_relative(args.reference_residuals),
        "spectrax_summary": _repo_relative(args.spectrax_summary),
        "spectrax_traces": _repo_relative(args.spectrax_traces),
        "compare_csv": _repo_relative(args.compare_csv),
        "rows": rows,
        "notes": (
            "This figure is a diagnostic audit, not a closed validation gate. It intentionally uses "
            "the paper-text line-first normalization rather than the clipped initial value visible in "
            "the published figure. The current artifact reaches the reference time windows, but the "
            "residual and late-envelope mismatches remain open physics/numerics work."
        ),
    }
    out_json.write_text(
        json.dumps(_json_clean(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def run_contract(argv: list[str] | None = None) -> int:
    args = _parse_contract_args(argv)
    out_csv = args.out_csv or args.out_png.with_suffix(".csv")
    out_json = args.out_json or args.out_png.with_suffix(".json")
    rows = load_audit_rows(
        args.compare_csv,
        residual_rtol=float(args.residual_rtol),
        envelope_atol=float(args.envelope_atol),
    )
    reference_traces = pd.read_csv(args.reference_traces)
    fig = contract_audit_figure(rows, reference_traces, args.spectrax_traces)
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=240, bbox_inches="tight")
    fig.savefig(args.out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    write_contract_rows(rows, out_csv)
    write_contract_metadata(
        rows=rows, out_json=out_json, out_csv=out_csv, out_png=args.out_png, args=args
    )
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_json}")
    print(f"Wrote {args.out_png}")
    return 0


DEFAULT_STATE_CONFIG = ROOT / "benchmarks" / "runtime_w7x_zonal_response_vmec.toml"
DEFAULT_STATE_OUT = ROOT / "docs" / "_static" / "w7x_zonal_state_convention_audit.png"


def _parse_state_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit W7-X zonal initial-state and observable conventions."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_STATE_CONFIG)
    parser.add_argument("--kx", type=float, default=0.07, help="Target zonal kx rho_i.")
    parser.add_argument("--ky", type=float, default=0.0, help="Target zonal ky rho_i.")
    parser.add_argument(
        "--Nl", type=int, default=None, help="Laguerre count; defaults to run.Nl."
    )
    parser.add_argument(
        "--Nm", type=int, default=None, help="Hermite count; defaults to run.Nm."
    )
    parser.add_argument("--out-png", type=Path, default=DEFAULT_STATE_OUT)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    return parser.parse_args(argv)


def _selected_indices(
    grid: SpectralGrid, *, ky_target: float, kx_target: float
) -> tuple[int, int]:
    ky = np.asarray(grid.ky, dtype=float)
    kx = np.asarray(grid.kx, dtype=float)
    ky_index = int(np.argmin(np.abs(ky - float(ky_target))))
    positive = np.flatnonzero(kx > 0.0)
    if positive.size:
        kx_index = int(positive[np.argmin(np.abs(kx[positive] - float(kx_target)))])
    else:
        kx_index = int(np.argmin(np.abs(kx - float(kx_target))))
    return ky_index, kx_index


def _expected_paper_phi(cfg: RuntimeConfig, z: np.ndarray) -> np.ndarray:
    center = 0.0
    width = float(cfg.init.gaussian_width)
    envelope = float(cfg.init.gaussian_envelope_constant) + float(
        cfg.init.gaussian_envelope_sine
    ) * np.sin(z - center)
    return float(cfg.init.init_amp) * envelope * np.exp(-(((z - center) / width) ** 2))


def _relative_l2(observed: np.ndarray, expected: np.ndarray) -> float:
    denom = float(np.linalg.norm(expected))
    if denom <= 0.0:
        return float(np.linalg.norm(observed))
    return float(np.linalg.norm(observed - expected) / denom)


def _relative_abs(observed: complex, expected: complex) -> float:
    denom = max(abs(expected), 1.0e-300)
    return float(abs(observed - expected) / denom)


def build_state_audit(
    cfg: RuntimeConfig,
    *,
    kx_target: float,
    ky_target: float = 0.0,
    Nl: int | None = None,
    Nm: int | None = None,
) -> dict[str, object]:
    """Build one state-level W7-X zonal convention audit from the runtime path."""

    raw_nl = 8 if Nl is None else int(Nl)
    raw_nm = 32 if Nm is None else int(Nm)
    geom = build_runtime_geometry(cfg)
    grid_cfg = apply_geometry_grid_defaults(geom, cfg.grid)
    if str(grid_cfg.boundary).lower() != "periodic":
        grid_cfg = replace(grid_cfg, boundary="periodic", jtwist=None, non_twist=True)
    grid_cfg = replace(grid_cfg, Lx=float(2.0 * np.pi / float(kx_target)))
    grid = build_spectral_grid(grid_cfg)
    geom_eff = ensure_flux_tube_geometry_data(geom, grid.z)
    ky_index, kx_index = _selected_indices(
        grid, ky_target=ky_target, kx_target=kx_target
    )

    params = build_runtime_linear_params(cfg, Nm=raw_nm, geom=geom_eff)
    cache = build_linear_cache(grid, geom_eff, params, raw_nl, raw_nm)
    terms = build_runtime_term_config(cfg)
    g0 = _build_initial_condition(
        grid,
        geom_eff,
        cfg,
        ky_index=ky_index,
        kx_index=kx_index,
        Nl=raw_nl,
        Nm=raw_nm,
        nspecies=sum(1 for species in cfg.species if bool(species.kinetic)),
    )
    fields = compute_fields_cached(
        jnp.asarray(g0), cache, params, terms=terms, use_custom_vjp=False
    )
    phi = np.asarray(fields.phi)
    target_phi = np.asarray(phi[ky_index, kx_index, :])
    expected = _expected_paper_phi(cfg, np.asarray(grid.z, dtype=float))
    vol_fac, _flux_fac = fieldline_quadrature_weights(geom_eff, grid)
    vol_np = np.asarray(vol_fac, dtype=float)
    line_helper = np.asarray(zonal_phi_line_kxt(jnp.asarray(phi), grid))
    mode_helper = np.asarray(zonal_phi_mode_kxt(jnp.asarray(phi), grid, vol_fac))
    manual_line = complex(np.mean(target_phi))
    manual_mode = complex(np.sum(target_phi * vol_np))
    line_value = complex(line_helper[kx_index])
    mode_value = complex(mode_helper[kx_index])

    mask = np.ones(phi.shape, dtype=bool)
    mask[ky_index, kx_index, :] = False
    target_amp = max(float(np.max(np.abs(target_phi))), 1.0e-300)
    non_target_max = float(np.max(np.abs(phi[mask]))) if np.any(mask) else 0.0
    density_seed = np.asarray(g0)[0, 0, 0, ky_index, kx_index, :]

    row = {
        "case": "w7x_zonal_state_convention_audit",
        "kx_target": float(kx_target),
        "kx_selected": float(np.asarray(grid.kx, dtype=float)[kx_index]),
        "ky_target": float(ky_target),
        "ky_selected": float(np.asarray(grid.ky, dtype=float)[ky_index]),
        "kx_index": int(kx_index),
        "ky_index": int(ky_index),
        "Nz": int(grid.z.size),
        "Nl": int(raw_nl),
        "Nm": int(raw_nm),
        "init_amp": float(cfg.init.init_amp),
        "gaussian_width": float(cfg.init.gaussian_width),
        "profile_relative_l2": _relative_l2(target_phi.real, expected),
        "profile_max_abs_error": float(np.max(np.abs(target_phi.real - expected))),
        "profile_max_relative_error": float(
            np.max(np.abs(target_phi.real - expected))
            / max(float(np.max(np.abs(expected))), 1.0e-300)
        ),
        "profile_imag_abs_max": float(np.max(np.abs(target_phi.imag))),
        "non_target_phi_abs_max": non_target_max,
        "non_target_phi_over_target": float(non_target_max / target_amp),
        "line_helper_vs_manual_rel": _relative_abs(line_value, manual_line),
        "mode_helper_vs_manual_rel": _relative_abs(mode_value, manual_mode),
        "line_first_initial_over_init_amp": float(
            abs(manual_line) / max(abs(float(cfg.init.init_amp)), 1.0e-300)
        ),
        "volume_initial_over_init_amp": float(
            abs(manual_mode) / max(abs(float(cfg.init.init_amp)), 1.0e-300)
        ),
        "line_vs_volume_relative_difference": _relative_abs(line_value, mode_value),
        "density_seed_max_over_init_amp": float(
            np.max(np.abs(density_seed)) / max(abs(float(cfg.init.init_amp)), 1.0e-300)
        ),
        "paper_initializer": "phi(theta)=init_amp*exp[-theta^2/gaussian_width^2] for ky=0",
        "paper_observable": "unweighted line average Phi_zonal_line_kxt normalized by its first nonzero sample",
    }
    passed = (
        float(row["profile_relative_l2"]) <= 1.0e-4
        and float(row["profile_imag_abs_max"])
        <= 1.0e-8 * max(abs(float(cfg.init.init_amp)), 1.0e-300)
        and float(row["non_target_phi_over_target"]) <= 1.0e-6
        and float(row["line_helper_vs_manual_rel"]) <= 1.0e-6
        and float(row["mode_helper_vs_manual_rel"]) <= 1.0e-6
    )
    return {
        "row": row,
        "passed": bool(passed),
        "z": np.asarray(grid.z, dtype=float),
        "phi_real": target_phi.real,
        "phi_imag": target_phi.imag,
        "expected_phi": expected,
        "line_value": line_value,
        "mode_value": mode_value,
    }


def state_audit_figure(audit: dict[str, object]) -> plt.Figure:
    """Create the W7-X state-convention audit figure."""

    row = dict(audit["row"])  # type: ignore[arg-type]
    z = np.asarray(audit["z"], dtype=float)
    phi = np.asarray(audit["phi_real"], dtype=float)
    expected = np.asarray(audit["expected_phi"], dtype=float)
    scale = max(float(np.max(np.abs(expected))), 1.0e-300)

    set_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.8), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(
        z,
        expected / scale,
        color="#111827",
        linewidth=2.5,
        label=r"paper $e^{-\theta^2}$",
    )
    ax.plot(
        z,
        phi / scale,
        color="#0f4c81",
        linestyle="--",
        linewidth=2.0,
        label="recovered field solve",
    )
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\phi/\max|\phi_\mathrm{paper}|$")
    ax.set_title("Initial potential profile")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[0, 1]
    keys = [
        ("line_first_initial_over_init_amp", "line avg / init_amp"),
        ("volume_initial_over_init_amp", "volume avg / init_amp"),
        ("line_vs_volume_relative_difference", "line-vs-volume rel diff"),
    ]
    ax.bar(
        [label for _key, label in keys],
        [float(row[key]) for key, _label in keys],
        color=["#0f4c81", "#2a9d8f", "#c2410c"],
    )
    ax.set_ylabel("dimensionless")
    ax.set_title("Observable convention")
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=18)

    ax = axes[1, 0]
    err_keys = [
        ("profile_relative_l2", r"profile $L^2$"),
        ("profile_max_relative_error", "profile max rel"),
        ("line_helper_vs_manual_rel", "line helper"),
        ("mode_helper_vs_manual_rel", "mode helper"),
        ("non_target_phi_over_target", "off-target"),
    ]
    values = [max(float(row[key]), 1.0e-18) for key, _label in err_keys]
    ax.bar([label for _key, label in err_keys], values, color="#7b2cbf", alpha=0.82)
    ax.set_yscale("log")
    ax.set_ylabel("relative error")
    ax.set_title("State-level closure checks")
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=20)

    ax = axes[1, 1]
    summary = (
        f"kx selected = {float(row['kx_selected']):.5f}\n"
        f"Nl,Nm = {int(row['Nl'])},{int(row['Nm'])}\n"
        f"line-first/init_amp = {float(row['line_first_initial_over_init_amp']):.5f}\n"
        f"profile L2 rel = {float(row['profile_relative_l2']):.2e}\n"
        f"non-target/target = {float(row['non_target_phi_over_target']):.2e}\n"
        f"state convention = {'closed' if bool(audit['passed']) else 'open'}"
    )
    ax.text(
        0.04, 0.96, summary, transform=ax.transAxes, va="top", ha="left", fontsize=11
    )
    ax.set_axis_off()
    fig.suptitle(
        "W7-X test-4 zonal state and observable convention audit",
        y=1.02,
        fontsize=14,
        fontweight="bold",
    )
    return fig


def write_state_outputs(
    audit: dict[str, object],
    *,
    out_png: Path,
    out_csv: Path,
    out_json: Path,
    config: Path,
) -> None:
    row = dict(audit["row"])  # type: ignore[arg-type]
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig = state_audit_figure(audit)
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(row.keys()), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerow(row)
    payload = {
        "case": "w7x_zonal_state_convention_audit",
        "validation_status": "state_convention_closed"
        if bool(audit["passed"])
        else "open",
        "gate_index_include": False,
        "config": _repo_relative(config),
        "audit_csv": _repo_relative(out_csv),
        "audit_png": _repo_relative(out_png),
        "reference": "Gonzalez-Jerez et al., J. Plasma Phys. 88, 905880310 (2022), W7-X test 4",
        "row": row,
        "notes": (
            "This state-level audit checks the paper-facing W7-X test-4 convention before time evolution: "
            "the runtime phi initializer recovers the prescribed Gaussian potential profile, the selected "
            "mode is ky=0 at the requested kx, the off-target spectral field is negligible, and the "
            "unweighted line-average observable differs explicitly from the volume-weighted zonal mode. "
            "It closes the initializer/observable convention layer but does not close the separate "
            "long-time recurrence and damping mismatch against the digitized stella/GENE traces."
        ),
    }
    out_json.write_text(
        json.dumps(_json_clean(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def run_state_convention(argv: list[str] | None = None) -> int:
    args = _parse_state_args(argv)
    cfg, raw = load_runtime_from_toml(args.config)
    run_cfg = dict(raw.get("run", {}))
    nl = int(args.Nl) if args.Nl is not None else int(run_cfg.get("Nl", 8))
    nm = int(args.Nm) if args.Nm is not None else int(run_cfg.get("Nm", 32))
    audit = build_state_audit(
        cfg, kx_target=float(args.kx), ky_target=float(args.ky), Nl=nl, Nm=nm
    )
    out_csv = args.out_csv or args.out_png.with_suffix(".csv")
    out_json = args.out_json or args.out_png.with_suffix(".json")
    write_state_outputs(
        audit,
        out_png=args.out_png,
        out_csv=out_csv,
        out_json=out_json,
        config=args.config,
    )
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_json}")
    print(f"Wrote {args.out_png}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("contract", "state-convention"))
    args, remainder = parser.parse_known_args(argv)
    if args.mode == "contract":
        return run_contract(remainder)
    return run_state_convention(remainder)


if __name__ == "__main__":
    raise SystemExit(main())
