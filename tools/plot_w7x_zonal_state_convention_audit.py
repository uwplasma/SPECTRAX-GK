#!/usr/bin/env python3
"""Audit the W7-X zonal-response initial state and observable conventions."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from spectraxgk.diagnostics import gx_phi_zonal_line_kxt, gx_phi_zonal_mode_kxt, gx_volume_factors  # noqa: E402
from spectraxgk.geometry import apply_geometry_grid_defaults, ensure_flux_tube_geometry_data  # noqa: E402
from spectraxgk.grids import SpectralGrid, build_spectral_grid  # noqa: E402
from spectraxgk.io import load_runtime_from_toml  # noqa: E402
from spectraxgk.linear import build_linear_cache  # noqa: E402
from spectraxgk.plotting import set_plot_style  # noqa: E402
from spectraxgk.runtime import (  # noqa: E402
    _build_initial_condition,
    build_runtime_geometry,
    build_runtime_linear_params,
    build_runtime_term_config,
)
from spectraxgk.runtime_config import RuntimeConfig  # noqa: E402
from spectraxgk.terms.assembly import compute_fields_cached  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "examples" / "benchmarks" / "runtime_w7x_zonal_response_vmec.toml"
DEFAULT_OUT = ROOT / "docs" / "_static" / "w7x_zonal_state_convention_audit.png"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--kx", type=float, default=0.07, help="Target zonal kx rho_i.")
    parser.add_argument("--ky", type=float, default=0.0, help="Target zonal ky rho_i.")
    parser.add_argument("--Nl", type=int, default=None, help="Laguerre moment count. Defaults to TOML run.Nl.")
    parser.add_argument("--Nm", type=int, default=None, help="Hermite moment count. Defaults to TOML run.Nm.")
    parser.add_argument("--out-png", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
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


def _selected_indices(grid: SpectralGrid, *, ky_target: float, kx_target: float) -> tuple[int, int]:
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
    envelope = float(cfg.init.gaussian_envelope_constant) + float(cfg.init.gaussian_envelope_sine) * np.sin(z - center)
    return float(cfg.init.init_amp) * envelope * np.exp(-((z - center) / width) ** 2)


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
    ky_index, kx_index = _selected_indices(grid, ky_target=ky_target, kx_target=kx_target)

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
    fields = compute_fields_cached(jnp.asarray(g0), cache, params, terms=terms, use_custom_vjp=False)
    phi = np.asarray(fields.phi)
    target_phi = np.asarray(phi[ky_index, kx_index, :])
    expected = _expected_paper_phi(cfg, np.asarray(grid.z, dtype=float))
    vol_fac, _flux_fac = gx_volume_factors(geom_eff, grid)
    vol_np = np.asarray(vol_fac, dtype=float)
    line_helper = np.asarray(gx_phi_zonal_line_kxt(jnp.asarray(phi), grid))
    mode_helper = np.asarray(gx_phi_zonal_mode_kxt(jnp.asarray(phi), grid, vol_fac))
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
        "profile_max_relative_error": float(np.max(np.abs(target_phi.real - expected)) / max(float(np.max(np.abs(expected))), 1.0e-300)),
        "profile_imag_abs_max": float(np.max(np.abs(target_phi.imag))),
        "non_target_phi_abs_max": non_target_max,
        "non_target_phi_over_target": float(non_target_max / target_amp),
        "line_helper_vs_manual_rel": _relative_abs(line_value, manual_line),
        "mode_helper_vs_manual_rel": _relative_abs(mode_value, manual_mode),
        "line_first_initial_over_init_amp": float(abs(manual_line) / max(abs(float(cfg.init.init_amp)), 1.0e-300)),
        "volume_initial_over_init_amp": float(abs(manual_mode) / max(abs(float(cfg.init.init_amp)), 1.0e-300)),
        "line_vs_volume_relative_difference": _relative_abs(line_value, mode_value),
        "density_seed_max_over_init_amp": float(np.max(np.abs(density_seed)) / max(abs(float(cfg.init.init_amp)), 1.0e-300)),
        "paper_initializer": "phi(theta)=init_amp*exp[-theta^2/gaussian_width^2] for ky=0",
        "paper_observable": "unweighted line average Phi_zonal_line_kxt normalized by its first nonzero sample",
    }
    passed = (
        float(row["profile_relative_l2"]) <= 1.0e-4
        and float(row["profile_imag_abs_max"]) <= 1.0e-8 * max(abs(float(cfg.init.init_amp)), 1.0e-300)
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


def audit_figure(audit: dict[str, object]) -> plt.Figure:
    """Create the W7-X state-convention audit figure."""

    row = dict(audit["row"])  # type: ignore[arg-type]
    z = np.asarray(audit["z"], dtype=float)
    phi = np.asarray(audit["phi_real"], dtype=float)
    expected = np.asarray(audit["expected_phi"], dtype=float)
    scale = max(float(np.max(np.abs(expected))), 1.0e-300)

    set_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.8), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(z, expected / scale, color="#111827", linewidth=2.5, label=r"paper $e^{-\theta^2}$")
    ax.plot(z, phi / scale, color="#0f4c81", linestyle="--", linewidth=2.0, label="recovered field solve")
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
    ax.bar([label for _key, label in keys], [float(row[key]) for key, _label in keys], color=["#0f4c81", "#2a9d8f", "#c2410c"])
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
    ax.text(0.04, 0.96, summary, transform=ax.transAxes, va="top", ha="left", fontsize=11)
    ax.set_axis_off()
    fig.suptitle("W7-X test-4 zonal state and observable convention audit", y=1.02, fontsize=14, fontweight="bold")
    return fig


def write_outputs(audit: dict[str, object], *, out_png: Path, out_csv: Path, out_json: Path, config: Path) -> None:
    row = dict(audit["row"])  # type: ignore[arg-type]
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig = audit_figure(audit)
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)
    payload = {
        "case": "w7x_zonal_state_convention_audit",
        "validation_status": "state_convention_closed" if bool(audit["passed"]) else "open",
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
    out_json.write_text(json.dumps(_json_clean(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg, raw = load_runtime_from_toml(args.config)
    run_cfg = dict(raw.get("run", {}))
    nl = int(args.Nl) if args.Nl is not None else int(run_cfg.get("Nl", 8))
    nm = int(args.Nm) if args.Nm is not None else int(run_cfg.get("Nm", 32))
    audit = build_state_audit(cfg, kx_target=float(args.kx), ky_target=float(args.ky), Nl=nl, Nm=nm)
    out_csv = args.out_csv or args.out_png.with_suffix(".csv")
    out_json = args.out_json or args.out_png.with_suffix(".json")
    write_outputs(audit, out_png=args.out_png, out_csv=out_csv, out_json=out_json, config=args.config)
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_json}")
    print(f"Wrote {args.out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
