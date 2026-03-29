#!/usr/bin/env python3
"""Shared helpers for config-backed example drivers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from spectraxgk.io import load_runtime_from_toml
from spectraxgk.runtime import run_runtime_linear, run_runtime_nonlinear

_FIT_KEYS = {
    "auto_window",
    "tmin",
    "tmax",
    "window_fraction",
    "min_points",
    "start_fraction",
    "growth_weight",
    "require_positive",
    "min_amp_fraction",
    "mode_method",
    "fit_signal",
}


def _load_runtime_case(config_path: str | Path) -> tuple[Any, dict[str, Any]]:
    cfg, raw = load_runtime_from_toml(config_path)
    return cfg, raw


def run_linear_case(
    config_path: str | Path,
    *,
    ky: float | None = None,
    Nl: int | None = None,
    Nm: int | None = None,
    solver: str | None = None,
    method: str | None = None,
    dt: float | None = None,
    steps: int | None = None,
    sample_stride: int | None = None,
) -> int:
    cfg, raw = _load_runtime_case(config_path)
    run_cfg = dict(raw.get("run", {}))
    fit_cfg = {k: v for k, v in raw.get("fit", {}).items() if k in _FIT_KEYS}

    result = run_runtime_linear(
        cfg,
        ky_target=float(ky if ky is not None else run_cfg.get("ky", 0.3)),
        Nl=int(Nl if Nl is not None else run_cfg.get("Nl", 24)),
        Nm=int(Nm if Nm is not None else run_cfg.get("Nm", 12)),
        solver=str(solver if solver is not None else run_cfg.get("solver", "auto")),
        method=method if method is not None else run_cfg.get("method", None),
        dt=dt if dt is not None else run_cfg.get("dt", None),
        steps=steps if steps is not None else run_cfg.get("steps", None),
        sample_stride=sample_stride if sample_stride is not None else raw.get("time", {}).get("sample_stride", None),
        **fit_cfg,
    )
    print(f"ky={result.ky:.6f} gamma={result.gamma:.8f} omega={result.omega:.8f}")
    return 0


def run_nonlinear_case(
    config_path: str | Path,
    *,
    ky: float | None = None,
    Nl: int | None = None,
    Nm: int | None = None,
    method: str | None = None,
    dt: float | None = None,
    steps: int | None = None,
    sample_stride: int | None = None,
    diagnostics_stride: int | None = None,
) -> int:
    cfg, raw = _load_runtime_case(config_path)
    run_cfg = dict(raw.get("run", {}))
    time_cfg = dict(raw.get("time", {}))

    result = run_runtime_nonlinear(
        cfg,
        ky_target=float(ky if ky is not None else run_cfg.get("ky", 0.3)),
        Nl=int(Nl if Nl is not None else run_cfg.get("Nl", 4)),
        Nm=int(Nm if Nm is not None else run_cfg.get("Nm", 8)),
        method=method if method is not None else run_cfg.get("method", None),
        dt=dt if dt is not None else time_cfg.get("dt", None),
        steps=steps if steps is not None else run_cfg.get("steps", None),
        sample_stride=sample_stride if sample_stride is not None else time_cfg.get("sample_stride", None),
        diagnostics_stride=(
            diagnostics_stride if diagnostics_stride is not None else time_cfg.get("diagnostics_stride", None)
        ),
        diagnostics=True,
    )
    if result.diagnostics is None or result.ky_selected is None:
        print("completed without streamed diagnostics")
        return 0
    diag = result.diagnostics
    print(
        "ky={:.6f} Wg={:.8e} Wphi={:.8e} heat={:.8e} pflux={:.8e}".format(
            float(result.ky_selected),
            float(diag.Wg_t[-1]),
            float(diag.Wphi_t[-1]),
            float(diag.heat_flux_t[-1]),
            float(diag.particle_flux_t[-1]),
        )
    )
    return 0
