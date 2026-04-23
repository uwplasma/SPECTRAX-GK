#!/usr/bin/env python3
"""Profile nonlinear runtime startup phases and first-compile costs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
import time
from typing import Any, Callable


@dataclass(frozen=True)
class PhaseTiming:
    phase: str
    seconds: float
    note: str = ""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear.toml"),
    )
    parser.add_argument("--ky", type=float, default=0.3)
    parser.add_argument("--kx", type=float, default=None)
    parser.add_argument("--Nl", type=int, default=4)
    parser.add_argument("--Nm", type=int, default=8)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--compile-steps", type=int, default=1)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--diagnostics-stride", type=int, default=1)
    parser.add_argument("--laguerre-mode", type=str, default=None)
    parser.add_argument("--trace-dir", type=Path, default=None)
    parser.add_argument("--memory-profile", type=Path, default=None)
    parser.add_argument("--xla-dump-dir", type=Path, default=None)
    parser.add_argument("--xla-hlo-pass-re", type=str, default=".*")
    parser.add_argument("--debug-log-cache", action="store_true", default=False)
    parser.add_argument("--explain-cache-misses", action="store_true", default=False)
    parser.add_argument("--csv-out", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def _configure_env(args: argparse.Namespace) -> None:
    if args.xla_dump_dir is not None:
        flags = os.environ.get("XLA_FLAGS", "")
        dump_flags = [
            f"--xla_dump_to={args.xla_dump_dir}",
            "--xla_dump_hlo_as_text",
            f"--xla_dump_hlo_pass_re={args.xla_hlo_pass_re}",
        ]
        os.environ["XLA_FLAGS"] = " ".join([flags] + dump_flags).strip()
    if args.debug_log_cache:
        os.environ.setdefault("JAX_DEBUG_LOG_MODULES", "jax._src.compiler,jax._src.lru_cache")
        os.environ.setdefault("JAX_LOGGING_LEVEL", "DEBUG")
    if args.explain_cache_misses:
        os.environ.setdefault("JAX_EXPLAIN_CACHE_MISSES", "1")


def _block_tree(tree: Any) -> None:
    import jax

    for leaf in jax.tree_util.tree_leaves(tree):
        try:
            jax.block_until_ready(leaf)
        except TypeError:
            continue


def _time_phase(
    timings: list[PhaseTiming],
    phase: str,
    fn: Callable[[], Any],
    *,
    note: str = "",
    block: bool = False,
) -> Any:
    from jax import profiler

    t0 = time.perf_counter()
    with profiler.TraceAnnotation(phase):
        out = fn()
        if block:
            _block_tree(out)
    timings.append(PhaseTiming(phase=phase, seconds=time.perf_counter() - t0, note=note))
    return out


def _write_phase_csv(path: Path, timings: list[PhaseTiming]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["phase", "seconds", "note"])
        writer.writeheader()
        for row in timings:
            writer.writerow(asdict(row))


def _write_phase_json(path: Path, timings: list[PhaseTiming], metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata,
        "phases": [asdict(row) for row in timings],
        "startup_total_s": sum(row.seconds for row in timings),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    _configure_env(args)

    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax import profiler
    from spectraxgk.geometry import apply_gx_geometry_grid_defaults
    from spectraxgk.grids import build_spectral_grid
    from spectraxgk.io import load_runtime_from_toml
    from spectraxgk.linear import build_linear_cache
    from spectraxgk.nonlinear import integrate_nonlinear_gx_diagnostics_state, nonlinear_rhs_cached
    from spectraxgk.runtime import (
        _build_initial_condition,
        _runtime_external_phi,
        _select_nonlinear_mode_indices,
        build_runtime_geometry,
        build_runtime_linear_params,
        build_runtime_term_config,
    )
    from spectraxgk.terms.assembly import assemble_rhs_cached_jit, compute_fields_cached

    timings: list[PhaseTiming] = []

    if args.trace_dir is not None:
        args.trace_dir.mkdir(parents=True, exist_ok=True)
        profiler.start_trace(str(args.trace_dir))

    try:
        cfg_loaded = _time_phase(
            timings,
            "load_runtime_config",
            lambda: load_runtime_from_toml(args.config),
            note=str(args.config),
        )
        cfg, _ = cfg_loaded
        geom = _time_phase(timings, "build_runtime_geometry", lambda: build_runtime_geometry(cfg), block=True)
        grid_cfg = _time_phase(
            timings,
            "apply_geometry_grid_defaults",
            lambda: apply_gx_geometry_grid_defaults(geom, cfg.grid),
        )
        grid = _time_phase(timings, "build_spectral_grid", lambda: build_spectral_grid(grid_cfg), block=True)
        params = _time_phase(
            timings,
            "build_runtime_linear_params",
            lambda: build_runtime_linear_params(cfg, Nm=args.Nm, geom=geom),
            block=True,
        )
        term_cfg = _time_phase(timings, "build_runtime_term_config", lambda: build_runtime_term_config(cfg))
        external_phi = _time_phase(timings, "resolve_external_phi", lambda: _runtime_external_phi(cfg), block=True)
        ky_index, kx_index = _time_phase(
            timings,
            "select_mode_indices",
            lambda: _select_nonlinear_mode_indices(
                grid,
                ky_target=args.ky,
                kx_target=args.kx,
                use_dealias_mask=bool(cfg.time.nonlinear_dealias),
            ),
        )
        laguerre_mode = cfg.time.laguerre_nonlinear_mode if args.laguerre_mode is None else str(args.laguerre_mode)
        G0 = _time_phase(
            timings,
            "build_initial_condition",
            lambda: _build_initial_condition(
                grid,
                geom,
                cfg,
                ky_index=ky_index,
                kx_index=kx_index,
                Nl=args.Nl,
                Nm=args.Nm,
                nspecies=len(cfg.species),
            ),
            block=True,
            note=f"ky_index={int(ky_index)} kx_index={int(kx_index)}",
        )
        cache = _time_phase(
            timings,
            "build_linear_cache",
            lambda: build_linear_cache(grid, geom, params, args.Nl, args.Nm),
            block=True,
        )

        G0 = jnp.asarray(G0)
        linear_term_cfg = replace(term_cfg, nonlinear=0.0)
        field_fn = jax.jit(lambda G: compute_fields_cached(G, cache, params, terms=term_cfg, external_phi=external_phi))
        linear_rhs_fn = jax.jit(
            lambda G: assemble_rhs_cached_jit(G, cache, params, linear_term_cfg, external_phi=external_phi)
        )
        full_rhs_fn = jax.jit(
            lambda G: nonlinear_rhs_cached(
                G,
                cache,
                params,
                term_cfg,
                gx_real_fft=bool(cfg.time.gx_real_fft),
                laguerre_mode=laguerre_mode,
                external_phi=external_phi,
            )
        )

        fields = _time_phase(
            timings,
            "compile_first_field_solve",
            lambda: field_fn(G0),
            block=True,
            note="jit+execute",
        )
        linear_rhs_out = _time_phase(
            timings,
            "compile_first_linear_rhs",
            lambda: linear_rhs_fn(G0),
            block=True,
            note="jit+execute",
        )
        full_rhs_out = _time_phase(
            timings,
            "compile_first_full_rhs",
            lambda: full_rhs_fn(G0),
            block=True,
            note="jit+execute",
        )

        dt_val = float(cfg.time.dt if args.dt is None else args.dt)
        if dt_val <= 0.0:
            raise ValueError("dt must be > 0")
        compile_steps = int(args.compile_steps)
        if compile_steps < 1:
            raise ValueError("--compile-steps must be >= 1")
        integrator_out = _time_phase(
            timings,
            "compile_first_integrator_run",
            lambda: integrate_nonlinear_gx_diagnostics_state(
                G0,
                grid,
                geom,
                params,
                dt=dt_val,
                steps=compile_steps,
                method=str(args.method or cfg.time.method),
                cache=cache,
                terms=term_cfg,
                sample_stride=int(args.sample_stride),
                diagnostics_stride=int(args.diagnostics_stride),
                use_dealias_mask=bool(cfg.time.nonlinear_dealias),
                gx_real_fft=bool(cfg.time.gx_real_fft),
                laguerre_mode=laguerre_mode,
                omega_ky_index=int(ky_index),
                omega_kx_index=int(kx_index),
                flux_scale=float(cfg.normalization.flux_scale),
                wphi_scale=float(cfg.normalization.wphi_scale),
                fixed_dt=bool(cfg.time.fixed_dt),
                dt_min=float(cfg.time.dt_min),
                dt_max=cfg.time.dt_max,
                cfl=float(cfg.time.cfl),
                cfl_fac=cfg.time.cfl_fac,
                collision_split=bool(cfg.time.collision_split),
                collision_scheme=str(cfg.time.collision_scheme),
                implicit_restart=int(cfg.time.implicit_restart),
                implicit_solve_method=str(cfg.time.implicit_solve_method),
                implicit_preconditioner=cfg.time.implicit_preconditioner,
                external_phi=external_phi,
                show_progress=False,
            ),
            block=True,
            note=f"steps={compile_steps}",
        )

        metadata = {
            "config": str(args.config),
            "ky": args.ky,
            "kx": args.kx,
            "Nl": args.Nl,
            "Nm": args.Nm,
            "dt": dt_val,
            "compile_steps": compile_steps,
            "method": str(args.method or cfg.time.method),
            "platform": os.environ.get("JAX_PLATFORM_NAME", ""),
            "device_count": int(jax.device_count()),
            "field_norm": float(np.asarray(jnp.linalg.norm(fields.phi))),
            "linear_rhs_norm": float(np.asarray(jnp.linalg.norm(linear_rhs_out[0]))),
            "full_rhs_norm": float(np.asarray(jnp.linalg.norm(full_rhs_out[0]))),
            "integrator_samples": int(np.asarray(integrator_out[0]).size),
        }
        total = sum(row.seconds for row in timings)
        for row in timings:
            note = f" {row.note}" if row.note else ""
            print(f"{row.phase}: seconds={row.seconds:.6f}{note}")
        print(f"startup_total_s={total:.6f}")

        if args.csv_out is not None:
            _write_phase_csv(args.csv_out, timings)
        if args.json_out is not None:
            _write_phase_json(args.json_out, timings, metadata)
        if args.memory_profile is not None:
            with profiler.TraceAnnotation("spectrax_memory_snapshot"):
                profiler.save_device_memory_profile(str(args.memory_profile))
    finally:
        if args.trace_dir is not None:
            profiler.stop_trace()


if __name__ == "__main__":
    main()
