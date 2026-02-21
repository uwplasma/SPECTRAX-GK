#!/usr/bin/env python3
"""Profile Cyclone time integration with diffrax (adaptive) and XLA dumps.

Example:
  python tools/profile_cyclone.py --trace-dir /tmp/spectrax_trace \
    --xla-dump-dir /tmp/spectrax_xla --t-max 10 --dt 0.01 --Nl 16 --Nm 48
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import replace


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile Cyclone time integration.")
    parser.add_argument("--ky", type=float, default=0.3)
    parser.add_argument("--Nl", type=int, default=16)
    parser.add_argument("--Nm", type=int, default=48)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--t-max", type=float, default=50.0)
    parser.add_argument("--use-diffrax", action="store_true", default=True)
    parser.add_argument("--no-diffrax", dest="use_diffrax", action="store_false")
    parser.add_argument("--solver", type=str, default="Dopri8")
    parser.add_argument("--adaptive", action="store_true", default=True)
    parser.add_argument("--no-adaptive", dest="adaptive", action="store_false")
    parser.add_argument("--rtol", type=float, default=1.0e-6)
    parser.add_argument("--atol", type=float, default=1.0e-8)
    parser.add_argument("--max-steps", type=int, default=200000)
    parser.add_argument("--progress-bar", action="store_true", default=False)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--trace-dir", type=str, default=None)
    parser.add_argument("--memory-profile", type=str, default=None)
    parser.add_argument("--xla-dump-dir", type=str, default=None)
    parser.add_argument("--xla-hlo-pass-re", type=str, default=".*")
    parser.add_argument("--warmup-only", action="store_true", default=False)
    return parser.parse_args()


def _configure_xla(args: argparse.Namespace) -> None:
    if not args.xla_dump_dir:
        return
    flags = os.environ.get("XLA_FLAGS", "")
    dump_flags = [
        f"--xla_dump_to={args.xla_dump_dir}",
        "--xla_dump_hlo_as_text",
        f"--xla_dump_hlo_pass_re={args.xla_hlo_pass_re}",
    ]
    combined = " ".join([flags] + dump_flags).strip()
    os.environ["XLA_FLAGS"] = combined


def main() -> None:
    args = _parse_args()
    _configure_xla(args)

    import jax
    import numpy as np
    from jax import profiler

    from spectraxgk.benchmarks import load_cyclone_reference, run_cyclone_linear
    from spectraxgk.config import CycloneBaseCase

    cfg = CycloneBaseCase()
    time_cfg = replace(
        cfg.time,
        use_diffrax=args.use_diffrax,
        diffrax_solver=args.solver,
        diffrax_adaptive=args.adaptive,
        diffrax_rtol=args.rtol,
        diffrax_atol=args.atol,
        diffrax_max_steps=args.max_steps,
        progress_bar=args.progress_bar,
        dt=args.dt,
        t_max=args.t_max,
        sample_stride=args.sample_stride,
    )

    def _run():
        return run_cyclone_linear(
            cfg=cfg,
            time_cfg=time_cfg,
            ky_target=args.ky,
            Nl=args.Nl,
            Nm=args.Nm,
            solver="time",
        )

    # Warmup: compile + run
    t0 = time.perf_counter()
    res = _run()
    t1 = time.perf_counter()

    if args.warmup_only:
        print(f"warmup_time_s={t1 - t0:.3f}")
        return

    # Profiled run (runtime only)
    if args.trace_dir:
        profiler.start_trace(args.trace_dir)
    t2 = time.perf_counter()
    res2 = _run()
    t3 = time.perf_counter()
    if args.trace_dir:
        profiler.stop_trace()

    if args.memory_profile:
        profiler.save_device_memory_profile(args.memory_profile)

    ref = load_cyclone_reference()
    idx = int(np.argmin(np.abs(ref.ky - args.ky)))

    print(
        "warmup_time_s={:.3f} run_time_s={:.3f}".format(t1 - t0, t3 - t2)
    )
    print(
        "gamma={:.6f} omega={:.6f} ref_gamma={:.6f} ref_omega={:.6f}".format(
            res.gamma, res.omega, ref.gamma[idx], ref.omega[idx]
        )
    )


if __name__ == "__main__":
    main()
