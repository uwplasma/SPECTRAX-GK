#!/usr/bin/env python3
"""Profile nonlinear Cyclone runs with optional XLA + Perfetto traces.

Example:
  python tools/profile_nonlinear_cyclone.py --trace-dir /tmp/spectrax_nl_trace \
    --xla-dump-dir /tmp/spectrax_nl_xla --steps 400 --dt 0.0377 --Nl 4 --Nm 8
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile nonlinear Cyclone runtime.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("examples/linear/axisymmetric/runtime_cyclone_nonlinear.toml"),
    )
    parser.add_argument("--ky", type=float, default=0.3)
    parser.add_argument("--Nl", type=int, default=4)
    parser.add_argument("--Nm", type=int, default=8)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--sample-stride", type=int, default=None)
    parser.add_argument("--diagnostics-stride", type=int, default=None)
    parser.add_argument("--trace-dir", type=Path, default=None)
    parser.add_argument("--memory-profile", type=Path, default=None)
    parser.add_argument("--xla-dump-dir", type=Path, default=None)
    parser.add_argument("--xla-hlo-pass-re", type=str, default=".*")
    parser.add_argument("--warmup-only", action="store_true", default=False)
    return parser.parse_args()


def _configure_xla(args: argparse.Namespace) -> None:
    if args.xla_dump_dir is None:
        return
    flags = os.environ.get("XLA_FLAGS", "")
    dump_flags = [
        f"--xla_dump_to={args.xla_dump_dir}",
        "--xla_dump_hlo_as_text",
        f"--xla_dump_hlo_pass_re={args.xla_hlo_pass_re}",
    ]
    os.environ["XLA_FLAGS"] = " ".join([flags] + dump_flags).strip()


def main() -> None:
    args = _parse_args()
    _configure_xla(args)

    from jax import profiler

    from spectraxgk.runtime import run_runtime_nonlinear
    from spectraxgk.io import load_runtime_from_toml

    cfg, _data = load_runtime_from_toml(args.config)

    def _run():
        return run_runtime_nonlinear(
            cfg,
            ky_target=args.ky,
            Nl=args.Nl,
            Nm=args.Nm,
            dt=args.dt,
            steps=args.steps,
            method=args.method,
            sample_stride=args.sample_stride,
            diagnostics_stride=args.diagnostics_stride,
            diagnostics=True,
        )

    t0 = time.perf_counter()
    _run()
    t1 = time.perf_counter()

    if args.warmup_only:
        print(f"warmup_time_s={t1 - t0:.3f}")
        return

    if args.trace_dir is not None:
        args.trace_dir.mkdir(parents=True, exist_ok=True)
        profiler.start_trace(str(args.trace_dir))
    t2 = time.perf_counter()
    _run()
    t3 = time.perf_counter()
    if args.trace_dir is not None:
        profiler.stop_trace()

    if args.memory_profile is not None:
        profiler.save_device_memory_profile(str(args.memory_profile))

    print(f"warmup_time_s={t1 - t0:.3f} run_time_s={t3 - t2:.3f}")


if __name__ == "__main__":
    main()
