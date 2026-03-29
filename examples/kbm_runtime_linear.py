#!/usr/bin/env python3
"""Run the config-backed KBM linear example."""

from __future__ import annotations

import argparse
from pathlib import Path

from _runtime_case import run_linear_case

CONFIG = Path(__file__).resolve().parent / "configs" / "runtime_kbm.toml"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ky", type=float, default=None)
    p.add_argument("--Nl", type=int, default=None)
    p.add_argument("--Nm", type=int, default=None)
    p.add_argument("--solver", type=str, default=None)
    args = p.parse_args()
    return run_linear_case(CONFIG, ky=args.ky, Nl=args.Nl, Nm=args.Nm, solver=args.solver)


if __name__ == "__main__":
    raise SystemExit(main())
