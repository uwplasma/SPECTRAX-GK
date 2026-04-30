#!/usr/bin/env python3
"""Build the full VMEC/Boozer-state quasilinear-gradient artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectraxgk.solver_objective_gradients import mode21_vmec_boozer_quasilinear_gradient_report  # noqa: E402
from tools.build_solver_objective_gradient_gate import _json_clean, write_solver_objective_gradient_artifacts  # noqa: E402

DEFAULT_OUT = ROOT / "docs" / "_static" / "vmec_boozer_quasilinear_gradient_gate.png"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--case-name", default="nfp4_QH_warm_start")
    parser.add_argument("--fd-step", type=float, default=1.0e-6)
    parser.add_argument("--rtol", type=float, default=2.0e-2)
    parser.add_argument("--atol", type=float, default=5.0e-2)
    parser.add_argument("--ntheta", type=int, default=4)
    parser.add_argument("--mboz", type=int, default=21)
    parser.add_argument("--nboz", type=int, default=21)
    parser.add_argument("--json-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = mode21_vmec_boozer_quasilinear_gradient_report(
        case_name=args.case_name,
        fd_step=args.fd_step,
        rtol=args.rtol,
        atol=args.atol,
        ntheta=args.ntheta,
        mboz=args.mboz,
        nboz=args.nboz,
    )
    if args.json_only:
        print(json.dumps(_json_clean(payload), indent=2, sort_keys=True))
        return 0
    paths = write_solver_objective_gradient_artifacts(payload, out=args.out)
    print(json.dumps(paths, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
