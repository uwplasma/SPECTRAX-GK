#!/usr/bin/env python3
"""Build VMEC/Boozer-state gradient gate artifacts.

This family command owns the frequency, quasilinear, and reduced nonlinear-window
VMEC/Boozer gradient gates.  The three gates share the same VMEC/Boozer state
setup, finite-difference controls, artifact writer, and claim boundary; only the
observable family and a few tolerances differ.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Literal

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectraxgk.objectives.vmec_boozer_gradients import (  # noqa: E402
    mode21_vmec_boozer_linear_frequency_gradient_report,
    mode21_vmec_boozer_nonlinear_window_gradient_report,
    mode21_vmec_boozer_quasilinear_gradient_report,
)
from tools.artifacts.build_solver_objective_gradient_gate import (  # noqa: E402
    _json_clean,
    write_solver_objective_gradient_artifacts,
)

GradientKind = Literal["frequency", "quasilinear", "nonlinear-window"]
_REPORT_BUILDER_FUNCTIONS = (
    mode21_vmec_boozer_linear_frequency_gradient_report,
    mode21_vmec_boozer_quasilinear_gradient_report,
    mode21_vmec_boozer_nonlinear_window_gradient_report,
)

DEFAULT_OUTS: dict[GradientKind, Path] = {
    "frequency": ROOT
    / "docs"
    / "_static"
    / "vmec_boozer_solver_frequency_gradient_gate.png",
    "quasilinear": ROOT
    / "docs"
    / "_static"
    / "vmec_boozer_quasilinear_gradient_gate.png",
    "nonlinear-window": ROOT
    / "docs"
    / "_static"
    / "vmec_boozer_nonlinear_window_gradient_gate.png",
}

DEFAULT_RTOL: dict[GradientKind, float] = {
    "frequency": 5.0e-2,
    "quasilinear": 2.0e-2,
    "nonlinear-window": 7.5e-2,
}
DEFAULT_ATOL: dict[GradientKind, float] = {
    "frequency": 2.0e-2,
    "quasilinear": 5.0e-2,
    "nonlinear-window": 5.0e-2,
}
REPORT_BUILDERS: dict[GradientKind, str] = {
    "frequency": "mode21_vmec_boozer_linear_frequency_gradient_report",
    "quasilinear": "mode21_vmec_boozer_quasilinear_gradient_report",
    "nonlinear-window": "mode21_vmec_boozer_nonlinear_window_gradient_report",
}


def _add_common_args(parser: argparse.ArgumentParser, *, kind: GradientKind) -> None:
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTS[kind])
    parser.add_argument("--case-name", default="nfp4_QH_warm_start")
    parser.add_argument("--radial-index", type=int, default=None)
    parser.add_argument("--mode-index", type=int, default=1)
    parser.add_argument("--parameter-family", default="Rcos")
    parser.add_argument("--surface-index", type=int, default=None)
    parser.add_argument("--fd-step", type=float, default=1.0e-6)
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL[kind])
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL[kind])
    parser.add_argument("--ntheta", type=int, default=4)
    parser.add_argument("--mboz", type=int, default=21)
    parser.add_argument("--nboz", type=int, default=21)
    parser.add_argument(
        "--surface-stencil-width",
        type=int,
        default=0,
        help="Boozer radial stencil width; 0 transforms all radial surfaces.",
    )
    parser.add_argument("--json-only", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="kind", required=True)
    for kind, help_text in (
        ("frequency", "Build the solver-frequency gradient gate."),
        ("quasilinear", "Build the quasilinear-gradient gate."),
        (
            "nonlinear-window",
            "Build the reduced nonlinear-window-gradient gate.",
        ),
    ):
        subparser = subparsers.add_parser(kind, help=help_text)
        _add_common_args(subparser, kind=kind)  # type: ignore[arg-type]
        if kind == "nonlinear-window":
            subparser.add_argument("--nonlinear-dt", type=float, default=0.18)
            subparser.add_argument("--nonlinear-steps", type=int, default=96)
            subparser.add_argument("--tail-fraction", type=float, default=0.30)
    return parser


def _payload_from_args(args: argparse.Namespace) -> dict[str, Any]:
    kind = str(args.kind)
    if kind not in REPORT_BUILDERS:
        raise ValueError(f"unsupported gradient kind {kind!r}")
    if kind == "nonlinear-window" and 0 < args.surface_stencil_width < 3:
        raise SystemExit("--surface-stencil-width must be 0 or at least 3")
    kwargs: dict[str, Any] = {
        "case_name": args.case_name,
        "radial_index": args.radial_index,
        "mode_index": args.mode_index,
        "parameter_family": args.parameter_family,
        "surface_index": args.surface_index,
        "fd_step": args.fd_step,
        "rtol": args.rtol,
        "atol": args.atol,
        "ntheta": args.ntheta,
        "mboz": args.mboz,
        "nboz": args.nboz,
        "surface_stencil_width": None
        if args.surface_stencil_width <= 0
        else args.surface_stencil_width,
    }
    if kind == "nonlinear-window":
        kwargs.update(
            nonlinear_dt=args.nonlinear_dt,
            nonlinear_steps=args.nonlinear_steps,
            tail_fraction=args.tail_fraction,
        )
    report_builder = globals()[REPORT_BUILDERS[kind]]  # type: ignore[index]
    return report_builder(**kwargs)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = _payload_from_args(args)
    if args.json_only:
        print(json.dumps(_json_clean(payload), indent=2, sort_keys=True))
        return 0
    paths = write_solver_objective_gradient_artifacts(payload, out=args.out)
    print(json.dumps(paths, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
