#!/usr/bin/env python3
"""Build a fail-closed nonlinear-audit redesign report.

The report connects a matched replicated nonlinear transport comparison to the
next reduced-objective design decision.  If a local reduced-metric candidate
does not transfer to late-window nonlinear heat flux, the output records the
blockers and the recommended multi-surface, multi-field-line, multi-ky sample
set for the next VMEC-JAX transport objective.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from spectraxgk.validation.stellarator.transport_audit import (  # noqa: E402
    build_nonlinear_audit_redesign_report,
)
from spectraxgk.validation.stellarator.transport_policies import (  # noqa: E402
    VMECJAXNonlinearAuditPolicy,
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _sample_set_from_args(args: argparse.Namespace) -> dict[str, list[float]] | None:
    if args.sample_set_json is not None:
        payload = _load_json(args.sample_set_json)
        return {
            "surfaces": [float(item) for item in payload.get("surfaces", ())],
            "alphas": [float(item) for item in payload.get("alphas", ())],
            "ky_values": [float(item) for item in payload.get("ky_values", ())],
        }
    if args.surface or args.alpha or args.ky:
        return {
            "surfaces": [float(item) for item in args.surface],
            "alphas": [float(item) for item in args.alpha],
            "ky_values": [float(item) for item in args.ky],
        }
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched-comparison", type=Path, required=True)
    parser.add_argument("--sample-set-json", type=Path)
    parser.add_argument("--surface", type=float, action="append", default=[])
    parser.add_argument("--alpha", type=float, action="append", default=[])
    parser.add_argument("--ky", type=float, action="append", default=[])
    parser.add_argument("--minimum-relative-reduction", type=float, default=0.02)
    parser.add_argument("--minimum-uncertainty-z-score", type=float, default=1.0)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--fail-on-redesign", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    policy = VMECJAXNonlinearAuditPolicy(
        minimum_relative_reduction=float(args.minimum_relative_reduction),
        minimum_uncertainty_z_score=float(args.minimum_uncertainty_z_score),
    )
    report = build_nonlinear_audit_redesign_report(
        _load_json(args.matched_comparison),
        objective_sample_set=_sample_set_from_args(args),
        policy=policy,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "nonlinear_audit_promoted": report["nonlinear_audit_promoted"],
                "requires_objective_redesign": report["requires_objective_redesign"],
                "blockers": report["blockers"],
                "recommended_sample_set": report["recommended_sample_set"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.fail_on_redesign and bool(report["requires_objective_redesign"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
