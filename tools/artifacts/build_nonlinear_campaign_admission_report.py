#!/usr/bin/env python3
"""Build the next-campaign admission gate for nonlinear optimization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from spectraxgk.diagnostics.stellarator_transport_reports import (  # noqa: E402
    build_nonlinear_campaign_admission_report,
)
from spectraxgk.objectives.vmec_transport_admission import (  # noqa: E402
    VMECJAXNonlinearCampaignPolicy,
)


DEFAULT_PRELAUNCH = (
    ROOT / "docs/_static/vmec_boundary_transport_landscape_p0p03_prelaunch_gate.json"
)
DEFAULT_LANDSCAPE = (
    ROOT / "docs/_static/vmec_boundary_transport_landscape_admission.json"
)
DEFAULT_OUT = ROOT / "docs/_static/nonlinear_campaign_admission_report.json"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def build_report(
    *,
    prelaunch_report: Path,
    landscape_admission: Path,
    policy: VMECJAXNonlinearCampaignPolicy,
) -> dict[str, Any]:
    """Load compact JSON artifacts and build the campaign-admission report."""

    report = build_nonlinear_campaign_admission_report(
        reduced_prelaunch_report=_load_json(prelaunch_report),
        landscape_admission_report=_load_json(landscape_admission),
        policy=policy,
    )
    report["artifacts"] = {
        "reduced_prelaunch_report": _repo_relative(prelaunch_report),
        "landscape_admission_report": _repo_relative(landscape_admission),
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prelaunch-report", type=Path, default=DEFAULT_PRELAUNCH)
    parser.add_argument("--landscape-admission", type=Path, default=DEFAULT_LANDSCAPE)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-landscape-relative-reduction", type=float, default=0.10)
    parser.add_argument("--min-landscape-z-score", type=float, default=3.0)
    parser.add_argument("--max-landscape-sem-rel", type=float, default=0.05)
    parser.add_argument("--min-landscape-replicates", type=int, default=3)
    parser.add_argument(
        "--allow-missing-cross-sample-gate",
        action="store_true",
        help="Do not block when older prelaunch artifacts lack reduced cross-sample statistics.",
    )
    parser.add_argument("--fail-on-blocked", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    policy = VMECJAXNonlinearCampaignPolicy(
        minimum_landscape_relative_reduction=float(
            args.min_landscape_relative_reduction
        ),
        minimum_landscape_uncertainty_z_score=float(args.min_landscape_z_score),
        maximum_landscape_sem_rel=float(args.max_landscape_sem_rel),
        minimum_landscape_replicate_count=int(args.min_landscape_replicates),
        require_reduced_cross_sample_gate=not bool(
            args.allow_missing_cross_sample_gate
        ),
    )
    report = build_report(
        prelaunch_report=args.prelaunch_report,
        landscape_admission=args.landscape_admission,
        policy=policy,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "campaign_admitted": report["campaign_admitted"],
                "selected_label": (
                    (report.get("selected_landscape_candidate") or {}).get("label")
                    if isinstance(report.get("selected_landscape_candidate"), dict)
                    else None
                ),
                "blockers": report["blockers"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.fail_on_blocked and not bool(report["campaign_admitted"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
