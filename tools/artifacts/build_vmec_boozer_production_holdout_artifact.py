#!/usr/bin/env python3
"""Build a production-scope VMEC/Boozer held-out nonlinear transport artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def _repo_relative(path: Path | str) -> str:
    raw = Path(path)
    try:
        return raw.resolve().relative_to(ROOT.resolve()).as_posix()
    except (OSError, ValueError):
        return str(path)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _transport_sample(manifest: dict[str, Any]) -> dict[str, Any]:
    sample = manifest.get("transport_sample")
    if not isinstance(sample, dict):
        raise ValueError("transport manifest must contain transport_sample")
    required = ("vmec_file", "torflux", "alpha", "ky")
    missing = [key for key in required if key not in sample]
    if missing:
        raise ValueError(
            "transport_sample missing required key(s): " + ", ".join(missing)
        )
    return dict(sample)


def _ensemble_passed(ensemble: dict[str, Any]) -> bool:
    if bool(ensemble.get("passed", False)):
        return True
    gate = ensemble.get("gate_report")
    return bool(isinstance(gate, dict) and gate.get("passed", False))


def _window(ensemble: dict[str, Any]) -> dict[str, Any]:
    window = ensemble.get("window")
    return dict(window) if isinstance(window, dict) else {}


def build_vmec_boozer_production_holdout_artifact(
    *,
    transport_manifest: str | Path,
    ensemble_json: str | Path,
    case: str | None = None,
) -> dict[str, Any]:
    """Return a promotion-gate input for a held-out nonlinear VMEC/Boozer audit."""

    manifest_path = Path(transport_manifest)
    ensemble_path = Path(ensemble_json)
    manifest = _load_json(manifest_path)
    ensemble = _load_json(ensemble_path)
    sample = _transport_sample(manifest)
    passed = _ensemble_passed(ensemble)
    case_name = str(
        case
        or manifest.get("case")
        or ensemble.get("case")
        or manifest_path.parent.name
    )
    holdout_sample = {
        "surface_index": None,
        "surface": float(sample["torflux"]),
        "torflux": float(sample["torflux"]),
        "alpha": float(sample["alpha"]),
        "ky": float(sample["ky"]),
        "selected_ky_index": f"ky={float(sample['ky']):.16g}",
        "vmec_file": str(sample["vmec_file"]),
        "npol": float(sample.get("npol", 1.0)),
    }
    return {
        "kind": "vmec_boozer_production_scope_heldout_nonlinear_transport_artifact",
        "case": case_name,
        "claim_level": (
            "production_scope_vmec_boozer_heldout_nonlinear_transport_average"
        ),
        "passed": passed,
        "promotion_gate": {
            "passed": passed,
            "blockers": []
            if passed
            else ["replicated_nonlinear_window_ensemble_failed"],
        },
        "transport_average_gate": passed,
        "samples": [holdout_sample],
        "holdout_samples": [holdout_sample],
        "source_manifest": _repo_relative(manifest_path),
        "nonlinear_ensemble_artifact": _repo_relative(ensemble_path),
        "nonlinear_ensemble_passed": passed,
        "window": _window(ensemble),
        "statistics": ensemble.get("statistics", {}),
        "notes": (
            "This artifact is generated only from a concrete VMEC transport manifest "
            "and a replicated post-transient nonlinear-window ensemble. It is the "
            "surface/field-line holdout companion consumed by "
            "check_vmec_boozer_aggregate_holdout_gate.py."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport-manifest", required=True, type=Path)
    parser.add_argument("--ensemble-json", required=True, type=Path)
    parser.add_argument("--case")
    parser.add_argument("--out", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact = build_vmec_boozer_production_holdout_artifact(
        transport_manifest=args.transport_manifest,
        ensemble_json=args.ensemble_json,
        case=args.case,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"saved {_repo_relative(args.out)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
