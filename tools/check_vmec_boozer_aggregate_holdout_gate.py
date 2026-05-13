#!/usr/bin/env python3
"""Check held-out surface/field-line evidence for VMEC/Boozer aggregate optimization."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AGGREGATE_ARTIFACT = ROOT / "docs/_static/vmec_boozer_aggregate_objective_gate.json"
DEFAULT_LINE_SEARCH_ARTIFACT = ROOT / "docs/_static/vmec_boozer_aggregate_line_search_gate.json"

NON_PROMOTABLE_CLAIM_MARKERS = (
    "not_transport",
    "not transport",
    "not_production",
    "not production",
    "not a nonlinear",
    "startup",
    "reduced",
    "plumbing",
    "exploratory",
    "feasibility",
    "pending",
    "negative",
)


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return payload


def _artifact_passed(payload: dict[str, Any]) -> bool:
    if bool(payload.get("passed", False)):
        return True
    if bool(payload.get("gate_passed", False)):
        return True
    for key in ("promotion_gate", "gate_report"):
        nested = payload.get(key)
        if isinstance(nested, dict) and bool(nested.get("passed", False)):
            return True
    return False


def _claim_scope_blocks_promotion(payload: dict[str, Any]) -> list[str]:
    claim_text = " ".join(
        str(payload.get(key, ""))
        for key in ("claim_level", "claim_scope", "notes", "next_action")
    ).lower()
    blockers = [marker for marker in NON_PROMOTABLE_CLAIM_MARKERS if marker in claim_text]
    if payload.get("transport_average_gate") is False:
        blockers.append("transport_average_gate_false")
    return sorted(set(blockers))


def _samples(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("samples", "holdout_samples", "validation_samples"):
        raw = payload.get(key)
        if isinstance(raw, list):
            return [item for item in raw if isinstance(item, dict)]
    return []


def _alpha(sample: dict[str, Any]) -> float | None:
    value = sample.get("alpha")
    try:
        alpha = float(value)
    except (TypeError, ValueError):
        return None
    return alpha if math.isfinite(alpha) else None


def _sample_identity(sample: dict[str, Any]) -> tuple[str, str, str]:
    surface = sample.get("surface_index")
    ky = sample.get("selected_ky_index")
    alpha = _alpha(sample)
    alpha_key = "" if alpha is None else f"{alpha:.16g}"
    return (str(surface), alpha_key, str(ky))


def _sample_set(payload: dict[str, Any]) -> set[tuple[str, str, str]]:
    return {_sample_identity(sample) for sample in _samples(payload)}


def _has_heldout_surface_or_field_line(
    training_samples: list[dict[str, Any]],
    holdout_samples: list[dict[str, Any]],
    *,
    alpha_atol: float,
) -> tuple[bool, str]:
    training_surfaces = {sample.get("surface_index") for sample in training_samples}
    training_alphas = [alpha for sample in training_samples if (alpha := _alpha(sample)) is not None]
    for sample in holdout_samples:
        surface = sample.get("surface_index")
        if surface is not None and surface not in training_surfaces:
            return True, f"held-out surface_index={surface}"
        alpha = _alpha(sample)
        if alpha is not None and all(abs(alpha - item) > alpha_atol for item in training_alphas):
            return True, f"held-out field-line alpha={alpha:.16g}"
    return False, "no passed holdout sample changes surface_index or field-line alpha"


def _gate(metric: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"metric": metric, "passed": bool(passed), "detail": detail}


def check_vmec_boozer_aggregate_holdout_gate(
    *,
    aggregate_artifact: str | Path = DEFAULT_AGGREGATE_ARTIFACT,
    line_search_artifact: str | Path = DEFAULT_LINE_SEARCH_ARTIFACT,
    holdout_artifacts: tuple[str | Path, ...] = (),
    alpha_atol: float = 1.0e-12,
) -> dict[str, Any]:
    """Return a JSON-ready promotion gate for aggregate optimization artifacts."""

    aggregate_path = Path(aggregate_artifact)
    line_search_path = Path(line_search_artifact)
    aggregate = _load_json_object(aggregate_path)
    line_search = _load_json_object(line_search_path)
    training_samples = _samples(aggregate)
    line_search_same_samples = bool(training_samples) and _sample_set(aggregate) == _sample_set(line_search)

    holdout_rows: list[dict[str, Any]] = []
    qualifying_holdout_reasons: list[str] = []
    for raw_path in holdout_artifacts:
        path = Path(raw_path)
        payload = _load_json_object(path)
        samples = _samples(payload)
        passed = _artifact_passed(payload)
        scope_blockers = _claim_scope_blocks_promotion(payload)
        has_holdout_sample, reason = _has_heldout_surface_or_field_line(
            training_samples,
            samples,
            alpha_atol=alpha_atol,
        )
        qualifies = bool(passed and not scope_blockers and has_holdout_sample)
        if qualifies:
            qualifying_holdout_reasons.append(f"{_repo_relative(path)}: {reason}")
        holdout_rows.append(
            {
                "path": _repo_relative(path),
                "passed": passed,
                "claim_scope_blockers": scope_blockers,
                "n_samples": len(samples),
                "heldout_surface_or_field_line": has_holdout_sample,
                "heldout_reason": reason,
                "qualifies_for_promotion": qualifies,
            }
        )

    gates = [
        _gate(
            "aggregate_finite_difference_artifact_passed",
            _artifact_passed(aggregate),
            _repo_relative(aggregate_path),
        ),
        _gate(
            "aggregate_line_search_artifact_passed",
            _artifact_passed(line_search),
            _repo_relative(line_search_path),
        ),
        _gate(
            "line_search_reuses_aggregate_sample_set",
            line_search_same_samples,
            "line-search samples must match the aggregate objective samples",
        ),
        _gate(
            "passed_holdout_surface_or_field_line_artifact",
            bool(qualifying_holdout_reasons),
            "; ".join(qualifying_holdout_reasons)
            if qualifying_holdout_reasons
            else "provide a passed holdout artifact with a new surface_index or alpha",
        ),
    ]
    blockers = [gate["metric"] for gate in gates if not bool(gate["passed"])]
    passed = not blockers
    return {
        "kind": "vmec_boozer_aggregate_holdout_promotion_gate",
        "claim_level": (
            "aggregate_optimization_promotion_requires_heldout_surface_or_field_line_validation"
        ),
        "passed": passed,
        "promotion_gate": {
            "passed": passed,
            "blockers": blockers,
            "requirements": [
                "aggregate finite-difference artifact passes",
                "aggregate line-search artifact passes on the same sample set",
                "at least one passed validation artifact covers a held-out surface_index or field-line alpha",
                "k_y-only holdouts do not satisfy the surface/field-line requirement",
            ],
        },
        "gates": gates,
        "training_sample_summary": {
            "n_samples": len(training_samples),
            "surfaces": sorted({str(sample.get("surface_index")) for sample in training_samples}),
            "alphas": sorted(
                {f"{alpha:.16g}" for sample in training_samples if (alpha := _alpha(sample)) is not None}
            ),
            "selected_ky_indices": sorted({str(sample.get("selected_ky_index")) for sample in training_samples}),
        },
        "holdout_artifacts": holdout_rows,
        "notes": (
            "This check gates claim promotion only. Passing aggregate reduced-objective "
            "FD and line-search artifacts proves optimizer plumbing; it does not by "
            "itself validate optimized-equilibrium nonlinear transport. Promotion "
            "requires independent held-out surface or field-line evidence."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-artifact", type=Path, default=DEFAULT_AGGREGATE_ARTIFACT)
    parser.add_argument("--line-search-artifact", type=Path, default=DEFAULT_LINE_SEARCH_ARTIFACT)
    parser.add_argument("--holdout-artifact", action="append", type=Path, default=[])
    parser.add_argument("--alpha-atol", type=float, default=1.0e-12)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--fail-on-blocked",
        action="store_true",
        help="Return non-zero when the promotion gate is blocked.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = check_vmec_boozer_aggregate_holdout_gate(
        aggregate_artifact=args.aggregate_artifact,
        line_search_artifact=args.line_search_artifact,
        holdout_artifacts=tuple(args.holdout_artifact),
        alpha_atol=args.alpha_atol,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
        print(f"saved {args.json_out}")
    else:
        print(text)
    if args.fail_on_blocked and not bool(report["passed"]):
        print(
            "VMEC/Boozer aggregate optimization promotion blocked: "
            + ", ".join(report["promotion_gate"]["blockers"]),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
