#!/usr/bin/env python3
"""Audit that quasilinear calibration points use passed nonlinear gates."""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from spectraxgk.artifacts.plotting import set_plot_style  # noqa: E402


ACCEPTABLE_EXTERNAL_VMEC_HOLDOUT_CLAIM_LEVELS = frozenset(
    {
        "passed_grid_convergence_candidate_for_transport_holdout",
        "passed_grid_converged_external_vmec_transport_window",
        "passed_high_grid_transport_holdout_admission_under_coarse_grid_exclusion",
        "passed_replicated_external_vmec_transport_holdout_under_explicit_spread_gate",
    }
)

EXTERNAL_VMEC_HOLDOUT_GATE_KINDS = frozenset(
    {
        "external_vmec_high_grid_admission_gate",
        "external_vmec_nonlinear_grid_convergence_gate",
        "external_vmec_replicate_admission_gate",
        "external_vmec_transport_window_summary",
    }
)


def _contains_external_vmec_marker(values: Iterable[object]) -> bool:
    return any("external_vmec" in str(value).lower() for value in values)


def is_external_vmec_holdout_gate(
    payload: dict[str, Any],
    *,
    artifact: str | Path | None = None,
    artifact_keys: Iterable[str] = (),
) -> bool:
    """Return whether a gate should use external-VMEC holdout admission rules."""

    kind = str(payload.get("kind", ""))
    claim_level = str(payload.get("claim_level", ""))
    if kind in EXTERNAL_VMEC_HOLDOUT_GATE_KINDS:
        return True
    if claim_level in ACCEPTABLE_EXTERNAL_VMEC_HOLDOUT_CLAIM_LEVELS:
        return True
    if claim_level.startswith("negative_grid_convergence_result"):
        return True
    case = str(payload.get("case", ""))
    values = (artifact, case, *tuple(artifact_keys))
    return _contains_external_vmec_marker(value for value in values if value)


def external_vmec_holdout_admission_status(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Return fail-closed calibration/model-selection admission metadata.

    External-VMEC holdout evidence may update quasilinear calibration ledgers
    only when its explicit promotion gate passes and its claim level is one of
    the scoped holdout-admission claim levels. Any other combination is negative
    evidence for admission, even if a lower-level convergence gate reports pass.
    """

    promotion_gate = payload.get("promotion_gate")
    promotion_gate = promotion_gate if isinstance(promotion_gate, dict) else {}
    gate_report = payload.get("gate_report")
    gate_report = gate_report if isinstance(gate_report, dict) else {}
    claim_level = str(payload.get("claim_level", ""))

    promotion_gate_passed = bool(promotion_gate.get("passed", False))
    claim_level_acceptable = (
        claim_level in ACCEPTABLE_EXTERNAL_VMEC_HOLDOUT_CLAIM_LEVELS
    )
    gate_report_passed = (
        bool(gate_report.get("passed", False)) if gate_report else None
    )
    top_level_passed = (
        bool(payload.get("passed", False)) if "passed" in payload else None
    )
    raw_gate_passed = bool(
        promotion_gate_passed or gate_report_passed is True or top_level_passed is True
    )

    blockers: list[str] = []
    if not promotion_gate_passed:
        blockers.append("promotion_gate_not_passed")
    if not claim_level_acceptable:
        blockers.append("claim_level_not_acceptable")
    if gate_report_passed is False:
        blockers.append("gate_report_not_passed")
    if top_level_passed is False:
        blockers.append("payload_not_passed")

    admitted = not blockers
    return {
        "admissible_for_calibration": admitted,
        "promotion_gate_passed": promotion_gate_passed,
        "claim_level": claim_level,
        "claim_level_acceptable": claim_level_acceptable,
        "gate_report_passed": gate_report_passed,
        "top_level_passed": top_level_passed,
        "raw_gate_passed": raw_gate_passed,
        "negative_evidence": not admitted,
        "admission_blockers": blockers,
    }


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GATE_GLOB = str(ROOT / "docs" / "_static" / "**" / "*.json")
DEFAULT_JSON = (
    ROOT / "docs" / "_static" / "quasilinear_validated_calibration_inputs.json"
)
DEFAULT_PNG = ROOT / "docs" / "_static" / "quasilinear_validated_calibration_inputs.png"


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, np.generic):
        return _json_clean(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _repo_relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _canonical_artifact_key(raw: object) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        try:
            return path.resolve().relative_to(ROOT.resolve()).as_posix()
        except ValueError:
            parts = path.parts
            for anchor in ("tools_out", "docs", "examples"):
                if anchor in parts:
                    return Path(*parts[parts.index(anchor) :]).as_posix()
            return path.as_posix()
    return path.as_posix().lstrip("./")


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return data


def _is_gate_passed(data: dict[str, Any]) -> bool | None:
    if isinstance(data.get("gate_report"), dict):
        return bool(data["gate_report"].get("passed", False))
    if isinstance(data.get("promotion_gate"), dict):
        return bool(data["promotion_gate"].get("passed", False))
    if "gate_passed" in data:
        return bool(data.get("gate_passed"))
    return None


def _gate_case(data: dict[str, Any], path: Path) -> str:
    if isinstance(data.get("gate_report"), dict):
        return str(data["gate_report"].get("case", data.get("case", path.stem)))
    return str(data.get("case", path.stem))


def _artifact_keys_from_gate(data: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for field in ("spectrax", "nonlinear_netcdf", "csv", "source"):
        key = _canonical_artifact_key(data.get(field))
        if key:
            keys.add(key)
    inputs = data.get("inputs")
    if isinstance(inputs, dict):
        for value in inputs.values():
            values = value if isinstance(value, list) else [value]
            for item in values:
                key = _canonical_artifact_key(item)
                if key:
                    keys.add(key)
    for run in data.get("runs", []):
        if isinstance(run, dict):
            for field in ("csv", "json", "source", "nonlinear_artifact"):
                key = _canonical_artifact_key(run.get(field))
                if key:
                    keys.add(key)
    return keys


def _expand_gate_paths(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(str(pattern), recursive=True)
        out.extend(Path(item) for item in matches)
    return sorted(set(out))


def build_gate_index(patterns: list[str]) -> dict[str, dict[str, Any]]:
    """Map nonlinear artifact paths to gate metadata."""

    index: dict[str, dict[str, Any]] = {}

    def prefer_gate(new: dict[str, Any], old: dict[str, Any] | None) -> bool:
        if old is None:
            return True
        new_score = (
            int(bool(new.get("admissible_for_calibration", False))),
            int(bool(new.get("promotion_gate_passed", False))),
            int(bool(new.get("claim_level_acceptable", False))),
            int(bool(new.get("raw_gate_passed", False))),
        )
        old_score = (
            int(bool(old.get("admissible_for_calibration", False))),
            int(bool(old.get("promotion_gate_passed", False))),
            int(bool(old.get("claim_level_acceptable", False))),
            int(bool(old.get("raw_gate_passed", False))),
        )
        return new_score >= old_score

    for path in _expand_gate_paths(patterns):
        try:
            data = _load_json(path)
        except Exception:
            continue
        passed = _is_gate_passed(data)
        if passed is None:
            continue
        artifact_keys = _artifact_keys_from_gate(data)
        artifact_keys.add(_repo_relative_path(path))
        path_key = _canonical_artifact_key(path)
        if path_key:
            artifact_keys.add(path_key)
        external_holdout_gate = is_external_vmec_holdout_gate(
            data,
            artifact=path,
            artifact_keys=artifact_keys,
        )
        admission = (
            external_vmec_holdout_admission_status(data)
            if external_holdout_gate
            else {
                "admissible_for_calibration": bool(passed),
                "promotion_gate_passed": bool(passed),
                "claim_level_acceptable": True,
                "raw_gate_passed": bool(passed),
                "negative_evidence": not bool(passed),
                "admission_blockers": [] if bool(passed) else ["gate_not_passed"],
            }
        )
        gate_metadata = {
            "artifact": _repo_relative_path(path),
            "case": _gate_case(data, path),
            "passed": bool(admission["admissible_for_calibration"]),
            "raw_gate_passed": bool(admission["raw_gate_passed"]),
            "promotion_gate_passed": bool(admission["promotion_gate_passed"]),
            "claim_level_acceptable": bool(admission["claim_level_acceptable"]),
            "admissible_for_calibration": bool(admission["admissible_for_calibration"]),
            "negative_evidence": bool(admission["negative_evidence"]),
            "admission_blockers": list(admission["admission_blockers"]),
            "kind": str(data.get("kind", "")),
            "claim_level": str(data.get("claim_level", "")),
        }
        for key in artifact_keys:
            if prefer_gate(gate_metadata, index.get(key)):
                index[key] = {
                    **gate_metadata,
                }
    return index


def _negative_evidence_rows(
    gate_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for gate in gate_index.values():
        if not bool(gate.get("negative_evidence", False)):
            continue
        key = (str(gate.get("artifact", "")), str(gate.get("case", "")))
        if key in seen:
            continue
        seen.add(key)
        rows.append(dict(gate))
    return rows


def _load_points_from_report(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "points" in data:
        points = data["points"]
    elif isinstance(data, list):
        points = data
    else:
        raise ValueError(f"{path} is not a calibration report or point list")
    if not isinstance(points, list):
        raise ValueError(f"{path} points field must be a list")
    return [dict(item) for item in points]


def audit_calibration_inputs(
    reports: list[str | Path],
    *,
    gate_patterns: list[str] | None = None,
    required_splits: tuple[str, ...] = ("train", "holdout"),
) -> dict[str, Any]:
    """Return a JSON-ready audit for quasilinear calibration inputs."""

    gate_index = build_gate_index(gate_patterns or [DEFAULT_GATE_GLOB])
    report_rows = []
    all_passed = True
    for report_path_raw in reports:
        report_path = Path(report_path_raw)
        points = _load_points_from_report(report_path)
        point_rows = []
        report_passed = True
        for point in points:
            split = str(point.get("split", ""))
            artifact_key = _canonical_artifact_key(point.get("nonlinear_artifact"))
            required = split in required_splits
            gate = gate_index.get(artifact_key or "")
            point_passed = (not required) or (gate is not None and bool(gate["passed"]))
            if not point_passed:
                report_passed = False
                all_passed = False
            reason = "not required split"
            if required and gate is None:
                reason = "no matching nonlinear validation/convergence gate"
            elif (
                required
                and gate is not None
                and bool(gate.get("negative_evidence", False))
            ):
                reason = "matching nonlinear gate is negative evidence for calibration admission"
            elif required and gate is not None and not bool(gate["passed"]):
                reason = "matching nonlinear gate is not passed"
            elif required:
                reason = "matched passed nonlinear gate"
            point_rows.append(
                {
                    "case": str(point.get("case", "")),
                    "split": split,
                    "required": required,
                    "nonlinear_artifact": artifact_key,
                    "passed": point_passed,
                    "reason": reason,
                    "matched_gate": None if gate is None else dict(gate),
                }
            )
        report_rows.append(
            {
                "report": _repo_relative_path(report_path),
                "passed": report_passed,
                "n_points": len(point_rows),
                "n_required": sum(1 for row in point_rows if row["required"]),
                "n_required_passed": sum(
                    1 for row in point_rows if row["required"] and row["passed"]
                ),
                "points": point_rows,
            }
        )
    return _json_clean(
        {
            "kind": "quasilinear_calibration_input_audit",
            "claim_level": "calibration_inputs_validated_by_passed_nonlinear_gates",
            "passed": all_passed,
            "required_splits": list(required_splits),
            "gate_patterns": [
                str(pattern) for pattern in (gate_patterns or [DEFAULT_GATE_GLOB])
            ],
            "n_gate_artifact_matches": len(gate_index),
            "n_negative_evidence": len(_negative_evidence_rows(gate_index)),
            "negative_evidence": _negative_evidence_rows(gate_index),
            "reports": report_rows,
        }
    )


def write_audit_plot(
    payload: dict[str, Any], out_png: str | Path = DEFAULT_PNG
) -> None:
    """Write a compact calibration-input audit plot."""

    rows = []
    seen: set[tuple[str, str, str, bool]] = set()
    for report in payload["reports"]:
        for point in report["points"]:
            if point["required"]:
                gate_case = (
                    ""
                    if point["matched_gate"] is None
                    else str(point["matched_gate"]["case"]).replace("_", " ")
                )
                key = (
                    str(point["case"]),
                    str(point["split"]),
                    gate_case,
                    bool(point["passed"]),
                )
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "report": Path(str(report["report"]))
                        .stem.replace("quasilinear_", "")
                        .replace("_report", ""),
                        "case": str(point["case"]).replace("_", " "),
                        "split": str(point["split"]),
                        "passed": bool(point["passed"]),
                        "gate": gate_case,
                    }
                )
    if not rows:
        return
    set_plot_style()
    height = max(3.2, 0.48 * len(rows) + 1.5)
    fig, ax = plt.subplots(figsize=(10.8, height), constrained_layout=True)
    y = np.arange(len(rows))
    colors = ["#2a9d55" if row["passed"] else "#c2410c" for row in rows]
    ax.barh(y, np.ones(len(rows)), color=colors, alpha=0.88)
    ax.set_yticks(y, [f"{row['case']} ({row['split']})" for row in rows])
    ax.set_xticks([])
    ax.set_xlim(0.0, 1.0)
    ax.invert_yaxis()
    ax.set_title("Quasilinear Calibration Inputs: Nonlinear Gate Audit")
    for idx, row in enumerate(rows):
        text = f"matched gate: {row['gate'] or 'missing/failed gate'}"
        ax.text(0.02, idx, text, va="center", ha="left", color="white", fontsize=8.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def write_audit(
    reports: list[str | Path],
    *,
    gate_patterns: list[str] | None = None,
    out_json: str | Path = DEFAULT_JSON,
    out_png: str | Path = DEFAULT_PNG,
    required_splits: tuple[str, ...] = ("train", "holdout"),
    no_plot: bool = False,
) -> dict[str, str]:
    """Write a quasilinear calibration input audit artifact set."""

    payload = audit_calibration_inputs(
        reports, gate_patterns=gate_patterns, required_splits=required_splits
    )
    json_path = Path(out_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    paths = {"json": str(json_path)}
    if not no_plot:
        write_audit_plot(payload, out_png)
        paths["png"] = str(out_png)
        paths["pdf"] = str(Path(out_png).with_suffix(".pdf"))
    return paths


def _parse_required_splits(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        action="append",
        required=True,
        help="Calibration report or point-list JSON.",
    )
    parser.add_argument(
        "--gate-json",
        action="append",
        dest="gate_patterns",
        default=None,
        help="Gate JSON glob.",
    )
    parser.add_argument("--required-splits", default="train,holdout")
    parser.add_argument("--out-json", default=str(DEFAULT_JSON))
    parser.add_argument("--out-png", default=str(DEFAULT_PNG))
    parser.add_argument("--no-plot", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = write_audit(
        args.report,
        gate_patterns=args.gate_patterns,
        out_json=args.out_json,
        out_png=args.out_png,
        required_splits=_parse_required_splits(args.required_splits),
        no_plot=args.no_plot,
    )
    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    print(json.dumps(paths, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
