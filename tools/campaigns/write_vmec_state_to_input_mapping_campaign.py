#!/usr/bin/env python3
"""Write a fail-closed VMEC state-to-input mapping launch campaign.

The quasilinear seed screen admits internal ``vmec_jax`` state controls, while
long-window nonlinear campaigns perturb VMEC input coefficients.  This tool
bridges that gap only as a launch artifact: it writes baseline/plus/minus VMEC
input decks for candidate input coefficients and records the response-matrix
protocol needed to prove a conditioned state-to-input mapping later.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.campaigns.write_vmec_boundary_campaigns import (  # noqa: E402
    CoefficientSpec,
    _json_clean,
    _parse_coefficient_spec,
    write_perturbation_inputs,
)


DEFAULT_OUT_PREFIX = (
    ROOT / "docs" / "_static" / "nonlinear_gradient_state_to_input_mapping_campaign"
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _admitted_controls(ql_seed_screen: Mapping[str, Any]) -> list[dict[str, Any]]:
    controls = ql_seed_screen.get("admitted_controls")
    if not isinstance(controls, Sequence):
        return []
    rows: list[dict[str, Any]] = []
    for raw in controls:
        if not isinstance(raw, Mapping):
            continue
        state_parameter = raw.get("state_parameter")
        if not isinstance(state_parameter, str) or not state_parameter:
            continue
        rows.append(
            {
                "state_parameter": state_parameter,
                "state_control_argument": raw.get("state_control_argument"),
                "state_control_family": raw.get("state_control_family"),
                "descent_direction_sign": raw.get("descent_direction_sign"),
                "mean_abs_sensitivity": raw.get("mean_abs_sensitivity"),
                "sign_consistency_fraction": raw.get("sign_consistency_fraction"),
                "source_case_count": raw.get("n_cases"),
            }
        )
    return rows


def _write_csv(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "row_type",
                "state_parameter",
                "state_control_argument",
                "coefficient",
                "coefficient_slug",
                "baseline_value",
                "delta_parameter",
                "manifest",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in report["admitted_state_controls"]:
            writer.writerow(
                {
                    "row_type": "state_control",
                    "state_parameter": row.get("state_parameter"),
                    "state_control_argument": row.get("state_control_argument"),
                }
            )
        for row in report["input_directions"]:
            writer.writerow(
                {
                    "row_type": "input_direction",
                    "coefficient": row.get("coefficient"),
                    "coefficient_slug": row.get("coefficient_slug"),
                    "baseline_value": row.get("baseline_value"),
                    "delta_parameter": row.get("delta_parameter"),
                    "manifest": row.get("manifest"),
                }
            )


def _plot(path: Path, report: Mapping[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spectraxgk.artifacts.plotting import set_plot_style

    state_labels = [
        str(row["state_parameter"]) for row in report["admitted_state_controls"]
    ]
    input_labels = [str(row["coefficient"]) for row in report["input_directions"]]
    if not state_labels:
        state_labels = ["no admitted state controls"]
    if not input_labels:
        input_labels = ["no input directions"]
    matrix = np.ones((len(state_labels), len(input_labels)))

    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.2), constrained_layout=True)
    axes[0].imshow(matrix, cmap="Blues", vmin=0.0, vmax=3.0, aspect="auto")
    axes[0].set_xticks(
        np.arange(len(input_labels)), input_labels, rotation=30, ha="right"
    )
    axes[0].set_yticks(np.arange(len(state_labels)), state_labels)
    axes[0].set_title("Planned response matrix")
    axes[0].set_xlabel("VMEC input perturbation")
    axes[0].set_ylabel("VMEC-state control")
    for y in range(len(state_labels)):
        for x in range(len(input_labels)):
            axes[0].text(
                x,
                y,
                "solve\npending",
                ha="center",
                va="center",
                fontsize=8.2,
                color="0.25",
            )

    deltas = np.asarray(
        [float(row["delta_parameter"]) for row in report["input_directions"]],
        dtype=float,
    )
    if deltas.size:
        x = np.arange(deltas.size)
        axes[1].bar(x, deltas, color="#377eb8", edgecolor="0.25")
        axes[1].set_xticks(x, input_labels, rotation=30, ha="right")
        axes[1].set_yscale("log")
    axes[1].set_title("Finite-difference input steps")
    axes[1].set_ylabel(r"$|\Delta c|$")
    axes[1].grid(True, axis="y", alpha=0.25)
    fig.suptitle("VMEC state-to-input mapping campaign is launch-only", fontsize=14)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _unique_coefficients(
    coefficients: Sequence[CoefficientSpec],
) -> tuple[CoefficientSpec, ...]:
    seen: set[CoefficientSpec] = set()
    out: list[CoefficientSpec] = []
    for coefficient in coefficients:
        if coefficient in seen:
            raise ValueError(f"duplicate coefficient {coefficient.label}")
        seen.add(coefficient)
        out.append(coefficient)
    if not out:
        raise ValueError("at least one candidate coefficient is required")
    return tuple(out)


def write_state_to_input_mapping_campaign(
    *,
    ql_seed_screen: Mapping[str, Any],
    ql_seed_screen_path: Path | None,
    baseline_input: Path,
    out_dir: Path,
    out_prefix: Path,
    case: str,
    coefficients: Sequence[CoefficientSpec],
    relative_delta: float = 0.02,
    vmec_command: str = "vmec_jax",
) -> dict[str, Any]:
    if not math.isfinite(float(relative_delta)) or relative_delta <= 0.0:
        raise ValueError("relative_delta must be finite and positive")
    admitted_controls = _admitted_controls(ql_seed_screen)
    if not admitted_controls:
        raise ValueError("ql_seed_screen does not contain admitted VMEC-state controls")
    unique_coefficients = _unique_coefficients(coefficients)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_directions: list[dict[str, Any]] = []
    vmec_run_commands: list[str] = []
    for coefficient in unique_coefficients:
        direction_case = f"{case}_{coefficient.slug}"
        direction_dir = out_dir / coefficient.slug
        manifest = write_perturbation_inputs(
            baseline_input=baseline_input,
            out_dir=direction_dir,
            case=direction_case,
            coefficient=coefficient,
            relative_delta=relative_delta,
            vmec_command=vmec_command,
        )
        clean_manifest = _json_clean(manifest)
        input_directions.append(
            {
                "coefficient": manifest["coefficient"],
                "coefficient_slug": manifest["coefficient_slug"],
                "baseline_value": manifest["baseline_value"],
                "delta_parameter": manifest["delta_parameter"],
                "relative_delta": manifest["relative_delta"],
                "state_values": clean_manifest["state_values"],
                "state_input_files": clean_manifest["state_input_files"],
                "expected_wout_files": clean_manifest["expected_wout_files"],
                "vmec_run_commands": clean_manifest["vmec_run_commands"],
                "manifest": clean_manifest["manifest"],
            }
        )
        vmec_run_commands.extend(
            str(command) for command in clean_manifest["vmec_run_commands"].values()
        )

    blockers = [
        "vmec_response_artifact_missing",
        "state_to_input_jacobian_not_extracted",
        "mapping_conditioning_not_measured",
    ]
    if len(input_directions) < len(admitted_controls):
        blockers.append("candidate_input_direction_count_below_state_control_count")

    report = {
        "kind": "vmec_state_to_input_mapping_campaign",
        "claim_level": "state_to_input_mapping_launch_plan_not_mapping_evidence",
        "case": str(case),
        "passed": False,
        "ready_for_nonlinear_launch": False,
        "baseline_input": baseline_input.resolve(),
        "ql_seed_screen": ql_seed_screen_path.resolve()
        if ql_seed_screen_path is not None
        else None,
        "admitted_state_controls": admitted_controls,
        "input_directions": input_directions,
        "planned_response_matrix_shape": [
            len(admitted_controls),
            len(input_directions),
        ],
        "rank_feasibility_precheck_passed": len(input_directions)
        >= len(admitted_controls),
        "blockers": blockers,
        "vmec_run_commands": vmec_run_commands,
        "postprocess_protocol": [
            "run vmec_jax for every baseline/plus/minus input deck listed in vmec_run_commands",
            "extract the admitted VMEC-state controls from each solved equilibrium",
            "form the central finite-difference response matrix d(state_control)/d(input_coefficient)",
            "solve the least-squares state-to-input map and record condition number plus relative residual",
            "only pass the mapping artifact to design_nonlinear_gradient.py state-control-runbook if the mapping is local, conditioned, and residual-bounded",
        ],
        "coefficient_convention": (
            "Candidate directions are stored exactly as they appear in the VMEC input file, "
            "for example RBC(1,1) or ZBS(1,0). Downstream mapping artifacts should also record "
            "explicit vmec_n/vmec_m fields before promoting any state-to-input direction."
        ),
        "mapping_artifact_schema_after_vmec_runs": {
            "kind": "vmec_state_to_input_control_mapping",
            "passed": "bool",
            "controls": [
                {
                    "state_parameter": "<admitted VMEC-state control>",
                    "input_control_argument": "RBC(m,n):weight or profile direction",
                    "passed": "bool",
                    "condition_number": "<finite positive float>",
                    "relative_residual": "<finite non-negative float>",
                    "input_direction": "<optional raw coefficient weights>",
                }
            ],
        },
        "next_action": (
            "run the VMEC perturbation solves and build a conditioned state-to-input mapping artifact; "
            "this launch plan alone must not be used as nonlinear-gradient evidence"
        ),
    }

    json_path = out_prefix.with_suffix(".json")
    csv_path = out_prefix.with_suffix(".csv")
    png_path = out_prefix.with_suffix(".png")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(_json_clean(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(csv_path, _json_clean(report))
    _plot(png_path, _json_clean(report))
    report["artifacts"] = {
        "json": json_path,
        "csv": csv_path,
        "png": png_path,
        "pdf": png_path.with_suffix(".pdf"),
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ql_seed_screen", type=Path)
    parser.add_argument("--baseline-input", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
    parser.add_argument("--case", default="qa_ess_state_to_input_mapping")
    parser.add_argument("--coefficient", action="append", required=True)
    parser.add_argument("--relative-delta", type=float, default=0.02)
    parser.add_argument("--vmec-command", default="vmec_jax")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ql_seed_screen = _load_json(args.ql_seed_screen)
    report = write_state_to_input_mapping_campaign(
        ql_seed_screen=ql_seed_screen,
        ql_seed_screen_path=Path(args.ql_seed_screen),
        baseline_input=Path(args.baseline_input),
        out_dir=Path(args.out_dir),
        out_prefix=Path(args.out_prefix),
        case=str(args.case),
        coefficients=tuple(_parse_coefficient_spec(raw) for raw in args.coefficient),
        relative_delta=float(args.relative_delta),
        vmec_command=str(args.vmec_command),
    )
    artifacts = _json_clean(report["artifacts"])
    print(
        json.dumps(
            {
                "json": artifacts["json"],
                "passed": report["passed"],
                "state_controls": len(report["admitted_state_controls"]),
                "input_directions": len(report["input_directions"]),
                "next_action": report["next_action"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
