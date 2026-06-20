#!/usr/bin/env python3
"""Write a LASYM=true VMEC state-to-input mapping launch campaign.

The first state-to-input campaign showed that stellarator-symmetric
``RBC/ZBS`` perturbations do not move the admitted asymmetric ``Rsin/Zcos``
VMEC-state controls.  This tool writes the symmetry-compatible follow-up:
matched baseline/plus/minus VMEC decks with ``LASYM = .TRUE.`` and explicit
``RBS/ZBC`` candidate input coefficients.  It is still launch-only evidence;
the resulting WOUT files must be solved and post-processed into a conditioned
response matrix before any nonlinear-gradient campaign can use the controls.
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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.write_nonlinear_turbulence_gradient_campaign import _repo_relative  # noqa: E402
from tools.write_vmec_boundary_perturbation_inputs import (  # noqa: E402
    CoefficientSpec,
    _json_clean,
    _parse_coefficient_spec,
)
from tools.write_vmec_state_to_input_mapping_campaign import _admitted_controls  # noqa: E402


DEFAULT_OUT_PREFIX = (
    ROOT / "docs" / "_static" / "nonlinear_gradient_asymmetric_state_to_input_mapping_campaign"
)
ASYMMETRIC_FAMILIES = frozenset({"RBS", "ZBC"})


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _import_namelist():
    try:
        from vmec_jax import namelist
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "vmec_jax.namelist is required to insert LASYM=true RBS/ZBC coefficients"
        ) from exc
    return namelist


def _validate_asymmetric_coefficients(coefficients: Sequence[CoefficientSpec]) -> tuple[CoefficientSpec, ...]:
    seen: set[CoefficientSpec] = set()
    out: list[CoefficientSpec] = []
    for coefficient in coefficients:
        if coefficient.family not in ASYMMETRIC_FAMILIES:
            raise ValueError("asymmetric campaign coefficients must use RBS or ZBC")
        if coefficient in seen:
            raise ValueError(f"duplicate coefficient {coefficient.label}")
        seen.add(coefficient)
        out.append(coefficient)
    if not out:
        raise ValueError("at least one asymmetric candidate coefficient is required")
    return tuple(out)


def _input_path(out_dir: Path, case: str, state: str) -> Path:
    return out_dir / f"input.{case}_{state}"


def _wout_path(out_dir: Path, case: str, state: str) -> Path:
    return out_dir / f"wout_{case}_{state}.nc"


def _vmec_command(vmec_command: str, input_path: Path | str, extra_args: str) -> str:
    suffix = f" {extra_args.strip()}" if extra_args.strip() else ""
    return f"{vmec_command} {_repo_relative(input_path)}{suffix}"


def _coefficient_value(indata: Any, coefficient: CoefficientSpec) -> tuple[float, bool]:
    family_values = indata.indexed.get(coefficient.family)
    if not isinstance(family_values, dict):
        return 0.0, True
    key = (coefficient.m, coefficient.n)
    if key not in family_values:
        return 0.0, True
    return float(family_values[key]), False


def write_asymmetric_perturbation_inputs(
    *,
    baseline_input: Path,
    out_dir: Path,
    case: str,
    coefficient: CoefficientSpec,
    delta: float,
    vmec_command: str = "vmec_jax",
    vmec_extra_args: str = "",
) -> dict[str, Any]:
    if coefficient.family not in ASYMMETRIC_FAMILIES:
        raise ValueError("asymmetric VMEC perturbations must use RBS or ZBC")
    if not math.isfinite(float(delta)) or delta <= 0.0:
        raise ValueError("coefficient perturbation delta must be finite and positive")

    namelist = _import_namelist()
    base_indata = namelist.read_indata(str(baseline_input))
    baseline_lasym = bool(base_indata.get_bool("LASYM"))
    base_value, inserted_missing = _coefficient_value(base_indata, coefficient)
    delta_value = float(delta)
    states = {
        "baseline": base_value,
        "plus_delta": base_value + delta_value,
        "minus_delta": base_value - delta_value,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    input_files: dict[str, Path] = {}
    wout_files: dict[str, Path] = {}
    for state, value in states.items():
        indata = namelist.read_indata(str(baseline_input))
        indata.scalars["LASYM"] = True
        indata.indexed.setdefault(coefficient.family, {})[(coefficient.m, coefficient.n)] = float(value)
        input_file = _input_path(out_dir, case, state)
        namelist.write_indata(str(input_file), indata)
        input_files[state] = input_file
        wout_files[state] = _wout_path(out_dir, case, state)

    run_commands = {
        state: f"cd {_repo_relative(out_dir)} && {_vmec_command(vmec_command, input_file.name, vmec_extra_args)}"
        for state, input_file in input_files.items()
    }
    manifest = {
        "kind": "vmec_asymmetric_boundary_perturbation_input_manifest",
        "claim_level": "lasym_true_launch_plan_not_simulation_claim",
        "baseline_input": baseline_input.resolve(),
        "baseline_lasym": baseline_lasym,
        "generated_lasym": True,
        "case": str(case),
        "coefficient": coefficient.label,
        "coefficient_family": coefficient.family,
        "vmec_m": coefficient.m,
        "vmec_n": coefficient.n,
        "coefficient_slug": coefficient.slug,
        "baseline_value": float(base_value),
        "inserted_missing_coefficient": bool(inserted_missing),
        "delta_parameter": float(delta_value),
        "relative_delta": None,
        "state_values": {state: float(value) for state, value in states.items()},
        "state_input_files": input_files,
        "expected_wout_files": wout_files,
        "vmec_run_commands": run_commands,
        "production_contract": (
            "Run vmec_jax on each generated LASYM=true deck. Only a finite, "
            "conditioned response matrix from solved WOUT files can promote this "
            "launch artifact into a nonlinear-gradient control."
        ),
    }
    manifest_path = out_dir / "vmec_asymmetric_boundary_perturbation_manifest.json"
    manifest_path.write_text(
        json.dumps(_json_clean(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest["manifest"] = manifest_path
    return manifest


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
                "inserted_missing_coefficient",
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
                    "inserted_missing_coefficient": row.get("inserted_missing_coefficient"),
                    "delta_parameter": row.get("delta_parameter"),
                    "manifest": row.get("manifest"),
                }
            )


def _plot(path: Path, report: Mapping[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spectraxgk.artifacts.plotting import set_plot_style

    state_labels = [str(row["state_parameter"]) for row in report["admitted_state_controls"]]
    input_labels = [str(row["coefficient"]) for row in report["input_directions"]]
    if not state_labels:
        state_labels = ["no admitted state controls"]
    if not input_labels:
        input_labels = ["no input directions"]
    matrix = np.ones((len(state_labels), len(input_labels)))

    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.2), constrained_layout=True)
    axes[0].imshow(matrix, cmap="Oranges", vmin=0.0, vmax=3.0, aspect="auto")
    axes[0].set_xticks(np.arange(len(input_labels)), input_labels, rotation=30, ha="right")
    axes[0].set_yticks(np.arange(len(state_labels)), state_labels)
    axes[0].set_title("LASYM=true response matrix plan")
    axes[0].set_xlabel("asymmetric VMEC input")
    axes[0].set_ylabel("VMEC-state control")
    for y in range(len(state_labels)):
        for x in range(len(input_labels)):
            axes[0].text(x, y, "solve\npending", ha="center", va="center", fontsize=8.2, color="0.25")

    deltas = np.asarray([float(row["delta_parameter"]) for row in report["input_directions"]], dtype=float)
    if deltas.size:
        x = np.arange(deltas.size)
        colors = [
            "#e6550d" if row.get("inserted_missing_coefficient") else "#fdae6b"
            for row in report["input_directions"]
        ]
        axes[1].bar(x, deltas, color=colors, edgecolor="0.25")
        axes[1].set_xticks(x, input_labels, rotation=30, ha="right")
        axes[1].set_yscale("log")
    axes[1].set_title("Absolute finite-difference steps")
    axes[1].set_ylabel(r"$|\Delta c|$")
    axes[1].grid(True, axis="y", alpha=0.25)
    fig.suptitle("Asymmetric VMEC state-to-input campaign is launch-only", fontsize=14)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_asymmetric_state_to_input_mapping_campaign(
    *,
    ql_seed_screen: Mapping[str, Any],
    ql_seed_screen_path: Path | None,
    baseline_input: Path,
    out_dir: Path,
    out_prefix: Path,
    case: str,
    coefficients: Sequence[CoefficientSpec],
    delta: float = 1.0e-3,
    vmec_command: str = "vmec_jax",
    vmec_extra_args: str = "",
) -> dict[str, Any]:
    if not math.isfinite(float(delta)) or delta <= 0.0:
        raise ValueError("delta must be finite and positive")
    admitted_controls = _admitted_controls(ql_seed_screen)
    if not admitted_controls:
        raise ValueError("ql_seed_screen does not contain admitted VMEC-state controls")
    unique_coefficients = _validate_asymmetric_coefficients(coefficients)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_directions: list[dict[str, Any]] = []
    vmec_run_commands: list[str] = []
    for coefficient in unique_coefficients:
        direction_case = f"{case}_{coefficient.slug}"
        direction_dir = out_dir / coefficient.slug
        manifest = write_asymmetric_perturbation_inputs(
            baseline_input=baseline_input,
            out_dir=direction_dir,
            case=direction_case,
            coefficient=coefficient,
            delta=delta,
            vmec_command=vmec_command,
            vmec_extra_args=vmec_extra_args,
        )
        clean_manifest = _json_clean(manifest)
        input_directions.append(
            {
                "coefficient": manifest["coefficient"],
                "coefficient_family": manifest["coefficient_family"],
                "vmec_m": manifest["vmec_m"],
                "vmec_n": manifest["vmec_n"],
                "coefficient_slug": manifest["coefficient_slug"],
                "baseline_value": manifest["baseline_value"],
                "inserted_missing_coefficient": manifest["inserted_missing_coefficient"],
                "delta_parameter": manifest["delta_parameter"],
                "relative_delta": None,
                "generated_lasym": True,
                "state_values": clean_manifest["state_values"],
                "state_input_files": clean_manifest["state_input_files"],
                "expected_wout_files": clean_manifest["expected_wout_files"],
                "vmec_run_commands": clean_manifest["vmec_run_commands"],
                "manifest": clean_manifest["manifest"],
            }
        )
        vmec_run_commands.extend(str(command) for command in clean_manifest["vmec_run_commands"].values())

    blockers = [
        "vmec_response_artifact_missing",
        "state_to_input_jacobian_not_extracted",
        "mapping_conditioning_not_measured",
    ]
    if len(input_directions) < len(admitted_controls):
        blockers.append("candidate_input_direction_count_below_state_control_count")

    report = {
        "kind": "vmec_asymmetric_state_to_input_mapping_campaign",
        "claim_level": "lasym_true_state_to_input_mapping_launch_plan_not_mapping_evidence",
        "case": str(case),
        "passed": False,
        "ready_for_nonlinear_launch": False,
        "symmetry_branch": "LASYM=true asymmetric RBS/ZBC input controls",
        "baseline_input": baseline_input.resolve(),
        "ql_seed_screen": ql_seed_screen_path.resolve() if ql_seed_screen_path is not None else None,
        "admitted_state_controls": admitted_controls,
        "input_directions": input_directions,
        "planned_response_matrix_shape": [len(admitted_controls), len(input_directions)],
        "rank_feasibility_precheck_passed": len(input_directions) >= len(admitted_controls),
        "blockers": blockers,
        "vmec_run_commands": vmec_run_commands,
        "postprocess_protocol": [
            "run vmec_jax for every LASYM=true baseline/plus/minus input deck listed in vmec_run_commands",
            "extract the admitted Rsin/Zcos VMEC-state controls from each solved equilibrium",
            "form the central finite-difference response matrix d(state_control)/d(RBS/ZBC coefficient)",
            "solve the least-squares state-to-input map and record condition number plus relative residual",
            "only pass the mapping artifact to design_nonlinear_gradient_state_control_runbook.py if the map is finite, full row-rank, conditioned, and residual-bounded",
        ],
        "coefficient_convention": (
            "Candidate directions are explicit VMEC input coefficients. The stored vmec_m/vmec_n "
            "fields preserve the same index ordering used by vmec_jax.namelist."
        ),
        "next_action": (
            "run the LASYM=true VMEC perturbation solves and build a conditioned state-to-input "
            "mapping artifact; this launch plan alone must not be used as nonlinear-gradient evidence"
        ),
    }

    json_path = out_prefix.with_suffix(".json")
    csv_path = out_prefix.with_suffix(".csv")
    png_path = out_prefix.with_suffix(".png")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(_json_clean(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
    parser.add_argument("--case", default="qa_lowres_asymmetric_state_to_input_mapping")
    parser.add_argument("--coefficient", action="append", required=True)
    parser.add_argument("--delta", type=float, default=1.0e-3)
    parser.add_argument("--vmec-command", default="vmec_jax")
    parser.add_argument("--vmec-extra-args", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ql_seed_screen = _load_json(args.ql_seed_screen)
    report = write_asymmetric_state_to_input_mapping_campaign(
        ql_seed_screen=ql_seed_screen,
        ql_seed_screen_path=Path(args.ql_seed_screen),
        baseline_input=Path(args.baseline_input),
        out_dir=Path(args.out_dir),
        out_prefix=Path(args.out_prefix),
        case=str(args.case),
        coefficients=tuple(_parse_coefficient_spec(raw) for raw in args.coefficient),
        delta=float(args.delta),
        vmec_command=str(args.vmec_command),
        vmec_extra_args=str(args.vmec_extra_args),
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
