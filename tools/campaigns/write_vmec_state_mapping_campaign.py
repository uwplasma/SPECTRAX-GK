#!/usr/bin/env python3
"""Write symmetric or asymmetric VMEC state-to-input mapping campaigns.

Both subcommands produce launch-only, fail-closed mapping evidence. Solved WOUT
responses and a conditioned response matrix are still required before a state
control can be used in a nonlinear-gradient campaign.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.campaigns.write_nonlinear_turbulence_gradient_campaign import _repo_relative
from tools.campaigns.write_vmec_boundary_campaigns import (
    CoefficientSpec, _coefficient_rows, _fortran_float, _json_clean,
    _parse_coefficient_spec,
    write_perturbation_inputs,
)

DEFAULT_SYMMETRIC_OUT_PREFIX = (
    ROOT / "docs" / "_static" / "nonlinear_gradient_state_to_input_mapping_campaign"
)
DEFAULT_ASYMMETRIC_OUT_PREFIX = (
    ROOT / "docs" / "_static" / "nonlinear_gradient_asymmetric_state_to_input_mapping_campaign"
)
ASYMMETRIC_FAMILIES = frozenset({"RBS", "ZBC"})
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


def build_symmetric_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ql_seed_screen", type=Path)
    parser.add_argument("--baseline-input", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_SYMMETRIC_OUT_PREFIX)
    parser.add_argument("--case", default="qa_ess_state_to_input_mapping")
    parser.add_argument("--coefficient", action="append", required=True)
    parser.add_argument("--relative-delta", type=float, default=0.02)
    parser.add_argument("--vmec-command", default="vmec_jax")
    return parser


def main_symmetric(argv: list[str] | None = None) -> int:
    args = build_symmetric_parser().parse_args(argv)
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


def _import_vmec_input():
    try:
        from vmec_jax import VmecInput
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "vmec_jax.VmecInput is required to insert LASYM=true RBS/ZBC coefficients"
        ) from exc
    return VmecInput


def _validate_asymmetric_coefficients(
    coefficients: Sequence[CoefficientSpec],
) -> tuple[CoefficientSpec, ...]:
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


def _vmec_input_coefficient_value(
    indata: Any, coefficient: CoefficientSpec
) -> float:
    values = np.asarray(getattr(indata, coefficient.family.lower()), dtype=float)
    n_index = int(coefficient.m) + int(indata.ntor)
    m_index = int(coefficient.n)
    if not (0 <= n_index < values.shape[0] and 0 <= m_index < values.shape[1]):
        raise ValueError(
            f"coefficient {coefficient.label} is outside the input NTOR/MPOL extent"
        )
    return float(values[n_index, m_index])


def _ensure_vmec_mode_extent(indata: Any, coefficient: CoefficientSpec) -> Any:
    if coefficient.n < 0:
        raise ValueError("the second VMEC boundary index must be nonnegative")
    ntor = max(int(indata.ntor), abs(int(coefficient.m)))
    mpol = max(int(indata.mpol), int(coefficient.n) + 1)
    if ntor == int(indata.ntor) and mpol == int(indata.mpol):
        return indata

    boundary_updates: dict[str, np.ndarray] = {}
    for family in ("rbc", "zbs", "rbs", "zbc"):
        old = np.asarray(getattr(indata, family), dtype=float)
        expanded = np.zeros((2 * ntor + 1, mpol), dtype=float)
        for old_n in range(-int(indata.ntor), int(indata.ntor) + 1):
            expanded[old_n + ntor, : int(indata.mpol)] = old[
                old_n + int(indata.ntor), :
            ]
        boundary_updates[family] = expanded

    axis_updates: dict[str, np.ndarray] = {}
    for name in ("raxis_c", "zaxis_s", "raxis_s", "zaxis_c"):
        old = np.asarray(getattr(indata, name), dtype=float)
        expanded = np.zeros(ntor + 1, dtype=float)
        expanded[: old.size] = old
        axis_updates[name] = expanded
    return replace(
        indata,
        ntor=ntor,
        mpol=mpol,
        **boundary_updates,
        **axis_updates,
    )


def _ensure_vmec_mode_extents(
    indata: Any, coefficients: Sequence[CoefficientSpec]
) -> Any:
    for coefficient in coefficients:
        indata = _ensure_vmec_mode_extent(indata, coefficient)
    return indata


def _write_vmec_input_coefficients(
    indata: Any,
    path: Path,
    values_by_coefficient: Mapping[CoefficientSpec, float],
    *,
    force_lasym: bool,
) -> None:
    indata = _ensure_vmec_mode_extents(indata, tuple(values_by_coefficient))
    boundary_updates = {
        family: np.asarray(getattr(indata, family), dtype=float).copy()
        for family in ("rbc", "zbs", "rbs", "zbc")
    }
    for coefficient, value in values_by_coefficient.items():
        boundary_updates[coefficient.family.lower()][
            int(coefficient.m) + int(indata.ntor), int(coefficient.n)
        ] = float(value)
    replace(
        indata,
        lasym=bool(indata.lasym) or bool(force_lasym),
        **boundary_updates,
    ).to_indata(path)

    # VmecInput intentionally omits zero Fourier coefficients. Keep every
    # explicitly controlled zero so launch decks expose the same FD coordinates.
    text = path.read_text(encoding="utf-8")
    zero_rows = "".join(
        f"  {coefficient.label} = {_fortran_float(value)}\n"
        for coefficient, value in values_by_coefficient.items()
        if float(value) == 0.0 and not _coefficient_rows(text, coefficient)
    )
    if zero_rows:
        marker = text.rfind("/")
        if marker < 0:
            raise ValueError(f"generated VMEC input has no namelist terminator: {path}")
        path.write_text(text[:marker] + zero_rows + text[marker:], encoding="utf-8")


def _write_vmec_input_coefficient(
    indata: Any, path: Path, coefficient: CoefficientSpec, value: float
) -> None:
    _write_vmec_input_coefficients(
        indata, path, {coefficient: float(value)}, force_lasym=True
    )


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

    VmecInput = _import_vmec_input()
    baseline_text = baseline_input.read_text(encoding="utf-8")
    explicit_rows = _coefficient_rows(baseline_text, coefficient)
    if len(explicit_rows) > 1:
        raise ValueError(
            f"coefficient {coefficient.label} appears {len(explicit_rows)} times"
        )
    base_indata = _ensure_vmec_mode_extent(
        VmecInput.from_file(baseline_input), coefficient
    )
    baseline_lasym = bool(base_indata.lasym)
    base_value = _vmec_input_coefficient_value(base_indata, coefficient)
    inserted_missing = not explicit_rows
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
        input_file = _input_path(out_dir, case, state)
        _write_vmec_input_coefficient(base_indata, input_file, coefficient, value)
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


def _write_asymmetric_csv(path: Path, report: Mapping[str, Any]) -> None:
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
                    "inserted_missing_coefficient": row.get(
                        "inserted_missing_coefficient"
                    ),
                    "delta_parameter": row.get("delta_parameter"),
                    "manifest": row.get("manifest"),
                }
            )


def _plot_asymmetric(path: Path, report: Mapping[str, Any]) -> None:
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
    axes[0].imshow(matrix, cmap="Oranges", vmin=0.0, vmax=3.0, aspect="auto")
    axes[0].set_xticks(
        np.arange(len(input_labels)), input_labels, rotation=30, ha="right"
    )
    axes[0].set_yticks(np.arange(len(state_labels)), state_labels)
    axes[0].set_title("LASYM=true response matrix plan")
    axes[0].set_xlabel("asymmetric VMEC input")
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
                "inserted_missing_coefficient": manifest[
                    "inserted_missing_coefficient"
                ],
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
        "kind": "vmec_asymmetric_state_to_input_mapping_campaign",
        "claim_level": "lasym_true_state_to_input_mapping_launch_plan_not_mapping_evidence",
        "case": str(case),
        "passed": False,
        "ready_for_nonlinear_launch": False,
        "symmetry_branch": "LASYM=true asymmetric RBS/ZBC input controls",
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
            "run vmec_jax for every LASYM=true baseline/plus/minus input deck listed in vmec_run_commands",
            "extract the admitted Rsin/Zcos VMEC-state controls from each solved equilibrium",
            "form the central finite-difference response matrix d(state_control)/d(RBS/ZBC coefficient)",
            "solve the least-squares state-to-input map and record condition number plus relative residual",
            "only pass the mapping artifact to design_nonlinear_gradient.py state-control-runbook if the map is finite, full row-rank, conditioned, and residual-bounded",
        ],
        "coefficient_convention": (
            "Candidate directions are explicit VMEC input coefficients. The stored vmec_m/vmec_n "
            "fields preserve the input-deck index ordering parsed by vmec_jax.VmecInput."
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
    json_path.write_text(
        json.dumps(_json_clean(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_asymmetric_csv(csv_path, _json_clean(report))
    _plot_asymmetric(png_path, _json_clean(report))
    report["artifacts"] = {
        "json": json_path,
        "csv": csv_path,
        "png": png_path,
        "pdf": png_path.with_suffix(".pdf"),
    }
    return report


def build_asymmetric_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ql_seed_screen", type=Path)
    parser.add_argument("--baseline-input", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_ASYMMETRIC_OUT_PREFIX)
    parser.add_argument("--case", default="qa_lowres_asymmetric_state_to_input_mapping")
    parser.add_argument("--coefficient", action="append", required=True)
    parser.add_argument("--delta", type=float, default=1.0e-3)
    parser.add_argument("--vmec-command", default="vmec_jax")
    parser.add_argument("--vmec-extra-args", default="")
    return parser


def main_asymmetric(argv: list[str] | None = None) -> int:
    args = build_asymmetric_parser().parse_args(argv)
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

_COMMANDS = {
    "symmetric": main_symmetric,
    "asymmetric": main_asymmetric,
}


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if not raw or raw[0] in {"-h", "--help"}:
        choices = ", ".join(_COMMANDS)
        print(f"usage: write_vmec_state_mapping_campaign.py <command> [options]\ncommands: {choices}")
        return 0
    command, *command_args = raw
    try:
        command_main = _COMMANDS[command]
    except KeyError as exc:
        choices = ", ".join(_COMMANDS)
        raise SystemExit(f"unknown command {command!r}; choose one of: {choices}") from exc
    return command_main(command_args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
