#!/usr/bin/env python3
"""Write VMEC launch decks for mapped VMEC-state short-bracket audits.

This consumes a passing nonlinear-gradient state-control runbook and turns each
mapped state control into explicit VMEC baseline/plus/minus input decks.  The
finite-difference scalar is an absolute multiplier on the least-squares input
direction stored in the runbook.  The output is a launch contract only; it is
not nonlinear-gradient evidence until VMEC and SPECTRAX-GK runs are completed
and post-processed through the usual replicated transport-window gates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.write_nonlinear_turbulence_gradient_campaign import PYTHON_CMD, _repo_relative  # noqa: E402
from tools.write_vmec_boundary_perturbation_inputs import (  # noqa: E402
    CoefficientSpec,
    _json_clean,
    _parse_coefficient_spec,
)


DEFAULT_OUT_PREFIX = ROOT / "docs" / "_static" / "nonlinear_gradient_state_control_short_bracket_launch"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _slug(raw: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")


def _import_namelist():
    try:
        from vmec_jax import namelist
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("vmec_jax.namelist is required to write VMEC launch decks") from exc
    return namelist


def _coefficient_value(indata: Any, spec: CoefficientSpec) -> tuple[float, bool]:
    family_values = indata.indexed.get(spec.family)
    if not isinstance(family_values, dict):
        return 0.0, True
    key = (spec.m, spec.n)
    if key not in family_values:
        return 0.0, True
    return float(family_values[key]), False


def _terms_from_control(row: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    input_direction = row.get("input_direction")
    if not isinstance(input_direction, Mapping):
        raise ValueError(f"{row.get('state_parameter')} missing input_direction")
    raw_terms = input_direction.get("terms")
    if not isinstance(raw_terms, Sequence) or not raw_terms:
        raise ValueError(f"{row.get('state_parameter')} missing input_direction terms")
    terms: list[dict[str, Any]] = []
    seen: set[CoefficientSpec] = set()
    for raw in raw_terms:
        if not isinstance(raw, Mapping):
            raise ValueError("input_direction terms must be JSON objects")
        coefficient = raw.get("coefficient")
        weight = float(raw.get("weight"))
        if not isinstance(coefficient, str) or not coefficient:
            raise ValueError("input_direction term missing coefficient")
        if not math.isfinite(weight) or weight == 0.0:
            raise ValueError(f"{coefficient} has nonfinite or zero weight")
        spec = _parse_coefficient_spec(coefficient)
        if spec in seen:
            raise ValueError(f"duplicate coefficient {spec.label}")
        seen.add(spec)
        terms.append(
            {
                "coefficient": spec.label,
                "coefficient_slug": spec.slug,
                "spec": spec,
                "weight": float(weight),
            }
        )
    return tuple(terms)


def _mapped_controls(runbook: Mapping[str, Any], state_parameter: str | None) -> list[Mapping[str, Any]]:
    if not bool(runbook.get("passed")):
        raise ValueError("state-control runbook must pass before writing launch decks")
    rows = runbook.get("mapped_controls") or runbook.get("controls")
    if not isinstance(rows, Sequence):
        raise ValueError("runbook does not contain mapped controls")
    out: list[Mapping[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        if not bool(row.get("mapping_ready")):
            continue
        if state_parameter is not None and row.get("state_parameter") != state_parameter:
            continue
        out.append(row)
    if not out:
        raise ValueError("no mapped controls matched the requested state parameter")
    return out


def _input_path(out_dir: Path, case: str, state: str) -> Path:
    return out_dir / f"input.{case}_{state}"


def _wout_path(out_dir: Path, case: str, state: str) -> Path:
    return out_dir / f"wout_{case}_{state}.nc"


def _vmec_command(vmec_command: str, input_path: Path | str, extra_args: str) -> str:
    suffix = f" {extra_args.strip()}" if extra_args.strip() else ""
    return f"{vmec_command} {_repo_relative(input_path)}{suffix}"


def _write_one_control(
    *,
    baseline_input: Path,
    out_dir: Path,
    case: str,
    control: Mapping[str, Any],
    alpha_delta: float,
    vmec_command: str,
    vmec_extra_args: str,
    horizons: str,
    grid: str,
    window_tmin: float,
    window_tmax: float,
    ky: float,
    dt: float,
    baseline_seed: int,
    seed_variants: tuple[int, ...],
    nl: int,
    nm: int,
    output_min_samples: int,
    output_min_window_samples: int,
    output_min_abs_window_mean: float,
) -> dict[str, Any]:
    namelist = _import_namelist()
    state_parameter = str(control["state_parameter"])
    terms = _terms_from_control(control)
    base_indata = namelist.read_indata(str(baseline_input))
    force_lasym = any(term["spec"].family in {"RBS", "ZBC"} for term in terms)
    baseline_lasym = bool(base_indata.get_bool("LASYM"))

    rows: list[dict[str, Any]] = []
    base_values: dict[CoefficientSpec, float] = {}
    for term in terms:
        spec = term["spec"]
        base_value, inserted_missing = _coefficient_value(base_indata, spec)
        base_values[spec] = base_value
        delta_value = float(alpha_delta) * float(term["weight"])
        rows.append(
            {
                "coefficient": spec.label,
                "coefficient_slug": spec.slug,
                "vmec_m": spec.m,
                "vmec_n": spec.n,
                "weight": float(term["weight"]),
                "baseline_value": float(base_value),
                "inserted_missing_coefficient": bool(inserted_missing),
                "coefficient_delta": float(delta_value),
                "plus_value": float(base_value + delta_value),
                "minus_value": float(base_value - delta_value),
            }
        )

    states = {
        "baseline": 0.0,
        "plus_delta": 1.0,
        "minus_delta": -1.0,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    input_files: dict[str, Path] = {}
    wout_files: dict[str, Path] = {}
    for state, sign in states.items():
        indata = namelist.read_indata(str(baseline_input))
        if force_lasym:
            indata.scalars["LASYM"] = True
        for term in terms:
            spec = term["spec"]
            value = base_values[spec] + sign * float(alpha_delta) * float(term["weight"])
            indata.indexed.setdefault(spec.family, {})[(spec.m, spec.n)] = float(value)
        input_file = _input_path(out_dir, case, state)
        namelist.write_indata(str(input_file), indata)
        input_files[state] = input_file
        wout_files[state] = _wout_path(out_dir, case, state)

    run_commands = {
        state: f"cd {_repo_relative(out_dir)} && {_vmec_command(vmec_command, input_file.name, vmec_extra_args)}"
        for state, input_file in input_files.items()
    }
    parameter_name = f"state_control_{_slug(state_parameter)}"
    campaign_command = (
        f"{PYTHON_CMD} tools/write_nonlinear_turbulence_gradient_campaign.py "
        f"--baseline-vmec-file {_repo_relative(wout_files['baseline'])} "
        f"--plus-vmec-file {_repo_relative(wout_files['plus_delta'])} "
        f"--minus-vmec-file {_repo_relative(wout_files['minus_delta'])} "
        f"--case {case}_nonlinear_gradient "
        f"--parameter-name {parameter_name} "
        f"--delta-parameter {float(alpha_delta):.16e} "
        f"--out-dir {_repo_relative(out_dir / 'nonlinear_campaign')} "
        f"--horizons {horizons} "
        f"--grid {grid} "
        f"--window-tmin {float(window_tmin):.12g} "
        f"--window-tmax {float(window_tmax):.12g} "
        f"--ky {float(ky):.16g} "
        f"--dt {float(dt):.12g} "
        f"--baseline-seed {int(baseline_seed)} "
        f"--Nl {int(nl)} --Nm {int(nm)} "
        f"--output-min-samples {int(output_min_samples)} "
        f"--output-min-window-samples {int(output_min_window_samples)} "
        f"--output-min-abs-window-mean {float(output_min_abs_window_mean):.12g}"
    )
    for seed in seed_variants:
        campaign_command += f" --seed-variant {int(seed)}"

    manifest = {
        "kind": "vmec_state_control_short_bracket_launch_manifest",
        "claim_level": "mapped_state_control_short_bracket_launch_plan_not_simulation_claim",
        "state_parameter": state_parameter,
        "state_control_argument": control.get("state_control_argument"),
        "input_control_argument": control.get("input_control_argument"),
        "baseline_input": baseline_input.resolve(),
        "baseline_lasym": baseline_lasym,
        "generated_lasym": bool(force_lasym),
        "case": str(case),
        "parameter_name": parameter_name,
        "alpha_delta": float(alpha_delta),
        "mapping_condition_number": control.get("condition_number"),
        "mapping_relative_residual": control.get("relative_residual"),
        "controls": rows,
        "state_input_files": input_files,
        "expected_wout_files": wout_files,
        "vmec_run_commands": run_commands,
        "campaign_command_after_vmec_runs": campaign_command,
        "run_contract": {
            "short_bracket_launch": True,
            "horizons": horizons,
            "grid": grid,
            "analysis_window": [float(window_tmin), float(window_tmax)],
            "ky": float(ky),
            "dt": float(dt),
            "replicates": [f"seed{seed}" for seed in seed_variants],
            "Nl": int(nl),
            "Nm": int(nm),
            "output_min_samples": int(output_min_samples),
            "output_min_window_samples": int(output_min_window_samples),
            "output_min_abs_window_mean": float(output_min_abs_window_mean),
        },
        "production_contract": (
            "Run vmec_jax on each generated input, then use the resulting WOUT "
            "files to create a nonlinear-gradient campaign. Promotion requires "
            "runtime-output gates, replicated nonlinear-window gates, central-FD "
            "conditioning, and final nonlinear-gradient evidence."
        ),
    }
    manifest_path = out_dir / "vmec_state_control_short_bracket_launch_manifest.json"
    manifest_path.write_text(json.dumps(_json_clean(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["manifest"] = manifest_path
    return manifest


def _write_csv(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "state_parameter",
                "coefficient",
                "weight",
                "alpha_delta",
                "coefficient_delta",
                "inserted_missing_coefficient",
                "manifest",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in report["launches"]:
            for control in row["controls"]:
                writer.writerow(
                    {
                        "state_parameter": row["state_parameter"],
                        "coefficient": control["coefficient"],
                        "weight": control["weight"],
                        "alpha_delta": row["alpha_delta"],
                        "coefficient_delta": control["coefficient_delta"],
                        "inserted_missing_coefficient": control["inserted_missing_coefficient"],
                        "manifest": row["manifest"],
                    }
                )


def _plot(path: Path, report: Mapping[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spectraxgk.plotting import set_plot_style

    launches = list(report["launches"])
    coeffs = sorted({control["coefficient"] for row in launches for control in row["controls"]})
    rows = [str(row["state_parameter"]) for row in launches]
    matrix = np.zeros((len(rows), len(coeffs)), dtype=float)
    for i, row in enumerate(launches):
        lookup = {control["coefficient"]: float(control["coefficient_delta"]) for control in row["controls"]}
        for j, coeff in enumerate(coeffs):
            matrix[i, j] = lookup.get(coeff, 0.0)
    scale = max(float(np.max(np.abs(matrix))), 1.0e-12)

    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.0), constrained_layout=True)
    im = axes[0].imshow(matrix, cmap="RdBu_r", vmin=-scale, vmax=scale, aspect="auto")
    axes[0].set_xticks(np.arange(len(coeffs)), coeffs, rotation=30, ha="right")
    axes[0].set_yticks(np.arange(len(rows)), rows)
    axes[0].set_title("Mapped coefficient deltas")
    axes[0].set_xlabel("VMEC input coefficient")
    axes[0].set_ylabel("state-control launch")
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            axes[0].text(x, y, f"{matrix[y, x]:.1e}", ha="center", va="center", fontsize=8.0)
    fig.colorbar(im, ax=axes[0], shrink=0.85, label=r"$\Delta c_j$")

    condition = np.asarray([float(row["mapping_condition_number"]) for row in launches], dtype=float)
    residual = np.asarray([float(row["mapping_relative_residual"]) for row in launches], dtype=float)
    x = np.arange(len(rows))
    axes[1].bar(x - 0.18, condition, width=0.36, color="#377eb8", edgecolor="0.25", label="condition")
    axes[1].bar(x + 0.18, np.maximum(residual, 1.0e-18), width=0.36, color="#ff7f00", edgecolor="0.25", label="residual")
    axes[1].set_yscale("log")
    axes[1].set_xticks(x, rows, rotation=24, ha="right")
    axes[1].set_title("Mapping quality")
    axes[1].legend(frameon=False)
    axes[1].grid(True, axis="y", alpha=0.25)
    fig.suptitle("VMEC-state short-bracket launch contract", fontsize=14)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_state_control_short_bracket_launch(
    *,
    runbook: Mapping[str, Any],
    runbook_path: Path | None,
    baseline_input: Path,
    out_dir: Path,
    out_prefix: Path,
    case: str,
    state_parameter: str | None = None,
    alpha_delta: float = 1.0e-3,
    vmec_command: str = "vmec_jax",
    vmec_extra_args: str = "",
    horizons: str = "150",
    grid: str = "n48:48:48:32:32",
    window_tmin: float = 75.0,
    window_tmax: float = 150.0,
    ky: float = 0.47619047619047616,
    dt: float = 0.05,
    baseline_seed: int = 22,
    seed_variants: tuple[int, ...] = (31, 32),
    nl: int = 4,
    nm: int = 8,
    output_min_samples: int = 60,
    output_min_window_samples: int = 30,
    output_min_abs_window_mean: float = 1.0e-4,
) -> dict[str, Any]:
    if not math.isfinite(float(alpha_delta)) or alpha_delta <= 0.0:
        raise ValueError("alpha_delta must be finite and positive")
    controls = _mapped_controls(runbook, state_parameter)
    out_dir.mkdir(parents=True, exist_ok=True)
    launches: list[dict[str, Any]] = []
    vmec_run_commands: list[str] = []
    campaign_commands: list[str] = []
    for control in controls:
        slug = _slug(str(control["state_parameter"]))
        control_case = f"{case}_{slug}"
        control_dir = out_dir / slug
        manifest = _write_one_control(
            baseline_input=baseline_input,
            out_dir=control_dir,
            case=control_case,
            control=control,
            alpha_delta=float(alpha_delta),
            vmec_command=vmec_command,
            vmec_extra_args=vmec_extra_args,
            horizons=horizons,
            grid=grid,
            window_tmin=float(window_tmin),
            window_tmax=float(window_tmax),
            ky=float(ky),
            dt=float(dt),
            baseline_seed=int(baseline_seed),
            seed_variants=tuple(seed_variants),
            nl=int(nl),
            nm=int(nm),
            output_min_samples=int(output_min_samples),
            output_min_window_samples=int(output_min_window_samples),
            output_min_abs_window_mean=float(output_min_abs_window_mean),
        )
        clean = _json_clean(manifest)
        launches.append(
            {
                "state_parameter": manifest["state_parameter"],
                "state_control_argument": manifest["state_control_argument"],
                "input_control_argument": manifest["input_control_argument"],
                "parameter_name": manifest["parameter_name"],
                "case": manifest["case"],
                "alpha_delta": manifest["alpha_delta"],
                "mapping_condition_number": manifest["mapping_condition_number"],
                "mapping_relative_residual": manifest["mapping_relative_residual"],
                "generated_lasym": manifest["generated_lasym"],
                "controls": clean["controls"],
                "state_input_files": clean["state_input_files"],
                "expected_wout_files": clean["expected_wout_files"],
                "vmec_run_commands": clean["vmec_run_commands"],
                "campaign_command_after_vmec_runs": clean["campaign_command_after_vmec_runs"],
                "manifest": clean["manifest"],
            }
        )
        vmec_run_commands.extend(str(command) for command in clean["vmec_run_commands"].values())
        campaign_commands.append(str(clean["campaign_command_after_vmec_runs"]))

    report = {
        "kind": "vmec_state_control_short_bracket_launch_plan",
        "claim_level": "mapped_state_control_short_bracket_launch_plan_not_simulation_claim",
        "case": str(case),
        "passed": False,
        "ready_for_nonlinear_launch": False,
        "source_runbook": None if runbook_path is None else runbook_path.resolve(),
        "baseline_input": baseline_input.resolve(),
        "alpha_delta": float(alpha_delta),
        "launches": launches,
        "vmec_run_commands": vmec_run_commands,
        "campaign_commands_after_vmec_runs": campaign_commands,
        "run_contract": {
            "short_bracket_launch": True,
            "horizons": horizons,
            "grid": grid,
            "analysis_window": [float(window_tmin), float(window_tmax)],
            "ky": float(ky),
            "dt": float(dt),
            "replicates": [f"seed{seed}" for seed in seed_variants],
            "Nl": int(nl),
            "Nm": int(nm),
            "output_min_samples": int(output_min_samples),
            "output_min_window_samples": int(output_min_window_samples),
            "output_min_abs_window_mean": float(output_min_abs_window_mean),
        },
        "next_action": (
            "run the generated VMEC decks, then write and execute the bounded "
            "nonlinear-gradient short-bracket campaigns; do not promote until "
            "central-FD and replicated-window evidence gates pass"
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
    parser.add_argument("runbook", type=Path)
    parser.add_argument("--baseline-input", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
    parser.add_argument("--case", default="qa_lowres_state_control_short_bracket")
    parser.add_argument("--state-parameter")
    parser.add_argument("--alpha-delta", type=float, default=1.0e-3)
    parser.add_argument("--vmec-command", default="vmec_jax")
    parser.add_argument("--vmec-extra-args", default="")
    parser.add_argument("--horizons", default="150")
    parser.add_argument("--grid", default="n48:48:48:32:32")
    parser.add_argument("--window-tmin", type=float, default=75.0)
    parser.add_argument("--window-tmax", type=float, default=150.0)
    parser.add_argument("--ky", type=float, default=0.47619047619047616)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--baseline-seed", type=int, default=22)
    parser.add_argument("--seed-variant", action="append", type=int, default=None)
    parser.add_argument("--Nl", type=int, default=4)
    parser.add_argument("--Nm", type=int, default=8)
    parser.add_argument("--output-min-samples", type=int, default=60)
    parser.add_argument("--output-min-window-samples", type=int, default=30)
    parser.add_argument("--output-min-abs-window-mean", type=float, default=1.0e-4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runbook = _load_json(args.runbook)
    report = write_state_control_short_bracket_launch(
        runbook=runbook,
        runbook_path=Path(args.runbook),
        baseline_input=Path(args.baseline_input),
        out_dir=Path(args.out_dir),
        out_prefix=Path(args.out_prefix),
        case=str(args.case),
        state_parameter=args.state_parameter,
        alpha_delta=float(args.alpha_delta),
        vmec_command=str(args.vmec_command),
        vmec_extra_args=str(args.vmec_extra_args),
        horizons=str(args.horizons),
        grid=str(args.grid),
        window_tmin=float(args.window_tmin),
        window_tmax=float(args.window_tmax),
        ky=float(args.ky),
        dt=float(args.dt),
        baseline_seed=int(args.baseline_seed),
        seed_variants=tuple(args.seed_variant or (31, 32)),
        nl=int(args.Nl),
        nm=int(args.Nm),
        output_min_samples=int(args.output_min_samples),
        output_min_window_samples=int(args.output_min_window_samples),
        output_min_abs_window_mean=float(args.output_min_abs_window_mean),
    )
    artifacts = _json_clean(report["artifacts"])
    print(
        json.dumps(
            {
                "json": artifacts["json"],
                "launch_count": len(report["launches"]),
                "next_action": report["next_action"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
