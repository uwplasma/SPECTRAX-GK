#!/usr/bin/env python3
"""Write matched VMEC inputs for a multi-coefficient boundary direction.

The output is a launch artifact, not nonlinear-gradient evidence.  It defines a
single scalar perturbation along a normalized vector of VMEC boundary
coefficients, writes baseline/plus/minus VMEC inputs, and records the downstream
long-window nonlinear-gradient campaign command.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, NamedTuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.write_nonlinear_turbulence_gradient_campaign import (  # noqa: E402
    DEFAULT_DT_VARIANT,
    DEFAULT_GRID,
    DEFAULT_HORIZONS,
    DEFAULT_SEEDS,
    DEFAULT_WINDOW,
    _repo_relative,
)
from tools.write_vmec_boundary_perturbation_inputs import (  # noqa: E402
    CoefficientSpec,
    _coefficient_value,
    _json_clean,
    _parse_coefficient_spec,
    _patch_coefficient,
)


CONTROL_RE = re.compile(
    r"^\s*(?P<coefficient>(?:RBC|RBS|ZBC|ZBS)\(\s*[+-]?\d+\s*,\s*[+-]?\d+\s*\))"
    r"\s*(?P<separator>[:=])\s*"
    r"(?P<weight>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)\s*$"
)


class WeightedCoefficient(NamedTuple):
    spec: CoefficientSpec
    weight: float


def _parse_weighted_coefficient(raw: str) -> WeightedCoefficient:
    match = CONTROL_RE.fullmatch(raw)
    if match is None:
        raise ValueError("control must have format RBC(m,n):weight, RBS(m,n):weight, ZBC(m,n):weight, or ZBS(m,n):weight")
    weight = float(match.group("weight").replace("D", "E").replace("d", "e"))
    if not math.isfinite(weight) or weight == 0.0:
        raise ValueError("control weight must be finite and non-zero")
    return WeightedCoefficient(_parse_coefficient_spec(match.group("coefficient")), weight)


def _direction_slug(controls: tuple[WeightedCoefficient, ...]) -> str:
    return "profile_direction_" + "_".join(str(control.spec.slug) for control in controls)


def _input_path(out_dir: Path, case: str, state: str) -> Path:
    return out_dir / f"input.{case}_{state}"


def _wout_path(out_dir: Path, case: str, state: str) -> Path:
    return out_dir / f"wout_{case}_{state}.nc"


def _vmec_command(vmec_command: str, input_path: Path | str) -> str:
    return f"{vmec_command} {_repo_relative(input_path)}"


def _patch_many(text: str, values: dict[CoefficientSpec, float]) -> str:
    patched = text
    for spec, value in values.items():
        patched = _patch_coefficient(patched, spec, value)
    return patched


def write_profile_direction_inputs(
    *,
    baseline_input: Path,
    out_dir: Path,
    case: str,
    controls: tuple[WeightedCoefficient, ...],
    relative_delta: float,
    vmec_command: str = "vmec_jax",
    horizons: str = DEFAULT_HORIZONS,
    grid: str = DEFAULT_GRID,
    window_tmin: float = DEFAULT_WINDOW[0],
    window_tmax: float = DEFAULT_WINDOW[1],
    ky: float = 0.47619047619047616,
    dt: float = 0.05,
    dt_variant: float = DEFAULT_DT_VARIANT,
    baseline_seed: int = 22,
    seed_variants: tuple[int, ...] = DEFAULT_SEEDS,
    nl: int = 4,
    nm: int = 8,
) -> dict[str, Any]:
    """Write a normalized multi-coefficient VMEC boundary direction.

    The largest absolute control weight receives the requested relative
    perturbation; other controls are scaled proportionally.  The scalar
    finite-difference parameter is the Euclidean norm of the coefficient
    perturbation vector, so downstream reports measure dQ/d||delta c|| along
    this direction.
    """

    if len(controls) < 2:
        raise ValueError("profile directions require at least two controls")
    if relative_delta <= 0.0:
        raise ValueError("relative_delta must be positive")
    specs = [control.spec for control in controls]
    if len(set(specs)) != len(specs):
        raise ValueError("control list contains duplicate coefficients")

    text = baseline_input.read_text(encoding="utf-8")
    max_abs_weight = max(abs(control.weight) for control in controls)
    rows: list[dict[str, Any]] = []
    plus_values: dict[CoefficientSpec, float] = {}
    minus_values: dict[CoefficientSpec, float] = {}
    direction_squared_norm = 0.0
    for control in controls:
        base_value = _coefficient_value(text, control.spec)
        if abs(base_value) <= 0.0:
            raise ValueError(f"relative_delta cannot be used with zero baseline coefficient {control.spec.label}")
        normalized_weight = float(control.weight) / max_abs_weight
        delta_value = abs(base_value) * float(relative_delta) * normalized_weight
        plus_values[control.spec] = base_value + delta_value
        minus_values[control.spec] = base_value - delta_value
        direction_squared_norm += delta_value * delta_value
        rows.append(
            {
                "coefficient": control.spec.label,
                "coefficient_slug": control.spec.slug,
                "baseline_value": float(base_value),
                "input_weight": float(control.weight),
                "normalized_weight": float(normalized_weight),
                "coefficient_delta_per_unit_alpha": float(delta_value),
                "relative_coefficient_delta": float(abs(delta_value) / abs(base_value)),
                "plus_value": float(base_value + delta_value),
                "minus_value": float(base_value - delta_value),
            }
        )
    direction_norm = math.sqrt(direction_squared_norm)
    if direction_norm <= 0.0:
        raise ValueError("profile direction has zero coefficient norm")

    out_dir.mkdir(parents=True, exist_ok=True)
    states = {
        "baseline": text,
        "plus_delta": _patch_many(text, plus_values),
        "minus_delta": _patch_many(text, minus_values),
    }
    input_files: dict[str, Path] = {}
    wout_files: dict[str, Path] = {}
    for state, state_text in states.items():
        input_file = _input_path(out_dir, case, state)
        input_file.write_text(state_text, encoding="utf-8")
        input_files[state] = input_file
        wout_files[state] = _wout_path(out_dir, case, state)

    run_commands = {
        state: f"cd {_repo_relative(out_dir)} && {_vmec_command(vmec_command, input_file.name)}"
        for state, input_file in input_files.items()
    }
    parameter_name = _direction_slug(controls)
    campaign_command = (
        "python tools/write_nonlinear_turbulence_gradient_campaign.py "
        f"--baseline-vmec-file {_repo_relative(wout_files['baseline'])} "
        f"--plus-vmec-file {_repo_relative(wout_files['plus_delta'])} "
        f"--minus-vmec-file {_repo_relative(wout_files['minus_delta'])} "
        f"--case {case}_nonlinear_gradient "
        f"--parameter-name {parameter_name} "
        f"--delta-parameter {direction_norm:.16e} "
        f"--out-dir {_repo_relative(out_dir / 'nonlinear_campaign')} "
        f"--horizons {horizons} "
        f"--grid {grid} "
        f"--window-tmin {float(window_tmin):.12g} "
        f"--window-tmax {float(window_tmax):.12g} "
        f"--ky {float(ky):.16g} "
        f"--dt {float(dt):.12g} "
        f"--dt-variant {float(dt_variant):.12g} "
        f"--baseline-seed {int(baseline_seed)} "
        f"--Nl {int(nl)} --Nm {int(nm)}"
    )
    for seed in seed_variants:
        campaign_command += f" --seed-variant {int(seed)}"

    manifest = {
        "kind": "vmec_boundary_profile_direction_perturbation_manifest",
        "claim_level": "multi_coefficient_profile_direction_launch_plan_not_simulation_claim",
        "baseline_input": baseline_input.resolve(),
        "case": str(case),
        "parameter_name": parameter_name,
        "relative_delta": float(relative_delta),
        "normalization": (
            "The largest absolute input weight receives relative_delta times the "
            "baseline coefficient magnitude. Other controls are scaled by "
            "weight/max(abs(weight)). The downstream finite-difference scalar is "
            "the Euclidean norm of the resulting coefficient perturbation vector."
        ),
        "delta_parameter": float(direction_norm),
        "controls": rows,
        "state_input_files": input_files,
        "expected_wout_files": wout_files,
        "vmec_run_commands": run_commands,
        "campaign_command_after_vmec_runs": campaign_command,
        "run_contract": {
            "same_numerics_except_profile_direction": True,
            "horizons": horizons,
            "grid": grid,
            "analysis_window": [float(window_tmin), float(window_tmax)],
            "ky": float(ky),
            "dt": float(dt),
            "dt_variant": float(dt_variant),
            "replicates": [f"seed{seed}" for seed in seed_variants]
            + [f"dt{float(dt_variant):.12g}".replace(".", "p").replace("-", "m")],
        },
        "production_contract": (
            "Run vmec_jax on each input first. Only distinct re-equilibrated wout "
            "files should be passed to the nonlinear turbulence-gradient campaign "
            "writer. Promotion still requires long-window ensemble gates and a "
            "central finite-difference audit."
        ),
    }
    manifest_path = out_dir / "vmec_boundary_profile_direction_manifest.json"
    manifest_path.write_text(
        json.dumps(_json_clean(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest["manifest"] = manifest_path
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-input", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--case", required=True)
    parser.add_argument("--control", action="append", required=True)
    parser.add_argument("--relative-delta", type=float, default=0.03)
    parser.add_argument("--vmec-command", default="vmec_jax")
    parser.add_argument("--horizons", default=DEFAULT_HORIZONS)
    parser.add_argument("--grid", default=DEFAULT_GRID)
    parser.add_argument("--window-tmin", type=float, default=DEFAULT_WINDOW[0])
    parser.add_argument("--window-tmax", type=float, default=DEFAULT_WINDOW[1])
    parser.add_argument("--ky", type=float, default=0.47619047619047616)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--dt-variant", type=float, default=DEFAULT_DT_VARIANT)
    parser.add_argument("--baseline-seed", type=int, default=22)
    parser.add_argument("--seed-variant", action="append", type=int, default=None)
    parser.add_argument("--Nl", type=int, default=4)
    parser.add_argument("--Nm", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = write_profile_direction_inputs(
        baseline_input=Path(args.baseline_input),
        out_dir=Path(args.out_dir),
        case=str(args.case),
        controls=tuple(_parse_weighted_coefficient(raw) for raw in args.control),
        relative_delta=float(args.relative_delta),
        vmec_command=str(args.vmec_command),
        horizons=str(args.horizons),
        grid=str(args.grid),
        window_tmin=float(args.window_tmin),
        window_tmax=float(args.window_tmax),
        ky=float(args.ky),
        dt=float(args.dt),
        dt_variant=float(args.dt_variant),
        baseline_seed=int(args.baseline_seed),
        seed_variants=tuple(args.seed_variant or DEFAULT_SEEDS),
        nl=int(args.Nl),
        nm=int(args.Nm),
    )
    print(
        json.dumps(
            {
                "manifest": _repo_relative(manifest["manifest"]),
                "case": manifest["case"],
                "parameter_name": manifest["parameter_name"],
                "delta_parameter": manifest["delta_parameter"],
                "control_count": len(manifest["controls"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
