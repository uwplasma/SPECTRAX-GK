#!/usr/bin/env python3
"""Write matched VMEC boundary perturbation campaign inputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, NamedTuple


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.campaigns.write_nonlinear_turbulence_gradient_campaign import (  # noqa: E402
    DEFAULT_DT_VARIANT,
    DEFAULT_GRID,
    DEFAULT_HORIZONS,
    DEFAULT_SEEDS,
    DEFAULT_WINDOW,
    PYTHON_CMD,
    _repo_relative,
)


COEFFICIENT_RE = re.compile(
    r"(?P<family>RBC|RBS|ZBC|ZBS)\("
    r"\s*(?P<m>[+-]?\d+)\s*,\s*(?P<n>[+-]?\d+)\s*\)\s*=\s*"
    r"(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"
)


class CoefficientSpec(NamedTuple):
    family: str
    m: int
    n: int

    @property
    def label(self) -> str:
        return f"{self.family}({self.m},{self.n})"

    @property
    def slug(self) -> str:
        return (
            self.label.replace("(", "_")
            .replace(")", "")
            .replace(",", "_")
            .replace("-", "m")
            .lower()
        )


def _parse_coefficient_spec(raw: str) -> CoefficientSpec:
    match = re.fullmatch(
        r"\s*(RBC|RBS|ZBC|ZBS)\(\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*\)\s*", raw
    )
    if match is None:
        raise ValueError(
            "coefficient must have format RBC(m,n), RBS(m,n), ZBC(m,n), or ZBS(m,n)"
        )
    return CoefficientSpec(match.group(1), int(match.group(2)), int(match.group(3)))


def _coefficient_rows(
    text: str, spec: CoefficientSpec
) -> list[tuple[int, re.Match[str]]]:
    rows: list[tuple[int, re.Match[str]]] = []
    for idx, line in enumerate(text.splitlines(keepends=True)):
        active_line = line.split("!", 1)[0]
        for match in COEFFICIENT_RE.finditer(active_line):
            row_spec = CoefficientSpec(
                match.group("family"), int(match.group("m")), int(match.group("n"))
            )
            if row_spec == spec:
                rows.append((idx, match))
    return rows


def _fortran_float(value: float) -> str:
    return f"{float(value):.16E}"


def _patch_coefficient(text: str, spec: CoefficientSpec, value: float) -> str:
    lines = text.splitlines(keepends=True)
    rows = _coefficient_rows(text, spec)
    if not rows:
        raise ValueError(f"coefficient {spec.label} not found")
    if len(rows) > 1:
        raise ValueError(
            f"coefficient {spec.label} appears {len(rows)} times; refusing ambiguous patch"
        )
    idx, match = rows[0]
    value_start = match.start("value")
    value_end = match.end("value")
    lines[idx] = (
        f"{lines[idx][:value_start]}{_fortran_float(value)}{lines[idx][value_end:]}"
    )
    return "".join(lines)


def _coefficient_value(text: str, spec: CoefficientSpec) -> float:
    rows = _coefficient_rows(text, spec)
    if not rows:
        raise ValueError(f"coefficient {spec.label} not found")
    if len(rows) > 1:
        raise ValueError(
            f"coefficient {spec.label} appears {len(rows)} times; refusing ambiguous read"
        )
    return float(rows[0][1].group("value").replace("D", "E").replace("d", "e"))


def _float_label(value: float) -> str:
    return f"{float(value):.12g}".replace(".", "p").replace("-", "m")


def _input_path(out_dir: Path, case: str, state: str) -> Path:
    return out_dir / f"input.{case}_{state}"


def _wout_path(out_dir: Path, case: str, state: str) -> Path:
    return out_dir / f"wout_{case}_{state}.nc"


def _vmec_command(vmec_command: str, input_path: Path | str) -> str:
    return f"{vmec_command} {_repo_relative(input_path)}"


def write_perturbation_inputs(
    *,
    baseline_input: Path,
    out_dir: Path,
    case: str,
    coefficient: CoefficientSpec,
    delta: float | None = None,
    relative_delta: float | None = None,
    vmec_command: str = "vmec_jax",
) -> dict[str, Any]:
    if (delta is None) == (relative_delta is None):
        raise ValueError("provide exactly one of delta or relative_delta")
    text = baseline_input.read_text(encoding="utf-8")
    base_value = _coefficient_value(text, coefficient)
    if relative_delta is not None:
        if not math.isfinite(float(relative_delta)):
            raise ValueError("relative_delta must be finite")
        if abs(base_value) <= 0.0:
            raise ValueError(
                "relative_delta cannot be used with a zero baseline coefficient"
            )
        delta_value = abs(base_value) * float(relative_delta)
    else:
        assert delta is not None
        if not math.isfinite(float(delta)):
            raise ValueError("coefficient perturbation delta must be finite")
        delta_value = float(delta)
    if not math.isfinite(delta_value) or delta_value <= 0.0:
        raise ValueError("coefficient perturbation delta must be positive")

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
        input_file.write_text(
            _patch_coefficient(text, coefficient, value), encoding="utf-8"
        )
        input_files[state] = input_file
        wout_files[state] = _wout_path(out_dir, case, state)

    run_commands = {
        state: f"cd {_repo_relative(out_dir)} && {_vmec_command(vmec_command, input_file.name)}"
        for state, input_file in input_files.items()
    }
    gradient_command = (
        f"{PYTHON_CMD} tools/campaigns/write_nonlinear_turbulence_gradient_campaign.py "
        f"--baseline-vmec-file {_repo_relative(wout_files['baseline'])} "
        f"--plus-vmec-file {_repo_relative(wout_files['plus_delta'])} "
        f"--minus-vmec-file {_repo_relative(wout_files['minus_delta'])} "
        f"--case {case}_nonlinear_gradient "
        f"--parameter-name {coefficient.slug} "
        f"--delta-parameter {delta_value:.16e}"
    )
    manifest = {
        "kind": "vmec_boundary_perturbation_input_manifest",
        "claim_level": "real_vmec_reequilibration_launch_plan_not_simulation_claim",
        "baseline_input": baseline_input.resolve(),
        "case": str(case),
        "coefficient": coefficient.label,
        "coefficient_slug": coefficient.slug,
        "baseline_value": float(base_value),
        "delta_parameter": float(delta_value),
        "relative_delta": None if relative_delta is None else float(relative_delta),
        "state_values": {state: float(value) for state, value in states.items()},
        "state_input_files": input_files,
        "expected_wout_files": wout_files,
        "vmec_run_commands": run_commands,
        "campaign_command_after_vmec_runs": gradient_command,
        "production_contract": (
            "Run vmec_jax on each input first. Only the resulting distinct wout files "
            "should be passed to the nonlinear turbulence-gradient campaign writer."
        ),
    }
    manifest_path = out_dir / "vmec_boundary_perturbation_manifest.json"
    manifest_path.write_text(
        json.dumps(_json_clean(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest["manifest"] = manifest_path
    return manifest


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, Path):
        return _repo_relative(value)
    return value


def build_single_coefficient_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write matched VMEC inputs for one boundary coefficient.")
    parser.add_argument("--baseline-input", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--case", required=True)
    parser.add_argument("--coefficient", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--delta", type=float)
    group.add_argument("--relative-delta", type=float)
    parser.add_argument("--vmec-command", default="vmec_jax")
    return parser


def main_single_coefficient(argv: list[str] | None = None) -> int:
    args = build_single_coefficient_parser().parse_args(argv)
    manifest = write_perturbation_inputs(
        baseline_input=Path(args.baseline_input),
        out_dir=Path(args.out_dir),
        case=str(args.case),
        coefficient=_parse_coefficient_spec(str(args.coefficient)),
        delta=args.delta,
        relative_delta=args.relative_delta,
        vmec_command=str(args.vmec_command),
    )
    print(
        json.dumps(
            {
                "manifest": _repo_relative(manifest["manifest"]),
                "case": manifest["case"],
                "coefficient": manifest["coefficient"],
                "delta_parameter": manifest["delta_parameter"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0



# Multi-coefficient profile-direction campaign support.
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
        raise ValueError(
            "control must have format RBC(m,n):weight, RBS(m,n):weight, ZBC(m,n):weight, or ZBS(m,n):weight"
        )
    weight = float(match.group("weight").replace("D", "E").replace("d", "e"))
    if not math.isfinite(weight) or weight == 0.0:
        raise ValueError("control weight must be finite and non-zero")
    return WeightedCoefficient(
        _parse_coefficient_spec(match.group("coefficient")), weight
    )


def _direction_slug(controls: tuple[WeightedCoefficient, ...]) -> str:
    return "profile_direction_" + "_".join(
        str(control.spec.slug) for control in controls
    )


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
    if not math.isfinite(float(relative_delta)) or relative_delta <= 0.0:
        raise ValueError("relative_delta must be finite and positive")
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
            raise ValueError(
                f"relative_delta cannot be used with zero baseline coefficient {control.spec.label}"
            )
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
    if not math.isfinite(direction_norm) or direction_norm <= 0.0:
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
        f"{PYTHON_CMD} tools/campaigns/write_nonlinear_turbulence_gradient_campaign.py "
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


def build_profile_direction_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write matched VMEC inputs for a multi-coefficient boundary direction.")
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


def main_profile_direction(argv: list[str] | None = None) -> int:
    args = build_profile_direction_parser().parse_args(argv)
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



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser(
        "single-coefficient",
        help="Write baseline/plus/minus inputs for one VMEC boundary coefficient.",
        parents=[build_single_coefficient_parser()],
        add_help=False,
    )
    single.set_defaults(func=main_single_coefficient)

    profile = subparsers.add_parser(
        "profile-direction",
        help="Write baseline/plus/minus inputs for a weighted coefficient direction.",
        parents=[build_profile_direction_parser()],
        add_help=False,
    )
    profile.set_defaults(func=main_profile_direction)
    return parser

def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if not raw or raw[0] in {"-h", "--help"}:
        build_parser().print_help()
        return 0
    command = raw.pop(0)
    runners = {
        "single-coefficient": main_single_coefficient,
        "profile-direction": main_profile_direction,
    }
    try:
        runner = runners[command]
    except KeyError as exc:
        raise SystemExit(f"unknown VMEC boundary campaign subcommand: {command}") from exc
    return int(runner(raw))

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
