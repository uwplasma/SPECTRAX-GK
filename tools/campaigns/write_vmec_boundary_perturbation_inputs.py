#!/usr/bin/env python3
"""Write matched VMEC input perturbations for nonlinear-gradient campaigns."""

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

from tools.campaigns.write_nonlinear_turbulence_gradient_campaign import _repo_relative  # noqa: E402
from tools.campaigns.write_nonlinear_turbulence_gradient_campaign import PYTHON_CMD  # noqa: E402


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-input", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--case", required=True)
    parser.add_argument("--coefficient", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--delta", type=float)
    group.add_argument("--relative-delta", type=float)
    parser.add_argument("--vmec-command", default="vmec_jax")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
