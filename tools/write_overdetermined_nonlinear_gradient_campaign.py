#!/usr/bin/env python3
"""Write a multi-control nonlinear turbulence-gradient campaign launch plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.write_vmec_boundary_perturbation_inputs import (  # noqa: E402
    CoefficientSpec,
    _json_clean,
    _parse_coefficient_spec,
    write_perturbation_inputs,
)
from tools.write_nonlinear_turbulence_gradient_campaign import (  # noqa: E402
    DEFAULT_DT_VARIANT,
    DEFAULT_GRID,
    DEFAULT_HORIZONS,
    DEFAULT_SEEDS,
    DEFAULT_WINDOW,
    _repo_relative,
)


DEFAULT_OUT_DIR = ROOT / "tools_out" / "overdetermined_nonlinear_gradient_campaign"


def _coefficient_slug(spec: CoefficientSpec) -> str:
    return str(spec.slug).replace("__", "_")


def _fd_artifact(case: str, coefficient: CoefficientSpec) -> Path:
    slug = _coefficient_slug(coefficient)
    return ROOT / "docs" / "_static" / f"{case}_{slug}_nonlinear_gradient_{slug}_central_fd_gradient_gate.json"


def _nonlinear_campaign_command(
    *,
    case: str,
    coefficient: CoefficientSpec,
    vmec_manifest: dict[str, Any],
    campaign_out_dir: Path,
    horizons: str,
    grid: str,
    window_tmin: float,
    window_tmax: float,
    ky: float,
    dt: float,
    dt_variant: float,
    baseline_seed: int,
    seed_variants: tuple[int, ...],
    nl: int,
    nm: int,
) -> str:
    slug = _coefficient_slug(coefficient)
    state_wouts = vmec_manifest["expected_wout_files"]
    command = (
        "python tools/write_nonlinear_turbulence_gradient_campaign.py "
        f"--baseline-vmec-file {state_wouts['baseline']} "
        f"--plus-vmec-file {state_wouts['plus_delta']} "
        f"--minus-vmec-file {state_wouts['minus_delta']} "
        f"--case {case}_{slug}_nonlinear_gradient "
        f"--parameter-name {slug} "
        f"--delta-parameter {float(vmec_manifest['delta_parameter']):.16e} "
        f"--out-dir {_repo_relative(campaign_out_dir / slug)} "
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
        command += f" --seed-variant {int(seed)}"
    return command


def _load_previous_ranking(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return payload


def write_overdetermined_campaign(
    *,
    baseline_input: Path,
    out_dir: Path,
    case: str,
    coefficients: tuple[CoefficientSpec, ...],
    relative_delta: float,
    vmec_command: str = "vmec_jax",
    campaign_out_dir: Path | None = None,
    previous_ranking: Path | None = None,
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
    if len(coefficients) < 2:
        raise ValueError("overdetermined nonlinear-gradient campaigns require at least two controls")
    if relative_delta <= 0.0:
        raise ValueError("relative_delta must be positive")
    if len({_coefficient_slug(spec) for spec in coefficients}) != len(coefficients):
        raise ValueError("coefficient list contains duplicate controls")

    out_dir.mkdir(parents=True, exist_ok=True)
    campaign_root = campaign_out_dir or (out_dir / "nonlinear_campaigns")
    previous = _load_previous_ranking(previous_ranking)
    controls: list[dict[str, Any]] = []
    fd_artifacts: list[Path] = []
    for coefficient in coefficients:
        slug = _coefficient_slug(coefficient)
        control_case = f"{case}_{slug}"
        control_out_dir = out_dir / slug
        vmec_manifest = write_perturbation_inputs(
            baseline_input=baseline_input,
            out_dir=control_out_dir,
            case=control_case,
            coefficient=coefficient,
            relative_delta=relative_delta,
            vmec_command=vmec_command,
        )
        fd_json = _fd_artifact(case, coefficient)
        fd_artifacts.append(fd_json)
        controls.append(
            {
                "coefficient": coefficient.label,
                "coefficient_slug": slug,
                "case": control_case,
                "vmec_manifest": _repo_relative(vmec_manifest["manifest"]),
                "delta_parameter": float(vmec_manifest["delta_parameter"]),
                "state_input_files": vmec_manifest["state_input_files"],
                "expected_wout_files": vmec_manifest["expected_wout_files"],
                "vmec_run_commands": vmec_manifest["vmec_run_commands"],
                "nonlinear_campaign_command_after_vmec_runs": _nonlinear_campaign_command(
                    case=case,
                    coefficient=coefficient,
                    vmec_manifest=vmec_manifest,
                    campaign_out_dir=campaign_root,
                    horizons=horizons,
                    grid=grid,
                    window_tmin=window_tmin,
                    window_tmax=window_tmax,
                    ky=ky,
                    dt=dt,
                    dt_variant=dt_variant,
                    baseline_seed=baseline_seed,
                    seed_variants=seed_variants,
                    nl=nl,
                    nm=nm,
                ),
                "expected_fd_artifact": _repo_relative(fd_json),
            }
        )

    ranking_json = ROOT / "docs" / "_static" / f"{case}_overdetermined_nonlinear_gradient_candidate_ranking.json"
    ranking_command = (
        "python tools/rank_nonlinear_turbulence_gradient_candidates.py "
        + " ".join(_repo_relative(path) for path in fd_artifacts)
        + f" --json-out {_repo_relative(ranking_json)}"
        + " --fail-on-no-promotable"
    )
    manifest = {
        "kind": "overdetermined_nonlinear_turbulence_gradient_campaign_manifest",
        "claim_level": "multi_control_profile_gradient_launch_plan_not_simulation_claim",
        "case": str(case),
        "baseline_input": baseline_input,
        "relative_delta": float(relative_delta),
        "control_count": len(controls),
        "controls": controls,
        "previous_ranking": None
        if previous is None
        else {
            "path": _repo_relative(previous_ranking) if previous_ranking is not None else None,
            "passed": bool(previous.get("passed", False)),
            "recommendation": str(previous.get("recommendation", "")),
            "best_candidate": previous.get("best_candidate"),
        },
        "run_contract": {
            "same_numerics_except_parameter": True,
            "overdetermined_controls": True,
            "horizons": horizons,
            "grid": grid,
            "analysis_window": [float(window_tmin), float(window_tmax)],
            "ky": float(ky),
            "dt": float(dt),
            "dt_variant": float(dt_variant),
            "replicates": [f"seed{seed}" for seed in seed_variants]
            + [f"dt{float(dt_variant):.12g}".replace(".", "p").replace("-", "m")],
        },
        "promotion_contract": {
            "claim_boundary": (
                "This manifest only launches an overdetermined campaign. "
                "Production nonlinear turbulence-gradient evidence still requires "
                "real re-equilibrated VMEC files, matched long post-transient nonlinear "
                "replicates, per-control central-FD gates, and a ranking/evidence report "
                "with at least one control passing all locality and uncertainty gates."
            ),
            "expected_fd_artifacts": [_repo_relative(path) for path in fd_artifacts],
            "candidate_ranking_json": _repo_relative(ranking_json),
            "candidate_ranking_command": ranking_command,
        },
    }
    manifest_path = out_dir / "overdetermined_nonlinear_gradient_campaign_manifest.json"
    manifest_path.write_text(json.dumps(_json_clean(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["manifest"] = manifest_path
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-input", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--case", required=True)
    parser.add_argument("--coefficient", action="append", required=True)
    parser.add_argument("--relative-delta", type=float, default=0.05)
    parser.add_argument("--vmec-command", default="vmec_jax")
    parser.add_argument("--campaign-out-dir", type=Path)
    parser.add_argument("--previous-ranking", type=Path)
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
    manifest = write_overdetermined_campaign(
        baseline_input=Path(args.baseline_input),
        out_dir=Path(args.out_dir),
        case=str(args.case),
        coefficients=tuple(_parse_coefficient_spec(raw) for raw in args.coefficient),
        relative_delta=float(args.relative_delta),
        vmec_command=str(args.vmec_command),
        campaign_out_dir=args.campaign_out_dir,
        previous_ranking=args.previous_ranking,
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
                "control_count": manifest["control_count"],
                "ranking_command": manifest["promotion_contract"]["candidate_ranking_command"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
