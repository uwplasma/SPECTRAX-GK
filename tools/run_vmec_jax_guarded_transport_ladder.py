#!/usr/bin/env python3
"""Run a fail-closed VMEC-JAX/SPECTRAX-GK transport-weight ladder.

The ladder starts from a solved QA candidate directory containing ``input.final``.
Each transport-weight refinement is run in its own output directory with
``--allow-failed-solved-wout-gate`` so failed branches remain inspectable. The
promotion rule is intentionally conservative: only candidates whose
``solved_wout_gate.json`` passes and whose lower-is-better transport metric
improves relative to the QA baseline may be selected. A separate long-window
nonlinear SPECTRAX-GK audit is still required before making turbulent-flux
claims.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spectraxgk.vmec_jax_candidate_gate import build_solved_vmec_candidate_gate  # type: ignore[import-untyped] # noqa: E402
from spectraxgk.vmec_jax_transport_admission import (  # type: ignore[import-untyped] # noqa: E402
    DEFAULT_TRANSPORT_METRIC_KEYS,
    VMECJAXTransportAdmissionPolicy,
    build_transport_admission_report,
)

DEFAULT_DRIVER = ROOT / "tools" / "vmec_jax_qa_low_turbulence_optimization.py"
DEFAULT_OUTDIR = ROOT / "tools_out" / "vmec_jax_guarded_transport_ladder"
DEFAULT_WEIGHTS = (5.0e-4, 1.0e-3, 2.5e-3, 5.0e-3, 1.0e-2)


def _read_json_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _history_with_transport_metric(history: dict[str, Any], metric_path: Path | None) -> dict[str, Any]:
    """Return history augmented by an explicit eval-only transport metric."""

    if metric_path is None:
        return history
    metric = _read_json_object(metric_path)
    merged = dict(history)
    for key in (
        "transport_objective_final",
        "spectrax_objective_final",
        "transport_metric_final",
        "transport_metric_kind",
        "transport_objective_source",
    ):
        if key in metric:
            merged[key] = metric[key]
    merged["transport_metric_json"] = str(metric_path)
    if "sample_set" in metric:
        merged["transport_metric_sample_set"] = metric["sample_set"]
    return merged


def _weight_token(weight: float) -> str:
    token = f"{float(weight):.8g}".replace("-", "m").replace(".", "p")
    return token.replace("+", "")


def _parse_weights(raw: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated weight")
    if any(weight <= 0.0 for weight in values):
        raise argparse.ArgumentTypeError("transport weights must be positive")
    return values


def _load_wout_iota_profiles(root: Path) -> tuple[Any, Any] | None:
    wout_path = root / "wout_final.nc"
    if not wout_path.exists():
        return None
    try:
        import vmec_jax as vj  # type: ignore[import-not-found]

        wout = vj.load_wout(wout_path)
        return getattr(wout, "iotas"), getattr(wout, "iotaf")
    except Exception:
        return None


def _candidate_gate_from_artifacts(
    root: Path,
    history: dict[str, Any],
    *,
    target_aspect: float,
    aspect_atol: float,
    min_abs_mean_iota: float,
    qs_residual_max: float,
    iota_profile_floor: float | None,
) -> tuple[dict[str, Any], str | None]:
    gate_path = root / "solved_wout_gate.json"
    if gate_path.exists():
        return _read_json_object(gate_path), str(gate_path)
    if not history:
        return {}, None
    profiles = _load_wout_iota_profiles(root)
    source = "wout_final.nc" if profiles is not None else "missing"
    return (
        build_solved_vmec_candidate_gate(
            history,
            target_aspect=target_aspect,
            aspect_atol=aspect_atol,
            min_abs_mean_iota=min_abs_mean_iota,
            qs_residual_max=qs_residual_max,
            iota_profile_floor=iota_profile_floor,
            iota_profiles=profiles,
            profile_source=source,
        ),
        None,
    )


def candidate_summary(
    root: Path,
    *,
    label: str,
    weight: float | None = None,
    baseline: bool = False,
    target_aspect: float = 6.0,
    aspect_atol: float = 5.0e-2,
    min_abs_mean_iota: float = 0.41,
    qs_residual_max: float = 5.0e-2,
    iota_profile_floor: float | None = 0.41,
    allow_reconstructed_gate: bool = False,
    transport_metric_json: Path | None = None,
) -> dict[str, Any]:
    """Return a compact promotion summary for a solved candidate directory."""

    history_path = root / "history.json"
    history = _read_json_object(history_path) if history_path.exists() else {}
    history = _history_with_transport_metric(history, transport_metric_json)
    gate, gate_path = _candidate_gate_from_artifacts(
        root,
        history,
        target_aspect=target_aspect,
        aspect_atol=aspect_atol,
        min_abs_mean_iota=min_abs_mean_iota,
        qs_residual_max=qs_residual_max,
        iota_profile_floor=iota_profile_floor,
    )
    gate_source = "solved_wout_gate.json" if gate_path is not None else ("reconstructed" if gate else None)
    gate_is_authoritative = gate_source == "solved_wout_gate.json" or (
        gate_source == "reconstructed" and bool(allow_reconstructed_gate)
    )
    gate_reported_passed = bool(gate.get("passed", False))
    passed = bool(gate_reported_passed and gate_is_authoritative)
    next_action = gate.get("next_action")
    if gate_source == "reconstructed" and not bool(allow_reconstructed_gate):
        next_action = (
            "reconstructed history/WOUT gate is advisory only; generate a fresh solved_wout_gate.json "
            "or rerun with --allow-reconstructed-gate for exploratory dry runs"
        )
    return {
        "label": label,
        "root": str(root),
        "baseline": bool(baseline),
        "transport_weight": None if weight is None else float(weight),
        "history_path": str(history_path) if history_path.exists() else None,
        "gate_path": gate_path,
        "gate_source": gate_source,
        "gate_is_authoritative": bool(gate_is_authoritative),
        "gate_reported_passed": gate_reported_passed,
        "passed": passed,
        "objective_final": history.get("objective_final"),
        "transport_objective_final": history.get("transport_objective_final"),
        "spectrax_objective_final": history.get("spectrax_objective_final"),
        "transport_metric_final": history.get("transport_metric_final"),
        "transport_metric_kind": history.get("transport_metric_kind"),
        "transport_metric_json": history.get("transport_metric_json"),
        "transport_metric_sample_set": history.get("transport_metric_sample_set"),
        "aspect_final": history.get("aspect_final"),
        "iota_final": history.get("iota_final"),
        "qs_final": history.get("qs_final"),
        "gate_checks": {name: check.get("passed") for name, check in gate.get("checks", {}).items()},
        "next_action": next_action,
    }


def select_promoted_candidate(
    summaries: list[dict[str, Any]],
    *,
    policy: VMECJAXTransportAdmissionPolicy | None = None,
) -> dict[str, Any] | None:
    """Select a physically admitted transport candidate, or the QA baseline."""

    report = build_transport_admission_report(summaries, policy=policy)
    promoted = report.get("promoted_candidate")
    return dict(promoted) if isinstance(promoted, dict) else None


def build_driver_command(
    *,
    python_executable: str,
    driver: Path,
    input_file: Path,
    outdir: Path,
    weight: float,
    driver_args: tuple[str, ...],
) -> list[str]:
    """Build one fail-closed transport-refinement command."""

    return [
        python_executable,
        str(driver),
        "--input",
        str(input_file),
        "--outdir",
        str(outdir),
        "--disable-mode-continuation",
        "--spectrax-weight",
        f"{float(weight):.16g}",
        "--allow-failed-solved-wout-gate",
        *driver_args,
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--constraints-dir", type=Path, required=True, help="Passing QA-only candidate directory")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="Transport ladder output root")
    parser.add_argument("--weights", type=_parse_weights, default=DEFAULT_WEIGHTS)
    parser.add_argument("--driver", type=Path, default=DEFAULT_DRIVER)
    parser.add_argument("--python", default=sys.executable, help="Python executable used for candidate runs")
    parser.add_argument(
        "--driver-args",
        default="",
        help="Additional arguments passed to the VMEC-JAX QA driver, parsed with shlex.split",
    )
    parser.add_argument(
        "--baseline-metric-json",
        type=Path,
        default=None,
        help=(
            "Optional eval-only SPECTRAX transport metric JSON for the constraints baseline. "
            "Use this when the baseline run was constraints-only and its history objective is not a transport metric."
        ),
    )
    parser.add_argument("--target-aspect", type=float, default=6.0, help="Solved-candidate aspect target")
    parser.add_argument("--aspect-atol", type=float, default=5.0e-2, help="Solved-candidate aspect tolerance")
    parser.add_argument("--min-iota", type=float, default=0.41, help="Minimum accepted absolute mean iota")
    parser.add_argument("--qs-max", type=float, default=5.0e-2, help="Maximum accepted QS residual")
    parser.add_argument("--iota-profile-floor", type=float, default=0.41, help="Minimum accepted solved iota profile")
    parser.add_argument(
        "--disable-iota-profile-floor",
        action="store_true",
        help=(
            "Disable profile-floor admission and forward the same convention to the candidate driver. "
            "Use this for strict upstream-QA baselines that gate high-weight mean iota instead."
        ),
    )
    parser.add_argument(
        "--allow-reconstructed-gate",
        action="store_true",
        help=(
            "Treat a gate reconstructed from history.json plus WOUT iota profiles as promotable. "
            "Default is fail-closed because history and restart-input QS conventions can drift."
        ),
    )
    parser.add_argument(
        "--continue-after-failed-gate",
        action="store_true",
        help="Continue trying larger transport weights after a transport candidate fails the solved-candidate gate.",
    )
    parser.add_argument(
        "--transport-metric-key",
        action="append",
        default=None,
        help=(
            "Candidate summary key used for lower-is-better transport admission. May be repeated. "
            f"Default order: {', '.join(DEFAULT_TRANSPORT_METRIC_KEYS)}."
        ),
    )
    parser.add_argument(
        "--min-transport-improvement",
        type=float,
        default=0.0,
        help="Minimum relative transport-metric improvement required before a transport branch is admitted.",
    )
    parser.add_argument("--timeout-s", type=float, default=0.0, help="Per-candidate subprocess timeout; 0 disables")
    parser.add_argument("--dry-run", action="store_true", help="Write the launch plan without running candidates")
    parser.add_argument("--out-json", type=Path, default=None, help="Summary JSON path; defaults inside --outdir")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    constraints_dir = Path(args.constraints_dir)
    input_file = constraints_dir / "input.final"
    if not input_file.exists():
        raise FileNotFoundError(f"missing constraints restart input: {input_file}")
    driver_args = tuple(shlex.split(str(args.driver_args)))
    if bool(args.disable_iota_profile_floor) and "--disable-iota-profile-floor" not in driver_args:
        driver_args = (*driver_args, "--disable-iota-profile-floor")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    commands: list[dict[str, Any]] = []
    gate_policy = {
        "target_aspect": float(args.target_aspect),
        "aspect_atol": float(args.aspect_atol),
        "min_abs_mean_iota": float(args.min_iota),
        "qs_residual_max": float(args.qs_max),
        "iota_profile_floor": None if bool(args.disable_iota_profile_floor) else float(args.iota_profile_floor),
    }
    summaries = [
        candidate_summary(
            constraints_dir,
            label="QA constraints restart",
            baseline=True,
            target_aspect=gate_policy["target_aspect"],
            aspect_atol=gate_policy["aspect_atol"],
            min_abs_mean_iota=gate_policy["min_abs_mean_iota"],
            qs_residual_max=gate_policy["qs_residual_max"],
            iota_profile_floor=gate_policy["iota_profile_floor"],
            allow_reconstructed_gate=bool(args.allow_reconstructed_gate),
            transport_metric_json=Path(args.baseline_metric_json) if args.baseline_metric_json is not None else None,
        )
    ]
    run_failures: list[dict[str, Any]] = []
    stopped_after_failed_gate = False
    for weight in tuple(float(w) for w in args.weights):
        candidate_dir = outdir / f"transport_weight_{_weight_token(weight)}"
        command = build_driver_command(
            python_executable=str(args.python),
            driver=Path(args.driver),
            input_file=input_file,
            outdir=candidate_dir,
            weight=weight,
            driver_args=driver_args,
        )
        commands.append({"transport_weight": weight, "outdir": str(candidate_dir), "command": command})
        if not bool(args.dry_run):
            try:
                subprocess.run(
                    command,
                    cwd=ROOT,
                    check=True,
                    timeout=None if float(args.timeout_s) <= 0.0 else float(args.timeout_s),
                )
            except subprocess.CalledProcessError as exc:
                run_failures.append({"transport_weight": weight, "outdir": str(candidate_dir), "returncode": exc.returncode})
            except subprocess.TimeoutExpired:
                run_failures.append({"transport_weight": weight, "outdir": str(candidate_dir), "timeout_s": float(args.timeout_s)})
            summary = candidate_summary(
                candidate_dir,
                label=f"transport weight {weight:.3g}",
                weight=weight,
                target_aspect=gate_policy["target_aspect"],
                aspect_atol=gate_policy["aspect_atol"],
                min_abs_mean_iota=gate_policy["min_abs_mean_iota"],
                qs_residual_max=gate_policy["qs_residual_max"],
                iota_profile_floor=gate_policy["iota_profile_floor"],
                allow_reconstructed_gate=bool(args.allow_reconstructed_gate),
            )
            summaries.append(summary)
            if not bool(summary.get("passed")) and not bool(args.continue_after_failed_gate):
                stopped_after_failed_gate = True
                break
    admission_policy = VMECJAXTransportAdmissionPolicy(
        metric_keys=tuple(args.transport_metric_key or DEFAULT_TRANSPORT_METRIC_KEYS),
        minimum_relative_improvement=float(args.min_transport_improvement),
        lower_is_better=True,
        require_authoritative_gate=not bool(args.allow_reconstructed_gate),
        allow_baseline_fallback=True,
    )
    transport_admission = build_transport_admission_report(summaries, policy=admission_policy)
    promoted = transport_admission.get("promoted_candidate")
    payload = {
        "kind": "vmec_jax_guarded_transport_ladder",
        "claim_scope": (
            "solved-candidate transport-weight admission only; candidates must pass physical gates "
            "and improve the selected transport metric before long-window nonlinear SPECTRAX-GK audits"
        ),
        "constraints_dir": str(constraints_dir),
        "restart_input": str(input_file),
        "baseline_metric_json": None if args.baseline_metric_json is None else str(args.baseline_metric_json),
        "gate_policy": gate_policy,
        "transport_admission_policy": admission_policy.to_dict(),
        "allow_reconstructed_gate": bool(args.allow_reconstructed_gate),
        "continue_after_failed_gate": bool(args.continue_after_failed_gate),
        "stopped_after_failed_gate": bool(stopped_after_failed_gate),
        "dry_run": bool(args.dry_run),
        "commands": commands,
        "candidates": summaries,
        "run_failures": run_failures,
        "transport_admission": transport_admission,
        "promoted_candidate": promoted,
        "transport_candidate_admitted": bool(transport_admission.get("transport_candidate_admitted")),
        "passed": promoted is not None,
        "next_action": transport_admission.get("next_action"),
    }
    out_json = Path(args.out_json) if args.out_json is not None else outdir / "guarded_transport_ladder.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": payload["passed"], "out_json": str(out_json), "promoted_candidate": promoted}, indent=2))
    return 0 if bool(payload["passed"]) or bool(args.dry_run) else 1


if __name__ == "__main__":
    raise SystemExit(main())
