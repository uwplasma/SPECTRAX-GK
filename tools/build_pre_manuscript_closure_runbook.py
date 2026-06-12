#!/usr/bin/env python3
"""Build an actionable runbook for strict pre-manuscript closure lanes.

The closure-status dashboard says whether broad manuscript claims are closed.
This companion artifact says what can be launched now, what is already running,
and what is scientifically blocked until a prerequisite screen or gate exists.
It intentionally fails closed for nonlinear holdouts: an unchanged replay of an
already represented or failed VMEC family is not a new absolute-flux holdout.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectraxgk.plotting import set_plot_style  # noqa: E402
from tools.build_pre_manuscript_closure_status import build_status_payload  # noqa: E402


DEFAULT_OUT = ROOT / "docs" / "_static" / "pre_manuscript_closure_runbook.png"
DEFAULT_INVENTORY = ROOT / "docs" / "_static" / "vmec_jax_equilibrium_inventory.json"
DEFAULT_SCREEN = ROOT / "docs" / "_static" / "external_vmec_candidate_linear_screen.csv"
DEFAULT_EXTERNAL_RUNBOOK = ROOT / "docs" / "_static" / "external_vmec_next_holdout_runbook.json"
DEFAULT_OPTIMIZER_MANIFEST = ROOT / "docs" / "_static" / "vmec_jax_qa_optimizer_comparison_manifest.json"
DEFAULT_LADDER_STATUS = ROOT / "docs" / "_static" / "vmec_jax_qa_optimizer_ladder_resume_status.json"
DEFAULT_OFFICE_ROOT = Path("/home/rjorge/spectrax_optimizer_ladder_20260609/SPECTRAX-GK")
DEFAULT_AUDIT_ROOT = Path("tools_out/pre_manuscript_nonlinear_audits")
MIN_LINEAR_LAUNCH_GAMMA = 0.02


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return _json_clean(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _slug(value: object) -> str:
    text = str(value).lower()
    text = text.removeprefix("wout_").removesuffix(".nc").removesuffix("_nc")
    return re.sub(r"[^a-z0-9]+", "", text)


def _screen_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _screened_slugs(rows: list[dict[str, str]]) -> set[str]:
    out: set[str] = set()
    for row in rows:
        for key in ("case", "vmec_file", "source", "geometry"):
            raw = row.get(key, "")
            if raw:
                out.add(_slug(Path(raw).name))
                out.add(_slug(raw))
    return out


def _screen_expansion_candidates(
    *,
    inventory: dict[str, Any],
    screen_rows: list[dict[str, str]],
    max_candidates: int,
) -> list[dict[str, Any]]:
    screened = _screened_slugs(screen_rows)
    rows = [row for row in inventory.get("rows", []) if isinstance(row, dict)]
    candidates: list[dict[str, Any]] = []
    for row in rows:
        name = str(row.get("name", ""))
        if not name or _slug(name) in screened:
            continue
        if not bool(row.get("reference_scale_valid", False)):
            continue
        family = str(row.get("family", ""))
        if family in {"axisymmetric"} and any("tokamak" in str(row.get("name", "")).lower() for _ in (0,)):
            priority_note = "axisymmetric reserve candidate; use only if stellarator candidates fail screen"
        else:
            priority_note = "independent VMEC candidate requiring linear screen before nonlinear launch"
        candidates.append(
            {
                "name": name,
                "path": row.get("path", name),
                "family": family,
                "aspect": row.get("aspect"),
                "iota_edge": row.get("iota_edge"),
                "candidate_score": row.get("candidate_score", 0.0),
                "priority_note": priority_note,
                "next_required_gate": "linear_ky_screen_with_gamma_ge_0p02_before_nonlinear_holdout_launch",
            }
        )
    candidates.sort(key=lambda item: (-float(item.get("candidate_score") or 0.0), str(item["name"])))
    return candidates[:max_candidates]


def _optimizer_audit_commands(
    *,
    office_root: Path,
    audit_root: Path,
    names: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, name in enumerate(names):
        wout = (
            office_root
            / "tools_out"
            / "vmec_jax_qa_optimizer_ladder_20260609"
            / "runs"
            / name
            / "wout_final_rerun.nc"
        )
        out_dir = audit_root / name
        case = f"premanuscript_{name}"
        generate = (
            "python3 tools/write_optimized_equilibrium_transport_configs.py "
            f"--vmec-file {wout.relative_to(office_root).as_posix()} "
            f"--case {case} "
            f"--out-dir {out_dir.as_posix()} "
            "--horizons 700,1100,1500 --grid n64:64:64:40:40 "
            "--torflux 0.64 --alpha 0.0 --npol 1.0 "
            "--window-tmin 1100 --window-tmax 1500 "
            "--dt-variant 0.04 --seed-variant 32 --seed-variant 33"
        )
        seed32 = (
            "CUDA_VISIBLE_DEVICES=0 python3 -m spectraxgk.cli run-runtime-nonlinear "
            f"--config {out_dir / f'{case}_nonlinear_t1500_n64_seed32.toml'} "
            "--steps 30000 --no-progress"
        )
        seed33 = (
            "CUDA_VISIBLE_DEVICES=1 python3 -m spectraxgk.cli run-runtime-nonlinear "
            f"--config {out_dir / f'{case}_nonlinear_t1500_n64_seed33.toml'} "
            "--steps 30000 --no-progress"
        )
        dt_variant = (
            "CUDA_VISIBLE_DEVICES=${DEVICE:-0} python3 -m spectraxgk.cli run-runtime-nonlinear "
            f"--config {out_dir / f'{case}_nonlinear_t1500_n64_dt0p04.toml'} "
            "--steps 37500 --no-progress"
        )
        rows.append(
            {
                "rank": idx + 1,
                "optimizer_case": name,
                "office_root": office_root.as_posix(),
                "wout": wout.relative_to(office_root).as_posix(),
                "manifest": (out_dir / "run_manifest.json").as_posix(),
                "generate_configs_command": generate,
                "seed_launch_commands": [seed32, seed33],
                "dt_variant_followup_command": dt_variant,
                "window": [1100.0, 1500.0],
                "claim_level": "launch_contract_or_running_audit_not_transport_promotion",
            }
        )
    return rows


def _vmec_boozer_holdout_transport_commands(
    *,
    office_root: Path,
    out_root: Path,
) -> list[dict[str, Any]]:
    """Return production-scope held-out VMEC/Boozer nonlinear launch contracts."""

    wout = Path("/home/rjorge/src/vmec_jax/examples/data/wout_nfp4_QH_warm_start.nc")
    case = "vmec_boozer_qh_torflux078_alpha120_holdout"
    out_dir = out_root / case
    generate = (
        "python3 tools/write_optimized_equilibrium_transport_configs.py "
        f"--vmec-file {wout.as_posix()} "
        f"--case {case} "
        f"--out-dir {out_dir.as_posix()} "
        "--horizons 250,350,450,700 --grid n64:64:64:40:40 "
        "--torflux 0.78 --alpha 1.2 --npol 1.0 --ky 0.2 "
        "--window-tmin 350 --window-tmax 700 "
        "--dt-variant 0.04 --seed-variant 31 --seed-variant 32"
    )
    outputs = [
        out_dir / f"{case}_nonlinear_t700_n64_seed31.out.nc",
        out_dir / f"{case}_nonlinear_t700_n64_seed32.out.nc",
        out_dir / f"{case}_nonlinear_t700_n64_dt0p04.out.nc",
    ]
    artifact_dir = office_root / "docs" / "_static" / "vmec_boozer_holdout_transport"
    ensemble_json = artifact_dir / f"{case}_ensemble_gate.json"
    readiness_json = artifact_dir / f"{case}_readiness.json"
    ensemble_png = artifact_dir / f"{case}_ensemble_gate.png"
    holdout_json = artifact_dir / f"{case}_production_holdout.json"
    output_gate_json = artifact_dir / f"{case}_output_gate.json"
    output_gate_command = (
        "python3 tools/check_nonlinear_runtime_outputs.py "
        + " ".join(path.as_posix() for path in outputs)
        + " --min-samples 200 --tmin 350 --tmax 700 --min-window-samples 80 "
        f"--min-abs-window-mean 0.0001 --json-out {output_gate_json.as_posix()}"
    )
    build_ensemble_command = (
        "python3 tools/build_external_vmec_replicate_ensemble.py "
        + " ".join(path.as_posix() for path in outputs)
        + f" --out-dir {artifact_dir.as_posix()}"
        + f" --case {case}_replicated_nonlinear_window"
        + " --tmin 350 --tmax 700"
        + " --artifact-prefix docs/_static/vmec_boozer_holdout_transport"
        + f" --readiness-json {readiness_json.name}"
        + f" --ensemble-json {ensemble_json.name}"
        + f" --out-png {ensemble_png.name}"
    )
    build_holdout_artifact_command = (
        "python3 tools/build_vmec_boozer_production_holdout_artifact.py "
        f"--transport-manifest {(out_dir / 'run_manifest.json').as_posix()} "
        f"--ensemble-json {ensemble_json.as_posix()} "
        f"--case {case} --out {holdout_json.as_posix()}"
    )
    promotion_gate_command = (
        "python3 tools/check_vmec_boozer_aggregate_holdout_gate.py "
        "--holdout-artifact docs/_static/vmec_boozer_aggregate_alpha_holdout_gate.json "
        "--holdout-artifact docs/_static/vmec_boozer_aggregate_surface_holdout_gate.json "
        f"--holdout-artifact {holdout_json.as_posix()} "
        "--nonlinear-ensemble-artifact docs/_static/external_vmec_dshape_replicates/dshape_replicate_t250_ensemble_gate.json "
        "--nonlinear-ensemble-artifact docs/_static/external_vmec_circular_replicates/circular_replicate_t700_ensemble_gate.json "
        f"--nonlinear-ensemble-artifact {ensemble_json.as_posix()} "
        "--json-out docs/_static/vmec_boozer_aggregate_holdout_promotion_gate.json"
    )
    return [
        {
            "case": case,
            "wout": wout.as_posix(),
            "transport_sample": {
                "torflux": 0.78,
                "alpha": 1.2,
                "ky": 0.2,
                "npol": 1.0,
                "role": "heldout_surface_field_line_transport",
            },
            "manifest": (out_dir / "run_manifest.json").as_posix(),
            "generate_configs_command": generate,
            "direct_full_horizon_launch_commands": [
                (
                    f"CUDA_VISIBLE_DEVICES={device} python3 -m spectraxgk.cli run-runtime-nonlinear "
                    f"--config {out_dir / f'{case}_nonlinear_t700_n64_{variant}.toml'} "
                    f"--steps {steps} --no-progress"
                )
                for device, variant, steps in (
                    (0, "seed31", 14000),
                    (1, "seed32", 14000),
                    ("${DEVICE:-0}", "dt0p04", 17500),
                )
            ],
            "expected_outputs": [path.as_posix() for path in outputs],
            "output_gate_command": output_gate_command,
            "build_ensemble_command": build_ensemble_command,
            "build_holdout_artifact_command": build_holdout_artifact_command,
            "promotion_gate_command": promotion_gate_command,
            "postprocess_artifacts": {
                "output_gate": output_gate_json.as_posix(),
                "ensemble_json": ensemble_json.as_posix(),
                "readiness_json": readiness_json.as_posix(),
                "ensemble_png": ensemble_png.as_posix(),
                "production_holdout_json": holdout_json.as_posix(),
            },
            "window": [350.0, 700.0],
            "claim_level": (
                "production_scope_vmec_boozer_surface_field_line_launch_contract_not_transport_promotion"
            ),
        }
    ]


def build_runbook_payload(
    *,
    root: Path = ROOT,
    inventory_path: Path = DEFAULT_INVENTORY,
    screen_path: Path = DEFAULT_SCREEN,
    external_runbook_path: Path = DEFAULT_EXTERNAL_RUNBOOK,
    optimizer_manifest_path: Path = DEFAULT_OPTIMIZER_MANIFEST,
    ladder_status_path: Path = DEFAULT_LADDER_STATUS,
    office_root: Path = DEFAULT_OFFICE_ROOT,
    audit_root: Path = DEFAULT_AUDIT_ROOT,
    max_screen_candidates: int = 8,
) -> dict[str, Any]:
    """Return a JSON-ready pre-manuscript action runbook."""

    status = build_status_payload(root)
    inventory = _read_json(inventory_path)
    screen_rows = _screen_rows(screen_path)
    external_runbook = _read_json(external_runbook_path)
    optimizer_manifest = _read_json(optimizer_manifest_path)
    ladder_status = _read_json(ladder_status_path)
    screen_candidates = _screen_expansion_candidates(
        inventory=inventory,
        screen_rows=screen_rows,
        max_candidates=max_screen_candidates,
    )
    optimizer_names = (
        "growth_scalar_trust_from_strict_baseline",
        "growth_lbfgs_adjoint_from_strict_baseline",
        "quasilinear_scalar_trust_from_strict_baseline",
    )
    optimizer_audits = _optimizer_audit_commands(
        office_root=office_root,
        audit_root=audit_root,
        names=optimizer_names,
    )
    heldout_transport = _vmec_boozer_holdout_transport_commands(
        office_root=office_root,
        out_root=office_root / "tools_out" / "vmec_boozer_holdout_transport",
    )
    external_has_launch = bool(external_runbook.get("passed", False)) and bool(
        external_runbook.get("launch_commands")
    )
    external_next_action = (
        "Launch or harvest the selected nonlinear holdout campaign, then admit it only through "
        "grid/window convergence, replicated post-transient transport, and QL recalibration gates."
        if external_has_launch
        else (
            "Run a linear ky screen on the listed unscreened VMEC candidates; only candidates "
            "with gamma >= 0.02 and valid flux-tube metrics may enter the nonlinear holdout runbook."
        )
    )
    selected_external = (
        external_runbook.get("selected_new_family_candidate")
        or external_runbook.get("selected_preferred_family_audit")
        or {}
    )
    lanes = {str(lane["lane"]): lane for lane in status.get("lanes", []) if isinstance(lane, dict)}
    domain_lane = lanes.get("Production nonlinear domain-decomposition speedup", {})
    payload = {
        "kind": "pre_manuscript_closure_runbook",
        "claim_scope": (
            "actionable campaign runbook only; generated or launched commands do not promote "
            "absolute quasilinear, broad nonlinear optimization, VMEC/Boozer optimization, or "
            "production nonlinear parallel speedup claims without the strict gates"
        ),
        "status_summary": status.get("summary", {}),
        "external_vmec_holdout_campaign": {
            "status": "blocked_on_new_linear_screen" if not external_has_launch else "launchable",
            "external_runbook": external_runbook_path.relative_to(root).as_posix(),
            "external_runbook_passed": bool(external_runbook.get("passed", False)),
            "min_launch_gamma": float(external_runbook.get("min_launch_gamma", MIN_LINEAR_LAUNCH_GAMMA)),
            "screen_rows": len(screen_rows),
            "inventory_equilibria": int(inventory.get("n_equilibria", 0) or 0),
            "selected_candidate": selected_external,
            "launch_commands": external_runbook.get("launch_commands", []),
            "unscreened_candidates": screen_candidates,
            "next_action": external_next_action,
        },
        "vmec_boozer_production_scope_artifacts": {
            "status": "launch_contracts_generated_on_office",
            "purpose": (
                "three matched long-window optimized-equilibrium nonlinear audits for broad "
                "VMEC/Boozer and nonlinear turbulent-flux optimization evidence, plus a held-out "
                "VMEC/Boozer surface/field-line nonlinear transport audit"
            ),
            "optimizer_manifest": optimizer_manifest_path.relative_to(root).as_posix(),
            "optimizer_manifest_entries": len(optimizer_manifest.get("entries", []) or []),
            "ladder_status": ladder_status_path.relative_to(root).as_posix(),
            "ladder_commands": len(ladder_status.get("commands", []) or []),
            "audit_commands": optimizer_audits,
            "heldout_transport_commands": heldout_transport,
            "office_seed_queue": {
                "launched": True,
                "pids": [3402448, 3402449],
                "queues": [
                    "GPU0 seed32: growth_scalar_trust -> growth_lbfgs_adjoint -> quasilinear_scalar_trust",
                    "GPU1 seed33: growth_scalar_trust -> growth_lbfgs_adjoint -> quasilinear_scalar_trust",
                ],
                "dt_variant_policy": "launch dt=0.04 variants after seed outputs are finite and logs show no NaNs",
            },
        },
        "nonlinear_optimization_audit_extension": {
            "status": "running_or_launchable",
            "required_count": 3,
            "candidate_count": len(optimizer_audits),
            "window": [1100.0, 1500.0],
            "acceptance": (
                "each candidate must pass finite-output, long-window sample count, running-mean/block-SEM, "
                "seed/timestep ensemble, and matched baseline-vs-optimized transport gates"
            ),
        },
        "nonlinear_domain_decomposition": {
            "status": "identity_route_extended_no_speedup_claim",
            "completion_percent": domain_lane.get("completion_percent"),
            "artifact": "docs/_static/nonlinear_spectral_communication_identity_gate.json",
            "new_gate": "logical_sharded_nonlinear_spectral_integrator_identity",
            "next_action": (
                "Replace the local logical tile route with real device communication/distributed FFT routing, "
                "then require serial-vs-decomposed transport-window identity and CPU/GPU speedup >= 1.5."
            ),
        },
    }
    payload["overall_next_actions"] = [
        "Harvest office t=1500 seed audit logs and outputs; launch dt=0.04 variants only if seed outputs are finite.",
        external_next_action,
        "Regenerate QL calibration only after a new independent nonlinear holdout passes convergence and replicate gates.",
        "Keep nonlinear decomposition as identity-only until production distributed routing and profiler speedup gates pass.",
    ]
    return _json_clean(payload)


def write_runbook_artifacts(payload: dict[str, Any], *, out: Path = DEFAULT_OUT) -> dict[str, str]:
    """Write JSON/CSV/PNG/PDF companions for the action runbook."""

    out.parent.mkdir(parents=True, exist_ok=True)
    json_path = out.with_suffix(".json")
    csv_path = out.with_suffix(".csv")
    pdf_path = out.with_suffix(".pdf")
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    rows = [
        {
            "lane": "external_vmec_holdout_campaign",
            "status": payload["external_vmec_holdout_campaign"]["status"],
            "next_action": payload["external_vmec_holdout_campaign"]["next_action"],
        },
        {
            "lane": "vmec_boozer_production_scope_artifacts",
            "status": payload["vmec_boozer_production_scope_artifacts"]["status"],
            "next_action": "harvest launched office t=1500 audits and build ensemble gates",
        },
        {
            "lane": "nonlinear_optimization_audit_extension",
            "status": payload["nonlinear_optimization_audit_extension"]["status"],
            "next_action": payload["nonlinear_optimization_audit_extension"]["acceptance"],
        },
        {
            "lane": "nonlinear_domain_decomposition",
            "status": payload["nonlinear_domain_decomposition"]["status"],
            "next_action": payload["nonlinear_domain_decomposition"]["next_action"],
        },
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("lane", "status", "next_action"), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    set_plot_style()
    labels = [row["lane"].replace("_", "\n") for row in rows]
    status_to_score = {
        "launchable": 0.75,
        "running_or_launchable": 0.72,
        "launch_contracts_generated_on_office": 0.68,
        "identity_route_extended_no_speedup_claim": 0.62,
        "blocked_on_new_linear_screen": 0.35,
    }
    scores = [status_to_score.get(row["status"], 0.5) * 100.0 for row in rows]
    colors = ["#d89c32" if "blocked" in row["status"] else "#2f7f5f" for row in rows]
    fig, ax = plt.subplots(figsize=(12.2, 5.4))
    x = np.arange(len(rows))
    ax.bar(x, scores, color=colors, edgecolor="#333333")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, 100.0)
    ax.set_ylabel("actionability score (%)")
    ax.set_title("Pre-manuscript action runbook")
    ax.grid(axis="y", alpha=0.25)
    for xi, score, row in zip(x, scores, rows, strict=True):
        ax.text(xi, score + 2.0, row["status"], ha="center", va="bottom", fontsize=8, rotation=12)
    fig.text(
        0.5,
        0.035,
        "Actionability is not claim closure: strict promotion gates remain authoritative.",
        ha="center",
        fontsize=8.5,
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.34)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(out), "pdf": str(pdf_path), "json": str(json_path), "csv": str(csv_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--screen", type=Path, default=DEFAULT_SCREEN)
    parser.add_argument("--external-runbook", type=Path, default=DEFAULT_EXTERNAL_RUNBOOK)
    parser.add_argument("--optimizer-manifest", type=Path, default=DEFAULT_OPTIMIZER_MANIFEST)
    parser.add_argument("--ladder-status", type=Path, default=DEFAULT_LADDER_STATUS)
    parser.add_argument("--office-root", type=Path, default=DEFAULT_OFFICE_ROOT)
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    parser.add_argument("--max-screen-candidates", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_runbook_payload(
        inventory_path=args.inventory,
        screen_path=args.screen,
        external_runbook_path=args.external_runbook,
        optimizer_manifest_path=args.optimizer_manifest,
        ladder_status_path=args.ladder_status,
        office_root=args.office_root,
        audit_root=args.audit_root,
        max_screen_candidates=int(args.max_screen_candidates),
    )
    paths = write_runbook_artifacts(payload, out=args.out)
    print(json.dumps(paths, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
