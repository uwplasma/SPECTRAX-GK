#!/usr/bin/env python3
"""Build a VMEC-JAX QA-only vs QA+transport candidate comparison panel."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spectraxgk.artifacts.plotting import set_plot_style  # type: ignore[import-untyped] # noqa: E402
from spectraxgk.objectives.vmec_candidate_admission import (  # type: ignore[import-untyped] # noqa: E402
    build_solved_vmec_candidate_gate,
)


DEFAULT_OUT = (
    ROOT / "docs" / "_static" / "vmec_jax_qa_transport_candidate_comparison.png"
)
DEFAULT_PAYLOAD_JSON = DEFAULT_OUT.with_suffix(".json")
DEFAULT_CONSTRAINTS_DIR = (
    ROOT / "tools_out" / "vmec_jax_qa_transport_authoritative_sidecar" / "constraints"
)
DEFAULT_TRANSPORT_DIR = (
    ROOT / "tools_out" / "vmec_jax_qa_transport_authoritative_sidecar" / "transport"
)
COLORS = {
    "QA constraints": "#244c66",
    "QA + SPECTRAX-GK transport": "#b45f2a",
}


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve(strict=False).relative_to(ROOT.resolve(strict=False)))
    except ValueError:
        return str(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--constraints-dir",
        type=Path,
        default=DEFAULT_CONSTRAINTS_DIR,
        help="VMEC-JAX output directory for the QA-only branch",
    )
    parser.add_argument(
        "--transport-dir",
        type=Path,
        default=DEFAULT_TRANSPORT_DIR,
        help="VMEC-JAX output directory for the QA+SPECTRAX-GK transport branch",
    )
    parser.add_argument(
        "--payload-json",
        type=Path,
        default=DEFAULT_PAYLOAD_JSON,
        help="Tracked payload used for replotting when candidate directories are absent",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="PNG output path")
    parser.add_argument("--pdf", action="store_true", help="also write a PDF companion")
    parser.add_argument("--target-aspect", type=float, default=6.0)
    parser.add_argument("--aspect-atol", type=float, default=5.0e-2)
    parser.add_argument("--min-iota", type=float, default=0.41)
    parser.add_argument("--qs-max", type=float, default=5.0e-2)
    return parser.parse_args()


def _load_history(root: Path) -> dict[str, Any]:
    path = root / "history.json"
    if not path.exists():
        raise FileNotFoundError(f"missing VMEC-JAX history file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_solved_gate(root: Path) -> dict[str, Any] | None:
    path = root / "solved_wout_gate.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _load_wout_reproducibility_gate(root: Path) -> dict[str, Any] | None:
    path = root / "wout_reproducibility_gate.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _load_rerun_wout_admission_gate(root: Path) -> dict[str, Any] | None:
    path = root / "rerun_wout_admission_gate.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _load_iota_profiles(
    root: Path, *, wout_name: str = "wout_final.nc"
) -> tuple[np.ndarray, np.ndarray]:
    import vmec_jax as vj  # type: ignore[import-not-found]

    path = root / wout_name
    if not path.exists():
        raise FileNotFoundError(f"missing final WOUT file: {path}")
    wout = vj.load_wout(path)
    return np.asarray(wout.iotas, dtype=float), np.asarray(wout.iotaf, dtype=float)


def _objective_series(history: dict[str, Any]) -> list[float]:
    entries = history.get("history", ())
    if not isinstance(entries, list):
        return []
    values = []
    for entry in entries:
        if isinstance(entry, dict) and entry.get("objective") is not None:
            try:
                values.append(float(entry["objective"]))
            except Exception:
                pass
    return values


def _branch_summary(
    label: str,
    root: Path,
    *,
    target_aspect: float,
    aspect_atol: float,
    min_iota: float,
    qs_max: float,
) -> dict[str, Any]:
    history = _load_history(root)
    gate_source = "solved_wout_gate.json"
    gate = _load_solved_gate(root)
    wout_repro_gate = _load_wout_reproducibility_gate(root)
    wout_repro_gate_passed = (
        None if wout_repro_gate is None else bool(wout_repro_gate.get("passed", False))
    )
    rerun_wout_admission_gate = _load_rerun_wout_admission_gate(root)
    rerun_wout_admission_gate_passed = (
        None
        if rerun_wout_admission_gate is None
        else bool(rerun_wout_admission_gate.get("passed", False))
    )
    uses_authoritative_rerun_wout = (
        wout_repro_gate_passed is False
        and rerun_wout_admission_gate_passed is True
        and (root / "wout_final_rerun.nc").exists()
    )
    authoritative_wout_name = (
        "wout_final_rerun.nc" if uses_authoritative_rerun_wout else "wout_final.nc"
    )
    iotas, iotaf = _load_iota_profiles(root, wout_name=authoritative_wout_name)
    s = np.linspace(0.0, 1.0, iotas.size)
    if gate is None:
        gate_source = "reconstructed_history_wout"
        gate = build_solved_vmec_candidate_gate(
            history,
            target_aspect=target_aspect,
            aspect_atol=aspect_atol,
            min_abs_mean_iota=min_iota,
            qs_residual_max=qs_max,
            iota_profile_floor=min_iota,
            iota_profiles=(iotas, iotaf),
            profile_source=authoritative_wout_name,
        )
    gate_is_authoritative = gate_source == "solved_wout_gate.json"
    gate_reported_passed = bool(gate.get("passed", False))
    admission_blockers = [
        name
        for name, check in gate.get("checks", {}).items()
        if isinstance(check, dict) and not bool(check.get("passed", False))
    ]
    if not gate_is_authoritative:
        admission_blockers.insert(0, "non_authoritative_reconstructed_gate")
    if wout_repro_gate_passed is False and not uses_authoritative_rerun_wout:
        admission_blockers.append("wout_reproducibility_gate_failed")
    admitted = bool(
        gate_reported_passed
        and gate_is_authoritative
        and (
            wout_repro_gate_passed is None
            or bool(wout_repro_gate_passed)
            or bool(uses_authoritative_rerun_wout)
        )
    )
    return {
        "label": label,
        "root": _repo_relative(root),
        "history": {
            key: history.get(key)
            for key in (
                "aspect_initial",
                "aspect_final",
                "iota_initial",
                "iota_final",
                "qs_initial",
                "qs_final",
                "objective_initial",
                "objective_final",
                "nfev",
                "success",
                "message",
                "total_wall_time_s",
            )
        },
        "objective_history": _objective_series(history),
        "iota_profile": {
            "s": s.tolist(),
            "iota": iotas.tolist(),
            "iotaf": iotaf.tolist(),
            "mean_including_axis": float(np.nanmean(iotas)),
            "mean_excluding_axis": float(np.nanmean(iotas[1:]))
            if iotas.size > 1
            else float(np.nanmean(iotas)),
            "min_excluding_axis": float(np.nanmin(iotas[1:]))
            if iotas.size > 1
            else float(np.nanmin(iotas)),
            "min_iotaf": float(np.nanmin(iotaf)),
            "edge": float(iotas[-1]),
        },
        "gate": gate,
        "gate_source": gate_source,
        "gate_is_authoritative": gate_is_authoritative,
        "gate_reported_passed": gate_reported_passed,
        "wout_reproducibility_gate": wout_repro_gate,
        "wout_reproducibility_gate_passed": wout_repro_gate_passed,
        "rerun_wout_admission_gate": rerun_wout_admission_gate,
        "rerun_wout_admission_gate_passed": rerun_wout_admission_gate_passed,
        "uses_authoritative_rerun_wout": uses_authoritative_rerun_wout,
        "authoritative_wout": _repo_relative(root / authoritative_wout_name),
        "admitted_for_long_window_nonlinear_audit": admitted,
        "admission_blockers": admission_blockers,
    }


def build_payload(
    constraints_dir: Path,
    transport_dir: Path,
    *,
    target_aspect: float = 6.0,
    aspect_atol: float = 5.0e-2,
    min_iota: float = 0.41,
    qs_max: float = 5.0e-2,
) -> dict[str, Any]:
    """Return JSON-ready comparison data from two VMEC-JAX output directories."""

    branches = [
        _branch_summary(
            "QA constraints",
            constraints_dir,
            target_aspect=target_aspect,
            aspect_atol=aspect_atol,
            min_iota=min_iota,
            qs_max=qs_max,
        ),
        _branch_summary(
            "QA + SPECTRAX-GK transport",
            transport_dir,
            target_aspect=target_aspect,
            aspect_atol=aspect_atol,
            min_iota=min_iota,
            qs_max=qs_max,
        ),
    ]
    ready = [
        branch["label"]
        for branch in branches
        if bool(branch["admitted_for_long_window_nonlinear_audit"])
    ]
    transport_branch = next(
        (
            branch
            for branch in branches
            if branch["label"] == "QA + SPECTRAX-GK transport"
        ),
        branches[-1],
    )
    transport_admitted = bool(
        transport_branch["admitted_for_long_window_nonlinear_audit"]
    )
    return {
        "kind": "vmec_jax_qa_transport_candidate_comparison",
        "claim_scope": (
            "bounded VMEC-JAX solved-boundary candidate comparison; validates objective assembly, "
            "trace-safe SPECTRAX-GK transport residual, WOUT writing, and authoritative solved-equilibrium "
            "admission gates; not a converged optimized-equilibrium nonlinear turbulent-flux claim"
        ),
        "target_aspect": float(target_aspect),
        "aspect_atol": float(aspect_atol),
        "iota_gate_policy": "lower_bound_admission_not_exact_upstream_mean_iota_target",
        "mean_iota_lower_bound": float(min_iota),
        "iota_profile_floor": float(min_iota),
        "target_mean_iota": float(min_iota),
        "target_iota_profile_floor": float(min_iota),
        "legacy_target_iota_fields_are_lower_bounds": True,
        "qs_residual_max": float(qs_max),
        "branches": branches,
        "summary": {
            "all_branches_passed_solved_wout_gate": all(
                bool(branch["admitted_for_long_window_nonlinear_audit"])
                for branch in branches
            ),
            "all_branches_have_authoritative_gate": all(
                bool(branch["gate_is_authoritative"]) for branch in branches
            ),
            "ready_for_long_window_nonlinear_audit": ready,
            "blocked_branches": [
                branch["label"]
                for branch in branches
                if not bool(branch["admitted_for_long_window_nonlinear_audit"])
            ],
            "transport_candidate_admitted": transport_admitted,
            "transport_optimization_status": (
                "candidate_admitted_pending_long_window_nonlinear_audit"
                if transport_admitted
                else "blocked_before_transport_claim"
            ),
            "next_step": (
                "Use only branches with authoritative passing solved-WOUT gates for matched long-window "
                "SPECTRAX-GK nonlinear audits. If the transport branch is blocked, switch to a "
                "constraint-preserving/projection/admission method instead of promoting scalar-weight sweeps."
            ),
        },
    }


def _gate_metric(branch: dict[str, Any], name: str) -> float:
    check = branch["gate"]["checks"][name]
    if name == "aspect":
        return float(check["absolute_tolerance"]) - float(check["absolute_error"])
    if name == "iota_profile":
        return float(check["minimum_iotas_excluding_axis"]) - float(check["floor"])
    margin = check.get("margin")
    return float(margin) if margin is not None else float("nan")


def plot_payload(payload: dict[str, Any], out: Path) -> None:
    """Render a publication-style candidate comparison panel."""

    set_plot_style()
    labels = [branch["label"] for branch in payload["branches"]]
    colors = [COLORS.get(label, "#586069") for label in labels]
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.8), constrained_layout=True)

    ax = axes[0, 0]
    for branch, color in zip(payload["branches"], colors, strict=True):
        profile = branch["iota_profile"]
        ax.plot(
            profile["s"], profile["iota"], lw=2.4, color=color, label=branch["label"]
        )
    iota_profile_floor = float(
        payload.get("iota_profile_floor", payload["target_iota_profile_floor"])
    )
    ax.axhline(iota_profile_floor, color="black", lw=1.2, ls=":")
    ax.set_xlabel("normalized toroidal flux")
    ax.set_ylabel(r"$\iota$")
    ax.set_title("Solved WOUT iota profiles")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    x = np.arange(len(labels), dtype=float)
    width = 0.18
    metric_specs = (
        (
            r"$|A-A_0|/\Delta A$",
            lambda branch: branch["gate"]["checks"]["aspect"]["absolute_error"]
            / branch["gate"]["checks"]["aspect"]["absolute_tolerance"],
        ),
        (
            r"$|\bar\iota|/\iota_{min}$",
            lambda branch: branch["gate"]["checks"]["mean_iota"]["value"]
            / branch["gate"]["checks"]["mean_iota"]["minimum_abs"],
        ),
        (
            r"$R_{QA}/R_{max}$",
            lambda branch: branch["gate"]["checks"]["quasisymmetry"]["value"]
            / branch["gate"]["checks"]["quasisymmetry"]["maximum"],
        ),
        (
            r"$\min\iota(s)/\iota_{min}$",
            lambda branch: branch["gate"]["checks"]["iota_profile"][
                "minimum_iotas_excluding_axis"
            ]
            / branch["gate"]["checks"]["iota_profile"]["floor"],
        ),
    )
    offsets = (-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width)
    ratio_floor = 1.0e-3
    for offset, (label, getter) in zip(offsets, metric_specs, strict=True):
        values = [
            max(float(getter(branch)), ratio_floor) for branch in payload["branches"]
        ]
        ax.bar(x + offset, values, width=width, label=label)
    ax.axhline(1.0, color="black", lw=1.0, ls=":", alpha=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(["QA", "QA+Q"], rotation=0)
    ax.set_yscale("log")
    ax.set_ylim(ratio_floor, None)
    ax.set_ylabel("ratio to gate threshold (log)")
    ax.set_title("Normalized gate quantities")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1, 0]
    gate_names = ("aspect", "mean_iota", "iota_profile", "quasisymmetry")
    gate_labels = ("aspect", r"$|\bar\iota|$", r"$\min\iota(s)$", "QS")
    xg = np.arange(len(gate_names), dtype=float)
    width = 0.36
    for offset, branch, color in zip(
        (-0.5 * width, 0.5 * width), payload["branches"], colors, strict=True
    ):
        margins = [_gate_metric(branch, name) for name in gate_names]
        ax.bar(xg + offset, margins, width=width, color=color, label=branch["label"])
    ax.axhline(0.0, color="black", lw=1.0)
    ax.set_xticks(xg)
    ax.set_xticklabels(gate_labels)
    ax.set_ylabel("positive margin passes")
    ax.set_title("Solved-candidate gate margins")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    for branch, color in zip(payload["branches"], colors, strict=True):
        history = branch.get("objective_history", [])
        if history:
            y = np.asarray(history, dtype=float)
            ax.semilogy(
                np.arange(y.size), y, lw=2.0, color=color, label=branch["label"]
            )
        else:
            hist = branch["history"]
            ax.semilogy(
                [0, 1],
                [float(hist["objective_initial"]), float(hist["objective_final"])],
                marker="o",
                lw=2.0,
                color=color,
                label=branch["label"],
            )
    ax.set_xlabel("optimizer callback")
    ax.set_ylabel(r"$||r||^2$")
    ax.set_title("Objective history")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle(
        "VMEC-JAX QA candidate gate: constraints-only vs SPECTRAX-GK transport residual",
        fontsize=14,
        fontweight="bold",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_csv(payload: dict[str, Any], path: Path) -> None:
    rows = []
    for branch in payload["branches"]:
        hist = branch["history"]
        gate = branch["gate"]
        profile = branch["iota_profile"]
        rows.append(
            {
                "label": branch["label"],
                "passed": gate["passed"],
                "admitted_for_long_window_nonlinear_audit": branch.get(
                    "admitted_for_long_window_nonlinear_audit",
                    False,
                ),
                "gate_source": branch.get("gate_source"),
                "gate_is_authoritative": branch.get("gate_is_authoritative"),
                "admission_blockers": ";".join(
                    str(item) for item in branch.get("admission_blockers", ())
                ),
                "aspect_final": hist["aspect_final"],
                "iota_final": hist["iota_final"],
                "qs_final": hist["qs_final"],
                "min_iota_profile": profile["min_excluding_axis"],
                "min_iotaf": profile["min_iotaf"],
                "objective_initial": hist["objective_initial"],
                "objective_final": hist["objective_final"],
                "next_action": gate["next_action"],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _load_or_build_payload(args: argparse.Namespace) -> dict[str, Any]:
    if (args.constraints_dir / "history.json").exists() and (
        args.transport_dir / "history.json"
    ).exists():
        return build_payload(
            args.constraints_dir,
            args.transport_dir,
            target_aspect=float(args.target_aspect),
            aspect_atol=float(args.aspect_atol),
            min_iota=float(args.min_iota),
            qs_max=float(args.qs_max),
        )
    if args.payload_json.exists():
        return json.loads(args.payload_json.read_text(encoding="utf-8"))
    raise FileNotFoundError(
        "candidate directories are missing and no tracked payload JSON is available; "
        "pass --constraints-dir/--transport-dir or --payload-json"
    )


def main() -> int:
    args = _parse_args()
    payload = _load_or_build_payload(args)
    out = args.out
    base = out.with_suffix("")
    base.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    _write_csv(payload, base.with_suffix(".csv"))
    plot_payload(payload, out)
    if args.pdf:
        plot_payload(payload, base.with_suffix(".pdf"))
    print(
        json.dumps(
            {
                "passed": payload["summary"].get(
                    "all_branches_passed_solved_wout_gate", False
                ),
                "ready_for_long_window_nonlinear_audit": payload["summary"].get(
                    "ready_for_long_window_nonlinear_audit",
                    [],
                ),
                "paths": {
                    "png": str(out),
                    "json": str(base.with_suffix(".json")),
                    "csv": str(base.with_suffix(".csv")),
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
