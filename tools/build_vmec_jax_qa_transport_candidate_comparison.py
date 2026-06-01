#!/usr/bin/env python3
"""Build a VMEC-JAX QA-only vs QA+transport candidate comparison panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from spectraxgk.plotting import set_plot_style  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "docs" / "_static" / "vmec_jax_qa_transport_candidate_comparison.png"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--constraints-dir",
        type=Path,
        default=Path("/tmp/spectraxgk_qa_pair_constraints_scalar"),
        help="VMEC-JAX output directory for the QA-only branch",
    )
    parser.add_argument(
        "--transport-dir",
        type=Path,
        default=Path("/tmp/spectraxgk_qa_pair_transport_scalar"),
        help="VMEC-JAX output directory for the QA+SPECTRAX-GK transport branch",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="PNG output path")
    parser.add_argument("--pdf", action="store_true", help="also write a PDF companion")
    return parser.parse_args()


def _load_history(root: Path) -> dict[str, Any]:
    path = root / "history.json"
    if not path.exists():
        raise FileNotFoundError(f"missing VMEC-JAX history file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_iota_profile(root: Path) -> tuple[np.ndarray, np.ndarray]:
    import vmec_jax as vj

    path = root / "wout_final.nc"
    if not path.exists():
        raise FileNotFoundError(f"missing final WOUT file: {path}")
    wout = vj.load_wout(path)
    iota = np.asarray(wout.iotas, dtype=float)
    s = np.linspace(0.0, 1.0, iota.size)
    return s, iota


def _branch_summary(label: str, root: Path) -> dict[str, Any]:
    history = _load_history(root)
    s, iota = _load_iota_profile(root)
    return {
        "label": label,
        "root": str(root),
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
        "iota_profile": {
            "s": s.tolist(),
            "iota": iota.tolist(),
            "mean_including_axis": float(np.nanmean(iota)),
            "mean_excluding_axis": float(np.nanmean(iota[1:])) if iota.size > 1 else float(np.nanmean(iota)),
            "min_excluding_axis": float(np.nanmin(iota[1:])) if iota.size > 1 else float(np.nanmin(iota)),
            "edge": float(iota[-1]),
        },
    }


def build_payload(constraints_dir: Path, transport_dir: Path) -> dict[str, Any]:
    """Return JSON-ready comparison data from two VMEC-JAX output directories."""

    branches = [
        _branch_summary("QA constraints", constraints_dir),
        _branch_summary("QA + SPECTRAX-GK transport", transport_dir),
    ]
    target_iota = 0.41
    target_aspect = 6.0
    for branch in branches:
        hist = branch["history"]
        branch["gates"] = {
            "aspect_near_target": abs(float(hist["aspect_final"]) - target_aspect) / target_aspect < 5.0e-3,
            "mean_iota_above_target": float(hist["iota_final"]) >= target_iota,
            "wout_iota_above_target": float(branch["iota_profile"]["min_excluding_axis"]) >= target_iota,
            "optimized": bool(hist.get("success")) and float(hist["objective_final"]) < float(hist["objective_initial"]),
        }
    return {
        "kind": "vmec_jax_qa_transport_candidate_comparison",
        "claim_scope": (
            "bounded VMEC-JAX solved-boundary candidate comparison; validates objective assembly, "
            "trace-safe SPECTRAX-GK transport residual, WOUT writing, and iota convention; not a "
            "converged optimized-equilibrium nonlinear turbulent-flux claim"
        ),
        "target_aspect": target_aspect,
        "target_mean_iota": target_iota,
        "target_iota_profile_floor": target_iota,
        "branches": branches,
        "summary": {
            "plumbing_passed": all(branch["gates"]["aspect_near_target"] for branch in branches)
            and all(branch["gates"]["mean_iota_above_target"] for branch in branches),
            "all_wout_iota_profiles_above_target": all(
                branch["gates"]["wout_iota_above_target"] for branch in branches
            ),
            "any_branch_optimized": any(branch["gates"]["optimized"] for branch in branches),
            "next_step": (
                "Run a longer VMEC-JAX optimization budget, then launch matched long-window "
                "SPECTRAX-GK nonlinear audits only after QS/objective reduction and held-out "
                "geometry gates pass."
            ),
        },
    }


def plot_payload(payload: dict[str, Any], out: Path) -> None:
    """Render a compact publication-style candidate comparison panel."""

    set_plot_style()
    colors = ["#244c66", "#b45f2a"]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6), constrained_layout=True)
    labels = [branch["label"] for branch in payload["branches"]]

    ax = axes[0]
    for branch, color in zip(payload["branches"], colors, strict=True):
        profile = branch["iota_profile"]
        ax.plot(profile["s"], profile["iota"], lw=2.2, color=color, label=branch["label"])
    ax.axhline(
        payload["target_iota_profile_floor"],
        color="black",
        lw=1.2,
        ls=":",
        label=r"profile floor $\iota=0.41$",
    )
    ax.set_xlabel("normalized toroidal flux")
    ax.set_ylabel(r"$\iota$")
    ax.set_title("Final WOUT iota profiles")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    x = np.arange(len(labels))
    width = 0.26
    metrics = [
        ("aspect_final", "aspect", payload["target_aspect"]),
        ("iota_final", "mean iota", payload["target_mean_iota"]),
        ("qs_final", "QS residual", None),
    ]
    for offset, (key, label, target) in zip((-width, 0.0, width), metrics, strict=True):
        values = [float(branch["history"][key]) for branch in payload["branches"]]
        ax.bar(x + offset, values, width, label=label)
        if target is not None and key != "aspect_final":
            ax.axhline(target, color="black", lw=1.0, ls=":", alpha=0.45)
    ax.set_xticks(x)
    ax.set_xticklabels(["QA", "QA+Q"], rotation=0)
    ax.set_title("Final scalar diagnostics")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    ax.axis("off")
    lines = [
        "Bounded VMEC-JAX candidate comparison",
        f"Target aspect: {payload['target_aspect']:.1f}",
        f"Target iota profile floor: {payload['target_iota_profile_floor']:.2f}",
        f"Plumbing passed: {payload['summary']['plumbing_passed']}",
        f"All WOUT iota profiles above target: {payload['summary']['all_wout_iota_profiles_above_target']}",
        f"Any branch optimized: {payload['summary']['any_branch_optimized']}",
        "",
        "Interpretation",
        "- Objective assembly and WOUT writing work.",
        "- Transport branch is trace-safe.",
        "- This small local budget is not an optimized design.",
        "- Longer VMEC-JAX solve + nonlinear audits are next.",
    ]
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.55", "fc": "#f7f4ee", "ec": "#d7c4a8", "alpha": 0.95},
    )
    fig.suptitle("VMEC-JAX QA candidate: constraints-only vs SPECTRAX-GK transport residual", fontsize=14)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def main() -> int:
    args = _parse_args()
    payload = build_payload(args.constraints_dir, args.transport_dir)
    out = args.out
    base = out.with_suffix("")
    base.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot_payload(payload, out)
    if args.pdf:
        plot_payload(payload, base.with_suffix(".pdf"))
    print(json.dumps({"passed": payload["summary"]["plumbing_passed"], "paths": {"png": str(out), "json": str(base.with_suffix('.json'))}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
