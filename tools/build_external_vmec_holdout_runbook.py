#!/usr/bin/env python3
"""Build the next external-VMEC nonlinear holdout launch runbook."""

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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectraxgk.validation.external_holdout import (  # noqa: E402
    build_external_holdout_runbook,
    read_external_holdout_screen,
)
from spectraxgk.artifacts.plotting import set_plot_style  # noqa: E402


DEFAULT_GAP_REPORT = ROOT / "docs" / "_static" / "quasilinear_holdout_gap_report.json"
DEFAULT_SCREEN = ROOT / "docs" / "_static" / "external_vmec_candidate_linear_screen.csv"
DEFAULT_OUT = ROOT / "docs" / "_static" / "external_vmec_next_holdout_runbook.png"


def _repo_relative(path: str | Path) -> str:
    raw = Path(path)
    try:
        return raw.resolve().relative_to(ROOT.resolve()).as_posix()
    except (OSError, ValueError):
        return str(path)


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    return value


def _short_label(value: object, *, width: int = 34) -> str:
    text = str(value)
    if len(text) <= width:
        return text
    return text[: max(1, width - 3)].rstrip("_-. ") + "..."


def _parse_horizons(value: str) -> tuple[float, ...]:
    try:
        horizons = tuple(float(chunk) for chunk in value.split(",") if chunk.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("horizons must be a comma-separated list of numbers") from exc
    if not horizons:
        raise argparse.ArgumentTypeError("at least one horizon is required")
    if any(horizon <= 0.0 for horizon in horizons):
        raise argparse.ArgumentTypeError("horizons must be positive")
    if horizons != tuple(sorted(horizons)):
        raise argparse.ArgumentTypeError("horizons must be sorted increasingly")
    return horizons


def _write_csv(path: Path, runbook: dict[str, Any]) -> None:
    fields = [
        "rank",
        "case",
        "family",
        "status",
        "best_gamma",
        "best_ky",
        "best_omega",
        "reason",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in runbook.get("ranked_candidates", []):
            if not isinstance(row, dict):
                continue
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_panel(path: Path, runbook: dict[str, Any], *, dpi: int = 220, write_pdf: bool = True) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [row for row in runbook.get("ranked_candidates", []) if isinstance(row, dict)]
    labels = [_short_label(row.get("case", "")) for row in rows]
    gammas = [float(row.get("best_gamma") or 0.0) for row in rows]
    colors = []
    for row in rows:
        status = str(row.get("status", ""))
        if status in {"preferred_family_new_holdout", "new_family_holdout_candidate"}:
            colors.append("#2f7f5f")
        elif status == "modified_protocol_failed_family_candidate":
            colors.append("#3b6ea8")
        elif status in {"preferred_family_already_represented", "preferred_family_audit_already_passed"}:
            colors.append("#d89c32")
        elif status == "represented_family_audit_candidate":
            colors.append("#78909c")
        else:
            colors.append("#b44a3c")

    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), constrained_layout=True)
    ax_bar, ax_text = axes
    y = list(range(len(labels)))
    ax_bar.barh(y, gammas, color=colors, alpha=0.9)
    ax_bar.set_yticks(y, labels)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("screened linear growth rate")
    ax_bar.set_title("Ranked external-VMEC holdout candidates")
    ax_bar.grid(True, axis="x", alpha=0.25)

    selected = runbook.get("selected_new_family_candidate") or runbook.get("selected_preferred_family_audit") or {}
    selected_case = selected.get("case", "none") if isinstance(selected, dict) else "none"
    nearest = runbook.get("nearest_tracked_gap", {})
    nearest_case = nearest.get("case", "none") if isinstance(nearest, dict) else "none"
    lines = [
        f"Status: {'READY' if runbook.get('passed') else 'BLOCKED'}",
        f"Preferred family: {runbook.get('preferred_family')}",
        f"Min launch gamma: {float(runbook.get('min_launch_gamma', 0.0)):.3g}",
        f"Selected next candidate: {selected_case}",
        f"Nearest tracked gap: {nearest_case}",
        f"Horizons: {', '.join(str(v) for v in runbook.get('recommended_horizons', []))}",
        f"Grids: {', '.join(str(v) for v in runbook.get('recommended_grids', []))}",
        "",
        "Acceptance gate:",
        "- split=holdout",
        "- passed grid/window convergence",
        "- post-transient transport window",
        "- independent of training reference",
        "",
        "Claim boundary:",
        "launch plan only; no absolute-flux",
        "predictor is promoted by this panel.",
    ]
    ax_text.axis("off")
    ax_text.text(0.02, 0.98, "\n".join(lines), ha="left", va="top", family="monospace", fontsize=10.5)
    fig.suptitle("External-VMEC nonlinear holdout runbook", fontsize=14)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    outputs = {"png": str(path)}
    if write_pdf:
        pdf = path.with_suffix(".pdf")
        fig.savefig(pdf, bbox_inches="tight")
        outputs["pdf"] = str(pdf)
    plt.close(fig)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gap-report", type=Path, default=DEFAULT_GAP_REPORT)
    parser.add_argument("--screen", type=Path, default=DEFAULT_SCREEN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--out-dir", default="tools_out/external_vmec_holdouts")
    parser.add_argument("--grid", action="append", default=None, help="Grid spec label:Nx:Ny:Nz:ntheta")
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument(
        "--horizons",
        type=_parse_horizons,
        default=None,
        help="Optional comma-separated nonlinear horizons overriding the gap-derived recommendation.",
    )
    parser.add_argument(
        "--allow-modified-protocol-family",
        action="append",
        default=None,
        help=(
            "External-VMEC family with a tracked failed gate that may be relaunched only because the protocol "
            "is materially changed. Requires --modified-protocol-note."
        ),
    )
    parser.add_argument(
        "--modified-protocol-note",
        default="",
        help="Required note explaining the grid/window/horizon/protocol change for any failed-family rerun.",
    )
    parser.add_argument(
        "--min-launch-gamma",
        type=float,
        default=0.02,
        help="Minimum positive linear growth rate required before writing nonlinear launch commands.",
    )
    parser.add_argument("--max-candidates", type=int, default=6)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--no-pdf", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.allow_modified_protocol_family and not str(args.modified_protocol_note).strip():
        parser.error("--modified-protocol-note is required with --allow-modified-protocol-family")
    gap_report = json.loads(args.gap_report.read_text(encoding="utf-8"))
    screen_rows = read_external_holdout_screen(args.screen)
    runbook = build_external_holdout_runbook(
        gap_report=gap_report,
        screen_rows=screen_rows,
        out_dir=str(args.out_dir),
        grids=tuple(args.grid or ("n48:48:48:32:32", "n64:64:64:40:40")),
        dt=float(args.dt),
        horizons=args.horizons,
        allow_modified_protocol_families=tuple(args.allow_modified_protocol_family or ()),
        modified_protocol_note=str(args.modified_protocol_note).strip(),
        min_launch_gamma=float(args.min_launch_gamma),
        max_candidates=int(args.max_candidates),
    )
    outputs = _write_panel(args.out, runbook, dpi=int(args.dpi), write_pdf=not bool(args.no_pdf))
    json_path = args.out.with_suffix(".json")
    csv_path = args.out.with_suffix(".csv")
    runbook = dict(runbook)
    runbook["inputs"] = {
        "gap_report": _repo_relative(args.gap_report),
        "screen": _repo_relative(args.screen),
    }
    runbook["png"] = _repo_relative(outputs["png"])
    if "pdf" in outputs:
        runbook["pdf"] = _repo_relative(outputs["pdf"])
    runbook["csv"] = _repo_relative(csv_path)
    json_path.write_text(json.dumps(_json_clean(runbook), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(csv_path, runbook)
    print(json.dumps({"passed": runbook["passed"], "json": str(json_path), **outputs}, indent=2, sort_keys=True))
    return 0 if bool(runbook["passed"]) else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
