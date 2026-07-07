#!/usr/bin/env python3
"""Build the next external-VMEC nonlinear holdout launch runbook."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectraxgk.artifacts.plotting import set_plot_style  # noqa: E402


@dataclass(frozen=True)
class ExternalHoldoutScreenRow:
    """One row from an external-VMEC linear candidate screen."""

    case: str
    vmec_file: str
    returncode: int
    best_ky: float | None
    best_gamma: float | None
    best_omega: float | None
    log: str = ""

    @property
    def family(self) -> str:
        """Geometry family inferred from case and source path."""

        return external_vmec_family(self.case, self.vmec_file)

    @property
    def unstable(self) -> bool:
        """Whether the screen row is finite, successful, and linearly unstable."""

        return (
            self.returncode == 0
            and self.best_ky is not None
            and self.best_gamma is not None
            and self.best_gamma > 0.0
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly representation."""

        payload = asdict(self)
        payload["family"] = self.family
        payload["unstable"] = self.unstable
        return payload


def _finite_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def external_vmec_family(case: str, source: str = "") -> str:
    """Return a stable family label for an external-VMEC candidate."""

    text = f"{case} {source}".lower()
    if "updown" in text or "up-down" in text or "up_down" in text:
        return "updown_asym_external_vmec"
    if "itermodel" in text:
        return "itermodel_external_vmec"
    if "li383" in text:
        return "li383_external_vmec"
    if "qi_stel" in text or "quasi-isodynamic" in text or "nfp3_qi" in text:
        return "qi_external_vmec"
    if "qa" in text and ("landremanpaul" in text or "quasi-axisymmetric" in text):
        return "qa_external_vmec"
    if "dshape" in text or "d-shaped" in text or "d_shaped" in text:
        return "dshape_external_vmec"
    if "circular" in text:
        return "circular_external_vmec"
    if "cth" in text:
        return "cth_like_external_vmec"
    if "qh" in text or "nfp4" in text:
        return "qh_external_vmec"
    if "basic_non_stellsym" in text or "non_stellsym" in text:
        return "non_stellsym_external_vmec"
    if "purely_toroidal" in text:
        return "purely_toroidal_external_vmec"
    if "solovev" in text:
        return "solovev_external_vmec"
    if "shaped_tokamak" in text:
        return "shaped_tokamak_external_vmec"
    return "external_vmec"


def read_external_holdout_screen(path: str | Path) -> list[ExternalHoldoutScreenRow]:
    """Load a candidate-screen CSV produced by the linear external-VMEC sweep."""

    rows: list[ExternalHoldoutScreenRow] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                ExternalHoldoutScreenRow(
                    case=str(row.get("case", "")).strip(),
                    vmec_file=str(row.get("vmec_file", "")).strip(),
                    returncode=int(str(row.get("returncode", "1")).strip() or "1"),
                    best_ky=_finite_float(row.get("best_ky")),
                    best_gamma=_finite_float(row.get("best_gamma")),
                    best_omega=_finite_float(row.get("best_omega")),
                    log=str(row.get("log", "")).strip(),
                )
            )
    return rows


def _families_from_rows(rows: Iterable[dict[str, Any]]) -> set[str]:
    families: set[str] = set()
    for row in rows:
        family = str(row.get("geometry", ""))
        if "external_vmec" in family:
            families.add(family)
    return families


def _failed_external_families(gap_report: dict[str, Any]) -> set[str]:
    """Return external-VMEC families with tracked failed convergence gates."""

    families: set[str] = set()
    excluded = gap_report.get("excluded_candidates", [])
    if not isinstance(excluded, list):
        return families
    for row in excluded:
        if not isinstance(row, dict):
            continue
        family = str(row.get("geometry", ""))
        if "external_vmec" not in family:
            continue
        status = str(row.get("status", ""))
        failed = row.get("gate_passed") is False or status == "excluded_failed_external_gate"
        if failed:
            families.add(family)
    return families


def _passed_training_audit_families(gap_report: dict[str, Any]) -> set[str]:
    """Return represented families with a passed same-family audit gate."""

    families: set[str] = set()
    excluded = gap_report.get("excluded_candidates", [])
    if not isinstance(excluded, list):
        return families
    for row in excluded:
        if not isinstance(row, dict):
            continue
        if str(row.get("status", "")) != "excluded_same_family_training_audit":
            continue
        family = str(row.get("geometry", ""))
        if "external_vmec" in family and row.get("gate_passed") is True:
            families.add(family)
    return families


def _first_nearest_gap(gap_report: dict[str, Any]) -> dict[str, Any]:
    needed = gap_report.get("next_actual_nonlinear_holdout_needed", {})
    if not isinstance(needed, dict):
        return {}
    nearest = needed.get("nearest_tracked_gap", {})
    return nearest if isinstance(nearest, dict) else {}


def _preferred_family(gap_report: dict[str, Any]) -> str | None:
    needed = gap_report.get("next_actual_nonlinear_holdout_needed", {})
    if not isinstance(needed, dict):
        return None
    family = needed.get("preferred_family")
    return str(family) if family else None


def _tracked_horizon(case: str) -> float | None:
    match = re.search(r"\bt(\d+(?:p\d+)?)\b", str(case))
    if not match:
        return None
    return float(match.group(1).replace("p", "."))


def _recommended_horizons(nearest_gap: dict[str, Any]) -> list[float]:
    horizon = _tracked_horizon(str(nearest_gap.get("case", "")))
    if horizon is None:
        return [150.0, 250.0, 350.0]
    return [horizon, horizon + 100.0, horizon + 200.0]


def _validated_horizons(horizons: Iterable[float]) -> list[float]:
    values = [float(value) for value in horizons]
    if not values:
        raise ValueError("at least one nonlinear holdout horizon is required")
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("nonlinear holdout horizons must be finite and positive")
    if values != sorted(values):
        raise ValueError("nonlinear holdout horizons must be sorted increasingly")
    return values


def _candidate_status(
    row: ExternalHoldoutScreenRow,
    *,
    preferred_family: str | None,
    represented_families: set[str],
    failed_external_families: set[str],
    passed_training_audit_families: set[str],
    min_launch_gamma: float,
    allow_modified_protocol_families: set[str],
    modified_protocol_note: str,
) -> tuple[str, float, str]:
    if not row.unstable:
        return ("screen_rejected_stable_or_failed", 9.0, "screen row did not finish with positive growth")
    if float(row.best_gamma or 0.0) < float(min_launch_gamma):
        return (
            "screen_marginal_needs_linear_refinement",
            7.0,
            "positive growth is below the nonlinear-launch threshold; refine the linear branch before launching a transport holdout",
        )
    if (
        row.family in failed_external_families
        and row.family in allow_modified_protocol_families
        and row.family not in passed_training_audit_families
    ):
        return (
            "modified_protocol_failed_family_candidate",
            1.5,
            "this external-VMEC family has a tracked failed convergence gate, but an explicit modified-protocol rerun was requested; "
            f"protocol change: {modified_protocol_note}",
        )
    if row.family in failed_external_families and row.family != preferred_family:
        return (
            "recent_family_failed_external_gate",
            6.0,
            "this external-VMEC family has a tracked failed convergence gate; rerun only with a modified higher-resolution protocol",
        )
    if preferred_family and row.family == preferred_family:
        if row.family in represented_families:
            if row.family in passed_training_audit_families:
                return (
                    "preferred_family_audit_already_passed",
                    5.0,
                    "same-family audit already passed; relaunch only with a different independent geometry or a materially changed protocol",
                )
            return (
                "preferred_family_already_represented",
                3.0,
                "preferred family is already used in the current calibration portfolio; use only with an independent split/window",
            )
        return ("preferred_family_new_holdout", 0.0, "matches preferred gap-report family")
    if row.family not in represented_families:
        return ("new_family_holdout_candidate", 2.0, "unstable electrostatic-compatible VMEC family not yet represented")
    return ("represented_family_audit_candidate", 4.0, "family already represented; useful as an audit, not first holdout leverage")


@dataclass(frozen=True)
class _ExternalHoldoutRunbookContext:
    """Derived gap-report policy used to rank external-VMEC holdout screens."""

    preferred_family: str | None
    represented_families: set[str]
    failed_external_families: set[str]
    passed_training_audit_families: set[str]
    nearest_gap: dict[str, Any]
    recommended_horizons: list[float]
    allowed_modified_families: set[str]
    modified_protocol_note: str


def _external_holdout_runbook_context(
    *,
    gap_report: dict[str, Any],
    horizons: tuple[float, ...] | None,
    allow_modified_protocol_families: tuple[str, ...],
    modified_protocol_note: str,
) -> _ExternalHoldoutRunbookContext:
    """Collect gap-report state before ranking candidate nonlinear holdouts."""

    preferred = _preferred_family(gap_report)
    admitted = gap_report.get("admitted_holdouts", [])
    training = gap_report.get("training_references", [])
    nearest_gap = _first_nearest_gap(gap_report)
    allowed_modified = {
        str(family).strip()
        for family in allow_modified_protocol_families
        if str(family).strip()
    }
    note = str(modified_protocol_note).strip()
    if allowed_modified and not note:
        raise ValueError(
            "modified_protocol_note is required when allowing failed-family modified-protocol reruns"
        )
    return _ExternalHoldoutRunbookContext(
        preferred_family=preferred,
        represented_families=_families_from_rows([*admitted, *training]),
        failed_external_families=_failed_external_families(gap_report),
        passed_training_audit_families=_passed_training_audit_families(gap_report),
        nearest_gap=nearest_gap,
        recommended_horizons=(
            _validated_horizons(horizons)
            if horizons is not None
            else _recommended_horizons(nearest_gap)
        ),
        allowed_modified_families=allowed_modified,
        modified_protocol_note=note,
    )


def _rank_external_holdout_candidates(
    *,
    screen_rows: Iterable[ExternalHoldoutScreenRow],
    context: _ExternalHoldoutRunbookContext,
    min_launch_gamma: float,
) -> list[dict[str, Any]]:
    """Rank linear-screen candidates for the next nonlinear holdout runbook."""

    ranked: list[dict[str, Any]] = []
    for row in screen_rows:
        status, priority, reason = _candidate_status(
            row,
            preferred_family=context.preferred_family,
            represented_families=context.represented_families,
            failed_external_families=context.failed_external_families,
            passed_training_audit_families=context.passed_training_audit_families,
            min_launch_gamma=float(min_launch_gamma),
            allow_modified_protocol_families=context.allowed_modified_families,
            modified_protocol_note=context.modified_protocol_note,
        )
        gamma_key = -(row.best_gamma if row.best_gamma is not None else -math.inf)
        ky_key = -(row.best_ky if row.best_ky is not None else -math.inf)
        ranked.append(
            {
                **row.to_dict(),
                "status": status,
                "priority": priority,
                "reason": reason,
                "_sort_key": [priority, gamma_key, ky_key, row.case],
            }
        )
    ranked = sorted(ranked, key=lambda item: tuple(item["_sort_key"]))
    for idx, ranked_row in enumerate(ranked, start=1):
        ranked_row["rank"] = idx
        ranked_row.pop("_sort_key", None)
    return ranked


def _select_external_holdout_candidates(
    ranked: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Select the new-family launch and optional preferred-family audit rows."""

    selected_new = next(
        (
            row
            for row in ranked
            if row["status"]
            in {
                "preferred_family_new_holdout",
                "new_family_holdout_candidate",
                "modified_protocol_failed_family_candidate",
            }
        ),
        None,
    )
    selected_preferred_audit = next(
        (row for row in ranked if row["status"] == "preferred_family_already_represented"),
        None,
    )
    return selected_new, selected_preferred_audit


def _external_holdout_launch_command(
    *,
    row: dict[str, Any],
    case_suffix: str,
    out_suffix: str,
    out_dir: str,
    dt: float,
    horizons: list[float],
    grid_args: str,
) -> str:
    """Build the replayable configuration-generation command for one row."""

    case = str(row["case"]).replace("_nc", "")
    return (
        "python tools/campaigns/write_external_vmec_holdout_configs.py "
        f"--case {case}{case_suffix} "
        f"--vmec-file {row['vmec_file']} "
        f"--out-dir {out_dir}/{case}{out_suffix} "
        f"--ky {float(row['best_ky']):.12g} "
        f"--dt {float(dt):.12g} "
        f"--horizons {','.join(f'{value:.12g}' for value in horizons)} "
        f"{grid_args}"
    )


def _external_holdout_launch_commands(
    *,
    selected_new: dict[str, Any] | None,
    selected_preferred_audit: dict[str, Any] | None,
    out_dir: str,
    grids: tuple[str, ...],
    dt: float,
    horizons: list[float],
) -> list[str]:
    """Build launch commands for the selected holdout rows."""

    grid_args = " ".join(f"--grid {grid}" for grid in grids)
    launch_commands: list[str] = []
    if selected_new is not None:
        launch_commands.append(
            _external_holdout_launch_command(
                row=selected_new,
                case_suffix="_holdout",
                out_suffix="",
                out_dir=out_dir,
                dt=dt,
                horizons=horizons,
                grid_args=grid_args,
            )
        )
    if selected_preferred_audit is not None:
        launch_commands.append(
            _external_holdout_launch_command(
                row=selected_preferred_audit,
                case_suffix="_independent_audit",
                out_suffix="_audit",
                out_dir=out_dir,
                dt=dt,
                horizons=horizons,
                grid_args=grid_args,
            )
        )
    return launch_commands


def _external_holdout_acceptance_gate(min_launch_gamma: float) -> dict[str, Any]:
    """Return the fail-closed nonlinear admission requirements for the runbook."""

    return {
        "required_split": "holdout",
        "required_claim_level": "passed_grid_convergence_candidate_for_transport_holdout",
        "requires_grid_window_convergence": True,
        "requires_post_transient_window": True,
        "requires_independent_from_training_reference": True,
        "requires_explicit_modified_protocol_note_for_failed_families": True,
        "minimum_screen_growth_rate_for_launch": float(min_launch_gamma),
    }


def build_external_holdout_runbook(
    *,
    gap_report: dict[str, Any],
    screen_rows: Iterable[ExternalHoldoutScreenRow],
    out_dir: str = "tools_out/external_vmec_holdouts",
    grids: tuple[str, ...] = ("n48:48:48:32:32", "n64:64:64:40:40"),
    dt: float = 0.05,
    min_launch_gamma: float = 0.02,
    max_candidates: int = 6,
    horizons: tuple[float, ...] | None = None,
    allow_modified_protocol_families: tuple[str, ...] = (),
    modified_protocol_note: str = "",
) -> dict[str, Any]:
    """Build a JSON-ready runbook for the next nonlinear holdout campaign.

    The ranking first preserves the gap-report priority, then prefers unstable
    families absent from the current train/holdout portfolio.  The runbook is a
    launch plan only; promotion still requires the generated nonlinear traces to
    pass grid/window convergence and then enter calibration metadata as
    ``split=holdout``.
    """

    context = _external_holdout_runbook_context(
        gap_report=gap_report,
        horizons=horizons,
        allow_modified_protocol_families=allow_modified_protocol_families,
        modified_protocol_note=modified_protocol_note,
    )
    ranked = _rank_external_holdout_candidates(
        screen_rows=screen_rows,
        context=context,
        min_launch_gamma=min_launch_gamma,
    )
    selected_new, selected_preferred_audit = _select_external_holdout_candidates(ranked)
    launch_commands = _external_holdout_launch_commands(
        selected_new=selected_new,
        selected_preferred_audit=selected_preferred_audit,
        out_dir=out_dir,
        grids=grids,
        dt=dt,
        horizons=context.recommended_horizons,
    )

    return {
        "kind": "external_vmec_holdout_runbook",
        "claim_level": "nonlinear_holdout_launch_plan_not_transport_validation",
        "passed": bool(selected_new is not None or selected_preferred_audit is not None),
        "absolute_flux_promoted": False,
        "preferred_family": context.preferred_family,
        "represented_families": sorted(context.represented_families),
        "failed_external_families": sorted(context.failed_external_families),
        "allow_modified_protocol_families": sorted(context.allowed_modified_families),
        "modified_protocol_note": context.modified_protocol_note,
        "nearest_tracked_gap": context.nearest_gap,
        "recommended_horizons": context.recommended_horizons,
        "recommended_grids": list(grids),
        "dt": float(dt),
        "min_launch_gamma": float(min_launch_gamma),
        "selected_new_family_candidate": selected_new,
        "selected_preferred_family_audit": selected_preferred_audit,
        "ranked_candidates": ranked[: int(max_candidates)],
        "launch_commands": launch_commands,
        "acceptance_gate": _external_holdout_acceptance_gate(min_launch_gamma),
        "notes": (
            "Run the selected configurations on the large-run host, build a convergence gate with "
            "tools/artifacts/plot_external_vmec_nonlinear_convergence_gate.py, and admit the resulting transport "
            "window to quasilinear calibration only if the gate passes and the split is holdout."
        ),
    }


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
        raise argparse.ArgumentTypeError(
            "horizons must be a comma-separated list of numbers"
        ) from exc
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


def _write_panel(
    path: Path, runbook: dict[str, Any], *, dpi: int = 220, write_pdf: bool = True
) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        row for row in runbook.get("ranked_candidates", []) if isinstance(row, dict)
    ]
    labels = [_short_label(row.get("case", "")) for row in rows]
    gammas = [float(row.get("best_gamma") or 0.0) for row in rows]
    colors = []
    for row in rows:
        status = str(row.get("status", ""))
        if status in {"preferred_family_new_holdout", "new_family_holdout_candidate"}:
            colors.append("#2f7f5f")
        elif status == "modified_protocol_failed_family_candidate":
            colors.append("#3b6ea8")
        elif status in {
            "preferred_family_already_represented",
            "preferred_family_audit_already_passed",
        }:
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

    selected = (
        runbook.get("selected_new_family_candidate")
        or runbook.get("selected_preferred_family_audit")
        or {}
    )
    selected_case = (
        selected.get("case", "none") if isinstance(selected, dict) else "none"
    )
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
    ax_text.text(
        0.02,
        0.98,
        "\n".join(lines),
        ha="left",
        va="top",
        family="monospace",
        fontsize=10.5,
    )
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
    parser.add_argument(
        "--grid", action="append", default=None, help="Grid spec label:Nx:Ny:Nz:ntheta"
    )
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
    if (
        args.allow_modified_protocol_family
        and not str(args.modified_protocol_note).strip()
    ):
        parser.error(
            "--modified-protocol-note is required with --allow-modified-protocol-family"
        )
    gap_report = json.loads(args.gap_report.read_text(encoding="utf-8"))
    screen_rows = read_external_holdout_screen(args.screen)
    runbook = build_external_holdout_runbook(
        gap_report=gap_report,
        screen_rows=screen_rows,
        out_dir=str(args.out_dir),
        grids=tuple(args.grid or ("n48:48:48:32:32", "n64:64:64:40:40")),
        dt=float(args.dt),
        horizons=args.horizons,
        allow_modified_protocol_families=tuple(
            args.allow_modified_protocol_family or ()
        ),
        modified_protocol_note=str(args.modified_protocol_note).strip(),
        min_launch_gamma=float(args.min_launch_gamma),
        max_candidates=int(args.max_candidates),
    )
    outputs = _write_panel(
        args.out, runbook, dpi=int(args.dpi), write_pdf=not bool(args.no_pdf)
    )
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
    json_path.write_text(
        json.dumps(_json_clean(runbook), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(csv_path, runbook)
    print(
        json.dumps(
            {"passed": runbook["passed"], "json": str(json_path), **outputs},
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if bool(runbook["passed"]) else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
