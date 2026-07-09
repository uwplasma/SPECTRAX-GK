"""Campaign follow-up planning for nonlinear replicate-spread blockers."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any
import math

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from spectraxgk.diagnostics.nonlinear_replicates import (  # noqa: E402
    NonlinearReplicateSpreadConfig,
    nonlinear_replicate_spread_report,
)

@dataclass(frozen=True)
class NonlinearReplicateFollowupConfig:
    """Options controlling the targeted replicate follow-up plan."""

    include_extra_nominal_seed: bool = True
    extra_seed_increment: int = 1
    max_runs_per_state: int = 3


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _finite_int(value: Any) -> int | None:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out if out >= 0 else None


def _variant_key(state: str, label: str) -> tuple[str, str]:
    return str(state), str(label)


def _metadata_lookup(
    variant_metadata: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in variant_metadata:
        state = row.get("state")
        label = row.get("variant_label") or row.get("label")
        if state is None or label is None:
            continue
        seed = _finite_int(row.get("seed"))
        timestep = _finite_float(row.get("timestep") or row.get("dt"))
        if seed is None or timestep is None:
            continue
        out[_variant_key(str(state), str(label))] = {
            "state": str(state),
            "variant_label": str(label),
            "variant_axis": str(row.get("variant_axis") or row.get("axis") or "unknown"),
            "seed": seed,
            "timestep": timestep,
            "source_config": row.get("source_config"),
            "source_output": row.get("source_output"),
        }
    return out


def _row_by_state(rows: Sequence[Mapping[str, Any]], state: str) -> dict[str, Any] | None:
    for row in rows:
        if str(row.get("state")) == state:
            return dict(row)
    return None


def _next_seed(metadata_rows: Sequence[Mapping[str, Any]], *, increment: int) -> int | None:
    seeds = [
        seed
        for seed in (_finite_int(row.get("seed")) for row in metadata_rows)
        if seed is not None
    ]
    if not seeds:
        return None
    return max(seeds) + max(1, int(increment))


def _planned_run(
    *,
    state: str,
    seed: int,
    timestep: float,
    reason: str,
    source_labels: Sequence[str],
) -> dict[str, Any]:
    timestep_label = f"{float(timestep):.12g}".replace(".", "p").replace("-", "m")
    return {
        "state": state,
        "variant_axis": "seed_timestep",
        "variant_label": f"seed{int(seed)}_dt{timestep_label}",
        "seed": int(seed),
        "timestep": float(timestep),
        "reason": reason,
        "source_variant_labels": list(source_labels),
    }


def _validated_config(
    config: NonlinearReplicateFollowupConfig | None,
) -> NonlinearReplicateFollowupConfig:
    cfg = config or NonlinearReplicateFollowupConfig()
    if cfg.max_runs_per_state <= 0:
        raise ValueError("max_runs_per_state must be positive")
    if cfg.extra_seed_increment <= 0:
        raise ValueError("extra_seed_increment must be positive")
    return cfg


def _state_rows_from_report(spread_report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    state_rows_raw = spread_report.get("state_rows")
    if not isinstance(state_rows_raw, Sequence):
        return []
    return [row for row in state_rows_raw if isinstance(row, Mapping)]


def _failed_states_from_report(spread_report: Mapping[str, Any]) -> list[Any]:
    summary = spread_report.get("summary")
    if not isinstance(summary, Mapping):
        return []
    return list(summary.get("failed_states", []))


def _state_metadata(
    lookup: Mapping[tuple[str, str], dict[str, Any]],
    state: str,
) -> list[dict[str, Any]]:
    return [row for key, row in lookup.items() if key[0] == state]


def _mixed_seed_timestep_runs(
    *,
    state: str,
    state_row: Mapping[str, Any],
    state_metadata: Sequence[Mapping[str, Any]],
    lookup: Mapping[tuple[str, str], dict[str, Any]],
    cfg: NonlinearReplicateFollowupConfig,
    missing_metadata: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    high_label = state_row.get("high_variant_label")
    low_label = state_row.get("low_variant_label")
    if not (high_label and low_label):
        return []
    high = lookup.get(_variant_key(state, str(high_label)))
    low = lookup.get(_variant_key(state, str(low_label)))
    if high is None or low is None:
        missing_metadata.append(
            {
                "state": state,
                "high_variant_label": high_label,
                "low_variant_label": low_label,
                "reason": "missing seed/timestep metadata for high or low variant",
            }
        )
        return []
    runs = [
        _planned_run(
            state=state,
            seed=int(low["seed"]),
            timestep=float(high["timestep"]),
            reason="test whether the low window follows the seed when the timestep is nominal",
            source_labels=[str(low_label), str(high_label)],
        ),
        _planned_run(
            state=state,
            seed=int(high["seed"]),
            timestep=float(low["timestep"]),
            reason="test whether the high window follows the seed when the timestep is refined",
            source_labels=[str(high_label), str(low_label)],
        ),
    ]
    if cfg.include_extra_nominal_seed:
        extra_seed = _next_seed(state_metadata, increment=cfg.extra_seed_increment)
        if extra_seed is not None:
            runs.append(
                _planned_run(
                    state=state,
                    seed=extra_seed,
                    timestep=float(high["timestep"]),
                    reason="add one independent nominal-timestep seed after the cross checks",
                    source_labels=[str(high_label)],
                )
            )
    return runs


def _seed_spread_runs(
    *,
    state: str,
    state_metadata: Sequence[Mapping[str, Any]],
    cfg: NonlinearReplicateFollowupConfig,
) -> list[dict[str, Any]]:
    extra_seed = _next_seed(state_metadata, increment=cfg.extra_seed_increment)
    nominal_dt = None
    for row in state_metadata:
        if str(row.get("variant_axis")) == "seed":
            nominal_dt = _finite_float(row.get("timestep"))
            break
    if extra_seed is None or nominal_dt is None:
        return []
    return [
        _planned_run(
            state=state,
            seed=extra_seed,
            timestep=nominal_dt,
            reason="seed spread dominates; add one independent nominal-timestep seed",
            source_labels=[],
        )
    ]


def _timestep_spread_runs(
    *,
    state: str,
    state_metadata: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    nominal_seed = None
    refined_dt = None
    for row in state_metadata:
        if str(row.get("variant_axis")) == "timestep":
            nominal_seed = _finite_int(row.get("seed"))
            refined_dt = _finite_float(row.get("timestep"))
            break
    if nominal_seed is None or refined_dt is None:
        return []
    return [
        _planned_run(
            state=state,
            seed=nominal_seed,
            timestep=refined_dt,
            reason="timestep spread dominates; repeat the refined-timestep replicate before promotion",
            source_labels=[],
        )
    ]


def _dedupe_and_limit_runs(
    runs: Sequence[Mapping[str, Any]],
    *,
    max_runs: int,
) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[int, float]] = set()
    for row in runs:
        key = (int(row["seed"]), float(row["timestep"]))
        if key not in seen:
            seen.add(key)
            deduped.append(dict(row))
    return deduped[: int(max_runs)]


def _runs_for_failed_state(
    *,
    state: str,
    state_row: Mapping[str, Any],
    state_metadata: Sequence[Mapping[str, Any]],
    lookup: Mapping[tuple[str, str], dict[str, Any]],
    cfg: NonlinearReplicateFollowupConfig,
    missing_metadata: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    classification = str(state_row.get("classification", ""))
    if classification == "mixed_seed_timestep_spread":
        runs = _mixed_seed_timestep_runs(
            state=state,
            state_row=state_row,
            state_metadata=state_metadata,
            lookup=lookup,
            cfg=cfg,
            missing_metadata=missing_metadata,
        )
    elif classification == "seed_spread_limited":
        runs = _seed_spread_runs(state=state, state_metadata=state_metadata, cfg=cfg)
    elif classification == "timestep_spread_limited":
        runs = _timestep_spread_runs(state=state, state_metadata=state_metadata)
    else:
        runs = []
    return classification, _dedupe_and_limit_runs(
        runs, max_runs=cfg.max_runs_per_state
    )


def _state_plan_payload(
    *,
    state: str,
    classification: str,
    planned_runs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "state": state,
        "classification": classification,
        "planned_run_count": len(planned_runs),
        "planned_runs": list(planned_runs),
        "recommendation": (
            "Run these targeted cross variants before rerunning the ensemble and central-FD gates."
            if planned_runs
            else "No runnable follow-up was selected; inspect metadata before spending more GPU time."
        ),
    }


def _followup_config_payload(cfg: NonlinearReplicateFollowupConfig) -> dict[str, Any]:
    return {
        "include_extra_nominal_seed": bool(cfg.include_extra_nominal_seed),
        "extra_seed_increment": int(cfg.extra_seed_increment),
        "max_runs_per_state": int(cfg.max_runs_per_state),
    }


def nonlinear_replicate_followup_plan(
    spread_report: Mapping[str, Any],
    *,
    variant_metadata: Sequence[Mapping[str, Any]],
    case: str = "nonlinear_replicate_followup_plan",
    config: NonlinearReplicateFollowupConfig | None = None,
) -> dict[str, Any]:
    """Return targeted cross-run follow-ups for failed replicate-spread states."""

    cfg = _validated_config(config)
    lookup = _metadata_lookup(variant_metadata)
    state_rows = _state_rows_from_report(spread_report)
    failed_states = _failed_states_from_report(spread_report)
    planned: list[dict[str, Any]] = []
    state_plans: list[dict[str, Any]] = []
    missing_metadata: list[dict[str, Any]] = []

    for raw_state in failed_states:
        state = str(raw_state)
        state_row = _row_by_state(state_rows, state)
        if state_row is None:
            missing_metadata.append(
                {"state": state, "reason": "missing state row in spread report"}
            )
            continue
        classification, deduped = _runs_for_failed_state(
            state=state,
            state_row=state_row,
            state_metadata=_state_metadata(lookup, state),
            lookup=lookup,
            cfg=cfg,
            missing_metadata=missing_metadata,
        )
        planned.extend(deduped)
        state_plans.append(
            _state_plan_payload(
                state=state,
                classification=classification,
                planned_runs=deduped,
            )
        )

    return {
        "kind": "nonlinear_replicate_followup_plan",
        "claim_level": "targeted_replicate_disambiguation_launch_plan_not_simulation_claim",
        "case": str(case),
        "passed": not planned and not missing_metadata,
        "summary": {
            "failed_state_count": len(failed_states),
            "planned_run_count": len(planned),
            "missing_metadata_count": len(missing_metadata),
            "recommendation": (
                "Run planned cross variants, rebuild the failed ensemble, then rerun the central-FD gate."
                if planned
                else "No additional cross variants are currently required."
            ),
        },
        "state_plans": state_plans,
        "planned_runs": planned,
        "missing_metadata": missing_metadata,
        "config": _followup_config_payload(cfg),
    }


def build_spread_summary_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize replicated nonlinear-window spread from ensemble JSON artifacts."
    )
    parser.add_argument(
        "ensembles", nargs="+", type=Path, help="Nonlinear ensemble JSON files."
    )
    parser.add_argument("--out-prefix", type=Path, required=True)
    parser.add_argument("--case", default="nonlinear_replicate_spread_diagnostic")
    parser.add_argument("--max-mean-rel-spread", type=float, default=0.15)
    parser.add_argument("--value-floor", type=float, default=1.0e-12)
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} did not contain a JSON object")
    return payload


def _candidate_paths(raw: object, *, ensemble_path: Path) -> list[Path]:
    if not isinstance(raw, str) or not raw:
        return []
    path = Path(raw)
    if path.is_absolute():
        return [path]
    return [
        ROOT / path,
        ensemble_path.parent / path,
        ensemble_path.parent / path.name,
    ]


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _convergence_path(summary_path: Path) -> Path | None:
    candidate = (
        summary_path.parent
        / "nonlinear_window_convergence_reports"
        / f"{summary_path.stem}.convergence.json"
    )
    if candidate.exists():
        return candidate
    sibling = summary_path.with_suffix(".convergence.json")
    return sibling if sibling.exists() else None


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _enrich_ensemble(payload: dict[str, Any], *, ensemble_path: Path) -> dict[str, Any]:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return payload
    for row in rows:
        if not isinstance(row, dict):
            continue
        summary_path = _first_existing(
            _candidate_paths(row.get("summary_artifact"), ensemble_path=ensemble_path)
        )
        if summary_path is not None:
            summary = _read_json(summary_path)
            for key in ("variant_label", "variant_axis", "variant", "seed", "dt"):
                if key in summary and key not in row:
                    row[key] = summary[key]
            convergence_path = _convergence_path(summary_path)
            if convergence_path is not None:
                convergence = _read_json(convergence_path)
                stats = convergence.get("statistics")
                if isinstance(stats, dict):
                    row["window_statistics"] = stats
                row["convergence_artifact"] = _display_path(convergence_path)
    return payload


def _write_spread_csv(report: dict[str, Any], out_csv: Path) -> None:
    rows = list(report.get("replicate_rows", []))
    fieldnames = [
        "state",
        "index",
        "variant_label",
        "variant_axis",
        "late_mean",
        "sem",
        "ensemble_mean",
        "relative_delta",
        "running_mean_rel_drift",
        "terminal_mean_rel_delta",
        "sem_rel",
        "n_blocks",
        "passed",
        "promotion_ready",
        "source_artifact",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _write_spread_png(report: dict[str, Any], out_png: Path) -> None:
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spectraxgk.artifacts.plotting import set_plot_style

    replicate_rows = list(report.get("replicate_rows", []))
    state_rows = {str(row["state"]): row for row in report.get("state_rows", [])}
    states = list(dict.fromkeys(str(row["state"]) for row in replicate_rows))
    offsets = {"seed": -0.24, "timestep": 0.24, "seed_timestep": 0.0, "unknown": 0.0}
    colors = {
        "seed": "#2563eb",
        "timestep": "#d97706",
        "seed_timestep": "#4b5563",
        "unknown": "#6b7280",
    }

    set_plot_style()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.2, 4.8), constrained_layout=True)
    plotted_values: list[float] = []
    for state_index, state in enumerate(states):
        group = [row for row in replicate_rows if str(row["state"]) == state]
        for local_index, row in enumerate(group):
            axis = str(row.get("variant_axis") or "unknown")
            offset = offsets.get(axis, 0.0) + 0.06 * (
                local_index - (len(group) - 1) / 2
            )
            x = state_index + offset
            mean = row.get("late_mean")
            if mean is None:
                continue
            sem = 0.0 if row.get("sem") is None else float(row["sem"])
            plotted_values.extend([float(mean) - sem, float(mean) + sem])
            ax.errorbar(
                [x],
                [float(mean)],
                yerr=[sem],
                fmt="o",
                ms=6.0,
                capsize=3.0,
                color=colors.get(axis, colors["unknown"]),
                label=axis if axis not in ax.get_legend_handles_labels()[1] else None,
            )
            ax.text(
                x,
                float(mean) + max(sem, 0.05),
                str(row.get("variant_label", "")),
                rotation=45,
                ha="left",
                va="bottom",
                fontsize=6.5,
            )
        state_summary = state_rows.get(state, {})
        ensemble_mean = state_summary.get("ensemble_mean")
        if ensemble_mean is not None:
            ax.hlines(
                float(ensemble_mean),
                state_index - 0.36,
                state_index + 0.36,
                color="0.2",
                lw=1.2,
                ls="--",
            )
        if state_summary.get("classification") != "passed_replicate_spread_gate":
            ax.axvspan(
                state_index - 0.45,
                state_index + 0.45,
                color="#fee2e2",
                alpha=0.35,
                lw=0,
            )

    ax.set_xticks(np.arange(len(states)), states)
    if plotted_values:
        ymin = min(plotted_values)
        ymax = max(plotted_values)
        pad = max(0.25, 0.16 * (ymax - ymin))
        ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_ylabel("post-transient ion heat flux")
    ax.set_title("Replicated nonlinear-window spread diagnostic")
    ax.grid(True, axis="y", alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    ax.legend(dedup.values(), dedup.keys(), frameon=False, loc="best", fontsize=8)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main_spread_summary(argv: list[str] | None = None) -> int:
    args = build_spread_summary_parser().parse_args(argv)
    ensembles = [
        _enrich_ensemble(_read_json(path), ensemble_path=path)
        for path in args.ensembles
    ]
    report = nonlinear_replicate_spread_report(
        ensembles,
        case=args.case,
        config=NonlinearReplicateSpreadConfig(
            max_mean_rel_spread=args.max_mean_rel_spread,
            value_floor=args.value_floor,
        ),
    )
    out_json = args.out_prefix.with_suffix(".json")
    out_csv = args.out_prefix.with_suffix(".csv")
    out_png = args.out_prefix.with_suffix(".png")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_spread_csv(report, out_csv)
    _write_spread_png(report, out_png)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("spread-summary",),
        help="Replicate follow-up artifact to build.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    if not tokens or tokens[0] in {"-h", "--help"}:
        build_parser().parse_args(tokens)
        return 0
    command, rest = tokens[0], tokens[1:]
    if command == "spread-summary":
        return main_spread_summary(rest)
    build_parser().parse_args([command])
    return 2


__all__ = [
    "NonlinearReplicateFollowupConfig",
    "main",
    "nonlinear_replicate_followup_plan",
]


if __name__ == "__main__":
    raise SystemExit(main())
