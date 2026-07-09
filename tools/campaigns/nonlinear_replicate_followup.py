"""Campaign follow-up planning for nonlinear replicate-spread blockers."""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import shlex
import sys
from typing import Any
import math

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from spectraxgk.diagnostics.nonlinear_replicates import (  # noqa: E402
    NonlinearReplicateSpreadConfig,
    nonlinear_replicate_spread_report,
)

from tools.campaigns.write_external_vmec_holdout_configs import (  # noqa: E402
    _parse_grid,
    write_configs,
    write_manifest,
)

tomllib: Any = importlib.import_module(
    "tomllib" if sys.version_info >= (3, 11) else "tomli"
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



STATE_ORDER = ("baseline", "plus_delta", "minus_delta")


def _repo_relative(path: Path | str) -> str:
    raw = Path(path)
    try:
        return raw.resolve().relative_to(ROOT.resolve()).as_posix()
    except (OSError, ValueError):
        return str(path)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _config_from_direct_command(command: str) -> Path:
    parts = shlex.split(command)
    try:
        raw = parts[parts.index("--config") + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"direct command is missing '--config': {command}") from exc
    return ROOT / raw


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _metadata_from_config(path: Path, *, state: str) -> dict[str, Any]:
    payload = _read_toml(path)
    metadata = payload.get("metadata", {})
    init = payload.get("init", {})
    time = payload.get("time", {})
    output = payload.get("output", {})
    if not isinstance(metadata, dict):
        metadata = {}
    return {
        "state": state,
        "variant_label": str(metadata.get("variant_label") or path.stem),
        "variant_axis": str(metadata.get("variant_axis") or "unknown"),
        "seed": int(metadata.get("seed", init.get("random_seed", 0))),
        "timestep": float(metadata.get("timestep", time.get("dt", 0.0))),
        "source_config": _repo_relative(path),
        "source_output": _repo_relative(
            path.parent / str(output.get("path", path.with_suffix(".out.nc").name))
        ),
    }


def collect_variant_metadata(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Read variant metadata from all direct full-horizon TOMLs in a manifest."""

    state_commands = manifest.get("state_ensemble_commands")
    if not isinstance(state_commands, dict):
        raise ValueError("manifest is missing state_ensemble_commands")
    rows: list[dict[str, Any]] = []
    ordered_states = [state for state in STATE_ORDER if state in state_commands]
    ordered_states.extend(
        sorted(state for state in state_commands if state not in ordered_states)
    )
    for state in ordered_states:
        raw_row = state_commands[state]
        if not isinstance(raw_row, dict):
            continue
        for command in raw_row.get("direct_full_horizon_launch_commands", []):
            config = _config_from_direct_command(str(command))
            rows.append(_metadata_from_config(config, state=state))
    return rows


def _state_reference_config(
    manifest: dict[str, Any], state: str
) -> tuple[Path, dict[str, Any]]:
    row = manifest["state_ensemble_commands"][state]
    commands = row.get("direct_full_horizon_launch_commands", [])
    if not commands:
        raise ValueError(f"state {state!r} has no direct full-horizon launch commands")
    config_path = _config_from_direct_command(str(commands[0]))
    return config_path, _read_toml(config_path)


def _grid_spec_from_toml(payload: dict[str, Any], *, fallback_label: str) -> str:
    grid = payload.get("grid")
    if not isinstance(grid, dict):
        raise ValueError("reference config is missing [grid]")
    return "{label}:{nx}:{ny}:{nz}:{ntheta}".format(
        label=fallback_label,
        nx=int(grid["Nx"]),
        ny=int(grid["Ny"]),
        nz=int(grid["Nz"]),
        ntheta=int(grid["ntheta"]),
    )


def _value(payload: dict[str, Any], section: str, key: str, default: Any) -> Any:
    raw = payload.get(section)
    if isinstance(raw, dict) and key in raw:
        return raw[key]
    return default


def _write_followup_configs(
    *,
    manifest: dict[str, Any],
    plan: dict[str, Any],
) -> dict[str, Any]:
    run_contract = manifest.get("run_contract")
    if not isinstance(run_contract, dict):
        raise ValueError("manifest is missing run_contract")
    tmax = float(
        run_contract.get(
            "analysis_window", [0.0, run_contract.get("minimum_tmax", 0.0)]
        )[1]
    )
    grid_label = str(run_contract.get("grid", "n64"))
    by_state: dict[str, list[dict[str, Any]]] = {}
    for row in plan.get("planned_runs", []):
        if isinstance(row, dict):
            by_state.setdefault(str(row["state"]), []).append(row)

    written_by_state: dict[str, Any] = {}
    for state, rows in by_state.items():
        reference_path, reference = _state_reference_config(manifest, state)
        geometry = reference.get("geometry")
        if not isinstance(geometry, dict):
            raise ValueError(f"reference config {reference_path} is missing [geometry]")
        vmec_file = (reference_path.parent / str(geometry["vmec_file"])).resolve()
        state_case = str(
            _value(reference, "metadata", "case", f"{manifest['case']}_{state}")
        )
        out_dir = reference_path.parent
        joint = tuple((int(row["seed"]), float(row["timestep"])) for row in rows)
        written = write_configs(
            case=state_case,
            vmec_file=vmec_file,
            out_dir=out_dir,
            grids=[
                _parse_grid(_grid_spec_from_toml(reference, fallback_label=grid_label))
            ],
            horizons=(tmax,),
            dt=float(_value(reference, "time", "dt", 0.05)),
            ky=float(_value(reference, "run", "ky", 0.47619047619047616)),
            nl=int(_value(reference, "run", "Nl", 4)),
            nm=int(_value(reference, "run", "Nm", 8)),
            torflux=float(_value(reference, "geometry", "torflux", 0.64)),
            alpha=float(_value(reference, "geometry", "alpha", 0.0)),
            npol=float(_value(reference, "geometry", "npol", 1.0)),
            tprim=float(reference.get("species", [{}])[0].get("tprim", 3.0)),
            fprim=float(reference.get("species", [{}])[0].get("fprim", 1.0)),
            nu=float(reference.get("species", [{}])[0].get("nu", 0.01)),
            init_amp=float(_value(reference, "init", "init_amp", 1.0e-3)),
            y0=float(_value(reference, "grid", "y0", 21.0)),
            lx=float(_value(reference, "grid", "Lx", 62.8)),
            ly=float(_value(reference, "grid", "Ly", 62.8)),
            sample_stride=int(_value(reference, "time", "sample_stride", 50)),
            diagnostics_stride=int(_value(reference, "time", "diagnostics_stride", 50)),
            progress_bar=bool(_value(reference, "time", "progress_bar", False)),
            baseline_seed=int(_value(reference, "init", "random_seed", 22)),
            seed_dt_variants=joint,
        )
        manifest_path = write_manifest(out_dir, written)
        direct_commands = [
            (
                "python3 -m spectraxgk.cli run-runtime-nonlinear "
                f"--config {_repo_relative(item.path)} "
                f"--steps {int(round(tmax / float(item.variant.dt if item.variant else _value(reference, 'time', 'dt', 0.05))))} "
                "--no-progress"
            )
            for item in written
        ]
        written_by_state[state] = {
            "state": state,
            "reference_config": _repo_relative(reference_path),
            "run_manifest": _repo_relative(manifest_path),
            "configs": [
                {
                    "path": _repo_relative(item.path),
                    "output": _repo_relative(item.output_path),
                    "variant_label": item.variant.label if item.variant else None,
                    "seed": item.variant.random_seed if item.variant else None,
                    "timestep": item.variant.dt if item.variant else None,
                    "steps": int(
                        round(
                            tmax
                            / float(
                                item.variant.dt
                                if item.variant
                                else _value(reference, "time", "dt", 0.05)
                            )
                        )
                    ),
                }
                for item in written
            ],
            "direct_full_horizon_launch_commands": direct_commands,
        }
    return written_by_state


def _planned_outputs_for_state(
    written_by_state: dict[str, Any], state: str
) -> list[str]:
    state_payload = written_by_state.get(state)
    if not isinstance(state_payload, dict):
        return []
    return [
        str(row["output"])
        for row in state_payload.get("configs", [])
        if isinstance(row, dict) and row.get("output")
    ]


def _postprocess_commands(
    *,
    manifest: dict[str, Any],
    written_by_state: dict[str, Any],
) -> dict[str, Any]:
    run_contract = manifest.get("run_contract")
    if not isinstance(run_contract, dict):
        return {}
    analysis_window = run_contract.get("analysis_window", [0.0, 0.0])
    tmin = float(analysis_window[0])
    tmax = float(analysis_window[1])
    t_label = (
        str(int(round(tmax)))
        if abs(tmax - round(tmax)) < 1.0e-12
        else f"{tmax:.12g}".replace(".", "p")
    )
    commands: dict[str, Any] = {}
    state_commands = manifest.get("state_ensemble_commands")
    if not isinstance(state_commands, dict):
        return commands
    for state in sorted(written_by_state):
        original = state_commands.get(state)
        if not isinstance(original, dict):
            continue
        ensemble_json = Path(str(original.get("ensemble_json", "")))
        if not ensemble_json.name:
            continue
        ensemble_dir = ensemble_json.parent
        existing_outputs = [str(path) for path in original.get("expected_outputs", [])]
        planned_outputs = _planned_outputs_for_state(written_by_state, state)
        all_outputs = existing_outputs + planned_outputs
        prefix = f"{manifest['case']}_{state}_t{t_label}_followup"
        output_gate_json = ensemble_dir / f"{prefix}_output_gate.json"
        ensemble_gate_json = f"{prefix}_ensemble_gate.json"
        readiness_json = f"{prefix}_ensemble_readiness.json"
        ensemble_png = f"{prefix}_ensemble_gate.png"
        commands[state] = {
            "all_expected_outputs": all_outputs,
            "output_gate_json": _repo_relative(output_gate_json),
            "output_gate_command": (
                "python3 tools/release/check_nonlinear_runtime_outputs.py "
                + " ".join(all_outputs)
                + f" --min-samples 200 --tmin {tmin:.12g} --tmax {tmax:.12g}"
                + " --min-window-samples 80 --min-abs-window-mean 1e-4"
                + f" --json-out {_repo_relative(output_gate_json)}"
            ),
            "ensemble_json": _repo_relative(ensemble_dir / ensemble_gate_json),
            "readiness_json": _repo_relative(ensemble_dir / readiness_json),
            "ensemble_png": _repo_relative(ensemble_dir / ensemble_png),
            "build_ensemble_command": (
                "python3 tools/artifacts/build_external_vmec_replicate_ensemble.py "
                + " ".join(all_outputs)
                + f" --out-dir {_repo_relative(ensemble_dir)}"
                + f" --case {manifest['case']}_{state}_replicated_nonlinear_window_followup"
                + f" --tmin {tmin:.12g} --tmax {tmax:.12g}"
                + f" --artifact-prefix {_repo_relative(ensemble_dir)}"
                + f" --readiness-json {readiness_json}"
                + f" --ensemble-json {ensemble_gate_json}"
                + f" --out-png {ensemble_png}"
            ),
        }

    baseline_json = state_commands.get("baseline", {}).get("ensemble_json")
    minus_json = state_commands.get("minus_delta", {}).get("ensemble_json")
    if baseline_json and minus_json:
        for state, row in commands.items():
            if state != "plus_delta":
                continue
            spread_prefix = (
                ROOT
                / "docs"
                / "_static"
                / f"{manifest['case']}_{state}_followup_replicate_spread_diagnostic"
            )
            fd_prefix = (
                ROOT
                / "docs"
                / "_static"
                / f"{manifest['case']}_{state}_followup_central_fd_gradient_gate"
            )
            evidence_json = (
                ROOT
                / "docs"
                / "_static"
                / f"{manifest['case']}_{state}_followup_evidence_status.json"
            )
            gap_json = (
                ROOT
                / "docs"
                / "_static"
                / f"{manifest['case']}_{state}_followup_evidence_gap_report.json"
            )
            row["replicate_spread_command"] = (
                f"python3 tools/campaigns/nonlinear_replicate_followup.py spread-summary {baseline_json} "
                f"{row['ensemble_json']} {minus_json} --out-prefix {_repo_relative(spread_prefix)} "
                f"--case {manifest['case']}_{state}_followup_replicate_spread"
            )
            row["central_fd_command"] = (
                "python3 tools/artifacts/build_nonlinear_turbulence_gradient_fd_gate.py "
                f"--baseline {baseline_json} --plus {row['ensemble_json']} --minus {minus_json} "
                f"--delta-parameter {float(manifest['delta_parameter']):.12g} "
                f"--parameter-name {manifest['parameter_name']} "
                f"--out-prefix {_repo_relative(fd_prefix)} --fail-on-blocked"
            )
            row["evidence_check_command"] = (
                "python3 tools/release/check_nonlinear_turbulence_gradient_evidence.py "
                f"--gradient-artifact {_repo_relative(fd_prefix.with_suffix('.json'))} "
                f"--window-artifact {baseline_json} --window-artifact {row['ensemble_json']} "
                f"--window-artifact {minus_json} --json-out {_repo_relative(evidence_json)} "
                f"--gap-json-out {_repo_relative(gap_json)} --fail-on-blocked"
            )
    return commands


def build_followup_campaign(
    *,
    campaign_manifest_path: Path,
    spread_diagnostic_path: Path,
    out_json: Path,
    case: str,
    include_extra_nominal_seed: bool,
    max_runs_per_state: int,
    dry_run: bool = False,
) -> dict[str, Any]:
    campaign_manifest = _load_json(campaign_manifest_path)
    spread_report = _load_json(spread_diagnostic_path)
    metadata = collect_variant_metadata(campaign_manifest)
    plan = nonlinear_replicate_followup_plan(
        spread_report,
        variant_metadata=metadata,
        case=case,
        config=NonlinearReplicateFollowupConfig(
            include_extra_nominal_seed=include_extra_nominal_seed,
            max_runs_per_state=max_runs_per_state,
        ),
    )
    written = (
        {}
        if dry_run
        else _write_followup_configs(manifest=campaign_manifest, plan=plan)
    )
    postprocess = (
        {}
        if dry_run
        else _postprocess_commands(manifest=campaign_manifest, written_by_state=written)
    )
    payload = {
        **plan,
        "campaign_manifest": _repo_relative(campaign_manifest_path),
        "spread_diagnostic": _repo_relative(spread_diagnostic_path),
        "variant_metadata": metadata,
        "written_configs_by_state": written,
        "postprocess_commands_by_state": postprocess,
        "dry_run": bool(dry_run),
        "next_action": (
            "Run the direct_full_horizon_launch_commands for each written state, rebuild the failed "
            "ensemble with the added outputs, then rerun the replicate-spread and central-FD gates."
        ),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def build_write_campaign_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-manifest", required=True, type=Path)
    parser.add_argument("--spread-diagnostic", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--case", default="nonlinear_replicate_followup")
    parser.add_argument("--no-extra-nominal-seed", action="store_true")
    parser.add_argument("--max-runs-per-state", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main_write_campaign(argv: list[str] | None = None) -> int:
    args = build_write_campaign_parser().parse_args(argv)
    payload = build_followup_campaign(
        campaign_manifest_path=Path(args.campaign_manifest),
        spread_diagnostic_path=Path(args.spread_diagnostic),
        out_json=Path(args.out_json),
        case=str(args.case),
        include_extra_nominal_seed=not bool(args.no_extra_nominal_seed),
        max_runs_per_state=int(args.max_runs_per_state),
        dry_run=bool(args.dry_run),
    )
    print(
        json.dumps(
            {
                "planned_run_count": payload["summary"]["planned_run_count"],
                "states": sorted(payload["written_configs_by_state"]),
                "out_json": _repo_relative(args.out_json),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("spread-summary", "write-campaign"),
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
    if command == "write-campaign":
        return main_write_campaign(rest)
    build_parser().parse_args([command])
    return 2


__all__ = [
    "NonlinearReplicateFollowupConfig",
    "build_followup_campaign",
    "main",
    "nonlinear_replicate_followup_plan",
]


if __name__ == "__main__":
    raise SystemExit(main())
