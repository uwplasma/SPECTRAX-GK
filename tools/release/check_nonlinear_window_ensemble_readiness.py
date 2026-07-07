#!/usr/bin/env python3
"""Convert transport-window summaries into ensemble-readiness metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from spectraxgk.validation.quasilinear.window_config import (  # noqa: E402
    NonlinearWindowConvergenceConfig,
    NonlinearWindowEnsembleManifestConfig,
)
from spectraxgk.validation.quasilinear.window_io import (  # noqa: E402
    nonlinear_window_convergence_from_summary,
)
from spectraxgk.validation.quasilinear.window_ensemble import (  # noqa: E402
    nonlinear_window_ensemble_artifact_manifest,
)


VARIANT_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "seed": ("seed", "random_seed", "rng_seed", "simulation_seed"),
    "timestep": ("timestep", "time_step", "dt"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "summaries",
        nargs="+",
        type=Path,
        help="Transport-window summary JSON files with trace CSV provenance.",
    )
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument(
        "--reports-dir",
        type=Path,
        help="Optional directory for per-summary convergence report JSON files.",
    )
    parser.add_argument("--case", default="nonlinear_window_ensemble_readiness")
    parser.add_argument("--time-column", default="t")
    parser.add_argument("--value-column", default="heat_flux")
    parser.add_argument("--min-samples", type=int, default=24)
    parser.add_argument("--min-blocks", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=256)
    parser.add_argument("--max-running-mean-rel-drift", type=float, default=0.15)
    parser.add_argument("--max-terminal-mean-rel-delta", type=float, default=0.10)
    parser.add_argument("--max-sem-rel", type=float, default=0.25)
    parser.add_argument("--min-replicates-per-case", type=int, default=2)
    parser.add_argument(
        "--variant-axis",
        action="append",
        choices=tuple(VARIANT_KEY_ALIASES),
        help="Required replicated variant axis. Defaults to seed and timestep.",
    )
    parser.add_argument(
        "--allow-failed-observed-window",
        action="store_true",
        help="Record failed convergence reports but do not block on them.",
    )
    return parser


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return payload


def _repo_relative(path: Path | str | None) -> str | None:
    if path is None:
        return None
    raw_path = Path(path)
    try:
        return raw_path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _safe_report_name(summary_path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", summary_path.stem)
    return f"{stem}.convergence.json"


def _nested_search(payload: dict[str, Any], aliases: tuple[str, ...]) -> Any | None:
    queue: list[dict[str, Any]] = [payload]
    seen = 0
    while queue and seen < 64:
        current = queue.pop(0)
        seen += 1
        for key in aliases:
            if key in current and current[key] not in (None, ""):
                return current[key]
        for key in (
            "variant",
            "metadata",
            "run",
            "simulation",
            "config",
            "nonlinear_config",
        ):
            nested = current.get(key)
            if isinstance(nested, dict):
                queue.append(nested)
    return None


def _variant_from_summary(
    summary: dict[str, Any], summary_path: Path
) -> dict[str, Any]:
    text = summary_path.stem.lower()
    variant: dict[str, Any] = {}
    for axis, aliases in VARIANT_KEY_ALIASES.items():
        value = _nested_search(summary, aliases)
        if value is None:
            regexes = (
                [r"(?:seed|rng)[_-]?([0-9]+)"]
                if axis == "seed"
                else [r"(?:dt|timestep)[_-]?([0-9]+(?:p[0-9]+)?)"]
            )
            for pattern in regexes:
                match = re.search(pattern, text)
                if match:
                    value = match.group(1).replace("p", ".")
                    break
        variant[axis] = value
    return variant


def _summary_case(summary: dict[str, Any], path: Path) -> str:
    return str(summary.get("case") or summary.get("case_name") or path.stem)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    axes = tuple(args.variant_axis or ("seed", "timestep"))
    records: list[dict[str, Any]] = []
    for summary_path in args.summaries:
        summary = _load_json_object(summary_path)
        convergence_config = NonlinearWindowConvergenceConfig(
            tmin=summary.get("tmin"),
            tmax=summary.get("tmax"),
            min_samples=args.min_samples,
            min_blocks=args.min_blocks,
            bootstrap_samples=args.bootstrap_samples,
            max_running_mean_rel_drift=args.max_running_mean_rel_drift,
            max_terminal_mean_rel_delta=args.max_terminal_mean_rel_delta,
            max_sem_rel=args.max_sem_rel,
        )
        report = nonlinear_window_convergence_from_summary(
            summary_path,
            time_column=args.time_column,
            value_column=args.value_column,
            case=_summary_case(summary, summary_path),
            config=convergence_config,
        )
        report_path: Path | None = None
        if args.reports_dir is not None:
            report_path = args.reports_dir / _safe_report_name(summary_path)
            _write_json(report_path, report)
        records.append(
            {
                "case": _summary_case(summary, summary_path),
                "summary_artifact": _repo_relative(summary_path),
                "source_artifact": _repo_relative(
                    report["provenance"]["source_artifact"]
                ),
                "convergence_report_artifact": _repo_relative(report_path),
                "variant": _variant_from_summary(summary, summary_path),
                "report": report,
            }
        )

    manifest = nonlinear_window_ensemble_artifact_manifest(
        records,
        case=args.case,
        config=NonlinearWindowEnsembleManifestConfig(
            min_replicates_per_case=args.min_replicates_per_case,
            required_variant_axes=axes,
            require_observed_windows_ready=not args.allow_failed_observed_window,
        ),
    )
    _write_json(args.out_json, manifest)
    print(json.dumps(manifest["promotion_gate"], indent=2, sort_keys=True))
    return 0 if bool(manifest["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
