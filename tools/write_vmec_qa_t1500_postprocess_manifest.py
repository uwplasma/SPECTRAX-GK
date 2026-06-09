#!/usr/bin/env python3
"""Write reproducible postprocess commands for strict QA ``t=1500`` audits."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = "python"
TMIN = 1100.0
TMAX = 1500.0


@dataclass(frozen=True)
class QAAuditCase:
    key: str
    run_dir: str
    ensemble_stem: str
    title: str

    @property
    def output_gate_json(self) -> Path:
        return Path("docs/_static") / f"vmec_qa_t1500_{self.key}_output_gate.json"

    @property
    def ensemble_json(self) -> Path:
        return (
            Path("docs/_static/vmec_qa_t1500_replicates")
            / f"{self.ensemble_stem}_t1500_ensemble_gate.json"
        )

    @property
    def ensemble_png(self) -> Path:
        return (
            Path("docs/_static/vmec_qa_t1500_replicates")
            / f"{self.ensemble_stem}_t1500_ensemble_gate.png"
        )

    @property
    def readiness_json_name(self) -> str:
        return f"{self.ensemble_stem}_t1500_readiness.json"

    @property
    def ensemble_json_name(self) -> str:
        return self.ensemble_json.name

    @property
    def ensemble_png_name(self) -> str:
        return self.ensemble_png.name


CASES: dict[str, QAAuditCase] = {
    "baseline": QAAuditCase(
        key="baseline",
        run_dir="vmec_qa_full_sweep_qa_baseline_scipy",
        ensemble_stem="qa_baseline_scipy",
        title="Strict QA baseline nonlinear audit, t=[1100,1500]",
    ),
    "growth": QAAuditCase(
        key="growth",
        run_dir="vmec_qa_full_sweep_growth_from_strict_baseline",
        ensemble_stem="growth_from_strict_baseline",
        title="QA growth-objective nonlinear audit, t=[1100,1500]",
    ),
    "quasilinear": QAAuditCase(
        key="quasilinear",
        run_dir="vmec_qa_full_sweep_quasilinear_from_strict_baseline",
        ensemble_stem="quasilinear_from_strict_baseline",
        title="QA quasilinear-objective nonlinear audit, t=[1100,1500]",
    ),
    "nonlinear_window": QAAuditCase(
        key="nonlinear_window",
        run_dir="vmec_qa_full_sweep_nonlinear_window_from_strict_baseline",
        ensemble_stem="nonlinear_window_from_strict_baseline",
        title="QA nonlinear-window-objective nonlinear audit, t=[1100,1500]",
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        default="tools_out/vmec_qa_full_sweep_nonlinear_audits",
        help="Root containing one subdirectory per strict QA audit case.",
    )
    parser.add_argument(
        "--netcdf-root",
        default=(
            "office:/home/rjorge/spectrax_strict_qa_t1500_20260609/"
            "SPECTRAX-GK/tools_out/vmec_qa_full_sweep_nonlinear_audits"
        ),
        help="Authoritative NetCDF root used in compact artifact provenance.",
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=tuple(CASES),
        help="Case to include. Defaults to all cases.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("docs/_static/vmec_qa_t1500_postprocess_manifest.json"),
    )
    parser.add_argument(
        "--min-relative-reduction",
        type=float,
        default=0.04,
        help="Promotion threshold used for matched baseline-vs-candidate comparisons.",
    )
    return parser


def _repo_relative(path: Path | str) -> str:
    raw = Path(path)
    try:
        return raw.resolve().relative_to(ROOT.resolve()).as_posix()
    except (OSError, ValueError):
        return str(path)


def _outputs(case: QAAuditCase, *, run_root: str) -> list[str]:
    root = Path(run_root) / case.run_dir
    return [
        _repo_relative(root / f"{case.run_dir}_nonlinear_t1500_n64_seed32.out.nc"),
        _repo_relative(root / f"{case.run_dir}_nonlinear_t1500_n64_seed33.out.nc"),
        _repo_relative(root / f"{case.run_dir}_nonlinear_t1500_n64_dt0p04.out.nc"),
    ]


def _quote_title(raw: str) -> str:
    return '"' + raw.replace('"', '\\"') + '"'


def _case_commands(case: QAAuditCase, *, run_root: str, netcdf_root: str) -> dict[str, Any]:
    outputs = _outputs(case, run_root=run_root)
    output_gate = _repo_relative(case.output_gate_json)
    ensemble_dir = "docs/_static/vmec_qa_t1500_replicates"
    ensemble = _repo_relative(case.ensemble_json)
    check_command = (
        f"{PYTHON} tools/check_nonlinear_runtime_outputs.py "
        + " ".join(outputs)
        + " --min-samples 200 --tmin 1100 --tmax 1500"
        + " --min-window-samples 80 --min-abs-window-mean 0.0001"
        + f" --json-out {output_gate}"
    )
    ensemble_command = (
        'PYTHONPATH="$PWD/src" '
        f"{PYTHON} tools/build_external_vmec_replicate_ensemble.py "
        + " ".join(outputs)
        + f" --out-dir {ensemble_dir}"
        + f" --case {case.run_dir}_t1500_replicated_nonlinear_window"
        + " --tmin 1100 --tmax 1500"
        + f" --artifact-prefix {ensemble_dir}"
        + f" --readiness-json {case.readiness_json_name}"
        + f" --ensemble-json {case.ensemble_json_name}"
        + f" --out-png {case.ensemble_png_name}"
        + f" --figure-title {_quote_title(case.title)}"
    )
    compact_command = (
        f"{PYTHON} tools/compact_replicate_ensemble_bundle.py"
        f" --ensemble-json {ensemble}"
        f" --output-gate-json {output_gate}"
        f" --netcdf-root {netcdf_root.rstrip('/')}"
    )
    return {
        "case": case.key,
        "run_dir": case.run_dir,
        "outputs": outputs,
        "output_gate_json": output_gate,
        "ensemble_json": ensemble,
        "ensemble_png": _repo_relative(case.ensemble_png),
        "check_outputs_command": check_command,
        "build_ensemble_command": ensemble_command,
        "compact_bundle_command": compact_command,
        "commands": [check_command, ensemble_command, compact_command],
    }


def _comparison_command(
    *,
    baseline: QAAuditCase,
    candidate: QAAuditCase,
    min_relative_reduction: float,
) -> dict[str, str]:
    stem = f"vmec_qa_t1500_baseline_to_{candidate.key}_comparison"
    out_json = Path("docs/_static") / f"{stem}.json"
    out_figure = Path("docs/_static") / f"{stem}.png"
    command = (
        f"{PYTHON} tools/build_matched_nonlinear_transport_comparison.py"
        f" --baseline-ensemble {_repo_relative(baseline.ensemble_json)}"
        f" --candidate-ensemble {_repo_relative(candidate.ensemble_json)}"
        f" --case {stem}"
        f" --min-relative-reduction {float(min_relative_reduction):.12g}"
        f" --out-json {_repo_relative(out_json)}"
        f" --out-figure {_repo_relative(out_figure)}"
    )
    return {
        "candidate": candidate.key,
        "out_json": _repo_relative(out_json),
        "out_figure": _repo_relative(out_figure),
        "command": command,
    }


def build_manifest(
    *,
    run_root: str,
    netcdf_root: str,
    cases: tuple[str, ...],
    min_relative_reduction: float,
) -> dict[str, Any]:
    selected = tuple(CASES[key] for key in cases)
    baseline = CASES["baseline"]
    return {
        "kind": "vmec_qa_t1500_postprocess_manifest",
        "claim_level": "postprocess_command_manifest_not_simulation_claim",
        "run_root": run_root,
        "netcdf_root": netcdf_root,
        "window": {"tmin": TMIN, "tmax": TMAX},
        "min_relative_reduction": float(min_relative_reduction),
        "case_commands": [
            _case_commands(case, run_root=run_root, netcdf_root=netcdf_root)
            for case in selected
        ],
        "comparison_commands": [
            _comparison_command(
                baseline=baseline,
                candidate=case,
                min_relative_reduction=float(min_relative_reduction),
            )
            for case in selected
            if case.key != "baseline"
        ],
        "next_actions": [
            "run check_outputs_command, build_ensemble_command, and compact_bundle_command for each completed case",
            "run comparison commands only after the baseline and candidate ensemble JSON files exist and pass",
            "do not promote candidates unless the matched comparison passes the configured relative-reduction gate",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cases = tuple(args.case or CASES.keys())
    manifest = build_manifest(
        run_root=str(args.run_root),
        netcdf_root=str(args.netcdf_root),
        cases=cases,
        min_relative_reduction=float(args.min_relative_reduction),
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"cases": cases, "out_json": _repo_relative(args.out_json)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
