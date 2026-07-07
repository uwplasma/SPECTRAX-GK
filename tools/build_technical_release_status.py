#!/usr/bin/env python3
"""Build a machine-readable technical release completion status report."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "docs" / "_static" / "technical_release_status.json"


@dataclass(frozen=True)
class EvidenceCheck:
    """One checkable release-readiness item."""

    name: str
    path: str
    snippet: str | None = None
    required: bool = True


def _read(root: Path, path: str) -> str | None:
    target = root / path
    if not target.exists():
        return None
    if target.suffix.lower() in {".png", ".pdf", ".nc"}:
        return "<binary>"
    return target.read_text(encoding="utf-8", errors="replace")


def _evaluate_check(root: Path, check: EvidenceCheck) -> dict[str, Any]:
    text = _read(root, check.path)
    exists = text is not None
    snippet_ok = True if check.snippet is None else bool(text and check.snippet in text)
    passed = bool(exists and snippet_ok)
    return {
        "name": check.name,
        "path": check.path,
        "required": check.required,
        "exists": exists,
        "snippet": check.snippet,
        "passed": passed,
    }


LANES: dict[str, tuple[EvidenceCheck, ...]] = {
    "ci_coverage_release": (
        EvidenceCheck("wide coverage matrix", ".github/workflows/ci.yml", "wide-coverage-shards"),
        EvidenceCheck("wide shard manifest", ".github/workflows/ci.yml", "coverage-wide-shard-manifest.json"),
        EvidenceCheck("95 percent gate", ".github/workflows/ci.yml", "--fail-under 95"),
        EvidenceCheck("measured manifest coverage audit", ".github/workflows/ci.yml", "--coverage-xml coverage-wide.xml"),
        EvidenceCheck("codecov upload", ".github/workflows/ci.yml", "codecov/codecov-action"),
        EvidenceCheck("release readiness check", ".github/workflows/ci.yml", "tools/check_release_readiness.py"),
    ),
    "parallelization_release_surface": (
        EvidenceCheck("parallelization policy docs", "docs/parallelization.rst", "Production-ready parallelism"),
        EvidenceCheck("runtime parallel input docs", "docs/inputs.rst", "strategy = \"batch\""),
        EvidenceCheck("independent ky scaling artifact", "docs/_static/independent_ky_scan_scaling_large.json", "not a nonlinear domain-decomposition"),
        EvidenceCheck("quasilinear UQ scaling artifact", "docs/_static/quasilinear_uq_ensemble_scaling_large.json", "not a promoted absolute nonlinear heat-flux predictor"),
        EvidenceCheck("parallelization completion status", "docs/_static/parallelization_completion_status.json", "production_completion_percent"),
        EvidenceCheck("nonlinear sharding scoped diagnostic", "docs/_static/nonlinear_sharding_strong_scaling_large.json", "not a production speedup claim"),
        EvidenceCheck("nonlinear domain identity gate", "docs/_static/nonlinear_domain_parallel_identity_gate.json", "no production routing or speedup claim"),
        EvidenceCheck("nonlinear spectral communication gate", "docs/_static/nonlinear_spectral_communication_identity_gate.json", "no production routing or speedup claim"),
        EvidenceCheck("parallel artifact checker", "tools/check_parallel_scaling_artifacts.py", "FAMILIES"),
    ),
    "refactor_modularity": (
        EvidenceCheck("architecture refactor plan", "docs/architecture_refactor_plan.rst", "authoritative refactor plan"),
        EvidenceCheck("package architecture manifest", "tools/package_architecture_manifest.toml", "allowed_root_prefix_modules"),
        EvidenceCheck("package architecture checker", "tools/release/check_package_architecture_manifest.py", "root-level prefix modules"),
        EvidenceCheck("operators package facade", "src/spectraxgk/operators/__init__.py", "hermite_streaming"),
        EvidenceCheck("linear operator package", "src/spectraxgk/operators/linear/__init__.py", "build_linear_cache"),
        EvidenceCheck("linear solver package", "src/spectraxgk/solvers/linear/__init__.py", "KrylovConfig"),
        EvidenceCheck("nonlinear operator package", "src/spectraxgk/operators/nonlinear/__init__.py", "nonlinear_rhs_cached_impl"),
        EvidenceCheck("nonlinear solver package", "src/spectraxgk/solvers/nonlinear/__init__.py", "solve_imex_step"),
        EvidenceCheck("runtime orchestration module", "src/spectraxgk/workflows/runtime/orchestration.py"),
        EvidenceCheck("runtime policy module", "src/spectraxgk/workflows/runtime/policies.py", "RuntimeIndependentParallelPlan"),
        EvidenceCheck("linear cache module", "src/spectraxgk/operators/linear/cache.py"),
        EvidenceCheck("linear moments module", "src/spectraxgk/operators/linear/moments.py"),
        EvidenceCheck("linear params module", "src/spectraxgk/operators/linear/params.py"),
        EvidenceCheck("linear parallel module", "src/spectraxgk/solvers/linear/parallel.py"),
        EvidenceCheck("nonlinear helper module", "src/spectraxgk/operators/nonlinear/policies.py"),
        EvidenceCheck("benchmark scan module", "src/spectraxgk/validation/benchmarks/scan.py"),
        EvidenceCheck("diagnostics channel module", "src/spectraxgk/diagnostics/channels.py"),
        EvidenceCheck("coverage manifest", "tools/validation_coverage_manifest.toml", "spectraxgk.workflows.runtime.orchestration"),
    ),
    "docs_release_hygiene": (
        EvidenceCheck("readme install", "README.md", "pip install spectraxgk"),
        EvidenceCheck("readme executable", "README.md", "spectraxgk"),
        EvidenceCheck("MIT license in README", "README.md", "MIT"),
        EvidenceCheck("release scope ledger", "docs/release_scope.rst", "Claim scope"),
        EvidenceCheck("roadmap", "docs/roadmap.rst", "Release-ready"),
        EvidenceCheck("examples docs", "docs/examples.rst", "parallelization"),
        EvidenceCheck("release workflow", ".github/workflows/release.yml", "gh-action-pypi-publish"),
    ),
    "performance_artifacts": (
        EvidenceCheck("runtime memory figure", "docs/_static/runtime_memory_benchmark.png"),
        EvidenceCheck("runtime memory manifest", "tools/runtime_memory_manifest.toml"),
        EvidenceCheck("performance manifest", "tools/performance_optimization_manifest.toml", "parallel_scaling"),
        EvidenceCheck("nonlinear RHS profile", "docs/_static/nonlinear_rhs_profile.json"),
        EvidenceCheck("full nonlinear RHS trace", "docs/_static/full_nonlinear_rhs_trace_summary.json"),
        EvidenceCheck("performance docs", "docs/performance.rst", "No speedup claim"),
    ),
    "scientific_gate_guardrails": (
        EvidenceCheck("validation gate index", "docs/_static/validation_gate_index.json"),
        EvidenceCheck("quasilinear guardrails", "docs/_static/quasilinear_promotion_guardrails.json"),
        EvidenceCheck(
            "VMEC/Boozer differentiability claim guard",
            "docs/_static/vmec_boozer_differentiability_claim_guard.json",
            "not_full_nonlinear_transport_optimization",
        ),
        EvidenceCheck(
            "finite-beta VMEC/Boozer frequency gradient gate",
            "docs/_static/vmec_boozer_shaped_pressure_solver_frequency_gradient_gate.json",
            "shaped_tokamak_pressure",
        ),
        EvidenceCheck(
            "finite-beta VMEC/Boozer quasilinear gradient gate",
            "docs/_static/vmec_boozer_shaped_pressure_quasilinear_gradient_gate.json",
            "shaped_tokamak_pressure",
        ),
        EvidenceCheck(
            "finite-beta VMEC/Boozer reduced nonlinear-window gradient gate",
            "docs/_static/vmec_boozer_shaped_pressure_nonlinear_window_gradient_gate.json",
            "shaped_tokamak_pressure",
        ),
        EvidenceCheck("manuscript readiness", "docs/_static/manuscript_readiness_status.json"),
        EvidenceCheck("open lane status", "docs/_static/open_research_lane_status.json"),
        EvidenceCheck("stellarator optimization docs", "docs/stellarator_optimization.rst", "finite-difference"),
        EvidenceCheck("quasilinear docs", "docs/quasilinear.rst", "absolute-flux"),
    ),
}


def build_technical_release_status(root: Path = REPO_ROOT) -> dict[str, Any]:
    """Return technical release completion status from tracked evidence."""

    root = root.resolve()
    lanes: dict[str, Any] = {}
    scores: list[float] = []
    failed_required: list[str] = []
    for lane, checks in LANES.items():
        evaluated = [_evaluate_check(root, check) for check in checks]
        required = [item for item in evaluated if item["required"]]
        passed_required = [item for item in required if item["passed"]]
        score = 100.0 * len(passed_required) / max(len(required), 1)
        scores.append(score)
        lane_failed = [item["name"] for item in required if not item["passed"]]
        failed_required.extend(f"{lane}: {name}" for name in lane_failed)
        lanes[lane] = {
            "completion_percent": score,
            "passed_required": len(passed_required),
            "required": len(required),
            "failed_required": lane_failed,
            "checks": evaluated,
        }
    overall = sum(scores) / max(len(scores), 1)
    return {
        "kind": "spectraxgk_technical_release_status",
        "root": str(root),
        "technical_release_completion_percent": overall,
        "target_percent": 98.0,
        "passed": overall >= 98.0 and not failed_required,
        "failed_required": failed_required,
        "lanes": lanes,
        "scope": (
            "Technical release readiness for CI, refactor, docs/release hygiene, "
            "parallelization artifacts, performance artifacts, and guardrails. "
            "Deferred manuscript physics lanes remain scoped separately."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--fail-under", type=float, default=98.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_technical_release_status(args.root)
    report["target_percent"] = float(args.fail_under)
    report["passed"] = (
        float(report["technical_release_completion_percent"]) >= float(args.fail_under)
        and not report["failed_required"]
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out_json is None:
        print(payload, end="")
    else:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(payload, encoding="utf-8")
        print(f"Wrote {args.out_json}")
    if not report["passed"]:
        print(payload, end="")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
