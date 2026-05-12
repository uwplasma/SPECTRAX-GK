#!/usr/bin/env python3
"""Fast local release-readiness checks for CI, packaging, and docs wiring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.check_release_version import validate_release_version

REQUIRED_CI_SNIPPETS = (
    "wide-coverage-shards",
    "coverage-wide-shard-manifest.json",
    "--require-shard-data",
    "codecov/codecov-action",
    "tools/check_parallel_scaling_artifacts.py",
    "tools/check_performance_optimization_manifest.py",
    "tools/check_quasilinear_promotion_guardrails.py",
    "tools/check_release_readiness.py",
)
REQUIRED_README_SNIPPETS = (
    "pip install spectraxgk",
    "spectraxgk",
    "MIT",
)
REQUIRED_STATIC_ARTIFACTS = (
    "docs/_static/runtime_memory_benchmark.png",
    "docs/_static/validation_gate_index.json",
    "docs/_static/validation_coverage_manifest_summary.json",
    "docs/_static/quasilinear_promotion_guardrails.json",
    "docs/_static/independent_ky_scan_scaling_large.json",
    "docs/_static/quasilinear_uq_ensemble_scaling_large.json",
    "docs/_static/nonlinear_sharding_strong_scaling_large.json",
)


class ReleaseReadinessError(RuntimeError):
    """Raised when a release-readiness contract is not satisfied."""


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ReleaseReadinessError(f"missing required file: {path}") from exc


def _missing_snippets(text: str, snippets: tuple[str, ...]) -> list[str]:
    return [snippet for snippet in snippets if snippet not in text]


def _project_metadata(root: Path) -> dict[str, Any]:
    with (root / "pyproject.toml").open("rb") as stream:
        pyproject = tomllib.load(stream)
    project = pyproject.get("project", {})
    scripts = project.get("scripts", {})
    return {
        "name": project.get("name"),
        "version": project.get("version"),
        "scripts": sorted(scripts),
    }


def check_release_readiness(root: Path = REPO_ROOT) -> dict[str, Any]:
    """Return a JSON-ready release-readiness report or raise on failure."""

    root = root.resolve()
    failures: list[str] = []

    version_report = validate_release_version(root=root)
    project = _project_metadata(root)
    if project["name"] != "spectraxgk":
        failures.append("pyproject project.name must be 'spectraxgk'")
    expected_scripts = {"spectraxgk", "spectrax-gk"}
    missing_scripts = sorted(expected_scripts - set(project["scripts"]))
    if missing_scripts:
        failures.append(f"missing executable entry points: {missing_scripts}")

    ci_text = _read(root / ".github" / "workflows" / "ci.yml")
    missing_ci = _missing_snippets(ci_text, REQUIRED_CI_SNIPPETS)
    if missing_ci:
        failures.append(f"ci.yml missing release checks: {missing_ci}")

    release_text = _read(root / ".github" / "workflows" / "release.yml")
    missing_release = _missing_snippets(
        release_text,
        ("name: Release", "gh-action-pypi-publish", "tools/check_release_version.py"),
    )
    if missing_release:
        failures.append(f"release.yml missing publish/version checks: {missing_release}")

    readme_text = _read(root / "README.md")
    missing_readme = _missing_snippets(readme_text, REQUIRED_README_SNIPPETS)
    if missing_readme:
        failures.append(f"README missing release-user snippets: {missing_readme}")

    missing_artifacts = [
        path for path in REQUIRED_STATIC_ARTIFACTS if not (root / path).exists()
    ]
    if missing_artifacts:
        failures.append(f"missing required docs/static release artifacts: {missing_artifacts}")

    report = {
        "kind": "spectraxgk_release_readiness",
        "root": str(root),
        "project": project,
        "version": version_report,
        "required_ci_snippets": list(REQUIRED_CI_SNIPPETS),
        "required_readme_snippets": list(REQUIRED_README_SNIPPETS),
        "required_static_artifacts": list(REQUIRED_STATIC_ARTIFACTS),
        "failures": failures,
        "passed": not failures,
    }
    if failures:
        raise ReleaseReadinessError("; ".join(failures))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--out-json", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = check_release_readiness(root=args.root)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out_json is None:
        print(payload, end="")
    else:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(payload, encoding="utf-8")
        print(f"Wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
