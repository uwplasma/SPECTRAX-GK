from __future__ import annotations

from pathlib import Path

import pytest

from tools.check_release_readiness import (
    ReleaseReadinessError,
    check_release_readiness,
)


def _write_release_ready_tree(root: Path) -> None:
    (root / "src" / "spectraxgk").mkdir(parents=True)
    (root / ".github" / "workflows").mkdir(parents=True)
    (root / "docs" / "_static").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        """
[project]
name = "spectraxgk"
version = "1.2.3"

[project.scripts]
spectraxgk = "spectraxgk.cli:main"
spectrax-gk = "spectraxgk.cli:main"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (root / "src" / "spectraxgk" / "_version.py").write_text(
        '__version__ = "1.2.3"\n',
        encoding="utf-8",
    )
    (root / ".github" / "workflows" / "ci.yml").write_text(
        "\n".join(
            [
                "wide-coverage-shards",
                "coverage-wide-shard-manifest.json",
                "--require-shard-data",
                "codecov/codecov-action",
                "tools/check_parallel_scaling_artifacts.py",
                "tools/check_performance_optimization_manifest.py",
                "tools/check_quasilinear_promotion_guardrails.py",
                "tools/check_release_readiness.py",
            ]
        ),
        encoding="utf-8",
    )
    (root / ".github" / "workflows" / "release.yml").write_text(
        "name: Release\n"
        "tools/check_release_version.py\n"
        "gh-action-pypi-publish\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text(
        "Install with pip install spectraxgk, run spectraxgk. License: MIT.\n",
        encoding="utf-8",
    )
    for artifact in (
        "runtime_memory_benchmark.png",
        "validation_gate_index.json",
        "validation_coverage_manifest_summary.json",
        "quasilinear_promotion_guardrails.json",
        "independent_ky_scan_scaling_large.json",
        "quasilinear_uq_ensemble_scaling_large.json",
        "nonlinear_sharding_strong_scaling_large.json",
    ):
        (root / "docs" / "_static" / artifact).write_text("{}", encoding="utf-8")


def test_release_readiness_accepts_ci_release_docs_and_artifact_contracts(
    tmp_path: Path,
) -> None:
    _write_release_ready_tree(tmp_path)

    report = check_release_readiness(tmp_path)

    assert report["passed"] is True
    assert report["project"]["name"] == "spectraxgk"
    assert report["project"]["scripts"] == ["spectrax-gk", "spectraxgk"]
    assert report["version"]["project_version"] == "1.2.3"


def test_release_readiness_rejects_missing_ci_guardrails(tmp_path: Path) -> None:
    _write_release_ready_tree(tmp_path)
    (tmp_path / ".github" / "workflows" / "ci.yml").write_text(
        "wide-coverage-shards\n",
        encoding="utf-8",
    )

    with pytest.raises(ReleaseReadinessError, match="ci.yml missing release checks"):
        check_release_readiness(tmp_path)
