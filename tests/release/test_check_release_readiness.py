from __future__ import annotations

from pathlib import Path

import pytest

from tools.release.check_release_readiness import TECHNICAL_COMPLETION_TARGET
from tools.release.check_release_readiness import (
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
                "--coverage-xml coverage-wide.xml",
                "--enforce-package-coverage",
                "codecov/codecov-action",
                "tools/release/check_parallel_scaling_artifacts.py",
                "tools/release/check_package_architecture_manifest.py",
                "tools/release/check_performance_optimization_manifest.py",
                "tools/release/check_quasilinear_promotion_guardrails.py",
                "tools/release/check_vmec_boozer_differentiability_claim.py",
                "tools/artifacts/build_parallelization_completion_status.py",
                "tools/artifacts/build_technical_release_status.py",
                "tools/release/check_release_readiness.py",
            ]
        ),
        encoding="utf-8",
    )
    (root / "codecov.yml").write_text(
        """
codecov:
  notify:
    after_n_builds: 2
    wait_for_ci: true

coverage:
  status:
    project:
      default:
        target: 95%
        threshold: 0.5%
        flags:
          - wide-package
    patch:
      default:
        informational: true
""".lstrip(),
        encoding="utf-8",
    )
    (root / ".github" / "workflows" / "release.yml").write_text(
        "name: Release\n"
        "tools/release/check_release_version.py\n"
        "tools/release/check_repository_size_manifest.py\n"
        "tools/release/check_release_artifact_manifest.py\n"
        "tools/release/check_package_architecture_manifest.py\n"
        "tools/release/check_performance_optimization_manifest.py\n"
        "tools/release/check_parallel_scaling_artifacts.py\n"
        "tools/release/check_quasilinear_promotion_guardrails.py\n"
        "tools/release/check_vmec_boozer_differentiability_claim.py\n"
        "tools/artifacts/build_parallelization_completion_status.py\n"
        "tools/artifacts/build_technical_release_status.py\n"
        "tools/release/check_release_readiness.py\n"
        "gh-action-pypi-publish\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text(
        "Install with pip install spectraxgk, run spectraxgk. License: MIT.\n",
        encoding="utf-8",
    )
    for artifact in (
        "runtime_memory_benchmark.png",
        "runtime_memory_summary_ship_refresh.json",
        "runtime_memory_results_ship_refresh.csv",
        "validation_gate_index.json",
        "validation_coverage_manifest_summary.json",
        "quasilinear_promotion_guardrails.json",
        "vmec_boozer_differentiability_claim_guard.json",
        "vmec_boozer_shaped_pressure_solver_frequency_gradient_gate.json",
        "vmec_boozer_shaped_pressure_quasilinear_gradient_gate.json",
        "vmec_boozer_shaped_pressure_nonlinear_window_gradient_gate.json",
        "technical_release_status.json",
        "independent_ky_scan_scaling_large.json",
        "quasilinear_uq_ensemble_scaling_large.json",
        "parallelization_completion_status.json",
        "nonlinear_sharding_strong_scaling_large.json",
        "nonlinear_domain_parallel_identity_gate.json",
        "nonlinear_spectral_communication_identity_gate.json",
        "vmec_jax_qa_transport_optimization_status.json",
        "vmec_boundary_transport_landscape_admission.json",
        "vmec_boundary_transport_prelaunch_gate.json",
        "nonlinear_campaign_admission_report.json",
        "strict_qa_top12_edge_prelaunch_gate.json",
    ):
        (root / "docs" / "_static" / artifact).write_text("{}", encoding="utf-8")
    (root / "docs" / "_static" / "technical_release_status.json").write_text(
        """
{
  "failed_required": [],
  "kind": "spectraxgk_technical_release_status",
  "lanes": {},
  "passed": true,
  "target_percent": 98.0,
  "technical_release_completion_percent": 100.0
}
""".lstrip(),
        encoding="utf-8",
    )
    (
        root / "docs" / "_static" / "vmec_jax_qa_transport_optimization_status.json"
    ).write_text(
        """
{
  "kind": "vmec_jax_qa_transport_optimization_status",
  "prelaunch_gates": [
    {
      "label": "replicated landscape admission",
      "path": "docs/_static/vmec_boundary_transport_landscape_admission.json",
      "passed": true,
      "expected_raw_passed": true,
      "raw_passed": true,
      "sample_count": 12.0,
      "blockers": []
    },
    {
      "label": "selected reduced prelaunch",
      "path": "docs/_static/vmec_boundary_transport_prelaunch_gate.json",
      "passed": true,
      "expected_raw_passed": true,
      "raw_passed": true,
      "sample_count": 18.0,
      "blockers": []
    },
    {
      "label": "weak reduced-margin reference",
      "path": "docs/_static/strict_qa_top12_edge_prelaunch_gate.json",
      "passed": true,
      "expected_raw_passed": false,
      "raw_passed": false,
      "sample_count": 18.0,
      "blockers": ["insufficient_reduced_margin_for_nonlinear_audit"]
    },
    {
      "label": "next nonlinear campaign admission",
      "path": "docs/_static/nonlinear_campaign_admission_report.json",
      "passed": true,
      "expected_raw_passed": true,
      "raw_passed": true,
      "sample_count": 18.0,
      "blockers": []
    }
  ],
  "summary": {
    "qa_baseline_gate_passed": true,
    "quasilinear_model_selection_passed": false,
    "simple_quasilinear_absolute_flux_promoted": false,
    "long_window_nonlinear_audit_passed": true,
    "nonlinear_prelaunch_policy_ready": true,
    "nonlinear_campaign_admission_ready": true,
    "negative_reference_blocks_weak_margin": true,
    "claim_evidence_level": "scoped_matched_replicated_nonlinear_audit",
    "claim_promotion_blockers": [
      "quasilinear_model_selection_not_promoted",
      "simple_quasilinear_absolute_flux_not_promoted"
    ]
  }
}
""".lstrip(),
        encoding="utf-8",
    )
    (root / "docs" / "_static" / "manuscript_readiness_status.json").write_text(
        """
{
  "claim_scope": "release_scope",
  "kind": "manuscript_readiness_status",
  "lanes": [
    {
      "claim_level": "release_claim",
      "lane": "CI/release hygiene and status automation",
      "status": "closed"
    },
    {
      "claim_level": "deferred_out_of_release_scope",
      "lane": "Future physics extension",
      "status": "deferred"
    }
  ],
  "summary": {
    "active_fraction_closed": 1.0,
    "n_active": 1,
    "n_blocked": 0,
    "n_closed": 1,
    "n_deferred": 1,
    "n_lanes": 2,
    "n_open": 0,
    "n_partial": 0
  }
}
""".lstrip(),
        encoding="utf-8",
    )
    (root / "docs" / "_static" / "open_research_lane_status.json").write_text(
        """
{
  "claim_scope": "post_release_tracking",
  "kind": "open_research_lane_status",
  "lanes": [
    {
      "claim_level": "open_research_not_release_claim",
      "lane": "Open research lane",
      "status": "open"
    }
  ],
  "summary": {"n_open": 1}
}
""".lstrip(),
        encoding="utf-8",
    )
    (root / "docs" / "_static" / "w7x_tem_extension_status.json").write_text(
        """
{
  "claim_scope": "extension_tracking",
  "kind": "w7x_tem_extension_status",
  "lanes": [
    {
      "claim_level": "partial_extension_not_release_claim",
      "lane": "W7-X TEM extension",
      "status": "partial"
    }
  ],
  "summary": {"n_partial": 1}
}
""".lstrip(),
        encoding="utf-8",
    )


def test_release_readiness_accepts_ci_release_docs_and_artifact_contracts(
    tmp_path: Path,
) -> None:
    _write_release_ready_tree(tmp_path)

    report = check_release_readiness(tmp_path)

    assert report["passed"] is True
    assert report["project"]["name"] == "spectraxgk"
    assert report["project"]["scripts"] == ["spectrax-gk", "spectraxgk"]
    assert report["version"]["project_version"] == "1.2.3"
    assert (
        report["release_target"]["technical_completion_fraction"]
        == TECHNICAL_COMPLETION_TARGET
    )
    assert report["lane_status"]["passed"] is True
    assert report["lane_status"]["active_fraction_closed"] == 1.0
    assert report["lane_status"]["release_scoped_open_or_blocked"] == 0
    assert report["technical_status"]["passed"] is True
    assert report["technical_status"]["completion_percent"] >= 98.0
    assert report["optimization_status"]["passed"] is True
    assert (
        report["optimization_status"]["summary"]["nonlinear_prelaunch_policy_ready"]
        is True
    )
    assert report["lane_status"]["status_artifacts"][
        "docs/_static/manuscript_readiness_status.json"
    ]["status_counts"] == {"closed": 1, "deferred": 1}


def test_release_readiness_rejects_missing_ci_guardrails(tmp_path: Path) -> None:
    _write_release_ready_tree(tmp_path)
    (tmp_path / ".github" / "workflows" / "ci.yml").write_text(
        "wide-coverage-shards\n",
        encoding="utf-8",
    )

    with pytest.raises(ReleaseReadinessError, match="ci.yml missing release checks"):
        check_release_readiness(tmp_path)


def test_release_readiness_rejects_missing_codecov_status_policy(
    tmp_path: Path,
) -> None:
    _write_release_ready_tree(tmp_path)
    (tmp_path / "codecov.yml").write_text(
        """
coverage:
  status:
    project:
      default:
        target: 95%
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(
        ReleaseReadinessError,
        match="codecov.yml missing wide-coverage status policy",
    ):
        check_release_readiness(tmp_path)


def test_release_readiness_rejects_missing_release_guardrails(tmp_path: Path) -> None:
    _write_release_ready_tree(tmp_path)
    (tmp_path / ".github" / "workflows" / "release.yml").write_text(
        "name: Release\n"
        "tools/release/check_release_version.py\n"
        "gh-action-pypi-publish\n",
        encoding="utf-8",
    )

    with pytest.raises(ReleaseReadinessError, match="release.yml missing publish/version checks"):
        check_release_readiness(tmp_path)


def test_release_readiness_rejects_below_target_release_completion(
    tmp_path: Path,
) -> None:
    _write_release_ready_tree(tmp_path)
    (tmp_path / "docs" / "_static" / "manuscript_readiness_status.json").write_text(
        """
{
  "claim_scope": "release_scope",
  "kind": "manuscript_readiness_status",
  "lanes": [
    {
      "claim_level": "release_claim",
      "lane": "CI/release hygiene and status automation",
      "status": "partial"
    }
  ],
  "summary": {
    "active_fraction_closed": 0.97,
    "n_active": 1,
    "n_blocked": 0,
    "n_closed": 0,
    "n_deferred": 0,
    "n_lanes": 1,
    "n_open": 0,
    "n_partial": 1
  }
}
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(
        ReleaseReadinessError,
        match="release-scoped technical completion below target",
    ):
        check_release_readiness(tmp_path)


def test_release_readiness_rejects_failed_technical_status(tmp_path: Path) -> None:
    _write_release_ready_tree(tmp_path)
    (tmp_path / "docs" / "_static" / "technical_release_status.json").write_text(
        """
{
  "failed_required": ["docs_release_hygiene: roadmap"],
  "kind": "spectraxgk_technical_release_status",
  "lanes": {},
  "passed": false,
  "target_percent": 98.0,
  "technical_release_completion_percent": 92.0
}
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(
        ReleaseReadinessError,
        match="technical release status below target",
    ):
        check_release_readiness(tmp_path)


def test_release_readiness_rejects_missing_optimization_prelaunch_policy(
    tmp_path: Path,
) -> None:
    _write_release_ready_tree(tmp_path)
    (
        tmp_path / "docs" / "_static" / "vmec_jax_qa_transport_optimization_status.json"
    ).write_text(
        """
{
  "kind": "vmec_jax_qa_transport_optimization_status",
  "prelaunch_gates": [],
  "summary": {
    "qa_baseline_gate_passed": true,
    "quasilinear_model_selection_passed": false,
    "simple_quasilinear_absolute_flux_promoted": false,
    "long_window_nonlinear_audit_passed": true,
    "nonlinear_prelaunch_policy_ready": false,
    "negative_reference_blocks_weak_margin": false
  }
}
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(
        ReleaseReadinessError,
        match="optimization status prelaunch/claim-boundary flags failed",
    ):
        check_release_readiness(tmp_path)


def test_release_readiness_requires_explicit_optimization_status_booleans(
    tmp_path: Path,
) -> None:
    _write_release_ready_tree(tmp_path)
    (
        tmp_path / "docs" / "_static" / "vmec_jax_qa_transport_optimization_status.json"
    ).write_text(
        """
{
  "kind": "vmec_jax_qa_transport_optimization_status",
  "prelaunch_gates": [
    {"label": "landscape", "passed": true},
    {"label": "positive", "passed": true},
    {"label": "negative", "passed": true}
  ],
  "summary": {
    "qa_baseline_gate_passed": true,
    "quasilinear_model_selection_passed": false,
    "long_window_nonlinear_audit_passed": true,
    "nonlinear_prelaunch_policy_ready": true,
    "negative_reference_blocks_weak_margin": true
  }
}
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(
        ReleaseReadinessError,
        match="optimization status prelaunch/claim-boundary flags failed",
    ):
        check_release_readiness(tmp_path)


def test_release_readiness_rejects_stale_prelaunch_gate_rows(
    tmp_path: Path,
) -> None:
    _write_release_ready_tree(tmp_path)
    (
        tmp_path / "docs" / "_static" / "vmec_jax_qa_transport_optimization_status.json"
    ).write_text(
        """
{
  "kind": "vmec_jax_qa_transport_optimization_status",
  "prelaunch_gates": [
    {
      "label": "replicated landscape admission",
      "path": "docs/_static/vmec_boundary_transport_landscape_admission.json",
      "passed": true,
      "expected_raw_passed": true,
      "raw_passed": true,
      "sample_count": 12.0,
      "blockers": []
    },
    {
      "label": "selected reduced prelaunch",
      "path": "docs/_static/vmec_boundary_transport_prelaunch_gate.json",
      "passed": true,
      "expected_raw_passed": true,
      "raw_passed": true,
      "sample_count": 1.0,
      "blockers": []
    },
    {
      "label": "weak reduced-margin reference",
      "path": "docs/_static/strict_qa_top12_edge_prelaunch_gate.json",
      "passed": true,
      "expected_raw_passed": true,
      "raw_passed": true,
      "sample_count": 18.0,
      "blockers": []
    },
    {
      "label": "next nonlinear campaign admission",
      "path": "docs/_static/nonlinear_campaign_admission_report.json",
      "passed": true,
      "expected_raw_passed": true,
      "raw_passed": true,
      "sample_count": 18.0,
      "blockers": []
    }
  ],
  "summary": {
    "qa_baseline_gate_passed": true,
    "quasilinear_model_selection_passed": false,
    "simple_quasilinear_absolute_flux_promoted": false,
    "long_window_nonlinear_audit_passed": true,
    "nonlinear_prelaunch_policy_ready": true,
    "nonlinear_campaign_admission_ready": true,
    "negative_reference_blocks_weak_margin": true,
    "claim_evidence_level": "scoped_matched_replicated_nonlinear_audit",
    "claim_promotion_blockers": [
      "quasilinear_model_selection_not_promoted",
      "simple_quasilinear_absolute_flux_not_promoted"
    ]
  }
}
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(
        ReleaseReadinessError,
        match="optimization status prelaunch/claim-boundary flags failed",
    ):
        check_release_readiness(tmp_path)
