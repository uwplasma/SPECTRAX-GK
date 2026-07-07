from __future__ import annotations

import pytest

from tools.check_package_architecture_manifest import validate_architecture_policy


def _manifest(*, allowed: list[str]) -> dict[str, object]:
    return {
        "metadata": {
            "schema_version": 1,
            "title": "test architecture policy",
            "layout_authority": "docs/architecture_refactor_plan.rst",
            "status": "active",
        },
        "root_prefix_policy": {
            "blocked_prefixes": ["runtime_", "nonlinear_"],
            "allowed_root_prefix_modules": allowed,
        },
        "package_policy": {
            "required_domain_packages": ["spectraxgk.operators"],
            "required_docs": ["docs/architecture_refactor_plan.rst"],
        },
    }


def _manifest_with_topology(*, count_path: str, baseline: int, target: int) -> dict[str, object]:
    data = _manifest(allowed=[])
    data["topology_policy"] = {
        "mode": "no_regression_until_target",
        "description": "test topology policy",
        "counts": [
            {
                "name": "test_python_files",
                "path": count_path,
                "pattern": "*.py",
                "recursive": True,
                "baseline": baseline,
                "target": target,
            }
        ],
    }
    return data


def test_validate_architecture_policy_accepts_manifested_root_facade(tmp_path):
    source_root = tmp_path / "spectraxgk"
    (source_root / "operators").mkdir(parents=True)
    (source_root / "operators" / "__init__.py").write_text("", encoding="utf-8")
    (source_root / "nonlinear_removed_helper.py").write_text("", encoding="utf-8")

    summary = validate_architecture_policy(
        _manifest(allowed=["spectraxgk.nonlinear_removed_helper"]),
        source_root=source_root,
        check_paths=False,
    )

    assert summary["n_current_root_prefix_modules"] == 1
    assert summary["n_allowed_root_prefix_modules"] == 1


def test_validate_architecture_policy_rejects_new_root_prefix_module(tmp_path):
    source_root = tmp_path / "spectraxgk"
    (source_root / "operators").mkdir(parents=True)
    (source_root / "operators" / "__init__.py").write_text("", encoding="utf-8")
    (source_root / "runtime_extra.py").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="root-level prefix modules"):
        validate_architecture_policy(
            _manifest(allowed=[]),
            source_root=source_root,
            check_paths=False,
        )


def test_validate_architecture_policy_reports_topology_gap(tmp_path):
    source_root = tmp_path / "spectraxgk"
    count_root = tmp_path / "counted"
    (source_root / "operators").mkdir(parents=True)
    (source_root / "operators" / "__init__.py").write_text("", encoding="utf-8")
    count_root.mkdir()
    for index in range(3):
        (count_root / f"module_{index}.py").write_text("", encoding="utf-8")

    summary = validate_architecture_policy(
        _manifest_with_topology(count_path=str(count_root), baseline=5, target=2),
        source_root=source_root,
        check_paths=False,
    )

    row = summary["topology_counts"][0]
    assert row["count"] == 3
    assert row["baseline"] == 5
    assert row["target"] == 2
    assert row["remaining_to_target"] == 1
    assert row["target_met"] is False
    assert summary["topology_targets_met"] is False


def test_validate_architecture_policy_rejects_topology_regression(tmp_path):
    source_root = tmp_path / "spectraxgk"
    count_root = tmp_path / "counted"
    (source_root / "operators").mkdir(parents=True)
    (source_root / "operators" / "__init__.py").write_text("", encoding="utf-8")
    count_root.mkdir()
    for index in range(3):
        (count_root / f"module_{index}.py").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="above baseline"):
        validate_architecture_policy(
            _manifest_with_topology(count_path=str(count_root), baseline=2, target=1),
            source_root=source_root,
            check_paths=False,
        )


def test_validate_architecture_policy_can_require_topology_targets(tmp_path):
    source_root = tmp_path / "spectraxgk"
    count_root = tmp_path / "counted"
    (source_root / "operators").mkdir(parents=True)
    (source_root / "operators" / "__init__.py").write_text("", encoding="utf-8")
    count_root.mkdir()
    for index in range(2):
        (count_root / f"module_{index}.py").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="target not met"):
        validate_architecture_policy(
            _manifest_with_topology(count_path=str(count_root), baseline=3, target=1),
            source_root=source_root,
            check_paths=False,
            require_topology_targets=True,
        )

    summary = validate_architecture_policy(
        _manifest_with_topology(count_path=str(count_root), baseline=3, target=2),
        source_root=source_root,
        check_paths=False,
        require_topology_targets=True,
    )
    assert summary["topology_targets_met"] is True


def test_validate_architecture_policy_rejects_stale_allowlist(tmp_path):
    source_root = tmp_path / "spectraxgk"
    (source_root / "operators").mkdir(parents=True)
    (source_root / "operators" / "__init__.py").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="allowlist contains modules"):
        validate_architecture_policy(
            _manifest(allowed=["spectraxgk.nonlinear_removed_helper"]),
            source_root=source_root,
            check_paths=False,
        )
