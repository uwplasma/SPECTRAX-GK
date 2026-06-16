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


def test_validate_architecture_policy_accepts_manifested_root_facade(tmp_path):
    source_root = tmp_path / "spectraxgk"
    (source_root / "operators").mkdir(parents=True)
    (source_root / "operators" / "__init__.py").write_text("", encoding="utf-8")
    (source_root / "nonlinear_rhs.py").write_text("", encoding="utf-8")

    summary = validate_architecture_policy(
        _manifest(allowed=["spectraxgk.nonlinear_rhs"]),
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


def test_validate_architecture_policy_rejects_stale_allowlist(tmp_path):
    source_root = tmp_path / "spectraxgk"
    (source_root / "operators").mkdir(parents=True)
    (source_root / "operators" / "__init__.py").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="allowlist contains modules"):
        validate_architecture_policy(
            _manifest(allowed=["spectraxgk.nonlinear_rhs"]),
            source_root=source_root,
            check_paths=False,
        )
