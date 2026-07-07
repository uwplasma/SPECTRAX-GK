from __future__ import annotations

from pathlib import Path

from tools.artifacts.build_technical_release_status import (
    LANES,
    build_technical_release_status,
)


def _write_minimal_tree(root: Path) -> None:
    text_by_path: dict[Path, list[str]] = {}
    for checks in LANES.values():
        for check in checks:
            path = root / check.path
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix.lower() in {".png", ".pdf"}:
                path.write_bytes(b"artifact")
                continue
            text_by_path.setdefault(path, []).append(check.snippet or "present")
    for path, snippets in text_by_path.items():
        path.write_text("\n".join(snippets) + "\n", encoding="utf-8")


def test_technical_release_status_passes_complete_evidence_tree(tmp_path: Path) -> None:
    _write_minimal_tree(tmp_path)

    report = build_technical_release_status(tmp_path)

    assert report["passed"] is True
    assert report["technical_release_completion_percent"] == 100.0
    assert not report["failed_required"]
    assert set(report["lanes"]) == set(LANES)


def test_technical_release_status_reports_missing_required_evidence(
    tmp_path: Path,
) -> None:
    _write_minimal_tree(tmp_path)
    (tmp_path / "docs" / "parallelization.rst").unlink()

    report = build_technical_release_status(tmp_path)

    assert report["passed"] is False
    assert report["technical_release_completion_percent"] < 100.0
    assert any(
        "parallelization_release_surface" in item for item in report["failed_required"]
    )
