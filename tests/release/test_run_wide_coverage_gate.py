from __future__ import annotations

from pathlib import Path

import pytest

from tools.release.run_wide_coverage_gate import (
    build_coverage_shard_report,
    _resolve_test_dir,
    discover_test_files,
    validate_coverage_shard_report,
    split_shards,
    write_json,
)


def test_split_shards_is_round_robin_and_complete() -> None:
    files = [Path(f"tests/test_{idx}.py") for idx in range(7)]
    shards = split_shards(files, 3)

    assert shards == [files[0::3], files[1::3], files[2::3]]
    assert sorted(path for shard in shards for path in shard) == files


def test_split_shards_isolates_known_high_cost_tests() -> None:
    expensive = [
        Path("tests/unit/solvers/test_diffrax_integrators_core.py"),
        Path("tests/integration/runtime/test_runtime_runner.py"),
    ]
    files = expensive + [Path(f"tests/test_light_{idx}.py") for idx in range(12)]
    shards = split_shards(files, 4)

    expensive_by_shard = [
        [path.name for path in shard if path in expensive] for shard in shards
    ]
    assert sorted(name for shard in expensive_by_shard for name in shard) == [
        "test_diffrax_integrators_core.py",
        "test_runtime_runner.py",
    ]
    assert all(len(shard_names) <= 1 for shard_names in expensive_by_shard)
    assert all(
        len(shard) == 1 for shard in shards if any(path in expensive for path in shard)
    )


def test_split_shards_rejects_nonpositive_count() -> None:
    with pytest.raises(ValueError, match="nshards"):
        split_shards([Path("tests/test_a.py")], 0)


def test_discover_test_files_returns_sorted_recursive_tests(tmp_path: Path) -> None:
    (tmp_path / "test_b.py").write_text("", encoding="utf-8")
    (tmp_path / "test_a.py").write_text("", encoding="utf-8")
    (tmp_path / "helper.py").write_text("", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "test_nested.py").write_text("", encoding="utf-8")

    assert [path.relative_to(tmp_path) for path in discover_test_files(tmp_path)] == [
        Path("nested/test_nested.py"),
        Path("test_a.py"),
        Path("test_b.py"),
    ]


def test_relative_test_dir_resolves_under_repository_root() -> None:
    resolved = _resolve_test_dir(Path("tests"))

    assert resolved.is_absolute()
    assert resolved.name == "tests"
    assert discover_test_files(Path("tests"))


def test_coverage_shard_report_tracks_labeled_data(tmp_path: Path) -> None:
    (tmp_path / ".coverage.shard-1.0").write_text("data", encoding="utf-8")
    (tmp_path / ".coverage.shard-2.0").write_text("data", encoding="utf-8")
    (tmp_path / ".coverage.local").write_text("data", encoding="utf-8")

    report = build_coverage_shard_report(tmp_path, 3)

    assert report["coverage_data_file_count"] == 3
    assert report["labeled_shards"] == {
        "1": [".coverage.shard-1.0"],
        "2": [".coverage.shard-2.0"],
    }
    assert report["unlabeled_coverage_data_files"] == [".coverage.local"]
    assert report["missing_labeled_shards"] == [3]
    failures = validate_coverage_shard_report(report, require_labeled_shards=True)
    assert "missing labeled coverage data for shards: [3]" in failures


def test_coverage_shard_report_rejects_empty_and_out_of_range_data(
    tmp_path: Path,
) -> None:
    (tmp_path / ".coverage.shard-1.0").write_text("data", encoding="utf-8")
    (tmp_path / ".coverage.shard-4.0").write_text("data", encoding="utf-8")
    (tmp_path / "EMPTY_SHARD_2").write_text("empty shard\n", encoding="utf-8")

    report = build_coverage_shard_report(tmp_path, 3)
    failures = validate_coverage_shard_report(report, require_labeled_shards=True)

    assert "empty shard markers found: ['EMPTY_SHARD_2']" in failures
    assert (
        "out-of-range labeled coverage data files found: ['.coverage.shard-4.0']"
        in failures
    )
    assert "missing labeled coverage data for shards: [2, 3]" in failures


def test_coverage_shard_report_requires_some_coverage_data(tmp_path: Path) -> None:
    report = build_coverage_shard_report(tmp_path, 2)

    assert validate_coverage_shard_report(report, require_labeled_shards=False) == [
        "no coverage.py data files were found"
    ]


def test_write_json_creates_parent_directory(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "manifest.json"

    write_json(out, {"b": 2, "a": 1})

    assert out.read_text(encoding="utf-8").splitlines() == [
        "{",
        '  "a": 1,',
        '  "b": 2',
        "}",
    ]
