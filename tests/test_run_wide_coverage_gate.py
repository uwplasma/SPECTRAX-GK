from __future__ import annotations

from pathlib import Path

import pytest

from tools.run_wide_coverage_gate import (
    _resolve_test_dir,
    discover_test_files,
    split_shards,
)


def test_split_shards_is_round_robin_and_complete() -> None:
    files = [Path(f"tests/test_{idx}.py") for idx in range(7)]
    shards = split_shards(files, 3)

    assert shards == [files[0::3], files[1::3], files[2::3]]
    assert sorted(path for shard in shards for path in shard) == files


def test_split_shards_rejects_nonpositive_count() -> None:
    with pytest.raises(ValueError, match="nshards"):
        split_shards([Path("tests/test_a.py")], 0)


def test_discover_test_files_returns_sorted_top_level_tests(tmp_path: Path) -> None:
    (tmp_path / "test_b.py").write_text("", encoding="utf-8")
    (tmp_path / "test_a.py").write_text("", encoding="utf-8")
    (tmp_path / "helper.py").write_text("", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "test_nested.py").write_text("", encoding="utf-8")

    assert [path.name for path in discover_test_files(tmp_path)] == [
        "test_a.py",
        "test_b.py",
    ]


def test_relative_test_dir_resolves_under_repository_root() -> None:
    resolved = _resolve_test_dir(Path("tests"))

    assert resolved.is_absolute()
    assert resolved.name == "tests"
    assert discover_test_files(Path("tests"))
