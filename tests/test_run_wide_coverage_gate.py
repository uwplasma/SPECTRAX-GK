from __future__ import annotations

from pathlib import Path

import pytest

from tools.run_wide_coverage_gate import discover_test_files, split_shards


def test_split_shards_is_contiguous_and_complete() -> None:
    files = [Path(f"tests/test_{idx}.py") for idx in range(7)]
    shards = split_shards(files, 3)

    assert shards == [files[:3], files[3:6], files[6:]]
    assert [path for shard in shards for path in shard] == files


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

    assert [path.name for path in discover_test_files(tmp_path)] == ["test_a.py", "test_b.py"]
