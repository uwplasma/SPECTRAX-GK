from __future__ import annotations

import subprocess
from pathlib import Path

from tools.release import run_tests_fast


def test_discover_test_files_returns_recursive_tests(tmp_path: Path) -> None:
    (tmp_path / "test_b.py").write_text("", encoding="utf-8")
    (tmp_path / "test_a.py").write_text("", encoding="utf-8")
    (tmp_path / "helper.py").write_text("", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "test_nested.py").write_text("", encoding="utf-8")

    assert [
        path.relative_to(tmp_path)
        for path in run_tests_fast.discover_test_files(tmp_path)
    ] == [
        Path("nested/test_nested.py"),
        Path("test_a.py"),
        Path("test_b.py"),
    ]


def test_relative_test_dir_resolves_under_repository_root() -> None:
    resolved = run_tests_fast._resolve_test_dir(Path("tests"))

    assert resolved.is_absolute()
    assert resolved.name == "tests"
    assert run_tests_fast.discover_test_files(Path("tests"))


def test_run_tests_uses_bounded_pytest_invocations(monkeypatch, tmp_path: Path) -> None:
    test_file = tmp_path / "test_sample.py"
    test_file.write_text("def test_ok(): assert True\n", encoding="utf-8")
    calls: list[tuple[list[str], float]] = []

    def _fake_run(cmd, *, cwd, check, timeout):
        del cwd, check
        calls.append((list(cmd), float(timeout)))

    monkeypatch.setattr(run_tests_fast.subprocess, "run", _fake_run)
    code, results = run_tests_fast.run_tests(
        [test_file],
        per_file_timeout_s=12.0,
        total_timeout_s=30.0,
        pytest_args=["-k", "sample"],
    )

    assert code == 0
    assert results[0][1] == "ok"
    assert calls[0][0][0:4] == [run_tests_fast.sys.executable, "-m", "pytest", "-q"]
    assert calls[0][0][-3:] == ["-k", "sample", str(test_file)]
    assert calls[0][1] <= 12.0


def test_run_tests_returns_124_on_timeout(monkeypatch, tmp_path: Path) -> None:
    test_file = tmp_path / "test_timeout.py"
    test_file.write_text("def test_slow(): assert True\n", encoding="utf-8")

    def _fake_run(cmd, *, cwd, check, timeout):
        del cwd, check
        raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(run_tests_fast.subprocess, "run", _fake_run)
    code, results = run_tests_fast.run_tests(
        [test_file],
        per_file_timeout_s=1.0,
        total_timeout_s=30.0,
    )

    assert code == 124
    assert results[0][1] == "timeout"


def test_run_tests_treats_pytest_no_tests_collected_as_skip(
    monkeypatch,
    tmp_path: Path,
) -> None:
    test_file = tmp_path / "test_integration_only.py"
    test_file.write_text(
        "import pytest\npytestmark = pytest.mark.integration\n",
        encoding="utf-8",
    )

    def _fake_run(cmd, *, cwd, check, timeout):
        del cwd, check, timeout
        raise subprocess.CalledProcessError(5, cmd)

    monkeypatch.setattr(run_tests_fast.subprocess, "run", _fake_run)
    code, results = run_tests_fast.run_tests(
        [test_file],
        per_file_timeout_s=1.0,
        total_timeout_s=30.0,
    )

    assert code == 0
    assert results[0][1] == "skipped(no_tests_collected)"


def test_run_tests_marks_remaining_files_after_total_timeout(
    monkeypatch, tmp_path: Path
) -> None:
    files = [tmp_path / "test_one.py", tmp_path / "test_two.py"]
    for path in files:
        path.write_text("def test_ok(): assert True\n", encoding="utf-8")
    monotonic_values = iter([0.0, 0.0, 0.1, 0.1, 0.2, 2.0])

    def _fake_monotonic() -> float:
        return next(monotonic_values)

    def _fake_run(cmd, *, cwd, check, timeout):
        del cmd, cwd, check, timeout

    monkeypatch.setattr(run_tests_fast.time, "monotonic", _fake_monotonic)
    monkeypatch.setattr(run_tests_fast.subprocess, "run", _fake_run)
    code, results = run_tests_fast.run_tests(
        files,
        per_file_timeout_s=10.0,
        total_timeout_s=1.0,
    )

    assert code == 124
    assert results[0][1] == "ok"
    assert results[1][1] == "not_run(total_timeout)"
