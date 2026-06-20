from __future__ import annotations

from pathlib import Path
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks" / "results" / "manifest.toml"
MAX_TRACKED_RESULT_BYTES = 1_000_000
MAX_ROOT_BENCHMARK_PAYLOAD_BYTES = 200_000


def _load_manifest() -> dict:
    with MANIFEST.open("rb") as fh:
        return tomllib.load(fh)


def test_benchmark_results_manifest_is_root_level_and_small() -> None:
    assert MANIFEST.exists()
    assert MANIFEST.relative_to(ROOT).parts[:2] == ("benchmarks", "results")
    assert MANIFEST.stat().st_size < 20_000

    readme = MANIFEST.parent / "README.md"
    assert readme.exists()
    assert readme.stat().st_size < 20_000


def test_benchmark_results_manifest_points_to_tracked_docs_outputs() -> None:
    manifest = _load_manifest()
    entries = [*manifest.get("figure", []), *manifest.get("table", [])]
    assert {entry["name"] for entry in entries} >= {
        "Core linear benchmark atlas",
        "Core nonlinear benchmark atlas",
        "Runtime and memory comparison",
        "Runtime and memory result rows",
    }

    for entry in entries:
        path = ROOT / entry["path"]
        assert path.exists(), entry["path"]
        assert path.is_file(), entry["path"]
        assert ROOT / "tools_out" not in path.parents
        assert ROOT / "docs" / "_build" not in path.parents
        assert path.stat().st_size <= MAX_TRACKED_RESULT_BYTES, entry["path"]

        source_manifest = ROOT / entry["source_manifest"]
        assert source_manifest.exists(), entry["source_manifest"]
        assert source_manifest.suffix == ".toml"
        assert entry["regenerate"].startswith("python ")
        assert entry["docs_page"].endswith((".rst", ".md"))
        assert entry["claim_scope"].strip()


def test_benchmark_results_manifest_documents_artifact_hygiene_policy() -> None:
    policy = _load_manifest()["policy"]
    assert policy["tracked_payload"] == "small pointers only"
    assert "tools_out" in policy["raw_outputs"]
    assert "docs/_static" in policy["docs_payload"]


def test_root_benchmark_manifest_is_reflected_in_docs() -> None:
    manifest = _load_manifest()
    docs_text = (ROOT / "docs" / "benchmarks.rst").read_text(encoding="utf-8")
    entries = [*manifest.get("figure", []), *manifest.get("table", [])]

    for entry in entries:
        assert entry["name"] in docs_text
        assert entry["path"] in docs_text
        assert entry["claim_scope"] in docs_text


def test_root_benchmark_payload_stays_lightweight() -> None:
    tracked_benchmark_files = [
        ROOT / path
        for path in subprocess.check_output(
            ["git", "ls-files", "benchmarks"],
            cwd=ROOT,
            text=True,
        ).splitlines()
    ]
    assert tracked_benchmark_files
    total_bytes = sum(path.stat().st_size for path in tracked_benchmark_files)
    assert total_bytes <= MAX_ROOT_BENCHMARK_PAYLOAD_BYTES
