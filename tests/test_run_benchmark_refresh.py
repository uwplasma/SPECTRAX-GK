from pathlib import Path

from tools.run_benchmark_refresh import _load_manifest, _select_jobs


def test_refresh_manifest_loads_jobs() -> None:
    manifest = Path("/path/to/SPECTRAX-GK/tools/benchmark_refresh_manifest.toml")
    jobs = _load_manifest(manifest)

    names = [job.name for job in jobs]
    assert "cyclone-core-assets" in names
    assert "benchmark-atlas" in names
    assert any(job.requires_env for job in jobs)


def test_refresh_job_selection_filters_named_jobs(tmp_path: Path) -> None:
    manifest = tmp_path / "mini.toml"
    manifest.write_text(
        """
[[job]]
name = "one"
description = "first"
command = "echo one"

[[job]]
name = "two"
description = "second"
command = "echo two"
enabled = false
""",
        encoding="utf-8",
    )
    jobs = _load_manifest(manifest)
    selected = _select_jobs(jobs, {"one", "two"})
    assert [job.name for job in selected] == ["one"]
