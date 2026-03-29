from pathlib import Path

from tools.benchmark_runtime_memory import _load_manifest, _parse_peak_rss_mb, _select_runs


ROOT = Path("/path/to/SPECTRAX-GK")


def test_runtime_memory_manifest_loads_runs() -> None:
    runs = _load_manifest(ROOT / "tools" / "runtime_memory_manifest.toml")
    assert any(run.case == "cyclone-linear" and run.backend == "spectrax_cpu" for run in runs)
    assert any(run.backend == "gx" for run in runs)


def test_runtime_memory_selection_filters_case_and_backend(tmp_path: Path) -> None:
    manifest = tmp_path / "mini.toml"
    manifest.write_text(
        """
[[run]]
case = "a"
label = "A"
backend = "spectrax_cpu"
command = "echo a"

[[run]]
case = "b"
label = "B"
backend = "gx"
command = "echo b"
host = "office"
enabled = false
""",
        encoding="utf-8",
    )
    runs = _load_manifest(manifest)
    assert runs[1].host == "office"
    selected = _select_runs(runs, {"a"}, {"spectrax_cpu"})
    assert len(selected) == 1
    assert selected[0].case == "a"
    assert selected[0].backend == "spectrax_cpu"


def test_parse_peak_rss_mb_supports_macos_and_linux_formats() -> None:
    assert _parse_peak_rss_mb("peak memory footprint: 1048576") == 1.0
    assert _parse_peak_rss_mb("Maximum resident set size (kbytes): 2048") == 2.0
