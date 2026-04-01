from pathlib import Path

from tools.benchmark_runtime_memory import _load_manifest, _load_summary_rows, _parse_peak_rss_mb, _render, _select_runs


ROOT = Path(__file__).resolve().parents[1]


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


def test_load_summary_rows_merges_matching_json_files(tmp_path: Path) -> None:
    first = tmp_path / "a.json"
    first.write_text('{"rows":[{"case":"a","backend":"spectrax_cpu","status":"success"}]}\n', encoding="utf-8")
    second = tmp_path / "b.json"
    second.write_text('{"rows":[{"case":"a","backend":"gx","status":"success"}]}\n', encoding="utf-8")
    rows = _load_summary_rows([str(tmp_path / "*.json")])
    assert len(rows) == 2
    assert {row["backend"] for row in rows} == {"spectrax_cpu", "gx"}


def test_render_expands_root_and_env(monkeypatch) -> None:
    monkeypatch.setenv("SPECTRAX_BENCH_ROOT", "/tmp/bench")
    rendered = _render("{root}:${SPECTRAX_BENCH_ROOT}")
    assert str(ROOT) in rendered
    assert "/tmp/bench" in rendered
