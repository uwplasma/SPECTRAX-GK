from pathlib import Path

from tools.benchmark_runtime_memory import (
    RuntimeBenchRun,
    _load_manifest,
    _load_summary_rows,
    _parse_profile_times,
    _parse_peak_rss_mb,
    _plot_results,
    _render,
    _run_command,
    _select_runs,
    _write_row_logs,
    _write_summary,
)


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
profile_command = "echo profile"
host = "office"
enabled = false
""",
        encoding="utf-8",
    )
    runs = _load_manifest(manifest)
    assert runs[1].host == "office"
    assert runs[1].profile_command == "echo profile"
    selected = _select_runs(runs, {"a"}, {"spectrax_cpu"})
    assert len(selected) == 1
    assert selected[0].case == "a"
    assert selected[0].backend == "spectrax_cpu"


def test_parse_peak_rss_mb_supports_macos_and_linux_formats() -> None:
    assert _parse_peak_rss_mb("peak memory footprint: 1048576") == 1.0
    assert _parse_peak_rss_mb("Maximum resident set size (kbytes): 2048") == 2.0


def test_parse_profile_times_extracts_warmup_and_run_fields() -> None:
    parsed = _parse_profile_times("warmup_time_s=30.776 run_time_s=14.081")
    assert parsed == {"warmup_time_s": 30.776, "run_time_s": 14.081}


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


def test_gx_runtime_memory_manifest_runs_in_isolated_tempdir() -> None:
    runs = _load_manifest(ROOT / "tools" / "runtime_memory_manifest.toml")
    gx_runs = [run for run in runs if run.backend == "gx"]
    assert gx_runs
    for run in gx_runs:
        assert "mktemp -d" in run.command
        assert "env " in run.command
        assert "-u DISPLAY" in run.command
        assert "HDF5_DISABLE_VERSION_CHECK=1" in run.command
        assert "CUDA_VISIBLE_DEVICES=${SPECTRAX_BENCH_CUDA_DEVICE}" in run.command


def test_gx_stellarator_runtime_manifest_uses_pregenerated_nc_geometry() -> None:
    runs = _load_manifest(ROOT / "tools" / "runtime_memory_manifest.toml")
    stellarator = [run for run in runs if run.backend == "gx" and run.case in {"w7x-linear", "w7x-nonlinear", "hsx-linear", "hsx-nonlinear"}]
    assert len(stellarator) == 4
    for run in stellarator:
        assert 'geo_option = "nc"' in run.command
        assert 'vmec_file' in run.command
        assert 'geo_file = "' in run.command
        assert "REFERENCE_GK_NETCDF_LIBDIR" in run.command
        assert "REFERENCE_GK_PYTHON_BIN" in run.command


def test_gpu_runtime_memory_manifest_pins_configured_cuda_device() -> None:
    runs = _load_manifest(ROOT / "tools" / "runtime_memory_manifest.toml")
    gpu_runs = [run for run in runs if run.backend == "spectrax_gpu"]
    assert gpu_runs
    for run in gpu_runs:
        assert "CUDA_VISIBLE_DEVICES=${SPECTRAX_BENCH_CUDA_DEVICE}" in run.command


def test_short_nonlinear_gpu_rows_request_warm_profile_pass() -> None:
    runs = _load_manifest(ROOT / "tools" / "runtime_memory_manifest.toml")
    selected = {
        (run.case, run.backend): run.profile_command
        for run in runs
        if run.backend == "spectrax_gpu" and run.case in {"cyclone-nonlinear", "kbm-nonlinear"}
    }
    assert "profile_nonlinear_cyclone.py" in str(selected[("cyclone-nonlinear", "spectrax_gpu")])
    assert "profile_nonlinear_cyclone.py" in str(selected[("kbm-nonlinear", "spectrax_gpu")])


def test_remote_runtime_memory_runs_disable_x11_forwarding(monkeypatch) -> None:
    captured = {}

    def fake_run(cmd, capture_output, text):  # type: ignore[no-untyped-def]
        captured["cmd"] = cmd
        class Proc:
            returncode = 0
            stdout = ""
            stderr = ""
        return Proc()

    monkeypatch.setattr("tools.benchmark_runtime_memory.subprocess.run", fake_run)
    run = RuntimeBenchRun(case="c", label="C", backend="gx", command="echo hi", cwd="/tmp", host="office")
    row = _run_command(run)
    assert row["status"] == "success"
    assert captured["cmd"][:2] == ["ssh", "-x"]


def test_runtime_memory_command_captures_profile_times(monkeypatch) -> None:
    def fake_run(cmd, shell, cwd, capture_output, text):  # type: ignore[no-untyped-def]
        class Proc:
            returncode = 0
            stdout = "warmup_time_s=12.5 run_time_s=3.25\n"
            stderr = "Maximum resident set size (kbytes): 2048\n"

        return Proc()

    monkeypatch.setattr("tools.benchmark_runtime_memory.subprocess.run", fake_run)
    run = RuntimeBenchRun(case="c", label="C", backend="spectrax_cpu", command="echo hi", cwd="/tmp", wrap_time=False)
    row = _run_command(run)
    assert row["runtime_s"] >= 0.0
    assert row["peak_rss_mb"] == 2.0
    assert row["warmup_time_s"] == 12.5
    assert row["run_time_s"] == 3.25


def test_runtime_memory_command_runs_profile_subcommand(monkeypatch) -> None:
    calls = []

    def fake_run(cmd, shell, cwd, capture_output, text):  # type: ignore[no-untyped-def]
        calls.append(cmd)

        class Proc:
            returncode = 0
            stdout = "main\n" if len(calls) == 1 else "warmup_time_s=20 run_time_s=7\n"
            stderr = "Maximum resident set size (kbytes): 2048\n" if len(calls) == 1 else ""

        return Proc()

    monkeypatch.setattr("tools.benchmark_runtime_memory.subprocess.run", fake_run)
    run = RuntimeBenchRun(
        case="c",
        label="C",
        backend="spectrax_gpu",
        command="echo cold",
        profile_command="echo warm",
        cwd="/tmp",
        wrap_time=False,
    )
    row = _run_command(run)
    assert len(calls) == 2
    assert row["status"] == "success"
    assert row["warmup_time_s"] == 20.0
    assert row["run_time_s"] == 7.0
    assert "--- profile stdout ---" in row["stdout"]


def test_runtime_memory_row_logs_are_written(tmp_path: Path) -> None:
    row = {
        "case": "cyclone-linear",
        "backend": "gx",
        "stdout": "ok",
        "stderr": "warn",
    }
    logs = _write_row_logs(tmp_path, row)
    assert Path(logs["stdout_log"]).read_text(encoding="utf-8") == "ok"
    assert Path(logs["stderr_log"]).read_text(encoding="utf-8") == "warn"


def test_runtime_memory_summary_is_written(tmp_path: Path) -> None:
    rows = [{"case": "a", "backend": "spectrax_cpu", "status": "success"}]
    out = tmp_path / "summary.json"
    _write_summary(out, rows)
    assert '"case": "a"' in out.read_text(encoding="utf-8")


def test_runtime_memory_plot_supports_warm_runtime_markers(tmp_path: Path) -> None:
    csv_path = tmp_path / "runtime.csv"
    csv_path.write_text(
        "\n".join(
            [
                "case,label,backend,status,returncode,runtime_s,warmup_time_s,run_time_s,peak_rss_mb,host,cwd,command,stdout_log,stderr_log",
                "cyclone-nonlinear,Cyclone ITG Nonlinear,spectrax_gpu,success,0,35.3,33.2,14.4,1878.4,office,/tmp,cmd,out,err",
                "cyclone-nonlinear,Cyclone ITG Nonlinear,gx,success,0,21.1,,,1900.0,office,/tmp,cmd,out,err",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    png_path = tmp_path / "runtime.png"
    pdf_path = tmp_path / "runtime.pdf"

    _plot_results(csv_path, png_path, pdf_path)

    assert png_path.exists()
    assert pdf_path.exists()
