"""Executable tests for basic command execution."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from spectraxgk import __version__
from spectraxgk.analysis import ModeSelection
from spectraxgk.cli import (
    _cmd_default_demo,
    _cmd_plot_saved_output,
    _cmd_run,
    _cmd_run_linear,
    _cmd_run_runtime_linear,
    _cmd_run_runtime_nonlinear,
    _cmd_scan_linear,
    _cmd_scan_runtime_linear,
    _default_example_config_path,
    _is_runtime_toml,
    _load_scan_ky,
    _resolve_case,
    _runtime_output_path,
    _should_show_progress,
    main,
)
from spectraxgk.diagnostics import SimulationDiagnostics
from spectraxgk.runtime import RuntimeLinearResult, RuntimeNonlinearResult
from spectraxgk.runtime_config import RuntimeConfig


def test_version_exposed():
    """Version string should be exported from the package."""
    assert __version__ == "1.2.0"


def test_cli_version_flag(capsys, monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["spectraxgk", "--version"])
    try:
        main()
    except SystemExit as exc:
        assert exc.code == 0
    out = capsys.readouterr().out
    assert "spectraxgk 1.2.0" in out


def test_cli_without_args_runs_default_demo(capsys, monkeypatch, tmp_path: Path) -> None:
    class _FakeFigure:
        def savefig(self, path, **_kwargs):
            Path(path).write_bytes(b"fake-image")

    class _FakeResult:
        t = np.asarray([0.1, 0.2])
        phi_t = np.asarray([[[[1.0 + 0.0j, 0.5 + 0.1j]]], [[[1.2 + 0.1j, 0.6 + 0.2j]]]])
        gamma = 0.12
        omega = -0.34
        ky = 0.3
        selection = ModeSelection(ky_index=0, kx_index=0, z_index=0)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("spectraxgk.cli._default_example_config_path", lambda: None)
    monkeypatch.setattr("spectraxgk.cli.run_cyclone_linear", lambda **_kwargs: _FakeResult())
    monkeypatch.setattr("spectraxgk.cli.extract_mode_time_series", lambda *_args, **_kwargs: np.asarray([1.0 + 0.0j, 1.2 + 0.1j]))
    monkeypatch.setattr("spectraxgk.cli.extract_eigenfunction", lambda *_args, **_kwargs: np.asarray([1.0 + 0.0j, 0.5 + 0.2j]))
    monkeypatch.setattr("spectraxgk.cli.normalize_eigenfunction", lambda eigen, _z: np.asarray(eigen))
    monkeypatch.setattr("spectraxgk.cli.build_spectral_grid", lambda _cfg: type("Grid", (), {"z": np.asarray([-1.0, 1.0])})())
    monkeypatch.setattr("spectraxgk.cli.linear_runtime_panel_figure", lambda **_kwargs: (_FakeFigure(), None))
    monkeypatch.setattr("matplotlib.pyplot.close", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(sys, "argv", ["spectraxgk"])

    code = main()
    out = capsys.readouterr().out
    assert code == 0
    assert "No input file specified" in out
    assert (tmp_path / "tools_out" / "spectraxgk_default_linear.png").exists()


def test_cli_global_plot_uses_saved_output_renderer(capsys, monkeypatch, tmp_path: Path) -> None:
    rendered = tmp_path / "rendered.png"
    monkeypatch.setattr("spectraxgk.cli.plot_saved_output", lambda path, out=None: rendered)
    monkeypatch.setattr(sys, "argv", ["spectraxgk", "--plot", "tools_out/linear_case.summary.json"])
    code = main()
    out = capsys.readouterr().out
    assert code == 0
    assert f"saved {rendered}" in out


def test_cli_global_plot_accepts_out_argument(capsys, monkeypatch, tmp_path: Path) -> None:
    rendered = tmp_path / "rendered-out.png"
    captured: dict[str, str | None] = {}

    def _plot(path, out=None):
        captured["path"] = path
        captured["out"] = out
        return rendered

    monkeypatch.setattr("spectraxgk.cli.plot_saved_output", _plot)
    monkeypatch.setattr(sys, "argv", ["spectraxgk", "--plot", "case.summary.json", "--out", "figure.png"])
    code = main()
    out = capsys.readouterr().out
    assert code == 0
    assert captured == {"path": "case.summary.json", "out": "figure.png"}
    assert f"saved {rendered}" in out


def test_cli_plot_usage_errors(capsys, monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["spectraxgk", "--plot"])
    assert main() == 1
    assert "usage: spectraxgk --plot" in capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", ["spectraxgk", "--plot", "a", "--bad"])
    assert main() == 1
    assert "usage: spectraxgk --plot" in capsys.readouterr().out


def test_cli_cyclone_info(capsys, monkeypatch):
    """The cyclone-info command should print default parameters."""
    monkeypatch.setattr(sys, "argv", ["spectrax-gk", "cyclone-info"])
    code = main()
    out = capsys.readouterr().out
    assert code == 0
    assert "Cyclone base case" in out


def test_cli_cyclone_kperp(capsys, monkeypatch):
    """The cyclone-kperp command should print k_perp^2 ranges."""
    monkeypatch.setattr(sys, "argv", ["spectrax-gk", "cyclone-kperp", "--kx0", "0.0", "--ky", "0.3"])
    code = main()
    out = capsys.readouterr().out
    assert code == 0
    assert "k_perp^2" in out


def test_cli_helper_predicates_and_dispatch_utils(monkeypatch, capsys) -> None:
    cfg = RuntimeConfig()
    args = argparse.Namespace(out="explicit.out", progress=False, no_progress=False)
    assert _runtime_output_path(args, cfg) == "explicit.out"
    args.out = None
    assert _runtime_output_path(args, cfg) == cfg.output.path

    assert _is_runtime_toml({"physics": {}}) is True
    assert _is_runtime_toml({"case": "cyclone"}) is False
    assert _is_runtime_toml({}) is True

    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    assert _should_show_progress(argparse.Namespace(progress=True, no_progress=False), False) is True
    assert _should_show_progress(argparse.Namespace(progress=False, no_progress=True), True) is False
    assert _should_show_progress(argparse.Namespace(progress=False, no_progress=False), True) is True
    assert _should_show_progress(argparse.Namespace(progress=False, no_progress=False), False) is False

    _resolve_case("cyclone")
    _resolve_case("etg")
    with pytest.raises(ValueError):
        _resolve_case("bad")

    assert np.allclose(_load_scan_ky({"scan": {"ky": [0.1, 0.2]}}), np.array([0.1, 0.2]))
    assert _load_scan_ky({}).size == 0


def test_cmd_run_handles_load_error_and_dispatches(monkeypatch, capsys) -> None:
    args = argparse.Namespace(config="bad.toml")

    def _boom(_path):
        raise RuntimeError("forced")

    monkeypatch.setattr("spectraxgk.cli.load_runtime_from_toml", _boom)
    assert _cmd_run(args) == 1
    assert "Error loading bad.toml" in capsys.readouterr().out

    nonlinear_cfg = type("Cfg", (), {"physics": type("Phys", (), {"nonlinear": True})()})()
    linear_cfg = type("Cfg", (), {"physics": type("Phys", (), {"nonlinear": False})()})()
    monkeypatch.setattr("spectraxgk.cli.load_runtime_from_toml", lambda _path: (nonlinear_cfg, {}))
    monkeypatch.setattr("spectraxgk.cli._cmd_run_runtime_nonlinear", lambda _args: 7)
    assert _cmd_run(args) == 7

    monkeypatch.setattr("spectraxgk.cli.load_runtime_from_toml", lambda _path: (linear_cfg, {}))
    monkeypatch.setattr("spectraxgk.cli._cmd_run_runtime_linear", lambda _args: 9)
    assert _cmd_run(args) == 9


def test_main_shorthand_dispatches_runtime_and_legacy(monkeypatch, tmp_path: Path) -> None:
    cfg_path = tmp_path / "case.toml"
    cfg_path.write_text("[physics]\n", encoding="utf-8")
    captured: list[list[str]] = []

    class _Parser:
        def parse_args(self, argv):
            captured.append(list(argv))
            return argparse.Namespace(func=lambda _args: 11)

    monkeypatch.setattr("spectraxgk.cli.load_toml", lambda _path: {"physics": {}})
    monkeypatch.setattr("spectraxgk.cli.build_parser", lambda: _Parser())
    monkeypatch.setattr(sys, "argv", ["spectraxgk", str(cfg_path)])
    assert main() == 11
    assert captured[-1][:2] == ["run", "--config"]

    monkeypatch.setattr("spectraxgk.cli.load_toml", lambda _path: {"case": "cyclone"})
    monkeypatch.setattr(sys, "argv", ["spectraxgk", str(cfg_path)])
    assert main() == 11
    assert captured[-1][:2] == ["run-linear", "--config"]


def test_default_example_config_path_exists_in_repo() -> None:
    path = _default_example_config_path()
    assert path is not None
    assert path.name == "cyclone.toml"
    assert path.exists()


def test_cmd_default_demo_uses_example_config_branch(monkeypatch, capsys, tmp_path: Path) -> None:
    class _FakeFigure:
        def savefig(self, path, **_kwargs):
            Path(path).write_bytes(b"fake-image")

    class _FakeResult:
        t = np.asarray([0.1, 0.2])
        phi_t = np.asarray([[[[1.0 + 0.0j, 0.5 + 0.1j]]], [[[1.2 + 0.1j, 0.6 + 0.2j]]]])
        gamma = 0.12
        omega = -0.34
        ky = 0.3
        selection = ModeSelection(ky_index=0, kx_index=0, z_index=0)

    example_path = tmp_path / "cyclone.toml"
    example_path.write_text("[run]\nky=0.4\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("spectraxgk.cli._default_example_config_path", lambda: example_path)
    monkeypatch.setattr(
        "spectraxgk.cli.load_case_from_toml",
        lambda *_args: (
            "cyclone",
            type("Cfg", (), {"time": type("TimeCfg", (), {"method": "rk2", "dt": 0.1, "t_max": 0.2})(), "grid": object()})(),
            {"run": {"ky": 0.4, "Nl": 8, "Nm": 6, "solver": "dense", "method": "rk4", "dt": 0.05, "steps": 6}, "fit": {"mode_method": "svd"}},
        ),
    )
    monkeypatch.setattr("spectraxgk.cli.run_cyclone_linear", lambda **_kwargs: _FakeResult())
    monkeypatch.setattr("spectraxgk.cli.extract_mode_time_series", lambda *_args, **_kwargs: np.asarray([1.0 + 0.0j, 1.2 + 0.1j]))
    monkeypatch.setattr("spectraxgk.cli.extract_eigenfunction", lambda *_args, **_kwargs: np.asarray([1.0 + 0.0j, 0.5 + 0.2j]))
    monkeypatch.setattr("spectraxgk.cli.normalize_eigenfunction", lambda eigen, _z: np.asarray(eigen))
    monkeypatch.setattr("spectraxgk.cli.build_spectral_grid", lambda _cfg: type("Grid", (), {"z": np.asarray([-1.0, 1.0])})())
    monkeypatch.setattr("spectraxgk.cli.linear_runtime_panel_figure", lambda **_kwargs: (_FakeFigure(), None))
    monkeypatch.setattr("matplotlib.pyplot.close", lambda *_args, **_kwargs: None)

    assert _cmd_default_demo() == 0
    out = capsys.readouterr().out
    assert "source=" in out
    assert "cyclone.toml" in out
    assert (tmp_path / "tools_out" / "spectraxgk_default_linear.png").exists()


def test_cli_run_runtime_linear(capsys, monkeypatch, tmp_path: Path):
    """The unified runtime command should run a tiny linear configuration."""
    cfg = """
[[species]]
name = "ion"
charge = 1.0
mass = 1.0
density = 1.0
temperature = 1.0
tprim = 2.49
fprim = 0.8
kinetic = true

[grid]
Nx = 1
Ny = 6
Nz = 16
Lx = 62.8
Ly = 62.8
boundary = "periodic"

[time]
t_max = 0.2
dt = 0.01
method = "rk2"
use_diffrax = false

[geometry]
q = 1.4
s_hat = 0.8
epsilon = 0.18
R0 = 2.77778

[init]
init_field = "density"
init_amp = 1e-8
gaussian_init = false

[physics]
electrostatic = true
electromagnetic = false
adiabatic_electrons = true
tau_e = 1.0

[normalization]
contract = "cyclone"
diagnostic_norm = "none"

[run]
ky = 0.2
Nl = 4
Nm = 6
solver = "krylov"
"""
    path = tmp_path / "runtime_cli.toml"
    path.write_text(cfg, encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["spectrax-gk", "run-runtime-linear", "--config", str(path)],
    )
    code = main()
    out = capsys.readouterr().out
    assert code == 0
    assert "starting runtime linear run" in out
    assert "gamma=" in out


def test_cli_run_runtime_linear_writes_artifacts(capsys, monkeypatch, tmp_path: Path) -> None:
    cfg = """
[[species]]
name = "ion"
charge = 1.0
mass = 1.0
density = 1.0
temperature = 1.0
tprim = 2.49
fprim = 0.8
kinetic = true

[grid]
Nx = 1
Ny = 6
Nz = 16
Lx = 62.8
Ly = 62.8
boundary = "periodic"

[time]
t_max = 0.2
dt = 0.01
method = "rk2"
use_diffrax = false

[geometry]
q = 1.4
s_hat = 0.8
epsilon = 0.18
R0 = 2.77778

[init]
init_field = "density"
init_amp = 1e-8
gaussian_init = false

[physics]
electrostatic = true
electromagnetic = false
adiabatic_electrons = true
tau_e = 1.0

[normalization]
contract = "cyclone"
diagnostic_norm = "none"
"""
    path = tmp_path / "runtime_cli_linear_out.toml"
    path.write_text(cfg, encoding="utf-8")
    out_base = tmp_path / "linear_bundle"

    def _fake_run_runtime_linear(_cfg, **_kwargs):
        return RuntimeLinearResult(
            ky=0.2,
            gamma=0.3,
            omega=-0.4,
            selection=ModeSelection(ky_index=0, kx_index=0, z_index=1),
            t=np.asarray([0.1, 0.2]),
            signal=np.asarray([1.0, 2.0]),
        )

    monkeypatch.setattr("spectraxgk.cli.run_runtime_linear", _fake_run_runtime_linear)
    monkeypatch.setattr(
        sys,
        "argv",
        ["spectrax-gk", "run-runtime-linear", "--config", str(path), "--out", str(out_base)],
    )
    code = main()
    out = capsys.readouterr().out
    assert code == 0
    assert "saved" in out
    assert (tmp_path / "linear_bundle.summary.json").exists()
    assert (tmp_path / "linear_bundle.timeseries.csv").exists()


def test_cmd_run_linear_plot_branch(monkeypatch, tmp_path: Path) -> None:
    class _FakeResult:
        ky = 0.2
        gamma = 0.3
        omega = -0.4
        t = np.asarray([0.1, 0.2])
        phi_t = np.ones((2, 1, 1, 2), dtype=np.complex128)
        selection = ModeSelection(ky_index=0, kx_index=0, z_index=0)

    class _FakeCase:
        grid = type("GridCfg", (), {"Nx": 1, "Ny": 2, "Nz": 2})()
        time = type("TimeCfg", (), {"method": "rk2", "dt": 0.1, "t_max": 0.2, "progress_bar": False})()

    class _Fig:
        def savefig(self, path, **_kwargs):
            Path(path).write_bytes(b"fig")

        def tight_layout(self):
            return None

    monkeypatch.setattr("spectraxgk.cli.load_case_from_toml", lambda *_args: ("cyclone", _FakeCase(), {"run": {}, "fit": {}}))
    monkeypatch.setattr("spectraxgk.cli._resolve_case", lambda _name: (object, lambda **_kwargs: _FakeResult()))
    monkeypatch.setattr("spectraxgk.cli.load_linear_terms_from_toml", lambda _data: None)
    monkeypatch.setattr("spectraxgk.cli.load_krylov_from_toml", lambda _data: None)
    monkeypatch.setattr("spectraxgk.cli.build_spectral_grid", lambda _cfg: type("Grid", (), {"z": np.asarray([-1.0, 1.0])})())
    monkeypatch.setattr("spectraxgk.cli.extract_mode_time_series", lambda *_args, **_kwargs: np.asarray([1.0 + 0.0j, 1.2 + 0.1j]))
    monkeypatch.setattr("spectraxgk.cli.growth_fit_figure", lambda *_args, **_kwargs: (_Fig(), None))
    monkeypatch.setattr("spectraxgk.cli.extract_eigenfunction", lambda *_args, **_kwargs: np.asarray([1.0 + 0.0j, 0.5 + 0.2j]))
    monkeypatch.setattr("spectraxgk.cli.normalize_eigenfunction", lambda eigen, _z: np.asarray(eigen))
    monkeypatch.setattr("spectraxgk.cli.set_plot_style", lambda: None)
    monkeypatch.setattr("matplotlib.pyplot.subplots", lambda *args, **kwargs: (_Fig(), type("Ax", (), {
        "plot": lambda *a, **k: None,
        "set_xlabel": lambda *a, **k: None,
        "set_ylabel": lambda *a, **k: None,
        "set_title": lambda *a, **k: None,
        "legend": lambda *a, **k: None,
    })()))
    monkeypatch.setattr("matplotlib.pyplot.close", lambda *_args, **_kwargs: None)
    args = argparse.Namespace(
        config="case.toml",
        case=None,
        ky=None,
        Nl=None,
        Nm=None,
        solver=None,
        method=None,
        dt=None,
        steps=None,
        fit_signal=None,
        plot=True,
        outdir=str(tmp_path),
        progress=False,
        no_progress=False,
    )
    assert _cmd_run_linear(args) == 0
    assert (tmp_path / "cyclone_ky0.200_fit.png").exists()
    assert (tmp_path / "cyclone_ky0.200_eig.png").exists()


def test_cmd_scan_linear_branches(monkeypatch, tmp_path: Path, capsys) -> None:
    class _Cfg:
        time = type("TimeCfg", (), {"method": "rk2", "dt": 0.1, "t_max": 0.2})()

    scan_result = type("Scan", (), {
        "ky": np.array([0.1, 0.2]),
        "gamma": np.array([0.3, 0.4]),
        "omega": np.array([-0.1, -0.2]),
    })()

    monkeypatch.setattr("spectraxgk.cli.load_case_from_toml", lambda *_args: ("other", _Cfg(), {"scan": {"ky": [0.1, 0.2]}, "fit": {}}))
    monkeypatch.setattr("spectraxgk.cli._resolve_case", lambda _name: (object, object()))
    monkeypatch.setattr("spectraxgk.cli.load_linear_terms_from_toml", lambda _data: None)
    monkeypatch.setattr("spectraxgk.cli.load_krylov_from_toml", lambda _data: None)
    monkeypatch.setattr("spectraxgk.cli.run_linear_scan", lambda **_kwargs: scan_result)
    args = argparse.Namespace(
        config="case.toml",
        case=None,
        ky_values=None,
        Nl=None,
        Nm=None,
        solver=None,
        method=None,
        dt=None,
        steps=None,
        fit_signal=None,
        plot=True,
        outdir=str(tmp_path),
    )
    assert _cmd_scan_linear(args) == 0
    assert "No reference available" in capsys.readouterr().out

    monkeypatch.setattr("spectraxgk.cli.load_case_from_toml", lambda *_args: ("cyclone", _Cfg(), {"scan": {}, "fit": {}}))
    args.ky_values = None
    with pytest.raises(ValueError):
        _cmd_scan_linear(args)


def test_cmd_scan_linear_with_reference_plot(monkeypatch, tmp_path: Path) -> None:
    class _Cfg:
        time = type("TimeCfg", (), {"method": "rk2", "dt": 0.1, "t_max": 0.2})()

    class _Fig:
        def savefig(self, path, **_kwargs):
            Path(path).write_bytes(b"fig")

    scan_result = type("Scan", (), {
        "ky": np.array([0.1, 0.2]),
        "gamma": np.array([0.3, 0.4]),
        "omega": np.array([-0.1, -0.2]),
    })()
    ref = type("Ref", (), {"ky": np.array([0.1, 0.2]), "gamma": np.array([0.3, 0.4]), "omega": np.array([-0.1, -0.2])})()
    monkeypatch.setattr("spectraxgk.cli.load_case_from_toml", lambda *_args: ("cyclone", _Cfg(), {"scan": {}, "fit": {}}))
    monkeypatch.setattr("spectraxgk.cli._resolve_case", lambda _name: (object, object()))
    monkeypatch.setattr("spectraxgk.cli.load_linear_terms_from_toml", lambda _data: None)
    monkeypatch.setattr("spectraxgk.cli.load_krylov_from_toml", lambda _data: None)
    monkeypatch.setattr("spectraxgk.cli.run_linear_scan", lambda **_kwargs: scan_result)
    monkeypatch.setattr("spectraxgk.cli.load_cyclone_reference", lambda: ref)
    monkeypatch.setattr("spectraxgk.cli.scan_comparison_figure", lambda *args, **kwargs: (_Fig(), None))
    args = argparse.Namespace(
        config="case.toml",
        case=None,
        ky_values="0.1,0.2",
        Nl=None,
        Nm=None,
        solver=None,
        method=None,
        dt=None,
        steps=None,
        fit_signal=None,
        plot=True,
        outdir=str(tmp_path),
    )
    assert _cmd_scan_linear(args) == 0
    assert (tmp_path / "cyclone_scan_comparison.png").exists()


def test_cmd_scan_runtime_linear_branches(monkeypatch, capsys) -> None:
    cfg = RuntimeConfig()
    scan = type("Scan", (), {
        "ky": np.array([0.1]),
        "gamma": np.array([0.2]),
        "omega": np.array([-0.3]),
    })()
    monkeypatch.setattr("spectraxgk.cli.load_runtime_from_toml", lambda _path: (cfg, {"scan": {"ky": [0.1]}, "fit": {}}))
    monkeypatch.setattr("spectraxgk.cli.run_runtime_scan", lambda *args, **kwargs: scan)
    args = argparse.Namespace(
        config="case.toml",
        ky_values="0.1",
        Nl=None,
        Nm=None,
        solver=None,
        fit_signal=None,
        method=None,
        dt=None,
        steps=None,
        sample_stride=None,
        batch_ky=True,
        progress=False,
        no_progress=True,
    )
    assert _cmd_scan_runtime_linear(args) == 0
    assert "ky=0.1000 gamma=0.200000 omega=-0.300000" in capsys.readouterr().out

    args.ky_values = None
    monkeypatch.setattr("spectraxgk.cli.load_runtime_from_toml", lambda _path: (cfg, {"scan": {}, "fit": {}}))
    with pytest.raises(ValueError):
        _cmd_scan_runtime_linear(args)


def test_cmd_run_runtime_nonlinear_branches(monkeypatch, capsys, tmp_path: Path) -> None:
    cfg = RuntimeConfig()
    diag = SimulationDiagnostics(
        t=np.asarray([0.1, 0.2]),
        dt_t=np.asarray([0.1, 0.1]),
        dt_mean=np.asarray(0.1),
        gamma_t=np.asarray([0.0, 0.0]),
        omega_t=np.asarray([0.0, 0.0]),
        Wg_t=np.asarray([1.0, 1.1]),
        Wphi_t=np.asarray([2.0, 2.1]),
        Wapar_t=np.asarray([0.0, 0.0]),
        heat_flux_t=np.asarray([3.0, 3.1]),
        particle_flux_t=np.asarray([4.0, 4.1]),
        energy_t=np.asarray([3.0, 3.2]),
        phi_mode_t=None,
    )
    result = RuntimeNonlinearResult(
        t=np.asarray([0.1, 0.2]),
        diagnostics=diag,
        ky_selected=0.2,
        kx_selected=0.0,
    )
    paths = {"summary": "a", "diagnostics": "b", "state": "c", "out": "d", "big": "e", "restart": "f"}
    monkeypatch.setattr("spectraxgk.cli.load_runtime_from_toml", lambda _path: (cfg, {"run": {"steps": 5}}))
    monkeypatch.setattr("spectraxgk.cli.run_runtime_nonlinear_with_artifacts", lambda *args, **kwargs: (result, paths))
    args = argparse.Namespace(
        config="case.toml",
        init_file=str(tmp_path / "seed.bin"),
        ky=None,
        Nl=None,
        Nm=None,
        dt=None,
        steps=None,
        method=None,
        sample_stride=None,
        diagnostics_stride=None,
        diagnostics=False,
        no_diagnostics=False,
        laguerre_mode=None,
        progress=False,
        no_progress=False,
        out="out.nc",
    )
    assert _cmd_run_runtime_nonlinear(args) == 0
    out = capsys.readouterr().out
    assert "starting runtime nonlinear run" in out
    assert "saved a" in out and "saved f" in out

    no_diag_result = RuntimeNonlinearResult(t=np.asarray([0.1]), diagnostics=None, ky_selected=0.2, kx_selected=0.0)
    monkeypatch.setattr("spectraxgk.cli.run_runtime_nonlinear_with_artifacts", lambda *args, **kwargs: (no_diag_result, {}))
    args.no_diagnostics = True
    assert _cmd_run_runtime_nonlinear(args) == 0
    assert "nonlinear run completed" in capsys.readouterr().out


def test_cmd_run_runtime_linear_prints_optional_artifact_paths(monkeypatch, capsys) -> None:
    cfg = RuntimeConfig()
    monkeypatch.setattr("spectraxgk.cli.load_runtime_from_toml", lambda _path: (cfg, {"run": {}}))
    monkeypatch.setattr(
        "spectraxgk.cli.run_runtime_linear",
        lambda *_args, **_kwargs: RuntimeLinearResult(
            ky=0.2,
            gamma=0.3,
            omega=-0.4,
            selection=ModeSelection(ky_index=0, kx_index=0, z_index=0),
            t=np.asarray([0.1, 0.2]),
            signal=np.asarray([1.0, 2.0]),
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.cli.write_runtime_linear_artifacts",
        lambda *_args, **_kwargs: {
            "summary": "sum.json",
            "timeseries": "diag.csv",
            "eigenfunction": "eig.csv",
            "state": "state.npy",
        },
    )
    args = argparse.Namespace(
        config="case.toml",
        ky=None,
        Nl=None,
        Nm=None,
        dt=None,
        steps=None,
        method=None,
        sample_stride=None,
        solver=None,
        fit_signal=None,
        progress=False,
        no_progress=False,
        out="bundle",
    )
    assert _cmd_run_runtime_linear(args) == 0
    out = capsys.readouterr().out
    assert "saved eig.csv" in out
    assert "saved state.npy" in out


def test_cmd_run_runtime_nonlinear_fixed_dt_and_explicit_diagnostics(monkeypatch, capsys) -> None:
    cfg = RuntimeConfig()
    cfg = RuntimeConfig(time=type(cfg.time)(**{**cfg.time.__dict__, "fixed_dt": True, "t_max": 1.0, "dt": 0.2}))
    captured: dict[str, object] = {}
    result = RuntimeNonlinearResult(t=np.asarray([0.1]), diagnostics=None, ky_selected=0.2, kx_selected=0.0)

    def _runner(*_args, **kwargs):
        captured.update(kwargs)
        return result, {}

    monkeypatch.setattr("spectraxgk.cli.load_runtime_from_toml", lambda _path: (cfg, {"run": {}}))
    monkeypatch.setattr("spectraxgk.cli.run_runtime_nonlinear_with_artifacts", _runner)
    args = argparse.Namespace(
        config="case.toml",
        init_file=None,
        ky=None,
        Nl=None,
        Nm=None,
        dt=None,
        steps=None,
        method=None,
        sample_stride=None,
        diagnostics_stride=None,
        diagnostics=True,
        no_diagnostics=False,
        laguerre_mode=None,
        progress=False,
        no_progress=False,
        out=None,
    )
    assert _cmd_run_runtime_nonlinear(args) == 0
    assert captured["steps"] == 5
    assert captured["diagnostics"] is True


def test_cli_run_runtime_linear_uses_toml_output_path(capsys, monkeypatch, tmp_path: Path) -> None:
    cfg = """
[[species]]
name = "ion"
charge = 1.0
mass = 1.0
density = 1.0
temperature = 1.0
tprim = 2.49
fprim = 0.8
kinetic = true

[grid]
Nx = 1
Ny = 6
Nz = 16
Lx = 62.8
Ly = 62.8
boundary = "periodic"

[time]
t_max = 0.2
dt = 0.01
method = "rk2"
use_diffrax = false

[geometry]
q = 1.4
s_hat = 0.8
epsilon = 0.18
R0 = 2.77778

[init]
init_field = "density"
init_amp = 1e-8
gaussian_init = false

[physics]
electrostatic = true
electromagnetic = false
adiabatic_electrons = true
tau_e = 1.0

[normalization]
contract = "cyclone"
diagnostic_norm = "none"

[output]
path = "artifacts/from_toml"
"""
    path = tmp_path / "runtime_cli_linear_toml_out.toml"
    path.write_text(cfg, encoding="utf-8")

    def _fake_run_runtime_linear(_cfg, **_kwargs):
        return RuntimeLinearResult(
            ky=0.2,
            gamma=0.3,
            omega=-0.4,
            selection=ModeSelection(ky_index=0, kx_index=0, z_index=1),
            t=np.asarray([0.1, 0.2]),
            signal=np.asarray([1.0, 2.0]),
        )

    monkeypatch.setattr("spectraxgk.cli.run_runtime_linear", _fake_run_runtime_linear)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["spectrax-gk", "run-runtime-linear", "--config", str(path)],
    )
    code = main()
    out = capsys.readouterr().out
    assert code == 0
    assert f"saved {tmp_path / 'artifacts' / 'from_toml.summary.json'}" in out
    assert (tmp_path / "artifacts" / "from_toml.summary.json").exists()
    assert (tmp_path / "artifacts" / "from_toml.timeseries.csv").exists()


def test_cli_run_runtime_linear_cli_out_overrides_toml_output_path(capsys, monkeypatch, tmp_path: Path) -> None:
    cfg = """
[[species]]
name = "ion"
charge = 1.0
mass = 1.0
density = 1.0
temperature = 1.0
tprim = 2.49
fprim = 0.8
kinetic = true

[grid]
Nx = 1
Ny = 6
Nz = 16
Lx = 62.8
Ly = 62.8
boundary = "periodic"

[time]
t_max = 0.2
dt = 0.01
method = "rk2"
use_diffrax = false

[geometry]
q = 1.4
s_hat = 0.8
epsilon = 0.18
R0 = 2.77778

[init]
init_field = "density"
init_amp = 1e-8
gaussian_init = false

[physics]
electrostatic = true
electromagnetic = false
adiabatic_electrons = true
tau_e = 1.0

[normalization]
contract = "cyclone"
diagnostic_norm = "none"

[output]
path = "artifacts/from_toml"
"""
    path = tmp_path / "runtime_cli_linear_toml_override.toml"
    path.write_text(cfg, encoding="utf-8")
    out_base = tmp_path / "cli_override"

    def _fake_run_runtime_linear(_cfg, **_kwargs):
        return RuntimeLinearResult(
            ky=0.2,
            gamma=0.3,
            omega=-0.4,
            selection=ModeSelection(ky_index=0, kx_index=0, z_index=1),
            t=np.asarray([0.1, 0.2]),
            signal=np.asarray([1.0, 2.0]),
        )

    monkeypatch.setattr("spectraxgk.cli.run_runtime_linear", _fake_run_runtime_linear)
    monkeypatch.setattr(
        sys,
        "argv",
        ["spectrax-gk", "run-runtime-linear", "--config", str(path), "--out", str(out_base)],
    )
    code = main()
    out = capsys.readouterr().out
    assert code == 0
    assert f"saved {out_base}.summary.json" in out
    assert (tmp_path / "cli_override.summary.json").exists()
    assert not (tmp_path / "artifacts" / "from_toml.summary.json").exists()


def test_cli_direct_config_shorthand_uses_toml_output_path(capsys, monkeypatch, tmp_path: Path) -> None:
    cfg = """
[[species]]
name = "ion"
charge = 1.0
mass = 1.0
density = 1.0
temperature = 1.0
tprim = 2.49
fprim = 0.8
kinetic = true

[grid]
Nx = 1
Ny = 6
Nz = 16
Lx = 62.8
Ly = 62.8
boundary = "periodic"

[time]
t_max = 0.2
dt = 0.01
method = "rk2"
use_diffrax = false

[geometry]
q = 1.4
s_hat = 0.8
epsilon = 0.18
R0 = 2.77778

[init]
init_field = "density"
init_amp = 1e-8
gaussian_init = false

[physics]
electrostatic = true
electromagnetic = false
adiabatic_electrons = true
tau_e = 1.0

[normalization]
contract = "cyclone"
diagnostic_norm = "none"

[output]
path = "artifacts/direct_shorthand"
"""
    path = tmp_path / "runtime_cli_direct.toml"
    path.write_text(cfg, encoding="utf-8")

    def _fake_run_runtime_linear(_cfg, **_kwargs):
        return RuntimeLinearResult(
            ky=0.2,
            gamma=0.3,
            omega=-0.4,
            selection=ModeSelection(ky_index=0, kx_index=0, z_index=1),
            t=np.asarray([0.1, 0.2]),
            signal=np.asarray([1.0, 2.0]),
        )

    monkeypatch.setattr("spectraxgk.cli.run_runtime_linear", _fake_run_runtime_linear)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["spectrax-gk", str(path)],
    )
    code = main()
    out = capsys.readouterr().out
    assert code == 0
    assert "starting runtime linear run" in out
    assert f"saved {tmp_path / 'artifacts' / 'direct_shorthand.summary.json'}" in out
    assert (tmp_path / "artifacts" / "direct_shorthand.summary.json").exists()


def test_cli_direct_config_shorthand_legacy_case_uses_run_linear_path(
    capsys, monkeypatch, tmp_path: Path
) -> None:
    cfg = """
[grid]
Nx = 8
Ny = 8
Nz = 16
Lx = 62.8
Ly = 62.8

[time]
t_max = 1.0
dt = 0.1

[geometry]
q = 1.4
s_hat = 0.8
epsilon = 0.18

[model]
R_over_LTi = 6.9
R_over_LTe = 0.0
R_over_Ln = 2.2
"""
    path = tmp_path / "cyclone_base_case.toml"
    path.write_text(cfg, encoding="utf-8")

    class _FakeResult:
        ky = 0.3
        gamma = 0.2
        omega = 0.4

    monkeypatch.setattr("spectraxgk.cli.run_cyclone_linear", lambda **_kwargs: _FakeResult())
    monkeypatch.setattr(sys, "argv", ["spectrax-gk", str(path)])

    code = main()
    out = capsys.readouterr().out
    assert code == 0
    assert "starting legacy linear cyclone run" in out
    assert "detected legacy case TOML; using run-linear path" in out
    assert "gamma=" in out


def test_cli_run_runtime_nonlinear(capsys, monkeypatch, tmp_path: Path):
    """The unified runtime nonlinear command should run a tiny configuration."""
    cfg = """
[[species]]
name = "ion"
charge = 1.0
mass = 1.0
density = 1.0
temperature = 1.0
tprim = 2.49
fprim = 0.8
kinetic = true

[grid]
Nx = 1
Ny = 6
Nz = 16
Lx = 62.8
Ly = 62.8
boundary = "periodic"

[time]
t_max = 0.1
dt = 0.01
method = "rk2"
use_diffrax = false

[geometry]
q = 1.4
s_hat = 0.8
epsilon = 0.18
R0 = 2.77778

[init]
init_field = "density"
init_amp = 1e-8
gaussian_init = false

[physics]
electrostatic = true
electromagnetic = false
adiabatic_electrons = true
tau_e = 1.0
nonlinear = true

[terms]
nonlinear = 1.0

[normalization]
contract = "cyclone"
diagnostic_norm = "none"

[run]
ky = 0.2
Nl = 3
Nm = 4
"""
    path = tmp_path / "runtime_cli_nonlinear.toml"
    path.write_text(cfg, encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["spectrax-gk", "run-runtime-nonlinear", "--config", str(path), "--steps", "3"],
    )
    code = main()
    out = capsys.readouterr().out
    assert code == 0
    assert "nonlinear" in out


def test_cli_direct_config_shorthand_runs_nonlinear(capsys, monkeypatch, tmp_path: Path):
    """Direct ``spectrax-gk path/to/config.toml`` should dispatch nonlinear configs."""
    cfg = """
[[species]]
name = "ion"
charge = 1.0
mass = 1.0
density = 1.0
temperature = 1.0
tprim = 2.49
fprim = 0.8
kinetic = true

[grid]
Nx = 1
Ny = 6
Nz = 16
Lx = 62.8
Ly = 62.8
boundary = "periodic"

[time]
t_max = 0.1
dt = 0.01
method = "rk2"
use_diffrax = false

[geometry]
q = 1.4
s_hat = 0.8
epsilon = 0.18
R0 = 2.77778

[init]
init_field = "density"
init_amp = 1e-8
gaussian_init = false

[physics]
electrostatic = true
electromagnetic = false
adiabatic_electrons = true
tau_e = 1.0
nonlinear = true

[terms]
nonlinear = 1.0

[normalization]
contract = "cyclone"
diagnostic_norm = "none"
"""
    path = tmp_path / "runtime_cli_shorthand.toml"
    path.write_text(cfg, encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["spectrax-gk", str(path), "--steps", "3"])
    code = main()
    out = capsys.readouterr().out
    assert code == 0
    assert "nonlinear" in out


def test_cli_direct_config_shorthand_accepts_no_progress(monkeypatch, tmp_path: Path) -> None:
    """Direct config shorthand should forward progress flags to nonlinear runtime runs."""
    cfg = """
[[species]]
name = "ion"
charge = 1.0
mass = 1.0
density = 1.0
temperature = 1.0
tprim = 2.49
fprim = 0.8
kinetic = true

[grid]
Nx = 1
Ny = 6
Nz = 16
Lx = 62.8
Ly = 62.8
boundary = "periodic"

[time]
t_max = 0.1
dt = 0.01
method = "rk2"
use_diffrax = false

[geometry]
q = 1.4
s_hat = 0.8
epsilon = 0.18
R0 = 2.77778

[init]
init_field = "density"
init_amp = 1e-8
gaussian_init = false

[physics]
electrostatic = true
electromagnetic = false
adiabatic_electrons = true
tau_e = 1.0
nonlinear = true

[terms]
nonlinear = 1.0
"""
    path = tmp_path / "runtime_cli_shorthand_progress.toml"
    path.write_text(cfg, encoding="utf-8")
    captured: dict[str, object] = {}

    def _fake_run_runtime_nonlinear_with_artifacts(_cfg, **kwargs):
        captured.update(kwargs)
        diag = SimulationDiagnostics(
            t=np.asarray([0.1]),
            dt_t=np.asarray([0.1]),
            dt_mean=np.asarray(0.1),
            gamma_t=np.asarray([0.0]),
            omega_t=np.asarray([0.0]),
            Wg_t=np.asarray([1.0]),
            Wphi_t=np.asarray([2.0]),
            Wapar_t=np.asarray([0.0]),
            heat_flux_t=np.asarray([3.0]),
            particle_flux_t=np.asarray([0.0]),
            energy_t=np.asarray([3.0]),
            heat_flux_species_t=None,
            particle_flux_species_t=None,
            phi_mode_t=None,
        )
        return (
            RuntimeNonlinearResult(
                t=np.asarray([0.1]),
                diagnostics=diag,
                ky_selected=0.2,
                kx_selected=0.0,
            ),
            {},
        )

    monkeypatch.setattr("spectraxgk.cli.run_runtime_nonlinear_with_artifacts", _fake_run_runtime_nonlinear_with_artifacts)
    monkeypatch.setattr(sys, "argv", ["spectrax-gk", str(path), "--steps", "3", "--no-progress"])
    code = main()
    assert code == 0
    assert captured["show_progress"] is False


def test_cli_run_runtime_nonlinear_outputs_species_flux_columns(capsys, monkeypatch, tmp_path: Path):
    """Nonlinear CSV output should include per-species flux diagnostics when available."""
    cfg = """
[[species]]
name = "ion"
charge = 1.0
mass = 1.0
density = 1.0
temperature = 1.0
tprim = 2.49
fprim = 0.8
kinetic = true

[grid]
Nx = 1
Ny = 6
Nz = 16
Lx = 62.8
Ly = 62.8
boundary = "periodic"

[time]
t_max = 0.1
dt = 0.01
method = "rk2"
use_diffrax = false

[geometry]
q = 1.4
s_hat = 0.8
epsilon = 0.18
R0 = 2.77778

[init]
init_field = "density"
init_amp = 1e-8
gaussian_init = false

[physics]
electrostatic = true
electromagnetic = false
adiabatic_electrons = true
tau_e = 1.0
nonlinear = true

[terms]
nonlinear = 1.0

[normalization]
contract = "cyclone"
diagnostic_norm = "none"

[run]
ky = 0.2
Nl = 3
Nm = 4
"""
    path = tmp_path / "runtime_cli_nonlinear_species.toml"
    out_path = tmp_path / "diag.csv"
    path.write_text(cfg, encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "spectrax-gk",
            "run-runtime-nonlinear",
            "--config",
            str(path),
            "--steps",
            "3",
            "--out",
            str(out_path),
        ],
    )
    code = main()
    _out = capsys.readouterr().out
    assert code == 0
    header = out_path.read_text(encoding="utf-8").splitlines()[0]
    assert "heat_flux_s0" in header
    assert "particle_flux_s0" in header


def test_cli_run_runtime_nonlinear_keeps_adaptive_steps_none(capsys, monkeypatch, tmp_path: Path):
    """Adaptive nonlinear executable runs should keep ``steps=None`` unless explicitly set."""

    cfg = """
[[species]]
name = "ion"
charge = 1.0
mass = 1.0
density = 1.0
temperature = 1.0
tprim = 2.49
fprim = 0.8
kinetic = true

[grid]
Nx = 1
Ny = 6
Nz = 16
Lx = 62.8
Ly = 62.8
boundary = "periodic"

[time]
t_max = 0.2
dt = 0.01
method = "rk2"
use_diffrax = false
fixed_dt = false

[geometry]
q = 1.4
s_hat = 0.8
epsilon = 0.18
R0 = 2.77778

[init]
init_field = "density"
init_amp = 1e-8
gaussian_init = false

[physics]
electrostatic = true
electromagnetic = false
adiabatic_electrons = true
tau_e = 1.0
nonlinear = true

[terms]
nonlinear = 1.0

[normalization]
contract = "cyclone"
diagnostic_norm = "none"

[run]
ky = 0.2
Nl = 3
Nm = 4
"""
    path = tmp_path / "runtime_cli_nonlinear_adaptive.toml"
    path.write_text(cfg, encoding="utf-8")

    captured: dict[str, object] = {}

    def _fake_run_runtime_nonlinear_with_artifacts(cfg, **kwargs):
        captured["steps"] = kwargs.get("steps")
        diag = SimulationDiagnostics(
            t=np.asarray([0.1]),
            dt_t=np.asarray([0.01]),
            dt_mean=np.asarray(0.01),
            gamma_t=np.asarray([0.0]),
            omega_t=np.asarray([0.0]),
            Wg_t=np.asarray([1.0]),
            Wphi_t=np.asarray([0.5]),
            Wapar_t=np.asarray([0.0]),
            heat_flux_t=np.asarray([0.0]),
            particle_flux_t=np.asarray([0.0]),
            energy_t=np.asarray([1.5]),
        )
        return (
            RuntimeNonlinearResult(
                t=np.asarray([0.1]),
                diagnostics=diag,
                ky_selected=0.2,
                kx_selected=0.0,
            ),
            {},
        )

    monkeypatch.setattr("spectraxgk.cli.run_runtime_nonlinear_with_artifacts", _fake_run_runtime_nonlinear_with_artifacts)
    monkeypatch.setattr(
        sys,
        "argv",
        ["spectrax-gk", "run-runtime-nonlinear", "--config", str(path)],
    )

    code = main()
    out = capsys.readouterr().out
    assert code == 0
    assert "nonlinear:" in out
    assert "t=0.1" in out
    assert captured["steps"] is None


_RUNTIME_LINEAR_TOML_WITH_VMEC = """
[[species]]
name = "ion"
charge = 1.0
mass = 1.0
density = 1.0
temperature = 1.0
tprim = 2.49
fprim = 0.8
kinetic = true

[grid]
Nx = 1
Ny = 6
Nz = 16
Lx = 62.8
Ly = 62.8
boundary = "periodic"

[time]
t_max = 0.2
dt = 0.01
method = "rk2"
use_diffrax = false

[geometry]
q = 1.4
s_hat = 0.8
epsilon = 0.18
R0 = 2.77778
vmec_file = "toml_wout.nc"

[init]
init_field = "density"
init_amp = 1e-8
gaussian_init = false

[physics]
electrostatic = true
electromagnetic = false
adiabatic_electrons = true
tau_e = 1.0

[normalization]
contract = "cyclone"
diagnostic_norm = "none"
"""


_RUNTIME_LINEAR_TOML_IMPORTED_GEOMETRY = """
[[species]]
name = "ion"
charge = 1.0
mass = 1.0
density = 1.0
temperature = 1.0
tprim = 2.49
fprim = 0.8
kinetic = true

[grid]
Nx = 1
Ny = 6
Nz = 16
Lx = 62.8
Ly = 62.8
boundary = "periodic"

[time]
t_max = 0.2
dt = 0.01
method = "rk2"
use_diffrax = false

[geometry]
model = "vmec-eik"
geometry_file = "from_config.eik.nc"

[init]
init_field = "density"
init_amp = 1e-8
gaussian_init = false

[physics]
electrostatic = true
electromagnetic = false
adiabatic_electrons = true
tau_e = 1.0

[normalization]
contract = "cyclone"
diagnostic_norm = "none"
"""


_RUNTIME_LINEAR_TOML_VMEC_MODEL = """
[[species]]
name = "ion"
charge = 1.0
mass = 1.0
density = 1.0
temperature = 1.0
tprim = 2.49
fprim = 0.8
kinetic = true

[grid]
Nx = 1
Ny = 6
Nz = 16
Lx = 62.8
Ly = 62.8
boundary = "periodic"

[time]
t_max = 0.2
dt = 0.01
method = "rk2"
use_diffrax = false

[geometry]
model = "vmec"
vmec_file = "wout_from_config.nc"
geometry_file = "generated_from_config.eik.nc"

[init]
init_field = "density"
init_amp = 1e-8
gaussian_init = false

[physics]
electrostatic = true
electromagnetic = false
adiabatic_electrons = true
tau_e = 1.0

[normalization]
contract = "cyclone"
diagnostic_norm = "none"
"""


_RUNTIME_NONLINEAR_TOML_MIN = """
[[species]]
name = "ion"
charge = 1.0
mass = 1.0
density = 1.0
temperature = 1.0
tprim = 2.49
fprim = 0.8
kinetic = true

[grid]
Nx = 1
Ny = 6
Nz = 16
Lx = 62.8
Ly = 62.8
boundary = "periodic"

[time]
t_max = 0.2
dt = 0.01
method = "rk2"
use_diffrax = false

[geometry]
q = 1.4
s_hat = 0.8
epsilon = 0.18
R0 = 2.77778
vmec_file = "toml_wout.nc"

[init]
init_field = "density"
init_amp = 1e-8
gaussian_init = false

[physics]
electrostatic = true
electromagnetic = false
adiabatic_electrons = true
tau_e = 1.0
nonlinear = true

[terms]
nonlinear = 1.0

[normalization]
contract = "cyclone"
diagnostic_norm = "none"

[run]
ky = 0.2
Nl = 3
Nm = 4
steps = 1
"""


def test_cli_run_runtime_linear_cli_vmec_file_resolves_against_cwd(monkeypatch, tmp_path: Path) -> None:
    """--vmec-file override lands on cfg.geometry.vmec_file, resolved relative to shell cwd."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "runtime.toml"
    config_path.write_text(_RUNTIME_LINEAR_TOML_WITH_VMEC, encoding="utf-8")

    shell_cwd = tmp_path / "shellwd"
    (shell_cwd / "sub").mkdir(parents=True)
    (shell_cwd / "sub" / "cli_wout.nc").write_text("", encoding="utf-8")
    monkeypatch.chdir(shell_cwd)

    captured: dict[str, object] = {}

    def _fake_run_runtime_linear(cfg, **_kwargs):
        captured["vmec_file"] = cfg.geometry.vmec_file
        return RuntimeLinearResult(
            ky=0.2,
            gamma=0.3,
            omega=-0.4,
            selection=ModeSelection(ky_index=0, kx_index=0, z_index=1),
            t=np.asarray([0.1, 0.2]),
            signal=np.asarray([1.0, 2.0]),
        )

    monkeypatch.setattr("spectraxgk.cli.run_runtime_linear", _fake_run_runtime_linear)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "spectrax-gk",
            "run-runtime-linear",
            "--config",
            str(config_path),
            "--vmec-file",
            "sub/cli_wout.nc",
        ],
    )
    assert main() == 0
    expected_cwd_resolved = str((shell_cwd / "sub" / "cli_wout.nc").resolve())
    assert captured["vmec_file"] == expected_cwd_resolved
    # Regression guard: must NOT resolve against the config file's parent directory.
    assert captured["vmec_file"] != str((config_dir / "sub" / "cli_wout.nc"))


def test_cli_run_runtime_nonlinear_init_file_expands_home(monkeypatch, tmp_path: Path) -> None:
    """--init-file with leading ~ should expand to $HOME (regression guard for prior bypass)."""
    config_path = tmp_path / "runtime.toml"
    config_path.write_text(_RUNTIME_NONLINEAR_TOML_MIN, encoding="utf-8")

    captured: dict[str, object] = {}

    def _fake_run_runtime_nonlinear_with_artifacts(cfg, **_kwargs):
        captured["init_file"] = cfg.init.init_file
        diag = SimulationDiagnostics(
            t=np.asarray([0.1]),
            dt_t=np.asarray([0.01]),
            dt_mean=np.asarray(0.01),
            gamma_t=np.asarray([0.0]),
            omega_t=np.asarray([0.0]),
            Wg_t=np.asarray([1.0]),
            Wphi_t=np.asarray([0.5]),
            Wapar_t=np.asarray([0.0]),
            heat_flux_t=np.asarray([0.0]),
            particle_flux_t=np.asarray([0.0]),
            energy_t=np.asarray([1.5]),
        )
        return (
            RuntimeNonlinearResult(
                t=np.asarray([0.1]),
                diagnostics=diag,
                ky_selected=0.2,
                kx_selected=0.0,
            ),
            {},
        )

    monkeypatch.setattr(
        "spectraxgk.cli.run_runtime_nonlinear_with_artifacts",
        _fake_run_runtime_nonlinear_with_artifacts,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "spectrax-gk",
            "run-runtime-nonlinear",
            "--config",
            str(config_path),
            "--init-file",
            "~/spectraxgk_test/g_state.h5",
            "--steps",
            "1",
        ],
    )
    assert main() == 0
    init_file = captured["init_file"]
    assert init_file is not None
    assert init_file.startswith(os.path.expanduser("~"))
    assert init_file.endswith("g_state.h5")
    assert "~" not in init_file


def test_cli_run_runtime_linear_cli_geometry_file_resolves_against_cwd_for_imported_geometry(
    monkeypatch, tmp_path: Path
) -> None:
    """--geometry-file overrides [geometry].geometry_file (cwd-resolved) without touching [geometry].model."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "runtime.toml"
    config_path.write_text(_RUNTIME_LINEAR_TOML_IMPORTED_GEOMETRY, encoding="utf-8")

    shell_cwd = tmp_path / "shellwd"
    (shell_cwd / "sub").mkdir(parents=True)
    (shell_cwd / "sub" / "cli.eik.nc").write_text("", encoding="utf-8")
    monkeypatch.chdir(shell_cwd)

    captured: dict[str, object] = {}

    def _fake_run_runtime_linear(cfg, **_kwargs):
        captured["geometry_file"] = cfg.geometry.geometry_file
        captured["model"] = cfg.geometry.model
        return RuntimeLinearResult(
            ky=0.2,
            gamma=0.3,
            omega=-0.4,
            selection=ModeSelection(ky_index=0, kx_index=0, z_index=1),
            t=np.asarray([0.1, 0.2]),
            signal=np.asarray([1.0, 2.0]),
        )

    monkeypatch.setattr("spectraxgk.cli.run_runtime_linear", _fake_run_runtime_linear)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "spectrax-gk",
            "run-runtime-linear",
            "--config",
            str(config_path),
            "--geometry-file",
            "sub/cli.eik.nc",
        ],
    )
    assert main() == 0
    expected_cwd_resolved = str((shell_cwd / "sub" / "cli.eik.nc").resolve())
    assert captured["geometry_file"] == expected_cwd_resolved
    # Override resolves against shell cwd, not the config file's parent directory.
    assert captured["geometry_file"] != str(config_dir / "sub" / "cli.eik.nc")
    # --geometry-file must not change [geometry].model: imported-geometry stays imported.
    assert captured["model"] == "vmec-eik"


def test_cli_run_runtime_linear_cli_geometry_file_does_not_change_vmec_model(
    monkeypatch, tmp_path: Path
) -> None:
    """--geometry-file on a model="vmec" TOML must not flip the model to imported-EIK."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "runtime.toml"
    config_path.write_text(_RUNTIME_LINEAR_TOML_VMEC_MODEL, encoding="utf-8")

    shell_cwd = tmp_path / "shellwd"
    (shell_cwd / "cache").mkdir(parents=True)
    monkeypatch.chdir(shell_cwd)

    captured: dict[str, object] = {}

    def _fake_run_runtime_linear(cfg, **_kwargs):
        captured["geometry_file"] = cfg.geometry.geometry_file
        captured["vmec_file"] = cfg.geometry.vmec_file
        captured["model"] = cfg.geometry.model
        return RuntimeLinearResult(
            ky=0.2,
            gamma=0.3,
            omega=-0.4,
            selection=ModeSelection(ky_index=0, kx_index=0, z_index=1),
            t=np.asarray([0.1, 0.2]),
            signal=np.asarray([1.0, 2.0]),
        )

    monkeypatch.setattr("spectraxgk.cli.run_runtime_linear", _fake_run_runtime_linear)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "spectrax-gk",
            "run-runtime-linear",
            "--config",
            str(config_path),
            "--geometry-file",
            "cache/generated_cli.eik.nc",
        ],
    )
    assert main() == 0
    expected_cwd_resolved = str((shell_cwd / "cache" / "generated_cli.eik.nc").resolve())
    assert captured["geometry_file"] == expected_cwd_resolved
    # --geometry-file must not promote a VMEC-backed run into imported-geometry mode.
    assert captured["model"] == "vmec"
