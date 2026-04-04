"""CLI tests for basic command execution."""

import sys
from pathlib import Path

import numpy as np

from spectraxgk import __version__
from spectraxgk.analysis import ModeSelection
from spectraxgk.cli import main
from spectraxgk.diagnostics import GXDiagnostics
from spectraxgk.runtime import RuntimeLinearResult, RuntimeNonlinearResult


def test_version_exposed():
    """Version string should be exported from the package."""
    assert __version__ == "0.0.0"


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
    assert "saved artifacts/from_toml.summary.json" in out
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
    assert "saved artifacts/direct_shorthand.summary.json" in out
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
        diag = GXDiagnostics(
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
    """Adaptive nonlinear CLI runs should keep ``steps=None`` unless explicitly set."""

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
        diag = GXDiagnostics(
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
