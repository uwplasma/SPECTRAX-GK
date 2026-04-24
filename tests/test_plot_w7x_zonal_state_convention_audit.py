from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from spectraxgk.config import GeometryConfig, GridConfig, InitializationConfig, TimeConfig
from spectraxgk.runtime_config import RuntimeConfig, RuntimePhysicsConfig, RuntimeSpeciesConfig


def _load_tool_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "plot_w7x_zonal_state_convention_audit.py"
    spec = importlib.util.spec_from_file_location("plot_w7x_zonal_state_convention_audit", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _cfg() -> RuntimeConfig:
    return RuntimeConfig(
        grid=GridConfig(Nx=6, Ny=4, Nz=32, Lx=2.0 * np.pi / 0.07, Ly=62.8, boundary="periodic"),
        time=TimeConfig(t_max=0.1, dt=0.01, method="rk4", use_diffrax=False),
        geometry=GeometryConfig(model="s-alpha", q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778),
        species=(RuntimeSpeciesConfig(name="ion", charge=1.0, density=1.0, temperature=1.0, kinetic=True),),
        init=InitializationConfig(
            init_field="phi",
            init_amp=0.25,
            gaussian_init=True,
            gaussian_width=1.0,
            init_single=True,
        ),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, tau_e=1.0),
    )


def test_w7x_zonal_state_convention_audit_closes_synthetic_phi_state(tmp_path: Path) -> None:
    mod = _load_tool_module()
    audit = mod.build_state_audit(_cfg(), kx_target=0.07, ky_target=0.0, Nl=2, Nm=2)

    row = audit["row"]
    assert audit["passed"] is True
    assert row["profile_relative_l2"] < 1.0e-4
    assert row["line_helper_vs_manual_rel"] < 1.0e-6
    assert row["mode_helper_vs_manual_rel"] < 1.0e-6
    assert row["line_first_initial_over_init_amp"] < 1.0

    out_png = tmp_path / "state.png"
    mod.write_outputs(
        audit,
        out_png=out_png,
        out_csv=out_png.with_suffix(".csv"),
        out_json=out_png.with_suffix(".json"),
        config=Path("synthetic.toml"),
    )

    assert out_png.exists()
    assert out_png.with_suffix(".pdf").exists()
    payload = json.loads(out_png.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["validation_status"] == "state_convention_closed"
    assert payload["gate_index_include"] is False
