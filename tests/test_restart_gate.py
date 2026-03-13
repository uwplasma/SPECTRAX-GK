from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from spectraxgk.config import GeometryConfig, GridConfig, InitializationConfig, TimeConfig
from spectraxgk.restart import write_gx_restart_state
from spectraxgk.runtime import run_runtime_nonlinear
from spectraxgk.runtime_config import (
    RuntimeCollisionConfig,
    RuntimeConfig,
    RuntimeNormalizationConfig,
    RuntimePhysicsConfig,
    RuntimeSpeciesConfig,
    RuntimeTermsConfig,
)


def _restart_base_cfg() -> RuntimeConfig:
    ion = RuntimeSpeciesConfig(
        name="ion",
        charge=1.0,
        mass=1.0,
        density=1.0,
        temperature=1.0,
        tprim=2.49,
        fprim=0.8,
    )
    return RuntimeConfig(
        grid=GridConfig(Nx=4, Ny=8, Nz=16, Lx=6.28, Ly=6.28, boundary="periodic"),
        time=TimeConfig(
            t_max=1.0,
            dt=0.02,
            method="rk2",
            use_diffrax=False,
            fixed_dt=True,
            diagnostics=True,
            sample_stride=1,
            diagnostics_stride=1,
        ),
        geometry=GeometryConfig(model="s-alpha", q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778),
        init=InitializationConfig(init_field="density", init_amp=1.0e-8, init_single=True, gaussian_init=False),
        species=(ion,),
        physics=RuntimePhysicsConfig(
            linear=False,
            nonlinear=True,
            adiabatic_electrons=True,
            tau_e=1.0,
            electrostatic=True,
            electromagnetic=False,
            collisions=False,
            hypercollisions=False,
        ),
        collisions=RuntimeCollisionConfig(damp_ends_amp=0.0),
        normalization=RuntimeNormalizationConfig(contract="cyclone", diagnostic_norm="none"),
        terms=RuntimeTermsConfig(nonlinear=1.0, end_damping=0.0, hypercollisions=0.0),
    )


def test_restart_gate_nonlinear_matches_continuous(tmp_path: Path) -> None:
    cfg = _restart_base_cfg()
    Nl = 4
    Nm = 6
    dt = 0.02
    steps1 = 7
    steps2 = 9
    ky = 0.2
    kx = 0.0

    full = run_runtime_nonlinear(
        cfg,
        ky_target=ky,
        kx_target=kx,
        Nl=Nl,
        Nm=Nm,
        dt=dt,
        steps=steps1 + steps2,
        sample_stride=1,
        diagnostics_stride=1,
        return_state=True,
    )
    part1 = run_runtime_nonlinear(
        cfg,
        ky_target=ky,
        kx_target=kx,
        Nl=Nl,
        Nm=Nm,
        dt=dt,
        steps=steps1,
        sample_stride=1,
        diagnostics_stride=1,
        return_state=True,
    )
    assert part1.state is not None
    assert full.state is not None

    restart_path = tmp_path / "restart.bin"
    write_gx_restart_state(restart_path, np.asarray(part1.state, dtype=np.complex64))

    cfg_restart = replace(
        cfg,
        init=replace(cfg.init, init_file=str(restart_path), init_file_scale=1.0, init_file_mode="replace"),
    )
    part2 = run_runtime_nonlinear(
        cfg_restart,
        ky_target=ky,
        kx_target=kx,
        Nl=Nl,
        Nm=Nm,
        dt=dt,
        steps=steps2,
        sample_stride=1,
        diagnostics_stride=1,
        return_state=True,
    )
    assert part2.state is not None

    np.testing.assert_array_equal(np.asarray(part2.state), np.asarray(full.state))

