"""Tests for unified runtime-configured linear runner."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from spectraxgk.config import GeometryConfig, GridConfig, InitializationConfig, TimeConfig
from spectraxgk.runtime import (
    build_runtime_linear_params,
    build_runtime_linear_terms,
    run_runtime_linear,
    run_runtime_scan,
)
from spectraxgk.runtime_config import (
    RuntimeConfig,
    RuntimeNormalizationConfig,
    RuntimePhysicsConfig,
    RuntimeSpeciesConfig,
    RuntimeTermsConfig,
)


def _base_runtime_cfg() -> RuntimeConfig:
    return RuntimeConfig(
        grid=GridConfig(Nx=1, Ny=8, Nz=16, Lx=6.28, Ly=6.28, boundary="periodic"),
        time=TimeConfig(t_max=0.2, dt=0.01, method="rk2", use_diffrax=False, sample_stride=1),
        geometry=GeometryConfig(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778),
        init=InitializationConfig(init_field="density", init_amp=1.0e-8, gaussian_init=False),
        terms=RuntimeTermsConfig(hypercollisions=0.0, end_damping=0.0),
    )


def test_runtime_linear_cyclone_etg_kbm_smoke() -> None:
    ion = RuntimeSpeciesConfig(
        name="ion", charge=1.0, mass=1.0, density=1.0, temperature=1.0, tprim=2.49, fprim=0.8
    )
    electron = RuntimeSpeciesConfig(
        name="electron",
        charge=-1.0,
        mass=1.0 / 3670.0,
        density=1.0,
        temperature=1.0,
        tprim=2.49,
        fprim=0.8,
    )

    cyclone = replace(
        _base_runtime_cfg(),
        species=(ion,),
        normalization=RuntimeNormalizationConfig(contract="cyclone", diagnostic_norm="none"),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, tau_e=1.0),
    )
    etg = replace(
        _base_runtime_cfg(),
        species=(electron,),
        normalization=RuntimeNormalizationConfig(contract="etg", diagnostic_norm="none"),
        physics=RuntimePhysicsConfig(
            adiabatic_electrons=False,
            adiabatic_ions=True,
            electrostatic=True,
            electromagnetic=False,
        ),
    )
    kbm = replace(
        _base_runtime_cfg(),
        grid=GridConfig(Nx=1, Ny=8, Nz=16, Lx=62.8, Ly=62.8, boundary="periodic"),
        init=InitializationConfig(init_field="all", init_amp=1.0e-8, gaussian_init=False),
        species=(ion, electron),
        normalization=RuntimeNormalizationConfig(contract="kbm", diagnostic_norm="none"),
        physics=RuntimePhysicsConfig(
            adiabatic_electrons=False,
            electrostatic=False,
            electromagnetic=True,
            use_apar=True,
            beta=0.2,
            hypercollisions=False,
        ),
    )

    for cfg, ky in ((cyclone, 0.2), (etg, 2.0), (kbm, 0.2)):
        res = run_runtime_linear(
            cfg,
            ky_target=ky,
            Nl=4,
            Nm=6,
            solver="krylov",
        )
        assert np.isfinite(res.gamma)
        assert np.isfinite(res.omega)


def test_runtime_terms_and_params_follow_toggles() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(contract="cyclone", diagnostic_norm="gx"),
        physics=RuntimePhysicsConfig(
            electrostatic=False,
            electromagnetic=True,
            use_apar=True,
            use_bpar=False,
            adiabatic_electrons=True,
            tau_e=1.0,
            beta=0.1,
            collisions=False,
            hypercollisions=False,
        ),
    )
    params = build_runtime_linear_params(cfg)
    terms = build_runtime_linear_terms(cfg)
    assert float(params.beta) == 0.1
    assert float(params.fapar) == 1.0
    assert terms.apar == 1.0
    assert terms.bpar == 0.0
    assert terms.collisions == 0.0
    assert terms.hypercollisions == 0.0


def test_runtime_scan_returns_arrays() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(contract="cyclone"),
    )
    scan = run_runtime_scan(
        cfg,
        ky_values=[0.1, 0.2],
        Nl=4,
        Nm=6,
        solver="krylov",
    )
    assert scan.ky.shape == (2,)
    assert scan.gamma.shape == (2,)
    assert scan.omega.shape == (2,)
