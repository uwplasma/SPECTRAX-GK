"""Tests for the imported-geometry GX linear comparison tool."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import compare_gx_imported_linear as imported_linear

from compare_gx_imported_linear import (
    GXInputContract,
    _build_imported_initial_condition,
    _infer_gx_linear_dt,
    _integrate_target_mode_series,
    _gx_Wg_by_ky,
    _gx_kyst_fac_mask_cached,
    _load_gx_input_contract,
    _match_local_kx_index,
    _resolve_imported_real_fft_ny,
    _run_single_ky,
    _select_geometry_source,
    _select_gx_kx_index,
    _write_scan_rows,
    build_parser,
)
from spectraxgk.config import GridConfig
from spectraxgk.grids import build_spectral_grid
from spectraxgk.gx_integrators import GXTimeConfig
from spectraxgk.linear import LinearTerms
from spectraxgk.species import Species


def test_compare_gx_imported_linear_parser_accepts_gx_input() -> None:
    args = build_parser().parse_args(
        [
            "--gx",
            "/tmp/run.out.nc",
            "--geometry-file",
            "/tmp/run.eik.nc",
            "--gx-input",
            "/tmp/run.in",
        ]
    )
    assert args.gx_input == Path("/tmp/run.in")


def test_load_gx_input_contract_reads_fix_aspect_and_species_contract(tmp_path: Path) -> None:
    path = tmp_path / "run.in"
    path.write_text(
        """
[Dimensions]
 ntheta = 48
 nperiod = 1
 ny = 96
 nx = 96
 nspecies = 1

[Domain]
 y0 = 21.0
 boundary = "fix aspect"

[Physics]
 beta = 0.01

[Time]
 dt = 0.005
 scheme = "rk3"

[Initialization]
 init_field = "density"
 init_amp = 1.0e-3
 ikpar_init = 0

[Diagnostics]
 nwrite = 50

[species]
 z = [1.0, -1.0]
 mass = [1.0, 0.00027]
 dens = [1.0, 1.0]
 temp = [1.0, 1.0]
 tprim = [3.0, 0.0]
 fprim = [1.0, 0.0]
 vnewk = [0.01, 0.0]

[Boltzmann]
 add_Boltzmann_species = true
 Boltzmann_type = "electrons"
 tau_fac = 1.0

[Dissipation]
 hypercollisions = true
 hyper = true
 D_hyper = 0.05
""".strip()
    )

    contract = _load_gx_input_contract(path)
    assert contract.Nx == 96
    assert contract.Ny == 96
    assert contract.nperiod == 1
    assert contract.ntheta == 48
    assert contract.boundary == "fix aspect"
    assert contract.geo_option == "s-alpha"
    assert contract.y0 == 21.0
    assert contract.beta == 0.01
    assert contract.tau_e == 1.0
    assert contract.dt == 0.005
    assert contract.scheme == "rk3"
    assert contract.nwrite == 50
    assert contract.init_field == "density"
    assert contract.init_amp == 1.0e-3
    assert contract.init_single is False
    assert contract.gaussian_init is False
    assert contract.kpar_init == 0.0
    assert contract.random_seed == 22
    assert contract.hypercollisions is True
    assert contract.hyper is True
    assert contract.D_hyper == 0.05
    assert contract.damp_ends_amp == 0.1
    assert contract.damp_ends_widthfrac == 1.0 / 8.0
    assert len(contract.species) == 1
    assert contract.species[0].charge == 1.0
    assert contract.species[0].tprim == 3.0


def test_imported_linear_uses_raw_damp_ends_rate() -> None:
    contract = _dummy_gx_contract(init_single=False)
    dt = 0.2
    params = imported_linear.build_linear_params(
        contract.species,
        tau_e=contract.tau_e,
        kpar_scale=1.0,
        beta=contract.beta,
    )
    params = replace(
        params,
        D_hyper=float(contract.D_hyper),
        damp_ends_amp=float(contract.damp_ends_amp),
        damp_ends_widthfrac=float(contract.damp_ends_widthfrac),
    )
    assert float(params.damp_ends_amp) == pytest.approx(0.1)
    assert float(params.damp_ends_amp) != pytest.approx(0.1 / dt)


def test_infer_gx_linear_dt_prefers_explicit_input_dt() -> None:
    contract = replace(_dummy_gx_contract(init_single=False), dt=0.025, nwrite=50)
    gx_time = np.asarray([1.25, 2.50, 3.75], dtype=float)
    assert _infer_gx_linear_dt(gx_time, contract) == pytest.approx(0.025)


def test_infer_gx_linear_dt_uses_diagnostic_spacing_without_input_dt() -> None:
    contract = replace(_dummy_gx_contract(init_single=False), dt=None, nwrite=100)
    gx_time = np.asarray([0.5, 1.0, 1.5, 2.0], dtype=float)
    assert _infer_gx_linear_dt(gx_time, contract) == pytest.approx(0.005)


def test_build_imported_initial_condition_uses_runtime_multikx_startup() -> None:
    class DummyGeom:
        s_hat = 1.0

    contract = _load_gx_input_contract(
        Path("/path/to/SPECTRAX-GK/.cache/gx_clean_main/linear/hsx/hsx_linear.in")
    )
    grid_full = build_spectral_grid(
        GridConfig(
            Nx=9,
            Ny=10,
            Nz=8,
            Lx=62.8,
            Ly=2.0 * np.pi * contract.y0,
            boundary="periodic",
            y0=contract.y0,
            nperiod=1,
            ntheta=8,
        )
    )
    g0 = _build_imported_initial_condition(
        grid=grid_full,
        geom=DummyGeom(),
        gx_contract=contract,
        species=contract.species,
        ky_index=1,
        kx_index=0,
        Nl=8,
        Nm=4,
    )
    g0_np = np.asarray(g0)
    nonzero_kx = np.flatnonzero(np.any(np.abs(g0_np[0, 0, 0, 1]) > 0.0, axis=-1))
    assert nonzero_kx.size > 1


def test_match_local_kx_index_uses_kx_value_not_raw_index() -> None:
    grid_kx = np.asarray([0.0, 0.05, 0.10, 0.15, -0.15, -0.10, -0.05], dtype=float)
    assert _match_local_kx_index(grid_kx, -0.10) == 5
    assert _match_local_kx_index(grid_kx, 0.15) == 3


def test_select_gx_kx_index_defaults_to_kx_zero_branch() -> None:
    gx_kx = np.asarray([-0.2, -0.1, 0.0, 0.1, 0.2], dtype=float)
    assert _select_gx_kx_index(gx_kx, None) == 2
    assert _select_gx_kx_index(gx_kx, _dummy_gx_contract(init_single=False)) == 2


def test_select_gx_kx_index_honors_explicit_single_mode_startup() -> None:
    gx_kx = np.asarray([-0.2, -0.1, 0.0, 0.1, 0.2], dtype=float)
    contract = replace(_dummy_gx_contract(init_single=True), ikx_single=4)
    assert _select_gx_kx_index(gx_kx, contract) == 4


def test_resolve_imported_real_fft_ny_uses_full_gx_ky_layout() -> None:
    gx_ky = np.asarray([0.0] + [0.05 * i for i in range(1, 16)], dtype=float)
    contract = replace(_dummy_gx_contract(init_single=False), Ny=16)
    assert _resolve_imported_real_fft_ny(gx_ky, contract) == 43


def test_resolve_imported_real_fft_ny_keeps_single_positive_ky_unmasked() -> None:
    gx_ky = np.asarray([0.0, 0.01], dtype=float)
    contract = replace(_dummy_gx_contract(init_single=False), Ny=2)
    assert _resolve_imported_real_fft_ny(gx_ky, contract) == 4


def _dummy_gx_contract(*, init_single: bool) -> GXInputContract:
    return GXInputContract(
        Nx=8,
        Ny=8,
        nperiod=1,
        ntheta=8,
        boundary="periodic",
        geo_option="s-alpha",
        y0=10.0,
        species=(Species(charge=1.0, mass=1.0, density=1.0, temperature=1.0, tprim=0.0, fprim=0.0),),
        tau_e=0.0,
        beta=0.0,
        dt=0.1,
        scheme="rk4",
        nwrite=1,
        init_field="density",
        init_amp=1.0e-5,
        init_single=init_single,
        ikx_single=0,
        iky_single=1,
        gaussian_init=False,
        gaussian_width=0.5,
        gaussian_envelope_constant=1.0,
        gaussian_envelope_sine=0.0,
        kpar_init=0.0,
        random_seed=22,
        init_electrons_only=False,
        random_init=False,
        hypercollisions=False,
        hyper=False,
        D_hyper=0.0,
        damp_ends_amp=0.1,
        damp_ends_widthfrac=1.0 / 8.0,
    )


def test_run_single_ky_uses_full_grid_for_imported_multimode(monkeypatch) -> None:
    grid_full = SimpleNamespace(
        ky=np.asarray([0.0, 0.1, 0.2], dtype=float),
        kx=np.asarray([0.0, 0.1], dtype=float),
        z=np.asarray([-1.0, 0.0, 1.0, 2.0], dtype=float),
    )
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        imported_linear,
        "_build_imported_initial_condition",
        lambda **_: np.zeros((1, 1, 1, 3, 2, 4), dtype=np.complex64),
    )
    monkeypatch.setattr(imported_linear, "build_linear_cache", lambda *_args, **_kwargs: "cache")

    def _fake_integrate(**kwargs):
        captured["grid"] = kwargs["grid"]
        captured["g_shape"] = tuple(np.asarray(kwargs["G0"]).shape)
        captured["ky_index"] = kwargs["ky_index"]
        return tuple(np.zeros(2, dtype=float) for _ in range(5))

    monkeypatch.setattr(imported_linear, "_integrate_target_mode_series", _fake_integrate)

    _run_single_ky(
        ky_target=0.1,
        geom=SimpleNamespace(),
        grid_full=grid_full,
        params=SimpleNamespace(),
        time_cfg=GXTimeConfig(dt=0.1, t_max=0.2, sample_stride=1, fixed_dt=True),
        gx_contract=_dummy_gx_contract(init_single=False),
        species=(Species(charge=1.0, mass=1.0, density=1.0, temperature=1.0, tprim=0.0, fprim=0.0),),
        Nl=1,
        Nm=1,
        sample_steps=np.arange(2, dtype=int),
        mode_method="z_index",
        kx_index=0,
        terms=LinearTerms(),
    )

    assert captured["grid"] is grid_full
    assert captured["g_shape"] == (1, 1, 1, 3, 2, 4)
    assert captured["ky_index"] == 1


def test_run_single_ky_preserves_single_ky_fallback_without_gx_contract(monkeypatch) -> None:
    grid_full = build_spectral_grid(
        GridConfig(
            Nx=4,
            Ny=6,
            Nz=4,
            Lx=10.0,
            Ly=20.0,
            boundary="periodic",
            y0=10.0,
        )
    )
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        imported_linear,
        "_build_imported_initial_condition",
        lambda **_: np.zeros((1, 1, 1, grid_full.ky.size, grid_full.kx.size, grid_full.z.size), dtype=np.complex64),
    )
    monkeypatch.setattr(imported_linear, "build_linear_cache", lambda *_args, **_kwargs: "cache")

    def _fake_integrate(**kwargs):
        captured["grid_ky"] = int(kwargs["grid"].ky.size)
        captured["g_shape"] = tuple(np.asarray(kwargs["G0"]).shape)
        captured["ky_index"] = kwargs["ky_index"]
        return tuple(np.zeros(2, dtype=float) for _ in range(5))

    monkeypatch.setattr(imported_linear, "_integrate_target_mode_series", _fake_integrate)

    _run_single_ky(
        ky_target=float(grid_full.ky[1]),
        geom=SimpleNamespace(),
        grid_full=grid_full,
        params=SimpleNamespace(),
        time_cfg=GXTimeConfig(dt=0.1, t_max=0.2, sample_stride=1, fixed_dt=True),
        gx_contract=None,
        species=(Species(charge=1.0, mass=1.0, density=1.0, temperature=1.0, tprim=0.0, fprim=0.0),),
        Nl=1,
        Nm=1,
        sample_steps=np.arange(2, dtype=int),
        mode_method="z_index",
        kx_index=0,
        terms=LinearTerms(),
    )

    assert captured["grid_ky"] == 1
    assert captured["g_shape"][3] == 1
    assert captured["ky_index"] == 0


def test_gx_kyst_fac_mask_cached_uses_positive_half_storage_on_full_ky_grid() -> None:
    cache = SimpleNamespace(
        ky=np.asarray([-0.2, 0.0, 0.2], dtype=np.float32),
        kx=np.asarray([0.0, 0.1], dtype=np.float32),
        dealias_mask=np.asarray([[1.0, 1.0], [1.0, 1.0], [1.0, 0.0]], dtype=np.float32),
    )
    fac = np.asarray(_gx_kyst_fac_mask_cached(cache, use_dealias=True), dtype=float)
    np.testing.assert_allclose(
        fac,
        np.asarray(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 0.0],
            ],
            dtype=float,
        ),
    )


def test_gx_Wg_by_ky_matches_gx_positive_ky_storage_contract() -> None:
    cache = SimpleNamespace(
        ky=np.asarray([-0.2, 0.0, 0.2], dtype=np.float32),
        kx=np.asarray([0.0], dtype=np.float32),
        dealias_mask=np.asarray([[1.0], [1.0], [1.0]], dtype=np.float32),
    )
    params = SimpleNamespace(density=1.0, temp=1.0)
    vol_fac = jnp.asarray([1.0], dtype=jnp.float32)
    G = jnp.ones((1, 1, 1, 3, 1, 1), dtype=jnp.complex64)
    Wg = np.asarray(_gx_Wg_by_ky(G, cache, params, vol_fac), dtype=float)
    assert np.allclose(Wg, np.asarray([0.0, 0.5, 1.0], dtype=float))


def test_select_geometry_source_prefers_gx_output_for_vmec_generated_runs() -> None:
    gx_out = Path("/tmp/run.out.nc")
    geom = Path("/tmp/run.eik.nc")
    vmec_contract = replace(_dummy_gx_contract(init_single=False), geo_option="vmec")
    desc_contract = replace(_dummy_gx_contract(init_single=False), geo_option="desc")
    nc_contract = replace(_dummy_gx_contract(init_single=False), geo_option="nc")
    assert _select_geometry_source(gx_out, geom, None) == geom
    assert _select_geometry_source(gx_out, geom, vmec_contract) == gx_out
    assert _select_geometry_source(gx_out, geom, desc_contract) == gx_out
    assert _select_geometry_source(gx_out, geom, nc_contract) == geom


def test_integrate_target_mode_series_collects_requested_sample_count(monkeypatch) -> None:
    monkeypatch.setattr(imported_linear.jax, "jit", lambda fn, donate_argnums=None: fn)
    monkeypatch.setattr(
        imported_linear,
        "assemble_rhs_cached",
        lambda *_args, **_kwargs: (
            None,
            SimpleNamespace(phi=jnp.zeros((2, 2, 3), dtype=jnp.complex64), apar=None),
        ),
    )
    monkeypatch.setattr(
        imported_linear,
        "_linear_explicit_step",
        lambda G_state, *_args, **_kwargs: (
            G_state,
            SimpleNamespace(phi=jnp.zeros((2, 2, 3), dtype=jnp.complex64), apar=None),
        ),
    )
    monkeypatch.setattr(
        imported_linear,
        "_gx_growth_rate_step",
        lambda *_args, **_kwargs: (
            jnp.ones((2, 2), dtype=jnp.float32),
            jnp.full((2, 2), 2.0, dtype=jnp.float32),
        ),
    )
    monkeypatch.setattr(imported_linear, "_gx_Wg_by_ky", lambda *_args, **_kwargs: jnp.asarray([0.0, 3.0]))
    monkeypatch.setattr(imported_linear, "_gx_Wphi_by_ky", lambda *_args, **_kwargs: jnp.asarray([0.0, 4.0]))
    monkeypatch.setattr(imported_linear, "_gx_Wapar_by_ky", lambda *_args, **_kwargs: jnp.asarray([0.0, 5.0]))

    gamma, omega, Wg, Wphi, Wapar = _integrate_target_mode_series(
        G0=jnp.zeros((1, 1, 1, 2, 2, 3), dtype=jnp.complex64),
        grid=SimpleNamespace(dealias_mask=np.ones((2, 2), dtype=bool), z=np.arange(3)),
        cache=SimpleNamespace(jacobian=jnp.ones(3, dtype=jnp.float32)),
        params=SimpleNamespace(),
        time_cfg=GXTimeConfig(dt=0.1, t_max=0.21, sample_stride=1, fixed_dt=True),
        terms=LinearTerms(),
        mode_method="z_index",
        ky_index=1,
        kx_index=0,
        sample_steps=np.arange(3, dtype=int),
    )

    np.testing.assert_allclose(gamma, np.ones(3, dtype=float))
    np.testing.assert_allclose(omega, np.full(3, 2.0, dtype=float))
    np.testing.assert_allclose(Wg, np.full(3, 3.0, dtype=float))
    np.testing.assert_allclose(Wphi, np.full(3, 4.0, dtype=float))
    np.testing.assert_allclose(Wapar, np.full(3, 5.0, dtype=float))


def test_write_scan_rows_checkpoints_sorted_csv(tmp_path: Path) -> None:
    out = tmp_path / "scan.csv"
    df = _write_scan_rows(
        [
            {"ky": 0.3, "mean_abs_gamma": 3.0},
            {"ky": 0.1, "mean_abs_gamma": 1.0},
        ],
        out,
    )
    assert list(df["ky"]) == [0.1, 0.3]
    saved = np.genfromtxt(out, delimiter=",", names=True)
    np.testing.assert_allclose(np.asarray(saved["ky"], dtype=float), np.asarray([0.1, 0.3], dtype=float))
