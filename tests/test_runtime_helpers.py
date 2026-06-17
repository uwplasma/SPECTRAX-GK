from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import spectraxgk.runtime as runtime
import spectraxgk.runtime_policies as runtime_policies
from spectraxgk.analysis import ModeSelection
from spectraxgk.runtime_diagnostics import fit_runtime_linear_diagnostics
from spectraxgk.runtime_orchestration import (
    build_runtime_progress_message,
    format_duration,
)
from spectraxgk.workflows.reduced_models import (
    CETGLinearRuntimeDeps,
    run_cetg_linear_runtime,
)
from spectraxgk.benchmarking import late_time_linear_metrics
from spectraxgk.config import (
    GeometryConfig,
    GridConfig,
    InitializationConfig,
    TimeConfig,
)
from spectraxgk.diagnostics import ResolvedDiagnostics, SimulationDiagnostics
from spectraxgk.grids import build_spectral_grid
from spectraxgk.runtime import (
    _build_initial_condition,
    _build_gaussian_profile,
    _concat_runtime_diagnostics,
    _enforce_full_ky_hermitian,
    _expand_ky,
    _centered_glibc_random_pairs,
    _default_hermite_hypercollision_exponent,
    _dealiased_initial_mode_pairs,
    _periodic_zp_from_grid,
    _infer_runtime_nonlinear_steps,
    _load_initial_state_from_file,
    _midplane_index,
    _normalize_linear_solver_name,
    _require_full_gk_runtime_model,
    _resolve_runtime_hl_dims,
    _reshape_netcdf_state,
    _runtime_external_phi,
    _runtime_default_krylov_config,
    _runtime_model_key,
    _select_nonlinear_mode_indices,
    _slice_runtime_diagnostics,
    _species_to_linear,
    _stride_runtime_diagnostics,
    _truncate_runtime_diagnostics,
    _zero_kx_index,
    _run_runtime_scan_batch,
    build_runtime_geometry,
    build_runtime_linear_params,
    build_runtime_linear_terms,
    build_runtime_term_config,
    run_runtime_linear,
    run_runtime_nonlinear,
    run_runtime_scan,
)
from spectraxgk.runtime_config import (
    RuntimeConfig,
    RuntimeExpertConfig,
    RuntimeNormalizationConfig,
    RuntimeParallelConfig,
    RuntimePhysicsConfig,
    RuntimeSpeciesConfig,
)
from spectraxgk.terms.config import FieldState


def _base_cfg() -> RuntimeConfig:
    return RuntimeConfig(
        grid=GridConfig(Nx=4, Ny=6, Nz=8, Lx=6.28, Ly=6.28, boundary="periodic"),
        time=TimeConfig(
            t_max=0.4, dt=0.1, method="rk2", use_diffrax=False, sample_stride=1
        ),
        geometry=GeometryConfig(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778),
        init=InitializationConfig(
            init_field="density", init_amp=1.0e-8, gaussian_init=False
        ),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(contract="cyclone"),
    )


def _diag(offset: float = 0.0, *, resolved: bool = True) -> SimulationDiagnostics:
    res = None
    if resolved:
        res = ResolvedDiagnostics(
            Phi2_kxt=np.ones((2, 4), dtype=float) + offset,
            Wg_kxst=np.ones((2, 1, 4), dtype=float) + offset,
        )
    return SimulationDiagnostics(
        t=np.asarray([0.1, 0.2]) + offset,
        dt_t=np.asarray([0.1, 0.1]),
        dt_mean=np.asarray(0.1),
        gamma_t=np.asarray([0.01, 0.02]) + offset,
        omega_t=np.asarray([0.03, 0.04]) + offset,
        Wg_t=np.asarray([1.0, 1.1]) + offset,
        Wphi_t=np.asarray([2.0, 2.1]) + offset,
        Wapar_t=np.asarray([0.5, 0.6]) + offset,
        heat_flux_t=np.asarray([3.0, 3.1]) + offset,
        particle_flux_t=np.asarray([4.0, 4.1]) + offset,
        energy_t=np.asarray([3.5, 3.8]) + offset,
        heat_flux_species_t=np.asarray([[3.0], [3.1]]) + offset,
        particle_flux_species_t=np.asarray([[4.0], [4.1]]) + offset,
        turbulent_heating_t=np.asarray([5.0, 5.1]) + offset,
        turbulent_heating_species_t=np.asarray([[5.0], [5.1]]) + offset,
        phi_mode_t=np.asarray([1.0 + 0.0j, 1.1 + 0.1j]),
        resolved=res,
    )


def test_runtime_linear_terms_disable_zero_collision_frequency() -> None:
    cfg_zero = replace(
        _base_cfg(),
        species=(RuntimeSpeciesConfig(name="ion", nu=0.0),),
        physics=RuntimePhysicsConfig(collisions=True, hypercollisions=False),
    )
    cfg_nonzero = replace(
        cfg_zero,
        species=(RuntimeSpeciesConfig(name="ion", nu=0.05),),
    )

    assert build_runtime_linear_terms(cfg_zero).collisions == 0.0
    assert build_runtime_linear_terms(cfg_nonzero).collisions == 1.0


def test_runtime_small_helper_functions() -> None:
    cfg = _base_cfg()
    grid = build_spectral_grid(cfg.grid)

    assert _normalize_linear_solver_name(" explicit_time ") == "explicit_time"
    assert _normalize_linear_solver_name("krylov") == "krylov"
    assert _midplane_index(grid) == min(grid.z.size // 2 + 1, grid.z.size - 1)
    assert _midplane_index(type("Grid", (), {"z": np.asarray([0.0])})()) == 0
    assert _zero_kx_index(grid) == int(np.argmin(np.abs(np.asarray(grid.kx))))
    assert _dealiased_initial_mode_pairs(grid)[0] == (0, 1)
    assert _periodic_zp_from_grid(np.asarray([0.0])) == 1.0
    assert _periodic_zp_from_grid(np.asarray([0.0, 0.0])) == 1.0
    assert _default_hermite_hypercollision_exponent(None) == 20.0
    assert _default_hermite_hypercollision_exponent(3) == 1.0
    assert _default_hermite_hypercollision_exponent(40) == 20.0
    assert _runtime_model_key(cfg) == "gyrokinetic"


def test_runtime_policy_helpers_preserve_legacy_runtime_exports() -> None:
    for name in runtime_policies.__all__:
        assert getattr(runtime, name) is getattr(runtime_policies, name)


def test_runtime_independent_parallel_plan_resolves_config_and_arguments() -> None:
    cfg = replace(
        _base_cfg(),
        parallel=RuntimeParallelConfig(
            strategy="batch", axis="ky", num_devices=4, backend="process"
        ),
    )

    plan = runtime_policies._runtime_independent_parallel_plan(
        cfg, problem_size=3, workers=1, executor="thread"
    )

    assert plan.requested_workers == 4
    assert plan.effective_workers == 3
    assert plan.executor == "process"
    assert plan.source == "runtime_config"
    assert plan.enabled is True
    assert plan.to_dict()["enabled"] is True

    explicit = runtime_policies._runtime_independent_parallel_plan(
        cfg, problem_size=3, workers=2, executor="threads"
    )

    assert explicit.requested_workers == 2
    assert explicit.executor == "thread"
    assert explicit.source == "arguments"


def test_runtime_independent_parallel_plan_rejects_invalid_policy() -> None:
    cfg_bad_backend = replace(
        _base_cfg(),
        parallel=RuntimeParallelConfig(
            strategy="batch", axis="ky", num_devices=2, backend="mpi"
        ),
    )
    cfg_bad_axis = replace(
        _base_cfg(),
        parallel=RuntimeParallelConfig(strategy="batch", axis="kx", num_devices=2),
    )

    with pytest.raises(ValueError, match="workers"):
        runtime_policies._runtime_independent_parallel_plan(
            _base_cfg(), problem_size=1, workers=0, executor="thread"
        )
    with pytest.raises(ValueError, match="backend"):
        runtime_policies._runtime_independent_parallel_plan(
            cfg_bad_backend, problem_size=2, workers=1, executor="thread"
        )
    with pytest.raises(ValueError, match="axis='ky'"):
        runtime_policies._runtime_independent_parallel_plan(
            cfg_bad_axis, problem_size=2, workers=1, executor="thread"
        )


def test_runtime_orchestration_progress_policy() -> None:
    message, snapshot = build_runtime_progress_message(
        label="nonlinear",
        chunk_index=3,
        t_elapsed=2.0,
        t_max=4.0,
        chunk_wall_seconds=61.0,
        elapsed_seconds=180.0,
    )

    assert format_duration(3661.0) == "1:01:01"
    assert snapshot.progress == pytest.approx(0.5)
    assert snapshot.eta_seconds == pytest.approx(180.0)
    assert "completed nonlinear chunk 3" in message
    assert "progress= 50.0%" in message
    assert "chunk_wall=01:01" in message
    assert "elapsed=03:00" in message
    assert "eta=03:00" in message


def test_runtime_random_pair_edge_cases() -> None:
    empty = _centered_glibc_random_pairs(3, 0)
    assert empty.shape == (0, 2)

    seed_zero = _centered_glibc_random_pairs(0, 3)
    seed_one = _centered_glibc_random_pairs(1, 3)
    np.testing.assert_allclose(seed_zero, seed_one)


def test_runtime_mode_index_selection_and_step_inference() -> None:
    cfg = _base_cfg()
    grid = build_spectral_grid(cfg.grid)
    ky_idx, kx_idx = _select_nonlinear_mode_indices(
        grid, ky_target=0.2, kx_target=None, use_dealias_mask=False
    )
    assert 0 <= ky_idx < grid.ky.size
    assert 0 <= kx_idx < grid.kx.size

    empty_mask_grid = type(
        "Grid",
        (),
        {
            "ky": np.asarray([0.0, 0.2, 0.4]),
            "kx": np.asarray([-0.5, 0.0, 0.5]),
            "dealias_mask": np.zeros((3, 3), dtype=bool),
        },
    )()
    ky_idx2, kx_idx2 = _select_nonlinear_mode_indices(
        empty_mask_grid, ky_target=0.4, kx_target=0.5, use_dealias_mask=True
    )
    assert (ky_idx2, kx_idx2) == (2, 2)

    dealiased_grid = type(
        "Grid",
        (),
        {
            "ky": np.asarray([0.0, 0.2, 0.4]),
            "kx": np.asarray([0.0, 0.5, 1.0]),
            "dealias_mask": np.asarray(
                [
                    [True, True, True],
                    [True, False, True],
                    [False, False, False],
                ],
                dtype=bool,
            ),
        },
    )()
    ky_idx3, kx_idx3 = _select_nonlinear_mode_indices(
        dealiased_grid, ky_target=0.39, kx_target=0.9, use_dealias_mask=True
    )
    assert (ky_idx3, kx_idx3) == (1, 2)

    assert _infer_runtime_nonlinear_steps(cfg, dt=0.1, steps=7) == 7
    assert (
        _infer_runtime_nonlinear_steps(
            replace(cfg, time=replace(cfg.time, fixed_dt=True)), dt=0.05, steps=None
        )
        == 4
    )
    adaptive_cfg = replace(cfg, time=replace(cfg.time, fixed_dt=False, dt_max=None))
    assert _infer_runtime_nonlinear_steps(adaptive_cfg, dt=0.2, steps=None) == 2
    with pytest.raises(ValueError):
        _infer_runtime_nonlinear_steps(
            replace(cfg, time=replace(cfg.time, t_max=0.0)), dt=0.1, steps=0
        )


def test_runtime_nonlinear_diagnostics_kwargs_policy() -> None:
    base = _base_cfg()
    cfg = replace(
        base,
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, nonlinear=True),
        time=replace(
            base.time,
            method="rk4",
            nonlinear_dealias=True,
            collision_split=True,
            collision_scheme="exp",
            implicit_restart=7,
            implicit_solve_method="gmres",
            implicit_preconditioner="jacobi",
            cfl_fac=None,
        ),
    )

    kwargs = runtime_policies.build_runtime_nonlinear_diagnostics_kwargs(
        cfg,
        dt=0.05,
        steps=9,
        method=None,
        term_config="terms",
        sample_stride=2,
        diagnostics_stride=3,
        laguerre_mode="grid",
        ky_index=1,
        kx_index=2,
        fixed_dt=False,
        fixed_mode_ky_index=4,
        fixed_mode_kx_index=5,
        external_phi=0.25,
        resolved_diagnostics=False,
        show_progress=True,
    )

    assert kwargs["dt"] == pytest.approx(0.05)
    assert kwargs["steps"] == 9
    assert kwargs["method"] == "rk4"
    assert kwargs["terms"] == "terms"
    assert kwargs["sample_stride"] == 2
    assert kwargs["diagnostics_stride"] == 3
    assert kwargs["use_dealias_mask"] is True
    assert kwargs["laguerre_mode"] == "grid"
    assert kwargs["omega_ky_index"] == 1
    assert kwargs["omega_kx_index"] == 2
    assert kwargs["fixed_dt"] is False
    assert kwargs["collision_split"] is True
    assert kwargs["collision_scheme"] == "exp"
    assert kwargs["implicit_restart"] == 7
    assert kwargs["implicit_solve_method"] == "gmres"
    assert kwargs["implicit_preconditioner"] == "jacobi"
    assert kwargs["fixed_mode_ky_index"] == 4
    assert kwargs["fixed_mode_kx_index"] == 5
    assert kwargs["external_phi"] == pytest.approx(0.25)
    assert kwargs["resolved_diagnostics"] is False
    assert kwargs["show_progress"] is True


def test_runtime_diagnostic_slice_stride_truncate_concat() -> None:
    diag = _diag()
    sliced = _slice_runtime_diagnostics(diag, 1)
    assert sliced.t.shape == (1,)
    assert sliced.resolved is not None and sliced.resolved.Phi2_kxt.shape[0] == 1
    zero = _slice_runtime_diagnostics(diag, 0)
    assert float(zero.dt_mean) == 0.0
    with pytest.raises(ValueError):
        _slice_runtime_diagnostics(diag, -1)

    truncated = _truncate_runtime_diagnostics(diag, t_max=0.15)
    assert truncated.t.shape == (2,)
    empty = _truncate_runtime_diagnostics(replace(diag, t=np.asarray([])), t_max=1.0)
    assert empty is not None

    strided = _stride_runtime_diagnostics(diag, stride=2)
    assert strided.t.shape == (1,)
    assert _stride_runtime_diagnostics(diag, stride=1) is diag

    concat = _concat_runtime_diagnostics([diag, _diag(offset=1.0)])
    assert concat.t.shape == (4,)
    assert concat.resolved is not None and concat.resolved.Phi2_kxt.shape[0] == 4
    concat_none = _concat_runtime_diagnostics(
        [replace(diag, resolved=None), replace(_diag(offset=1.0), resolved=None)]
    )
    assert concat_none.resolved is None
    with pytest.raises(ValueError):
        _concat_runtime_diagnostics([])


def test_fit_runtime_linear_diagnostics_density_fit_contract() -> None:
    t = np.asarray([0.1, 0.2, 0.3, 0.4])
    phi = np.ones((4, 1, 1, 2), dtype=np.complex128)
    density = np.asarray([1.0, 1.5, 2.25, 3.375], dtype=np.complex128)[
        :, None, None, None
    ] * np.ones((1, 1, 1, 2), dtype=np.complex128)
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=0)

    out = fit_runtime_linear_diagnostics(
        t=t,
        phi_t=phi,
        density_t=density,
        selection=sel,
        z=np.asarray([-1.0, 1.0]),
        fit_signal="density",
        mode_method="z_index",
        auto_window=True,
        tmin=None,
        tmax=None,
        window_fraction=1.0,
        min_points=3,
        start_fraction=0.0,
        growth_weight=0.0,
        require_positive=True,
        min_amp_fraction=0.0,
    )

    assert out.fit_signal_used == "density"
    assert out.gamma > 0.0
    assert out.fit_window_tmin is not None
    assert out.fit_window_tmax is not None
    np.testing.assert_allclose(out.signal, density[:, 0, 0, 0])


def test_run_cetg_linear_runtime_dependency_contract() -> None:
    cfg = replace(_base_cfg(), physics=replace(_base_cfg().physics, reduced_model="cetg"))
    grid = build_spectral_grid(cfg.grid)
    statuses: list[str] = []

    def _fake_integrator(*_args, **_kwargs):
        diag = SimpleNamespace(
            t=np.asarray([0.1, 0.2, 0.3]),
            phi_mode_t=np.asarray([1.0 + 0.0j, 1.1 + 0.1j, 1.4 + 0.2j]),
            gamma_t=np.asarray([0.1, 0.2, 0.3]),
            omega_t=np.asarray([-0.1, -0.2, -0.3]),
        )
        return (
            np.asarray(diag.t),
            diag,
            np.ones((2, 1, 1, 1, grid.z.size), dtype=np.complex64),
            object(),
        )

    out = run_cetg_linear_runtime(
        cfg,
        deps=CETGLinearRuntimeDeps(
            build_runtime_geometry=build_runtime_geometry,
            validate_cetg_runtime_config=lambda *_args, **_kwargs: None,
            build_initial_condition=lambda *_args, **_kwargs: np.zeros(
                (2, 1, 1, 1, grid.z.size), dtype=np.complex64
            ),
            build_runtime_term_config=lambda _cfg: object(),
            build_cetg_model_params=lambda *_args, **_kwargs: object(),
            integrate_cetg_explicit_diagnostics_state=_fake_integrator,
            fit_growth_rate_auto=lambda *_args, **_kwargs: (0.2, -0.3, 0.1, 0.3),
            fit_growth_rate=lambda *_args, **_kwargs: (0.2, -0.3),
        ),
        ky_target=0.1,
        Nl=2,
        Nm=1,
        solver="time",
        method=None,
        dt=0.1,
        steps=3,
        sample_stride=None,
        auto_window=True,
        tmin=None,
        tmax=None,
        window_fraction=1.0,
        min_points=3,
        start_fraction=0.0,
        growth_weight=0.0,
        require_positive=True,
        min_amp_fraction=0.0,
        return_state=True,
        status_callback=statuses.append,
    )

    assert out.gamma == pytest.approx(0.2)
    assert out.omega == pytest.approx(-0.3)
    assert out.state is not None
    assert out.fit_signal_used == "phi"
    assert any("running cETG time integration" in msg for msg in statuses)


def test_runtime_diagnostic_concat_rejects_misaligned_optional_channels() -> None:
    """Optional species channels must stay aligned with the common time axis."""

    diag0 = replace(_diag(resolved=False), heat_flux_species_t=None)
    diag1 = _diag(offset=1.0, resolved=False)

    with pytest.raises(ValueError, match="optional diagnostic heat_flux_species_t"):
        _concat_runtime_diagnostics([diag0, diag1])

    with pytest.raises(ValueError, match="resolved diagnostics"):
        _concat_runtime_diagnostics(
            [replace(_diag(), resolved=None), _diag(offset=1.0)]
        )

    partial0 = replace(
        _diag(),
        resolved=ResolvedDiagnostics(
            Phi2_kxt=np.ones((2, 4), dtype=float),
            Wg_kxst=None,
        ),
    )
    partial1 = replace(
        _diag(offset=1.0),
        resolved=ResolvedDiagnostics(
            Phi2_kxt=np.ones((2, 4), dtype=float),
            Wg_kxst=np.ones((2, 1, 4), dtype=float),
        ),
    )
    with pytest.raises(ValueError, match="resolved diagnostic Wg_kxst"):
        _concat_runtime_diagnostics([partial0, partial1])


def test_runtime_species_and_model_helpers() -> None:
    cfg = _base_cfg()
    species = _species_to_linear(cfg.species)
    assert len(species) == 1
    with pytest.raises(ValueError):
        _species_to_linear((RuntimeSpeciesConfig(name="adiabatic", kinetic=False),))

    etg_cfg = replace(
        cfg,
        species=(RuntimeSpeciesConfig(name="electron", charge=-1.0, kinetic=True),),
        normalization=RuntimeNormalizationConfig(contract="etg"),
        physics=RuntimePhysicsConfig(
            adiabatic_electrons=False,
            adiabatic_ions=True,
            electrostatic=True,
            electromagnetic=False,
        ),
    )
    krylov = _runtime_default_krylov_config(etg_cfg)
    assert krylov.method == "shift_invert"
    assert krylov.mode_family == "etg"
    assert _runtime_default_krylov_config(cfg).method != "shift_invert"

    assert _resolve_runtime_hl_dims(cfg, Nl=None, Nm=None) == (24, 12)
    cetg_cfg = replace(cfg, physics=replace(cfg.physics, reduced_model="cetg"))
    assert _resolve_runtime_hl_dims(cetg_cfg, Nl=2, Nm=1) == (2, 1)
    with pytest.raises(ValueError):
        _resolve_runtime_hl_dims(cetg_cfg, Nl=3, Nm=1)
    with pytest.raises(NotImplementedError):
        _resolve_runtime_hl_dims(
            replace(cfg, physics=replace(cfg.physics, reduced_model="krehm")),
            Nl=None,
            Nm=None,
        )
    with pytest.raises(ValueError):
        _resolve_runtime_hl_dims(
            replace(cfg, physics=replace(cfg.physics, reduced_model="mystery")),
            Nl=None,
            Nm=None,
        )

    _require_full_gk_runtime_model(cfg)
    with pytest.raises(NotImplementedError):
        _require_full_gk_runtime_model(cetg_cfg)
    with pytest.raises(NotImplementedError):
        _require_full_gk_runtime_model(
            replace(cfg, physics=replace(cfg.physics, reduced_model="krehm"))
        )
    with pytest.raises(ValueError):
        _require_full_gk_runtime_model(
            replace(cfg, physics=replace(cfg.physics, reduced_model="mystery"))
        )


def test_runtime_wrapper_patch_surfaces(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _base_cfg()
    captured: dict[str, object] = {}
    geom = object()

    def _fake_build_geom(_cfg):
        captured["geom_called"] = True
        return geom

    def _fake_build_params(_cfg, *, Nm, geom):
        captured["params"] = {"Nm": Nm, "geom": geom}
        return "params"

    def _fake_build_terms(_cfg):
        captured["terms"] = _cfg
        return "terms"

    def _fake_build_term_config(_cfg):
        captured["term_cfg"] = _cfg
        return "term_cfg"

    monkeypatch.setattr("spectraxgk.runtime.build_runtime_geometry", _fake_build_geom)
    monkeypatch.setattr(
        "spectraxgk.runtime_startup.build_runtime_linear_params", _fake_build_params
    )
    monkeypatch.setattr(
        "spectraxgk.runtime_startup.build_runtime_linear_terms", _fake_build_terms
    )
    monkeypatch.setattr(
        "spectraxgk.runtime_startup.build_runtime_term_config", _fake_build_term_config
    )

    assert build_runtime_linear_params(cfg, Nm=7) == "params"
    assert captured["geom_called"] is True
    assert captured["params"] == {"Nm": 7, "geom": geom}

    captured.clear()
    explicit_geom = object()
    assert build_runtime_linear_params(cfg, Nm=5, geom=explicit_geom) == "params"
    assert "geom_called" not in captured
    assert captured["params"] == {"Nm": 5, "geom": explicit_geom}

    assert build_runtime_linear_terms(cfg) == "terms"
    assert captured["terms"] is cfg
    assert build_runtime_term_config(cfg) == "term_cfg"
    assert captured["term_cfg"] is cfg


def test_runtime_external_phi_helper() -> None:
    cfg = _base_cfg()

    assert _runtime_external_phi(cfg) is None
    assert (
        _runtime_external_phi(
            replace(cfg, expert=RuntimeExpertConfig(source=" default "))
        )
        is None
    )
    assert _runtime_external_phi(
        replace(cfg, expert=RuntimeExpertConfig(source="phiext_full", phi_ext=0.375))
    ) == pytest.approx(0.375)

    with pytest.raises(ValueError, match="unsupported expert.source"):
        _runtime_external_phi(
            replace(cfg, expert=RuntimeExpertConfig(source="bad_source", phi_ext=1.0))
        )


def test_runtime_build_geometry_vmec_and_miller_branches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = _base_cfg()
    captured: list[tuple[str, str | None]] = []

    def _fake_build(geom_cfg):
        captured.append((geom_cfg.model, geom_cfg.geometry_file))
        return geom_cfg

    vmec_path = tmp_path / "vmec.eik.nc"
    miller_path = tmp_path / "miller.eik.nc"
    vmec_path.write_bytes(b"x")
    miller_path.write_bytes(b"x")

    monkeypatch.setattr("spectraxgk.runtime.build_flux_tube_geometry", _fake_build)
    monkeypatch.setattr(
        "spectraxgk.runtime.generate_runtime_vmec_eik", lambda _cfg: vmec_path
    )
    monkeypatch.setattr(
        "spectraxgk.runtime.generate_runtime_miller_eik", lambda _cfg: miller_path
    )

    build_runtime_geometry(replace(cfg, geometry=GeometryConfig(model="vmec")))
    build_runtime_geometry(replace(cfg, geometry=GeometryConfig(model="miller")))
    build_runtime_geometry(cfg)
    assert captured[0] == ("vmec-eik", str(vmec_path))
    assert captured[1] == ("imported-eik", str(miller_path))
    assert captured[2][0] == cfg.geometry.model


def test_runtime_initial_state_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    z = np.linspace(-1.0, 1.0, 5)
    profile = _build_gaussian_profile(
        z,
        kx=0.2,
        ky=0.1,
        s_hat=0.8,
        width=0.5,
        envelope_constant=1.0,
        envelope_sine=0.2,
    )
    assert profile.shape == z.shape
    assert np.allclose(
        _build_gaussian_profile(
            z,
            kx=0.2,
            ky=0.0,
            s_hat=0.8,
            width=0.5,
            envelope_constant=1.0,
            envelope_sine=0.2,
        ),
        np.zeros_like(z),
    )

    raw = np.arange(2 * 3 * 2 * 4 * 5, dtype=np.float32).astype(np.complex64)
    reshaped = _reshape_netcdf_state(raw, nspec=1, nl=2, nm=3, nyc=2, nx=4, nz=5)
    assert reshaped.shape == (1, 2, 3, 2, 4, 5)

    expanded = _expand_ky(np.ones((1, 2, 3, 4, 5), dtype=np.complex64), nyc=3)
    assert expanded.shape[-3] == 4
    assert (
        _expand_ky(np.ones((1, 2, 3, 4, 5), dtype=np.complex64), nyc=2).shape[-3] == 3
    )


def test_runtime_single_mode_init_populates_zonal_ky0_branch() -> None:
    cfg = replace(
        _base_cfg(),
        grid=GridConfig(Nx=6, Ny=8, Nz=8, Lx=6.28, Ly=6.28, boundary="periodic"),
        init=InitializationConfig(
            init_field="density",
            init_amp=1.0,
            gaussian_init=False,
            init_single=True,
        ),
    )
    geom = build_runtime_geometry(cfg)
    grid = build_spectral_grid(cfg.grid)
    g0 = np.asarray(
        _build_initial_condition(
            grid, geom, cfg, ky_index=0, kx_index=1, Nl=1, Nm=1, nspecies=1
        )
    )

    assert np.max(np.abs(g0)) > 0.0
    assert np.max(np.abs(g0[0, 0, 0, 0, 1, :])) > 0.0


def test_runtime_gaussian_single_mode_init_populates_zonal_ky0_branch() -> None:
    cfg = replace(
        _base_cfg(),
        grid=GridConfig(Nx=6, Ny=8, Nz=8, Lx=6.28, Ly=6.28, boundary="periodic"),
        init=InitializationConfig(
            init_field="density",
            init_amp=1.0,
            gaussian_init=True,
            init_single=True,
            gaussian_width=0.35,
        ),
    )
    geom = build_runtime_geometry(cfg)
    grid = build_spectral_grid(cfg.grid)
    g0 = np.asarray(
        _build_initial_condition(
            grid, geom, cfg, ky_index=0, kx_index=1, Nl=1, Nm=1, nspecies=1
        )
    )

    assert np.max(np.abs(g0)) > 0.0
    assert np.max(np.abs(g0[0, 0, 0, 0, 1, :])) > 0.0


def test_runtime_initial_state_loading_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    full = np.zeros((1, 1, 1, 4, 4, 2), dtype=np.complex64)
    full[..., 1, :, :] = 1.0 + 2.0j
    herm = _enforce_full_ky_hermitian(full)
    assert herm.shape == full.shape
    assert np.allclose(
        _enforce_full_ky_hermitian(np.ones((1, 1, 1, 1, 2), dtype=np.complex64)),
        np.ones((1, 1, 1, 1, 2), dtype=np.complex64),
    )

    nc_path = tmp_path / "restart.nc"
    monkeypatch.setattr(
        "spectraxgk.runtime.load_netcdf_restart_state",
        lambda *_args, **_kwargs: np.ones((1, 2, 3, 4, 4, 5), dtype=np.complex64),
    )
    assert _load_initial_state_from_file(
        nc_path, nspecies=1, Nl=2, Nm=3, ny=4, nx=4, nz=5
    ).shape == (1, 2, 3, 4, 4, 5)

    ny = 4
    nx = 4
    nz = 5
    nyc = ny // 2 + 1
    nyc_raw = np.ones(1 * 2 * 3 * nyc * nx * nz, dtype=np.complex64)
    nyc_path = tmp_path / "restart.bin"
    nyc_raw.tofile(nyc_path)
    assert _load_initial_state_from_file(
        nyc_path, nspecies=1, Nl=2, Nm=3, ny=ny, nx=nx, nz=nz
    ).shape == (1, 2, 3, 4, 4, 5)

    full_raw = np.ones(1 * 2 * 3 * ny * nx * nz, dtype=np.complex64)
    full_path = tmp_path / "restart_full.bin"
    full_raw.tofile(full_path)
    assert _load_initial_state_from_file(
        full_path, nspecies=1, Nl=2, Nm=3, ny=ny, nx=nx, nz=nz
    ).shape == (1, 2, 3, 4, 4, 5)

    bad_path = tmp_path / "restart_bad.bin"
    np.ones(7, dtype=np.complex64).tofile(bad_path)
    with pytest.raises(ValueError):
        _load_initial_state_from_file(
            bad_path, nspecies=1, Nl=2, Nm=3, ny=ny, nx=nx, nz=nz
        )


def test_runtime_initial_condition_validation_branches() -> None:
    cfg = _base_cfg()
    geom = build_runtime_geometry(cfg)
    grid = build_spectral_grid(cfg.grid)

    with pytest.raises(ValueError):
        _build_initial_condition(
            grid,
            geom,
            replace(cfg, init=replace(cfg.init, gaussian_width=0.0)),
            ky_index=1,
            kx_index=0,
            Nl=2,
            Nm=2,
            nspecies=1,
        )

    with pytest.raises(ValueError):
        _build_initial_condition(
            grid,
            geom,
            replace(cfg, init=replace(cfg.init, init_file_mode="bad")),
            ky_index=1,
            kx_index=0,
            Nl=2,
            Nm=2,
            nspecies=1,
        )

    with pytest.raises(ValueError):
        _build_initial_condition(
            grid,
            geom,
            replace(cfg, init=replace(cfg.init, init_field="bad")),
            ky_index=1,
            kx_index=0,
            Nl=2,
            Nm=2,
            nspecies=1,
        )


def test_run_runtime_scan_batch_validation_and_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _base_cfg()
    with pytest.raises(ValueError):
        _run_runtime_scan_batch(
            cfg,
            np.asarray([], dtype=float),
            Nl=2,
            Nm=3,
            method="rk2",
            dt=0.1,
            steps=2,
            sample_stride=1,
            auto_window=True,
            tmin=None,
            tmax=None,
            window_fraction=0.4,
            min_points=2,
            start_fraction=0.0,
            growth_weight=0.0,
            require_positive=False,
            min_amp_fraction=0.0,
            mode_method="project",
            fit_signal="phi",
            show_progress=False,
        )

    grid = build_spectral_grid(cfg.grid)
    geom = object()
    params = type("Params", (), {"rho_star": np.asarray(1.0)})()
    monkeypatch.setattr("spectraxgk.runtime.build_runtime_geometry", lambda _cfg: geom)
    monkeypatch.setattr(
        "spectraxgk.runtime.apply_geometry_grid_defaults",
        lambda _geom, grid_cfg: grid_cfg,
    )
    monkeypatch.setattr("spectraxgk.runtime.build_spectral_grid", lambda _cfg: grid)
    monkeypatch.setattr(
        "spectraxgk.runtime.build_runtime_linear_params",
        lambda *_args, **_kwargs: params,
    )
    monkeypatch.setattr(
        "spectraxgk.runtime.build_runtime_linear_terms", lambda _cfg: object()
    )
    monkeypatch.setattr(
        "spectraxgk.runtime._build_initial_condition",
        lambda *_args, **_kwargs: np.ones(
            (1, 2, 3, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.runtime.integrate_linear_diagnostics",
        lambda *_args, **_kwargs: (
            None,
            np.ones((3, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64),
            2.0
            * np.ones((3, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64),
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.runtime.extract_mode_time_series",
        lambda arr, sel, method="project": np.asarray(
            arr[:, sel.ky_index, sel.kx_index, 0]
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.runtime.fit_growth_rate_auto_with_stats",
        lambda t, signal, **kwargs: (
            0.2,
            0.3,
            0.0,
            0.2,
            2.0 if np.max(np.abs(signal)) < 1.5 else 1.0,
            0.0,
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.runtime.fit_growth_rate_auto",
        lambda *args, **kwargs: (0.4, 0.5, 0.0, 0.2),
    )
    monkeypatch.setattr(
        "spectraxgk.runtime.fit_growth_rate", lambda *args, **kwargs: (0.6, 0.7)
    )
    monkeypatch.setattr(
        "spectraxgk.runtime.apply_diagnostic_normalization",
        lambda g, o, **kwargs: (g, o),
    )

    scan_auto = _run_runtime_scan_batch(
        cfg,
        np.asarray([0.1, 0.2], dtype=float),
        Nl=2,
        Nm=3,
        method="rk2",
        dt=0.1,
        steps=2,
        sample_stride=1,
        auto_window=True,
        tmin=None,
        tmax=None,
        window_fraction=0.4,
        min_points=2,
        start_fraction=0.0,
        growth_weight=0.0,
        require_positive=False,
        min_amp_fraction=0.0,
        mode_method="project",
        fit_signal="auto",
        show_progress=False,
    )
    assert scan_auto.gamma.shape == (2,)

    scan_density = _run_runtime_scan_batch(
        cfg,
        np.asarray([0.1], dtype=float),
        Nl=2,
        Nm=3,
        method="rk2",
        dt=0.1,
        steps=2,
        sample_stride=1,
        auto_window=False,
        tmin=0.0,
        tmax=0.2,
        window_fraction=0.4,
        min_points=2,
        start_fraction=0.0,
        growth_weight=0.0,
        require_positive=False,
        min_amp_fraction=0.0,
        mode_method="project",
        fit_signal="density",
        show_progress=False,
    )
    assert np.allclose(scan_density.gamma, np.array([0.6]))

    with pytest.raises(ValueError):
        _run_runtime_scan_batch(
            cfg,
            np.asarray([0.1], dtype=float),
            Nl=2,
            Nm=3,
            method="rk2",
            dt=0.1,
            steps=2,
            sample_stride=1,
            auto_window=True,
            tmin=None,
            tmax=None,
            window_fraction=0.4,
            min_points=2,
            start_fraction=0.0,
            growth_weight=0.0,
            require_positive=False,
            min_amp_fraction=0.0,
            mode_method="project",
            fit_signal="invalid",
            show_progress=False,
        )


def test_run_runtime_scan_default_parallel_config_keeps_serial_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import spectraxgk.runtime as runtime

    cfg = _base_cfg()
    calls: list[float] = []

    def _unexpected_batch(*_args, **_kwargs):
        raise AssertionError(
            "default runtime parallel config should not use the combined-ky batch path"
        )

    def _fake_run_runtime_linear(_cfg, **kwargs):
        ky = float(kwargs["ky_target"])
        calls.append(ky)
        return SimpleNamespace(gamma=10.0 + ky, omega=-(20.0 + ky), quasilinear=None)

    monkeypatch.setattr(runtime, "_run_runtime_scan_batch", _unexpected_batch)
    monkeypatch.setattr(runtime, "run_runtime_linear", _fake_run_runtime_linear)

    result = run_runtime_scan(
        cfg,
        [0.3, 0.1],
        solver="time",
        workers=1,
        parallel_executor="thread",
        show_progress=False,
    )

    np.testing.assert_allclose(calls, [0.3, 0.1])
    np.testing.assert_allclose(result.ky, [0.3, 0.1])
    np.testing.assert_allclose(result.gamma, [10.3, 10.1])
    np.testing.assert_allclose(result.omega, [-20.3, -20.1])
    assert result.parallel is not None
    assert result.parallel["requested_workers"] == 1
    assert result.parallel["effective_workers"] == 1
    assert result.parallel["executor"] == "thread"
    assert "serial ky ordering" in result.parallel["identity_contract"]
    assert result.parallel["quasilinear_state_extraction"] is False


def test_run_runtime_scan_collects_quasilinear_payloads_and_worker_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import spectraxgk.runtime as runtime

    cfg = _base_cfg()
    calls: list[float] = []

    def _fake_run_runtime_linear(_cfg, **kwargs):
        ky = float(kwargs["ky_target"])
        calls.append(ky)
        return SimpleNamespace(
            gamma=1.0 + ky,
            omega=-(2.0 + ky),
            quasilinear={
                "ky": ky,
                "heat_flux_weight_total": 10.0 * ky,
                "claim_level": "bounded_unit_contract",
            },
        )

    monkeypatch.setattr(runtime, "run_runtime_linear", _fake_run_runtime_linear)

    result = run_runtime_scan(
        cfg,
        [0.4, 0.2, 0.1],
        solver="time",
        workers=8,
        parallel_executor="thread",
        show_progress=False,
    )

    np.testing.assert_allclose(calls, [0.4, 0.2, 0.1])
    np.testing.assert_allclose(result.gamma, [1.4, 1.2, 1.1])
    assert result.quasilinear is not None
    assert [payload["ky"] for payload in result.quasilinear] == [0.4, 0.2, 0.1]
    assert result.parallel is not None
    assert result.parallel["requested_workers"] == 8
    assert result.parallel["effective_workers"] == 3
    assert result.parallel["quasilinear_state_extraction"] is True


def test_run_runtime_scan_parallel_config_requests_combined_ky_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import spectraxgk.runtime as runtime

    cfg = replace(
        _base_cfg(),
        parallel=RuntimeParallelConfig(strategy="combined_ky", axis="ky"),
    )
    captured: dict[str, object] = {}
    sentinel = SimpleNamespace(
        ky=np.asarray([0.2, 0.4]),
        gamma=np.asarray([0.1, 0.2]),
        omega=np.asarray([-0.3, -0.4]),
        quasilinear=None,
    )

    def _fake_batch(_cfg, ky_arr, **kwargs):
        captured["ky_arr"] = np.asarray(ky_arr)
        captured["solverless_kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(runtime, "_run_runtime_scan_batch", _fake_batch)
    monkeypatch.setattr(
        runtime,
        "run_runtime_linear",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("combined-ky scan must not dispatch per-ky workers")
        ),
    )

    result = run_runtime_scan(
        cfg,
        [0.2, 0.4],
        solver="time",
        method="rk2",
        sample_stride=2,
        show_progress=True,
    )

    assert result is sentinel
    np.testing.assert_allclose(captured["ky_arr"], [0.2, 0.4])
    assert captured["solverless_kwargs"]["method"] == "rk2"
    assert captured["solverless_kwargs"]["sample_stride"] == 2
    assert captured["solverless_kwargs"]["show_progress"] is True


def test_run_runtime_linear_diffrax_contract_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import spectraxgk.runtime as runtime

    cfg0 = _base_cfg()
    cfg = replace(
        cfg0,
        time=replace(cfg0.time, use_diffrax=True, dt=0.01, t_max=0.03, sample_stride=1),
    )
    geom = build_runtime_geometry(cfg)
    grid = build_spectral_grid(cfg.grid)
    gamma_ref = 0.25
    omega_ref = -0.12
    t_saved = np.asarray([0.01, 0.02, 0.03], dtype=float)

    monkeypatch.setattr(runtime, "build_runtime_geometry", lambda _cfg: geom)
    monkeypatch.setattr(
        runtime,
        "_build_initial_condition",
        lambda *args, **kwargs: np.zeros(
            (1, 3, 4, 1, 1, grid.z.size), dtype=np.complex64
        ),
    )

    calls: list[tuple[object, str]] = []

    def _fake_integrate(*args, **kwargs):
        calls.append((kwargs["save_mode"], kwargs["save_field"]))
        if kwargs["save_field"] == "phi+density":
            phi_t = np.ones((3, 1, 1, grid.z.size), dtype=np.complex64)
            density_t = 3.0 * np.ones((3, 1, 1, grid.z.size), dtype=np.complex64)
            return np.zeros((1, 3, 4, 1, 1, grid.z.size), dtype=np.complex64), (
                phi_t,
                density_t,
            )
        phi_t = np.ones((3, 1, 1, grid.z.size), dtype=np.complex64)
        return np.zeros((1, 3, 4, 1, 1, grid.z.size), dtype=np.complex64), phi_t

    monkeypatch.setattr(runtime, "integrate_linear_from_config", _fake_integrate)
    monkeypatch.setattr(
        runtime,
        "extract_mode_time_series",
        lambda arr, sel, method="project": (
            np.exp((gamma_ref - 1j * omega_ref) * t_saved).astype(np.complex128)
            if np.max(np.abs(arr)) < 2.0
            else np.asarray([1.0, 2.0, 4.0], dtype=np.complex128)
        ),
    )
    monkeypatch.setattr(
        runtime,
        "fit_growth_rate_auto_with_stats",
        lambda t, signal, **kwargs: (
            (0.05, -0.02, 0.01, 0.03, 1.0, 0.0)
            if np.max(np.abs(signal)) < 2.0
            else (0.2, -0.08, 0.01, 0.03, 2.0, 0.0)
        ),
    )
    monkeypatch.setattr(
        runtime,
        "extract_eigenfunction",
        lambda *args, **kwargs: np.ones(grid.z.size, dtype=np.complex128),
    )
    monkeypatch.setattr(
        runtime,
        "apply_diagnostic_normalization",
        lambda gamma, omega, **kwargs: (gamma, omega),
    )

    res_phi = run_runtime_linear(
        cfg,
        ky_target=0.1,
        Nl=3,
        Nm=4,
        solver="time",
        fit_signal="phi",
        mode_method="project",
    )
    res_auto = run_runtime_linear(
        cfg,
        ky_target=0.1,
        Nl=3,
        Nm=4,
        solver="time",
        fit_signal="auto",
        mode_method="project",
    )

    assert calls[0] == (None, "phi")
    assert calls[1] == (None, "phi+density")
    metrics = late_time_linear_metrics(res_phi, tail_fraction=2.0 / 3.0)
    assert res_phi.fit_signal_used == "phi"
    assert metrics.gamma_fit == pytest.approx(gamma_ref, rel=1.0e-3)
    assert metrics.omega_fit == pytest.approx(omega_ref, rel=1.0e-3)
    assert res_auto.fit_signal_used == "density"
    np.testing.assert_allclose(res_auto.signal, [1.0, 2.0, 4.0])


def test_run_runtime_nonlinear_final_state_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import spectraxgk.runtime as runtime

    cfg = replace(
        _base_cfg(),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, nonlinear=True),
    )
    geom = build_runtime_geometry(cfg)
    grid = build_spectral_grid(cfg.grid)
    captured: dict[str, object] = {}

    monkeypatch.setattr(runtime, "build_runtime_geometry", lambda _cfg: geom)
    monkeypatch.setattr(
        runtime,
        "build_runtime_linear_params",
        lambda *args, **kwargs: type("P", (), {"rho_star": np.asarray(1.0)})(),
    )
    monkeypatch.setattr(runtime, "build_runtime_term_config", lambda _cfg: object())
    monkeypatch.setattr(
        runtime, "_select_nonlinear_mode_indices", lambda *args, **kwargs: (1, 0)
    )
    monkeypatch.setattr(
        runtime,
        "_build_initial_condition",
        lambda *args, **kwargs: np.zeros(
            (1, 3, 4, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64
        ),
    )

    def _fake_final_state(*args, **kwargs):
        captured["show_progress"] = kwargs.get("show_progress")
        return (
            np.ones(
                (1, 3, 4, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64
            ),
            FieldState(
                phi=np.ones(
                    (grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64
                ),
                apar=None,
                bpar=None,
            ),
        )

    monkeypatch.setattr(runtime, "integrate_nonlinear_from_config", _fake_final_state)

    out = run_runtime_nonlinear(
        cfg,
        ky_target=0.2,
        Nl=3,
        Nm=4,
        diagnostics=False,
        show_progress=True,
    )

    assert captured["show_progress"] is True
    assert out.diagnostics is None
    assert out.phi2 is not None
    assert out.state is None


def test_run_runtime_nonlinear_return_state_uses_diagnostics_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import spectraxgk.runtime as runtime

    cfg = replace(
        _base_cfg(),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, nonlinear=True),
    )
    geom = build_runtime_geometry(cfg)
    grid = build_spectral_grid(cfg.grid)

    monkeypatch.setattr(runtime, "build_runtime_geometry", lambda _cfg: geom)
    monkeypatch.setattr(
        runtime,
        "build_runtime_linear_params",
        lambda *args, **kwargs: type("P", (), {"rho_star": np.asarray(1.0)})(),
    )
    monkeypatch.setattr(runtime, "build_runtime_term_config", lambda _cfg: object())
    monkeypatch.setattr(
        runtime, "_select_nonlinear_mode_indices", lambda *args, **kwargs: (1, 0)
    )
    monkeypatch.setattr(
        runtime,
        "_build_initial_condition",
        lambda *args, **kwargs: np.zeros(
            (1, 3, 4, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64
        ),
    )

    def _fake_diag_integrator(*args, **kwargs):
        t = np.asarray([0.1, 0.2], dtype=float)
        diag = SimulationDiagnostics(
            t=t,
            dt_t=t,
            dt_mean=float(t[-1]),
            gamma_t=np.zeros_like(t),
            omega_t=np.zeros_like(t),
            Wg_t=np.zeros_like(t),
            Wphi_t=np.zeros_like(t),
            Wapar_t=np.zeros_like(t),
            heat_flux_t=np.zeros_like(t),
            particle_flux_t=np.zeros_like(t),
            energy_t=np.zeros_like(t),
        )
        return (
            t,
            diag,
            np.ones(
                (1, 3, 4, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64
            ),
            FieldState(
                phi=np.ones(
                    (grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64
                ),
                apar=None,
                bpar=None,
            ),
        )

    monkeypatch.setattr(
        runtime, "integrate_nonlinear_explicit_diagnostics_state", _fake_diag_integrator
    )

    out = run_runtime_nonlinear(
        cfg,
        ky_target=0.2,
        Nl=3,
        Nm=4,
        diagnostics=False,
        return_state=True,
    )

    assert out.diagnostics is None
    assert out.state is not None
    assert out.phi2 is not None


def test_run_runtime_nonlinear_fixed_mode_requires_indices() -> None:
    cfg = replace(
        _base_cfg(),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, nonlinear=True),
        expert=RuntimeExpertConfig(fixed_mode=True, iky_fixed=None, ikx_fixed=None),
    )
    with pytest.raises(ValueError, match="expert.iky_fixed and expert.ikx_fixed"):
        run_runtime_nonlinear(cfg, ky_target=0.2, Nl=3, Nm=4)


def test_run_runtime_nonlinear_rejects_unknown_external_source() -> None:
    cfg = replace(
        _base_cfg(),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, nonlinear=True),
        expert=RuntimeExpertConfig(source="bad_source"),
    )
    with pytest.raises(ValueError, match="unsupported expert.source"):
        run_runtime_nonlinear(cfg, ky_target=0.2, Nl=3, Nm=4)


def test_run_runtime_nonlinear_adaptive_chunk_requires_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import spectraxgk.runtime as runtime

    cfg = replace(
        _base_cfg(),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, nonlinear=True),
        time=replace(
            _base_cfg().time, fixed_dt=False, t_max=0.3, dt=0.1, diagnostics=True
        ),
    )
    geom = build_runtime_geometry(cfg)
    grid = build_spectral_grid(cfg.grid)

    monkeypatch.setattr(runtime, "build_runtime_geometry", lambda _cfg: geom)
    monkeypatch.setattr(
        runtime,
        "build_runtime_linear_params",
        lambda *args, **kwargs: type("P", (), {"rho_star": np.asarray(1.0)})(),
    )
    monkeypatch.setattr(runtime, "build_runtime_term_config", lambda _cfg: object())
    monkeypatch.setattr(
        runtime, "_select_nonlinear_mode_indices", lambda *args, **kwargs: (1, 0)
    )
    monkeypatch.setattr(
        runtime,
        "_build_initial_condition",
        lambda *args, **kwargs: np.zeros(
            (1, 3, 4, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64
        ),
    )

    def _fake_diag_integrator(*args, **kwargs):
        t = np.asarray([0.0], dtype=float)
        diag = SimulationDiagnostics(
            t=t,
            dt_t=np.asarray([0.1], dtype=float),
            dt_mean=float(0.1),
            gamma_t=np.zeros_like(t),
            omega_t=np.zeros_like(t),
            Wg_t=np.zeros_like(t),
            Wphi_t=np.zeros_like(t),
            Wapar_t=np.zeros_like(t),
            heat_flux_t=np.zeros_like(t),
            particle_flux_t=np.zeros_like(t),
            energy_t=np.zeros_like(t),
        )
        return (
            t,
            diag,
            np.zeros(
                (1, 3, 4, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64
            ),
            FieldState(
                phi=np.ones(
                    (grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64
                ),
                apar=None,
                bpar=None,
            ),
        )

    monkeypatch.setattr(
        runtime, "integrate_nonlinear_explicit_diagnostics_state", _fake_diag_integrator
    )

    with pytest.raises(RuntimeError, match="made no time-step progress"):
        run_runtime_nonlinear(cfg, ky_target=0.2, Nl=3, Nm=4, diagnostics=True)


def test_run_runtime_nonlinear_phiext_source_uses_diagnostics_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import spectraxgk.runtime as runtime

    cfg = replace(
        _base_cfg(),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, nonlinear=True),
        expert=RuntimeExpertConfig(source="phiext_full", phi_ext=0.25),
    )
    geom = build_runtime_geometry(cfg)
    grid = build_spectral_grid(cfg.grid)
    captured: dict[str, object] = {}

    monkeypatch.setattr(runtime, "build_runtime_geometry", lambda _cfg: geom)
    monkeypatch.setattr(
        runtime,
        "build_runtime_linear_params",
        lambda *args, **kwargs: type("P", (), {"rho_star": np.asarray(1.0)})(),
    )
    monkeypatch.setattr(runtime, "build_runtime_term_config", lambda _cfg: object())
    monkeypatch.setattr(
        runtime, "_select_nonlinear_mode_indices", lambda *args, **kwargs: (1, 0)
    )
    monkeypatch.setattr(
        runtime,
        "_build_initial_condition",
        lambda *args, **kwargs: np.zeros(
            (1, 3, 4, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64
        ),
    )

    def _fake_diag_integrator(*args, **kwargs):
        captured.update(kwargs)
        t = np.asarray([0.1, 0.2], dtype=float)
        diag = SimulationDiagnostics(
            t=t,
            dt_t=t,
            dt_mean=float(t[-1]),
            gamma_t=np.zeros_like(t),
            omega_t=np.zeros_like(t),
            Wg_t=np.zeros_like(t),
            Wphi_t=np.asarray([1.0, 1.1]),
            Wapar_t=np.zeros_like(t),
            heat_flux_t=np.zeros_like(t),
            particle_flux_t=np.zeros_like(t),
            energy_t=np.zeros_like(t),
        )
        return (
            t,
            diag,
            np.ones(
                (1, 3, 4, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64
            ),
            FieldState(
                phi=np.ones(
                    (grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64
                ),
                apar=None,
                bpar=None,
            ),
        )

    monkeypatch.setattr(
        runtime, "integrate_nonlinear_explicit_diagnostics_state", _fake_diag_integrator
    )

    out = run_runtime_nonlinear(cfg, ky_target=0.2, Nl=3, Nm=4, diagnostics=False)

    assert captured["external_phi"] == pytest.approx(0.25)
    assert out.diagnostics is None
    assert out.phi2 is not None


def test_run_runtime_nonlinear_adaptive_chunk_forwards_fixed_mode_and_collision_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import spectraxgk.runtime as runtime

    base = _base_cfg()
    cfg = replace(
        base,
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, nonlinear=True),
        time=replace(
            base.time,
            fixed_dt=False,
            t_max=0.35,
            dt=0.1,
            diagnostics=True,
            collision_split=True,
            collision_scheme="exp",
        ),
        expert=RuntimeExpertConfig(fixed_mode=True, iky_fixed=1, ikx_fixed=0),
    )
    geom = build_runtime_geometry(cfg)
    grid = build_spectral_grid(cfg.grid)
    captured: list[dict[str, object]] = []

    monkeypatch.setattr(runtime, "build_runtime_geometry", lambda _cfg: geom)
    monkeypatch.setattr(
        runtime,
        "build_runtime_linear_params",
        lambda *args, **kwargs: type("P", (), {"rho_star": np.asarray(1.0)})(),
    )
    monkeypatch.setattr(runtime, "build_runtime_term_config", lambda _cfg: object())
    monkeypatch.setattr(
        runtime, "_select_nonlinear_mode_indices", lambda *args, **kwargs: (1, 0)
    )
    monkeypatch.setattr(
        runtime,
        "_build_initial_condition",
        lambda *args, **kwargs: np.zeros(
            (1, 3, 4, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64
        ),
    )

    def _fake_diag_integrator(*args, **kwargs):
        captured.append(dict(kwargs))
        t = np.asarray([0.1, 0.2], dtype=float)
        diag = SimulationDiagnostics(
            t=t,
            dt_t=np.asarray([0.1, 0.1], dtype=float),
            dt_mean=float(0.1),
            gamma_t=np.asarray([0.0, 0.0], dtype=float),
            omega_t=np.asarray([0.0, 0.0], dtype=float),
            Wg_t=np.asarray([0.0, 0.0], dtype=float),
            Wphi_t=np.asarray([1.0, 1.2], dtype=float),
            Wapar_t=np.asarray([0.0, 0.0], dtype=float),
            heat_flux_t=np.asarray([0.0, 0.0], dtype=float),
            particle_flux_t=np.asarray([0.0, 0.0], dtype=float),
            energy_t=np.asarray([0.0, 0.0], dtype=float),
        )
        return (
            t,
            diag,
            np.ones(
                (1, 3, 4, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64
            ),
            FieldState(
                phi=np.ones(
                    (grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64
                ),
                apar=None,
                bpar=None,
            ),
        )

    monkeypatch.setattr(
        runtime, "integrate_nonlinear_explicit_diagnostics_state", _fake_diag_integrator
    )

    out = run_runtime_nonlinear(
        cfg, ky_target=0.2, Nl=3, Nm=4, diagnostics=True, show_progress=True
    )

    assert len(captured) == 2
    assert captured[0]["fixed_dt"] is False
    assert captured[0]["collision_split"] is True
    assert captured[0]["collision_scheme"] == "exp"
    assert captured[0]["fixed_mode_ky_index"] == 1
    assert captured[0]["fixed_mode_kx_index"] == 0
    assert captured[0]["show_progress"] is True
    assert out.diagnostics is not None
    assert out.state is None
    assert out.fields is not None
    np.testing.assert_allclose(out.diagnostics.t, [0.1, 0.2, 0.3, 0.4])
