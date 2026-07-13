from __future__ import annotations

import io
import subprocess
import sys
from contextlib import redirect_stdout

import jax.numpy as jnp
import numpy as np
import pytest

import spectraxgk.config as public_config
from spectraxgk.config import (
    CycloneBaseCase,
    REFERENCE_ELECTRON_MASS,
    GeometryConfig,
    GridConfig,
    KBMBaseCase,
    ModelConfig,
    TimeConfig,
    explicit_method_default_cfl_fac,
    resolve_cfl_fac,
)
from spectraxgk.core.contracts import (
    DifferentiabilityContract,
    ExtensionPointContract,
    ModuleRefactorContract,
    ShapeContract,
    ValidationGateContract,
)
from spectraxgk.core.extension_points import (
    ArtifactWriter,
    BasisFamily,
    CollisionContext,
    CollisionOperator,
    Diagnostic,
    FieldSolver,
    GeometryProvider,
    LinearRHS,
    NonlinearRHS,
    Objective,
    SplitCollisionOperator,
)
from spectraxgk.core.grid import (
    SpectralGrid,
    build_spectral_grid,
    real_fft_ordered_kx,
    real_fft_unique_ky,
    select_ky_grid,
    select_real_fft_ky_grid,
)
from spectraxgk.utils.callbacks import (
    _PROGRESS_START,
    _emit_progress,
    _format_duration,
    print_callback,
    progress_update_stride,
    should_emit_progress,
)


def test_shape_and_differentiability_contracts_validate_core_metadata() -> None:
    shape = ShapeContract(
        ("species", "ky", "kx", "z"), "distribution state", dtype="complex"
    )
    assert shape.axes == ("species", "ky", "kx", "z")

    contract = DifferentiabilityContract(
        differentiable=True,
        jit_safe=True,
        vmap_safe=True,
        static_arg_names=("model",),
        dynamic_arg_names=("state", "geometry"),
        gradient_checks=("finite_difference", "jvp"),
    )
    assert contract.custom_derivative == "none"

    with pytest.raises(ValueError, match="both static and dynamic"):
        DifferentiabilityContract(
            differentiable=True,
            jit_safe=True,
            vmap_safe=False,
            static_arg_names=("model",),
            dynamic_arg_names=("model",),
            gradient_checks=("finite_difference",),
        )

    with pytest.raises(ValueError, match="gradient_checks"):
        DifferentiabilityContract(differentiable=True, jit_safe=True, vmap_safe=True)

    with pytest.raises(ValueError, match="non-empty string"):
        ShapeContract(("",), "bad")

    with pytest.raises(ValueError, match="duplicates"):
        ShapeContract(("z", "z"), "bad")

    with pytest.raises(ValueError, match="description"):
        ShapeContract(("z",), "")

    with pytest.raises(ValueError, match="dtype"):
        ShapeContract(("z",), "bad", dtype="")

    with pytest.raises(ValueError, match="custom derivatives"):
        DifferentiabilityContract(
            differentiable=False,
            jit_safe=True,
            vmap_safe=True,
            custom_derivative="vjp",
        )


def test_module_refactor_contract_tracks_facades_gates_and_extension_points() -> None:
    gate = ValidationGateContract(
        name="linear-growth-fd",
        category="autodiff",
        observable="gamma",
        evidence=("tests/unit/objectives/test_autodiff_solver_objectives.py",),
        tolerance="rtol <= 1e-3",
        literature_anchors=("Cyclone Base Case",),
    )
    extension = ExtensionPointContract(
        name="collision operator",
        protocol="CollisionOperator",
        required_methods=("apply",),
        validation_tests=("tests/unit/core/test_core_contracts.py",),
        documentation="Collision operators must preserve state layout.",
    )
    contract = ModuleRefactorContract(
        source_module="spectraxgk.linear",
        facade_module="spectraxgk.linear",
        target_modules=(
            "spectraxgk.solvers.linear.rhs",
            "spectraxgk.solvers.linear.fields",
        ),
        public_api_compatible=True,
        max_lines_target=800,
        validation_gates=(gate,),
        extension_points=(extension,),
    )
    assert contract.gate_names == ("linear-growth-fd",)
    assert contract.target_package_names == ("spectraxgk.solvers.linear",)
    assert contract.public_api_compatible is True

    with pytest.raises(ValueError, match="validation_gates"):
        ModuleRefactorContract(
            source_module="spectraxgk.linear",
            facade_module="spectraxgk.linear",
            target_modules=("spectraxgk.solvers.linear.rhs",),
            public_api_compatible=True,
            max_lines_target=800,
            validation_gates=(),
        )

    with pytest.raises(ValueError, match="name"):
        ValidationGateContract("", "physics", "gamma", ("artifact.json",))

    with pytest.raises(ValueError, match="observable"):
        ValidationGateContract("gate", "physics", "", ("artifact.json",))

    with pytest.raises(ValueError, match="evidence"):
        ValidationGateContract("gate", "physics", "gamma", ())

    with pytest.raises(ValueError, match="literature_anchors"):
        ValidationGateContract(
            "gate", "physics", "gamma", ("artifact.json",), literature_anchors=("",)
        )

    with pytest.raises(ValueError, match="tolerance"):
        ValidationGateContract(
            "gate", "physics", "gamma", ("artifact.json",), tolerance=""
        )

    with pytest.raises(ValueError, match="name"):
        ExtensionPointContract(
            "",
            "CollisionOperator",
            ("apply",),
            ("tests/unit/core/test_core_contracts.py",),
            "doc",
        )

    with pytest.raises(ValueError, match="protocol"):
        ExtensionPointContract(
            "collision",
            "",
            ("apply",),
            ("tests/unit/core/test_core_contracts.py",),
            "doc",
        )

    with pytest.raises(ValueError, match="required_methods"):
        ExtensionPointContract(
            "collision",
            "CollisionOperator",
            (),
            ("tests/unit/core/test_core_contracts.py",),
            "doc",
        )

    with pytest.raises(ValueError, match="validation_tests"):
        ExtensionPointContract("collision", "CollisionOperator", ("apply",), (), "doc")

    with pytest.raises(ValueError, match="documentation"):
        ExtensionPointContract(
            "collision",
            "CollisionOperator",
            ("apply",),
            ("tests/unit/core/test_core_contracts.py",),
            "",
        )

    with pytest.raises(ValueError, match="source_module"):
        ModuleRefactorContract(
            source_module="linear",
            facade_module="spectraxgk.linear",
            target_modules=("spectraxgk.solvers.linear.rhs",),
            public_api_compatible=True,
            max_lines_target=800,
            validation_gates=(gate,),
        )

    with pytest.raises(ValueError, match="facade_module"):
        ModuleRefactorContract(
            source_module="spectraxgk.linear",
            facade_module="linear",
            target_modules=("spectraxgk.solvers.linear.rhs",),
            public_api_compatible=True,
            max_lines_target=800,
            validation_gates=(gate,),
        )

    with pytest.raises(ValueError, match="target_modules"):
        ModuleRefactorContract(
            source_module="spectraxgk.linear",
            facade_module="spectraxgk.linear",
            target_modules=(),
            public_api_compatible=True,
            max_lines_target=800,
            validation_gates=(gate,),
        )

    with pytest.raises(ValueError, match="max_lines_target"):
        ModuleRefactorContract(
            source_module="spectraxgk.linear",
            facade_module="spectraxgk.linear",
            target_modules=("spectraxgk.solvers.linear.rhs",),
            public_api_compatible=True,
            max_lines_target=0,
            validation_gates=(gate,),
        )


def test_extension_point_protocols_accept_structural_implementations() -> None:
    class ToyBasis:
        def recurrence_coefficients(self, order: int) -> tuple[int, ...]:
            return tuple(range(order))

    class ToyGeometry:
        def sample_flux_tube(self, parameters):
            return {"geometry": parameters}

    class ToyCollision:
        def apply(self, context):
            return context.distribution

    class ToySplitCollision(ToyCollision):
        def split_step(self, context, dt):
            return context.distribution

    class ToyFieldSolver:
        def solve_fields(self, distribution, geometry, parameters):
            return {"phi": distribution}

    class ToyRHS:
        def __call__(self, time, state, parameters):
            return state

    class ToyDiagnostic:
        def evaluate(self, state, fields, geometry, parameters):
            return {"energy": 1.0}

    class ToyObjective:
        def evaluate(self, parameters):
            return 0.0

    class ToyWriter:
        def write(self, payload, destination):
            return {"destination": destination, "payload": payload}

    context = CollisionContext(
        distribution=jnp.ones(1),
        hamiltonian=2.0 * jnp.ones(1),
        fields={"phi": jnp.zeros(1)},
        cache={"Jl": jnp.ones(1)},
        parameters={"nu": 0.1},
    )
    assert jnp.allclose(ToyCollision().apply(context), context.distribution)
    assert isinstance(ToyBasis(), BasisFamily)
    assert isinstance(ToyGeometry(), GeometryProvider)
    assert isinstance(ToyCollision(), CollisionOperator)
    assert isinstance(ToySplitCollision(), SplitCollisionOperator)
    assert not isinstance(ToyCollision(), SplitCollisionOperator)
    assert isinstance(ToyFieldSolver(), FieldSolver)
    assert isinstance(ToyRHS(), LinearRHS)
    assert isinstance(ToyRHS(), NonlinearRHS)
    assert isinstance(ToyDiagnostic(), Diagnostic)
    assert isinstance(ToyObjective(), Objective)
    assert isinstance(ToyWriter(), ArtifactWriter)


def test_linear_terms_import_has_no_facade_order_dependency() -> None:
    """Low-level term imports must work in a fresh interpreter."""

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from spectraxgk.terms.linear_dissipation import "
                "collisions_contribution; "
                "from spectraxgk.terms.linear_terms import "
                "conservative_full_f_dougherty_cross_moments, "
                "drift_kinetic_dougherty_contribution; "
                "from spectraxgk.operators.linear import linear_rhs"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_nonlinear_operator_facade_resolves_lazy_public_exports() -> None:
    """The public facade must not require eager RHS assembly at import time."""

    from spectraxgk.operators import nonlinear

    assert nonlinear.NonlinearDiagnosticKernels.__module__.endswith("diagnostic_state")
    assert callable(nonlinear.compute_nonlinear_diagnostic_tuple)
    assert callable(nonlinear.make_nonlinear_diagnostic_tuple_fn)
    assert callable(nonlinear.linear_rhs_jit_for_terms_impl)
    assert callable(nonlinear.nonlinear_em_term_cached_impl)
    assert callable(nonlinear.nonlinear_rhs_cached_impl)


def test_velocity_basis_orthonormality_and_validation() -> None:
    from spectraxgk.core.velocity import hermite_ladder_coeffs, hermite_normed, laguerre

    xh = jnp.linspace(-6.0, 6.0, 4001)
    dxh = xh[1] - xh[0]
    h = hermite_normed(xh, 4)
    wh = jnp.exp(-xh * xh)
    gram_h = jnp.einsum("ix,jx,x->ij", h, h, wh) * dxh
    assert jnp.allclose(gram_h, jnp.eye(5), atol=2e-2)

    xl = jnp.linspace(0.0, 40.0, 8001)
    dxl = xl[1] - xl[0]
    lag = laguerre(xl, 4)
    wl = jnp.exp(-xl)
    gram_l = jnp.einsum("ix,jx,x->ij", lag, lag, wl) * dxl
    assert jnp.allclose(gram_l, jnp.eye(5), atol=2e-2)

    with pytest.raises(ValueError):
        hermite_normed(jnp.array([0.0]), -1)
    with pytest.raises(ValueError):
        laguerre(jnp.array([0.0]), -1)
    with pytest.raises(ValueError):
        hermite_ladder_coeffs(-1)

    h0 = hermite_normed(jnp.array([0.0, 1.0]), 0)
    l0 = laguerre(jnp.array([0.0, 1.0]), 0)
    assert h0.shape == (1, 2)
    assert l0.shape == (1, 2)


def test_species_builder_core_contracts() -> None:
    from spectraxgk.core.species import Species, build_linear_params

    ion = Species(
        charge=1.0, mass=1.0, density=1.0, temperature=1.0, tprim=2.0, fprim=1.0
    )
    ele = Species(
        charge=-1.0, mass=0.001, density=1.0, temperature=1.0, tprim=2.0, fprim=1.0
    )
    params = build_linear_params([ion, ele], beta=1.0e-4, fapar=1.0)
    assert params.charge_sign.shape == (2,)
    assert params.vth.shape == (2,)
    assert np.isclose(params.vth[0], 1.0)
    assert np.isclose(params.tz[1], -1.0)
    assert params.beta == 1.0e-4


def test_normalization_and_benchmark_public_contracts() -> None:
    import spectraxgk.benchmarks as benchmark_defaults
    from spectraxgk import benchmarks
    from spectraxgk.diagnostics.normalization import (
        apply_diagnostic_normalization,
        get_normalization_contract,
    )

    kin = get_normalization_contract("kinetic")
    assert kin == get_normalization_contract("kinetic_itg")
    with pytest.raises(ValueError, match="Unknown normalization case"):
        get_normalization_contract("not-a-case")

    gamma, omega = apply_diagnostic_normalization(
        0.2, -0.3, rho_star=0.5, diagnostic_norm="rho_star"
    )
    assert gamma == pytest.approx(0.1)
    assert omega == pytest.approx(-0.15)
    gamma_none, omega_none = apply_diagnostic_normalization(
        0.2, -0.3, rho_star=0.5, diagnostic_norm="none"
    )
    assert gamma_none == pytest.approx(0.2)
    assert omega_none == pytest.approx(-0.3)

    cyclone = get_normalization_contract("cyclone")
    etg = get_normalization_contract("etg")
    kinetic = get_normalization_contract("kinetic")
    kbm = get_normalization_contract("kbm")
    assert benchmarks.CYCLONE_OMEGA_D_SCALE == pytest.approx(cyclone.omega_d_scale)
    assert benchmarks.CYCLONE_OMEGA_STAR_SCALE == pytest.approx(
        cyclone.omega_star_scale
    )
    assert benchmarks.CYCLONE_RHO_STAR == pytest.approx(cyclone.rho_star)
    assert benchmarks.ETG_OMEGA_D_SCALE == pytest.approx(etg.omega_d_scale)
    assert benchmarks.ETG_OMEGA_STAR_SCALE == pytest.approx(etg.omega_star_scale)
    assert benchmarks.ETG_RHO_STAR == pytest.approx(etg.rho_star)
    assert benchmarks.KINETIC_OMEGA_D_SCALE == pytest.approx(kinetic.omega_d_scale)
    assert benchmarks.KINETIC_OMEGA_STAR_SCALE == pytest.approx(
        kinetic.omega_star_scale
    )
    assert benchmarks.KINETIC_RHO_STAR == pytest.approx(kinetic.rho_star)
    assert benchmarks.KBM_OMEGA_D_SCALE == pytest.approx(kbm.omega_d_scale)
    assert benchmarks.KBM_OMEGA_STAR_SCALE == pytest.approx(kbm.omega_star_scale)
    assert benchmarks.KBM_RHO_STAR == pytest.approx(kbm.rho_star)

    for name in benchmark_defaults.__all__:
        assert getattr(benchmarks, name) is getattr(benchmark_defaults, name)
    assert benchmarks.KINETIC_KRYLOV_REFERENCE_ALIGNED.shift_source == "history"
    assert benchmarks.KINETIC_KRYLOV_DEFAULT.shift_source == "target"
    assert benchmarks.KBM_KRYLOV_DEFAULT.mode_family == "kbm"
    assert benchmarks.KBM_KRYLOV_DEFAULT.omega_sign == 1
    assert benchmarks.ETG_KRYLOV_DEFAULT.omega_sign == -1


def test_public_api_facades_and_lazy_import_contracts() -> None:
    import subprocess
    import sys

    import spectraxgk
    import spectraxgk.api as public_api
    from support.paths import REPO_ROOT

    assert public_api.__all__ == list(public_api._EXPORT_TARGETS)
    assert len(public_api.__all__) == len(set(public_api.__all__))
    assert spectraxgk.ExplicitTimeConfig.__name__ == "ExplicitTimeConfig"
    assert callable(spectraxgk.integrate_nonlinear_explicit_diagnostics)
    assert "LinearExplicitTimeConfig" not in spectraxgk.__all__
    assert "integrate_nonlinear_diagnostics" not in spectraxgk.__all__

    root_script = f"""
import sys
sys.path.insert(0, {str(REPO_ROOT / "src")!r})
import spectraxgk
assert "numpy" not in sys.modules
assert "jax" not in sys.modules
from spectraxgk.parallel.decomposition import build_independent_portfolio_decomposition
contract = build_independent_portfolio_decomposition(
    4, requested_shards=2, workload="independent_ky_scan"
)
assert contract.actual_shards == 2
assert "numpy" not in sys.modules
assert "jax" not in sys.modules
"""
    subprocess.run([sys.executable, "-S", "-c", root_script], check=True)

    api_script = f"""
import sys
sys.path.insert(0, {str(REPO_ROOT / "src")!r})
import spectraxgk.api as api
assert "numpy" not in sys.modules
assert "jax" not in sys.modules
assert "LinearParams" in api.__all__
"""
    subprocess.run([sys.executable, "-S", "-c", api_script], check=True)


# Progress callback contracts.
def test_emit_progress_reports_one_based_step_once() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        _emit_progress(4, 5, 1.0, -2.0, 3.0, 4.0, sim_time=2.0, sim_total=2.5)
    out = buf.getvalue().strip()
    assert "step=5/5" in out
    assert "progress=100.0%" in out
    assert "t=2/2.5" in out
    assert "elapsed=" in out
    assert "eta=00:00" in out
    assert "step=6/5" not in out


def test_format_duration_clamps_and_rolls_over() -> None:
    assert _format_duration(-2.0) == "00:00"
    assert _format_duration(65.4) == "01:05"
    assert _format_duration(3661.0) == "1:01:01"


def test_progress_update_stride_caps_long_runs() -> None:
    assert progress_update_stride(5) == 1
    assert progress_update_stride(50) == 1
    assert progress_update_stride(51) == 2
    assert progress_update_stride(500) == 10


def test_progress_update_stride_sanitizes_inputs() -> None:
    assert progress_update_stride(0) == 1
    assert progress_update_stride(-10, target_updates=0) == 1
    assert progress_update_stride(9, target_updates=4) == 3


def test_should_emit_progress_reports_first_interval_and_last() -> None:
    assert bool(should_emit_progress(0, 200)) is True
    assert bool(should_emit_progress(3, 200)) is True
    assert bool(should_emit_progress(4, 200)) is False
    assert bool(should_emit_progress(199, 200)) is True


def test_should_emit_progress_sanitizes_steps_and_targets() -> None:
    assert bool(should_emit_progress(0, 0, target_updates=0)) is True
    assert bool(should_emit_progress(1, 9, target_updates=4)) is False
    assert bool(should_emit_progress(2, 9, target_updates=4)) is True


def test_emit_progress_handles_time_variants_and_metric_labels(monkeypatch) -> None:
    ticks = iter([10.0, 12.0])
    monkeypatch.setattr(
        "spectraxgk.utils.callbacks.time.perf_counter", lambda: next(ticks)
    )
    _PROGRESS_START.clear()

    first = io.StringIO()
    with redirect_stdout(first):
        _emit_progress(
            0, 3, 0.1, 0.2, 0.3, 0.4, sim_time=1.25, metric_labels=("A", "B")
        )
    first_out = first.getvalue()
    assert "step=1/3" in first_out
    assert "t=1.25" in first_out
    assert "eta=--:--" in first_out
    assert "A=0.3 B=0.4" in first_out

    second = io.StringIO()
    with redirect_stdout(second):
        _emit_progress(1, 3, 0.1, 0.2, 0.3, 0.4, sim_time=2.0, sim_total=0.0)
    second_out = second.getvalue()
    assert "step=2/3" in second_out
    assert "t=2" in second_out
    assert "/0" not in second_out
    assert "eta=00:01" in second_out


def test_print_callback_returns_state_and_forwards_values(monkeypatch) -> None:
    calls = []

    def fake_callback(fn, *args):
        calls.append(args)
        fn(*args)

    monkeypatch.setattr("spectraxgk.utils.callbacks.jax.debug.callback", fake_callback)

    state = {"unchanged": True}
    buf = io.StringIO()
    with redirect_stdout(buf):
        returned = print_callback(
            state,
            0,
            1,
            1.5,
            -0.5,
            2.0,
            3.0,
            sim_time=None,
            sim_total=None,
            metric_labels=("heat", "free"),
        )

    assert returned is state
    assert calls == [(0, 1, 1.5, -0.5, 2.0, 3.0, None, None)]
    out = buf.getvalue()
    assert "heat=2" in out
    assert "free=3" in out


# Configuration contracts.
def test_config_to_dict():
    """All config dataclasses should serialize to dictionaries."""
    cfg = CycloneBaseCase()
    d = cfg.to_dict()
    assert set(d.keys()) == {
        "grid",
        "time",
        "geometry",
        "model",
        "init",
        "reference_alignment",
    }
    assert d["geometry"]["q"] == cfg.geometry.q
    assert d["grid"]["y0"] == 20.0
    assert d["grid"]["ntheta"] == 32
    assert d["grid"]["nperiod"] == 2
    assert d["reference_alignment"]["enabled"] is True


def test_benchmark_case_presets_keep_stable_public_exports() -> None:
    """Benchmark presets are owned directly by the public config module."""

    for name in (
        "ModelConfig",
        "CycloneBaseCase",
        "KineticElectronModelConfig",
        "KBMBaseCase",
    ):
        assert hasattr(public_config, name)
        assert name in public_config.__all__


def test_config_override():
    """Overrides should propagate into the serialized representation."""
    grid = GridConfig(Nx=12, Ny=10, Nz=8)
    geom = GeometryConfig(q=1.7, s_hat=0.9, epsilon=0.2)
    model = ModelConfig(R_over_LTi=7.0, R_over_LTe=1.0, R_over_Ln=2.5)
    time = TimeConfig(t_max=1.0, dt=0.05, compressed_real_fft=False)
    cfg = CycloneBaseCase(grid=grid, time=time, geometry=geom, model=model)
    d = cfg.to_dict()
    assert d["grid"]["Nx"] == 12
    assert d["geometry"]["q"] == 1.7
    assert d["model"]["R_over_LTe"] == 1.0
    assert d["time"]["dt"] == 0.05
    assert d["time"]["compressed_real_fft"] is False


def test_reference_aligned_mass_ratio_defaults() -> None:
    """Reference-aligned benchmark defaults should use the tracked electron mass."""

    cfg = KBMBaseCase()
    assert (1.0 / cfg.model.mass_ratio) == pytest.approx(REFERENCE_ELECTRON_MASS)


def test_kbm_config_to_dict():
    """KBM configuration should serialize to dictionaries."""
    cfg = KBMBaseCase()
    d = cfg.to_dict()
    assert d["model"]["beta"] == cfg.model.beta


def test_explicit_method_default_cfl_fac_is_method_resolved() -> None:
    assert explicit_method_default_cfl_fac("rk2") == pytest.approx(1.0)
    assert explicit_method_default_cfl_fac("rk3") == pytest.approx(1.73)
    assert explicit_method_default_cfl_fac("sspx3") == pytest.approx(1.73)
    assert explicit_method_default_cfl_fac("rk4") == pytest.approx(2.82)


def test_explicit_method_default_cfl_fac_alias_is_method_resolved() -> None:
    assert explicit_method_default_cfl_fac("rk2") == pytest.approx(1.0)
    assert explicit_method_default_cfl_fac("rk3") == pytest.approx(1.73)
    assert explicit_method_default_cfl_fac("sspx3") == pytest.approx(1.73)
    assert explicit_method_default_cfl_fac("rk4") == pytest.approx(2.82)


def test_resolve_cfl_fac_preserves_explicit_override() -> None:
    assert resolve_cfl_fac("rk3", None) == pytest.approx(1.73)
    assert resolve_cfl_fac("rk4", 1.25) == pytest.approx(1.25)


# Spectral grid contracts.
def test_build_spectral_grid_shapes():
    """Grid arrays should have consistent shapes."""
    cfg = GridConfig(Nx=8, Ny=6, Nz=4, Lx=2.0, Ly=3.0)
    grid = build_spectral_grid(cfg)
    assert grid.kx.shape == (cfg.Nx,)
    assert grid.ky.shape == (cfg.Ny,)
    assert grid.z.shape == (cfg.Nz,)
    assert grid.kx_grid.shape == (cfg.Ny, cfg.Nx)
    assert grid.ky_grid.shape == (cfg.Ny, cfg.Nx)
    assert grid.dealias_mask.shape == (cfg.Ny, cfg.Nx)


def test_build_spectral_grid_spacing():
    """Fourier spacing should match 2*pi/L for each direction."""
    cfg = GridConfig(Nx=8, Ny=6, Nz=4, Lx=2.0, Ly=3.0)
    grid = build_spectral_grid(cfg)
    dkx = grid.kx[1] - grid.kx[0]
    dky = grid.ky[1] - grid.ky[0]
    assert jnp.isclose(dkx, 2.0 * jnp.pi / cfg.Lx)
    assert jnp.isclose(dky, 2.0 * jnp.pi / cfg.Ly)


def test_spectral_grid_tree_roundtrip():
    """SpectralGrid pytree should round-trip through flatten/unflatten."""
    cfg = GridConfig(Nx=4, Ny=4, Nz=4, Lx=2.0, Ly=2.0)
    grid = build_spectral_grid(cfg)
    children, aux = grid.tree_flatten()
    grid2 = SpectralGrid.tree_unflatten(aux, children)
    assert jnp.allclose(grid2.kx, grid.kx)
    assert jnp.allclose(grid2.ky, grid.ky)
    assert jnp.allclose(grid2.z, grid.z)


def test_grid_config_y0_and_ntheta():
    """Field-aligned grid inputs should map to expected ky and z spacing."""
    cfg = GridConfig(Nx=4, Ny=12, Nz=4, Lx=2.0, Ly=3.0, y0=20.0, ntheta=8, nperiod=2)
    grid = build_spectral_grid(cfg)
    assert grid.z.shape[0] == 8 * 3
    dz = grid.z[1] - grid.z[0]
    assert jnp.isclose(dz, 2.0 * jnp.pi / 8.0)
    dky = grid.ky[1] - grid.ky[0]
    assert jnp.isclose(dky, 1.0 / 20.0)


def test_grid_config_ntheta_default_zp():
    """ntheta without nperiod should default to Zp=1."""
    cfg = GridConfig(Nx=4, Ny=4, Nz=4, Lx=2.0, Ly=3.0, ntheta=6)
    grid = build_spectral_grid(cfg)
    assert grid.z.shape[0] == 6
    assert jnp.isclose(grid.z[0], -jnp.pi)
    dz = grid.z[1] - grid.z[0]
    assert jnp.isclose(dz, 2.0 * jnp.pi / 6.0)


def test_grid_config_explicit_zp():
    """Explicit Zp should override nperiod when provided."""
    cfg = GridConfig(Nx=4, Ny=4, Nz=4, Lx=2.0, Ly=3.0, ntheta=5, zp=3)
    grid = build_spectral_grid(cfg)
    assert grid.z.shape[0] == 15
    assert jnp.isclose(grid.z[0], -jnp.pi * 3.0)


def test_compressed_real_fft_wavenumbers_match_gx_native_layout():
    """compressed real-FFT helpers should expose positive Nyquist multipliers."""

    cfg = GridConfig(Nx=4, Ny=10, Nz=4, Lx=2.0, Ly=20.0)
    grid = build_spectral_grid(cfg)
    dkx = 2.0 * jnp.pi / cfg.Lx
    dky = 2.0 * jnp.pi / cfg.Ly
    assert jnp.allclose(
        real_fft_ordered_kx(grid.kx), jnp.asarray([0.0, dkx, 2.0 * dkx, -dkx])
    )
    assert jnp.allclose(
        real_fft_unique_ky(grid.ky),
        jnp.asarray([0.0, dky, 2.0 * dky, 3.0 * dky, 4.0 * dky, 5.0 * dky]),
    )


def test_select_real_fft_ky_grid_uses_explicit_positive_dump_values():
    """GX dump grids should not inherit the negative Nyquist sign from fftfreq order."""

    cfg = GridConfig(Nx=4, Ny=6, Nz=4, Lx=2.0, Ly=6.0)
    grid = build_spectral_grid(cfg)
    gx_ky = jnp.asarray(
        [
            0.0,
            2.0 * jnp.pi / cfg.Ly,
            2.0 * 2.0 * jnp.pi / cfg.Ly,
            3.0 * 2.0 * jnp.pi / cfg.Ly,
        ]
    )
    gx_grid = select_real_fft_ky_grid(grid, gx_ky)

    assert jnp.allclose(gx_grid.ky, gx_ky)
    assert jnp.all(gx_grid.ky >= 0.0)
    assert jnp.allclose(gx_grid.kx, real_fft_ordered_kx(grid.kx))
    assert gx_grid.dealias_mask.shape == (gx_ky.shape[0], cfg.Nx)
    assert jnp.allclose(gx_grid.ky_grid[:, 0], gx_ky)


def test_twothirds_mask_matches_strict_twothirds_cutoff():
    """The nonlinear two-thirds mask excludes the |k| = 1/3 shell."""

    cfg = GridConfig(Nx=96, Ny=96, Nz=4, Lx=2.0 * jnp.pi, Ly=96.0)
    grid = build_spectral_grid(cfg)
    gx_grid = select_real_fft_ky_grid(grid, real_fft_unique_ky(grid.ky))
    mask = jnp.asarray(gx_grid.dealias_mask)

    # Positive ky rows retained by GX on a 96-point padded grid are 0..31.
    assert int(mask[:, 0].sum()) == 32
    assert bool(mask[31, 0])
    assert not bool(mask[32, 0])

    # Retained kx modes are -31..31 in FFT ordering.
    assert int(mask[0, :].sum()) == 63
    assert bool(mask[0, 31])
    assert not bool(mask[0, 32])


def test_select_ky_grid_disables_nonlinear_dealias_mask_for_linear_slices():
    """Linear ky slices should not zero modes dealiased only for nonlinear products."""

    cfg = GridConfig(Nx=1, Ny=12, Nz=4, Lx=2.0 * jnp.pi, y0=10.0)
    grid = build_spectral_grid(cfg)
    # The strict nonlinear two-thirds mask removes the boundary shell at ky index 4.
    assert not bool(grid.dealias_mask[4, 0])

    sliced = select_ky_grid(grid, [3, 4])

    assert jnp.allclose(sliced.ky, grid.ky[jnp.asarray([3, 4])])
    assert jnp.all(sliced.dealias_mask)
