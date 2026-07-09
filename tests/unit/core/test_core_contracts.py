from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

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
    CollisionOperator,
    Diagnostic,
    FieldSolver,
    GeometryProvider,
    LinearRHS,
    NonlinearRHS,
    Objective,
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
        evidence=("tests/unit/objectives/test_solver_objective_gradients.py",),
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
        def apply(self, state, geometry, parameters):
            return state

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

    assert isinstance(ToyBasis(), BasisFamily)
    assert isinstance(ToyGeometry(), GeometryProvider)
    assert isinstance(ToyCollision(), CollisionOperator)
    assert isinstance(ToyFieldSolver(), FieldSolver)
    assert isinstance(ToyRHS(), LinearRHS)
    assert isinstance(ToyRHS(), NonlinearRHS)
    assert isinstance(ToyDiagnostic(), Diagnostic)
    assert isinstance(ToyObjective(), Objective)
    assert isinstance(ToyWriter(), ArtifactWriter)


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


def test_species_builder_and_runtime_toml_core_contracts(tmp_path: Path) -> None:
    from spectraxgk.core.species import Species, build_linear_params
    from spectraxgk.workflows.runtime.toml import (
        load_case_from_toml,
        load_linear_terms_from_toml,
    )

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

    flag_path = tmp_path / "case_flag.toml"
    flag_path.write_text('case = "cyclone"\nreference_alignment = true\n', encoding="utf-8")
    case_name, cfg, _data = load_case_from_toml(flag_path)
    assert case_name == "cyclone"
    assert cfg.reference_aligned is True

    table_path = tmp_path / "case_table.toml"
    table_path.write_text(
        'case = "cyclone"\n\n[reference_alignment]\nenabled = false\n',
        encoding="utf-8",
    )
    case_name, cfg, _data = load_case_from_toml(table_path)
    assert case_name == "cyclone"
    assert cfg.reference_aligned is False

    terms = load_linear_terms_from_toml(
        {"terms": {"streaming": 0.0, "apar": 0.0, "bpar": 0.0, "nonlinear": 0.0}}
    )
    assert terms is not None
    assert terms.streaming == 0.0
    assert terms.apar == 0.0
    assert terms.bpar == 0.0


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
    assert benchmarks.CYCLONE_OMEGA_STAR_SCALE == pytest.approx(cyclone.omega_star_scale)
    assert benchmarks.CYCLONE_RHO_STAR == pytest.approx(cyclone.rho_star)
    assert benchmarks.ETG_OMEGA_D_SCALE == pytest.approx(etg.omega_d_scale)
    assert benchmarks.ETG_OMEGA_STAR_SCALE == pytest.approx(etg.omega_star_scale)
    assert benchmarks.ETG_RHO_STAR == pytest.approx(etg.rho_star)
    assert benchmarks.KINETIC_OMEGA_D_SCALE == pytest.approx(kinetic.omega_d_scale)
    assert benchmarks.KINETIC_OMEGA_STAR_SCALE == pytest.approx(kinetic.omega_star_scale)
    assert benchmarks.KINETIC_RHO_STAR == pytest.approx(kinetic.rho_star)
    assert benchmarks.KBM_OMEGA_D_SCALE == pytest.approx(kbm.omega_d_scale)
    assert benchmarks.KBM_OMEGA_STAR_SCALE == pytest.approx(kbm.omega_star_scale)
    assert benchmarks.KBM_RHO_STAR == pytest.approx(kbm.rho_star)

    for name in benchmark_defaults.__all__:
        assert getattr(benchmarks, name) is getattr(benchmark_defaults, name)
    assert benchmarks.KINETIC_KRYLOV_REFERENCE_ALIGNED.shift_source == "history"
    assert benchmarks.KINETIC_KRYLOV_DEFAULT.shift_source == "target"
    assert benchmarks.KBM_KRYLOV_DEFAULT.mode_family == "kbm"
    assert benchmarks.ETG_KRYLOV_DEFAULT.omega_sign == -1


def test_public_api_facades_and_lazy_import_contracts() -> None:
    import subprocess
    import sys

    import spectraxgk
    from support.paths import REPO_ROOT

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
