from __future__ import annotations

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
    shape = ShapeContract(("species", "ky", "kx", "z"), "distribution state", dtype="complex")
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
        evidence=("tests/test_solver_objective_gradients.py",),
        tolerance="rtol <= 1e-3",
        literature_anchors=("Cyclone Base Case",),
    )
    extension = ExtensionPointContract(
        name="collision operator",
        protocol="CollisionOperator",
        required_methods=("apply",),
        validation_tests=("tests/test_core_contracts.py",),
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
        ValidationGateContract("gate", "physics", "gamma", ("artifact.json",), literature_anchors=("",))

    with pytest.raises(ValueError, match="tolerance"):
        ValidationGateContract("gate", "physics", "gamma", ("artifact.json",), tolerance="")

    with pytest.raises(ValueError, match="name"):
        ExtensionPointContract("", "CollisionOperator", ("apply",), ("tests/test_core_contracts.py",), "doc")

    with pytest.raises(ValueError, match="protocol"):
        ExtensionPointContract("collision", "", ("apply",), ("tests/test_core_contracts.py",), "doc")

    with pytest.raises(ValueError, match="required_methods"):
        ExtensionPointContract("collision", "CollisionOperator", (), ("tests/test_core_contracts.py",), "doc")

    with pytest.raises(ValueError, match="validation_tests"):
        ExtensionPointContract("collision", "CollisionOperator", ("apply",), (), "doc")

    with pytest.raises(ValueError, match="documentation"):
        ExtensionPointContract("collision", "CollisionOperator", ("apply",), ("tests/test_core_contracts.py",), "")

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
