from __future__ import annotations

import spectraxgk.operators as operators
import spectraxgk.operators.linear as linear_operators
import spectraxgk.operators.nonlinear as nonlinear_operators
import spectraxgk.operators.nonlinear.diagnostic_state as operator_diagnostics
import spectraxgk.operators.nonlinear.rhs as operator_rhs
import spectraxgk.solvers.nonlinear as nonlinear_solvers
import spectraxgk.solvers.nonlinear.explicit as solver_explicit
import spectraxgk.solvers.nonlinear.imex as solver_imex


def test_operator_package_preserves_public_linear_export_identity() -> None:
    assert operators.hermite_streaming is linear_operators.hermite_streaming


def test_nonlinear_operator_package_reexports_rhs_implementation() -> None:
    assert nonlinear_operators.RhsCallable is operator_rhs.RhsCallable
    assert (
        nonlinear_operators.linear_rhs_jit_for_terms_impl
        is operator_rhs.linear_rhs_jit_for_terms_impl
    )
    assert nonlinear_operators.nonlinear_rhs_cached_impl is operator_rhs.nonlinear_rhs_cached_impl
    assert (
        nonlinear_operators.nonlinear_em_term_cached_impl
        is operator_rhs.nonlinear_em_term_cached_impl
    )


def test_nonlinear_operator_package_reexports_diagnostic_implementation() -> None:
    assert nonlinear_operators.NonlinearDiagnosticKernels is (
        operator_diagnostics.NonlinearDiagnosticKernels
    )
    assert (
        nonlinear_operators.compute_nonlinear_diagnostic_tuple
        is operator_diagnostics.compute_nonlinear_diagnostic_tuple
    )


def test_nonlinear_solver_package_reexports_implementations() -> None:
    assert (
        nonlinear_solvers.advance_explicit_nonlinear_state
        is solver_explicit.advance_explicit_nonlinear_state
    )
    assert (
        nonlinear_solvers.checkpoint_explicit_step
        is solver_explicit.checkpoint_explicit_step
    )
    assert nonlinear_solvers.imex_fixed_point_guess is solver_imex.imex_fixed_point_guess
    assert nonlinear_solvers.solve_imex_step is solver_imex.solve_imex_step
