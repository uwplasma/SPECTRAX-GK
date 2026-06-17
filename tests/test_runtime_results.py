from __future__ import annotations

import numpy as np
import pytest

from spectraxgk.workflows.runtime.results import (
    build_runtime_nonlinear_result,
    nonlinear_field_phi2,
)
from spectraxgk.terms.config import FieldState


def test_nonlinear_field_phi2_uses_mean_complex_field_energy_density() -> None:
    fields = FieldState(
        phi=np.asarray(
            [
                [1.0 + 0.0j, 0.0 + 2.0j],
                [3.0 + 4.0j, 0.0 + 0.0j],
            ],
            dtype=np.complex64,
        )
    )

    np.testing.assert_allclose(nonlinear_field_phi2(fields), np.asarray(7.5))


def test_build_runtime_nonlinear_result_summarizes_final_fields_only() -> None:
    fields = FieldState(phi=np.asarray([1.0 + 1.0j, 2.0 + 0.0j]))
    state = np.asarray([3.0])

    result = build_runtime_nonlinear_result(
        t=np.asarray([0.1, 0.2]),
        diagnostics=None,
        fields=fields,
        state=state,
        ky_selected=0.3,
        kx_selected=-0.5,
        summarize_fields=True,
    )

    assert result.t.size == 0
    assert result.diagnostics is None
    assert result.fields is fields
    assert result.state is state
    assert result.ky_selected == pytest.approx(0.3)
    assert result.kx_selected == pytest.approx(-0.5)
    np.testing.assert_allclose(result.phi2, np.asarray(3.0))


def test_build_runtime_nonlinear_result_preserves_diagnostics_without_summary() -> None:
    t = np.asarray([0.1, 0.2])

    result = build_runtime_nonlinear_result(
        t=t,
        diagnostics=None,
        fields=None,
        state=None,
        ky_selected=None,
        kx_selected=None,
        summarize_fields=False,
    )

    np.testing.assert_allclose(result.t, t)
    assert result.diagnostics is None
    assert result.phi2 is None
    assert result.fields is None


def test_build_runtime_nonlinear_result_requires_fields_for_summary() -> None:
    with pytest.raises(RuntimeError, match="final fields are required"):
        build_runtime_nonlinear_result(
            t=np.asarray([0.1]),
            diagnostics=None,
            fields=None,
            state=None,
            ky_selected=None,
            kx_selected=None,
            summarize_fields=True,
        )
