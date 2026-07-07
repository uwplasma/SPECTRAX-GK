from __future__ import annotations

from pathlib import Path

import numpy as np

from examples.theory_and_demos.quasilinear_implicit_sensitivity import run_demo


def test_quasilinear_implicit_sensitivity_demo_summary(tmp_path: Path) -> None:
    summary = run_demo(outdir=tmp_path, plot=False, write_files=False)

    assert summary["passed"] is True
    assert summary["branch_isolated"] is True
    assert summary["sensitivity_method"] == "implicit_left_right_eigenpair"
    assert len(summary["observable_labels"]) == 5
    assert len(summary["parameter_labels"]) == 2
    jac_impl = np.asarray(summary["jacobian_implicit"], dtype=float)
    jac_fd = np.asarray(summary["jacobian_fd"], dtype=float)
    np.testing.assert_allclose(jac_impl, jac_fd, rtol=5.0e-2, atol=2.0e-3)
