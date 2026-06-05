from __future__ import annotations

from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
from netCDF4 import Dataset

from tools.build_vmec_boundary_transport_landscape import (
    DEFAULT_FRACTIONS,
    _load_nonlinear_ensemble,
    _parse_float_list,
    _reuse_reduced_metrics_from_report,
    _sample_standard_error,
    _write_scan_inputs,
)
from tools.patch_vmec_jax_wout_metadata import patch_wout
from tools.write_vmec_boundary_perturbation_inputs import _parse_coefficient_spec


def _coefficient_value(text: str, name: str) -> float:
    match = re.search(rf"{re.escape(name)}\s*=\s*([+\-\d.Ee]+)", text)
    if match is None:
        raise AssertionError(f"missing {name}")
    return float(match.group(1))


def test_landscape_scan_inputs_patch_selected_boundary_coefficient(tmp_path: Path) -> None:
    baseline = tmp_path / "input.final"
    baseline.write_text(
        "\n".join(
            [
                "&INDATA",
                "  RBC(0,0) = 1.0000000000000000E+00",
                "  RBC(0,1) = 2.0000000000000000E-01",
                "  ZBS(0,1) = 1.0000000000000000E-01",
                "/",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    base, amplitude, reference, rows = _write_scan_inputs(
        baseline_input=baseline,
        coefficient=_parse_coefficient_spec("RBC(0,1)"),
        fractions=(-0.1, 0.0, 0.1),
        out_dir=tmp_path / "scan",
    )

    assert base == 0.2
    assert amplitude == 0.2
    assert reference == "RBC(0,1)"
    assert [row["relative_fraction"] for row in rows] == [-0.1, 0.0, 0.1]
    patched = [Path(row["input_path"]).read_text(encoding="utf-8") for row in rows]
    assert _coefficient_value(patched[0], "RBC(0,1)") == 0.18000000000000002
    assert _coefficient_value(patched[1], "RBC(0,1)") == 0.2
    assert _coefficient_value(patched[2], "RBC(0,1)") == 0.22000000000000003
    assert all("ZBS(0,1) = 1.0000000000000000E-01" in text for text in patched)


def test_landscape_scan_inputs_use_reference_amplitude_for_zero_coefficient(tmp_path: Path) -> None:
    baseline = tmp_path / "input.final"
    baseline.write_text(
        "\n".join(
            [
                "&INDATA",
                "  RBC(0,1) = 2.0000000000000000E-01",
                "  RBC(1,0) = -1.5000000000000000E-01",
                "  RBC(1,1) = 0.0000000000000000E+00",
                "/",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    base, amplitude, reference, rows = _write_scan_inputs(
        baseline_input=baseline,
        coefficient=_parse_coefficient_spec("RBC(1,1)"),
        fractions=(-0.5, 0.0, 0.5),
        out_dir=tmp_path / "scan",
        zero_reference_coefficients=(
            _parse_coefficient_spec("RBC(1,0)"),
            _parse_coefficient_spec("RBC(0,1)"),
        ),
    )

    assert base == 0.0
    assert amplitude == 0.2
    assert reference == "RBC(0,1)"
    patched = [Path(row["input_path"]).read_text(encoding="utf-8") for row in rows]
    assert _coefficient_value(patched[0], "RBC(1,1)") == -0.1
    assert _coefficient_value(patched[1], "RBC(1,1)") == 0.0
    assert _coefficient_value(patched[2], "RBC(1,1)") == 0.1


def test_parse_float_list_rejects_empty_lists() -> None:
    assert _parse_float_list("0.45,0.64") == (0.45, 0.64)

    try:
        _parse_float_list(" , ")
    except Exception as exc:
        assert "expected at least one" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("empty float list was accepted")


def test_default_landscape_scan_spans_dense_half_range() -> None:
    assert len(DEFAULT_FRACTIONS) == 21
    assert DEFAULT_FRACTIONS[0] == -0.5
    assert DEFAULT_FRACTIONS[-1] == 0.5
    assert 0.0 in DEFAULT_FRACTIONS


def test_reuse_reduced_metrics_validates_sample_set_and_point_values(tmp_path: Path) -> None:
    baseline = tmp_path / "input.final"
    baseline.write_text(
        "\n".join(
            [
                "&INDATA",
                "  RBC(0,1) = 2.0000000000000000E-01",
                "/",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    base, _amplitude, _reference, rows = _write_scan_inputs(
        baseline_input=baseline,
        coefficient=_parse_coefficient_spec("RBC(0,1)"),
        fractions=(0.0, 0.1),
        out_dir=tmp_path / "scan",
    )
    for row in rows:
        row["reduced_metrics"] = {}
        row["reduced_metric_reports"] = {}
    reusable = tmp_path / "landscape.json"
    reusable.write_text(
        """
{
  "baseline_coefficient_value": 0.2,
  "coefficient": "RBC(0,1)",
  "sample_set": {"surfaces": [0.64], "alphas": [0.0], "ky_values": [0.3, 0.5]},
  "rows": [
    {
      "label": "0",
      "coefficient_value": 0.2,
      "reduced_metrics": {"growth": 1.2, "quasilinear_flux": 2.3},
      "reduced_metric_reports": {
        "growth": {"payload": {"sample_statistics": {"weighted_standard_error": 0.05}}}
      }
    },
    {"label": "p0p1", "coefficient_value": 0.22000000000000003, "reduced_metrics": {"growth": 0.9, "quasilinear_flux": 1.7}}
  ]
}
""",
        encoding="utf-8",
    )
    args = SimpleNamespace(surfaces="0.64", alphas="0.0", ky_values="0.3,0.5")

    _reuse_reduced_metrics_from_report(
        rows=rows,
        kinds=("growth", "quasilinear_flux"),
        path=reusable,
        coefficient_label="RBC(0,1)",
        baseline_value=base,
        args=args,
    )

    assert [row["reduced_metrics"]["growth"] for row in rows] == [1.2, 0.9]
    assert rows[0]["reduced_metric_reports"]["growth"]["reused_from"] == reusable
    assert _sample_standard_error(rows[0], "growth") == 0.05

    bad_args = SimpleNamespace(surfaces="0.64,0.7", alphas="0.0", ky_values="0.3,0.5")
    try:
        _reuse_reduced_metrics_from_report(
            rows=rows,
            kinds=("growth",),
            path=reusable,
            coefficient_label="RBC(0,1)",
            baseline_value=base,
            args=bad_args,
        )
    except ValueError as exc:
        assert "sample_set.surfaces" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("mismatched sample set was accepted")


def test_load_nonlinear_ensemble_preserves_uncertainty_and_pass_flag(tmp_path: Path) -> None:
    sidecar = tmp_path / "candidate_ensemble_gate.json"
    sidecar.write_text(
        """
{
  "case": "landscape_rbc_1_1_m0p5_replicated_nonlinear_window",
  "passed": false,
  "statistics": {
    "ensemble_mean": 14.43919388557596,
    "combined_sem": 0.5831708511946153
  }
}
""",
        encoding="utf-8",
    )

    point = _load_nonlinear_ensemble(f"0.06331225406918571:{sidecar}")

    assert point["coefficient_value"] == 0.06331225406918571
    assert point["mean"] == 14.43919388557596
    assert point["sem"] == 0.5831708511946153
    assert point["passed"] is False
    assert point["case"] == "landscape_rbc_1_1_m0p5_replicated_nonlinear_window"


def test_patch_vmec_jax_wout_metadata_fills_zero_scalars(tmp_path: Path) -> None:
    path = tmp_path / "wout_test.nc"
    with Dataset(path, "w") as ds:
        ds.createDimension("mn", 2)
        ds.createDimension("ns", 2)
        ds.createVariable("xm", "f8", ("mn",))[:] = np.asarray([0.0, 1.0])
        ds.createVariable("xn", "f8", ("mn",))[:] = np.asarray([0.0, 0.0])
        rmnc = ds.createVariable("rmnc", "f8", ("ns", "mn"))
        rmnc[:, :] = np.asarray([[1.0, 0.1], [1.0, 0.2]])
        for name in ("Aminor_p", "Rmajor_p", "aspect", "volume_p"):
            ds.createVariable(name, "f8").assignValue(0.0)

    report = patch_wout(path, ntheta=64, nphi=4)

    assert set(report["patched"]) == {"Aminor_p", "Rmajor_p", "aspect", "volume_p"}
    assert report["after"]["Aminor_p"] > 0.0
    assert report["after"]["Rmajor_p"] > 0.0
    assert report["after"]["aspect"] > 0.0
    with Dataset(path) as ds:
        assert float(ds.variables["Aminor_p"][:]) == report["after"]["Aminor_p"]
