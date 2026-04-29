"""Tests for quasilinear calibration artifact helpers."""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import numpy as np
import pytest

import spectraxgk
from spectraxgk.quasilinear_calibration import (
    QuasilinearCalibrationPoint,
    calibration_point_from_nonlinear_window_summary,
    calibration_point_from_spectrum_and_nonlinear_window,
    integrated_quasilinear_flux_from_spectrum,
    quasilinear_calibration_report,
    write_quasilinear_calibration_report,
)


def _load_build_tool_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "build_quasilinear_calibration_report.py"
    spec = importlib.util.spec_from_file_location("build_quasilinear_calibration_report", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_quasilinear_calibration_report_tracks_train_holdout_claim_level(tmp_path: Path) -> None:
    assert spectraxgk.QuasilinearCalibrationPoint is QuasilinearCalibrationPoint
    points = [
        QuasilinearCalibrationPoint(
            case="cyclone_ky0p2",
            split="train",
            predicted_heat_flux=1.0,
            observed_heat_flux=1.1,
            saturation_rule="mixing_length",
            geometry="cyclone",
            electron_model="adiabatic",
        ),
        {
            "case": "cyclone_ky0p3",
            "split": "holdout",
            "predicted_heat_flux": 0.9,
            "observed_heat_flux": 1.0,
            "saturation_rule": "mixing_length",
            "geometry": "cyclone",
            "electron_model": "adiabatic",
        },
    ]

    report = quasilinear_calibration_report(
        points,
        saturation_rule="mixing_length",
        holdout_mean_rel_gate=0.2,
        metadata={"calibration_policy": "one_constant_train_holdout"},
    )

    assert report["passed"] is True
    assert report["claim_level"] == "calibrated_absolute_flux"
    assert report["by_split"]["train"]["n"] == 1
    assert report["by_split"]["holdout"]["mean_abs_relative_error"] == pytest.approx(0.1)
    out = write_quasilinear_calibration_report(tmp_path / "ql_calibration.json", report)
    assert json.loads(out.read_text(encoding="utf-8"))["passed"] is True


def test_quasilinear_calibration_report_demotes_missing_holdout_or_failed_gate() -> None:
    train_only = quasilinear_calibration_report(
        [
            QuasilinearCalibrationPoint(
                case="cyclone_train",
                split="train",
                predicted_heat_flux=1.0,
                observed_heat_flux=1.0,
                saturation_rule="mixing_length",
            )
        ],
        saturation_rule="mixing_length",
    )
    assert train_only["passed"] is False
    assert train_only["claim_level"] == "training_or_audit_only"

    failed = quasilinear_calibration_report(
        [
            QuasilinearCalibrationPoint(
                case="train",
                split="train",
                predicted_heat_flux=1.0,
                observed_heat_flux=1.0,
                saturation_rule="mixing_length",
            ),
            QuasilinearCalibrationPoint(
                case="holdout",
                split="holdout",
                predicted_heat_flux=2.0,
                observed_heat_flux=1.0,
                saturation_rule="mixing_length",
            ),
        ],
        saturation_rule="mixing_length",
        holdout_mean_rel_gate=0.2,
    )
    assert failed["passed"] is False
    assert failed["claim_level"] == "calibration_dataset"
    assert failed["by_split"]["holdout"]["max_abs_relative_error"] == pytest.approx(1.0)


def test_quasilinear_calibration_report_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        quasilinear_calibration_report([], saturation_rule="mixing_length")
    with pytest.raises(ValueError):
        quasilinear_calibration_report(
            [
                QuasilinearCalibrationPoint(
                    case="bad",
                    split="validation",
                    predicted_heat_flux=1.0,
                    observed_heat_flux=1.0,
                    saturation_rule="mixing_length",
                )
            ],
            saturation_rule="mixing_length",
        )
    with pytest.raises(ValueError):
        quasilinear_calibration_report(
            [
                QuasilinearCalibrationPoint(
                    case="bad",
                    split="train",
                    predicted_heat_flux=1.0,
                    observed_heat_flux=1.0,
                    saturation_rule="mixing_length",
                )
            ],
            saturation_rule="mixing_length",
            observed_floor=0.0,
        )
    metrics = quasilinear_calibration_report(
        [
            QuasilinearCalibrationPoint(
                case="floor",
                split="audit",
                predicted_heat_flux=1.0e-6,
                observed_heat_flux=0.0,
                saturation_rule="mixing_length",
            )
        ],
        saturation_rule="mixing_length",
        observed_floor=1.0e-5,
    )["metrics"]
    assert np.isfinite(float(metrics["mean_abs_relative_error"]))


def test_calibration_point_from_nonlinear_window_summary(tmp_path: Path) -> None:
    diag = tmp_path / "diag.csv"
    diag.write_text(
        "t,heat_flux,particle_flux\n0.0,1.0,0.0\n1.0,2.0,0.0\n2.0,4.0,0.0\n",
        encoding="utf-8",
    )
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "case": "cyclone_window",
                "spectrax": str(diag),
                "tmin": 0.5,
                "tmax": 2.0,
            }
        ),
        encoding="utf-8",
    )

    point = calibration_point_from_nonlinear_window_summary(
        summary,
        predicted_heat_flux=2.5,
        split="holdout",
        saturation_rule="mixing_length",
        geometry="cyclone",
        electron_model="adiabatic",
    )

    assert point.case == "cyclone_window"
    assert point.observed_heat_flux == pytest.approx(3.0)
    assert point.observed_heat_flux_std == pytest.approx(1.0)
    assert point.nonlinear_artifact == str(diag)


def test_integrated_quasilinear_flux_from_spectrum_and_window_point(tmp_path: Path) -> None:
    spectrum = tmp_path / "ql.csv"
    spectrum.write_text(
        "ky,saturated_heat_flux_total,other\n0.1,1.0,0\n0.2,2.0,0\n0.4,4.0,0\n",
        encoding="utf-8",
    )
    summed = integrated_quasilinear_flux_from_spectrum(spectrum)
    assert spectraxgk.integrated_quasilinear_flux_from_spectrum is integrated_quasilinear_flux_from_spectrum
    assert summed["estimate"] == pytest.approx(7.0)
    assert summed["n_samples"] == 3
    trapezoid = integrated_quasilinear_flux_from_spectrum(spectrum, method="trapezoid")
    assert trapezoid["estimate"] == pytest.approx(0.75)
    with pytest.raises(ValueError):
        integrated_quasilinear_flux_from_spectrum(spectrum, column="missing")

    diag = tmp_path / "diag.csv"
    diag.write_text("t,heat_flux\n0.0,2.0\n1.0,4.0\n", encoding="utf-8")
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"case": "c", "spectrax": str(diag)}), encoding="utf-8")

    point = calibration_point_from_spectrum_and_nonlinear_window(
        spectrum,
        summary,
        split="audit",
        saturation_rule="mixing_length",
        spectrum_method="sum",
        geometry="cyclone",
        electron_model="adiabatic",
    )

    assert point.predicted_heat_flux == pytest.approx(7.0)
    assert point.observed_heat_flux == pytest.approx(3.0)
    assert point.quasilinear_artifact == str(spectrum)
    assert "observed_to_predicted" in str(point.notes)


def test_build_calibration_report_tool_can_generate_point_from_artifacts(tmp_path: Path) -> None:
    mod = _load_build_tool_module()
    spectrum = tmp_path / "ql.csv"
    spectrum.write_text("ky,saturated_heat_flux_total\n0.1,1.0\n0.2,2.0\n", encoding="utf-8")
    diag = tmp_path / "diag.csv"
    diag.write_text("t,heat_flux\n0.0,4.0\n1.0,6.0\n", encoding="utf-8")
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"case": "generated", "spectrax": str(diag)}), encoding="utf-8")
    out = tmp_path / "report.json"

    assert (
        mod.main(
            [
                "--spectrum",
                str(spectrum),
                "--nonlinear-summary",
                str(summary),
                "--split",
                "audit",
                "--out",
                str(out),
            ]
        )
        == 0
    )

    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["claim_level"] == "training_or_audit_only"
    assert report["points"][0]["predicted_heat_flux"] == pytest.approx(3.0)
    assert report["points"][0]["observed_heat_flux"] == pytest.approx(5.0)


def test_calibration_point_from_nonlinear_window_summary_rejects_unsupported_sources(tmp_path: Path) -> None:
    diag = tmp_path / "diag.csv"
    diag.write_text("t,heat_flux\n0.0,1.0\n", encoding="utf-8")
    missing_col = tmp_path / "missing.csv"
    missing_col.write_text("t,Wphi\n0.0,1.0\n", encoding="utf-8")

    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"case": "c", "spectrax": str(missing_col)}), encoding="utf-8")
    with pytest.raises(ValueError):
        calibration_point_from_nonlinear_window_summary(
            summary,
            predicted_heat_flux=1.0,
            split="audit",
            saturation_rule="mixing_length",
        )

    nc_summary = tmp_path / "summary_nc.json"
    nc_summary.write_text(json.dumps({"case": "c", "spectrax": str(tmp_path / "run.out.nc")}), encoding="utf-8")
    with pytest.raises(NotImplementedError):
        calibration_point_from_nonlinear_window_summary(
            nc_summary,
            predicted_heat_flux=1.0,
            split="audit",
            saturation_rule="mixing_length",
        )
