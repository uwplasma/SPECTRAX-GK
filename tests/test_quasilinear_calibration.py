"""Tests for quasilinear calibration artifact helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import spectraxgk
from spectraxgk.quasilinear_calibration import (
    QuasilinearCalibrationPoint,
    calibration_point_from_nonlinear_window_summary,
    quasilinear_calibration_report,
    write_quasilinear_calibration_report,
)


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
