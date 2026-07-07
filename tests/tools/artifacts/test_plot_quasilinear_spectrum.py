"""Tests for the quasilinear spectrum plotting tool."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load_tool():
    path = Path(__file__).resolve().parents[3] / "tools" / "plot_quasilinear_spectrum.py"
    spec = importlib.util.spec_from_file_location("plot_quasilinear_spectrum", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_spectrum(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "ky,gamma,omega,kperp_eff2,heat_flux_weight_total,particle_flux_weight_total,amplitude2,saturated_heat_flux_total,saturated_particle_flux_total",
                "0.2,0.1,-0.4,0.8,1.2,0.1,nan,nan,nan",
                "0.3,0.2,-0.5,1.0,1.5,0.2,0.4,0.6,0.08",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_load_quasilinear_spectrum_requires_columns(tmp_path: Path) -> None:
    mod = _load_tool()
    path = tmp_path / "spectrum.csv"
    _write_spectrum(path)
    data = mod.load_quasilinear_spectrum(path)
    np.testing.assert_allclose(data["ky"], [0.2, 0.3])

    bad = tmp_path / "bad.csv"
    bad.write_text("ky,gamma\n0.2,0.1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        mod.load_quasilinear_spectrum(bad)


def test_write_quasilinear_spectrum_figure(tmp_path: Path) -> None:
    mod = _load_tool()
    path = tmp_path / "spectrum.csv"
    _write_spectrum(path)
    out = tmp_path / "ql_spectrum.png"

    paths = mod.write_quasilinear_spectrum_figure(path, out=out, title="Test QL Spectrum")

    assert Path(paths["png"]).exists()
    assert Path(paths["pdf"]).exists()
    assert Path(paths["json"]).exists()
