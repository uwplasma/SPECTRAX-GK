"""Tests for the GX secondary comparison helper."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from netCDF4 import Dataset
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from compare_gx_secondary import _load_gx_modes, _load_gx_readme_targets, build_parser


def test_load_gx_readme_targets_parses_sidebands_and_zero_modes(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        "\n".join(
            (
                "The correct final output should be roughly",
                "0.0000\t-0.0500\t-0.000160\t4.901835",
                "0.0000\t0.0000",
                "0.0000\t0.0500\t0.000160\t4.901835",
                "",
                "0.1000\t-0.0500\t0.000164\t4.901835",
                "0.1000\t0.0000",
                "0.1000\t0.0500\t-0.000164\t4.901835",
            )
        ),
        encoding="utf-8",
    )
    df = _load_gx_readme_targets(
        readme,
        (
            (0.0, -0.05),
            (0.0, 0.0),
            (0.0, 0.05),
            (0.1, -0.05),
            (0.1, 0.0),
            (0.1, 0.05),
        ),
    )
    assert list(df.columns) == ["ky", "kx", "gamma_gx", "omega_gx"]
    assert len(df) == 6
    center = df[(df["ky"] == 0.1) & (df["kx"] == 0.0)].iloc[0]
    assert float(center["gamma_gx"]) == 0.0
    assert float(center["omega_gx"]) == 0.0
    side = df[(df["ky"] == 0.0) & (df["kx"] == -0.05)].iloc[0]
    assert float(side["gamma_gx"]) == pytest.approx(4.901835)
    assert float(side["omega_gx"]) == pytest.approx(-1.6e-4)


def test_compare_gx_secondary_parser_defaults_to_readme_source() -> None:
    args = build_parser().parse_args(["--gx-readme", "README.md"])
    assert args.gx_source == "readme"
    assert args.gx_out is None
    assert args.gx_tail_fraction == pytest.approx(0.5)
    assert args.stage2_tmax == pytest.approx(100.0)
    assert args.fit_fraction == pytest.approx(0.5)


def test_load_gx_modes_uses_tail_mean_for_out_nc(tmp_path: Path) -> None:
    path = tmp_path / "kh01a.out.nc"
    root = Dataset(path, "w")
    grids = root.createGroup("Grids")
    diag = root.createGroup("Diagnostics")
    root.createDimension("time", 4)
    root.createDimension("ky", 2)
    root.createDimension("kx", 3)
    root.createDimension("ri", 2)
    grids.createVariable("ky", "f8", ("ky",))[:] = np.array([0.0, 0.1])
    grids.createVariable("kx", "f8", ("kx",))[:] = np.array([-0.05, 0.0, 0.05])
    data = np.zeros((4, 2, 3, 2))
    data[:, 0, 0, 0] = np.array([0.0, 0.1, 0.2, 0.3])
    data[:, 0, 0, 1] = np.array([4.8, 4.9, 5.0, 5.1])
    diag.createVariable("omega_kxkyt", "f8", ("time", "ky", "kx", "ri"))[:] = data
    root.close()

    df = _load_gx_modes(path, ((0.0, -0.05),), tail_fraction=0.5)
    row = df.iloc[0]
    assert float(row["omega_gx"]) == pytest.approx(0.25)
    assert float(row["gamma_gx"]) == pytest.approx(5.05)
