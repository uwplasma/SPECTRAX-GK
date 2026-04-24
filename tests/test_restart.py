"""Tests for GX restart-state IO helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from spectraxgk.restart import load_gx_restart_state


def test_load_gx_restart_state_accepts_full_ky_reduced_kx_layout(tmp_path: Path) -> None:
    path = tmp_path / "gx.restart.nc"
    root = Dataset(path, "w")
    root.createDimension("Nspecies", 1)
    root.createDimension("Nm", 2)
    root.createDimension("Nl", 2)
    root.createDimension("Nz", 3)
    root.createDimension("Nkx", 3)
    root.createDimension("Nky", 4)
    root.createDimension("ri", 2)
    data = np.zeros((1, 2, 2, 3, 3, 4, 2), dtype=np.float32)
    data[0, 0, 0, 0, 0, 1, 0] = 1.5
    data[0, 0, 0, 1, 2, 3, 1] = -0.25
    root.createVariable("G", "f4", ("Nspecies", "Nm", "Nl", "Nz", "Nkx", "Nky", "ri"))[:] = data
    root.close()

    state = load_gx_restart_state(path, nspecies=1, Nl=2, Nm=2, ny=4, nx=4, nz=3)

    assert state.shape == (1, 2, 2, 4, 4, 3)
    # GX's reduced kx axis is stored in active spectral order [negative, zero,
    # positive]. For nx=4 that maps active Nkx indices [0, 1, 2] to full kx
    # indices [3, 0, 1], matching the restart writer.
    assert state[0, 0, 0, 1, 3, 0] == np.complex64(1.5 + 0.0j)
    assert state[0, 0, 0, 3, 1, 1] == np.complex64(0.0 - 0.25j)
    assert state[0, 0, 0, 1, 0, 0] == np.complex64(0.0 + 0.0j)
