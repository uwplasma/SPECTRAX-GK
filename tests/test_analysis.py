"""Analysis helpers for mode extraction and indexing."""

import numpy as np

from spectraxgk.analysis import ModeSelection, extract_mode, select_ky_index


def test_select_ky_index():
    """Selecting ky should choose the nearest grid point."""
    ky = np.array([0.0, 0.1, 0.2, 0.3])
    assert select_ky_index(ky, 0.26) == 3
    assert select_ky_index(ky, 0.04) == 0


def test_extract_mode():
    """Extracted mode should match direct slicing."""
    phi_t = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
    sel = ModeSelection(ky_index=1, kx_index=2, z_index=3)
    mode = extract_mode(phi_t, sel)
    assert np.allclose(mode, phi_t[:, 1, 2, 3])
