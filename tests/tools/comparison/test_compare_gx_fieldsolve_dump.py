from __future__ import annotations

import numpy as np

from tools.compare_gx_fieldsolve_dump import _load_complex_packed_fields


def test_load_complex_packed_fields_infers_two_block_layout(tmp_path) -> None:
    path = tmp_path / "field_nbar.bin"
    blocks = [
        np.arange(6, dtype=np.float32).astype(np.complex64),
        (10 + np.arange(6, dtype=np.float32)).astype(np.complex64),
    ]
    np.concatenate(blocks).astype(np.complex64).tofile(path)

    out = _load_complex_packed_fields(path, nyc=1, nx=1, nz=6)

    assert len(out) == 2
    np.testing.assert_allclose(out[0].reshape(-1), blocks[0])
    np.testing.assert_allclose(out[1].reshape(-1), blocks[1])
