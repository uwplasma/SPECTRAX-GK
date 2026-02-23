import numpy as np

from spectraxgk.benchmarks import (
    load_cyclone_reference_gs2,
    load_cyclone_reference_stella,
    load_etg_reference_gs2,
    load_etg_reference_stella,
)


def _check_ref(ref) -> None:
    assert ref.ky.size > 0
    assert ref.gamma.size == ref.ky.size
    assert ref.omega.size == ref.ky.size
    assert np.all(np.isfinite(ref.ky))
    assert np.all(np.isfinite(ref.gamma))
    assert np.all(np.isfinite(ref.omega))


def test_load_cyclone_crosscode_references():
    _check_ref(load_cyclone_reference_gs2())
    _check_ref(load_cyclone_reference_stella())


def test_load_etg_crosscode_references():
    _check_ref(load_etg_reference_gs2())
    _check_ref(load_etg_reference_stella())
