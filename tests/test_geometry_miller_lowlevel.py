from __future__ import annotations

import numpy as np

from spectraxgk.geometry.miller import derm, dermv, generate_miller_eik, nperiod_data_extend, reflect_n_append


def test_miller_derm_covers_parity_and_direction_branches() -> None:
    arr = np.asarray([0.0, 1.0, 4.0, 9.0])

    even_theta = derm(arr, "l", par="e")
    odd_theta = derm(arr, "l", par="o")
    radial = derm(arr, "r")

    assert even_theta.shape == (1, 4)
    np.testing.assert_allclose(even_theta[0], [0.0, 4.0, 8.0, 0.0])
    np.testing.assert_allclose(odd_theta[0, [0, -1]], [2.0, 10.0])
    assert radial.shape == (4, 1)
    np.testing.assert_allclose(radial[:, 0], [2.0, 4.0, 8.0, 10.0])

    surface = np.vstack([arr, arr + 10.0, arr + 20.0])
    radial_2d = derm(surface, "r")
    even_2d = derm(surface, "l", par="e")
    odd_2d = derm(surface, "l", par="o")

    np.testing.assert_allclose(radial_2d[:, 0], [20.0, 20.0, 20.0])
    np.testing.assert_allclose(even_2d[:, [0, -1]], 0.0)
    np.testing.assert_allclose(odd_2d[:, 0], 2.0)


def test_miller_dermv_matches_weighted_finite_difference_branches() -> None:
    x = np.asarray([0.0, 1.0, 2.0, 3.0])
    f = x**2

    even_theta = dermv(f, x, "l", par="e")
    odd_theta = dermv(f, x, "l", par="o")
    radial = dermv(f, x.reshape(-1, 1), "r")

    np.testing.assert_allclose(even_theta[0, 1:-1], [2.0, 4.0])
    np.testing.assert_allclose(odd_theta[0, [0, -1]], [0.0, 6.0])
    np.testing.assert_allclose(radial[:, 0], [1.0, 2.0, 4.0, 5.0])

    surface_x = np.tile(x, (3, 1))
    surface_f = surface_x**2 + np.arange(3)[:, None]
    radial_2d = dermv(surface_f, surface_x + np.arange(3)[:, None], "r")
    even_2d = dermv(surface_f, surface_x, "l", par="e")
    odd_2d = dermv(surface_f, surface_x, "l", par="o")

    assert radial_2d.shape == surface_f.shape
    np.testing.assert_allclose(even_2d[:, 1:-1], [[2.0, 4.0]] * 3)
    np.testing.assert_allclose(odd_2d[:, [0, -1]], [[1.0, 5.0]] * 3)


def test_miller_periodic_extension_and_reflection_helpers() -> None:
    theta = np.asarray([0.0, np.pi / 2.0, np.pi])
    vals = np.asarray([1.0, 2.0, 3.0])

    theta_ext = nperiod_data_extend(theta, 2, istheta=1)
    even_ext = nperiod_data_extend(vals, 2, par="e")
    odd_ext = nperiod_data_extend(vals, 2, par="o")

    assert theta_ext.size == 7
    np.testing.assert_allclose(even_ext, [1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(odd_ext, [1.0, 2.0, 3.0, -2.0, -1.0, 2.0, 3.0])
    np.testing.assert_allclose(reflect_n_append(vals, "e"), [3.0, 2.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(reflect_n_append(vals, "o"), [-3.0, -2.0, 0.0, 2.0, 3.0])


def test_generate_miller_eik_writes_minimal_geometry_file(tmp_path) -> None:
    output = tmp_path / "miller.eik.nc"
    generate_miller_eik(
        {
            "Dimensions": {"ntheta": 8, "nperiod": 1},
            "Geometry": {
                "rhoc": 0.5,
                "q": 1.4,
                "s_hat": 0.8,
                "R0": 1.7,
                "akappa": 1.2,
            },
        },
        output,
    )

    assert output.exists()
