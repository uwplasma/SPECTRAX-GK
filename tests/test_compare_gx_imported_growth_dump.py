from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from compare_gx_imported_growth_dump import (
    _expand_gx_restart_state_to_full_positive_ky,
    _load_growth_dt,
    _load_gx_restart_state,
    _load_gx_restart_time,
    build_parser,
)


def test_compare_gx_imported_growth_dump_parser_accepts_required_paths() -> None:
    args = build_parser().parse_args(
        [
            "--gx-dir-start",
            "/tmp/start",
            "--gx-dir-stop",
            "/tmp/stop",
            "--gx-out",
            "/tmp/run.out.nc",
            "--gx-input",
            "/tmp/run.in",
            "--geometry-file",
            "/tmp/geom.nc",
            "--time-index-start",
            "10",
            "--time-index-stop",
            "11",
        ]
    )
    assert args.gx_dir_start == Path("/tmp/start")
    assert args.gx_dir_stop == Path("/tmp/stop")
    assert args.gx_out == Path("/tmp/run.out.nc")
    assert args.gx_input == Path("/tmp/run.in")
    assert args.geometry_file == Path("/tmp/geom.nc")
    assert args.time_index_start == 10
    assert args.time_index_stop == 11


def test_compare_gx_imported_growth_dump_parser_accepts_restart_start() -> None:
    args = build_parser().parse_args(
        [
            "--gx-dir-start",
            "/tmp/start",
            "--gx-dir-stop",
            "/tmp/stop",
            "--gx-restart-start",
            "/tmp/restart.nc",
            "--gx-out",
            "/tmp/run.out.nc",
            "--gx-input",
            "/tmp/run.in",
            "--geometry-file",
            "/tmp/geom.nc",
            "--time-index-start",
            "10",
            "--time-index-stop",
            "11",
        ]
    )
    assert args.gx_restart_start == Path("/tmp/restart.nc")


def test_load_growth_dt_accepts_float64_scalar(tmp_path: Path) -> None:
    path = tmp_path / "diag_growth_dt_t45.bin"
    import numpy as np

    np.asarray([2.5e-4], dtype=np.float64).tofile(path)
    assert _load_growth_dt(path) == 2.5e-4


def test_load_gx_restart_state_transposes_to_spectrax_layout(tmp_path: Path) -> None:
    import numpy as np
    from netCDF4 import Dataset

    path = tmp_path / "restart.nc"
    with Dataset(path, "w") as root:
        root.createDimension("Nspecies", 1)
        root.createDimension("Nm", 3)
        root.createDimension("Nl", 2)
        root.createDimension("Nz", 5)
        root.createDimension("Nkx", 1)
        root.createDimension("Nky", 4)
        root.createDimension("ri", 2)
        t = root.createVariable("time", "f8", ())
        t.assignValue(3.25)
        g = root.createVariable("G", "f4", ("Nspecies", "Nm", "Nl", "Nz", "Nkx", "Nky", "ri"))
        raw = np.zeros((1, 3, 2, 5, 1, 4, 2), dtype=np.float32)
        raw[0, 2, 1, 4, 0, 3, 0] = 7.0
        raw[0, 2, 1, 4, 0, 3, 1] = -2.0
        g[:] = raw

    state = _load_gx_restart_state(path)
    assert state.shape == (1, 2, 3, 4, 1, 5)
    assert state[0, 1, 2, 3, 0, 4] == np.complex64(7.0 - 2.0j)
    assert _load_gx_restart_time(path) == 3.25


def test_expand_gx_restart_state_to_full_positive_ky_embeds_dealiased_kx() -> None:
    import numpy as np

    # ny_full=16 -> nyc_full=9, active naky=6; nx_full=4 -> active nakx=3
    active = np.zeros((1, 1, 1, 6, 3, 2), dtype=np.complex64)
    active[0, 0, 0, 5, 0, 1] = 1.0 + 2.0j
    active[0, 0, 0, 5, 1, 1] = 3.0 + 4.0j
    active[0, 0, 0, 5, 2, 1] = 5.0 + 6.0j
    full = _expand_gx_restart_state_to_full_positive_ky(active, ny_full=16, nx_full=4)
    assert full.shape == (1, 1, 1, 9, 4, 2)
    assert full[0, 0, 0, 5, 0, 1] == np.complex64(1.0 + 2.0j)
    assert full[0, 0, 0, 5, 1, 1] == np.complex64(3.0 + 4.0j)
    assert full[0, 0, 0, 5, 3, 1] == np.complex64(5.0 + 6.0j)
    assert full[0, 0, 0, 6, 0, 1] == 0.0j
