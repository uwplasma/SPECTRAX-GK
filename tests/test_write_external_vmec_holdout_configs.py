from pathlib import Path
import json

import pytest

from tools.write_external_vmec_holdout_configs import (
    _parse_grid,
    _parse_horizons,
    write_configs,
    write_manifest,
)


def test_write_external_vmec_holdout_configs_restart_ladder(tmp_path: Path) -> None:
    vmec_file = tmp_path / "wout_fixture.nc"
    vmec_file.write_text("placeholder", encoding="utf-8")

    written = write_configs(
        case="candidate",
        vmec_file=vmec_file,
        out_dir=tmp_path / "runs",
        grids=[_parse_grid("n8:8:8:6:6"), _parse_grid("n10:10:10:8:8")],
        horizons=(1.0, 1.5, 2.0),
        dt=0.25,
        ky=0.3,
        nl=2,
        nm=3,
    )
    assert len(written) == 6
    assert [item.steps for item in written] == [4, 4, 2, 2, 2, 2]
    assert [item.restart_if_exists for item in written] == [False, False, True, True, True, True]

    first_config = written[0].path.read_text(encoding="utf-8")
    assert f'vmec_file = "{vmec_file.resolve().as_posix()}"' in first_config
    assert "Nx = 8" in first_config
    assert "Nz = 6" in first_config
    assert "ky = 0.3" in first_config
    assert "steps = 4" in first_config
    assert "restart_if_exists = false" in first_config
    assert 'path = "candidate_nonlinear_t1_n8.out.nc"' in first_config

    continuation_config = written[2].path.read_text(encoding="utf-8")
    assert "t_max = 1.5" in continuation_config
    assert "steps = 2" in continuation_config
    assert "restart_if_exists = true" in continuation_config

    manifest = write_manifest(tmp_path / "runs", written)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["kind"] == "external_vmec_holdout_config_manifest"
    assert len(payload["configs"]) == 6
    assert len(payload["launch_commands"]) == 6
    assert len(payload["restart_seed_commands"]) == 4
    assert "python3 -m spectraxgk.cli run" in payload["launch_commands"][0]
    assert "restart.nc" in payload["restart_seed_commands"][0]
    assert "candidate_nonlinear_t1p5_n8" in payload["restart_seed_commands"][0]
    assert "candidate_nonlinear_t1p5_n8" in payload["restart_seed_commands"][2]
    assert "candidate_nonlinear_t2_n8" in payload["restart_seed_commands"][2]


def test_write_external_vmec_holdout_configs_replicate_variants(tmp_path: Path) -> None:
    vmec_file = tmp_path / "wout_fixture.nc"
    vmec_file.write_text("placeholder", encoding="utf-8")

    written = write_configs(
        case="replicate",
        vmec_file=vmec_file,
        out_dir=tmp_path / "runs",
        grids=[_parse_grid("n8:8:8:6:6")],
        horizons=(1.0, 2.0),
        dt=0.25,
        ky=0.3,
        nl=2,
        nm=3,
        baseline_seed=22,
        seed_variants=[31, 32],
        dt_variants=[0.2, 0.125],
    )

    assert len(written) == 8
    labels = [item.variant.label if item.variant is not None else "" for item in written]
    assert labels == [
        "seed31",
        "seed31",
        "seed32",
        "seed32",
        "dt0p2",
        "dt0p2",
        "dt0p125",
        "dt0p125",
    ]
    assert [item.steps for item in written] == [4, 4, 4, 4, 5, 5, 8, 8]

    seed_config = written[0].path.read_text(encoding="utf-8")
    assert 'path = "replicate_nonlinear_t1_n8_seed31.out.nc"' in seed_config
    assert "random_seed = 31" in seed_config
    assert 'variant_axis = "seed"' in seed_config
    assert 'variant_label = "seed31"' in seed_config
    assert "timestep = 0.25" in seed_config

    dt_config = written[-1].path.read_text(encoding="utf-8")
    assert 'path = "replicate_nonlinear_t2_n8_dt0p125.out.nc"' in dt_config
    assert "dt = 0.125" in dt_config
    assert "random_seed = 22" in dt_config
    assert 'variant_axis = "timestep"' in dt_config
    assert "timestep = 0.125" in dt_config

    manifest = write_manifest(tmp_path / "runs", written)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert len(payload["restart_seed_commands"]) == 4
    assert "replicate_nonlinear_t1_n8_seed31" in payload["restart_seed_commands"][0]
    assert "replicate_nonlinear_t2_n8_seed31" in payload["restart_seed_commands"][0]
    assert "replicate_nonlinear_t1_n8_dt0p125" in payload["restart_seed_commands"][-1]
    assert "replicate_nonlinear_t2_n8_dt0p125" in payload["restart_seed_commands"][-1]
    variants = [item["variant"] for item in payload["configs"]]
    assert variants[0] == {"axis": "seed", "label": "seed31", "seed": 31, "timestep": 0.25}
    assert variants[-1] == {"axis": "timestep", "label": "dt0p125", "seed": 22, "timestep": 0.125}


def test_external_vmec_holdout_config_parsers_reject_bad_inputs(tmp_path: Path) -> None:
    assert _parse_horizons("1,2,3") == (1.0, 2.0, 3.0)
    with pytest.raises(ValueError, match="sorted"):
        _parse_horizons("2,1")
    with pytest.raises(ValueError, match="label:Nx:Ny:Nz:ntheta"):
        _parse_grid("bad")
    with pytest.raises(ValueError, match="positive"):
        _parse_grid("n0:0:8:8:8")
    with pytest.raises(ValueError, match="dt variants must be positive"):
        write_configs(
            case="bad",
            vmec_file=Path("wout.nc"),
            out_dir=tmp_path / "unused",
            grids=[_parse_grid("n8:8:8:6:6")],
            dt_variants=[0.0],
        )
