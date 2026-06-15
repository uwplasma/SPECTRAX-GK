"""Tests for reduced-model contract parsing."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

from spectraxgk import gx_reduced_models as legacy_reduced_models
from spectraxgk.reduced_model_contracts import load_reduced_model_contract

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from inspect_gx_reduced_model import build_parser


def test_load_reduced_model_contract_parses_cetg_input(tmp_path: Path) -> None:
    gx_input = tmp_path / "cetg.in"
    gx_input.write_text(
        """
[Dimensions]
ntheta = 48
ny = 128
nx = 128

[Domain]
x0 = 6.366
y0 = 6.366
z0 = 3.183
boundary = "periodic"

[Collisional_slab_ETG]
cetg = true

[Time]
t_max = 500.0
dt = 0.005
cfl = 1.0

[Initialization]
ikpar_init = 1
init_field = "density"
init_amp = 1.0e-2

[Geometry]
zero_shat = true

[Boltzmann]
Boltzmann_type = "ions"
tau_fac = 1.0
Z_ion = 1.0

[Dissipation]
hyper = true
D_hyper = 5.0e-4

[Expert]
dealias_kz = true
""",
        encoding="utf-8",
    )

    contract = load_reduced_model_contract(gx_input)

    assert contract.model == "cetg"
    assert contract.Nl == 2
    assert contract.Nm == 1
    assert contract.zero_shat is True
    assert contract.dealias_kz is True
    assert contract.D_hyper == pytest.approx(5.0e-4)


def test_legacy_reduced_model_module_alias_still_resolves(tmp_path: Path) -> None:
    gx_input = tmp_path / "cetg.in"
    gx_input.write_text(
        """
[Dimensions]
ntheta = 4
ny = 8
nx = 8

[Domain]
x0 = 1.0
y0 = 1.0
boundary = "periodic"

[Collisional_slab_ETG]
cetg = true

[Time]
dt = 0.1

[Initialization]
init_field = "density"
init_amp = 1.0e-3
""",
        encoding="utf-8",
    )

    contract = legacy_reduced_models.load_gx_reduced_model_contract(gx_input)
    assert contract.model == "cetg"
    assert (
        legacy_reduced_models.load_reduced_model_contract is load_reduced_model_contract
    )


def test_load_reduced_model_contract_parses_krehm_and_serializes(
    tmp_path: Path,
) -> None:
    gx_input = tmp_path / "krehm.in"
    gx_input.write_text(
        """
[Dimensions]
ntheta = 12
ny = 16
nx = 8
nlaguerre = 3
nhermite = 9

[Domain]
x0 = 2.0
y0 = 4.0
boundary = "linked"

[KREHM]
krehm = true

[Time]
dt = 0.025

[Initialization]
init_field = "density"
init_amp = 2.0e-4
""",
        encoding="utf-8",
    )

    contract = load_reduced_model_contract(gx_input)
    payload = contract.to_dict()

    assert contract.model == "krehm"
    assert contract.Nl == 3
    assert contract.Nm == 9
    assert contract.t_max is None
    assert payload["boundary"] == "linked"


def test_load_reduced_model_contract_rejects_unmarked_inputs(tmp_path: Path) -> None:
    gx_input = tmp_path / "ordinary.in"
    gx_input.write_text(
        """
[Dimensions]
ntheta = 4
ny = 8
nx = 8
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not a recognized reduced-model input"):
        load_reduced_model_contract(gx_input)


def test_inspect_gx_reduced_model_parser_accepts_json_flag() -> None:
    args = build_parser().parse_args(["/tmp/cetg.in", "--json"])
    assert args.gx_input == Path("/tmp/cetg.in")
    assert args.json is True
