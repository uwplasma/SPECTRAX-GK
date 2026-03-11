"""Tests for GX reduced-model contract parsing."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

from spectraxgk.gx_reduced_models import load_gx_reduced_model_contract

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from inspect_gx_reduced_model import build_parser


def test_load_gx_reduced_model_contract_parses_cetg_input(tmp_path: Path) -> None:
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

    contract = load_gx_reduced_model_contract(gx_input)

    assert contract.model == "cetg"
    assert contract.Nl == 2
    assert contract.Nm == 1
    assert contract.zero_shat is True
    assert contract.dealias_kz is True
    assert contract.D_hyper == pytest.approx(5.0e-4)


def test_inspect_gx_reduced_model_parser_accepts_json_flag() -> None:
    args = build_parser().parse_args(["/tmp/cetg.in", "--json"])
    assert args.gx_input == Path("/tmp/cetg.in")
    assert args.json is True
