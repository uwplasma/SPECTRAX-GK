"""Tests for canonical normalization contracts."""

from __future__ import annotations

import pytest

from spectraxgk import benchmarks
from spectraxgk.normalization import (
    apply_diagnostic_normalization,
    get_normalization_contract,
)


def test_get_normalization_contract_aliases() -> None:
    kin = get_normalization_contract("kinetic")
    kin_alias = get_normalization_contract("kinetic_itg")
    assert kin == kin_alias


def test_get_normalization_contract_unknown_case() -> None:
    with pytest.raises(ValueError, match="Unknown normalization case"):
        get_normalization_contract("not-a-case")


def test_apply_diagnostic_normalization_modes() -> None:
    gamma, omega = apply_diagnostic_normalization(
        0.2, -0.3, rho_star=0.5, diagnostic_norm="gx"
    )
    assert gamma == pytest.approx(0.1)
    assert omega == pytest.approx(-0.15)

    gamma_none, omega_none = apply_diagnostic_normalization(
        0.2, -0.3, rho_star=0.5, diagnostic_norm="none"
    )
    assert gamma_none == pytest.approx(0.2)
    assert omega_none == pytest.approx(-0.3)


def test_benchmark_constants_follow_contract() -> None:
    cyclone = get_normalization_contract("cyclone")
    etg = get_normalization_contract("etg")
    kbm = get_normalization_contract("kbm")

    assert benchmarks.CYCLONE_OMEGA_D_SCALE == pytest.approx(cyclone.omega_d_scale)
    assert benchmarks.CYCLONE_OMEGA_STAR_SCALE == pytest.approx(cyclone.omega_star_scale)
    assert benchmarks.CYCLONE_RHO_STAR == pytest.approx(cyclone.rho_star)

    assert benchmarks.ETG_OMEGA_D_SCALE == pytest.approx(etg.omega_d_scale)
    assert benchmarks.ETG_OMEGA_STAR_SCALE == pytest.approx(etg.omega_star_scale)
    assert benchmarks.ETG_RHO_STAR == pytest.approx(etg.rho_star)

    assert benchmarks.KBM_OMEGA_D_SCALE == pytest.approx(kbm.omega_d_scale)
    assert benchmarks.KBM_OMEGA_STAR_SCALE == pytest.approx(kbm.omega_star_scale)
    assert benchmarks.KBM_RHO_STAR == pytest.approx(kbm.rho_star)

