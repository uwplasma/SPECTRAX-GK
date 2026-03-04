"""Configuration object tests."""

from spectraxgk.config import (
    CycloneBaseCase,
    ETGBaseCase,
    ETGModelConfig,
    GeometryConfig,
    GridConfig,
    KBMBaseCase,
    KineticElectronBaseCase,
    ModelConfig,
    TEMBaseCase,
    TimeConfig,
)


def test_config_to_dict():
    """All config dataclasses should serialize to dictionaries."""
    cfg = CycloneBaseCase()
    d = cfg.to_dict()
    assert set(d.keys()) == {"grid", "time", "geometry", "model", "init", "gx_reference"}
    assert d["geometry"]["q"] == cfg.geometry.q
    assert d["grid"]["y0"] == 20.0
    assert d["grid"]["ntheta"] == 32
    assert d["grid"]["nperiod"] == 2
    assert d["gx_reference"]["enabled"] is True


def test_config_override():
    """Overrides should propagate into the serialized representation."""
    grid = GridConfig(Nx=12, Ny=10, Nz=8)
    geom = GeometryConfig(q=1.7, s_hat=0.9, epsilon=0.2)
    model = ModelConfig(R_over_LTi=7.0, R_over_LTe=1.0, R_over_Ln=2.5)
    time = TimeConfig(t_max=1.0, dt=0.05, gx_real_fft=False)
    cfg = CycloneBaseCase(grid=grid, time=time, geometry=geom, model=model)
    d = cfg.to_dict()
    assert d["grid"]["Nx"] == 12
    assert d["geometry"]["q"] == 1.7
    assert d["model"]["R_over_LTe"] == 1.0
    assert d["time"]["dt"] == 0.05
    assert d["time"]["gx_real_fft"] is False


def test_etg_config_to_dict():
    """ETG configuration should serialize to dictionaries."""
    cfg = ETGBaseCase()
    d = cfg.to_dict()
    assert set(d.keys()) == {"grid", "time", "geometry", "model", "init"}
    assert d["model"]["R_over_LTe"] == cfg.model.R_over_LTe


def test_kinetic_config_to_dict():
    """Kinetic-electron configuration should serialize to dictionaries."""
    cfg = KineticElectronBaseCase()
    d = cfg.to_dict()
    assert d["model"]["R_over_LTi"] == cfg.model.R_over_LTi


def test_kbm_config_to_dict():
    """KBM configuration should serialize to dictionaries."""
    cfg = KBMBaseCase()
    d = cfg.to_dict()
    assert d["model"]["beta"] == cfg.model.beta


def test_tem_config_to_dict():
    """TEM configuration should serialize to dictionaries."""
    cfg = TEMBaseCase()
    d = cfg.to_dict()
    assert d["geometry"]["q"] == cfg.geometry.q
