from spectraxgk.config import CycloneBaseCase, GeometryConfig, GridConfig, ModelConfig, TimeConfig


def test_config_to_dict():
    cfg = CycloneBaseCase()
    d = cfg.to_dict()
    assert set(d.keys()) == {"grid", "time", "geometry", "model"}
    assert d["geometry"]["q"] == cfg.geometry.q


def test_config_override():
    grid = GridConfig(Nx=12, Ny=10, Nz=8)
    geom = GeometryConfig(q=1.7, s_hat=0.9, epsilon=0.2)
    model = ModelConfig(R_over_LTi=7.0, R_over_LTe=1.0, R_over_Ln=2.5)
    time = TimeConfig(t_max=1.0, dt=0.05)
    cfg = CycloneBaseCase(grid=grid, time=time, geometry=geom, model=model)
    d = cfg.to_dict()
    assert d["grid"]["Nx"] == 12
    assert d["geometry"]["q"] == 1.7
    assert d["model"]["R_over_LTe"] == 1.0
    assert d["time"]["dt"] == 0.05
