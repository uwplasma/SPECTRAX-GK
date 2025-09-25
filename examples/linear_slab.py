from spectraxgk.io_config import read_toml
from spectraxgk.solver import run_simulation
from spectraxgk.post import load_result, plot_energy

cfg = read_toml("examples/linear_slab.toml")
info = run_simulation(cfg)
res = load_result(info["outfile"])
plot_energy(res)