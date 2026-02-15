import numpy as np

from spectraxgk.benchmarks import load_cyclone_reference, run_cyclone_linear
from spectraxgk.plotting import cyclone_reference_figure


def main():
    ref = load_cyclone_reference()
    fig, _axes = cyclone_reference_figure(ref)
    fig.savefig("cyclone_reference.png", dpi=200)

    ky_target = 0.3
    result = run_cyclone_linear(ky_target=ky_target, steps=200, dt=0.05, tmin=5.0)
    idx = int(np.argmin(np.abs(ref.ky - ky_target)))
    print(f"Cyclone reference ky={result.ky:.3f} gamma_ref={ref.gamma[idx]:.6f} omega_ref={ref.omega[idx]:.6f}")
    print(f"SPECTRAX-GK (streaming-only) gamma={result.gamma:.6f} omega={result.omega:.6f}")


if __name__ == "__main__":
    main()
