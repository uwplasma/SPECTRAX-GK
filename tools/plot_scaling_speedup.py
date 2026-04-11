"""Plot speedup curves from scaling_speedup_data.csv."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from spectraxgk.plotting import set_plot_style


def main() -> None:
    data_path = Path("docs/_static/scaling_speedup_data.csv")
    df = pd.read_csv(data_path)

    set_plot_style()
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 6.0), sharex=True)
    ax0, ax1 = axes

    for backend, color in [("cpu", "#1f77b4"), ("cuda", "#ff7f0e")]:
        sub = df[df["backend"] == backend]
        steps = sorted(sub["steps"].unique())
        speedups = []
        runtimes_1 = []
        runtimes_2 = []
        for step in steps:
            sub_step = sub[sub["steps"] == step]
            t1 = float(sub_step[sub_step["devices"] == 1]["elapsed_s"].iloc[0])
            t2 = float(sub_step[sub_step["devices"] == 2]["elapsed_s"].iloc[0])
            speedups.append(t1 / t2)
            runtimes_1.append(t1)
            runtimes_2.append(t2)
        ax0.plot(steps, speedups, marker="o", color=color, label=f"{backend.upper()} 2x")
        ax1.plot(steps, runtimes_1, marker="o", linestyle="--", color=color, label=f"{backend.upper()} 1x")
        ax1.plot(steps, runtimes_2, marker="s", linestyle="-", color=color, label=f"{backend.upper()} 2x")

    ax0.axhline(2.0, color="#444444", linestyle=":", linewidth=1.0, label="ideal")
    ax0.set_ylabel("Speedup (1x / 2x)")
    ax0.set_title("SPECTRAX-GK scaling (Ny=64, Nz=128, Nl=6, Nm=6)")
    ax0.legend(loc="best")

    ax1.set_xlabel("Steps (dt=0.05, Tsit5)")
    ax1.set_ylabel("Wall time [s]")
    ax1.legend(loc="best", ncol=2)

    fig.tight_layout()
    out_png = Path("docs/_static/scaling_speedup.png")
    out_pdf = Path("docs/_static/scaling_speedup.pdf")
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
