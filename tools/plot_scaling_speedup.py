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
    fig, ax0 = plt.subplots(1, 1, figsize=(7.2, 4.6))

    for backend, color in [("cpu", "#1f77b4"), ("cuda", "#ff7f0e")]:
        sub = df[df["backend"] == backend]
        steps = sorted(sub["steps"].unique())
        speedups = []
        for step in steps:
            sub_step = sub[sub["steps"] == step]
            t1 = float(sub_step[sub_step["devices"] == 1]["elapsed_s"].iloc[0])
            t2 = float(sub_step[sub_step["devices"] == 2]["elapsed_s"].iloc[0])
            speedups.append(t1 / t2)
        ax0.plot(steps, speedups, marker="o", linewidth=2.4, color=color, label=f"{backend.upper()} 2 devices")

    ax0.axhline(2.0, color="#444444", linestyle=":", linewidth=1.0, label="ideal")
    ax0.set_ylabel("Speedup (1x / 2x)")
    ax0.set_xlabel("Linear integration steps per run")
    ax0.set_xticks(sorted(df[df["backend"].isin(["cpu", "cuda"])]["steps"].unique()))
    ax0.set_title("Two-device diffrax scaling (Ny=64, Nz=128, Nl=6, Nm=6)")
    ax0.legend(loc="lower right", frameon=False)
    ax0.grid(True, alpha=0.25)

    fig.tight_layout()
    out_png = Path("docs/_static/scaling_speedup.png")
    out_pdf = Path("docs/_static/scaling_speedup.pdf")
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
