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
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 7.0))
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

    ax0.axhline(2.0, color="#444444", linestyle=":", linewidth=1.0, label="ideal")
    ax0.set_ylabel("Speedup (1x / 2x)")
    ax0.set_title("SPECTRAX-GK 2x scaling (Ny=64, Nz=128, Nl=6, Nm=6)")
    ax0.legend(loc="best")

    strong_backends = [
        ("cpu_sharded_large", "CPU", "#1f77b4"),
        ("cuda_sharded_large", "GPU", "#ff7f0e"),
    ]
    max_devices = 0
    for backend, label, color in strong_backends:
        sub = df[df["backend"] == backend]
        if sub.empty:
            continue
        devices = sorted(sub["devices"].unique())
        t1 = float(sub[sub["devices"] == 1]["elapsed_s"].iloc[0])
        speedups = [t1 / float(sub[sub["devices"] == d]["elapsed_s"].iloc[0]) for d in devices]
        ax1.plot(devices, speedups, marker="o", color=color, label=label)
        max_devices = max(max_devices, max(devices))
    if max_devices:
        ax1.plot(
            list(range(1, max_devices + 1)),
            list(range(1, max_devices + 1)),
            linestyle=":",
            color="#444444",
            label="ideal",
        )
        ax1.set_xticks(list(range(1, max_devices + 1)))
    ax1.set_xlabel("Devices")
    ax1.set_ylabel("Speedup (1x / N)")
    ax1.set_title("Strong scaling (sharded linear RK2, Ny=128, Nz=256, Nl=8, Nm=8)")
    ax1.legend(loc="best", ncol=2)

    fig.tight_layout()
    out_png = Path("docs/_static/scaling_speedup.png")
    out_pdf = Path("docs/_static/scaling_speedup.pdf")
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
