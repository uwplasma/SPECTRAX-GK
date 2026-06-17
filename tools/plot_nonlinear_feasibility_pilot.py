#!/usr/bin/env python3
"""Plot finite nonlinear feasibility pilots without promoting transport gates."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from spectraxgk.plotting import set_plot_style  # noqa: E402
from spectraxgk.workflows.runtime.artifacts import load_nonlinear_netcdf_diagnostics  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "docs" / "_static" / "external_vmec_nonlinear_feasibility.png"
DEFAULT_FRACTIONS = (0.5, 0.6, 0.7, 0.8)


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, np.generic):
        return _json_clean(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _as_1d(values: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return arr


def window_summaries(
    t: Any,
    heat_flux: Any,
    wphi: Any,
    *,
    start_fractions: tuple[float, ...] = DEFAULT_FRACTIONS,
) -> list[dict[str, float | int]]:
    """Return late-window statistics for a nonlinear feasibility trace."""

    t_arr = _as_1d(t, name="t")
    heat_arr = _as_1d(heat_flux, name="heat_flux")
    wphi_arr = _as_1d(wphi, name="wphi")
    if not (t_arr.size == heat_arr.size == wphi_arr.size):
        raise ValueError("t, heat_flux, and wphi must have the same length")
    if t_arr.size < 3:
        raise ValueError("at least three samples are required")
    out: list[dict[str, float | int]] = []
    for fraction in start_fractions:
        if not 0.0 <= float(fraction) < 1.0:
            raise ValueError("start fractions must satisfy 0 <= fraction < 1")
        start = min(int(t_arr.size * float(fraction)), t_arr.size - 2)
        tt = t_arr[start:]
        heat = heat_arr[start:]
        wphi_win = wphi_arr[start:]
        slope = float(np.polyfit(tt, heat, 1)[0]) if tt.size >= 2 else float("nan")
        heat_mean = float(np.mean(heat))
        out.append(
            {
                "start_fraction": float(fraction),
                "start_index": int(start),
                "tmin": float(tt[0]),
                "tmax": float(tt[-1]),
                "n_samples": int(tt.size),
                "heat_flux_mean": heat_mean,
                "heat_flux_std": float(np.std(heat)),
                "heat_flux_last": float(heat_arr[-1]),
                "heat_flux_slope": slope,
                "heat_flux_relative_slope_per_time": float(
                    slope / max(abs(heat_mean), 1.0e-300)
                ),
                "wphi_mean": float(np.mean(wphi_win)),
                "wphi_std": float(np.std(wphi_win)),
                "wphi_last": float(wphi_arr[-1]),
            }
        )
    return out


def load_trace_from_gx_netcdf(path: str | Path) -> dict[str, np.ndarray]:
    """Load the scalar trace needed for a nonlinear pilot panel."""

    diag = load_nonlinear_netcdf_diagnostics(path)
    return {
        "t": _as_1d(diag.t, name="t"),
        "heat_flux": _as_1d(diag.heat_flux_t, name="heat_flux"),
        "wphi": _as_1d(diag.Wphi_t, name="wphi"),
        "wg": _as_1d(diag.Wg_t, name="wg"),
    }


def write_trace_csv(path: str | Path, trace: dict[str, np.ndarray]) -> None:
    """Write scalar nonlinear pilot traces to CSV."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["t", "heat_flux", "wphi", "wg"]
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(keys)
        for values in zip(
            *(np.asarray(trace[key], dtype=float) for key in keys), strict=True
        ):
            writer.writerow([f"{float(value):.16e}" for value in values])


def write_pilot_panel(
    trace: dict[str, np.ndarray],
    *,
    out: str | Path = DEFAULT_OUT,
    source: str | Path | None = None,
    title: str = "Nonlinear Feasibility Pilot",
    label: str = "external VMEC",
    claim_level: str = "finite_nonlinear_feasibility_not_transport_validation",
    start_fractions: tuple[float, ...] = DEFAULT_FRACTIONS,
) -> dict[str, str]:
    """Write a nonlinear feasibility PNG/PDF/JSON/CSV artifact set."""

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t = _as_1d(trace["t"], name="t")
    heat = _as_1d(trace["heat_flux"], name="heat_flux")
    wphi = _as_1d(trace["wphi"], name="wphi")
    wg = _as_1d(trace.get("wg", np.zeros_like(t)), name="wg")
    summaries = window_summaries(t, heat, wphi, start_fractions=start_fractions)
    chosen = min(
        summaries,
        key=lambda item: abs(float(item["heat_flux_relative_slope_per_time"])),
    )

    set_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.0), constrained_layout=True)
    ax_heat, ax_wphi, ax_zoom, ax_text = axes.ravel()

    heat_plot = np.maximum(np.abs(heat), 1.0e-300)
    wphi_plot = np.maximum(np.abs(wphi), 1.0e-300)
    ax_heat.semilogy(
        t, heat_plot, marker="o", markersize=3.2, linewidth=2.0, color="#0f4c81"
    )
    ax_heat.set_xlabel("time")
    ax_heat.set_ylabel("|heat flux|")
    ax_heat.set_title("Transport trace")
    ax_heat.grid(True, alpha=0.25)

    ax_wphi.semilogy(
        t, wphi_plot, marker="s", markersize=3.2, linewidth=2.0, color="#c44e52"
    )
    ax_wphi.set_xlabel("time")
    ax_wphi.set_ylabel(r"$W_\phi$")
    ax_wphi.set_title("Field-energy trace")
    ax_wphi.grid(True, alpha=0.25)

    colors = ["#2a9d8f", "#b45309", "#7c3aed", "#6b7280"]
    for summary, color in zip(summaries, colors, strict=False):
        ax_heat.axvspan(
            float(summary["tmin"]), float(summary["tmax"]), color=color, alpha=0.08
        )
        ax_wphi.axvspan(
            float(summary["tmin"]), float(summary["tmax"]), color=color, alpha=0.08
        )
    start_idx = int(chosen["start_index"])
    late_t = t[start_idx:]
    late_heat = heat[start_idx:]
    ax_zoom.plot(
        late_t,
        late_heat,
        marker="o",
        markersize=3.5,
        linewidth=2.0,
        color="#0f4c81",
        label="heat flux",
    )
    ax_zoom.axhline(
        float(chosen["heat_flux_mean"]),
        color="#c44e52",
        linewidth=1.8,
        label="window mean",
    )
    ax_zoom.fill_between(
        late_t,
        float(chosen["heat_flux_mean"]) - float(chosen["heat_flux_std"]),
        float(chosen["heat_flux_mean"]) + float(chosen["heat_flux_std"]),
        color="#c44e52",
        alpha=0.12,
        label=r"$\pm 1\sigma$",
    )
    ax_zoom.set_xlabel("time")
    ax_zoom.set_ylabel("heat flux")
    ax_zoom.set_title(f"Least-trending window starts at t={float(chosen['tmin']):.2f}")
    ax_zoom.grid(True, alpha=0.25)
    ax_zoom.legend(frameon=False)

    ax_text.axis("off")
    lines = [
        label,
        f"claim: {claim_level}",
        f"samples: {t.size}, t=[{t[0]:.3g}, {t[-1]:.3g}]",
        f"final heat flux: {heat[-1]:.4g}",
        f"final W_phi: {wphi[-1]:.4g}",
        f"least-trending window: [{float(chosen['tmin']):.3g}, {float(chosen['tmax']):.3g}]",
        f"window heat mean: {float(chosen['heat_flux_mean']):.4g}",
        f"window heat std: {float(chosen['heat_flux_std']):.4g}",
        f"relative slope/time: {float(chosen['heat_flux_relative_slope_per_time']):.3g}",
        "",
        "Interpretation:",
        "finite long pilot; not promoted unless",
        "a saturated-window gate is defined and passed.",
    ]
    ax_text.text(
        0.02,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
    )

    fig.suptitle(title, fontsize=14)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    csv_path = out_path.with_suffix(".traces.csv")
    write_trace_csv(csv_path, {"t": t, "heat_flux": heat, "wphi": wphi, "wg": wg})
    json_path = out_path.with_suffix(".json")
    payload = {
        "kind": "nonlinear_feasibility_pilot",
        "claim_level": claim_level,
        "label": label,
        "source": None if source is None else str(source),
        "png": str(out_path),
        "pdf": str(pdf_path),
        "csv": str(csv_path),
        "n_samples": int(t.size),
        "tmin": float(t[0]),
        "tmax": float(t[-1]),
        "heat_flux_last": float(heat[-1]),
        "wphi_last": float(wphi[-1]),
        "wg_last": float(wg[-1]),
        "window_summaries": summaries,
        "least_trending_window": chosen,
        "promotion_gate": {
            "passed": False,
            "reason": "feasibility pilot only; no external nonlinear transport acceptance gate is defined",
        },
    }
    json_path.write_text(
        json.dumps(_json_clean(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "png": str(out_path),
        "pdf": str(pdf_path),
        "json": str(json_path),
        "csv": str(csv_path),
    }


def _parse_fractions(raw: str) -> tuple[float, ...]:
    return tuple(float(item) for item in raw.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, help="GX-style nonlinear *.out.nc file."
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output PNG path.")
    parser.add_argument("--title", default="Nonlinear Feasibility Pilot")
    parser.add_argument("--label", default="external VMEC")
    parser.add_argument(
        "--claim-level", default="finite_nonlinear_feasibility_not_transport_validation"
    )
    parser.add_argument("--fractions", default="0.5,0.6,0.7,0.8")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    trace = load_trace_from_gx_netcdf(args.input)
    paths = write_pilot_panel(
        trace,
        out=args.out,
        source=args.input,
        title=args.title,
        label=args.label,
        claim_level=args.claim_level,
        start_fractions=_parse_fractions(args.fractions),
    )
    print(f"saved {paths['png']}")
    print(f"saved {paths['pdf']}")
    print(f"saved {paths['json']}")
    print(f"saved {paths['csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
