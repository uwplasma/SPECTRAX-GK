#!/usr/bin/env python3
"""Digitize the W7-X test-4 zonal-flow reference curves from the paper figure."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from spectraxgk.plotting import set_plot_style


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_IMAGE_SHAPE = (1152, 2160)


@dataclass(frozen=True)
class PanelCalibration:
    label: str
    kx_rhoi: float
    main_box: tuple[int, int, int, int]
    inset_box: tuple[int, int, int, int]
    t_range: tuple[float, float]
    inset_t_range: tuple[float, float]
    y_range: tuple[float, float] = (-0.5, 1.0)
    inset_y_range: tuple[float, float] = (-0.1, 0.2)


PANEL_CALIBRATIONS = (
    PanelCalibration("a", 0.05, (216, 972, 57, 460), (648, 941, 74, 207), (0.0, 3500.0), (3300.0, 3500.0)),
    PanelCalibration("b", 0.07, (1188, 1944, 57, 460), (1620, 1913, 74, 207), (0.0, 2000.0), (1800.0, 2000.0)),
    PanelCalibration("c", 0.10, (216, 972, 610, 1013), (648, 941, 627, 760), (0.0, 2000.0), (1800.0, 2000.0)),
    PanelCalibration("d", 0.30, (1188, 1944, 610, 1013), (1620, 1913, 627, 760), (0.0, 2000.0), (1800.0, 2000.0)),
)

EXCLUSION_BOXES = (
    (560, 945, 70, 260),
    (1530, 1918, 70, 260),
    (560, 945, 625, 812),
    (1530, 1918, 625, 812),
    (705, 960, 335, 450),
    (1678, 1935, 335, 450),
    (705, 960, 888, 1005),
    (1678, 1935, 888, 1005),
)

REFERENCE_METADATA = {
    "paper": "Gonzalez-Jerez et al., Journal of Plasma Physics 88, 905880310 (2022)",
    "arxiv": "2107.06060",
    "figure": "Figure 11 / source figs/ZF.pdf",
    "observable": "line-averaged electrostatic potential normalized to its t=0 value",
    "test": "W7-X high-mirror bean-tube test 4 zonal-flow relaxation",
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure",
        type=Path,
        required=True,
        help="Path to the arXiv-source figs/ZF.pdf or a 2160x1152 PNG rendered from it at 2x.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=ROOT / "docs" / "_static" / "w7x_zonal_reference_digitized.csv",
        help="Output CSV containing the digitized main traces.",
    )
    parser.add_argument(
        "--out-residual-csv",
        type=Path,
        default=ROOT / "docs" / "_static" / "w7x_zonal_reference_digitized_residuals.csv",
        help="Output CSV containing the digitized inset residual levels.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=ROOT / "docs" / "_static" / "w7x_zonal_reference_digitized.json",
        help="Output JSON metadata path.",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=ROOT / "docs" / "_static" / "w7x_zonal_reference_digitized.png",
        help="Output QA plot path.",
    )
    parser.add_argument("--samples-per-trace", type=int, default=1001)
    return parser.parse_args(argv)


def _load_reference_image(path: Path) -> np.ndarray:
    """Load the source figure as RGB pixels using the calibration render size."""

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            import fitz
        except ImportError as exc:  # pragma: no cover - exercised only without PyMuPDF
            raise RuntimeError("PyMuPDF/fitz is required to render the source PDF figure") from exc
        doc = fitz.open(path)
        if len(doc) != 1:
            raise ValueError(f"{path} should contain one page")
        pix = doc[0].get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
        image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    else:
        image = Image.open(path).convert("RGB")
    arr = np.asarray(image, dtype=np.uint8)
    if arr.shape[:2] != EXPECTED_IMAGE_SHAPE:
        raise ValueError(
            f"{path} renders to {arr.shape[:2]}, expected {EXPECTED_IMAGE_SHAPE}; "
            "use the arXiv source figs/ZF.pdf or a 2x render of that file"
        )
    return arr


def _curve_mask(rgb: np.ndarray, code: str) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.int16)
    red = arr[:, :, 0]
    green = arr[:, :, 1]
    blue = arr[:, :, 2]
    if code == "stella":
        return (red > 170) & (green < 110) & (blue < 110) & ((red - green) > 70) & ((red - blue) > 70)
    if code == "GENE":
        return (blue > 140) & (red < 130) & (green < 150) & ((blue - red) > 50) & ((blue - green) > 30)
    raise ValueError("code must be 'stella' or 'GENE'")


def _apply_exclusions(mask: np.ndarray, *, box: tuple[int, int, int, int], exclusions: Iterable[tuple[int, int, int, int]]) -> np.ndarray:
    x0, _x1, y0, _y1 = box
    yy, xx = np.indices(mask.shape)
    global_x = xx + int(x0)
    global_y = yy + int(y0)
    out = np.array(mask, dtype=bool, copy=True)
    for ex0, ex1, ey0, ey1 in exclusions:
        out &= ~((global_x >= ex0) & (global_x <= ex1) & (global_y >= ey0) & (global_y <= ey1))
    return out


def _pixel_to_data(
    x_pixels: np.ndarray,
    y_pixels: np.ndarray,
    *,
    box: tuple[int, int, int, int],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    x0, x1, y0, y1 = box
    x = float(x_range[0]) + (np.asarray(x_pixels, dtype=float) - float(x0)) / float(x1 - x0) * float(x_range[1] - x_range[0])
    y = float(y_range[1]) - (np.asarray(y_pixels, dtype=float) - float(y0)) / float(y1 - y0) * float(y_range[1] - y_range[0])
    return x, y


def _digitize_box(
    image: np.ndarray,
    *,
    box: tuple[int, int, int, int],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    code: str,
    exclusions: Iterable[tuple[int, int, int, int]] = (),
) -> tuple[np.ndarray, np.ndarray]:
    x0, x1, y0, y1 = box
    sub = image[y0 : y1 + 1, x0 : x1 + 1]
    mask = _curve_mask(sub, code)
    if exclusions:
        mask = _apply_exclusions(mask, box=box, exclusions=exclusions)
    x_pixels: list[int] = []
    y_pixels: list[float] = []
    for local_x in range(mask.shape[1]):
        local_y = np.flatnonzero(mask[:, local_x])
        if local_y.size:
            x_pixels.append(local_x + x0)
            y_pixels.append(float(np.median(local_y + y0)))
    if not x_pixels:
        raise ValueError(f"no {code} pixels found in calibrated box {box}")
    x_data, y_data = _pixel_to_data(
        np.asarray(x_pixels, dtype=float),
        np.asarray(y_pixels, dtype=float),
        box=box,
        x_range=x_range,
        y_range=y_range,
    )
    order = np.argsort(x_data)
    return x_data[order], y_data[order]


def digitize_reference(image: np.ndarray, *, samples_per_trace: int = 1001) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return digitized main traces and inset residual summaries."""

    trace_rows: list[dict[str, object]] = []
    residual_rows: list[dict[str, object]] = []
    for panel in PANEL_CALIBRATIONS:
        for code in ("stella", "GENE"):
            t_raw, y_raw = _digitize_box(
                image,
                box=panel.main_box,
                x_range=panel.t_range,
                y_range=panel.y_range,
                code=code,
                exclusions=EXCLUSION_BOXES,
            )
            t_grid = np.linspace(float(panel.t_range[0]), float(panel.t_range[1]), int(samples_per_trace))
            y_grid = np.interp(t_grid, t_raw, y_raw)
            for t_value, response in zip(t_grid, y_grid, strict=True):
                trace_rows.append(
                    {
                        "panel": panel.label,
                        "kx_rhoi": panel.kx_rhoi,
                        "code": code,
                        "t_vti_over_a": float(t_value),
                        "response": float(response),
                    }
                )

            _t_inset, y_inset = _digitize_box(
                image,
                box=panel.inset_box,
                x_range=panel.inset_t_range,
                y_range=panel.inset_y_range,
                code=code,
            )
            residual_rows.append(
                {
                    "panel": panel.label,
                    "kx_rhoi": panel.kx_rhoi,
                    "code": code,
                    "residual_mean": float(np.mean(y_inset)),
                    "residual_median": float(np.median(y_inset)),
                    "residual_min": float(np.min(y_inset)),
                    "residual_max": float(np.max(y_inset)),
                    "n_pixels": int(y_inset.size),
                }
            )
    return pd.DataFrame(trace_rows), pd.DataFrame(residual_rows)


def _write_plot(trace_df: pd.DataFrame, residual_df: pd.DataFrame, out_png: Path) -> None:
    set_plot_style()
    colors = {"stella": "#d62728", "GENE": "#1f4fff"}
    styles = {"stella": "-", "GENE": "--"}
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.4), sharey=True, constrained_layout=True)
    for axis, panel in zip(axes.flat, PANEL_CALIBRATIONS, strict=True):
        subset = trace_df[trace_df["panel"] == panel.label]
        residual_subset = residual_df[residual_df["panel"] == panel.label]
        for code in ("stella", "GENE"):
            rows = subset[subset["code"] == code]
            axis.plot(
                rows["t_vti_over_a"],
                rows["response"],
                color=colors[code],
                linestyle=styles[code],
                linewidth=2.0,
                label=code,
            )
            residual = residual_subset[residual_subset["code"] == code]["residual_median"].iloc[0]
            axis.axhline(float(residual), color=colors[code], linestyle=styles[code], linewidth=1.2, alpha=0.55)
        axis.set_title(fr"$k_x \rho_i = {panel.kx_rhoi:.2f}$")
        axis.set_xlabel(r"$t v_{th,i}/a$")
        axis.set_ylabel(r"$\langle \mathrm{Re}(\hat{\phi}_k)\rangle_z / \langle \mathrm{Re}(\hat{\phi}_k)\rangle_z(t=0)$")
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False, loc="upper right")
    fig.suptitle("Digitized W7-X test-4 zonal-flow reference (Gonzalez-Jerez et al. 2022)")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    image = _load_reference_image(args.figure)
    trace_df, residual_df = digitize_reference(image, samples_per_trace=int(args.samples_per_trace))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    trace_df.to_csv(args.out_csv, index=False)
    residual_df.to_csv(args.out_residual_csv, index=False)
    _write_plot(trace_df, residual_df, args.out_png)

    payload = {
        **REFERENCE_METADATA,
        "source_figure": str(args.figure),
        "digitization_method": (
            "Color-threshold digitization of the arXiv source Figure 11 PDF rendered at 2x. "
            "Main traces exclude inset and legend boxes; residual levels are read from the insets."
        ),
        "image_shape": list(image.shape[:2]),
        "trace_csv": _repo_relative(args.out_csv),
        "residual_csv": _repo_relative(args.out_residual_csv),
        "qa_plot": _repo_relative(args.out_png),
        "calibrations": [asdict(item) for item in PANEL_CALIBRATIONS],
        "residual_summary": residual_df.to_dict(orient="records"),
        "validation_status": "reference",
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_residual_csv}")
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
