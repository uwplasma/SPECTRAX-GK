"""Digitize reference curves from published figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import rgb2hsv


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "src" / "spectraxgk" / "data"


def _curve_from_mask(mask: np.ndarray, x0: int, x1: int, y0: int, y1: int) -> np.ndarray:
    curve = []
    for x in range(x0, x1):
        ys = np.where(mask[y0:y1, x])[0]
        if ys.size:
            y = y0 + float(np.median(ys))
            curve.append((float(x), y))
    if not curve:
        raise RuntimeError("No curve pixels found in selected region")
    return np.array(curve)


def _map_curve(
    curve: np.ndarray,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    x_pix = curve[:, 0]
    y_pix = curve[:, 1]
    x_data = x_min + (x_pix - x0) / (x1 - x0) * (x_max - x_min)
    y_data = y_max - (y_pix - y0) / (y1 - y0) * (y_max - y_min)
    return x_data, y_data


def _cluster_ranges(indices: np.ndarray, gap: int = 1) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start: int | None = None
    prev: int | None = None
    for idx in indices:
        if start is None:
            start = int(idx)
            prev = int(idx)
            continue
        if idx == prev + gap:
            prev = int(idx)
            continue
        ranges.append((start, int(prev)))
        start = int(idx)
        prev = int(idx)
    if start is not None and prev is not None:
        ranges.append((start, int(prev)))
    return ranges


def _detect_horizontal_lines(
    gray: np.ndarray,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    *,
    thresh: int = 120,
    frac: float = 0.6,
) -> tuple[int, int]:
    sub = gray[y0:y1, x0:x1]
    mask = sub < thresh
    counts = mask.sum(axis=1)
    rows = np.where(counts > frac * (x1 - x0))[0]
    if rows.size == 0:
        raise RuntimeError("Unable to detect horizontal axis lines for ETG digitization")
    ranges = _cluster_ranges(rows)
    return y0 + ranges[0][0], y0 + ranges[-1][0]


def _detect_vertical_lines(
    gray: np.ndarray,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    *,
    thresh: int = 120,
    frac: float = 0.6,
) -> tuple[int, int]:
    sub = gray[y0:y1, x0:x1]
    mask = sub < thresh
    counts = mask.sum(axis=0)
    cols = np.where(counts > frac * (y1 - y0))[0]
    if cols.size == 0:
        raise RuntimeError("Unable to detect vertical axis lines for ETG digitization")
    ranges = _cluster_ranges(cols)
    if len(ranges) == 1:
        return x0 + ranges[0][0], x0 + ranges[0][1]
    return x0 + ranges[0][0], x0 + ranges[-1][0]


def digitize_etg(fig_path: Path, out_path: Path) -> None:
    """Digitize ETG growth rates and frequencies from the GX paper figure."""

    arr = np.array(Image.open(fig_path).convert("RGB"))
    gray = np.array(Image.open(fig_path).convert("L"))
    hsv = rgb2hsv(arr / 255.0)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    blue = (h > 0.55) & (h < 0.75) & (s > 0.3) & (v > 0.3)

    # Rough bounding box for Fig. 2(b) in the GX paper (page render).
    x0_guess, x1_guess = 520, 1100
    y_top_guess0, y_top_guess1 = 700, 900
    y_bot_guess0, y_bot_guess1 = 940, 1130

    x_left, x_right = _detect_vertical_lines(gray, x0_guess, x1_guess, y_bot_guess0, y_bot_guess1)
    yt0, yt1 = _detect_horizontal_lines(gray, x0_guess, x1_guess, y_top_guess0, y_top_guess1)
    yb0, yb1 = _detect_horizontal_lines(gray, x0_guess, x1_guess, y_bot_guess0, y_bot_guess1)

    curve_top = _curve_from_mask(blue, x_left, x_right + 1, yt0, yt1)
    curve_bot = _curve_from_mask(blue, x_left, x_right + 1, yb0, yb1)

    x_min, x_max = 0.0, 50.0
    y_gamma_min, y_gamma_max = 0.0, 10.0
    y_omega_min, y_omega_max = -40.0, 0.0

    x_top, y_top = _map_curve(curve_top, x_left, x_right, yt0, yt1, x_min, x_max, y_gamma_min, y_gamma_max)
    x_bot, y_bot = _map_curve(curve_bot, x_left, x_right, yb0, yb1, x_min, x_max, y_omega_min, y_omega_max)

    ky = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50], dtype=float)
    order = np.argsort(x_top)
    gamma = np.interp(ky, x_top[order], y_top[order])
    order = np.argsort(x_bot)
    omega = np.interp(ky, x_bot[order], y_bot[order])

    rows = ["ky,omega,gamma"]
    for k, w, g in zip(ky, omega, gamma):
        rows.append(f"{k:.6f},{w:.6f},{g:.6f}")
    out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def digitize_tem(fig_path: Path, out_path: Path) -> None:
    """Digitize the low-ky TEM branch (gamma and omega) from the TEM reference figure."""

    arr = np.array(Image.open(fig_path).convert("RGB"))
    hsv = rgb2hsv(arr / 255.0)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = ((h < 0.05) | (h > 0.95)) & (s > 0.3) & (v > 0.3)
    blue = (h > 0.55) & (h < 0.75) & (s > 0.3) & (v > 0.3)

    # Axis bounds for Fig. 4 in Frei et al. (TEM reference)
    x0, x1 = 277, 992
    y0, y1 = 104, 343

    curve_red = _curve_from_mask(red, x0, x1, y0, y1)
    curve_blue = _curve_from_mask(blue, x0, x1, y0, y1)

    x_min, x_max = 0.15, 0.8
    y_min, y_max = -2.0, 6.0

    x_red, y_red = _map_curve(curve_red, x0, x1, y0, y1, x_min, x_max, y_min, y_max)
    x_blue, y_blue = _map_curve(curve_blue, x0, x1, y0, y1, x_min, x_max, y_min, y_max)

    # Focus on the low-ky TEM branch.
    ky = np.array([0.2, 0.25, 0.3, 0.35, 0.4, 0.45], dtype=float)
    red_mask = x_red <= 0.5
    blue_mask = x_blue <= 0.5
    order = np.argsort(x_red[red_mask])
    gamma = np.interp(ky, x_red[red_mask][order], y_red[red_mask][order])
    order = np.argsort(x_blue[blue_mask])
    omega = np.interp(ky, x_blue[blue_mask][order], y_blue[blue_mask][order])

    rows = ["ky,omega,gamma"]
    for k, w, g in zip(ky, omega, gamma):
        rows.append(f"{k:.6f},{w:.6f},{g:.6f}")
    out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    etg_fig = Path("/tmp/gx_paper_fig2_page.png")
    tem_fig = Path("/tmp/tem_fig4_crop2.png")

    if not etg_fig.exists() or not tem_fig.exists():
        raise SystemExit(
            "Missing reference figures in /tmp. Render gx_paper_fig2_page.png and tem_fig4_crop2.png before running."
        )

    digitize_etg(etg_fig, DATA_DIR / "etg_reference.csv")
    digitize_tem(tem_fig, DATA_DIR / "tem_reference.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
