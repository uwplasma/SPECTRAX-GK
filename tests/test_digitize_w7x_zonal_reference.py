from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw
import numpy as np
import pytest


def _load_tool_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "digitize_w7x_zonal_reference.py"
    spec = importlib.util.spec_from_file_location("digitize_w7x_zonal_reference", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _pixel_y(value: float, box: tuple[int, int, int, int], y_range: tuple[float, float]) -> int:
    _x0, _x1, y0, y1 = box
    return int(round(y0 + (y_range[1] - value) / (y_range[1] - y_range[0]) * (y1 - y0)))


def _synthetic_reference_image(mod) -> Image.Image:
    image = Image.new("RGB", (mod.EXPECTED_IMAGE_SHAPE[1], mod.EXPECTED_IMAGE_SHAPE[0]), "white")
    draw = ImageDraw.Draw(image)
    for panel in mod.PANEL_CALIBRATIONS:
        for code, color, value in (("stella", (255, 0, 0), 0.12), ("GENE", (0, 0, 255), 0.18)):
            y_main = _pixel_y(value, panel.main_box, panel.y_range)
            y_inset = _pixel_y(value, panel.inset_box, panel.inset_y_range)
            draw.line((panel.main_box[0], y_main, panel.main_box[1], y_main), fill=color, width=5)
            draw.line((panel.inset_box[0], y_inset, panel.inset_box[1], y_inset), fill=color, width=5)
    return image


def test_w7x_zonal_digitizer_axis_mapping_round_trip() -> None:
    mod = _load_tool_module()
    panel = mod.PANEL_CALIBRATIONS[0]
    x, y = mod._pixel_to_data(
        np.array([panel.main_box[0], panel.main_box[1]]),
        np.array([panel.main_box[2], panel.main_box[3]]),
        box=panel.main_box,
        x_range=panel.t_range,
        y_range=panel.y_range,
    )

    assert np.allclose(x, np.array(panel.t_range))
    assert y[0] == pytest.approx(panel.y_range[1])
    assert y[1] == pytest.approx(panel.y_range[0])


def test_w7x_zonal_digitizer_extracts_synthetic_residuals() -> None:
    mod = _load_tool_module()
    image = np.asarray(_synthetic_reference_image(mod), dtype=np.uint8)

    trace_df, residual_df = mod.digitize_reference(image, samples_per_trace=11)

    assert set(trace_df["code"]) == {"stella", "GENE"}
    assert len(trace_df) == len(mod.PANEL_CALIBRATIONS) * 2 * 11
    medians = residual_df.set_index("code")["residual_median"].to_dict()
    assert medians["stella"] == pytest.approx(0.12, abs=5.0e-3)
    assert medians["GENE"] == pytest.approx(0.18, abs=5.0e-3)


def test_w7x_zonal_digitizer_main_writes_artifacts(tmp_path: Path) -> None:
    mod = _load_tool_module()
    figure = tmp_path / "synthetic_zf.png"
    _synthetic_reference_image(mod).save(figure)
    out_csv = tmp_path / "trace.csv"
    out_residuals = tmp_path / "residuals.csv"
    out_json = tmp_path / "meta.json"
    out_png = tmp_path / "qa.png"

    rc = mod.main(
        [
            "--figure",
            str(figure),
            "--out-csv",
            str(out_csv),
            "--out-residual-csv",
            str(out_residuals),
            "--out-json",
            str(out_json),
            "--out-png",
            str(out_png),
            "--samples-per-trace",
            "17",
        ]
    )

    assert rc == 0
    assert out_csv.exists()
    assert out_residuals.exists()
    assert out_png.exists()
    assert out_png.with_suffix(".pdf").exists()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["validation_status"] == "reference"
    assert payload["figure"] == "Figure 11 / source figs/ZF.pdf"
