from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from support.paths import load_artifact_tool


mod = load_artifact_tool("build_qa_itg_optimization_readme_panel")


def test_readme_panel_builder_treats_missing_optional_transport_artifacts_as_pending(
    tmp_path: Path,
) -> None:
    geometry_png = tmp_path / "geometry.png"
    fig, ax = plt.subplots(figsize=(1.0, 1.0))
    ax.imshow([[0.0, 1.0], [1.0, 0.0]])
    ax.set_axis_off()
    fig.savefig(geometry_png)
    plt.close(fig)

    sweep_json = tmp_path / "sweep.json"
    sweep_json.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "qa_baseline_scipy",
                        "iota_profile": {"s_iotaf": [0.2, 0.7], "iotaf": [0.41, 0.42]},
                        "q_traces": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    landscape_json = tmp_path / "landscape.json"
    landscape_json.write_text(
        json.dumps(
            {
                "coefficient": "RBC(1,1)",
                "rows": [
                    {
                        "label": "baseline",
                        "reduced_metric_reports": {
                            "growth": {
                                "payload": {
                                    "sample_statistics": {
                                        "weighted_standard_error": 0.01
                                    }
                                }
                            }
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    landscape_csv = tmp_path / "landscape.csv"
    landscape_csv.write_text(
        "label,relative_fraction,coefficient_value,growth,quasilinear_flux\n"
        "baseline,0.0,0.0,1.0,2.0\n",
        encoding="utf-8",
    )
    out = tmp_path / "panel.png"

    sidecar = mod.build_panel(
        geometry_png=geometry_png,
        sweep_json=sweep_json,
        landscape_json=landscape_json,
        landscape_csv=landscape_csv,
        admission_json=tmp_path / "missing_admission.json",
        matched_json=tmp_path / "missing_matched.json",
        out=out,
    )

    assert out.exists()
    assert sidecar["sources"]["landscape_admission_present"] is False
    assert sidecar["sources"]["matched_nonlinear_present"] is False
    assert sidecar["selected_landscape_candidate"] is None
    assert sidecar["matched_projected_candidate"] is None
