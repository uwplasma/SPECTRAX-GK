from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd


def _load_tool_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "plot_w7x_exact_state_audit.py"
    spec = importlib.util.spec_from_file_location("plot_w7x_exact_state_audit", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_audit_dir(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "startup.log").write_text(
        "\n".join(
            [
                "g_state      max|ref|=6.751e-04 max|test|=6.751e-04 max|diff|=6.000e-11 max|rel|=1.332e-07 rms_rel=7.303e-08 idx=(0,)",
                "phi          max|ref|=5.252e-04 max|test|=5.252e-04 max|diff|=1.567e-10 max|rel|=7.362e-07 rms_rel=1.218e-07 idx=(0,)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (path / "diag_state.log").write_text(
        "\n".join(
            [
                "kperp2       max|ref|=1.474e+01 max|test|=1.474e+01 max|diff|=3.338e-06 max|rel|=7.082e-07 rms_rel=9.822e-08 idx=(0,)",
                "fluxfac      max|ref|=1.950e-02 max|test|=1.950e-02 max|diff|=5.588e-09 max|rel|=3.055e-07 rms_rel=1.379e-07 idx=(0,)",
                "apar         max|ref|=0.000e+00 max|test|=0.000e+00 max|diff|=0.000e+00 max|rel|=nan rms_rel=nan idx=(0,)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "time_index": 10,
                "t": 32.4,
                "metric": "Wg",
                "gx_out": 11.0,
                "spectrax_dump": 11.0 * (1 + 1.0e-7),
                "rel_dump": 1.0e-7,
                "spectrax_solve": 11.0 * (1 + 2.0e-7),
                "rel_solve": 2.0e-7,
            },
            {
                "time_index": 10,
                "t": 32.4,
                "metric": "Wapar",
                "gx_out": 0.0,
                "spectrax_dump": 0.0,
                "rel_dump": 0.0,
                "spectrax_solve": 0.0,
                "rel_solve": 0.0,
            },
        ]
    ).to_csv(path / "diag_state.csv", index=False)


def test_w7x_exact_state_audit_parses_and_writes_outputs(tmp_path: Path) -> None:
    mod = _load_tool_module()
    audit_dir = tmp_path / "audit" / "w7x_vmec"
    _write_audit_dir(audit_dir)

    rows = mod.build_rows(audit_dir)
    assert {row["phase"] for row in rows} == {"startup", "late arrays", "late diagnostics"}
    assert max(float(row["value"]) for row in rows if row["value"] == row["value"]) < 1.0e-6

    out_png = tmp_path / "w7x_exact_state_audit.png"
    rc = mod.main(["--audit-dir", str(audit_dir), "--out-png", str(out_png)])

    assert rc == 0
    assert out_png.exists()
    assert out_png.with_suffix(".pdf").exists()
    payload = json.loads(out_png.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["validation_status"] == "closed"
    assert payload["gate_index_include"] is False
    assert payload["max_finite_relative_error"] < 1.0e-6
