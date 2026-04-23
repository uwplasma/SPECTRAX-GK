from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_tool_module():
    path = Path("/Users/rogeriojorge/local/SPECTRAX-GK/tools/make_validation_gate_index.py")
    spec = importlib.util.spec_from_file_location("make_validation_gate_index", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_gate(path: Path, *, case: str, passed: bool) -> None:
    path.write_text(
        json.dumps(
            {
                "gate_report": {
                    "case": case,
                    "source": "synthetic",
                    "passed": passed,
                    "max_abs_error": 0.1,
                    "max_rel_error": 0.2,
                    "gates": [
                        {
                            "metric": "metric_a",
                            "passed": passed,
                        }
                    ],
                }
            }
        )
    )


def test_collect_gate_entries_reads_top_level_gate_report(tmp_path: Path) -> None:
    mod = _load_tool_module()
    _write_gate(tmp_path / "pass.json", case="passed_case", passed=True)
    _write_gate(tmp_path / "open.json", case="open_case", passed=False)
    (tmp_path / "ignored.json").write_text(json.dumps({"case": "no_gate"}))

    index = mod.build_index([str(tmp_path / "*.json")])

    assert index["n_reports"] == 2
    assert index["n_passed"] == 1
    assert index["n_open"] == 1
    rows = {row["case"]: row for row in index["reports"]}
    assert rows["open_case"]["failed_metrics"] == "metric_a"
    assert rows["passed_case"]["n_failed"] == 0


def test_make_validation_gate_index_main_writes_json_csv_and_plot(tmp_path: Path) -> None:
    mod = _load_tool_module()
    _write_gate(tmp_path / "gate.json", case="case_a", passed=True)
    out_json = tmp_path / "index.json"
    out_csv = tmp_path / "index.csv"
    out_png = tmp_path / "index.png"

    assert (
        mod.main(
            [
                "--glob",
                str(tmp_path / "*.json"),
                "--out-json",
                str(out_json),
                "--out-csv",
                str(out_csv),
                "--out-png",
                str(out_png),
            ]
        )
        == 0
    )

    payload = json.loads(out_json.read_text())
    assert payload["n_reports"] == 1
    assert out_csv.exists()
    assert out_png.exists()
    assert out_png.with_suffix(".pdf").exists()
