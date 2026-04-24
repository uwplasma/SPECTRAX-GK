from pathlib import Path
import json

from tools.profile_runtime_startup import PhaseTiming, _write_phase_csv, _write_phase_json


def test_profile_runtime_startup_writes_csv_and_json(tmp_path: Path) -> None:
    phases = [
        PhaseTiming(phase="a", seconds=1.25, note="first"),
        PhaseTiming(phase="b", seconds=2.75, note="second"),
    ]
    csv_path = tmp_path / "startup.csv"
    json_path = tmp_path / "startup.json"

    _write_phase_csv(csv_path, phases)
    _write_phase_json(json_path, phases, {"config": "case.toml", "device_count": 1})

    csv_text = csv_path.read_text(encoding="utf-8")
    assert "phase,seconds,note" in csv_text
    assert "a,1.25,first" in csv_text

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["config"] == "case.toml"
    assert payload["startup_total_s"] == 4.0
    assert payload["phases"][1]["phase"] == "b"
