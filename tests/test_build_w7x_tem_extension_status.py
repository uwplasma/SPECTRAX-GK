from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "build_w7x_tem_extension_status.py"
spec = importlib.util.spec_from_file_location("build_w7x_tem_extension_status", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def test_w7x_tem_extension_status_tracks_open_tem_and_multiflux(tmp_path: Path) -> None:
    spectrum = tmp_path / "w7x_spectrum.json"
    spectrum.write_text(
        json.dumps(
            {
                "source_gate_passed": True,
                "time_samples": 12,
                "dominant_phi_ky": 0.19,
                "dominant_heat_flux_ky": 1.28,
            }
        ),
        encoding="utf-8",
    )
    tem = tmp_path / "tem.csv"
    tem.write_text(
        "ky,gamma_ref,omega_ref,gamma_spectrax,omega_spectrax,rel_gamma,rel_omega\n"
        "0.2,2.0,1.0,3.0,0.5,0.5,-0.5\n"
        "0.3,2.0,1.0,2.1,1.1,0.05,0.1\n",
        encoding="utf-8",
    )

    payload = mod.build_status_payload(w7x_spectrum=spectrum, tem_table=tem)
    rows = {row["lane"]: row for row in payload["rows"]}

    assert payload["summary"] == {"n_rows": 4, "n_closed": 1, "n_partial": 0, "n_open": 3}
    assert rows["W7-X nonlinear fluctuation spectrum"]["status"] == "closed"
    assert rows["TEM / kinetic-electron linear parity"]["status"] == "open"
    assert rows["TEM / kinetic-electron linear parity"]["key_metrics"]["max_abs_rel_gamma"] == 0.5
    assert rows["W7-X multi-flux-tube and multi-surface scan"]["status"] == "open"
    assert rows["W7-X kinetic-electron/TEM nonlinear window"]["status"] == "open"


def test_w7x_tem_extension_status_writes_artifacts(tmp_path: Path) -> None:
    payload = {
        "kind": "w7x_tem_extension_status",
        "rows": [
            {
                "lane": "W7-X nonlinear fluctuation spectrum",
                "status": "closed",
                "claim_level": "validated",
                "primary_artifact": "w7x.json",
                "key_metrics": {"time_samples": 4, "dominant_phi_ky": 0.2},
                "next_action": "Keep scoped.",
            },
            {
                "lane": "TEM / kinetic-electron linear parity",
                "status": "open",
                "claim_level": "open",
                "primary_artifact": "tem.csv",
                "key_metrics": {"max_abs_rel_gamma": 0.75},
                "next_action": "Fix mismatch.",
            },
        ],
        "summary": {"n_rows": 2, "n_closed": 1, "n_partial": 0, "n_open": 1},
    }

    paths = mod.write_artifacts(payload, out_png=tmp_path / "status.png")

    for path in paths.values():
        assert Path(path).exists()
    assert json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))["summary"]["n_open"] == 1
