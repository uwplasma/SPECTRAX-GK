from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "build_tem_branch_parity_audit.py"
spec = importlib.util.spec_from_file_location("build_tem_branch_parity_audit", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def test_tem_branch_audit_tracks_sign_and_branch_mismatch(tmp_path: Path) -> None:
    table = tmp_path / "tem_mismatch.csv"
    table.write_text(
        "ky,gamma_ref,omega_ref,gamma_spectrax,omega_spectrax,rel_gamma,rel_omega\n"
        "0.2,1.0,2.0,1.2,-1.0,0.2,-1.5\n"
        "0.3,2.0,1.0,2.2,0.5,0.1,-0.5\n"
        "0.4,-1.0,-0.5,0.5,1.0,-1.5,-3.0\n",
        encoding="utf-8",
    )
    reference = tmp_path / "tem_reference.csv"
    reference.write_text("ky,omega,gamma\n0.2,2.0,1.0\n", encoding="utf-8")

    payload = mod.build_audit_payload(table=table, reference=reference)
    metrics = payload["metrics"]

    assert payload["status"] == "open"
    assert metrics["gamma_sign_mismatch_count"] == 1
    assert metrics["omega_sign_mismatch_count"] == 2
    assert metrics["omega_branch_inversion"] is True
    assert metrics["max_abs_rel_gamma"] == 1.5
    assert metrics["max_abs_rel_omega_ref_ge_0p2"] == 3.0


def test_tem_branch_audit_writes_artifacts(tmp_path: Path) -> None:
    table = tmp_path / "tem_mismatch.csv"
    table.write_text(
        "ky,gamma_ref,omega_ref,gamma_spectrax,omega_spectrax,rel_gamma,rel_omega\n"
        "0.2,1.0,2.0,1.1,1.8,0.1,-0.1\n"
        "0.3,1.5,1.0,1.4,0.8,-0.0666667,-0.2\n",
        encoding="utf-8",
    )

    payload = mod.build_audit_payload(table=table, reference=tmp_path / "missing.csv")
    paths = mod.write_artifacts(payload, out_png=tmp_path / "tem_audit.png")

    for path in paths.values():
        assert Path(path).exists()
    written = json.loads((tmp_path / "tem_audit.json").read_text(encoding="utf-8"))
    assert written["kind"] == "tem_branch_parity_audit"
    assert written["reference"]["available"] is False
