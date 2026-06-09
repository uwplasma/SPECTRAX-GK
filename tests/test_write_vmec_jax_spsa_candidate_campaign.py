from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "write_vmec_jax_spsa_candidate_campaign.py"
spec = importlib.util.spec_from_file_location("write_vmec_jax_spsa_candidate_campaign", SCRIPT)
assert spec is not None
assert spec.loader is not None
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def _write_input(path: Path) -> None:
    path.write_text(
        """
&INDATA
  RBC(0,0) = 1.0000000000000000E+00
  ZBS(1,0) = -2.0000000000000000E-02
  ZBS(1,1) = 5.0000000000000000E-02
  RBC(1,1) = 1.0000000000000000E-01
/
""".lstrip(),
        encoding="utf-8",
    )


def test_spsa_candidate_campaign_writes_plus_minus_common_random_number_commands(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "input.final"
    out_dir = tmp_path / "spsa"
    _write_input(baseline)

    args = mod.parse_args(
        [
            "--baseline-input",
            str(baseline),
            "--out-dir",
            str(out_dir),
            "--controls",
            "ZBS(1,0);ZBS(1,1);RBC(1,1)",
            "--iterations",
            "2",
            "--seed",
            "123",
            "--relative-delta",
            "0.1",
            "--audit-seed-variant",
            "41",
            "--audit-seed-variant",
            "42",
        ]
    )

    payload = mod.build_campaign(args)

    assert payload["kind"] == "vmec_jax_spsa_transport_candidate_campaign"
    assert payload["controls"] == ["ZBS(1,0)", "ZBS(1,1)", "RBC(1,1)"]
    assert payload["common_random_number_policy"]["audit_seed_variants"] == [41, 42]
    assert len(payload["pairs"]) == 2
    first = payload["pairs"][0]
    assert set(first["states"]) == {"plus", "minus"}
    assert first["states"]["plus"]["input"].endswith("iter_000/plus/input.final")
    plus_text = (out_dir / "iter_000" / "plus" / "input.final").read_text(encoding="utf-8")
    minus_text = (out_dir / "iter_000" / "minus" / "input.final").read_text(encoding="utf-8")
    assert plus_text != minus_text
    assert "RBC(1,1)" in plus_text
    metric_command = first["states"]["plus"]["metric_eval_command"]
    metric_parts = shlex.split(metric_command)
    assert metric_parts[:2] == ["python3", "tools/evaluate_vmec_jax_spectrax_transport_metric.py"]
    assert metric_parts[metric_parts.index("--transport-kind") + 1] == "nonlinear_window_heat_flux"
    assert metric_parts[metric_parts.index("--mboz") + 1] == "21"
    audit_command = first["states"]["plus"]["nonlinear_audit_command"]
    assert "--seed-variant 41 --seed-variant 42" in audit_command
    assert "--window-tmin 1100 --window-tmax 1500" in audit_command
    assert "dJ/dx_i" in first["gradient_estimator"]["RBC(1,1)"]
    manifest = json.loads((out_dir / "vmec_jax_spsa_candidate_manifest.json").read_text(encoding="utf-8"))
    assert manifest["claim_scope"].startswith("SPSA common-random-number candidate generation")


def test_spsa_candidate_campaign_cli_writes_manifest(tmp_path: Path) -> None:
    baseline = tmp_path / "input.final"
    out_dir = tmp_path / "spsa"
    _write_input(baseline)

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--baseline-input",
            str(baseline),
            "--out-dir",
            str(out_dir),
            "--controls",
            "ZBS(1,0);RBC(1,1)",
            "--iterations",
            "1",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

    status = json.loads(completed.stdout)
    assert status["pairs"] == 1
    assert status["controls"] == ["ZBS(1,0)", "RBC(1,1)"]
    assert (out_dir / "vmec_jax_spsa_candidate_manifest.json").exists()
