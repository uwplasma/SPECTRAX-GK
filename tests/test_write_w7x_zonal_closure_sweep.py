from __future__ import annotations

import json
from pathlib import Path

from tools.write_w7x_zonal_closure_sweep import (
    DEFAULT_CASES,
    SweepCase,
    build_manifest,
    write_manifest,
)


def test_build_w7x_zonal_closure_sweep_manifest_contains_operator_families(tmp_path: Path) -> None:
    payload = build_manifest(
        config=tmp_path / "runtime_w7x.toml",
        out_dir=tmp_path / "runs",
        kx=0.07,
        dt=0.05,
        steps=2000,
        nl=16,
        nm=64,
        sample_stride=4,
    )

    assert payload["kind"] == "w7x_zonal_closure_sweep_manifest"
    assert payload["kx"] == 0.07
    assert payload["Nl"] == 16
    assert payload["Nm"] == 64
    assert len(payload["cases"]) == len(DEFAULT_CASES)
    assert len(payload["launch_commands"]) == len(DEFAULT_CASES)
    assert any(case["family"] == "constant_mixed_lm" for case in payload["cases"])
    assert any(case["family"] == "constant_laguerre" for case in payload["cases"])
    assert any(case["family"] == "constant_isotropic" for case in payload["cases"])
    assert any("--nu-hyper-lm 0.01" in command for command in payload["launch_commands"])
    assert any("--nu-hyper-l 0.03" in command for command in payload["launch_commands"])
    assert any("--nu-hyper 0.01" in command for command in payload["launch_commands"])
    assert any("--hypercollisions-kz 1" in command or "--hypercollisions-kz 1.0" in command for command in payload["launch_commands"])
    assert "plot_w7x_zonal_closure_ladder.py" in payload["plot_command"]


def test_w7x_zonal_closure_sweep_manifest_writes_json(tmp_path: Path) -> None:
    payload = build_manifest(
        config=tmp_path / "runtime_w7x.toml",
        out_dir=tmp_path / "runs",
        cases=(
            SweepCase(
                slug="baseline",
                label="baseline",
                family="baseline",
                hypercollisions_const=0.0,
                hypercollisions_kz=0.0,
            ),
        ),
    )
    path = write_manifest(tmp_path / "manifest.json", payload)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["kind"] == "w7x_zonal_closure_sweep_manifest"
    assert loaded["cases"][0]["slug"] == "baseline"
    assert loaded["launch_commands"][0].startswith("python3 tools/generate_w7x_zonal_response_panel.py")
