from __future__ import annotations

import json
from pathlib import Path

from support.paths import REPO_ROOT, load_release_tool


ROOT = REPO_ROOT
mod = load_release_tool("check_matched_nonlinear_transport_matrix_progress")


def _write_manifest(
    tmp_path: Path, outputs: list[Path], *, include_dt: bool = False
) -> Path:
    config = {"window": {"tmin": 10.0, "tmax": 20.0}}
    if include_dt:
        config.update({"dt": 0.05, "dt_variants": [0.04]})
    manifest = {
        "kind": "matched_nonlinear_transport_matrix_campaign",
        "config": config,
        "samples": [
            {
                "sample_id": "s0p45_a0_ky0p1",
                "surface_torflux": 0.45,
                "alpha": 0.0,
                "ky": 0.1,
                "states": {
                    "baseline": {"label": "base", "final_outputs": [str(outputs[0])]},
                    "candidate": {"label": "cand", "final_outputs": [str(outputs[1])]},
                },
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _touch_bundle(output: Path) -> None:
    stem = (
        output.name[: -len(".out.nc")]
        if output.name.endswith(".out.nc")
        else output.stem
    )
    base = output.with_name(stem)
    for suffix in ("out.nc", "restart.nc", "big.nc"):
        Path(f"{base}.{suffix}").write_text("stub\n", encoding="utf-8")


def test_progress_requires_target_time_even_when_bundle_exists(
    tmp_path: Path, monkeypatch
) -> None:
    base = tmp_path / "base.out.nc"
    cand = tmp_path / "cand.out.nc"
    _touch_bundle(base)
    _touch_bundle(cand)
    manifest = _write_manifest(tmp_path, [base, cand])
    monkeypatch.setattr(mod, "_read_output_tmax", lambda _path: 19.0)

    report = mod.build_report(matrix_manifest=manifest)

    assert report["summary"]["expected_outputs"] == 2
    assert report["summary"]["complete_bundles"] == 2
    assert report["summary"]["target_time_confirmed"] == 0
    assert report["summary"]["ready_for_postprocess"] is False


def test_progress_passes_when_all_bundles_reach_target_time(
    tmp_path: Path, monkeypatch
) -> None:
    base = tmp_path / "base.out.nc"
    cand = tmp_path / "cand.out.nc"
    _touch_bundle(base)
    _touch_bundle(cand)
    manifest = _write_manifest(tmp_path, [base, cand])
    monkeypatch.setattr(mod, "_read_output_tmax", lambda _path: 20.0)

    report = mod.build_report(matrix_manifest=manifest)

    assert report["summary"]["complete_bundles"] == 2
    assert report["summary"]["target_time_confirmed"] == 2
    assert report["summary"]["ready_for_postprocess"] is True
    assert all(row["bundle_complete"] for row in report["rows"])
    assert all(row["target_time_confirmed"] for row in report["rows"])


def test_progress_accepts_fixed_step_output_within_manifest_dt_tolerance(
    tmp_path: Path, monkeypatch
) -> None:
    base = tmp_path / "base.out.nc"
    cand = tmp_path / "cand.out.nc"
    _touch_bundle(base)
    _touch_bundle(cand)
    manifest = _write_manifest(tmp_path, [base, cand], include_dt=True)
    monkeypatch.setattr(mod, "_read_output_tmax", lambda _path: 19.927)

    report = mod.build_report(matrix_manifest=manifest)

    assert report["time_tolerance"] == 0.1
    assert report["summary"]["target_time_confirmed"] == 2
    assert report["summary"]["ready_for_postprocess"] is True


def test_progress_keeps_checkpoint_below_dt_tolerance_incomplete(
    tmp_path: Path, monkeypatch
) -> None:
    base = tmp_path / "base.out.nc"
    cand = tmp_path / "cand.out.nc"
    _touch_bundle(base)
    _touch_bundle(cand)
    manifest = _write_manifest(tmp_path, [base, cand], include_dt=True)
    monkeypatch.setattr(mod, "_read_output_tmax", lambda _path: 19.85)

    report = mod.build_report(matrix_manifest=manifest)

    assert report["time_tolerance"] == 0.1
    assert report["summary"]["target_time_confirmed"] == 0
    assert report["summary"]["ready_for_postprocess"] is False


def test_progress_cli_uses_manifest_dt_tolerance_by_default(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    base = tmp_path / "base.out.nc"
    cand = tmp_path / "cand.out.nc"
    _touch_bundle(base)
    _touch_bundle(cand)
    manifest = _write_manifest(tmp_path, [base, cand], include_dt=True)
    out_json = tmp_path / "progress.json"
    monkeypatch.setattr(mod, "_read_output_tmax", lambda _path: 19.927)

    rc = mod.main(["--matrix-manifest", str(manifest), "--out-json", str(out_json)])
    stdout = capsys.readouterr().out
    report = json.loads(out_json.read_text(encoding="utf-8"))

    assert rc == 0
    assert '"ready_for_postprocess": true' in stdout.lower()
    assert report["time_tolerance"] == 0.1
    assert report["summary"]["target_time_confirmed"] == 2


def test_skip_time_check_does_not_read_output_time(tmp_path: Path, monkeypatch) -> None:
    base = tmp_path / "base.out.nc"
    cand = tmp_path / "cand.out.nc"
    _touch_bundle(base)
    _touch_bundle(cand)
    manifest = _write_manifest(tmp_path, [base, cand], include_dt=True)

    def fail_if_called(_path):
        raise AssertionError("skip_time_check should not read NetCDF times")

    monkeypatch.setattr(mod, "_read_output_tmax", fail_if_called)
    report = mod.build_report(matrix_manifest=manifest, skip_time_check=True)

    assert report["skip_time_check"] is True
    assert report["summary"]["complete_bundles"] == 2
    assert report["summary"]["target_time_confirmed"] == 0
    assert report["summary"]["ready_for_postprocess"] is False
    assert report["summary"]["time_check_skipped"] is True
    assert all(not row["target_time_confirmed"] for row in report["rows"])
    assert all(row["output_tmax"] is None for row in report["rows"])
