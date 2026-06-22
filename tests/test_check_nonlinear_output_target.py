from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "check_nonlinear_output_target.py"
spec = importlib.util.spec_from_file_location("check_nonlinear_output_target", SCRIPT)
assert spec is not None
assert spec.loader is not None
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def _touch_bundle(output: Path) -> None:
    stem = output.name[: -len(".out.nc")] if output.name.endswith(".out.nc") else output.stem
    base = output.with_name(stem)
    for suffix in ("out.nc", "restart.nc", "big.nc"):
        Path(f"{base}.{suffix}").write_text("stub\n", encoding="utf-8")


def test_output_target_checker_accepts_near_horizon_bundle(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "run.out.nc"
    _touch_bundle(output)
    monkeypatch.setattr(mod, "_read_output_tmax", lambda _path: 1499.927)

    report = mod.build_report(output=output, target_time=1500.0, time_tolerance=0.1)

    assert report["bundle_complete"] is True
    assert report["target_time_confirmed"] is True


def test_output_target_checker_rejects_partial_checkpoint_bundle(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "run.out.nc"
    _touch_bundle(output)
    monkeypatch.setattr(mod, "_read_output_tmax", lambda _path: 400.0)

    report = mod.build_report(output=output, target_time=1500.0, time_tolerance=0.1)

    assert report["bundle_complete"] is True
    assert report["target_time_confirmed"] is False


def test_output_target_checker_cli_status_codes(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "run.out.nc"
    _touch_bundle(output)
    monkeypatch.setattr(mod, "_read_output_tmax", lambda _path: 19.95)

    assert (
        mod.main(
            [
                "--output",
                str(output),
                "--target-time",
                "20",
                "--time-tolerance",
                "0.1",
                "--quiet",
            ]
        )
        == 0
    )
    assert (
        mod.main(
            [
                "--output",
                str(output),
                "--target-time",
                "20",
                "--time-tolerance",
                "0.01",
                "--quiet",
            ]
        )
        == 1
    )
