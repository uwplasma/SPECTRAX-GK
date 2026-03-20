#!/usr/bin/env python3
"""Run manifest-driven CPU/GPU short-window parity gates."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
import json
import os
from pathlib import Path
from typing import Any

import jax
import numpy as np

from spectraxgk.io import load_runtime_from_toml, load_toml
from spectraxgk.runtime import RuntimeNonlinearResult, run_runtime_nonlinear


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True, help="TOML manifest describing each device-parity lane.")
    p.add_argument("--lane", type=str, default=None, help="Optional single lane key to run.")
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("tools_out") / "device_parity_gate",
        help="Output directory root.",
    )
    return p


def _lane_section(manifest: dict[str, Any]) -> dict[str, Any]:
    lanes = manifest.get("lane")
    if not isinstance(lanes, dict):
        raise ValueError("Manifest must contain a [lane.<name>] table per lane.")
    return lanes


def _resolve_manifest_path(value: str | Path, *, manifest_dir: Path) -> Path:
    raw = os.path.expandvars(str(value))
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = manifest_dir / path
    return path.resolve()


def _expand_env_value(value: object) -> str:
    text = str(value)
    for _ in range(4):
        expanded = os.path.expanduser(os.path.expandvars(text))
        if expanded == text:
            return expanded
        text = expanded
    return text


@contextmanager
def _temporary_env(env_updates: dict[str, str] | None):
    if not env_updates:
        yield
        return
    previous: dict[str, str | None] = {key: os.environ.get(key) for key in env_updates}
    try:
        for key, value in env_updates.items():
            os.environ[str(key)] = str(value)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _raw_run_value(raw: dict[str, Any], key: str, *, default: Any = None) -> Any:
    run_cfg = raw.get("run")
    if not isinstance(run_cfg, dict):
        return default
    return run_cfg.get(key, default)


def _device_result_summary(result: RuntimeNonlinearResult) -> dict[str, float]:
    if result.state is None:
        raise RuntimeError("Device parity gate requires return_state=True results.")
    summary = {
        "state_norm": float(np.linalg.norm(np.asarray(result.state).ravel())),
    }
    diag = result.diagnostics
    if diag is not None:
        for name, arr in (
            ("Wg", diag.Wg_t),
            ("Wphi", diag.Wphi_t),
            ("Wapar", diag.Wapar_t),
            ("heat", diag.heat_flux_t),
            ("pflux", diag.particle_flux_t),
        ):
            arr_np = np.asarray(arr, dtype=float)
            if arr_np.size:
                summary[name] = float(arr_np[-1])
    return summary


def _scalar_rel(ref: float, test: float) -> dict[str, float]:
    diff = abs(test - ref)
    denom = max(abs(ref), 1.0e-30)
    return {"abs": float(diff), "rel": float(diff / denom)}


def _run_lane(
    cfg_path: Path,
    lane_cfg: dict[str, Any],
) -> dict[str, Any]:
    env_cfg = lane_cfg.get("env")
    env_updates = (
        {str(key): _expand_env_value(value) for key, value in env_cfg.items()}
        if isinstance(env_cfg, dict)
        else None
    )
    with _temporary_env(env_updates):
        cfg, raw = load_runtime_from_toml(cfg_path)
        ky = float(lane_cfg.get("ky", _raw_run_value(raw, "ky", default=0.3)))
        kx_target = lane_cfg.get("kx_target", _raw_run_value(raw, "kx", default=0.0))
        if kx_target is not None:
            kx_target = float(kx_target)
        Nl = int(lane_cfg.get("Nl", _raw_run_value(raw, "Nl", default=4)))
        Nm = int(lane_cfg.get("Nm", _raw_run_value(raw, "Nm", default=8)))
        dt = float(lane_cfg.get("dt", cfg.time.dt))
        steps = int(lane_cfg["steps"])
        sample_stride = int(lane_cfg.get("sample_stride", 1))
        diagnostics_stride = int(lane_cfg.get("diagnostics_stride", 1))
        fixed_dt = bool(lane_cfg.get("fixed_dt", cfg.time.fixed_dt))
        rtol = float(lane_cfg.get("rtol", 2.0e-4))
        atol = float(lane_cfg.get("atol", 1.0e-7))
        required_nonzero = tuple(str(v) for v in lane_cfg.get("required_nonzero", ("state_norm", "Wphi")))

        cpu_devices = jax.devices("cpu")
        gpu_devices = jax.devices("gpu")
        if not cpu_devices or not gpu_devices:
            raise RuntimeError("CPU and GPU backends are both required for the device parity gate.")
        cpu = cpu_devices[0]
        gpu = gpu_devices[0]

        time_cfg = replace(
            cfg.time,
            dt=dt,
            fixed_dt=fixed_dt,
            diagnostics=True,
            sample_stride=sample_stride,
            diagnostics_stride=diagnostics_stride,
            t_max=dt * float(steps),
        )
        cfg_gate = replace(cfg, time=time_cfg)

        common = dict(
            ky_target=ky,
            kx_target=kx_target,
            Nl=Nl,
            Nm=Nm,
            dt=dt,
            steps=steps,
            sample_stride=sample_stride,
            diagnostics_stride=diagnostics_stride,
            return_state=True,
        )

        def _run_on(device) -> dict[str, float]:
            with jax.default_device(device):
                result = run_runtime_nonlinear(cfg_gate, **common)
            return _device_result_summary(result)

        cpu_summary = _run_on(cpu)
        gpu_summary = _run_on(gpu)
        metric_names = sorted(cpu_summary.keys() & gpu_summary.keys())
        metrics = {name: _scalar_rel(cpu_summary[name], gpu_summary[name]) for name in metric_names}
        missing_nonzero = [name for name in required_nonzero if abs(cpu_summary.get(name, 0.0)) <= 1.0e-30]
        ok = not missing_nonzero and all(
            values["abs"] <= atol or values["rel"] <= rtol for values in metrics.values()
        )
        return {
            "config": str(cfg_path),
            "env": env_updates,
            "ky": ky,
            "kx_target": kx_target,
            "Nl": Nl,
            "Nm": Nm,
            "dt": dt,
            "steps": steps,
            "fixed_dt": fixed_dt,
            "required_nonzero": list(required_nonzero),
            "missing_nonzero": missing_nonzero,
            "cpu": cpu_summary,
            "gpu": gpu_summary,
            "metrics": metrics,
            "tolerances": {"rtol": rtol, "atol": atol},
            "ok": bool(ok),
        }


def main() -> None:
    args = build_parser().parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    manifest_dir = manifest_path.parent
    manifest = load_toml(manifest_path)
    lanes = _lane_section(manifest)
    selected = [args.lane] if args.lane is not None else list(lanes.keys())
    out_root = args.outdir.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {"manifest": str(manifest_path), "lanes": {}}
    failed: list[str] = []
    for lane_key in selected:
        lane_cfg = lanes.get(lane_key)
        if not isinstance(lane_cfg, dict):
            raise SystemExit(f"Lane config must be a table: lane.{lane_key}")
        config_path = _resolve_manifest_path(lane_cfg["config"], manifest_dir=manifest_dir)
        lane_summary = _run_lane(config_path, lane_cfg)
        summary["lanes"][lane_key] = lane_summary
        lane_out = out_root / lane_key
        lane_out.mkdir(parents=True, exist_ok=True)
        (lane_out / "summary.json").write_text(json.dumps(lane_summary, indent=2, sort_keys=True), encoding="utf-8")
        if not bool(lane_summary["ok"]):
            failed.append(lane_key)

    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"saved {summary_path}")
    if failed:
        raise SystemExit(f"Device parity gate failed for lanes: {', '.join(failed)}")


if __name__ == "__main__":
    main()
