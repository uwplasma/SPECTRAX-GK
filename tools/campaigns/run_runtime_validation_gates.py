#!/usr/bin/env python3
"""Run manifest-driven runtime validation gates.

The subcommands here cover short CPU/GPU device parity, nonlinear restart
continuation parity, and VMEC-to-geometry roundtrip determinism. They share the
same lane-manifest contract so developers have one place to run runtime gates.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from typing import Any

import jax
import numpy as np

from spectraxgk.artifacts.restart import write_netcdf_restart_state
from spectraxgk.geometry import load_imported_geometry_netcdf
from spectraxgk.geometry.vmec_eik import generate_runtime_vmec_eik
from spectraxgk.runtime import RuntimeNonlinearResult, run_runtime_nonlinear
from spectraxgk.workflows.runtime.toml import load_runtime_from_toml, load_toml

ROUNDTRIP_FIELDS = (
    "theta",
    "bmag_profile",
    "gds2_profile",
    "gds21_profile",
    "gds22_profile",
    "cv_profile",
    "gb_profile",
    "jacobian_profile",
    "grho_profile",
)


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


def _lane_env(lane_cfg: dict[str, Any]) -> dict[str, str] | None:
    env_cfg = lane_cfg.get("env")
    return (
        {str(key): _expand_env_value(value) for key, value in env_cfg.items()}
        if isinstance(env_cfg, dict)
        else None
    )


def _raw_run_value(raw: dict[str, Any], key: str, *, default: Any = None) -> Any:
    run_cfg = raw.get("run")
    if not isinstance(run_cfg, dict):
        return default
    return run_cfg.get(key, default)


def _scalar_rel(ref: float, test: float) -> dict[str, float]:
    diff = abs(test - ref)
    denom = max(abs(ref), 1.0e-30)
    return {"abs": float(diff), "rel": float(diff / denom)}


def _complex_rel_metrics(ref: np.ndarray, test: np.ndarray) -> dict[str, float]:
    ref_arr = np.asarray(ref, dtype=np.complex128)
    test_arr = np.asarray(test, dtype=np.complex128)
    diff = np.abs(test_arr - ref_arr)
    denom = np.maximum(np.abs(ref_arr), 1.0e-30)
    rel = diff / denom
    return {
        "max_abs": float(np.max(diff)),
        "max_rel": float(np.max(rel)),
        "rms_rel": float(np.sqrt(np.mean(rel**2))),
        "norm_ref": float(np.linalg.norm(ref_arr.ravel())),
        "norm_test": float(np.linalg.norm(test_arr.ravel())),
    }


def _diag_scalar_map(result: RuntimeNonlinearResult) -> dict[str, float]:
    if result.diagnostics is None:
        return {}
    diag = result.diagnostics
    out: dict[str, float] = {}
    for name, arr in (
        ("Wg", diag.Wg_t),
        ("Wphi", diag.Wphi_t),
        ("Wapar", diag.Wapar_t),
        ("heat", diag.heat_flux_t),
        ("pflux", diag.particle_flux_t),
    ):
        arr_np = np.asarray(arr, dtype=float)
        if arr_np.size:
            out[name] = float(arr_np[-1])
    return out


def _device_result_summary(result: RuntimeNonlinearResult) -> dict[str, float]:
    if result.state is None:
        raise RuntimeError("Device parity gate requires return_state=True results.")
    summary = {
        "state_norm": float(np.linalg.norm(np.asarray(result.state).ravel())),
    }
    summary.update(_diag_scalar_map(result))
    return summary


def build_device_parity_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run CPU/GPU short-window parity gates."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lane", type=str, default=None)
    parser.add_argument(
        "--outdir", type=Path, default=Path("tools_out") / "device_parity_gate"
    )
    return parser


def _run_device_lane(cfg_path: Path, lane_cfg: dict[str, Any]) -> dict[str, Any]:
    env_updates = _lane_env(lane_cfg)
    with _temporary_env(env_updates):
        cfg, raw = load_runtime_from_toml(cfg_path)
        ky = float(lane_cfg.get("ky", _raw_run_value(raw, "ky", default=0.3)))
        kx_target = lane_cfg.get("kx_target", _raw_run_value(raw, "kx", default=0.0))
        if kx_target is not None:
            kx_target = float(kx_target)
        nl = int(lane_cfg.get("Nl", _raw_run_value(raw, "Nl", default=4)))
        nm = int(lane_cfg.get("Nm", _raw_run_value(raw, "Nm", default=8)))
        dt = float(lane_cfg.get("dt", cfg.time.dt))
        steps = int(lane_cfg["steps"])
        sample_stride = int(lane_cfg.get("sample_stride", 1))
        diagnostics_stride = int(lane_cfg.get("diagnostics_stride", 1))
        fixed_dt = bool(lane_cfg.get("fixed_dt", cfg.time.fixed_dt))
        rtol = float(lane_cfg.get("rtol", 2.0e-4))
        atol = float(lane_cfg.get("atol", 1.0e-7))
        required_nonzero = tuple(
            str(v) for v in lane_cfg.get("required_nonzero", ("state_norm", "Wphi"))
        )

        cpu_devices = jax.devices("cpu")
        gpu_devices = jax.devices("gpu")
        if not cpu_devices or not gpu_devices:
            raise RuntimeError(
                "CPU and GPU backends are both required for the device parity gate."
            )

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
            Nl=nl,
            Nm=nm,
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

        cpu_summary = _run_on(cpu_devices[0])
        gpu_summary = _run_on(gpu_devices[0])
        metric_names = sorted(cpu_summary.keys() & gpu_summary.keys())
        metrics = {
            name: _scalar_rel(cpu_summary[name], gpu_summary[name])
            for name in metric_names
        }
        missing_nonzero = [
            name
            for name in required_nonzero
            if abs(cpu_summary.get(name, 0.0)) <= 1.0e-30
        ]
        ok = not missing_nonzero and all(
            values["abs"] <= atol or values["rel"] <= rtol
            for values in metrics.values()
        )
        return {
            "config": str(cfg_path),
            "env": env_updates,
            "ky": ky,
            "kx_target": kx_target,
            "Nl": nl,
            "Nm": nm,
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


def main_device_parity(argv: list[str] | None = None) -> int:
    args = build_device_parity_parser().parse_args(argv)
    manifest_path = args.manifest.expanduser().resolve()
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
        config_path = _resolve_manifest_path(
            lane_cfg["config"], manifest_dir=manifest_path.parent
        )
        lane_summary = _run_device_lane(config_path, lane_cfg)
        summary["lanes"][lane_key] = lane_summary
        lane_out = out_root / lane_key
        lane_out.mkdir(parents=True, exist_ok=True)
        (lane_out / "summary.json").write_text(
            json.dumps(lane_summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        if not bool(lane_summary["ok"]):
            failed.append(lane_key)

    summary_path = out_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"saved {summary_path}")
    if failed:
        raise SystemExit(f"Device parity gate failed for lanes: {', '.join(failed)}")
    return 0


def build_vmec_roundtrip_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run VMEC roundtrip determinism gates."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lane", type=str, default=None)
    parser.add_argument(
        "--outdir", type=Path, default=Path("tools_out") / "vmec_roundtrip_gate"
    )
    return parser


def _array_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    diff = np.asarray(a) - np.asarray(b)
    return {
        "max_abs": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "l2_abs": float(np.linalg.norm(diff.ravel())) if diff.size else 0.0,
    }


def _run_vmec_roundtrip_lane(
    cfg_path: Path, lane_cfg: dict[str, Any], *, out_dir: Path
) -> dict[str, Any]:
    env_updates = _lane_env(lane_cfg)
    with _temporary_env(env_updates):
        cfg, _raw = load_runtime_from_toml(cfg_path)
        out1 = out_dir / "geom1.eik.nc"
        out2 = out_dir / "geom2.eik.nc"
        generate_runtime_vmec_eik(cfg, output_path=out1, force=True)
        generate_runtime_vmec_eik(cfg, output_path=out2, force=True)
        g1 = load_imported_geometry_netcdf(out1)
        g2 = load_imported_geometry_netcdf(out2)
        field_metrics = {
            name: _array_metrics(
                np.asarray(getattr(g1, name)), np.asarray(getattr(g2, name))
            )
            for name in ROUNDTRIP_FIELDS
        }
        ok = all(values["max_abs"] == 0.0 for values in field_metrics.values())
        return {
            "config": str(cfg_path),
            "env": env_updates,
            "out1": str(out1),
            "out2": str(out2),
            "fields": field_metrics,
            "ok": bool(ok),
        }


def main_vmec_roundtrip(argv: list[str] | None = None) -> int:
    args = build_vmec_roundtrip_parser().parse_args(argv)
    manifest_path = args.manifest.expanduser().resolve()
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
        config_path = _resolve_manifest_path(
            lane_cfg["config"], manifest_dir=manifest_path.parent
        )
        lane_out = out_root / lane_key
        lane_out.mkdir(parents=True, exist_ok=True)
        lane_summary = _run_vmec_roundtrip_lane(config_path, lane_cfg, out_dir=lane_out)
        summary["lanes"][lane_key] = lane_summary
        (lane_out / "summary.json").write_text(
            json.dumps(lane_summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        if not bool(lane_summary["ok"]):
            failed.append(lane_key)

    summary_path = out_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"saved {summary_path}")
    if failed:
        raise SystemExit(f"VMEC roundtrip gate failed for lanes: {', '.join(failed)}")
    return 0


def build_restart_parity_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run nonlinear restart/continuation parity gates."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lane", type=str, default=None)
    parser.add_argument(
        "--outdir", type=Path, default=Path("tools_out") / "restart_parity_gate"
    )
    return parser


def _restart_lane_summary(
    cfg_path: Path,
    lane_cfg: dict[str, Any],
    *,
    out_dir: Path,
) -> dict[str, Any]:
    env_updates = _lane_env(lane_cfg)
    with _temporary_env(env_updates):
        cfg, raw = load_runtime_from_toml(cfg_path)
        ky = float(lane_cfg.get("ky", _raw_run_value(raw, "ky", default=0.3)))
        kx_target = lane_cfg.get("kx_target", _raw_run_value(raw, "kx", default=0.0))
        if kx_target is not None:
            kx_target = float(kx_target)
        nl = int(lane_cfg.get("Nl", _raw_run_value(raw, "Nl", default=4)))
        nm = int(lane_cfg.get("Nm", _raw_run_value(raw, "Nm", default=8)))
        dt = float(lane_cfg.get("dt", cfg.time.dt))
        steps_first = int(lane_cfg["steps_first"])
        steps_second = int(lane_cfg["steps_second"])
        fixed_dt = bool(lane_cfg.get("fixed_dt", cfg.time.fixed_dt))
        sample_stride = int(lane_cfg.get("sample_stride", 1))
        diagnostics_stride = int(lane_cfg.get("diagnostics_stride", 1))
        state_rtol = float(lane_cfg.get("state_rtol", 1.0e-6))
        state_atol = float(lane_cfg.get("state_atol", 1.0e-9))
        diag_rtol = float(lane_cfg.get("diag_rtol", 1.0e-6))
        diag_atol = float(lane_cfg.get("diag_atol", 1.0e-9))

        time_cfg = replace(
            cfg.time,
            dt=dt,
            fixed_dt=fixed_dt,
            diagnostics=True,
            sample_stride=sample_stride,
            diagnostics_stride=diagnostics_stride,
            t_max=dt * float(steps_first + steps_second),
        )
        cfg_gate = replace(cfg, time=time_cfg)
        common = dict(
            ky_target=ky,
            kx_target=kx_target,
            Nl=nl,
            Nm=nm,
            dt=dt,
            sample_stride=sample_stride,
            diagnostics_stride=diagnostics_stride,
            return_state=True,
        )
        full = run_runtime_nonlinear(
            cfg_gate, steps=steps_first + steps_second, **common
        )
        part1 = run_runtime_nonlinear(cfg_gate, steps=steps_first, **common)
        if full.state is None or part1.state is None:
            raise RuntimeError(
                "Restart parity gate requires return_state=True results."
            )

        restart_path = out_dir / "restart.bin"
        write_netcdf_restart_state(
            restart_path, np.asarray(part1.state, dtype=np.complex64)
        )
        cfg_restart = replace(
            cfg_gate,
            init=replace(
                cfg_gate.init,
                init_file=str(restart_path),
                init_file_scale=1.0,
                init_file_mode="replace",
            ),
        )
        cont = run_runtime_nonlinear(cfg_restart, steps=steps_second, **common)
        if cont.state is None:
            raise RuntimeError("Restart continuation did not return a final state.")

        state_metrics = _complex_rel_metrics(
            np.asarray(full.state), np.asarray(cont.state)
        )
        diag_full = _diag_scalar_map(full)
        diag_cont = _diag_scalar_map(cont)
        diag_metrics = {
            name: _scalar_rel(diag_full[name], diag_cont[name])
            for name in diag_full.keys() & diag_cont.keys()
        }
        state_ok = bool(
            state_metrics["max_abs"] <= state_atol
            or state_metrics["max_rel"] <= state_rtol
        )
        diag_ok = all(
            values["abs"] <= diag_atol or values["rel"] <= diag_rtol
            for values in diag_metrics.values()
        )
        ok = bool(state_ok and diag_ok)
        summary = {
            "config": str(cfg_path),
            "env": env_updates,
            "ky": ky,
            "kx_target": kx_target,
            "Nl": nl,
            "Nm": nm,
            "dt": dt,
            "fixed_dt": fixed_dt,
            "steps_first": steps_first,
            "steps_second": steps_second,
            "state_metrics": state_metrics,
            "diag_metrics": diag_metrics,
            "state_tolerances": {"rtol": state_rtol, "atol": state_atol},
            "diag_tolerances": {"rtol": diag_rtol, "atol": diag_atol},
            "ok": ok,
        }
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        return summary


def main_restart_parity(argv: list[str] | None = None) -> int:
    args = build_restart_parity_parser().parse_args(argv)
    manifest_path = args.manifest.expanduser().resolve()
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
        config_path = _resolve_manifest_path(
            lane_cfg["config"], manifest_dir=manifest_path.parent
        )
        lane_out = out_root / lane_key
        lane_out.mkdir(parents=True, exist_ok=True)
        lane_summary = _restart_lane_summary(config_path, lane_cfg, out_dir=lane_out)
        summary["lanes"][lane_key] = lane_summary
        if not bool(lane_summary["ok"]):
            failed.append(lane_key)

    summary_path = out_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"saved {summary_path}")
    if failed:
        raise SystemExit(f"Restart parity gate failed for lanes: {', '.join(failed)}")
    return 0


SUBCOMMANDS = {
    "device-parity": main_device_parity,
    "restart-parity": main_restart_parity,
    "vmec-roundtrip": main_vmec_roundtrip,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=sorted(SUBCOMMANDS))
    return parser


def main(argv: list[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    if not tokens or tokens[0] in {"-h", "--help"}:
        build_parser().parse_args(tokens)
        return 0
    command, rest = tokens[0], tokens[1:]
    try:
        handler = SUBCOMMANDS[command]
    except KeyError:
        build_parser().parse_args([command])
        return 2
    return handler(rest)


if __name__ == "__main__":
    raise SystemExit(main())
