#!/usr/bin/env python3
"""Run manifest-driven VMEC roundtrip determinism gates."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from spectraxgk.geometry import load_gx_geometry_netcdf
from spectraxgk.io import load_runtime_from_toml, load_toml
from spectraxgk.vmec_eik import generate_runtime_vmec_eik


FIELDS = (
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True, help="TOML manifest describing each VMEC roundtrip lane.")
    p.add_argument("--lane", type=str, default=None, help="Optional single lane key to run.")
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("tools_out") / "vmec_roundtrip_gate",
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


def _array_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    diff = np.asarray(a) - np.asarray(b)
    return {
        "max_abs": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "l2_abs": float(np.linalg.norm(diff.ravel())) if diff.size else 0.0,
    }


def _run_lane(cfg_path: Path, lane_cfg: dict[str, Any], *, out_dir: Path) -> dict[str, Any]:
    env_cfg = lane_cfg.get("env")
    env_updates = (
        {str(key): _expand_env_value(value) for key, value in env_cfg.items()}
        if isinstance(env_cfg, dict)
        else None
    )
    with _temporary_env(env_updates):
        cfg, _raw = load_runtime_from_toml(cfg_path)
        out1 = out_dir / "geom1.eik.nc"
        out2 = out_dir / "geom2.eik.nc"
        generate_runtime_vmec_eik(cfg, output_path=out1, force=True)
        generate_runtime_vmec_eik(cfg, output_path=out2, force=True)
        g1 = load_gx_geometry_netcdf(out1)
        g2 = load_gx_geometry_netcdf(out2)

        field_metrics = {
            name: _array_metrics(np.asarray(getattr(g1, name)), np.asarray(getattr(g2, name))) for name in FIELDS
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
        lane_out = out_root / lane_key
        lane_out.mkdir(parents=True, exist_ok=True)
        lane_summary = _run_lane(config_path, lane_cfg, out_dir=lane_out)
        summary["lanes"][lane_key] = lane_summary
        (lane_out / "summary.json").write_text(json.dumps(lane_summary, indent=2, sort_keys=True), encoding="utf-8")
        if not bool(lane_summary["ok"]):
            failed.append(lane_key)

    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"saved {summary_path}")
    if failed:
        raise SystemExit(f"VMEC roundtrip gate failed for lanes: {', '.join(failed)}")


if __name__ == "__main__":
    main()
