#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import numpy as np
from netCDF4 import Dataset


def _write_gx_vmec_input(path: Path, request: object) -> None:
    lines = [
        "debug = false",
        "",
        "[Dimensions]",
        f"ntheta = {int(request.ntheta)}",
        "",
        "[Domain]",
        f"boundary = \"{str(request.boundary)}\"",
        f"y0 = {float(request.y0)!r}",
        "",
        "[Geometry]",
        'geo_option = "vmec"',
        f"vmec_file = \"{str(Path(request.vmec_file).resolve())}\"",
        f"torflux = {float(request.torflux)!r}",
        f"alpha = {float(request.alpha)!r}",
        f"npol = {float(request.npol)!r}",
        f"isaxisym = {str(bool(request.isaxisym)).lower()}",
        "",
        "[Physics]",
        f"beta = {float(request.beta)!r}",
        "",
        "[species]",
        f"z = [{', '.join(repr(float(v)) for v in request.z)}]",
        f"mass = [{', '.join(repr(float(v)) for v in request.mass)}]",
        f"dens = [{', '.join(repr(float(v)) for v in request.dens)}]",
        f"temp = [{', '.join(repr(float(v)) for v in request.temp)}]",
        f"tprim = [{', '.join(repr(float(v)) for v in request.tprim)}]",
        f"fprim = [{', '.join(repr(float(v)) for v in request.fprim)}]",
        f"vnewk = [{', '.join(repr(float(v)) for v in request.vnewk)}]",
        f"type = [{', '.join(repr(str(v)) for v in request.species_type)}]",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _stats(path: Path, name: str) -> tuple[float, float, float]:
    with Dataset(path, "r") as ds:
        arr = np.asarray(ds.variables[name][:], dtype=float)
    return float(np.min(arr)), float(np.max(arr)), float(np.linalg.norm(arr))


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root / "src") not in sys.path:
        sys.path.insert(0, str(repo_root / "src"))

    from spectraxgk.io import load_runtime_from_toml
    from spectraxgk.vmec_eik import build_gx_vmec_geometry_request, generate_runtime_vmec_eik

    cfg_path = (
        repo_root
        / "examples"
        / "nonlinear"
        / "non-axisymmetric"
        / "runtime_hsx_nonlinear_vmec_geometry.toml"
    )
    gx_repo = (repo_root / ".." / "gx").resolve()

    cfg, _ = load_runtime_from_toml(str(cfg_path))
    req = build_gx_vmec_geometry_request(cfg)

    out_dir = repo_root / "tmp_vmec_debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    gx_in = out_dir / "gx_vmec_geometry.toml"
    gx_nc = out_dir / "gx_vmec_geometry.eik.nc"
    internal_nc = out_dir / "internal_vmec.eik.nc"

    _write_gx_vmec_input(gx_in, req)

    gx_script = gx_repo / "geometry_modules" / "pyvmec" / "gx_geo_vmec.py"
    proc = subprocess.run(
        [sys.executable, str(gx_script), str(gx_in), str(gx_nc)],
        cwd=str(gx_script.parent),
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print("GX failed")
        print(proc.stdout)
        print(proc.stderr)
        return 2

    cfg_internal = replace(
        cfg,
        geometry=replace(cfg.geometry, geometry_backend="internal", geometry_file=str(internal_nc)),
    )
    generate_runtime_vmec_eik(cfg_internal, output_path=internal_nc, force=True)

    print(f"gx_file={gx_nc}")
    print(f"internal_file={internal_nc}")
    print(f"same_path={gx_nc.resolve() == internal_nc.resolve()}")

    for name in ["bmag", "gradpar", "gds2", "gds21", "gbdrift", "Rplot", "Zplot"]:
        gmin, gmax, gnorm = _stats(gx_nc, name)
        imin, imax, inorm = _stats(internal_nc, name)
        print(
            f"{name}: gx[min,max,norm]=({gmin:.6e},{gmax:.6e},{gnorm:.6e}) "
            f"internal[min,max,norm]=({imin:.6e},{imax:.6e},{inorm:.6e})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
