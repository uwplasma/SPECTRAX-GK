"""Compare SPECTRAX cETG runtime output against legacy GX cETG NetCDF."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from spectraxgk.gx_legacy_output import load_gx_legacy_cetg_output
from spectraxgk.io import load_runtime_from_toml
from spectraxgk.runtime import run_runtime_nonlinear


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gx-nc", required=True, type=Path, help="Legacy GX cETG NetCDF file")
    parser.add_argument("--config", required=True, type=Path, help="SPECTRAX runtime TOML config")
    parser.add_argument("--ky", type=float, default=None, help="Diagnostic ky target override")
    parser.add_argument("--kx", type=float, default=0.0, help="Diagnostic kx target override")
    parser.add_argument("--dt", type=float, default=None, help="Time step override")
    parser.add_argument("--steps", type=int, default=None, help="Step-count override")
    parser.add_argument("--sample-stride", type=int, default=1, help="Runtime diagnostic stride override")
    parser.add_argument("--out", type=Path, default=None, help="Optional CSV output path")
    return parser


def _interp_complex(x_new: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    y_real = np.interp(x_new, x, np.real(y))
    y_imag = np.interp(x_new, x, np.imag(y))
    return y_real + 1j * y_imag


def _rel_err(a: np.ndarray, b: np.ndarray, floor: float = 1.0e-30) -> np.ndarray:
    denom = np.maximum(np.abs(b), floor)
    return np.abs(a - b) / denom


def _leading_finite_prefix_mask(*series: np.ndarray) -> np.ndarray:
    if not series:
        raise ValueError("at least one series is required")
    mask = np.ones_like(np.asarray(series[0], dtype=bool), dtype=bool)
    for arr in series:
        arr_np = np.asarray(arr)
        if np.iscomplexobj(arr_np):
            finite = np.isfinite(arr_np.real) & np.isfinite(arr_np.imag)
        else:
            finite = np.isfinite(arr_np)
        mask &= finite
    prefix = np.cumprod(mask.astype(np.int8), dtype=np.int64).astype(bool)
    return prefix


def main() -> int:
    args = build_parser().parse_args()
    gx = load_gx_legacy_cetg_output(args.gx_nc)
    cfg, _data = load_runtime_from_toml(args.config)

    ky = float(args.ky) if args.ky is not None else float(gx.ky[1] if gx.ky.size > 1 else gx.ky[0])
    out = run_runtime_nonlinear(
        cfg,
        ky_target=ky,
        kx_target=float(args.kx),
        dt=args.dt,
        steps=args.steps,
        sample_stride=int(args.sample_stride),
        diagnostics=True,
    )
    if out.diagnostics is None:
        raise RuntimeError("SPECTRAX cETG run did not return diagnostics")

    t_s = np.asarray(out.diagnostics.t, dtype=float)
    t_gx = np.asarray(gx.time, dtype=float)
    if t_s.size == 0 or t_gx.size == 0:
        raise RuntimeError("both GX and SPECTRAX traces must contain at least one time sample")
    t_max = min(float(t_s[-1]), float(t_gx[-1]))
    t_common = t_gx[(t_gx >= float(max(t_s[0], t_gx[0]))) & (t_gx <= t_max)]
    if t_common.size == 0:
        raise RuntimeError("no overlapping time window between GX and SPECTRAX traces")

    W_s = np.interp(t_common, t_s, np.asarray(out.diagnostics.Wg_t, dtype=float))
    Phi2_s = np.interp(t_common, t_s, np.asarray(out.diagnostics.Wphi_t, dtype=float))
    qflux_s = np.interp(t_common, t_s, np.asarray(out.diagnostics.heat_flux_t, dtype=float))
    pflux_s = np.interp(t_common, t_s, np.asarray(out.diagnostics.particle_flux_t, dtype=float))
    phi_mode_s = _interp_complex(t_common, t_s, np.asarray(out.diagnostics.phi_mode_t, dtype=np.complex128))

    W_gx = np.interp(t_common, t_gx, np.asarray(gx.W, dtype=float))
    Phi2_gx = np.interp(t_common, t_gx, np.asarray(gx.Phi2, dtype=float))
    qflux_gx = np.interp(t_common, t_gx, np.asarray(gx.qflux[:, 0], dtype=float))
    pflux_gx = np.interp(t_common, t_gx, np.asarray(gx.pflux[:, 0], dtype=float))

    finite_prefix = _leading_finite_prefix_mask(
        W_s,
        Phi2_s,
        qflux_s,
        pflux_s,
        phi_mode_s,
        W_gx,
        Phi2_gx,
        qflux_gx,
        pflux_gx,
    )
    if not np.any(finite_prefix):
        raise RuntimeError("no leading finite overlap between GX and SPECTRAX cETG traces")
    t_eval = t_common[finite_prefix]
    W_eval = W_s[finite_prefix]
    W_gx_eval = W_gx[finite_prefix]
    Phi2_eval = Phi2_s[finite_prefix]
    Phi2_gx_eval = Phi2_gx[finite_prefix]
    qflux_eval = qflux_s[finite_prefix]
    qflux_gx_eval = qflux_gx[finite_prefix]
    pflux_eval = pflux_s[finite_prefix]
    pflux_gx_eval = pflux_gx[finite_prefix]

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        rows = np.column_stack(
            [
                t_common,
                W_s,
                W_gx,
                Phi2_s,
                Phi2_gx,
                qflux_s,
                qflux_gx,
                pflux_s,
                pflux_gx,
                np.real(phi_mode_s),
                np.imag(phi_mode_s),
            ]
        )
        np.savetxt(
            args.out,
            rows,
            delimiter=",",
            header="t,W_spectrax,W_gx,Phi2_spectrax,Phi2_gx,qflux_spectrax,qflux_gx,pflux_spectrax,pflux_gx,phi_mode_re,phi_mode_im",
            comments="",
        )

    print(
        f"window: samples={t_common.size} t_max={t_common[-1]:.6g} "
        f"finite_prefix_samples={t_eval.size} finite_prefix_t_max={t_eval[-1]:.6g}"
    )
    print(f"W rel_err_mean={np.mean(_rel_err(W_eval, W_gx_eval)):.6e}")
    print(f"Phi2 rel_err_mean={np.mean(_rel_err(Phi2_eval, Phi2_gx_eval)):.6e}")
    print(f"qflux rel_err_mean={np.mean(_rel_err(qflux_eval, qflux_gx_eval)):.6e}")
    print(f"pflux rel_err_mean={np.mean(_rel_err(pflux_eval, pflux_gx_eval)):.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
