# =============================================================================
# 常规 CBED（Mo [001]，512x512 高分辨率）—— FourierMultislice 无条件稳定
#   用户指示：先用"常规 CBED"（背散射修正 RRS 在 20kV+细采样会级数发散，故先走常规）
#   * Probe 会聚束 + FourierMultislice（倒空间传播子，无条件稳定）
#   * 512x512 网格；|k| 窗口裁剪输出
#   * 输出：CBED 衍射盘图（log）、k 窗口裁剪图、径向剖面
#
# 用法：
#   python cbed_512_conventional.py --quick     # 冒烟(更小)
#   python cbed_512_conventional.py             # 512x512 默认
#   python cbed_512_conventional.py --kmax 5 --semiangle 30
# =============================================================================
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import ase.io
import abtem
from abtem.multislice import FourierMultislice

BASE = Path(__file__).resolve().parents[1]
CIF_PATH = BASE / "atomic_model" / "Mo.cif"
OUT_DIR = BASE / "figures"
OUT_DIR.mkdir(exist_ok=True)

VOLTAGE = 20e3
SLICE_THICKNESS = 0.5
DW_ANG2 = 0.4
PARAMETRIZATION = "lobato"


def apply_log(I, C=1.5e5):
    imax = max(float(np.max(np.abs(I))), 1e-30)
    return np.log(1.0 + C * np.abs(I) / imax)


def main():
    p = argparse.ArgumentParser(description="Mo [001] 常规 CBED (512x512, Fourier)")
    p.add_argument("--voltage", type=float, default=VOLTAGE)
    p.add_argument("--in-plane", type=int, default=2)
    p.add_argument("--z-repeat", type=int, default=16)
    p.add_argument("--ppc", type=int, default=256, help="每晶胞采样(512x512格子=2*256)")
    p.add_argument("--slice-thickness", type=float, default=SLICE_THICKNESS)
    p.add_argument("--semiangle", type=float, default=20.0, help="探针会聚半角[mrad]")
    p.add_argument("--kmax", type=float, default=5.0, help="输出窗口 |k|<kmax [1/A]")
    p.add_argument("--no-gpu", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.in_plane, args.z_repeat, args.ppc = 2, 8, 64

    cell = ase.io.read(CIF_PATH)
    a = float(cell.cell.lengths()[0])
    supercell = cell.repeat((args.in_plane, args.in_plane, args.z_repeat))
    extent = args.in_plane * a
    sampling = a / args.ppc
    N = args.in_plane * args.ppc
    supercell.set_array("debye_waller", np.full(len(supercell), DW_ANG2))

    device = "cpu" if args.no_gpu else "gpu"
    pot = abtem.Potential(supercell, parametrization=PARAMETRIZATION, sampling=sampling,
                          slice_thickness=args.slice_thickness, projection="finite", device=device)
    ns = len(pot)
    wl = abtem.PlaneWave(energy=args.voltage, sampling=sampling, extent=(extent, extent)).accelerator.wavelength

    print("=" * 72)
    print(f" Mo [001] 常规 CBED (FourierMultislice) [{args.ppc}ppc -> {N}x{N}]"
          f"{' [QUICK]' if args.quick else ''}")
    print(f"  a={a:.4f} A | {args.in_plane}a x {args.z_repeat}c = {extent:.2f}x{args.z_repeat*a:.1f} A "
          f"| sampling={sampling:.5f} A | ns={ns} | wl={wl:.5f} A")
    print(f"  probe semiangle={args.semiangle} mrad | 输出窗口 |k|<{args.kmax} 1/A")
    print("=" * 72)

    probe = abtem.Probe(semiangle_cutoff=args.semiangle, sampling=sampling,
                        extent=(extent, extent), energy=args.voltage)
    t0 = time.perf_counter()
    ew = probe.multislice(pot, algorithm=FourierMultislice(), lazy=False)
    t_ms = time.perf_counter() - t0
    dp = ew.diffraction_patterns(max_angle=None, renormalize=False)
    I = np.abs(np.asarray(dp.array)) ** 2 if np.iscomplexobj(np.asarray(dp.array)) else np.abs(np.asarray(dp.array))
    print(f"  multislice={t_ms:.1f}s  diffraction shape={I.shape}", flush=True)

    dk = 1.0 / extent
    kk = (np.arange(N) - (N - 1) / 2) * dk
    mask = np.abs(kk) <= args.kmax
    idx = np.where(mask)[0]
    Ic = I[idx[0]:idx[-1] + 1, idx[0]:idx[-1] + 1]
    kc = kk[idx]

    # ---- 图1：全范围 log ---- 
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    im = axes[0].imshow(apply_log(I), extent=(kk[0], kk[-1], kk[0], kk[-1]),
                        origin="lower", cmap="inferno")
    axes[0].set_xlabel("kx [1/Å]"); axes[0].set_ylabel("ky [1/Å]")
    axes[0].set_title(f"Full range {N}x{N} (Probe {args.semiangle}mrad)", fontsize=10)
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label="log intensity")

    # ---- 图2：k 窗口裁剪（菊池/CBED 盘核心区）----
    im = axes[1].imshow(apply_log(Ic), extent=(kc[0], kc[-1], kc[0], kc[-1]),
                        origin="lower", cmap="inferno")
    axes[1].set_xlabel("kx [1/Å]"); axes[1].set_ylabel("ky [1/Å]")
    axes[1].set_title(f"|k|<{args.kmax} 1/A (window)", fontsize=10)
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="log intensity")
    fig.suptitle(f"Mo [001] conventional CBED (Fourier) @ {args.voltage/1e3:.0f} kV, "
                 f"{args.ppc}ppc/{N}^2, DW={DW_ANG2} A^2", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out1 = OUT_DIR / f"cbed_mo_512_conventional.png"
    fig.savefig(out1, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- 径向剖面（峰位）----
    Xg, Yg = np.meshgrid(kc, kc); R = np.sqrt(Xg ** 2 + Yg ** 2)
    nb = 120; edges = np.linspace(0, R.max(), nb + 1)
    dig = np.digitize(R.ravel(), edges)
    prof = np.array([Ic.ravel()[dig == b].mean() if (dig == b).any() else 0 for b in range(1, nb + 1)])
    rb = 0.5 * (edges[1:] + edges[:-1])
    th = np.degrees(np.arcsin(np.clip(wl * rb, 0, 0.999)))
    peak_idx = np.where((prof[1:-1] > prof[:-2]) & (prof[1:-1] > prof[2:]))[0] + 1
    peak_idx = peak_idx[prof[peak_idx] > prof.max() * 0.2]
    print("\n径向峰位 (theta=asin(lambda*|k|)):")
    for i in peak_idx[:8]:
        print(f"  {th[i]:5.2f} deg  I={prof[i]:.3e}")

    np.savez(OUT_DIR / f"cbed_mo_512_conventional.npz", I=I.astype(np.float32),
             Ic=Ic.astype(np.float32), kk=kk, kc=kc, wl=wl, a=a,
             prof=prof, th=th, semiangle=args.semiangle, sampling=sampling)
    print(f"saved: {out1}; npz")


if __name__ == "__main__":
    main()
