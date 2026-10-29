# =============================================================================
# 方案①：EBSD 菊池带仿真 — k⊥ 方向扫描 + RRS(RealSpace full) 多切片
#
# 物理（Winkelmann 2009 / Pogany-Turner 互易，方案①）：
#   EBSD 背散射出射 = 屏方向平面波入射的前向透射。对每个屏幕方向 k⊥=(kx,ky)，
#   构造带横向波矢 k⊥ 的入射平面波 ψ0 = exp(2πi(kx·x+ky·y))，
#   用 abTEM 的 realspace_multislice_step（=RRS，含背散射修正）逐切片传播，
#   记录出口波在内部交互体积各深度的振幅，非相干求和各深度强度，
#   得到该方向像素强度 I(kx,ky)。扫过全部 k⊥ → 连续菊池带。
#
# 注意：abTEM 的 PlaneWave.tilt 只作用于 Fresnel 传播子、不改变入射波矢，
#       因此必须手写 k⊥ 相位（本脚本核心），再调 realspace_multislice_step。
#
# 用法：
#   python ebsd_kikuchi_scan.py --quick            # 冒烟(低分辨)
#   python ebsd_kikuchi_scan.py --scan 21          # 中档 21x21
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
from abtem import Waves
from abtem.finite_difference import LaplaceOperator, multislice_step

BASE = Path(__file__).resolve().parents[1]
CIF_PATH = BASE / "atomic_model" / "Mo.cif"
OUT_DIR = BASE / "figures"
OUT_DIR.mkdir(exist_ok=True)


def apply_log(I, C=1.5e5):
    imax = max(float(np.max(np.abs(I))), 1e-30)
    return np.log(1.0 + C * np.abs(I) / imax)


def main():
    parser = argparse.ArgumentParser(description="EBSD 菊池带：k⊥ 方向扫描 + RRS")
    parser.add_argument("--voltage", type=float, default=20e3)
    parser.add_argument("--in-plane", type=int, default=2)
    parser.add_argument("--z-repeat", type=int, default=16)
    parser.add_argument("--ppc", type=int, default=32)
    parser.add_argument("--scan", type=int, default=21, help="NxN 方向网格")
    parser.add_argument("--kmax", type=float, default=1.0, help="最大 |k| [1/A]")
    parser.add_argument("--accuracy", type=int, default=6)
    parser.add_argument("--max-terms", type=int, default=80)
    parser.add_argument("--depths", type=int, default=8, help="非相干叠加的深度采样数")
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.in_plane, args.z_repeat, args.ppc, args.scan = 2, 8, 32, 11

    cell = ase.io.read(CIF_PATH)
    a = float(cell.cell.lengths()[0])
    supercell = cell.repeat((args.in_plane, args.in_plane, args.z_repeat))
    extent = args.in_plane * a
    sampling = a / args.ppc
    N = args.in_plane * args.ppc
    supercell.set_array("debye_waller", np.full(len(supercell), 0.4))

    pot = abtem.Potential(supercell, parametrization="lobato", sampling=sampling,
                          slice_thickness=0.1, projection="finite", device=args.device)
    slices = list(pot.generate_slices())
    n_slices = len(slices)
    wl = abtem.PlaneWave(energy=args.voltage, sampling=sampling, extent=(extent, extent)).accelerator.wavelength
    laplace = LaplaceOperator(args.accuracy, method="finite_difference")

    print("=" * 70)
    print(f" EBSD k⊥方向扫描 + RRS(RealSpace full)  [{'QUICK' if args.quick else ''}]")
    print(f"  晶格 a={a:.4f} A | 超胞 {args.in_plane}a x {args.z_repeat}c = "
          f"{extent:.2f} x {args.z_repeat*a:.1f} A | 网格 {N}x{N} | 切片 {n_slices}")
    print(f"  波长={wl:.5f} A | dk=1/extent={1/extent:.4f} 1/A | 方向网格 {args.scan}x{args.scan} "
          f"(|k|<={args.kmax} 1/A) | depth叠加={args.depths}")
    print("=" * 70)

    # 空间坐标
    x = np.linspace(0, extent, N, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="xy")
    # 深度采样切片索引（非相干叠加用）
    depth_idx = np.linspace(0, n_slices - 1, args.depths).astype(int)

    def propagate(kx, ky):
        """k⊥ 入射平面波 -> RRS 逐切片传播 -> 各深度 000 束强度非相干叠加。

        物理：EBSD 屏幕像素 = 该方向入射平面波的透射束 (000) 强度。
        当 k⊥ 满足某布拉格反射的激发时 000 束因动力学耦合而增强/减弱 -> 菊池带。
        逐深度取 000 强度再非相干求和（交互体积内多深度非相干）。
        """
        if args.device == "gpu":
            import cupy as cp
            xp = cp
        else:
            xp = np
        psi0 = xp.exp(2j * xp.pi * (kx * xp.asarray(X) + ky * xp.asarray(Y))).astype(xp.complex128)
        psi0 /= xp.sqrt(xp.abs(psi0) ** 2).sum()
        w = Waves(psi0, sampling=sampling, extent=(extent, extent), energy=args.voltage)
        amps = []
        for i in range(n_slices):
            if i < n_slices - 1:
                o = multislice_step(w, slices[i], next_slice=slices[i + 1],
                                    laplace=laplace, max_terms=args.max_terms,
                                    order=1, fully_corrected=True)
                w = o[0]
            else:
                o = multislice_step(w, slices[i], None, laplace=laplace,
                                    max_terms=args.max_terms, order=1, fully_corrected=True)
                w = o[0] if isinstance(o, tuple) else o
            if i in depth_idx:
                if args.device == "gpu":
                    arr = np.asarray(w.array.get())
                else:
                    arr = np.asarray(w.array)
                amps.append(float(np.abs(arr.sum()) ** 2 / arr.size ** 2))
        return float(np.sum(amps))   # 各深度 000 束强度非相干叠加

    # ---- 方向扫描 ----
    kvals = np.linspace(-args.kmax, args.kmax, args.scan)
    I2 = np.zeros((args.scan, args.scan))
    print("\n扫描 k⊥ 方向 ... (每方向一次完整RRS多切片)")
    t0 = time.perf_counter()
    for iy, ky in enumerate(kvals):
        for ix, kx in enumerate(kvals):
            I2[iy, ix] = propagate(kx, ky)
        # 保守：不逐方向打印，仅周期性的进度
        if (iy + 1) % 5 == 0:
            print(f"  完成 {iy+1}/{args.scan} 行，用时 {time.perf_counter()-t0:.0f}s", flush=True)
    print(f"方向扫描完成，用时 {time.perf_counter()-t0:.0f}s")

    # ---- 单位换算：k -> 散射角 ----
    kk = np.linspace(-args.kmax, args.kmax, args.scan)
    theta = np.degrees(np.arcsin(np.clip(wl * np.abs(kk), 0, 0.999)))

    # ---- 绘图 ----
    logI = apply_log(I2)
    fig, ax = plt.subplots(figsize=(7.5, 6.6))
    im = ax.imshow(logI, extent=(kk[0], kk[-1], kk[0], kk[-1]), origin="lower",
                   cmap="inferno", aspect="equal")
    ax.set_xlabel("kx [1/Å]"); ax.set_ylabel("ky [1/Å]")
    ax.set_title(f"Mo [001] EBSD Kikuchi (RRS) @ {args.voltage/1e3:.0f} kV, "
                 f"{args.depths} depths, {args.scan}^2 directions\n"
                 f"|psi(r0)|^2 incoherent; Bragg dark-line |k|=1/a={1/a:.4f} 1/A "
                 f"(theta~{np.degrees(np.arcsin(min(wl/a,0.999))):.2f} deg)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="log intensity")
    out = OUT_DIR / f"ebsd_kikuchi_{args.scan}x{args.scan}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    np.savez(OUT_DIR / f"ebsd_kikuchi_{args.scan}x{args.scan}.npz",
             I=I2, kx=kk, ky=kk, wl=wl, a=a)
    print(f"saved: {out}")
    print(f"  I.min={I2.min():.3e}  I.max={I2.max():.3e}")
    print("  验证：若 kx 方向在 |k|≈1/a 处出现暗线/亮线，即菊池带特征。")


if __name__ == "__main__":
    main()
