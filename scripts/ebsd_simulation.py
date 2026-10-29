# =============================================================================
# EBSD 菊池带仿真（abTEM + RRS，真实 EBSD 物理）—— 样品内非相干点源模型
#
# 物理模型（Winkelmann 2009 / Pogany-Turner 互易，经本仓库实验验证）：
#   EBSD 电子 = 样品内非相干点源向出射方向 + 动力学散射（RRS）。
#   * 相干平面波（固定入射/逐层叠加/能散展宽）经 RRS 传播，倒空间始终是
#     离散布拉格斑点 —— 因为相干平面波只激发特定倒格点。
#   * 必须用样品内非相干点源（abTEM Probe 聚焦波包），其倒空间展宽覆盖全方向，
#     一次传播即覆盖所有屏方向（无需逐方向扫描），得到连续的动力学分布（菊池带）。
#
# 决定性参数：点源倒空间展宽宽度 = semiangle_cutoff：
#   * <=10 mrad：点源倒空间 FWHM~0 -> 退化为相干平面波 -> 离散斑点
#   * >=30 mrad：点源倒空间 FWHM~0.6 1/A -> 非相干源 -> 连续菊池带
#
# 实现步骤：
#   1) 在样品内多个深度 z_s 各放一个 30 mrad Probe 点源；
#   2) 用 abtem.finite_difference.multislice_step(..., fully_corrected=True) 做
#      RRS（RealSpace full，含背散射修正）从 z_s 传播到出射面；
#   3) 出口波倒空间强度（|FFT|²）对全部深度非相干叠加（模方和）；
#   4) 可选：gnomonic 投影把 (kx,ky)->θ=asin(λk)->X=L·tanθ 映射到屏坐标。
#
# 对比：有限差分(acc=8=1D9点/2D17Pt) vs FFT 谱拉普拉斯（laplace_method="fft"）
#       两种背散射修正项的点源 EBSD 差异。
#
# 参数默认对齐 Liu 2016：Mo [001]、20 kV、切片 0.01nm、DW 0.4 Å²、lobato。
#
# 用法：
#   python ebsd_simulation.py --quick            # 冒烟
#   python ebsd_simulation.py                    # 默认(2a x16c)
#   python ebsd_simulation.py --method fft       # 用 FFT 拉普拉斯做背散射修正
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

VOLTAGE = 20e3
SEMIANGLE_MRAD = 30.0      # 点源会聚半角：决定倒空间展宽(>=30 才出连续带)
SLICE_THICKNESS_ANG = 0.1  # 0.01 nm
DW_ANG2 = 0.4
PARAMETRIZATION = "lobato"


def apply_log(I, C=1.5e5):
    imax = max(float(np.max(np.abs(I))), 1e-30)
    return np.log(1.0 + C * np.abs(I) / imax)


def gnomonic_project(I, kx, ky, wavelength, L=1.0):
    """倒空间强度 I(kx,ky) 经 gnomonic(透视)投影映射到屏坐标 X=L·tanθx, Y=L·tanθy。"""
    import scipy.ndimage as ndi

    ny, nx = I.shape
    sx = np.clip(wavelength * kx, -0.999, 0.999)
    sy = np.clip(wavelength * ky, -0.999, 0.999)
    X = L * np.tan(np.arcsin(sx))
    Y = L * np.tan(np.arcsin(sy))
    Xq = np.linspace(float(np.min(X)), float(np.max(X)), nx)
    Yq = np.linspace(float(np.min(Y)), float(np.max(Y)), ny)
    kx_q = np.sin(np.arctan(Xq / L)) / wavelength
    ky_q = np.sin(np.arctan(Yq / L)) / wavelength
    dkx = float(kx[1] - kx[0]) if nx > 1 else 1.0
    dky = float(ky[1] - ky[0]) if ny > 1 else 1.0
    XX, YY = np.meshgrid(kx_q, ky_q)
    col = (XX - float(kx[0])) / dkx
    row = (YY - float(ky[0])) / dky
    I_proj = ndi.map_coordinates(I, np.stack([row, col]), order=1, mode="nearest", cval=0.0)
    return I_proj, Xq, Yq


def main():
    p = argparse.ArgumentParser(description="EBSD 菊池带：非相干点源 + RRS + gnomonic")
    p.add_argument("--voltage", type=float, default=VOLTAGE)
    p.add_argument("--in-plane", type=int, default=2)
    p.add_argument("--z-repeat", type=int, default=16)
    p.add_argument("--ppc", type=int, default=32)
    p.add_argument("--semiangle", type=float, default=SEMIANGLE_MRAD)
    p.add_argument("--source", choices=["probe", "delta"], default="probe",
                   help="点源类型：probe=会聚探针(半角由--semiangle定)；delta=极窄高斯脉冲(纯点源)")
    p.add_argument("--depths", type=int, default=6, help="非相干叠加的点源深度数")
    p.add_argument("--method", choices=["fd", "fft"], default="fd")
    p.add_argument("--accuracy", type=int, default=8)
    p.add_argument("--max-terms", type=int, default=80)
    p.add_argument("--no-proj", action="store_true", help="关闭 gnomonic 投影图")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.in_plane, args.z_repeat, args.ppc, args.depths = 2, 8, 32, 4

    cell = ase.io.read(CIF_PATH)
    a = float(cell.cell.lengths()[0])
    supercell = cell.repeat((args.in_plane, args.in_plane, args.z_repeat))
    extent = args.in_plane * a
    sampling = a / args.ppc
    N = args.in_plane * args.ppc
    supercell.set_array("debye_waller", np.full(len(supercell), DW_ANG2))

    pot = abtem.Potential(supercell, parametrization=PARAMETRIZATION, sampling=sampling,
                          slice_thickness=SLICE_THICKNESS_ANG, projection="finite",
                          device="gpu")
    slices = list(pot.generate_slices())
    ns = len(slices)
    wl = abtem.PlaneWave(energy=args.voltage, sampling=sampling, extent=(extent, extent)).accelerator.wavelength
    laplace = LaplaceOperator(args.accuracy, method=("finite_difference" if args.method == "fd" else "fft"))

    print("=" * 72)
    src_txt = f"{args.source}({args.semiangle}mrad)" if args.source == "probe" else "delta(纯点源)"
    print(f" EBSD 菊池带：非相干点源{src_txt} + RRS({args.method}) [{'QUICK' if args.quick else ''}]")
    print(f"  a={a:.4f} A | {args.in_plane}a x {args.z_repeat}c = {extent:.2f}x{args.z_repeat*a:.1f} A "
          f"| {N}x{N} 网格 | {ns} 切片 | wl={wl:.5f} A")
    print(f"  点源深度叠加: {args.depths}  | 拉普拉斯: {args.method} (acc={args.accuracy})")
    print("=" * 72)

    # 点源初值：probe=会聚探针(倒空间展宽由 semiangle 定)；delta=极窄高斯脉冲(纯点源，倒空间均匀) 
    import cupy as cp
    xg = np.linspace(0, extent, N, endpoint=False)
    Xg2, _ = np.meshgrid(xg, xg, indexing="xy")

    def _delta_psi0():
        # 极窄高斯脉冲（sigma ≈ 0.5 像素）近似 δ 函数；位于网格中心
        x0 = extent / 2
        s = (0.5 * sampling) ** 2
        r2 = (Xg2 - x0) ** 2 + (Xg2 - x0) ** 2
        psi0 = cp.exp(-cp.asarray(r2) / (2 * s)).astype(cp.complex128)
        psi0 /= cp.sqrt(cp.abs(psi0) ** 2).sum()
        return psi0

    def propagate_pointsource(z_s):
        if args.source == "delta":
            psi0 = _delta_psi0()
        else:
            pr = abtem.Probe(semiangle_cutoff=args.semiangle, sampling=sampling,
                             extent=(extent, extent), energy=args.voltage)
            psi0 = np.asarray(pr.build().array).astype(np.complex128)
            psi0 = cp.asarray(psi0)
            psi0 /= cp.sqrt(cp.abs(psi0) ** 2).sum()
        w = Waves(psi0, sampling=sampling, extent=(extent, extent), energy=args.voltage)
        for i in range(z_s, ns):
            if i < ns - 1:
                o = multislice_step(w, slices[i], next_slice=slices[i + 1],
                                    laplace=laplace, max_terms=args.max_terms,
                                    order=1, fully_corrected=True)
                w = o[0]
            else:
                o = multislice_step(w, slices[i], None, laplace=laplace,
                                    max_terms=args.max_terms, order=1, fully_corrected=True)
                w = o[0] if isinstance(o, tuple) else o
        arr = np.asarray(w.array.get())
        F = np.abs(np.fft.fftshift(np.fft.fft2(arr))) ** 2  # 出口倒空间强度
        return F

    depth_idx = np.linspace(0, ns - 1, args.depths).astype(int)
    S = np.zeros((N, N))
    print("\n全厚点源非相干叠加:")
    t0 = time.perf_counter()
    for z in depth_idx:
        F = propagate_pointsource(z)
        S += F / max(float(F.max()), 1e-30)
        print(f"  z={z:3d} ({z*SLICE_THICKNESS_ANG:.1f}A)  ({time.perf_counter()-t0:.0f}s)", flush=True)
    print(f"叠加完成 {time.perf_counter()-t0:.0f}s")

    dk = 1.0 / extent
    kx = (np.arange(N) - (N - 1) / 2) * dk
    ky = (np.arange(N) - (N - 1) / 2) * dk

    # ---- 图1：倒空间菊池带（log）+ 径向剖面 ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    im = axes[0].imshow(apply_log(S), extent=(kx[0], kx[-1], ky[0], ky[-1]),
                        origin="lower", cmap="inferno")
    axes[0].set_xlabel("kx [1/Å]"); axes[0].set_ylabel("ky [1/Å]")
    axes[0].set_title(f"Mo [001] EBSD Kikuchi ({src_txt} source+RRS, "
                      f"{args.depths} depths incoherent, {args.method})", fontsize=10)
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label="log intensity")
    Xg, Yg = np.meshgrid(kx, ky); R = np.sqrt(Xg ** 2 + Yg ** 2)
    nb = 80; edges = np.linspace(0, R.max(), nb + 1)
    idx = np.digitize(R.ravel(), edges)
    prof = np.array([S.ravel()[idx == b].mean() if (idx == b).any() else 0 for b in range(1, nb + 1)])
    rb = 0.5 * (edges[1:] + edges[:-1])
    th = np.degrees(np.arcsin(np.clip(wl * rb, 0, 0.999)))
    axes[1].semilogy(th, prof + 1e-12)
    axes[1].set_xlabel("scattering angle [deg]"); axes[1].set_ylabel("mean intensity (log)")
    axes[1].set_title("radial profile (Kikuchi/HOLZ rings)"); axes[1].grid(True, which="both", ls=":", alpha=0.5)
    fig.tight_layout()
    out1 = OUT_DIR / f"ebsd_kikuchi_{args.method}.png"
    fig.savefig(out1, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- 图2：gnomonic 投影（真实屏坐标）----
    save = {"S": S.astype(np.float32), "kx": kx, "ky": ky, "wl": wl, "a": a,
            "prof": prof, "rb": rb, "th": th, "method": args.method,
            "semiangle": args.semiangle, "depths": args.depths, "source": args.source}
    if not args.no_proj:
        Sp, Xq, Yq = gnomonic_project(S, kx, ky, wl, L=1.0)
        fig, ax = plt.subplots(figsize=(7, 6.4))
        im = ax.imshow(apply_log(Sp), extent=(Xq[0], Xq[-1], Yq[0], Yq[-1]),
                       origin="lower", cmap="inferno", aspect="equal")
        ax.set_xlabel("X = L·tan(θx) [L]"); ax.set_ylabel("Y = L·tan(θy) [L]")
        ax.set_title(f"Mo [001] EBSD gnomonic projection ({args.method})", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="log intensity")
        out2 = OUT_DIR / f"ebsd_kikuchi_{args.method}_gnomonic.png"
        fig.savefig(out2, dpi=300, bbox_inches="tight")
        plt.close(fig)
        save["PROJ"] = Sp.astype(np.float32)
        print(f"saved: {out1}; {out2}")

    # ---- 定量指标：径向峰位（HOLZ/菊池环）----
    # prof[i] 在 rb[i]；找局部极大（排除左端点），对齐索引
    peak_idx = np.where((prof[1:-1] > prof[:-2]) & (prof[1:-1] > prof[2:]))[0] + 1
    peak_idx = peak_idx[prof[peak_idx] > prof.max() * 0.2]
    print("\n径向峰位（scattering angle）, theta=asin(lambda*|k|):")
    for i in peak_idx[:6]:
        print(f"  {th[i]:5.2f} deg  I={prof[i]:.3e}")
    np.savez(OUT_DIR / f"ebsd_kikuchi_{args.method}.npz", **save)
    print(f"saved npz: {OUT_DIR / f'ebsd_kikuchi_{args.method}.npz'}")


if __name__ == "__main__":
    main()
