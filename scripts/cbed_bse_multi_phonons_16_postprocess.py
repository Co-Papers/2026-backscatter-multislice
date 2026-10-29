# =============================================================================
# 后处理：从 npz 数据生成「FD(17Pt) vs FFT 拉普拉斯」背散射 CBED 对比图与指标
#
# 说明
# ----
#   本脚本**不做任何 multislice 计算**，只读取已算好的 npz 数据，产出：
#      * 一行三列 CBED log 对比：(a) 常规 FourierMultislice（无背散射）、
#        (b) RealSpace FD acc=8（1D 9 点/2D 17Pt，有背散射）、
#        (c) RealSpace FFT Laplacian（有背散射）；
#      * 指标（FFT vs FD）：max|I_FFT-I_FD|、max|I_FFT-I_FD|/max(I_FD)、
#        (ΣI_FFT-ΣI_FD)/ΣI_FD；
#      * 计时（从计算脚本保存，若有）。
#
# 数据来源
# --------
#   主数据 npz：`cbed_bse_multi_phonons_16_fd_vs_fft.npz`（由计算脚本
#   cbed_bse_multi_phonons_16_fft.py 生成），字段 cbed_reference / cbed_fd / cbed_fft。
#   若其中缺 reference/fd，回退到已有多态 npz `cbed_bse_phonons_16_5keV.npz`
#   （字段 cbed_without / cbed_with，第 --thickness-index 个厚度）。
#
# 用法
# ----
#   python cbed_bse_multi_phonons_16_postprocess.py
#   python cbed_bse_multi_phonons_16_postprocess.py --data x.npz --out my.png
#   python cbed_bse_multi_phonons_16_postprocess.py --thickness-index -1
#
# 输出（到 paper/figures_multi_phonons_16/）
# ----
#   cbed_bse_multi_phonons_16_fd_vs_fft.png   （300 dpi）
# =============================================================================
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "figures_multi_phonons_16"
TAG = "fd_vs_fft"
FD_ACCURACY = 8   # 与计算脚本一致：1D 9 点/2D 17Pt


def apply_log(I, C=1.5e5):
    imax = max(float(np.max(np.abs(I))), 1e-30)
    return np.log(1.0 + C * np.abs(I) / imax)


def main():
    parser = argparse.ArgumentParser(description="从 npz 出 FD vs FFT 背散射 CBED 对比图（后处理，不计算）")
    parser.add_argument("--data", type=str,
                        default=str(OUT_DIR / f"cbed_bse_multi_phonons_16_{TAG}.npz"),
                        help="主数据 npz（计算脚本输出，含 cbed_fft）")
    parser.add_argument("--reference-npz", type=str,
                        default=str(OUT_DIR / "cbed_bse_phonons_16_5keV.npz"),
                        help="已有多态 npz，用于回退加载 cbed_without/cbed_with")
    parser.add_argument("--thickness-index", type=int, default=-1,
                        help="从多态 npz 取的厚度索引（默认 -1 = 底部全厚约 195 Å）")
    parser.add_argument("--out", type=str,
                        default=str(OUT_DIR / f"cbed_bse_multi_phonons_16_{TAG}.png"),
                        help="输出 PNG 路径")
    args = parser.parse_args()

    # ---- 加载主数据 npz ----
    d = np.load(args.data, allow_pickle=True)
    cb_ref = d["cbed_reference"] if "cbed_reference" in d else None
    cb_fd = d["cbed_fd"] if "cbed_fd" in d else None
    cb_fft = d["cbed_fft"] if "cbed_fft" in d else None
    energy = int(d["energy_kev"]) if "energy_kev" in d else 5
    n_cfg = int(d["n_configs"]) if "n_configs" in d else 16
    thickness = float(d["thickness_angstrom"]) if "thickness_angstrom" in d else 195.6

    # ---- 缺 ref/fd 时回退到已有多态 npz ----
    if cb_ref is None or cb_fd is None:
        dr = np.load(args.reference_npz, allow_pickle=True)
        idx = args.thickness_index
        if cb_ref is None and "cbed_without" in dr:
            cb_ref = dr["cbed_without"][idx]
        if cb_fd is None and "cbed_with" in dr:
            cb_fd = dr["cbed_with"][idx]
        # 元数据回退
        if "sampling" in dr:
            sampling = np.asarray(dr["sampling"])
            wl = float(dr["wavelength"])
            dx, dy = np.abs(np.asarray(dr["extent"]))

    if "sampling" in d:
        sampling = np.asarray(d["sampling"])
        wl = float(d["wavelength"]) if "wavelength" in d else 0.173
        dx, dy = np.abs(np.asarray(d.get("extent", [13.35, 13.35])))
    elif "sampling" not in locals():
        raise ValueError("主/参考 npz 均无 sampling/extent 元数据，请确认数据文件")

    for name, a in (("reference", cb_ref), ("fd", cb_fd), ("fft", cb_fft)):
        if a is None:
            raise ValueError(f"缺少 {name} 数据（{args.data} 或 {args.reference_npz}），先运行计算脚本")
    cb_ref = np.asarray(cb_ref, float)
    cb_fd = np.asarray(cb_fd, float)
    cb_fft = np.asarray(cb_fft, float)

    # ---- 指标：FFT vs FD ----
    diff = np.abs(cb_fft - cb_fd)
    max_abs = float(diff.max())
    max_fd = float(cb_fd.max())
    rel = max_abs / max(max_fd, 1e-30)
    int_cons = (float(cb_fft.sum()) - float(cb_fd.sum())) / max(float(cb_fd.sum()), 1e-30)

    print("=" * 78)
    print(f" 数据: {args.data}")
    print(f" 能量={energy} keV, 声子态={n_cfg}, 厚度≈{thickness:.1f} Å, 尺寸={cb_fft.shape}")
    print("-" * 78)
    print(" 指标（FFT-Laplacian 背散射 vs FD(17Pt) 背散射）:")
    print(f"   max|I_FFT - I_FD|                = {max_abs:.6e}")
    print(f"   max|I_FFT - I_FD| / max(I_FD)    = {rel:.6e}")
    print(f"   (ΣI_FFT - ΣI_FD) / ΣI_FD         = {int_cons:+.6e}")
    if "max_abs" in d:
        print(f"   （npz 内已存指标: max_abs={d['max_abs']:.3e}, rel={d['rel']:.3e}, "
              f"int_cons={d['intensity_conservation']:+.3e}）")
    print("=" * 78)

    # ---- 图：一行三列 CBED log 对比 ----
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))
    panels = [
        (cb_ref, "(a) FourierMultislice (no backscattering)"),
        (cb_fd, f"(b) RealSpace FD acc={FD_ACCURACY} ({2 * FD_ACCURACY + 1}Pt, with backscattering)"),
        (cb_fft, "(c) RealSpace FFT Laplacian (with backscattering)"),
    ]
    for ax, (I, ttl) in zip(axes, panels):
        im = ax.imshow(apply_log(I), extent=(-dx, dx, -dy, dy), origin="lower", cmap="inferno")
        ax.set_xlabel("kx [1/Å]"); ax.set_ylabel("ky [1/Å]")
        ax.set_title(ttl, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="log intensity")

    fig.suptitle(
        f"{energy} keV CBED, {n_cfg} phonon configs — FD ({2 * FD_ACCURACY + 1}Pt) "
        f"vs FFT Laplacian comparison",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figure: {args.out}")


if __name__ == "__main__":
    main()
