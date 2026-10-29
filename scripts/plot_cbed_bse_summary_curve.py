# =============================================================================
# 从 cbed_bse_*keV.npz 生成"背散射效应 vs 厚度"汇总曲线图（2×3 六面板，共享 x=厚度 [Å]）
#   面板 (a) 绝对峰值：max|with-without|                                  （log y）
#   面板 (b) 相对峰值：max|with-without| / max|without|  [%]               （log y）
#   面板 (c) 相对总量(总变化量 L1)：Σ|with-without| / Σ|without|   [%]     （log y）
#   面板 (d) 环形平均相对峰值：maxθ RA(|diff|) / maxθ RA(|without|) [%]   （log y）
#   面板 (e) 净强度相对变化：(ΣI_with − ΣI_without)/ΣI_without   [%]       （线性，≈0=守恒）
#   面板 (f) 不平衡占比：|net| / Σ|diff|  [%]                             （线性，小=重分布）
#   每条线 = 一个能量档。
#
# 【c 与 e 不是同一数据】——都源自 with/without 数组、都除以 Σ|without|，但分子不同：
#   * 面板 c：Σ|with − without| / Σ|without|   分子 = 每像素差值取绝对值再求和（L1 总量），
#     度量"总共被搬动/新增了多少强度"，恒 ≥ 0；不区分"搬走"还是"净增"。
#   * 面板 e：(Σwith − Σwithout) / Σwithout    分子 = 带符号净差（不取绝对值），
#     度量"总强度净增/净减"，可正可负。
#   由三角不等式 Σ|Δ| ≥ |ΣΔ| 立得 **c ≥ |e|**。若 c、e 数值不同（如 5 keV：c≈99%、e=+13%），
#   则说明除搬动外还有明显净变化，不能把 c 当纯重分布。
#   二者关系正是面板 f：unbal = |net| / Σ|diff| = |e| / c。
#      f 很小（净量 ≪ 总量）→ 强度主要被"搬动"、总强度≈不变 → 重分布（c 可当重分布用）；
#      f 大（5 keV：|e|/c≈13/99≈0.13）→ 搬动之外还有净变化 → c = 重分布 + 净增益。
#
# 用法：
#   python plot_cbed_bse_summary_curve.py                    # 默认 paper/figures
#   python plot_cbed_bse_summary_curve.py paper/figures_multi16
#
# =============================================================================
# 【重要】"redistribution(重分布)"的严谨定义 —— 别把面板 (c) 无条件当成"重分布"。
#
#   重分布 = 强度被在前向探测器窗口内"搬动"，且**总强度近似守恒（net≈0）**。
#   只有满足这个前提，出现"差异"才能叫重分布。
#
#   * 面板 (c)  Σ|I_with − I_without| / ΣI_without 度量的是**总变化量(L1)**，
#     它并不自动等于"重分布"：
#       - 高电压(60–300 keV)：net 相对变化仅 ~0.2%（总强度几乎不变）
#         → 此时 (c) 确实 ≈ 纯重分布，(d) 的径向平均差异才是重分布的体现。
#       - 低电压(5 keV)：net 达 +13%（总强度反而**增加**）→ 此时 (c) 是
#         "重分布 + 净增益"的混合，**不能**解读为纯重分布。
#
#   * 判断是否重分布的正确依据：看面板 (e) 净相对变化是否 ≈ 0（守恒检验）。
#     只有 "net≈0" 且 "仍有差异(见 c/d)" 同时成立，才算重分布。
#     例：高电压下 总强度≈不变 + 径向平均差异(面板 d) 非零 → 这才是重分布；
#         5 keV 总强度增加 13% → 是重分布+净变化，非纯重分布。
# =============================================================================
# 【数据与坐标轴口径】面板 a–d 的"差异"度量全部基于**原始强度**，不是 log 变换值。
#
#   * npz 里存的是原始强度：cbed_without / cbed_with / cbed_diff = |with − without|。
#     （见 cbed_bse_energy_sweep.py 的 np.savez。）
#   * apply_log(I)=ln(1+C·I/I_max) 是非线性压缩变换，仅用于单套「每厚度×4列」
#     展示图的 imshow，**从未写入 npz**。若在 log 域做差，log(with)−log(without)
#     不再对应物理强度变化，那些百分比会失真，所以必须用原始强度。
#   * 注意：a–d 用 set_yscale("log")，所以**图上那几条曲线就是展示在对数尺度下**的
#     （等价于把 max|diff|、Σ|diff| 等度量值取 log 后画在[线性轴]上，曲线形状完全相同）。
#     取对数轴只因各能量档量级相差几个数量级（5 keV 峰值差 ~1e-2 → 300 keV ~3e-5），
#     线性轴会贴底看不清。**被取对数的只是"度量曲线"（显示层面），不是计算度量所用的
#     原始 CBED 强度**——度量 max|diff| 等都是先在原始强度数组上算出来，再落到 log 轴展示。
#     面板 e/f 是线性轴，所以没有 10^0 这种刻度。
# =============================================================================
import sys
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

out_dir = Path(__file__).resolve().parent.parent / "figures"


def parse_energy(path):
    m = re.search(r"(\d+)keV", Path(path).name)
    return int(m.group(1)) if m else None


def radial_annular_average(intensity, sampling, wavelength, bin_width=None):
    """径向（环形）平均：按不同半径分环取平均，再重 bin 到均匀角网格，空 bin 插值。
    theta_mrad = 1000 * wavelength * |k|。输入为强度。"""
    intensity = np.asarray(intensity, dtype=float)
    Ny, Nx = intensity.shape
    ky = (np.arange(Ny) - (Ny - 1) / 2) * sampling[0]
    kx = (np.arange(Nx) - (Nx - 1) / 2) * sampling[1]
    K = np.sqrt(kx[None, :] ** 2 + ky[:, None] ** 2)
    theta = (1000.0 * wavelength * K).ravel()
    I = intensity.ravel()
    key = np.round(theta, 3)
    uniq = np.unique(key)
    idx = np.searchsorted(uniq, key)
    cnt = np.bincount(idx, minlength=uniq.size)
    tot = np.bincount(idx, weights=I, minlength=uniq.size)
    prof_uniq = tot / np.maximum(cnt, 1)
    if bin_width is None:
        bin_width = 1000.0 * wavelength * float(sampling[0])
    edges = np.arange(uniq.min(), uniq.max() + bin_width, bin_width)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bidx = np.clip(np.digitize(uniq, edges) - 1, 0, centers.size - 1)
    bcnt = np.bincount(bidx, minlength=centers.size)
    btot = np.bincount(bidx, weights=prof_uniq, minlength=centers.size)
    profile = btot / np.maximum(bcnt, 1)
    empty = bcnt == 0
    if empty.any() and (~empty).any():
        profile[empty] = np.interp(centers[empty], centers[~empty], profile[~empty])
    return centers, profile


def ann_peak(intensity, sampling, wl):
    """环形平均后的（剖面峰值, mrad 中心）。"""
    centers, prof = radial_annular_average(intensity, sampling, wl)
    if len(centers) == 0 or not np.all(np.isfinite(prof)):
        return 0.0, 0.0
    ip = int(np.argmax(prof))
    return float(prof[ip]), float(centers[ip])


def main(data_dir=out_dir):
    data_dir = Path(data_dir)
    files = sorted(glob.glob(str(data_dir / "cbed_bse_*keV.npz")),
                   key=lambda f: parse_energy(f))
    print(f"{data_dir}: {len(files)} npz files", flush=True)

    rows = []
    for f in files:
        E = parse_energy(f)
        if E is None:
            continue
        d = np.load(f)
        rows.append((E, np.asarray(d["thickness_angstrom"]),
                     np.asarray(d["cbed_diff"]), np.asarray(d["cbed_without"]),
                     np.asarray(d["cbed_with"]),
                     np.asarray(d["sampling"]), float(np.asarray(d["wavelength"]))))
    rows.sort(key=lambda r: r[0])
    if not rows:
        print("no data found"); return

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True)
    cmap = plt.get_cmap("viridis")
    n = len(rows)

    handles, labels = None, None
    for i, (E, t, diff, without, with_, sampling, wl) in enumerate(rows):
        off = np.array([float(np.abs(diff[j]).max()) for j in range(len(t))])
        rel_max = off / np.array([max(float(np.abs(without[j]).max()), 1e-30) for j in range(len(t))]) * 100.0
        rel_tot = (np.array([float(np.abs(diff[j]).sum()) for j in range(len(t))])
                   / np.array([max(float(np.abs(without[j]).sum()), 1e-30) for j in range(len(t))]) * 100.0)
        ann_diff = np.array([ann_peak(np.abs(diff[j]), sampling, wl)[0] for j in range(len(t))])
        ann_with = np.array([ann_peak(np.abs(without[j]), sampling, wl)[0] for j in range(len(t))])
        rel_ann = ann_diff / np.maximum(ann_with, 1e-30) * 100.0
        # 守恒性：净相对变化 & 不平衡占比（证明强度是被"搬动"而非"净增/净减"）
        sum_with = np.array([float(np.abs(with_[j]).sum()) for j in range(len(t))])
        sum_without = np.array([float(np.abs(without[j]).sum()) for j in range(len(t))])
        net_rel = (sum_with - sum_without) / np.maximum(sum_without, 1e-30) * 100.0
        unbal = (np.abs(sum_with - sum_without)
                 / np.maximum(np.array([float(np.abs(diff[j]).sum()) for j in range(len(t))]), 1e-30) * 100.0)

        c = cmap(i / max(n - 1, 1))
        l0, = axes[0, 0].plot(t, off, marker="o", ms=3, color=c, label=f"{E} keV")
        axes[0, 1].plot(t, rel_max, marker="o", ms=3, color=c)
        axes[0, 2].plot(t, rel_tot, marker="o", ms=3, color=c)
        axes[1, 0].plot(t, rel_ann, marker="o", ms=3, color=c)
        axes[1, 1].plot(t, net_rel, marker="o", ms=3, color=c)
        axes[1, 2].plot(t, unbal, marker="o", ms=3, color=c)
        if handles is None:
            handles, labels = [l0], [f"{E} keV"]

    panels = [
        (0, 0, "max|with - without|   (log y)", "(a) absolute peak difference", "log"),
        (0, 1, "max diff / max without  [%]  (log y)", "(b) relative peak difference", "log"),
        (0, 2, "sum|diff| / sum|without|  [%]  (log y)", "(c) relative total change (L1)", "log"),
        (1, 0, "max_theta RA(|diff|) / RA(|without|)  [%]  (log y)", "(d) annular-averaged relative difference", "log"),
        (1, 1, "(sum_with - sum_without)/sum_without  [%]",
                "(e) NET intensity change (conservation check; ~0 = redistribution)", "linear"),
        (1, 2, "|net| / sum|diff|  [%]",
                "(f) unbalanced fraction (small = redistribution)", "linear"),
    ]
    for r, c, ylab, tit, scale in panels:
        ax = axes[r, c]
        if scale == "log":
            ax.set_yscale("log")
        ax.set_xlabel("thickness [Å]")
        ax.set_ylabel(ylab)
        ax.set_title(tit, fontsize=10)
        ax.grid(alpha=0.3, which="both")

    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, 1.0))
    out_png = out_dir / "cbed_bse_energy_thickness_curve.png"
    fig.suptitle("Backscattering effect vs thickness (single phonon config)\n"
                 "Note: 'redistribution' requires net intensity change ~0 (see (e)). "
                 "At 5 keV net increases ~13% (gain), so (c) = redistribution + net gain, not pure redistribution.",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_png}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(Path(sys.argv[1]))
    else:
        main(out_dir)
