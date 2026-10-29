# =============================================================================
# 多电压背散射 CBED 厚度系列扫描：5/10/30/60/80/100/200/300 keV
#   - 单冷冻声子态（NUM_CONFIGS=1，测两算法差异足够）
#   - 对数分布 8 个厚度点，每个电压输出：
#       * cbed_bse_{E}keV.npz   原始 CBED 数据（with / without / |diff| + 元数据）
#       * cbed_bse_{E}keV.png   8(厚度) × 3(无/有/差值) 网格图（log）
#   - 依赖对 abtem/finite_difference.py 的 cupy 改（背散射 GPU Laplacian 走 cupy）。
# 关键点：
#   * exit_planes=厚度系列切片索引：abtem 会把开头 -1 当入口剖面、且不标记任何切片
#     为出射面（见 cbed_bse.py 注释），所以用 None 或真实底部索引——厚度系列则用
#     聚合的切片索引元组，一次传出全部厚度（返回 (1, n_exit, H, W)，带 ThicknessAxis）。
#   * 背散射用 RealSpaceMultislice(expansion_scope="full")，Fourier 固定"propagator"
#     不能背散射；return_backscattered=False 做前向修正并避开应"Wrong shapes" bug。
#   * slice_thickness=0.05 防实空间 full 指数级数发散（厚片单层相位过大）。
#   * apply_log 仅用于网格图显示，npz 存原始强度；见 apply_log 函数注释的口径说明。
# =============================================================================
import abtem
import ase.io
import numpy as np
import matplotlib.pyplot as plt
import time
import psutil
import cupy as cp
from pathlib import Path
from abtem.multislice import RealSpaceMultislice, FourierMultislice

cif_path = Path(__file__).resolve().parents[1] / "atomic_model" / "SrTiO3.cif"
out_dir = Path(__file__).resolve().parent.parent / "figures"
out_dir.mkdir(exist_ok=True)

ENERGIES_KEV = [5, 10, 30, 60, 80, 100, 200, 300]
NUM_CONFIGS = 1                  # 单冷冻声子态（测算法差异足够）
SLICE_THICKNESS = 0.05           # 保"full"展开收敛
MAX_ANGLE = "cutoff"             # 整个倒空间（±13.35 1/Å，约 419×419）
THICKNESS_A = [5, 10, 20, 40, 70, 100, 140, 195]   # 厚度系列 [Å]（对数分布）

srtio3 = ase.io.read(cif_path).repeat((8, 8, 50))
frozen_phonons = abtem.FrozenPhonons(
    srtio3, NUM_CONFIGS, {"Sr": 0.078, "Ti": 0.078, "O": 0.078}
)


def to_numpy(x):
    if hasattr(x, "compute"):
        x = x.compute()
    if hasattr(x, "get"):
        x = x.get()
    return np.asarray(x)


def mem_report(tag):
    rss = psutil.Process().memory_info().rss / 1e6
    free, total = cp.cuda.runtime.memGetInfo()
    print(f"[mem] {tag}: RSS={rss:.0f}MB, GPU used={(total-free)/1e6:.0f}MiB / total={total/1e6:.0f}MiB", flush=True)


def apply_log(I, C=1.5e5):
    """仅用于显示的 log 压缩：ln(1+C·I/I_max)。
    【口径】这只用于下面 n_exit×4 网格图的 imshow/径向平均展示，
    **从不写进 npz**——np.savez 存的 cbed_without / cbed_with / cbed_diff
    全是**原始强度**（|with−without| 在原始强度上取）。
    度量背散射效应必须用原始强度：若在 log 域做差，
    log(with)−log(without) 不再对应物理强度变化，百分比会失真。
    面板 a–d 那种"log 轴"只是展示层（set_yscale("log")），曲线形状等于把
    度量值取 log 后画在线性轴上，但计算度量的原始强度未做变换。"""
    imax = max(float(np.max(np.abs(I))), 1e-30)
    return np.log(1.0 + C * np.abs(I) / imax)


def radial_annular_average(intensity, sampling, wavelength, bin_width=None):
    """径向（环形）平均：按不同半径分环取平均，再重 bin 到均匀角网格，空 bin 插值。
    theta_mrad = 1000 * wavelength * |k|。输入应是已 log 变换的强度。"""
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


def extract_exit(arr):
    """取多出射面测量里的 2D CBED 数组。

    形状：(config, n_exit, ny, nx) → (n_exit, ny, nx)；
         (n_exit, ny, nx)        → 原样。
    """
    arr = to_numpy(arr)
    if arr.ndim == 4:
        return arr[0]
    return arr


def depth_slice_indices(num_slices):
    """由厚度 [Å] 换算成切片索引（去重、clamp，最后一个对齐到全厚度底部）。"""
    idx = []
    for t in THICKNESS_A:
        i = int(round(t / SLICE_THICKNESS))
        i = min(max(i, 0), num_slices - 1)
        if not idx or i > idx[-1]:
            idx.append(i)
    # 最后一厚度点 = 全厚度底部出射（去重，避免与前一近邻点重复）
    if idx[-1] != num_slices - 1:
        idx[-1] = num_slices - 1
        if len(idx) > 1 and idx[-1] == idx[-2]:
            idx.pop()
    return tuple(idx)


def run_energy(energy_kev, timing_lines):
    print(f"\n=== energy = {energy_kev} keV ===", flush=True)
    potential = abtem.Potential(
        frozen_phonons,
        parametrization="lobato",
        sampling=0.05,
        slice_thickness=SLICE_THICKNESS,
        projection="finite",
        exit_planes=None,               # 先建一次以取其 num_slices
        device="gpu",
    )
    num_slices = potential.num_slices
    depth_slices = depth_slice_indices(num_slices)
    thickness_angstrom = np.asarray([(i + 1) * SLICE_THICKNESS for i in depth_slices])

    # 重建：exit_planes 设为厚度系列切片索引，一次传出全部厚度
    potential = abtem.Potential(
        frozen_phonons,
        parametrization="lobato",
        sampling=0.05,
        slice_thickness=SLICE_THICKNESS,
        projection="finite",
        exit_planes=depth_slices,
        device="gpu",
    )
    probe = abtem.Probe(energy=energy_kev * 1e3, semiangle_cutoff=9.4, device="gpu")
    probe.grid.match(potential)
    cell_x, cell_y, _ = srtio3.cell.lengths()

    print(f"  num_slices={num_slices}  depth_slices={depth_slices}"
          f"  thickness_angstrom={np.round(thickness_angstrom,1)}", flush=True)

    print("  backscatter (RealSpace full) ...", flush=True)
    mem_report("before backscatter")
    t0 = time.perf_counter()
    ew_bse = probe.multislice(potential, scan=[[cell_x / 2, cell_y / 2]],
                              algorithm=RealSpaceMultislice(expansion_scope="full"),
                              return_backscattered=False, lazy=False)
    t_bse = time.perf_counter() - t0
    mem_report("after backscatter")

    print("  reference (Fourier) ...", flush=True)
    t0 = time.perf_counter()
    ew_ref = probe.multislice(potential, scan=[[cell_x / 2, cell_y / 2]],
                              algorithm=FourierMultislice(), lazy=False)
    t_ref = time.perf_counter() - t0
    mem_report("after reference")
    print(f"  forward={t_ref:.2f}s  backscatter={t_bse:.2f}s  ratio={t_bse/t_ref:.2f}x", flush=True)
    timing_lines.append((energy_kev, t_ref, t_bse))

    # 每能量计时柱状图（背散射 full vs 常规 Fourier）
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["regular forward\n(Fourier)", "backscattering effect\n(RealSpace full)"]
    times = [t_ref, t_bse]
    bars = ax.bar(labels, times, color=["tab:blue", "tab:orange"])
    for b, t in zip(bars, times):
        ax.text(b.get_x() + b.get_width() / 2, t, f"{t:.1f} s", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("time [s]"); ax.set_ylim(0, max(times) * 1.25)
    ax.set_title(f"{energy_kev} keV  (backscatter/regular = {t_bse/t_ref:.2f}x)")
    fig.tight_layout()
    fig.savefig(out_dir / f"cbed_bse_{energy_kev}keV_timing.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    cbed_bse = ew_bse.diffraction_patterns(max_angle=MAX_ANGLE)
    cbed_ref = ew_ref.diffraction_patterns(max_angle=MAX_ANGLE)
    cb_bse = extract_exit(cbed_bse.array)          # (n_exit, ny, nx)
    cb_ref = extract_exit(cbed_ref.array)          # (n_exit, ny, nx)
    n_exit, ny, nx = cb_bse.shape
    print(f"  cbed shapes: {cb_bse.shape}  extent={tuple(cbed_bse.extent)}", flush=True)

    dx, dy = np.abs(cbed_bse.extent)
    sampling = np.asarray(cbed_bse.sampling)
    wl = probe.accelerator.wavelength

    diff = np.abs(cb_bse - cb_ref)

    # ---- 原始数据 npz ----
    np.savez(
        out_dir / f"cbed_bse_{energy_kev}keV.npz",
        energy_kev=energy_kev,
        thickness_angstrom=thickness_angstrom,
        cbed_without=cb_ref,
        cbed_with=cb_bse,
        cbed_diff=diff,
        max_angle=MAX_ANGLE,
        sampling=sampling,
        wavelength=wl,
        extent=np.asarray([dx, dy]),
    )
    print(f"  saved npz: {out_dir / f'cbed_bse_{energy_kev}keV.npz'}", flush=True)

    # ---- n_exit(厚度) × 4 网格图，log ----
    log_ref = apply_log(cb_ref)
    log_bse = apply_log(cb_bse)
    log_diff = apply_log(diff)

    fig, axes = plt.subplots(n_exit, 4, figsize=(16, n_exit * 2.6))
    cols = ["no backscattering (log)", "with backscattering effect (log)",
            "|with - without| (log)", "radial avg of |diff| (log)"]
    for i in range(n_exit):
        # 前三列：CBED 图像
        for j, (arr, t) in enumerate([(log_ref[i], cols[0]),
                                      (log_bse[i], cols[1]),
                                      (log_diff[i], cols[2])]):
            ax = axes[i, j] if n_exit > 1 else axes[j]
            im = ax.imshow(arr, extent=(-dx, dx, -dy, dy), origin="lower", cmap="inferno")
            ax.set_xlabel("kx [1/Å]"); ax.set_ylabel("ky [1/Å]")
            if i == 0:
                ax.set_title(t, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"t={thickness_angstrom[i]:.0f} Å\nky [1/Å]", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="log intensity")
        # 第四列：径向平均差异（log），x=mrad
        rax = axes[i, 3] if n_exit > 1 else axes[3]
        r_diff, ra_diff = radial_annular_average(log_diff[i], sampling, wl)
        rax.plot(r_diff, ra_diff)
        rax.set_xlabel("scattering angle [mrad]"); rax.set_ylabel("log intensity")
        if i == 0:
            rax.set_title(cols[3], fontsize=10)
        rax.grid(alpha=0.3)
    fig.suptitle(f"{energy_kev} keV CBED thickness series (1 phonon config)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / f"cbed_bse_{energy_kev}keV.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figure: {out_dir / f'cbed_bse_{energy_kev}keV.png'}", flush=True)


if __name__ == "__main__":
    timing_lines = []
    for E in ENERGIES_KEV:
        run_energy(E, timing_lines)

    # 计时文本（仅记录）
    report = "energy_keV, forward_s, backscatter_s, ratio\n"
    for E, tr, tb in timing_lines:
        report += f"{E}, {tr:.2f}, {tb:.2f}, {tb/tr:.2f}\n"
    with open(out_dir / "cbed_bse_energy_timing.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nsaved timing: {out_dir / 'cbed_bse_energy_timing.txt'}", flush=True)

    # 多能量汇总计时柱状图
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(timing_lines)); w = 0.4
    ax.bar(x - w / 2, [tr for _, tr, _ in timing_lines], w, label="regular forward")
    ax.bar(x + w / 2, [tb for _, tb, _ in timing_lines], w, label="backscattering effect")
    for xi, (_, tr, tb) in enumerate(timing_lines):
        ax.text(xi - w / 2, tr, f"{tr:.0f}", ha="center", va="bottom", fontsize=8)
        ax.text(xi + w / 2, tb, f"{tb:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f"{E}" for E in ENERGIES_KEV])
    ax.set_xlabel("energy [keV]"); ax.set_ylabel("time [s]")
    ax.set_title("Multislice time vs energy (1 phonon config)")
    ax.legend(); fig.tight_layout()
    fig.savefig(out_dir / "cbed_bse_energy_timing.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved timing summary: {out_dir / 'cbed_bse_energy_timing.png'}", flush=True)
    print("done: energy sweep (thickness series, 1 phonon config)", flush=True)
