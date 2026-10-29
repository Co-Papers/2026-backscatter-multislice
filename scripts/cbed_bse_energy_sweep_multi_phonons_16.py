# =============================================================================
# 多冷冻声子态（16 态）背散射 CBED 厚度系列扫描：5/10/30/60/80/100/200/300 keV
#   与单态版 cbed_bse_energy_sweep.py 逻辑一致，差异仅：
#     * NUM_CONFIGS = 16（多冷冻声子态热平均）
#     * extract_exit 对构型求强度平均（mean(axis=0)）——单态脚本取 arr[0]，
#       这里若不平均会只剩第 0 个构型。
#     * 输出到独立文件夹 paper/figures_multi_phonons_16/，文件名含 "phonons_16" 标记。
#   依赖对 abtem/finite_difference.py 的 cupy 改（背散射 GPU Laplacian 走 cupy）。
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
out_dir = Path(__file__).resolve().parent.parent / "figures_multi_phonons_16"
out_dir.mkdir(exist_ok=True)

ENERGIES_KEV = [5, 10, 30, 60, 80, 120, 200, 300]
NUM_CONFIGS = 16                 # 多冷冻声子态
SLICE_THICKNESS = 0.05           # 保"full"展开收敛
MAX_ANGLE = "cutoff"             # 整个倒空间（±13.35 1/Å，约 419×419）
THICKNESS_A = [5, 10, 20, 40, 70, 100, 140, 195]   # 厚度系列 [Å]（对数分布）
TAG = "phonons_16"                      # 文件名标记：多冷冻声子态

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
    """多冷冻声子态测量 → 取构型热平均后的 (n_exit, ny, nx)。

    形状：(n_config, n_exit, ny, nx) → 对构型求强度平均 → (n_exit, ny, nx)。
         (n_config, ny, nx)         → 同（n_exit=1 情形）。
         (n_exit, ny, nx)           → 原样。
    """
    arr = to_numpy(arr)
    if arr.ndim == 4:
        return arr.mean(axis=0)               # (n_config, n_exit, ny, nx)
    if arr.ndim == 3:
        return arr.mean(axis=0) if arr.shape[0] > 1 else arr[0]
    return arr


def depth_slice_indices(num_slices):
    """由厚度 [Å] 换算成切片索引（去重、clamp，最后一个对齐到全厚度底部）。"""
    idx = []
    for t in THICKNESS_A:
        i = int(round(t / SLICE_THICKNESS))
        i = min(max(i, 0), num_slices - 1)
        if not idx or i > idx[-1]:
            idx.append(i)
    if idx[-1] != num_slices - 1:
        idx[-1] = num_slices - 1
        if len(idx) > 1 and idx[-1] == idx[-2]:
            idx.pop()
    return tuple(idx)


def run_energy(energy_kev, timing_lines):
    print(f"\n=== energy = {energy_kev} keV ({TAG}) ===", flush=True)
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
    fig.savefig(out_dir / f"cbed_bse_{TAG}_{energy_kev}keV_timing.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    cbed_bse = ew_bse.diffraction_patterns(max_angle=MAX_ANGLE)
    cbed_ref = ew_ref.diffraction_patterns(max_angle=MAX_ANGLE)
    cb_bse = extract_exit(cbed_bse.array)          # (n_exit, ny, nx) 构型平均后
    cb_ref = extract_exit(cbed_ref.array)          # (n_exit, ny, nx)
    n_exit, ny, nx = cb_bse.shape
    print(f"  cbed shapes: {cb_bse.shape}  extent={tuple(cbed_bse.extent)}", flush=True)

    dx, dy = np.abs(cbed_bse.extent)
    sampling = np.asarray(cbed_bse.sampling)
    wl = probe.accelerator.wavelength

    diff = np.abs(cb_bse - cb_ref)

    # ---- 原始数据 npz ----
    np.savez(
        out_dir / f"cbed_bse_{TAG}_{energy_kev}keV.npz",
        energy_kev=energy_kev,
        n_configs=NUM_CONFIGS,
        thickness_angstrom=thickness_angstrom,
        cbed_without=cb_ref,
        cbed_with=cb_bse,
        cbed_diff=diff,
        max_angle=MAX_ANGLE,
        sampling=sampling,
        wavelength=wl,
        extent=np.asarray([dx, dy]),
    )
    print(f"  saved npz: {out_dir / f'cbed_bse_{TAG}_{energy_kev}keV.npz'}", flush=True)

    # ---- n_exit(厚度) × 4 网格图，log ----
    log_ref = apply_log(cb_ref)
    log_bse = apply_log(cb_bse)
    log_diff = apply_log(diff)

    fig, axes = plt.subplots(n_exit, 4, figsize=(16, n_exit * 2.6))
    cols = ["no backscattering (log)", "with backscattering effect (log)",
            "|with - without| (log)", "radial avg of |diff| (log)"]
    for i in range(n_exit):
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
        rax = axes[i, 3] if n_exit > 1 else axes[3]
        r_diff, ra_diff = radial_annular_average(log_diff[i], sampling, wl)
        rax.plot(r_diff, ra_diff)
        rax.set_xlabel("scattering angle [mrad]"); rax.set_ylabel("log intensity")
        if i == 0:
            rax.set_title(cols[3], fontsize=10)
        rax.grid(alpha=0.3)
    fig.suptitle(f"{energy_kev} keV CBED thickness series ({NUM_CONFIGS} phonon configs)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / f"cbed_bse_{TAG}_{energy_kev}keV.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figure: {out_dir / f'cbed_bse_{TAG}_{energy_kev}keV.png'}", flush=True)


if __name__ == "__main__":
    timing_lines = []
    for E in ENERGIES_KEV:
        run_energy(E, timing_lines)

    report = "energy_keV, forward_s, backscatter_s, ratio\n"
    for E, tr, tb in timing_lines:
        report += f"{E}, {tr:.2f}, {tb:.2f}, {tb/tr:.2f}\n"
    with open(out_dir / f"cbed_bse_{TAG}_energy_timing.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nsaved timing: {out_dir / f'cbed_bse_{TAG}_energy_timing.txt'}", flush=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(timing_lines)); w = 0.4
    ax.bar(x - w / 2, [tr for _, tr, _ in timing_lines], w, label="regular forward")
    ax.bar(x + w / 2, [tb for _, tb, _ in timing_lines], w, label="backscattering effect")
    for xi, (_, tr, tb) in enumerate(timing_lines):
        ax.text(xi - w / 2, tr, f"{tr:.0f}", ha="center", va="bottom", fontsize=8)
        ax.text(xi + w / 2, tb, f"{tb:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f"{E}" for E in ENERGIES_KEV])
    ax.set_xlabel("energy [keV]"); ax.set_ylabel("time [s]")
    ax.set_title(f"Multislice time vs energy ({NUM_CONFIGS} phonon configs)")
    ax.legend(); fig.tight_layout()
    fig.savefig(out_dir / f"cbed_bse_{TAG}_energy_timing.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved timing summary: {out_dir / f'cbed_bse_{TAG}_energy_timing.png'}", flush=True)
    print(f"done: energy sweep ({NUM_CONFIGS} phonon configs)", flush=True)
