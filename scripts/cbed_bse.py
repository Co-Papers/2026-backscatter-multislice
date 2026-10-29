# =============================================================================
# 考虑背散射效应的 CBED 计算（GPU，1 个冷冻声子态，便于测试）
#
# 术语澄清（本脚本遵循）：
#   * 背散射 (backscattering)   ：被样品向后散射回去的那部分波/电子本身。
#   * 背散射效应 (backscattering effect)：背散射项对「前散射」的修正——
#       abTEM 实空间多切片在每层用 (Micron 190 (2025) 103778, Eq.13) 计算一个
#       背散射修正项，并 `前向波 -= 背散射项`。也就是说背散射效应是加在前向波上的。
#   * 本脚本计算的是「考虑背散射效应后的前散射 CBED」，并把它与「无背散射」的
#      前散射 CBED 对比，以量化背散射效应。
#
# 关键实现点：
#   * 背散射必须用 RealSpaceMultislice(expansion_scope="full")（FourierMultislice
#    固定 expansion_scope="propagator"，不支持背散射）。
#   * 背散射的 GPU Laplacian 原走 numba-CUDA，需要 libNVVM；本机 numba 0.67 只
#    支持 CUDA<=12.4（驱动 CUDA 13.2），所以在 abtem/finite_difference.py 里把
#    _laplace_stencil_gpu 改成了 cupy 实现（等价模板），从而 GPU 可跑。
#   * 此处用 return_backscattered=False（前向修正，返回带背散射效应的前向波），
#     避开该版本 return_backscattered=True 后处理里 "Wrong shapes" 的 bug。
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


def to_numpy(x):
    if hasattr(x, "compute"):
        x = x.compute()
    if hasattr(x, "get"):
        x = x.get()
    return np.asarray(x)


def mem_report(tag):
    """打印当前系统 RSS 与 GPU 显存占用（用于监控是否爆显存）。"""
    rss = psutil.Process().memory_info().rss / 1e6                    # MB
    free, total = cp.cuda.runtime.memGetInfo()                        # bytes
    gpu_used = (total - free) / 1e6                                    # MiB
    print(f"[mem] {tag}: RSS={rss:.0f}MB, GPU used={gpu_used:.0f}MiB / total={total/1e6:.0f}MiB", flush=True)


# -----------------------------------------------------------------------------
# 1) 原子模型 + 势场：1 个冷冻声子态（快速测试）
#    exit_planes=None：默认在"最后一个切片"（即穿过整个厚度后的出射面）取测量。
#    注意：不要写成 exit_planes=(-1,)——abtem 把开头的 -1 理解为"入口剖面"，
#    并因 (-1,) 里没有其它下标而不再标记任何切片为出射面；多构型（measurements
#    被分配）路径会在入口处把"入射探针波"当作测量写进去，导致 with/without
#    背散射逐位相同（差值恒为 0）。exit_planes=None 才会取真正的底部出射波。
#    这里不需要多出射面——"full" 展开的逐层背散射修正是渐进施加的，与出射面无关。
#    slice_thickness=0.05：实空间 "full" 展开的指数级数在切片太厚时会发散
#    （厚 z 内某些强切片的单层相位过大）。取更薄切片可摊薄每层相位，保证收敛。
# -----------------------------------------------------------------------------
srtio3 = ase.io.read(cif_path).repeat((8, 8, 50))
frozen_phonons = abtem.FrozenPhonons(srtio3, 1, {"Sr": 0.078, "Ti": 0.078, "O": 0.078})
potential = abtem.Potential(
    frozen_phonons,
    parametrization="lobato",
    sampling=0.05,
    slice_thickness=0.05,
    projection="finite",
    exit_planes=None,
    device="gpu",
)

# -----------------------------------------------------------------------------
# 2) 探针（同 cbed.py），网格匹配势场
# -----------------------------------------------------------------------------
probe = abtem.Probe(energy=5e3, semiangle_cutoff=9.4, device="gpu")  # 5 keV 测试
probe.grid.match(potential)
cell_x, cell_y, _ = srtio3.cell.lengths()

# -----------------------------------------------------------------------------
# 3) 前散射 CBED：带背散射效应（RealSpace full，前向修正） vs 无背散射（Fourier）
#    并对两种多层法分别计时
# -----------------------------------------------------------------------------
print("computing backscatter-affected forward CBED ...")
mem_report("before backscatter multislice")
_t0 = time.perf_counter()
exit_wave_bse = probe.multislice(
    potential,
    scan=[[cell_x / 2, cell_y / 2]],
    algorithm=RealSpaceMultislice(expansion_scope="full"),
    return_backscattered=False,   # 只要前向修正，不触发有 bug 的背散射返回路径
    lazy=False,
)
_t_bse = time.perf_counter() - _t0
mem_report("after backscatter multislice")
print(f"  backscattering multislice: {_t_bse:.2f} s")

print("computing reference (no backscattering) CBED ...")
mem_report("before regular forward multislice")
_t0 = time.perf_counter()
exit_wave_ref = probe.multislice(
    potential,
    scan=[[cell_x / 2, cell_y / 2]],
    algorithm=FourierMultislice(),  # 常规前向多切片，无背散射
    lazy=False,
)
_t_ref = time.perf_counter() - _t0
mem_report("after regular forward multislice")
print(f"  regular forward multislice: {_t_ref:.2f} s")
print(f"  ratio (backscatter / regular): {_t_bse / _t_ref:.2f}x")

# 把计时写进脚本输出：同时打印并落盘到文本文件
_st = float(potential.slice_thickness[0]) if np.ndim(potential.slice_thickness) else float(potential.slice_thickness)
_timing_report = (
    "cbed_bse timing\n"
    f"grid gpts: {probe.grid.gpts}, slices: {potential.num_slices}, slice_thickness: {_st:.3g} A\n"
    f"regular forward multislice (FourierMultislice): {_t_ref:.3f} s\n"
    f"backscattering multislice (RealSpaceMultislice full): {_t_bse:.3f} s\n"
    f"ratio (backscatter / regular): {_t_bse / _t_ref:.2f} x\n"
)
print(_timing_report, end="")
with open(out_dir / "cbed_bse_timing.txt", "w", encoding="utf-8") as f:
    f.write(_timing_report)
print(f"timing written: {out_dir / 'cbed_bse_timing.txt'}")

# 时间对比柱状图（与上面同一次运行测到的耗时）
_ratio = _t_bse / _t_ref
labels = ["regular forward\n(Fourier multislice)", "backscattering effect\n(RealSpace multislice full)"]
times_s = [_t_ref, _t_bse]
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(labels, times_s, color=["tab:blue", "tab:orange"])
for b, t in zip(bars, times_s):
    ax.text(b.get_x() + b.get_width() / 2, t, f"{t:.1f} s", ha="center", va="bottom", fontsize=10)
ax.set_ylabel("time [s]")
ax.set_ylim(0, max(times_s) * 1.25)
ax.set_title(f"Multislice time comparison  (backscatter/regular = {_ratio:.2f}x)")
fig.tight_layout()
fig.savefig(out_dir / "cbed_bse_timing.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"saved: {out_dir / 'cbed_bse_timing.png'}")

max_angle = "cutoff"
cbed_bse = exit_wave_bse.diffraction_patterns(max_angle=max_angle)
cbed_ref = exit_wave_ref.diffraction_patterns(max_angle=max_angle)


def extract_exit(arr):
    """从测量数组里取最后一个出射平面的 2D CBED。

    形状可能是 (config, Ny, Nx)（单出射面被压缩）或 (config, exit_planes, Ny, Nx)。
    """
    arr = to_numpy(arr)
    if arr.ndim == 4:
        return arr[0, -1]
    if arr.ndim == 3:
        return arr[0]
    return arr


cb_bse = extract_exit(cbed_bse.array)   # (Ny, Nx) 带背散射效应
cb_ref = extract_exit(cbed_ref.array)   # (Ny, Nx) 无背散射
dx, dy = cbed_bse.extent                   # 倒空间对称 ±extent

# -----------------------------------------------------------------------------
# 4) log 变换（I' = ln[1 + C*I/I_max]，C=1.5e5，与 cbed.py/radial_average.py 一致）
# -----------------------------------------------------------------------------
def apply_log(I, C=1.5e5):
    imax = max(float(np.max(np.abs(I))), 1e-30)
    return np.log(1.0 + C * np.abs(I) / imax)


def radial_annular_average(intensity, sampling, wavelength, bin_width=None):
    """径向（环形）平均：按出现的每个不同半径分环取平均，再重bin到均匀角网格，
    空 bin 用邻近值插值，保证曲线连续。theta_mrad = 1000 * wavelength * |k|。"""
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


# -----------------------------------------------------------------------------
# 5) 绘图：2×3 —— 第一行 CBED(log)；第二行 径向环形平均(log)
# -----------------------------------------------------------------------------
log_ref = apply_log(cb_ref)
log_bse = apply_log(cb_bse)
log_diff = apply_log(np.abs(cb_bse - cb_ref))

sampling = np.asarray(cbed_bse.sampling)      # 倒空间采样 [1/Å]
wl = probe.accelerator.wavelength             # 电子波长 [Å]
r_ref, ra_ref = radial_annular_average(log_ref, sampling, wl)
r_bse, ra_bse = radial_annular_average(log_bse, sampling, wl)
r_diff, ra_diff = radial_annular_average(log_diff, sampling, wl)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
cmap = "inferno"

im0 = axes[0, 0].imshow(log_ref, extent=(-dx, dx, -dy, dy), origin="lower", cmap=cmap)
axes[0, 0].set_title("(a) no backscattering (log)")
axes[0, 0].set_xlabel("kx [1/Å]"); axes[0, 0].set_ylabel("ky [1/Å]")
fig.colorbar(im0, ax=axes[0, 0], label="log intensity")

im1 = axes[0, 1].imshow(log_bse, extent=(-dx, dx, -dy, dy), origin="lower", cmap=cmap)
axes[0, 1].set_title("(b) with backscattering effect (log)")
axes[0, 1].set_xlabel("kx [1/Å]"); axes[0, 1].set_ylabel("ky [1/Å]")
fig.colorbar(im1, ax=axes[0, 1], label="log intensity")

im2 = axes[0, 2].imshow(log_diff, extent=(-dx, dx, -dy, dy), origin="lower", cmap=cmap)
axes[0, 2].set_title(r"(c) |with - without| (log, C=1.5e5)")
axes[0, 2].set_xlabel("kx [1/Å]"); axes[0, 2].set_ylabel("ky [1/Å]")
fig.colorbar(im2, ax=axes[0, 2], label="log |diff|")

axes[1, 0].plot(r_ref, ra_ref)
axes[1, 0].set_title("(d) radial avg of (a)")
axes[1, 0].set_xlabel("scattering angle [mrad]"); axes[1, 0].set_ylabel("log intensity")

axes[1, 1].plot(r_bse, ra_bse)
axes[1, 1].set_title("(e) radial avg of (b)")
axes[1, 1].set_xlabel("scattering angle [mrad]"); axes[1, 1].set_ylabel("log intensity")

axes[1, 2].plot(r_diff, ra_diff)
axes[1, 2].set_title("(f) radial avg of (c)")
axes[1, 2].set_xlabel("scattering angle [mrad]"); axes[1, 2].set_ylabel("log intensity")

fig.tight_layout()
fig.savefig(out_dir / "cbed_bse.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"saved: {out_dir / 'cbed_bse.png'}")

# -----------------------------------------------------------------------------
# 5) 汇总：量化背散射效应对前散射 CBED 的影响
# -----------------------------------------------------------------------------
diff_abs = float(np.abs(cb_bse - cb_ref).max())
rel = diff_abs / max(float(np.abs(cb_ref).max()), 1e-30)
print(f"CBED grid: {cb_bse.shape}, reciprocal extent: ({dx:.3f}, {dy:.3f}) 1/A")
print(f"backscattering effect: max |with-without| = {diff_abs:.3e}, relative to ref max = {rel:.3e}")
print("done: backscatter-affected CBED (forward correction)")
