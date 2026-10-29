# =============================================================================
# CBED 径向（环形）平均：把二维倒空间强度按到中心(000)的径向距离分环取平均，
# 用于精细化表征 CBED 特征。
#   横轴 = 散射角 theta [mrad]
#   纵轴 = 原始强度 或 log 强度（取决定调用方传入的是原始还是 log 数组）
# 注：abTEM 内置 polar_binning()/polar_spatial_frequencies() 也能实现；
#     这里给出一个独立、可控、可加的实现，并复用了 abtem 的波长/采样换算。
# 换算关系：angular_sampling[mrad] = 1000 * wavelength[A] * reciprocal_sampling[1/A]
# =============================================================================
import abtem
import ase.io
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

cif_path = Path(__file__).resolve().parents[1] / "atomic_model" / "SrTiO3.cif"
out_dir = Path(__file__).resolve().parent.parent / "figures"
out_dir.mkdir(exist_ok=True)


# -----------------------------------------------------------------------------
# 核心算法：径向环形平均
# -----------------------------------------------------------------------------
def radial_annular_average(intensity, sampling, wavelength, nbins=200, bin_width=None):
    """对二维强度做径向（环形）平均。

    intensity : (Ny, Nx) 2D 强度数组——原始强度或 log 强度均可，由调用方决定；
       本函数只做分环平均，不改变其量纲。
    sampling  : (sy, sx) 倒空间采样 [1/A]（即 diffraction pattern 的 sampling）。
    wavelength: 电子波长 [A]。
    nbins     : 预留（保留参数，仅当 bin_width 未给定且无法由采样推算时使用）。
    bin_width : 径向 bin 的角宽 [mrad]。默认取像素角分辨率(1000*wavelength*sampling[0])，
       避免 bin 比数据分辨率还细——这正是之前曲线毛刺/断续的原因。

    算法分两步：
      1) 按"每个出现的不同半径"分组取平均——保证每个环有像素、无空环；
      2) 把上一步结果重bin到均匀角网格（宽 bin_width），空 bin 用邻近有值 bin 线性插值——
         得到平滑、连续、分辨合理的径向剖面。
    返回 (theta_mrad, profile)：环中心角度 [mrad] 与各环平均强度。
    """
    intensity = np.asarray(intensity, dtype=float)
    Ny, Nx = intensity.shape
    # 以数组中心为 000 的倒空间坐标 (ky, kx) [1/Å]
    ky = (np.arange(Ny) - (Ny - 1) / 2) * sampling[0]
    kx = (np.arange(Nx) - (Nx - 1) / 2) * sampling[1]
    K = np.sqrt(kx[None, :] ** 2 + ky[:, None] ** 2)      # |k| [1/Å]
    theta = (1000.0 * wavelength * K).ravel()             # 散射角 [mrad]

    I = intensity.ravel()
    # 第一步：按不同半径分组（四舍五入到足够精度，避免浮点误差拆分同一环）
    key = np.round(theta, 3)
    uniq = np.unique(key)
    idx = np.searchsorted(uniq, key)
    cnt = np.bincount(idx, minlength=uniq.size)
    tot = np.bincount(idx, weights=I, minlength=uniq.size)
    prof_uniq = tot / np.maximum(cnt, 1)

    # 第二步：重bin到均匀角网格
    if bin_width is None:
        bin_width = 1000.0 * wavelength * float(sampling[0])   # 像素角分辨率 [mrad]
    edges = np.arange(uniq.min(), uniq.max() + bin_width, bin_width)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bidx = np.clip(np.digitize(uniq, edges) - 1, 0, centers.size - 1)
    bcnt = np.bincount(bidx, minlength=centers.size)
    btot = np.bincount(bidx, weights=prof_uniq, minlength=centers.size)
    profile = btot / np.maximum(bcnt, 1)

    # 空 bin 用邻近有值 bin 线性插值填充，保证曲线连续
    empty = bcnt == 0
    if empty.any() and (not empty).any():
        profile[empty] = np.interp(centers[empty], centers[~empty], profile[~empty])

    return centers, profile


def apply_log(I, C=1.5e5):
    """I' = ln[1 + C * |I| / I_max]，与 cbed.py 的 log 变换一致。"""
    imax = max(float(np.max(np.abs(I))), 1e-30)
    return np.log(1.0 + C * np.abs(I) / imax)


def to_numpy(x):
    """dask/cupy -> numpy（并保留数组数据）"""
    if hasattr(x, "compute"):
        x = x.compute()
    if hasattr(x, "get"):
        x = x.get()
    return np.asarray(x)


# -----------------------------------------------------------------------------
# 计算 CBED（与 cbed.py 相同的物理设置）
# -----------------------------------------------------------------------------
srtio3 = ase.io.read(cif_path).repeat((8, 8, 50))
frozen_phonons = abtem.FrozenPhonons(srtio3, 1, {"Sr": 0.078, "Ti": 0.078, "O": 0.078})
potential = abtem.Potential(
    frozen_phonons, parametrization="lobato", sampling=0.05,
    slice_thickness=1, projection="finite", device="gpu",
)
probe = abtem.Probe(energy=100e3, semiangle_cutoff=9.4, device="gpu")
probe.grid.match(potential)
cell_x, cell_y, _ = srtio3.cell.lengths()

exit_wave = probe.multislice(potential, scan=[[cell_x / 2, cell_y / 2]], lazy=False)
cbed = exit_wave.diffraction_patterns(max_angle="cutoff")
cbed_mean = to_numpy(cbed.array).mean(axis=0)          # 声子热平均 -> (Ny, Nx) 原始强度
sampling = np.asarray(cbed.sampling)                   # [1/Å]
wavelength = probe.accelerator.wavelength              # [Å]

# -----------------------------------------------------------------------------
# 径向环形平均：原始强度 与 log 强度
# -----------------------------------------------------------------------------
cbed_log = apply_log(cbed_mean)

theta, prof_raw = radial_annular_average(cbed_mean, sampling, wavelength)
theta, prof_log = radial_annular_average(cbed_log, sampling, wavelength)

# -----------------------------------------------------------------------------
# 绘图：横轴 mrad，纵轴分别为原始 / log 强度
# -----------------------------------------------------------------------------
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.5))

ax0.plot(theta, prof_raw)
ax0.set_xlabel("scattering angle [mrad]")
ax0.set_ylabel("intensity")
ax0.set_ylim(0, prof_raw[np.isfinite(prof_raw)].max() * 1.05)
ax0.set_title("(a) Radial average (raw intensity)")

ax1.plot(theta, prof_log)
ax1.set_xlabel("scattering angle [mrad]")
ax1.set_ylabel("log intensity")
ax1.set_title(r"(b) Radial average (log, C=1.5e5)")

fig.tight_layout()
fig.savefig(out_dir / "radial_average.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"saved: {out_dir / 'radial_average.png'}")

# -----------------------------------------------------------------------------
# 汇总信息
# -----------------------------------------------------------------------------
print(f"CBED grid: {cbed_mean.shape}, reciprocal sampling: {sampling} 1/A, wavelength: {wavelength:.5f} A")
for prof, label in ((prof_raw, "raw"), (prof_log, "log")):
    v = np.isfinite(prof)
    if v.any():
        i = int(np.nanargmax(prof))
        print(f"{label}: range {theta[v].min():.2f}..{theta[v].max():.2f} mrad; peak at {theta[i]:.2f} mrad, value {prof[i]:.4g}")
print("done: CBED radial annular average (raw + log)")
