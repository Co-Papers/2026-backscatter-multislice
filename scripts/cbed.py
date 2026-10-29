# =============================================================================
# 计算 CBED（会聚束电子衍射）：用会聚探针穿过 SrTiO3 超胞的势场，
# multislice 后在倒空间收集衍射花样，并对冷冻声子系综取平均（热平均），
# 300 dpi 保存为 cbed.png。
# 网格用 probe.grid.match(potential) 与势场一致；device="gpu" 用 cupy 计算。
# =============================================================================
import abtem
import ase.io
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 路径均相对脚本自身解析，与运行时的当前工作目录无关
cif_path = Path(__file__).resolve().parents[1] / "atomic_model" / "SrTiO3.cif"
out_dir = Path(__file__).resolve().parent.parent / "figures"
out_dir.mkdir(exist_ok=True)


def to_numpy(x):
    """把 dask/cupy/numpy 数组统一转成 numpy 数组。"""
    if hasattr(x, "compute"):  # dask
        x = x.compute()
    if hasattr(x, "get"):      # cupy
        x = x.get()
    return np.asarray(x)


def apply_log(I, C=1.5e5):
    imax = max(float(np.max(np.abs(I))), 1e-30)
    return np.log(1.0 + C * np.abs(I) / imax)


# -----------------------------------------------------------------------------
# 1) 原子模型：SrTiO3 超胞，z 方向 50 个晶胞厚（与 potential.py/probe.py 一致）
# -----------------------------------------------------------------------------
srtio3 = ase.io.read(cif_path).repeat((8, 8, 50))

# -----------------------------------------------------------------------------
# 2) 冷冻声子 + 势场构建器（惰性，multislice 时会逐配置/逐切片流式用到）
# -----------------------------------------------------------------------------
frozen_phonons = abtem.FrozenPhonons(srtio3, 16, {"Sr": 0.078, "Ti": 0.078, "O": 0.078})
potential = abtem.Potential(
    frozen_phonons,
    parametrization="lobato",
    sampling=0.05,        # x/y 网格采样 [Å]
    slice_thickness=1,    # 沿 z 的切片厚度 [Å]
    projection="finite",
    device="gpu",
)

# -----------------------------------------------------------------------------
# 3) 探针：100 keV，聚光镜截止半角 9.4 mrad，网格匹配势场（同 probe.py）
# -----------------------------------------------------------------------------
probe = abtem.Probe(energy=300e3, semiangle_cutoff=21.4, defocus=10.0, device="gpu")
probe.grid.match(potential)

cell_x, cell_y, _ = srtio3.cell.lengths()

# -----------------------------------------------------------------------------
# 4) multislice：探针穿过势场，得到出射波（含声子系综维）
# -----------------------------------------------------------------------------
exit_wave = probe.multislice(potential, scan=[[cell_x / 2, cell_y / 2]], lazy=False)

# CBED = 出射波衍射花样（倒空间），含系综维，先做热平均
# max_angle=30 [mrad]：截到孔径附近的倒空间范围，避免整块网格把盘压到中心看不清
cbed = exit_wave.diffraction_patterns(max_angle='cutoff')
cbed_arr = to_numpy(cbed.array).mean(axis=0)  # 对冷冻声子配置取平均 -> (Ny, Nx)
dx, dy = cbed.extent  # 倒空间为对称的 ±extent

# -----------------------------------------------------------------------------
# 5) 保存 CBED 图（线性 + log 变换并排，300 dpi）
#    线性强度被中心盘占主导，log 变换 I' = ln[1 + C*I/I_max] 凸出弱盘与细节
# -----------------------------------------------------------------------------
cbed_log = apply_log(cbed_arr)

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 5))

im0 = ax0.imshow(cbed_arr, extent=(-dx, dx, -dy, dy), origin="lower", cmap="inferno")
ax0.set_title("(a) CBED (linear)")
ax0.set_xlabel("kx [1/Å]")
ax0.set_ylabel("ky [1/Å]")
fig.colorbar(im0, ax=ax0, label="intensity")

im1 = ax1.imshow(cbed_log, extent=(-dx, dx, -dy, dy), origin="lower", cmap="inferno")
ax1.set_title(r"(b) log[1+C*I/I_max], C=1.5e5")
ax1.set_xlabel("kx [1/Å]")
ax1.set_ylabel("ky [1/Å]")
fig.colorbar(im1, ax=ax1, label="log intensity")

fig.savefig(out_dir / "cbed.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"saved: {out_dir / 'cbed.png'}")

# -----------------------------------------------------------------------------
# 6) 汇总信息
# -----------------------------------------------------------------------------
print("max angles:", cbed.max_angles, "mrad")
print(f"grid gpts: {probe.grid.gpts}, slices: {potential.num_slices}, configs: {potential.num_configurations}")
print(f"CBED grid: {cbed_arr.shape}, reciprocal extent: ({dx:.3f}, {dy:.3f}) 1/A")
print("done: CBED computed (thermal average over phonon configs)")
