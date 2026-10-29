# =============================================================================
# 计算 STEM 电子探针（Probe），把「强度 | 相位 | 剖线」合成一张一行三列的图，
# 以 300 dpi 保存为 probe.png（不另存 npz / 衍射花样）。
# 网格通过 probe.grid.match(potential) 与势场一致；device="gpu" 用 cupy 计算。
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

# -----------------------------------------------------------------------------
# 1) 超胞 + 势场构建器（仅用于让探针网格与势场一致，不做势场计算）
# -----------------------------------------------------------------------------
srtio3 = ase.io.read(cif_path).repeat((8, 8, 50))
potential = abtem.Potential(srtio3, sampling=0.05, slice_thickness=1)

# -----------------------------------------------------------------------------
# 2) 探针：100 keV，聚光镜截止半角 9.4 mrad，GPU 计算
# -----------------------------------------------------------------------------
probe = abtem.Probe(energy=100e3, semiangle_cutoff=9.4, device="gpu")
probe.grid.match(potential)  # 把探针网格匹配到势场网格

# 在超胞中心建一个探针波函数（scan 为单点）
cell_x, cell_y, _ = srtio3.cell.lengths()
waves = probe.build(scan=[[cell_x / 2, cell_y / 2]], lazy=False)


def host_copy(x):
    """GPU(cupy) -> CPU numpy 拷贝；CPU(numpy) 则拷贝断开视图，供 matplotlib 使用。"""
    if hasattr(x, "get"):
        x = x.get()
    return np.asarray(x).copy()


# -----------------------------------------------------------------------------
# 3) 取数据：强度(2D)、相位(2D)、剖线(1D)
# -----------------------------------------------------------------------------
intensity = host_copy(waves.intensity().array)   # |psi|^2，归一化
phase = np.angle(host_copy(waves.array))          # 复波函数的相位 [rad]
extent_real = (0, cell_x, 0, cell_y)              # 实空间坐标范围 [Å]

profile = probe.profiles(angle=0.0)
prof_arr = host_copy(profile.array.compute())     # profiles() 默认 lazy，先物化
sampling = float(profile.sampling)
r = np.arange(prof_arr.size) * sampling           # 沿剖线距离 [Å]

# -----------------------------------------------------------------------------
# 4) 一行三列图：强度 | 相位 | 剖线，300 dpi 保存
# -----------------------------------------------------------------------------
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(12, 4))

im0 = ax0.imshow(intensity, extent=extent_real, origin="lower", cmap="inferno")
ax0.set_title("(a) Intensity")
ax0.set_xlabel("x [Å]")
ax0.set_ylabel("y [Å]")
fig.colorbar(im0, ax=ax0, label="intensity")

im1 = ax1.imshow(phase, extent=extent_real, origin="lower", cmap="twilight")
ax1.set_title("(b) Phase")
ax1.set_xlabel("x [Å]")
ax1.set_ylabel("y [Å]")
fig.colorbar(im1, ax=ax1, label="phase [rad]")

ax2.plot(r, prof_arr)
ax2.set_title("(c) Profile")
ax2.set_xlabel("r [Å]")
ax2.set_ylabel("intensity")

fig.tight_layout()
fig.savefig(out_dir / "probe.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"saved: {out_dir / 'probe.png'}")

# -----------------------------------------------------------------------------
# 5) 汇总信息
# -----------------------------------------------------------------------------
print(f"grid gpts: {probe.grid.gpts}, extent: ({cell_x:.2f}, {cell_y:.2f}) A")
print("done: probe intensity / phase / profile saved as one 1x3 figure")
