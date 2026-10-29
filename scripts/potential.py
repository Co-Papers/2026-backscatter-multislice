# =============================================================================
# 计算 SrTiO3 超胞的静电势场，并导出两张 GIF：
#   1) 不同声子态的势场（同一中间切片对比）  ->  potential_phonon_states.gif
#   2) 第一个声子态沿深度方向的不同切片       ->  potential_slices_state0.gif
#
# 内存策略：用 potential.generate_blocks() 逐配置（逐声子态）惰性计算，
# 每个配置算完后立刻释放，峰值内存只占单个配置的切片（约几百 MB），
# 避免一次性撑起整块 (32 x num_slices x 627 x 627) 数组（约 10 GB）。
# device="gpu"：使用 cupy 在 GPU 上计算（需已安装 cupy），取帧时用 host_copy() 搬回 CPU。
# =============================================================================
import abtem
import ase.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

# 路径均相对脚本自身解析，与运行时的当前工作目录无关
cif_path = Path(__file__).resolve().parents[1] / "atomic_model" / "SrTiO3.cif"
out_dir = Path(__file__).resolve().parent.parent / "figures"
out_dir.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# 1) 原子模型：SrTiO3 超胞，z 方向 50 个晶胞厚
# -----------------------------------------------------------------------------
srtio3 = ase.io.read(cif_path)
srtio3 = srtio3.repeat((8, 8, 50))

# -----------------------------------------------------------------------------
# 2) 冷冻声子：32 个随机位移配置，各元素均方根位移 0.078 Å
# -----------------------------------------------------------------------------
frozen_phonons = abtem.FrozenPhonons(srtio3, 32, {"Sr": 0.078, "Ti": 0.078, "O": 0.078})

# -----------------------------------------------------------------------------
# 3) 势场构建器（此时尚未计算任何数据）
# -----------------------------------------------------------------------------
potential = abtem.Potential(
    frozen_phonons,
    parametrization="lobato",
    sampling=0.05,        # x/y 网格采样 [Å]
    slice_thickness=1,    # 沿 z 的切片厚度 [Å]
    projection="finite",  # 有限投影：在切片边界间数值积分
    exit_planes=60,       # 每 60 个切片记录一个出射面（multislice 厚度序列用）
    device="gpu",         # 用 GPU 计算（需已安装 cupy）
)

num_configs = potential.num_configurations
num_slices = potential.num_slices
extent = potential.extent  # (x, y) 尺寸 [Å]，用于 imshow 坐标轴
mid_slice = num_slices // 2

# -----------------------------------------------------------------------------
# 4) 惰性逐配置计算：每次只算一个声子配置的全部切片，算完即释放
# -----------------------------------------------------------------------------
phonon_frames = []  # GIF1 帧：每个声子态的中间切片
slice_frames = []   # GIF2 帧：第一个声子态（config 0）的各切片

def host_copy(x):
    """把切片转成独立的 numpy 数组，供 matplotlib 使用。

    - GPU(cupy) 数组：.get() 搬到 CPU，返回新 numpy 数组；
    - CPU(numpy) 数组：.copy() 断开视图引用。
    目的是切断帧对底层整块数组的引用，del arr 后才能真正释放内存。
    """
    if hasattr(x, "get"):
        x = x.get()
    return np.asarray(x).copy()


for cfg_idx, (indices, slics, chunk) in enumerate(potential.generate_blocks()):
    # chunk 是 object 数组，chunk[0] 为当前配置的 Potential，计算后形状 (1, num_slices, H, W)
    arr = chunk[0].build().compute().array

    # GIF1：取该声子态的中间切片作为代表（转 numpy 副本）
    phonon_frames.append(host_copy(arr[0, mid_slice]))

    # GIF2：只取第一个声子态的所有切片（转 numpy 副本）
    if cfg_idx == 0:
        slice_frames = [host_copy(arr[0, s]) for s in range(num_slices)]

    del arr  # 立即释放当前配置的整块数组，控制峰值内存
    print(f"processed config {cfg_idx + 1}/{num_configs}  (slice_dim={num_slices})")

# -----------------------------------------------------------------------------
# 5) 写 GIF：所有帧共用同一 vmin/vmax，保证不同帧之间色标可比
# -----------------------------------------------------------------------------
def save_gif(planes, fname, title, cmap="inferno", fps=5):
    """把一组二维势场切片保存成 GIF。"""
    vmin = min(float(np.min(p)) for p in planes)
    vmax = max(float(np.max(p)) for p in planes)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(
        planes[0],
        extent=(0, extent[0], 0, extent[1]),
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    fig.colorbar(im, ax=ax, label="potential [V]")
    ax.set_xlabel("x [Å]")
    ax.set_ylabel("y [Å]")
    ax.set_title(title)

    def update(i):
        im.set_data(planes[i])
        ax.set_title(f"{title} — {i + 1}/{len(planes)}")
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(planes), interval=1000 / fps, blit=False)
    ani.save(fname, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"saved: {fname} ({len(planes)} frames)")


save_gif(phonon_frames, str(out_dir / "potential_phonon_states.gif"), "Different phonon states")
save_gif(slice_frames, str(out_dir / "potential_slices_state0.gif"), "Slices of phonon state 0")

print(f"done: num_configurations={num_configs}, num_slices={num_slices}")
