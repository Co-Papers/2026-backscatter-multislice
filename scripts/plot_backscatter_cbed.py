# =============================================================================
# CBED：出口面前向波 vs 入口面背散射波（倒空间衍射盘）
#   参照 abTEM 官方 real_space 教程，但用更稳健的"按 ThicknessAxis(label=='z')
#   语义化定位出射面轴"取面，替代官方直接 forward[-1]/backward[0]（后者在
#   多个冷冻声子态下会取错轴）。生成一行两列 CBED 对比图。
# =============================================================================
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import abtem, ase.io, time
from pathlib import Path
from abtem.multislice import RealSpaceMultislice

REPEAT = (4, 4, 16); ENERGY_KEV = 5; NUM_CONFIGS = 1
SLICE_THICKNESS = 0.05; SEMIANGLE = 9.4
MAX_ANGLE = "cutoff"
cif_path = Path(__file__).resolve().parents[1] / "atomic_model" / "SrTiO3.cif"
srtio3 = ase.io.read(cif_path).repeat(REPEAT)
frozen_phonons = abtem.FrozenPhonons(srtio3, NUM_CONFIGS, {"Sr": 0.078, "Ti": 0.078, "O": 0.078})

potential = abtem.Potential(frozen_phonons, parametrization="lobato", sampling=0.05,
                            slice_thickness=SLICE_THICKNESS, projection="finite",
                            exit_planes=(100, 200, 400, 626), device="gpu")
probe = abtem.Probe(energy=ENERGY_KEV * 1e3, semiangle_cutoff=SEMIANGLE, device="gpu")
probe.grid.match(potential)
cell_x, cell_y, _ = srtio3.cell.lengths()


def exit_plane_axis(waves):
    """按元数据定位出射面轴（ThicknessAxis label=='z'）。不依赖硬编码轴位置。"""
    for i, ax in enumerate(waves.ensemble_axes_metadata):
        if getattr(ax, "label", None) == "z":
            return i
    raise RuntimeError("找不到 ThicknessAxis(label='z') 出射面轴")


def take_exit_plane(waves, which="exit"):
    """语义化取 'exit'（出口面/最后一个出射面）或 'entrance'（入口面/第一个出射面）。"""
    ax = exit_plane_axis(waves)
    n = waves.shape[ax]
    idx = n - 1 if which == "exit" else 0
    items = [0] * len(waves.shape[:-2])  # 前面 ensemble 轴取第 0 个 config
    items[ax] = idx
    return waves[tuple(items)]


print("computing return_backscattered=True (order=3, full) ...", flush=True)
t0 = time.perf_counter()
out = probe.multislice(potential, scan=[[cell_x/2, cell_y/2]],
                       algorithm=RealSpaceMultislice(order=3, expansion_scope="full"),
                       return_backscattered=True, lazy=False)
print(f"  done {time.perf_counter()-t0:.1f}s", flush=True)

fwd, bwd = out[0], out[1]
fwd_exit = take_exit_plane(fwd, "exit")       # 出口面前向波
bwd_entr = take_exit_plane(bwd, "entrance")   # 入口面背散射波
print(f"fwd.shape={fwd.shape}  bwd.shape={bwd.shape}  "
      f"exit_axis={exit_plane_axis(bwd)}", flush=True)

# ---- CBED 衍射（倒空间）----
fwd_dp = fwd_exit.diffraction_patterns(max_angle=MAX_ANGLE)
bwd_dp = bwd_entr.diffraction_patterns(max_angle=MAX_ANGLE)
f = fwd_dp.array.get() if hasattr(fwd_dp.array, 'get') else np.asarray(fwd_dp.array)
b = bwd_dp.array.get() if hasattr(bwd_dp.array, 'get') else np.asarray(bwd_dp.array)
# 归一化到 2D（若还有前导 size-1 轴）
f = f[0] if f.ndim == 3 else f
b = b[0] if b.ndim == 3 else b
print(f"fwd_dp={f.shape}  bwd_dp={b.shape}", flush=True)


def apply_log(I, C=1.5e5):
    imax = max(float(np.max(np.abs(I))), 1e-30)
    return np.log(1.0 + C * np.abs(I) / imax)


# ---- 图：一行两列（左出口前向 CBED，右入口背散射 CBED）----
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
im = axes[0].imshow(apply_log(f), origin="lower", cmap="inferno")
axes[0].set_title("Forward-scattered wave\nat the exit surface", fontsize=11)
axes[0].set_xlabel("kx [1/A]"); axes[0].set_ylabel("ky [1/A]")
fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label="log intensity")
im = axes[1].imshow(apply_log(b), origin="lower", cmap="inferno")
axes[1].set_title("Coherent backscattered wave\nat the entrance surface", fontsize=11)
axes[1].set_xlabel("kx [1/A]"); axes[1].set_ylabel("ky [1/A]")
fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="log intensity")
fig.suptitle(f"SrTiO3 CBED 5keV (order=3, RealSpace full): exit forward vs entrance backscattered  "
             f"(scan ({cell_x/2:.2f},{cell_y/2:.2f}) A, semiangle {SEMIANGLE} mrad)", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.94))
out_png = r"F:\papers\backscatter\paper\figures\cbed_fwd_vs_backscatter.png"
fig.savefig(out_png, dpi=200, bbox_inches="tight")
print("saved", out_png, flush=True)

# ---- 定量指标 ----
ratio = float(np.sum(np.abs(b)**2)) / max(float(np.sum(np.abs(f)**2)), 1e-30)
print(f"backward/forward total intensity ratio = {ratio:.3e}", flush=True)
