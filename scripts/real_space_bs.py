# =============================================================================
# 背散射波 + 前散射波 计算脚本 —— 完全对齐 abTEM 官方 real_space 教程
#   https://abtem.github.io/doc/user_guide/tutorials/real_space.html
#   （章节 "Coherent backscattered wave"）
#
#   【官方参数（本脚本逐项对齐）】
#     * 样品    : unit_cell (SrTiO3, a=3.9127 A) * (6,6,24)
#     * Potential: gpts=(6*40,6*40), slice_thickness=0.5, projection='finite', exit_planes=1
#     * 光源    : converged_probe = Probe(semiangle_cutoff=20, energy=30e3).match_grid(potential)
#     * 算法    : RealSpaceMultislice(order=3, expansion_scope='full')
#     * 调用    : probe.multislice(potential=..., scan=[[0,0]], algorithm=..., return_backscattered=True)
#     * 取面    : forward[-1] (出口面) + backward[0] (入口面)，abtem.stack 后 .diffraction_patterns()
#
#   【与官方唯一的差别：语义化取面】
#     官方向量 forward[-1]/backward[0]（标量索引，作用于第一个 ensemble 轴）。
#     在多冷冻声子态(>1)下会取错轴，故这里改用按 ThicknessAxis(label=='z') 语义化定位
#     的 take_exit_plane，结果与官方等价但对任意 config 数稳健。
#     官方该例是单声子态，[-1]/[0] 恰好正确；本脚本做了稳健替代。
#
#   输出：figures/real_space_bs_fwd_bwd.png（左=出口面前向波，右=入口面背散射波）
#   用法：python real_space_bs.py
# =============================================================================
import abtem
import ase
import numpy as np
from pathlib import Path

from abtem.multislice import FourierMultislice, RealSpaceMultislice

# 官方把 device 设为 cpu；这里保留 GPU 加速（结果一致），如需严格对齐可改 abtem.config.set({'device':'cpu'})
# abtem.config.set({"device": "cpu"})
abtem.config.set({"diagnostics.progress_bar": False})

OUT_DIR = Path(__file__).resolve().parent.parent / "figures"
OUT_PNG = OUT_DIR / "real_space_bs_fwd_bwd.png"


# ---------------------------------------------------------------------------
# 1) 模型样品：SrTiO3 unit_cell * (6,6,24)   （与官方逐字对齐）
# ---------------------------------------------------------------------------
unit_cell = ase.Atoms(
    symbols="SrTiO3",
    scaled_positions=[
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5],
    ],
    cell=[3.9127, 3.9127, 3.9127],
    pbc=True,
)

atoms = unit_cell * (6, 6, 24)


# ---------------------------------------------------------------------------
# 2) Potential（official: gpts=(6*40,6*40), slice_thickness=0.5, projection='finite',
#    exit_planes=1）
# ---------------------------------------------------------------------------
potential_exit_planes = abtem.Potential(
    atoms,
    gpts=(6 * 40, 6 * 40),
    slice_thickness=0.5,
    exit_planes=1,          # 官方：每 1 片一个出射面（记录前向波深度演化）
    projection="finite",
)


# ---------------------------------------------------------------------------
# 3) focused probe（official: semiangle_cutoff=20, energy=30e3, match_grid）
# ---------------------------------------------------------------------------
converged_probe = abtem.Probe(
    semiangle_cutoff=20,
    energy=30e3,
).match_grid(potential_exit_planes)


# ---------------------------------------------------------------------------
# 4) 语义化取面（替代官方 forward[-1]/backward[0]，对多声子态稳健）
# ---------------------------------------------------------------------------
def exit_plane_axis(waves):
    for i, ax in enumerate(waves.ensemble_axes_metadata):
        if getattr(ax, "label", None) == "z":
            return i
    raise RuntimeError("找不到 ThicknessAxis(label='z') 出射面轴")


def take_exit_plane(waves, which="exit"):
    ax = exit_plane_axis(waves)
    idx = (waves.shape[ax] - 1) if which == "exit" else 0
    items = [0] * len(waves.shape[:-2])   # 前面 ensemble 轴取第 0 个 config
    items[ax] = idx
    return waves[tuple(items)]


# ---------------------------------------------------------------------------
# 5) 计算背散射（official: RealSpaceMultislice(order=3, expansion_scope='full'),
#    return_backscattered=True)
# ---------------------------------------------------------------------------
print("computing forward/backscattered waves (order=3, full, return_backscattered=True) ...", flush=True)
(
    forward_exit_waves_realspace_fc,
    backward_exit_waves_realspace_fc,
) = converged_probe.multislice(
    potential=potential_exit_planes,
    scan=[[0, 0]],
    algorithm=RealSpaceMultislice(order=3, expansion_scope="full"),
    return_backscattered=True,
)

# 官方：forward[-1] (出口面) + backward[0] (入口面)
fwd_exit = take_exit_plane(forward_exit_waves_realspace_fc, "exit")
bwd_entr = take_exit_plane(backward_exit_waves_realspace_fc, "entrance")
print(f"  forward_exit shape={fwd_exit.shape}  backward_entrance shape={bwd_entr.shape}", flush=True)


# ---------------------------------------------------------------------------
# 6) 作图（official: abtem.stack(...) .diffraction_patterns().show(explode=True)）
#    这里用 matplotlib 直观显示两幅 CBED，等价于 stack 的"出口前向 vs 入口背散射"。
# ---------------------------------------------------------------------------
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def to_np(arr):
    return arr.get() if hasattr(arr, "get") else np.asarray(arr)


def apply_log(I, C=1.5e5):
    imax = max(float(np.max(np.abs(I))), 1e-30)
    return np.log(1.0 + C * np.abs(I) / imax)


fdp = to_np(fwd_exit.diffraction_patterns(max_angle="cutoff").array)
fdp = fdp[0] if fdp.ndim == 3 else fdp
bdp = to_np(bwd_entr.diffraction_patterns(max_angle="cutoff").array)
bdp = bdp[0] if bdp.ndim == 3 else bdp

# 保存原始强度数据（便于复用，免重跑 multislice）
np.savez(OUT_DIR / "real_space_bs_fwd_bwd.npz", fdp=fdp.astype(np.float32),
         bdp=bdp.astype(np.float32))
print(f"  saved data npz (fdp={fdp.shape}, bdp={bdp.shape})", flush=True)

# 线性强度图：两幅均用原始强度 |psi|^2，各自独立线性归一化（背散射极弱需独立标尺）
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
im = axes[0].imshow(np.abs(fdp), origin="lower", cmap="inferno")
axes[0].set_title("Forward-scattered wave\nat the exit surface (linear)", fontsize=11)
axes[0].set_xlabel("kx [1/A]"); axes[0].set_ylabel("ky [1/A]")
fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label="|psi|^2 (linear)")
im = axes[1].imshow(np.abs(bdp), origin="lower", cmap="inferno")
axes[1].set_title("Coherent backscattered wave\nat the entrance surface (linear)", fontsize=11)
axes[1].set_xlabel("kx [1/A]"); axes[1].set_ylabel("ky [1/A]")
fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="|psi|^2 (linear)")
fig.suptitle("Official real-space tutorial: SrTiO3, 30keV, order=3/full — "
             "backward@entrance vs forward@exit (LINEAR intensity)", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.94))
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
plt.close(fig)
print("saved", OUT_PNG, flush=True)
