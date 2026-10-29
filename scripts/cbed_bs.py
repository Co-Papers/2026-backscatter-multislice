# =============================================================================
# 背散射波 (backscattered wave) 计算 + 作图 —— 一体脚本
#
#   【物理模型（重点，不要搞错）】
#     * return_backscattered=True 返回 (forward, backward)。
#     * backward = 背散射波：multislice_step 在【每一个片层传播时】都会算出一个
#       背散射分量（finite_difference.py 的 fully_corrected + next_slice 分支），
#       这些分量全部经 _back_propagate_backscattered_waves 反向传播、在
#       【上表面（入口面）】叠加成最终背散射波。
#     * 背散射波 ONLY 在入射表面（上表面）有一个物理结果 -> backward[0]（入口面），
#       它是所有片层背散射分量之和，非零。
#
#   【exit_planes 的作用（重要，别和背散射混）】
#     * exit_planes 只是"记录哪些厚度的【前向出射波 forward】"的开关：
#       在哪些深度保存 forward 波。它【不影响】背散射波的物理计算。
#     * 本脚本同时把 forward 在 exit_planes 各厚度的 CBED 也画出来（展示前向随深度演化），
#       背散射则只画上表面 backward[0] 一张。
#     * 注意：abTEM 的 return_backscattered=True 要求 num_exit_planes>1（实现限制，
#       multislice.py），所以 exit_planes 需给 ≥2 个值作为 forward 的记录厚度。
#
#   配置：5 keV / 无冷冻声子(FrozenPhonons? 这里用静态结构 srtio3) / 小体系(REPEAT)。
#   用法：python cbed_bs.py
# =============================================================================
import abtem
import ase.io
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from abtem.multislice import RealSpaceMultislice

REPEAT = (2, 2, 8)           # 小体系：快、少占 GPU
ENERGY_KEV = 5
SLICE_THICKNESS = 0.05
MAX_ANGLE = "cutoff"         # 不裁切，全分辨率衍射盘
SEMIANGLE = 9.4              # mrad

# 记录 forward 的厚度（仅用于记录前向出射波，与背散射无关）。
# 只要 ≥2 个即可（满足 return_backscattered 的实现限制）。
EXIT_PLANES_FORWARD = (100, 200, 400, 626)   # 对应厚度 5.04/10.03/20.02/31.30 Å

OUT_PNG = (
    Path(__file__).resolve().parents[1] / "figures" / "cbed_bs_forward_backward.png"
)

cif_path = Path(__file__).resolve().parents[1] / "atomic_model" / "SrTiO3.cif"
srtio3 = ase.io.read(cif_path).repeat(REPEAT)


# ---------------------------------------------------------------------------
# 语义化取面（按 ThicknessAxis(label=='z') 定位出射面轴）
# ---------------------------------------------------------------------------
def exit_plane_axis(waves):
    for i, ax in enumerate(waves.ensemble_axes_metadata):
        if getattr(ax, "label", None) == "z":
            return i
    raise RuntimeError("找不到 ThicknessAxis(label='z') 出射面轴")


def take_exit_plane(waves, which="exit"):
    """语义化取单一出射面：'exit'=最后一个(出口面)，'entrance'=第一个(入口面/上表面)。"""
    ax = exit_plane_axis(waves)
    idx = (waves.shape[ax] - 1) if which == "exit" else 0
    items = [0] * len(waves.shape[:-2])  # 前面 ensemble 轴取第 0 个 config
    items[ax] = idx
    return waves[tuple(items)]


def take_all_exit_planes(waves):
    """取全部出射面（用于展示 forward 随厚度演化），保留出射面轴为第一维。"""
    ax = exit_plane_axis(waves)
    items = [0] * len(waves.shape[:-2])
    items[ax] = slice(None)
    return waves[tuple(items)]


def to_np(arr):
    return arr.get() if hasattr(arr, "get") else np.asarray(arr)


def apply_log(I, C=1.5e5):
    imax = max(float(np.max(np.abs(I))), 1e-30)
    return np.log(1.0 + C * np.abs(I) / imax)


# ---------------------------------------------------------------------------
# 作图：forward 各记录厚度（前 n 列） + 入口面背散射波（最后一列）
# ---------------------------------------------------------------------------
def plot_forward_backward(fwd_all, bwd_entr, zlist, label):
    zlist = np.asarray(zlist, dtype=float)
    n_exit = fwd_all.shape[0]
    fig, axes = plt.subplots(1, n_exit + 1, figsize=(3.8 * (n_exit + 1), 4.2))
    if n_exit == 1:
        axes = axes[None, :]

    # forward 各记录厚度
    for k in range(n_exit):
        fk = fwd_all[k]
        fdp = to_np(fk.diffraction_patterns(max_angle=MAX_ANGLE).array)
        fdp = fdp[0] if fdp.ndim == 3 else fdp
        im = axes[k].imshow(apply_log(np.abs(fdp)), origin="lower", cmap="inferno")
        axes[k].set_title(f"forward CBED\nz={zlist[k]:.1f} A", fontsize=10)
        axes[k].set_xlabel("kx"); axes[k].set_ylabel("ky")
        fig.colorbar(im, ax=axes[k], fraction=0.046, pad=0.04, label="log I")

    # 背散射：只画上表面（入口面）
    bdp = to_np(bwd_entr.diffraction_patterns(max_angle=MAX_ANGLE).array)
    bdp = bdp[0] if bdp.ndim == 3 else bdp
    im = axes[n_exit].imshow(apply_log(np.abs(bdp)), origin="lower", cmap="inferno")
    axes[n_exit].set_title("coherent backscattered wave\nat entrance surface (top)", fontsize=10)
    axes[n_exit].set_xlabel("kx"); axes[n_exit].set_ylabel("ky")
    fig.colorbar(im, ax=axes[n_exit], fraction=0.046, pad=0.04, label="log I")

    fig.suptitle(f"SrTiO3 CBED 5keV (RealSpace full): forward (recorded thicknesses) vs "
                 f"backscattered (entrance surface)  [{label}]", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PLOT] saved {OUT_PNG}", flush=True)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def try_run():
    label = f"exit_planes={EXIT_PLANES_FORWARD}"
    print(f"\n===== {label} =====", flush=True)
    potential = abtem.Potential(
        srtio3,
        parametrization="lobato",
        sampling=0.05,
        slice_thickness=SLICE_THICKNESS,
        projection="finite",
        exit_planes=EXIT_PLANES_FORWARD,
        device="gpu",
    )
    num_slices = potential.num_slices
    probe = abtem.Probe(energy=ENERGY_KEV * 1e3, semiangle_cutoff=SEMIANGLE, device="gpu")
    probe.grid.match(potential)
    cell_x, cell_y, _ = srtio3.cell.lengths()
    print(f"  num_slices={num_slices}", flush=True)

    t0 = time.perf_counter()
    try:
        out = probe.multislice(
            potential,
            scan=[[cell_x / 2, cell_y / 2]],
            algorithm=RealSpaceMultislice(expansion_scope="full"),
            return_backscattered=True,
            lazy=False,
        )
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
        return

    print(f"  OK in {time.perf_counter()-t0:.1f}s", flush=True)
    fwd, bwd = out[0], out[1]
    zlist = potential.exit_thicknesses

    # 背散射：只取上表面（入口面=第一个出射面）。这是所有片层背散射反向叠加的最终结果。
    bwd_entr = take_exit_plane(bwd, "entrance")
    # forward：取记录的各厚度（展示前向随深度演化）
    fwd_all = take_all_exit_planes(fwd)
    print(f"  exit_axis={exit_plane_axis(bwd)}  fwd_all.shape={fwd_all.shape}  "
          f"bwd_entrance.shape={bwd_entr.shape}  zlist={np.round(zlist,2)}", flush=True)

    # 作图
    plot_forward_backward(fwd_all, bwd_entr, zlist, label)

    # 定量：背散射(入口面) / 前向(出口面) 总强度比
    fwd_exit = take_exit_plane(fwd, "exit")
    fdp = to_np(fwd_exit.diffraction_patterns(max_angle=MAX_ANGLE).array)
    fdp = fdp[0] if fdp.ndim == 3 else fdp
    bdp = to_np(bwd_entr.diffraction_patterns(max_angle=MAX_ANGLE).array)
    bdp = bdp[0] if bdp.ndim == 3 else bdp
    ratio = float(np.sum(np.abs(bdp) ** 2)) / max(float(np.sum(np.abs(fdp) ** 2)), 1e-30)
    print(f"  [ratio] backward(entrance)/forward(exit) total intensity = {ratio:.3e}", flush=True)


if __name__ == "__main__":
    try_run()
    print("\ndone", flush=True)
