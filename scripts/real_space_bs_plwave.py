# =============================================================================
# 平面波电子衍射版本 —— STO(30keV) real-space + 背散射
#   参照官方 real_space 教程，但光源改用 PlaneWave（平行/平面波入射）。
#   平面波入射会得到【衍射斑点】(SAED 类) 而非会聚束的 CBED 盘。
#   参数与 real_space_bs.py 一致，仅把 converged_probe 换成 plane_wave。
#
#   注意：低能(30keV) + 细片厚(0.5A) + order=3/full 可能触发 DivergedError，
#         故捕获异常并给出降阶建议。
# =============================================================================
import abtem
import ase
import numpy as np
from pathlib import Path

from abtem.multislice import FourierMultislice, RealSpaceMultislice
from abtem.finite_difference import DivergedError

abtem.config.set({"diagnostics.progress_bar": False})

OUT_DIR = Path(__file__).resolve().parent.parent / "figures"
OUT_PNG = OUT_DIR / "real_space_plwave_fwd_bwd.png"

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

potential_exit_planes = abtem.Potential(
    atoms,
    gpts=(6 * 40, 6 * 40),
    slice_thickness=0.5,
    exit_planes=1,
    projection="finite",
)


def exit_plane_axis(waves):
    for i, ax in enumerate(waves.ensemble_axes_metadata):
        if getattr(ax, "label", None) == "z":
            return i
    raise RuntimeError("找不到 ThicknessAxis(label='z') 出射面轴")


def take_exit_plane(waves, which="exit"):
    ax = exit_plane_axis(waves)
    idx = (waves.shape[ax] - 1) if which == "exit" else 0
    items = [0] * len(waves.shape[:-2])
    items[ax] = idx
    return waves[tuple(items)]


# 平面波：与会聚束 CBED 版(real_space_bs.py)一致用 30keV。
# 实测 30keV 平面波 order=3/full 收敛（无发散）。之前误用 200keV 是多余保守。
ENERGY = 30e3
plane_wave = abtem.PlaneWave(energy=ENERGY)
plane_wave.grid.match(potential_exit_planes)

print(f"computing planewave ({ENERGY/1e3:.0f} keV) forward/backscattered "
      f"(order=3, full, return_backscattered=True) ...", flush=True)
try:
    (
        forward_exit_waves_realspace_fc,
        backward_exit_waves_realspace_fc,
    ) = plane_wave.multislice(
        potential=potential_exit_planes,
        algorithm=RealSpaceMultislice(order=3, expansion_scope="full"),
        return_backscattered=True,
    )
except DivergedError:
    print("DivergedError: 30/200kV + thin slice real-space expansion diverged.\n"
          "  -> 建议改用较小 order (如 order=1) 或增大 slice_thickness。", flush=True)
    raise

fwd_exit = take_exit_plane(forward_exit_waves_realspace_fc, "exit")
bwd_entr = take_exit_plane(backward_exit_waves_realspace_fc, "entrance")
print(f"  forward_exit shape={fwd_exit.shape}  backward_entrance shape={bwd_entr.shape}", flush=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def to_np(arr):
    return arr.get() if hasattr(arr, "get") else np.asarray(arr)


fdp = to_np(fwd_exit.diffraction_patterns(max_angle="cutoff").array)
fdp = fdp[0] if fdp.ndim == 3 else fdp
bdp = to_np(bwd_entr.diffraction_patterns(max_angle="cutoff").array)
bdp = bdp[0] if bdp.ndim == 3 else bdp
np.savez(OUT_DIR / "real_space_plwave_fwd_bwd.npz", fdp=fdp.astype(np.float32),
         bdp=bdp.astype(np.float32))
print(f"  saved data npz (fdp={fdp.shape}, bdp={bdp.shape})", flush=True)


def apply_log(I, C=1.5e5):
    imax = max(float(np.max(np.abs(I))), 1e-30)
    return np.log(1.0 + C * np.abs(I) / imax)


fig, axes = plt.subplots(1, 2, figsize=(13, 6))
im = axes[0].imshow(apply_log(np.abs(fdp)), origin="lower", cmap="inferno")
axes[0].set_title("Forward-scattered wave (log)\nat the exit surface", fontsize=11)
axes[0].set_xlabel("kx [1/A]"); axes[0].set_ylabel("ky [1/A]")
fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label="log intensity")
im = axes[1].imshow(apply_log(np.abs(bdp)), origin="lower", cmap="inferno")
axes[1].set_title("Coherent backscattered wave (log)\nat the entrance surface", fontsize=11)
axes[1].set_xlabel("kx [1/A]"); axes[1].set_ylabel("ky [1/A]")
fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="log intensity")
fig.suptitle(f"PLANE WAVE, SrTiO3 {ENERGY/1e3:.0f}keV, order=3/full — "
             "backward@entrance vs forward@exit", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.94))
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
plt.close(fig)
print("saved", OUT_PNG, flush=True)
