# =============================================================================
# 背散射 + 前散射波计算（SrTiO3, 8x8x50, 5keV, 16 frozen phonons, focused probe）
#   ——— 监控内存 & 发散 ——
#
#   对应 cbed_bs.py 的"背散射波 vs 前散射波"取面约定：
#     * backward 只取入口面（上表面）[take_exit_plane(bwd,"entrance")]
#     * forward   取出口面           [take_exit_plane(fwd,"exit")]
#
#   用法：
#     python probe_bs_fwd_bwd.py --smoke      # 小规模冒烟：2x2x8 + 2 声子态，验证不发散/参数正确
#     python probe_bs_fwd_bwd.py              # 全量：8x8x50 + 16 声子态 + 5keV
#     python probe_bs_fwd_bwd.py --configs 16 --inplane 8 --zrepeat 50   # 显式全量参数
# =============================================================================
import argparse
import time
import traceback
from pathlib import Path

import numpy as np
import ase.io
import abtem
from abtem.multislice import RealSpaceMultislice
from abtem.finite_difference import DivergedError, NotConvergedError
import psutil

abtem.config.set({"diagnostics.progress_bar": True})

# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------
CIF = Path(__file__).resolve().parent.parent / "atomic_model" / "SrTiO3.cif"
OUT_DIR = Path(__file__).resolve().parent.parent / "figures_multi_phonons_16"
ENERGY_KEV = 5
SAMPLING = 0.05            # Å
SLICE_THICKNESS = 0.05     # Å
SEMIANGLE = 9.4            # mrad
NUM_CONFIGS = 16
INPLANE = 8
ZREPEAT = 50
DEBYE_WALLER = {"Sr": 0.078, "Ti": 0.078, "O": 0.078}
EXIT_PLANES = 1            # 每 1 片记录一个出射面（记录前向深度演化；≥2 出射面满足背散射前提）


def mem_report(tag):
    rss_mb = psutil.Process().memory_info().rss / 1024**2
    gpu = ""
    try:
        import cupy as cp
        free, total = cp.cuda.runtime.memGetInfo()
        gpu = f"  GPU free={free/1024**3:.2f}G / {total/1024**3:.2f}G"
    except Exception:
        pass
    print(f"[mem] {tag}:  RSS~{rss_mb:.0f}MB{gpu}", flush=True)


def build(atoms_num_configs, inplane, zrepeat, device):
    a = 3.91270131
    srtio3 = ase.io.read(CIF).repeat((inplane, inplane, zrepeat))
    frozen = abtem.FrozenPhonons(srtio3, atoms_num_configs, DEBYE_WALLER)
    extent = inplane * a
    sampling = SAMPLING
    potential = abtem.Potential(
        frozen, parametrization="lobato", sampling=sampling,
        slice_thickness=SLICE_THICKNESS, projection="finite",
        exit_planes=EXIT_PLANES, device=device,
    )
    probe = abtem.Probe(energy=ENERGY_KEV * 1e3, semiangle_cutoff=SEMIANGLE, device=device)
    probe.grid.match(potential)
    return potential, probe, frozen, extent


def exit_plane_axis(waves):
    for i, ax in enumerate(waves.ensemble_axes_metadata):
        if getattr(ax, "label", None) == "z":
            return i
    raise RuntimeError("exhausted: no z axis")


def take_exit_plane(waves, which="exit"):
    ax = exit_plane_axis(waves)
    idx = (waves.shape[ax] - 1) if which == "exit" else 0
    items = [0] * len(waves.shape[:-2])
    items[ax] = idx
    return waves[tuple(items)]


def to_np(arr):
    return arr.get() if hasattr(arr, "get") else np.asarray(arr)


def apply_log(I, C=1.5e5):
    imax = max(float(np.max(np.abs(I))), 1e-30)
    return np.log(1.0 + C * np.abs(I) / imax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="小规模冒烟（2x2x8 + 2 声子态）")
    ap.add_argument("--configs", type=int, default=NUM_CONFIGS)
    ap.add_argument("--inplane", type=int, default=INPLANE)
    ap.add_argument("--zrepeat", type=int, default=ZREPEAT)
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--order", type=int, default=1,
                    help="RealSpace 展开阶数。5keV+大网格(627^2)下 order=3 会 DivergedError，"
                         "用 order=1/2 稳定（已验证）。")
    args = ap.parse_args()

    cfg = args.configs
    inpl = args.inplane
    zrep = args.zrepeat
    if args.smoke:
        cfg, inpl, zrep = 2, 2, 8

    device = args.device
    tag = f"cbed_bs_fwd_bwd_{ENERGY_KEV}keV_{inpl}x{inpl}x{zrep}_phonons{cfg}"
    print(f"=== {tag} ===  5keV, sampling={SAMPLING}, slice={SLICE_THICKNESS}, "
          f"configs={cfg}, inplane={inpl}, zrepeat={zrep}, device={device}", flush=True)

    # 建 potential（含16声子态，构建就耗内存）
    mem_report("before potential build")
    potential, probe, frozen, extent = build(cfg, inpl, zrep, device)
    mem_report(f"after potential build  num_slices={potential.num_slices}  "
               f"exit_planes={potential.exit_planes}  grid={potential.gpts}")
    print(f"  extent={extent:.2f}  num_slices={potential.num_slices}  "
          f"num_exit_planes={potential.num_exit_planes}", flush=True)

    # 计算前向+背散射
    t0 = time.perf_counter()
    print("  computing forward/backscattered (return_backscattered=True) ...", flush=True)
    mem_report("before multislice")
    try:
        fwd, bwd = probe.multislice(
            potential=potential, scan=[[extent / 2, extent / 2]],
            algorithm=RealSpaceMultislice(order=args.order, expansion_scope="full"),
            return_backscattered=True, lazy=False,
        )
    except (DivergedError, NotConvergedError) as e:
        print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
        print("  -> 建议：降低 order(3->1) 或 增大 slice_thickness，或改用 FourierMultislice", flush=True)
        traceback.print_exc()
        return
    dt = time.perf_counter() - t0
    mem_report(f"after multislice  ({dt:.1f}s)")
    print(f"  done in {dt:.1f}s  fwd.shape={fwd.shape}  bwd.shape={bwd.shape}", flush=True)

    # 取面
    fwd_exit = take_exit_plane(fwd, "exit")
    bwd_entr = take_exit_plane(bwd, "entrance")
    print(f"  fwd_exit={fwd_exit.shape}  bwd_entrance={bwd_entr.shape}", flush=True)

    # 衍射（倒空间）
    fdp = to_np(fwd_exit.diffraction_patterns(max_angle="cutoff").array)
    bdp = to_np(bwd_entr.diffraction_patterns(max_angle="cutoff").array)
    fdp = fdp[0] if fdp.ndim == 3 else fdp
    bdp = bdp[0] if bdp.ndim == 3 else bdp
    out_npz = OUT_DIR / f"{tag}.npz"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, fdp=fdp.astype(np.float32), bdp=bdp.astype(np.float32),
             extent=extent, sampling=SAMPLING)
    print(f"  saved {out_npz}", flush=True)

    # 作图（log）
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_png = OUT_DIR / f"{tag}.png"
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    im = axes[0].imshow(apply_log(np.abs(fdp)), origin="lower", cmap="inferno")
    axes[0].set_title("Forward-scattered wave (log)\nat exit surface", fontsize=11)
    axes[0].set_xlabel("kx [1/A]"); axes[0].set_ylabel("ky [1/A]")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label="log intensity")
    im = axes[1].imshow(apply_log(np.abs(bdp)), origin="lower", cmap="inferno")
    axes[1].set_title("Coherent backscattered wave (log)\nat entrance surface", fontsize=11)
    axes[1].set_xlabel("kx [1/A]"); axes[1].set_ylabel("ky [1/A]")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="log intensity")
    fig.suptitle(f"SrTiO3 CBED {ENERGY_KEV}keV (order=3/full): fwd@exit vs bwd@entrance  "
                 f"({inpl}x{inpl}x{zrep}, {cfg} phonons)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_png}", flush=True)

    # 强度比（背散射相对于前向，量化）
    ratio = float(np.sum(np.abs(bdp) ** 2)) / max(float(np.sum(np.abs(fdp) ** 2)), 1e-30)
    print(f"  [ratio] bwd/fwd total intensity = {ratio:.3e}", flush=True)


if __name__ == "__main__":
    main()
