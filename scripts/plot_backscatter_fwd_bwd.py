# =============================================================================
# 可视化 forward（前散射）vs backward（背散射）波
#   return_backscattered=True -> (forward, backward)，形状 (1, n_exit, ny, nx)
#   对 4 个出射面绘制实空间强度 |psi|^2（对数尺度，因背散射极弱）
# =============================================================================
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import abtem, ase.io, time
from pathlib import Path
from abtem.multislice import RealSpaceMultislice

REPEAT = (8, 8, 50); ENERGY_KEV = 5; NUM_CONFIGS = 1
SLICE_THICKNESS = 0.05; SEMIANGLE = 9.4
cif_path = Path(__file__).resolve().parents[1] / "atomic_model" / "SrTiO3.cif"
srtio3 = ase.io.read(cif_path).repeat(REPEAT)
frozen_phonons = abtem.FrozenPhonons(srtio3, NUM_CONFIGS, {"Sr": 0.078, "Ti": 0.078, "O": 0.078})

potential = abtem.Potential(frozen_phonons, parametrization="lobato", sampling=0.05,
                            slice_thickness=SLICE_THICKNESS, projection="finite",
                            exit_planes=(100, 200, 400, 626), device="gpu")
probe = abtem.Probe(energy=ENERGY_KEV * 1e3, semiangle_cutoff=SEMIANGLE, device="gpu")
probe.grid.match(potential)
cell_x, cell_y, _ = srtio3.cell.lengths()

print("computing return_backscattered=True ...", flush=True)
t0 = time.perf_counter()
out = probe.multislice(potential, scan=[[cell_x/2, cell_y/2]],
                       algorithm=RealSpaceMultislice(expansion_scope="full"),
                       return_backscattered=True, lazy=False)
print(f"  done {time.perf_counter()-t0:.1f}s", flush=True)

fwd = out[0]; bwd = out[1]
f = fwd.array.get() if hasattr(fwd.array, 'get') else np.asarray(fwd.array)
b = bwd.array.get() if hasattr(bwd.array, 'get') else np.asarray(bwd.array)
print("fwd shape", f.shape, "bwd shape", b.shape, flush=True)

n_exit = f.shape[1]
zlist = potential.exit_thicknesses
extent = float(potential.extent[0]) if not np.isscalar(potential.extent) else float(potential.extent)

def logscale(I, C=1e3):
    imax = max(float(np.max(np.abs(I))), 1e-30)
    return np.log1p(C * np.abs(I) / imax)

# ---- 图：n_exit 行 x 2 列（左 forward，右 backward）----
fig, axes = plt.subplots(n_exit, 2, figsize=(11, 3.4 * n_exit))
if n_exit == 1:
    axes = axes[None, :]
for k in range(n_exit):
    Fk = np.abs(f[0, k])**2
    Bk = np.abs(b[0, k])**2
    im = axes[k, 0].imshow(logscale(Fk), extent=(0, extent, 0, extent), origin="lower", cmap="inferno")
    axes[k, 0].set_title(f"forward @ z={zlist[k]:.1f} A", fontsize=10)
    axes[k, 0].set_ylabel("y [A]")
    fig.colorbar(im, ax=axes[k, 0], fraction=0.046, pad=0.04, label="log|psi|^2")
    # backward: 独立归一化（它太弱，不能与 forward 同标尺）
    im = axes[k, 1].imshow(logscale(Bk), extent=(0, extent, 0, extent), origin="lower", cmap="inferno")
    axes[k, 1].set_title(f"backward @ z={zlist[k]:.1f} A", fontsize=10)
    axes[k, 1].set_xlabel("x [A]"); axes[k, 1].set_ylabel("y [A]")
    fig.colorbar(im, ax=axes[k, 1], fraction=0.046, pad=0.04, label="log|psi|^2")

fig.suptitle(f"SrTiO3 CBED 5keV: forward vs backscattered wave  "
             f"(scan ({cell_x/2:.2f},{cell_y/2:.2f}) A, semiangle {SEMIANGLE} mrad)", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.96))
out_png = r"F:\papers\backscatter\paper\figures\backscatter_fwd_bwd.png"
fig.savefig(out_png, dpi=200, bbox_inches="tight")
print("saved", out_png, flush=True)

# ---- 保存数据 ----
np.savez(r"F:\papers\backscatter\paper\figures\backscatter_fwd_bwd.npz",
         fwd=f.astype(np.float32), bwd=b.astype(np.float32),
         zlist=np.array(zlist), extent=extent)
print("saved npz", flush=True)
