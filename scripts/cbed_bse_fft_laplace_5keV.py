# =============================================================================
# FFT Laplacian 对照实验（5 keV）：检验幺正性违反是否来自 FD 模板
#   与基准 conv_slice_0p05 唯一差异：laplace_method="fft"（谱精确拉普拉斯）
#   若 FFT 下 sum_with 仍 > 1 → 违反来自背散射修正项本身（算法固有）
#   若 FFT 下 sum_with ≤ 1 → 违反来自 FD 数值色散
# 输出：paper/figures/convergence/conv_fft_laplace_5keV.npz
# =============================================================================
import time
from pathlib import Path

import abtem
import ase.io
import numpy as np
import cupy as cp
from abtem.multislice import RealSpaceMultislice, FourierMultislice

CIF = Path(__file__).resolve().parents[1] / "atomic_model" / "SrTiO3.cif"
OUT = Path(__file__).resolve().parent.parent / "figures" / "convergence"
OUT.mkdir(parents=True, exist_ok=True)

ENERGY_KEV = 5
SLICE = 0.05
SAMPLING = 0.05
THICKNESS_A = [20, 70, 140, 195]

atoms = ase.io.read(CIF).repeat((8, 8, 50))
phonons = abtem.FrozenPhonons(atoms, 1, {"Sr": 0.078, "Ti": 0.078, "O": 0.078})


def to_numpy(x):
    if hasattr(x, "compute"):
        x = x.compute()
    if hasattr(x, "get"):
        x = x.get()
    return np.asarray(x)


def extract_exit(arr):
    arr = to_numpy(arr)
    return arr[0] if arr.ndim == 4 else arr


def depth_slice_indices(num_slices):
    idx = []
    for t in THICKNESS_A:
        i = min(max(int(round(t / SLICE)), 0), num_slices - 1)
        if not idx or i > idx[-1]:
            idx.append(i)
    if idx[-1] != num_slices - 1:
        idx[-1] = num_slices - 1
        if len(idx) > 1 and idx[-1] == idx[-2]:
            idx.pop()
    return tuple(idx)


potential = abtem.Potential(phonons, parametrization="lobato", sampling=SAMPLING,
                            slice_thickness=SLICE, projection="finite",
                            exit_planes=None, device="gpu")
ns = potential.num_slices
ds = depth_slice_indices(ns)
thick = np.asarray([(i + 1) * SLICE for i in ds])
del potential
cp.get_default_memory_pool().free_all_blocks()

potential = abtem.Potential(phonons, parametrization="lobato", sampling=SAMPLING,
                            slice_thickness=SLICE, projection="finite",
                            exit_planes=ds, device="gpu")
probe = abtem.Probe(energy=ENERGY_KEV * 1e3, semiangle_cutoff=9.4, device="gpu")
probe.grid.match(potential)
cx, cy, _ = atoms.cell.lengths()
print(f"gpts={probe.grid.gpts} num_slices={ns} depth={ds}", flush=True)

t0 = time.perf_counter()
ew_bse = probe.multislice(potential, scan=[[cx / 2, cy / 2]],
                          algorithm=RealSpaceMultislice(order=1, expansion_scope="full",
                                                        laplace_method="fft"),
                          return_backscattered=False, lazy=False)
t_bse = time.perf_counter() - t0
print(f"backscatter (fft laplace): {t_bse:.1f}s", flush=True)

t0 = time.perf_counter()
ew_ref = probe.multislice(potential, scan=[[cx / 2, cy / 2]],
                          algorithm=FourierMultislice(), lazy=False)
t_ref = time.perf_counter() - t0

cb_bse = extract_exit(ew_bse.diffraction_patterns(max_angle="cutoff").array)
cb_ref = extract_exit(ew_ref.diffraction_patterns(max_angle="cutoff").array)

sum_w = np.abs(cb_bse).reshape(cb_bse.shape[0], -1).sum(axis=1)
sum_wo = np.abs(cb_ref).reshape(cb_ref.shape[0], -1).sum(axis=1)
net = (sum_w - sum_wo) / sum_wo * 100

for i in range(len(thick)):
    print(f"t={thick[i]:7.2f} A: without={sum_wo[i]:.5f}  with={sum_w[i]:.5f}  "
          f"bound={(1-sum_wo[i])/sum_wo[i]*100:.1f}%  net={net[i]:+.2f}%  "
          f"unitarity_ok={sum_w[i] <= 1.0}", flush=True)

np.savez(OUT / "conv_fft_laplace_5keV.npz",
         thickness_angstrom=thick, cbed_with=cb_bse.astype(np.float32),
         cbed_without=cb_ref.astype(np.float32),
         sum_with=sum_w, sum_without=sum_wo, net_percent=net,
         t_bse=t_bse, t_ref=t_ref)
print(f"saved: {OUT / 'conv_fft_laplace_5keV.npz'}", flush=True)
