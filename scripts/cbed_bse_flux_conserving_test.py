# =============================================================================
# 验证 conserve_flux=True 修复（5 keV, 8x8x50）：
#   对照三组：full（原始，非守恒）/ full+conserve_flux（修复）/ Fourier（参考）
#   判据：修复后 sum_with ≤ 1 所有厚度；净增益 ≤ 物理上界；花样形态保持。
# 输出：paper/figures/convergence/conv_flux_conserving_5keV.npz
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


potential = abtem.Potential(phonons, parametrization="lobato", sampling=SAMPLING,
                            slice_thickness=SLICE, projection="finite",
                            exit_planes=None, device="gpu")
ns = potential.num_slices
del potential
cp.get_default_memory_pool().free_all_blocks()

idx = []
for t in THICKNESS_A:
    i = min(max(int(round(t / SLICE)), 0), ns - 1)
    if not idx or i > idx[-1]:
        idx.append(i)
if idx[-1] != ns - 1:
    idx[-1] = ns - 1
    if len(idx) > 1 and idx[-1] == idx[-2]:
        idx.pop()
ds = tuple(idx)
thick = np.asarray([(i + 1) * SLICE for i in ds])

potential = abtem.Potential(phonons, parametrization="lobato", sampling=SAMPLING,
                            slice_thickness=SLICE, projection="finite",
                            exit_planes=ds, device="gpu")
probe = abtem.Probe(energy=ENERGY_KEV * 1e3, semiangle_cutoff=9.4, device="gpu")
probe.grid.match(potential)
cx, cy, _ = atoms.cell.lengths()
print(f"gpts={probe.grid.gpts} num_slices={ns} depth={ds}", flush=True)

results = {}
for tag, algo in [
    ("full_nocons", RealSpaceMultislice(order=1, expansion_scope="full")),
    ("full_cons", RealSpaceMultislice(order=1, expansion_scope="full", conserve_flux=True)),
    ("fourier", FourierMultislice()),
]:
    t0 = time.perf_counter()
    ew = probe.multislice(potential, scan=[[cx / 2, cy / 2]],
                          algorithm=algo, return_backscattered=False, lazy=False)
    dt = time.perf_counter() - t0
    cb = extract_exit(ew.diffraction_patterns(max_angle="cutoff").array)
    s = np.abs(cb).reshape(cb.shape[0], -1).sum(axis=1)
    results[tag] = (s, cb, dt)
    print(f"[{tag}] {dt:.1f}s  sums={np.round(s, 5)}", flush=True)

s_nc, _, _ = results["full_nocons"]
s_c, cb_c, _ = results["full_cons"]
s_f, cb_f, _ = results["fourier"]

print("\nthickness | Fourier | full(原) | full(守恒修复) | 修复后net% | 上界%", flush=True)
for k in range(len(thick)):
    bound = (1 - s_f[k]) / s_f[k] * 100
    net_c = (s_c[k] - s_f[k]) / s_f[k] * 100
    ok = "OK" if s_c[k] <= 1.0 else "VIOLATION"
    print(f"{thick[k]:7.2f} A | {s_f[k]:.5f} | {s_nc[k]:.5f} | {s_c[k]:.5f} | "
          f"{net_c:+.2f}% | {bound:.1f}% | {ok}", flush=True)

# 花样形态保持：修复版与原版的 CBED 差异
diff = np.abs(cb_c - extract_exit(results["full_nocons"][1]))
rel = np.abs(diff).sum() / np.abs(cb_f).sum() * 100
print(f"\n修复版 vs 原版 CBED 总差异（L1, 相对 Fourier 总量）: {rel:.2f}%", flush=True)

np.savez(OUT / "conv_flux_conserving_5keV.npz",
         thickness_angstrom=thick, sum_fourier=s_f, sum_full_nocons=s_nc,
         sum_full_conserving=s_c, cbed_conserving=cb_c.astype(np.float32),
         cbed_nocons=results["full_nocons"][1].astype(np.float32),
         cbed_fourier=cb_f.astype(np.float32))
print(f"saved: {OUT / 'conv_flux_conserving_5keV.npz'}", flush=True)
