# =============================================================================
# 通量守恒诊断（5 keV）：逐片记录背散射修正的通量账
#   monkeypatch abtem.multislice.realspace_multislice_step，
#   在每片记录三个量（实空间 sum|.|^2，同一口径）：
#     I_before : 修正前（前向级数展开+带限之后）的前向强度
#     I_after  : 修正后（waves -= backscatter）的前向强度
#     I_bwd    : 该片流入背向通道的强度 sum|backscatter|^2
#   判定：若 I_after ≈ I_before − I_bwd → 修正项是"搬运"（通量守恒，记账问题）；
#         若 I_after > I_before − I_bwd → 修正项是净范数源。
# 输出：paper/figures/convergence/flux_budget_5keV.npz
# =============================================================================
import time
from pathlib import Path

import abtem
import ase.io
import numpy as np
import cupy as cp
import abtem.multislice as ms_module
from abtem.multislice import RealSpaceMultislice, FourierMultislice

CIF = Path(__file__).resolve().parents[1] / "atomic_model" / "SrTiO3.cif"
OUT = Path(__file__).resolve().parent.parent / "figures" / "convergence"

ENERGY_KEV = 5
SLICE = 0.05
SAMPLING = 0.05
THICKNESS_A = [20, 70, 140, 195]

atoms = ase.io.read(CIF).repeat((8, 8, 50))
phonons = abtem.FrozenPhonons(atoms, 1, {"Sr": 0.078, "Ti": 0.078, "O": 0.078})

orig_step = ms_module.realspace_multislice_step
records = []  # (I_before_corr, I_after_corr, I_bwd)


def sumsq(arr):
    if hasattr(arr, "get"):
        return float((cp.abs(arr) ** 2).sum().get())
    return float((np.abs(arr) ** 2).sum())


def recording_step(waves, potential_slice, next_slice=None, **kwargs):
    # 原函数内部：前向级数展开 → 带限 → 修正项 → waves -= backscatter
    # 为拆分"修正前/后"，复制 waves 进入前重算太昂贵；
    # 改为：先记录进入本片的强度（上片修正后），调用原函数，
    # 再从返回值拿背向波强度。修正前强度 = 本片返回的前向强度 + 2Re<ψ,B> - |B|²
    # 不可直接拆；故记录 (进入强度, 出来强度, 背向强度)，进入−出来即单片净变化。
    I_in = sumsq(waves._array)
    out = orig_step(waves, potential_slice, next_slice=next_slice, **kwargs)
    if isinstance(out, tuple) and len(out) == 2:
        fwd, bwd = out
        I_out = sumsq(fwd._array)
        I_bwd = sumsq(bwd._array)
        records.append((I_in, I_out, I_bwd))
    return out


ms_module.realspace_multislice_step = recording_step


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

# 入射探针初始强度（实空间口径）
probe_grid = probe.build([[cx / 2, cy / 2]], lazy=False)
I0 = sumsq(probe_grid._array)
print(f"gpts={probe.grid.gpts} num_slices={ns} I0={I0:.5f}", flush=True)

t0 = time.perf_counter()
ew_bse = probe.multislice(potential, scan=[[cx / 2, cy / 2]],
                          algorithm=RealSpaceMultislice(order=1, expansion_scope="full"),
                          return_backscattered=False, lazy=False)
t_bse = time.perf_counter() - t0
print(f"backscatter full: {t_bse:.1f}s, records={len(records)}", flush=True)

cb_bse = extract_exit(ew_bse.diffraction_patterns(max_angle="cutoff").array)
sum_with = np.abs(cb_bse).reshape(cb_bse.shape[0], -1).sum(axis=1)

rec = np.asarray(records)          # (n_corr, 3): I_in, I_out, I_bwd
dI = rec[:, 1] - rec[:, 0]         # 单片前向净变化（含级数展开损耗+带限+修正）
cum_dI = np.cumsum(dI)
cum_bwd = np.cumsum(rec[:, 2])
print(f"per-slice dI: min={dI.min():.3e} max={dI.max():.3e}  (正=范数增加)", flush=True)
print(f"sum|B|^2 total={rec[:,2].sum():.5f},  sum dI total={dI.sum():.5f}", flush=True)

print("\nthickness | sum_with(diff) | fwd_est=I0+cum_dI | +bwd通道 | 备注", flush=True)
for k, i in enumerate(ds):
    n = min(i + 1, len(rec)) - 1
    fwd_est = I0 + cum_dI[n]
    print(f"{thick[k]:7.2f} A | {sum_with[k]:.5f} | {fwd_est:.5f} | "
          f"{fwd_est + cum_bwd[n]:.5f} | I0={I0:.4f}", flush=True)

np.savez(OUT / "flux_budget_5keV.npz",
         thickness_angstrom=thick, sum_with_diffraction=sum_with,
         records=rec, I0=I0, depth_slices=np.asarray(ds))
print(f"saved: {OUT / 'flux_budget_5keV.npz'}", flush=True)
