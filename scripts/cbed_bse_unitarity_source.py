# =============================================================================
# 幺正性来源定位对照（5 keV, 8x8x50）：
#   A) slice=0.2 A, expansion_scope="full"      —— 补全切片厚度标度链
#   B) slice=0.05 A, expansion_scope="propagator" —— 关闭背散射修正项，
#      隔离前向级数展开本身的幺正性（决定性对照）
# 输出：paper/figures/convergence/conv_unit_src_{slice02,prop_only}.npz
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


def run(tag, slice_thickness, scope):
    potential = abtem.Potential(phonons, parametrization="lobato", sampling=SAMPLING,
                                slice_thickness=slice_thickness, projection="finite",
                                exit_planes=None, device="gpu")
    ns = potential.num_slices
    idx = []
    for t in THICKNESS_A:
        i = min(max(int(round(t / slice_thickness)), 0), ns - 1)
        if not idx or i > idx[-1]:
            idx.append(i)
    if idx[-1] != ns - 1:
        idx[-1] = ns - 1
        if len(idx) > 1 and idx[-1] == idx[-2]:
            idx.pop()
    ds = tuple(idx)
    thick = np.asarray([(i + 1) * slice_thickness for i in ds])
    del potential
    cp.get_default_memory_pool().free_all_blocks()

    potential = abtem.Potential(phonons, parametrization="lobato", sampling=SAMPLING,
                                slice_thickness=slice_thickness, projection="finite",
                                exit_planes=ds, device="gpu")
    probe = abtem.Probe(energy=ENERGY_KEV * 1e3, semiangle_cutoff=9.4, device="gpu")
    probe.grid.match(potential)
    cx, cy, _ = atoms.cell.lengths()
    print(f"[{tag}] scope={scope} slice={slice_thickness} num_slices={ns} depth={ds}", flush=True)

    t0 = time.perf_counter()
    ew = probe.multislice(potential, scan=[[cx / 2, cy / 2]],
                          algorithm=RealSpaceMultislice(order=1, expansion_scope=scope),
                          return_backscattered=False, lazy=False)
    t_rs = time.perf_counter() - t0

    t0 = time.perf_counter()
    ew_ref = probe.multislice(potential, scan=[[cx / 2, cy / 2]],
                              algorithm=FourierMultislice(), lazy=False)
    t_ref = time.perf_counter() - t0

    cb = extract_exit(ew.diffraction_patterns(max_angle="cutoff").array)
    cb_ref = extract_exit(ew_ref.diffraction_patterns(max_angle="cutoff").array)

    s = np.abs(cb).reshape(cb.shape[0], -1).sum(axis=1)
    s_ref = np.abs(cb_ref).reshape(cb_ref.shape[0], -1).sum(axis=1)

    print(f"[{tag}] time rs={t_rs:.1f}s ref={t_ref:.1f}s", flush=True)
    for i in range(len(thick)):
        print(f"[{tag}] t={thick[i]:7.2f} A: RS={s[i]:.5f}  Fourier={s_ref[i]:.5f}  "
              f"RS>1? {s[i] > 1.0}", flush=True)

    np.savez(OUT / f"conv_unit_src_{tag}.npz",
             thickness_angstrom=thick, sum_rs=s, sum_fourier=s_ref,
             slice_thickness=slice_thickness, scope=scope)

    del potential, probe
    cp.get_default_memory_pool().free_all_blocks()


for tag, st, scope in [("slice02", 0.2, "full"), ("prop_only", 0.05, "propagator")]:
    try:
        run(tag, st, scope)
    except Exception as e:
        print(f"[{tag}] FAILED: {type(e).__name__}: {e}", flush=True)
        cp.get_default_memory_pool().free_all_blocks()
