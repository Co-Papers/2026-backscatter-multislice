# =============================================================================
# 收敛性测试：验证 5 keV 下背散射净增益/非单调特征是物理效应而非数值伪影
#
# 基准配置（与 cbed_bse_energy_sweep.py 一致）：
#   5 keV, SrTiO3, sampling=0.05 Å, slice_thickness=0.05 Å,
#   RealSpaceMultislice(order=1, expansion_scope="full",
#                       derivative_accuracy=6, max_terms=80)
#
# 扫描轴（每轴独立扫描，其余保持基准值；命名：<轴>_<值>）：
#   slice                : slice_thickness = [0.1, 0.05, 0.025] Å
#   sampling             : sampling        = [0.06, 0.05, 0.04] Å
#   fdacc (derivative_accuracy) : [2, 4, 6]      （FD 模板阶数）
#   terms (max_terms)    : [40, 80, 160]         （指数级数截断）
#
# 每组配置输出（npz 落盘到 paper/figures/convergence/）：
#   cbed_without / cbed_with : (n_exit, ny, nx) 原始强度（无光阑外截断，max_angle="cutoff"）
#   thickness_angstrom       : (n_exit,)
#   net_percent              : (n_exit,) 净强度相对变化 (%)
#   计时与收敛标志
#
# 用法：
#   python cbed_bse_convergence.py --smoke      # 2x2x8 小晶胞冒烟
#   python cbed_bse_convergence.py              # 8x8x50 全尺寸
#   python cbed_bse_convergence.py --only slice,sampling   # 只跑指定轴
# =============================================================================
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import abtem
import ase.io
import numpy as np
import psutil
import cupy as cp
from abtem.multislice import RealSpaceMultislice, FourierMultislice

CIF_PATH = Path(__file__).resolve().parents[1] / "atomic_model" / "SrTiO3.cif"
OUT_DIR = Path(__file__).resolve().parent.parent / "figures" / "convergence"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ENERGY_KEV = 5
SEMIANGLE = 9.4
MAX_ANGLE = "cutoff"
NUM_CONFIGS = 1
DEBYE_WALLER = {"Sr": 0.078, "Ti": 0.078, "O": 0.078}
THICKNESS_A = [20, 70, 140, 195]          # 4 个厚度点看趋势足够

# 基准参数
BASE = dict(sampling=0.05, slice_thickness=0.05, derivative_accuracy=6, max_terms=80)

# 扫描轴定义：axis 名为输出文件名前缀，元组第一个元素必须是 BASE 中的真实键名
AXES = {
    "slice": [("slice_thickness", v) for v in (0.1, 0.05, 0.025)],
    "sampling": [("sampling", v) for v in (0.06, 0.05, 0.04)],
    "fdacc": [("derivative_accuracy", v) for v in (2, 4, 6)],
    "terms": [("max_terms", v) for v in (40, 80, 160)],
}


def to_numpy(x):
    if hasattr(x, "compute"):
        x = x.compute()
    if hasattr(x, "get"):
        x = x.get()
    return np.asarray(x)


def mem_report(tag):
    rss = psutil.Process().memory_info().rss / 1e6
    free, total = cp.cuda.runtime.memGetInfo()
    print(f"[mem] {tag}: RSS={rss:.0f}MB, GPU used={(total-free)/1e6:.0f}MiB / total={total/1e6:.0f}MiB",
          flush=True)


def free_gpu():
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def extract_exit(arr):
    arr = to_numpy(arr)
    if arr.ndim == 4:
        return arr[0]
    return arr


def depth_slice_indices(num_slices, slice_thickness):
    idx = []
    for t in THICKNESS_A:
        i = int(round(t / slice_thickness))
        i = min(max(i, 0), num_slices - 1)
        if not idx or i > idx[-1]:
            idx.append(i)
    if idx[-1] != num_slices - 1:
        idx[-1] = num_slices - 1
        if len(idx) > 1 and idx[-1] == idx[-2]:
            idx.pop()
    return tuple(idx)


def run_config(name, params, frozen_phonons, atoms, repeat):
    """跑一组配置：背散射(RS full) + 参考(Fourier)，返回结果 dict。"""
    sampling = params["sampling"]
    st = params["slice_thickness"]
    print(f"\n=== config {name}: {params} ===", flush=True)

    # 先建一次取 num_slices
    potential = abtem.Potential(
        frozen_phonons, parametrization="lobato", sampling=sampling,
        slice_thickness=st, projection="finite", exit_planes=None, device="gpu",
    )
    num_slices = potential.num_slices
    depth_slices = depth_slice_indices(num_slices, st)
    thickness_angstrom = np.asarray([(i + 1) * st for i in depth_slices])
    del potential
    free_gpu()

    potential = abtem.Potential(
        frozen_phonons, parametrization="lobato", sampling=sampling,
        slice_thickness=st, projection="finite", exit_planes=depth_slices, device="gpu",
    )
    probe = abtem.Probe(energy=ENERGY_KEV * 1e3, semiangle_cutoff=SEMIANGLE, device="gpu")
    probe.grid.match(potential)
    cell_x, cell_y, _ = atoms.cell.lengths()

    print(f"  gpts={probe.grid.gpts}  num_slices={num_slices}  depth_slices={depth_slices}", flush=True)
    mem_report("built potential")

    algo = RealSpaceMultislice(
        order=1, expansion_scope="full",
        derivative_accuracy=params["derivative_accuracy"],
        max_terms=params["max_terms"],
    )

    result = dict(name=name, params={k: float(v) for k, v in params.items()},
                  repeat=repeat, num_slices=num_slices, converged=True, error="")

    t0 = time.perf_counter()
    try:
        ew_bse = probe.multislice(
            potential, scan=[[cell_x / 2, cell_y / 2]],
            algorithm=algo, return_backscattered=False, lazy=False,
        )
        t_bse = time.perf_counter() - t0
        mem_report("after backscatter")

        t0 = time.perf_counter()
        ew_ref = probe.multislice(
            potential, scan=[[cell_x / 2, cell_y / 2]],
            algorithm=FourierMultislice(), lazy=False,
        )
        t_ref = time.perf_counter() - t0

        cbed_bse = ew_bse.diffraction_patterns(max_angle=MAX_ANGLE)
        cbed_ref = ew_ref.diffraction_patterns(max_angle=MAX_ANGLE)
        cb_bse = extract_exit(cbed_bse.array)
        cb_ref = extract_exit(cbed_ref.array)

        sum_w = np.abs(cb_bse).reshape(cb_bse.shape[0], -1).sum(axis=1)
        sum_wo = np.abs(cb_ref).reshape(cb_ref.shape[0], -1).sum(axis=1)
        net_percent = (sum_w - sum_wo) / np.maximum(sum_wo, 1e-30) * 100.0

        result.update(
            t_bse=t_bse, t_ref=t_ref,
            thickness_angstrom=thickness_angstrom,
            net_percent=net_percent,
            sum_with=sum_w, sum_without=sum_wo,
            sampling_recip=np.asarray(cbed_bse.sampling),
            wavelength=float(probe.accelerator.wavelength),
        )
        np.savez(
            OUT_DIR / f"conv_{name}.npz",
            cbed_with=cb_bse.astype(np.float32),
            cbed_without=cb_ref.astype(np.float32),
            **{k: v for k, v in result.items() if k not in ("error",)},
        )
        print(f"  net%(t) = {np.round(net_percent, 3)}", flush=True)
        print(f"  forward={t_ref:.1f}s  backscatter={t_bse:.1f}s", flush=True)
    except Exception as e:  # NotConvergedError / OOM 等：记录并继续下一组
        result["converged"] = False
        result["error"] = f"{type(e).__name__}: {e}"
        print(f"  FAILED: {result['error']}", flush=True)
        traceback.print_exc()
        with open(OUT_DIR / f"conv_{name}_FAILED.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    del potential, probe
    free_gpu()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="2x2x8 小晶胞冒烟测试")
    ap.add_argument("--only", type=str, default="", help="只跑指定轴, 逗号分隔: slice,sampling,fdacc,terms")
    args = ap.parse_args()

    repeat = (2, 2, 8) if args.smoke else (8, 8, 50)
    print(f"repeat={repeat}  (smoke={args.smoke})", flush=True)

    atoms = ase.io.read(CIF_PATH).repeat(repeat)
    frozen_phonons = abtem.FrozenPhonons(atoms, NUM_CONFIGS, DEBYE_WALLER)

    only = set(args.only.split(",")) if args.only else set(AXES)
    results = []
    for axis, variants in AXES.items():
        if axis not in only:
            continue
        for key, value in variants:
            params = dict(BASE)
            params[key] = value
            name = f"{axis}_{str(value).replace('.', 'p')}"
            # 与基准完全相同的配置若已跑过则跳过重复
            if (OUT_DIR / f"conv_{name}.npz").exists() and not args.smoke:
                print(f"skip {name} (npz exists)", flush=True)
                continue
            results.append(run_config(name, params, frozen_phonons, atoms, repeat))

    summary = [
        {k: v for k, v in r.items() if k in ("name", "params", "converged", "error",
                                             "t_bse", "t_ref", "num_slices")}
        for r in results
    ]
    suffix = "_smoke" if args.smoke else ""
    with open(OUT_DIR / f"summary{suffix}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nsummary written: {OUT_DIR / f'summary{suffix}.json'}", flush=True)


if __name__ == "__main__":
    main()
