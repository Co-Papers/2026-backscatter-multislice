# =============================================================================
# abTEM 背散射 CBED：FD(17Pt) vs FFT 拉普拉斯 对比 —— 计算脚本（只产数据）
#
# 说明
# ----
#   本脚本做**重计算**（多切片），并按 `--compute` 控制算哪一类 CBED：
#      ref : 常规 FourierMultislice（无背散射）
#      fd  : RealSpace full + 有限差分 acc=8（1D 9 点/2D 17Pt，有背散射）
#      fft : RealSpace full + laplace_method="fft"（倒空间 -k²，有背散射）
#   未列入 --compute 的类型，若已有多态 npz（cbed_bse_phonons_16_*keV.npz，字段
#   cbed_without / cbed_with）则**直接加载**，不重复计算。FFT 结果必须现算。
#   脚本只把三类 CBED 保存到 npz；绘图/指标由
#   `cbed_bse_multi_phonons_16_postprocess.py` 从该 npz 出图。
#
# 依赖 abTEM 源码级选项（见 abtem/finite_difference.py、abtem/multislice.py）：
#   * LaplaceOperator(accuracy, method="finite_difference"|"fft")
#   * RealSpaceMultislice(..., laplace_method="finite_difference"|"fft")
#
# 参数与 cbed_bse_energy_sweep_multi_phonons_16.py 一致：sampling=0.05 Å、
# slice_thickness=0.05 Å、semiangle_cutoff=9.4 mrad、lobato、finite、16 声子态、5 keV。
#
# 用法
# ----
#   python cbed_bse_multi_phonons_16_fft.py                 # 只算 fft，ref/fd 从 npz 加载
#   python cbed_bse_multi_phonons_16_fft.py --compute fft fd # 算 fft + fd（本机 17Pt）
#   python cbed_bse_multi_phonons_16_fft.py --compute ref fd fft  # 三者全算（很慢）
#   python cbed_bse_multi_phonons_16_fft.py --configs 2 --thickness 40   # 快速调试
#
# 输出（到 paper/figures_multi_phonons_16/）
# ----
#   cbed_bse_multi_phonons_16_fd_vs_fft.npz   （cbed_reference/cbed_fd/cbed_fft + 元数据 + 指标）
# =============================================================================
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import abtem
import ase.io
from abtem.multislice import RealSpaceMultislice, FourierMultislice

CIF_PATH = Path(__file__).resolve().parents[1] / "atomic_model" / "SrTiO3.cif"
OUT_DIR = Path(__file__).resolve().parent.parent / "figures_multi_phonons_16"
OUT_DIR.mkdir(exist_ok=True)
TAG = "fd_vs_fft"

# 与 cbed_bse_energy_sweep_multi_phonons_16.py 一致
SAMPLING = 0.05
SLICE_THICKNESS = 0.05
SEMIANGLE_CUTOFF = 9.4
MAX_ANGLE = "cutoff"
PARAMETRIZATION = "lobato"
REPEAT = (8, 8, 50)
NUM_CONFIGS_DEFAULT = 16
ENERGY_KEV_DEFAULT = 5.0
FD_ACCURACY = 8            # 1D 9 点 / 2D 17Pt
DEFAULT_REF_NPZ = OUT_DIR / "cbed_bse_phonons_16_5keV.npz"   # 已有多态（常规 ref/fd）

# 已有多态 npz 中，厚度序列的最后一项 = 底部全厚（约 195 Å）—— 与默认全部出射一致
LOAD_THICKNESS_INDEX = -1


def to_numpy(x):
    if hasattr(x, "compute"):
        x = x.compute()
    if hasattr(x, "get"):
        x = x.get()
    return np.asarray(x)


def extract_exit(arr):
    arr = to_numpy(arr)
    if arr.ndim == 4:
        return arr.mean(axis=0)[-1]
    if arr.ndim == 3:
        a = arr.mean(axis=0) if arr.shape[0] > 1 else arr[0]
        return a[-1] if a.ndim == 3 else a
    return arr


def load_cb(npz_path, key, index=-1):
    """从已有多态 npz 加载某厚度处的 CBED 二维强度图。"""
    d = np.load(npz_path, allow_pickle=True)
    a = d[key]
    return np.asarray(a[index] if a.ndim == 3 else a, dtype=float)


def load_meta(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    return {
        "sampling": np.asarray(d["sampling"], dtype=float),
        "wavelength": float(d["wavelength"]),
        "extent": np.abs(np.asarray(d["extent"], dtype=float)),
    }


def run_multislice(potential, probe, scan, algorithm):
    t0 = time.perf_counter()
    ew = probe.multislice(potential, scan=scan, algorithm=algorithm,
                          return_backscattered=False, lazy=False)
    t = time.perf_counter() - t0
    return ew, t


def main():
    parser = argparse.ArgumentParser(description="abTEM 背散射 CBED：FD(17Pt) vs FFT 拉普拉斯（计算，只产数据）")
    parser.add_argument("--compute", nargs="+", choices=["ref", "fd", "fft"], default=["fft"],
                        help="要现算的类型；未指定的会尝试从已有多态 npz 加载")
    parser.add_argument("--energy", type=float, default=ENERGY_KEV_DEFAULT)
    parser.add_argument("--configs", type=int, default=NUM_CONFIGS_DEFAULT)
    parser.add_argument("--thickness", type=float, default=None, help="出射面厚度 [Å]（默认全厚≈195Å）")
    parser.add_argument("--fd-accuracy", type=int, default=FD_ACCURACY,
                        help="现算 fd 时的差分精度（默认 8 = 1D 9 点 / 2D 17Pt）")
    parser.add_argument("--no-gpu", action="store_true")
    args = parser.parse_args()

    compute = args.compute
    device = "cpu" if args.no_gpu else "gpu"
    num_configs = args.configs

    # FFT 结果无法从现有 npz 加载，必须现算
    if "fft" not in compute:
        raise ValueError("fft 结果无法从现有 npz 加载，必须包含在 --compute 中")

    # ---- 若需现算，构造势场/探针 ----
    potential = probe = scan = None
    full_thickness = 195.6
    if compute:
        srtio3 = ase.io.read(CIF_PATH).repeat(REPEAT)
        frozen_phonons = abtem.FrozenPhonons(
            srtio3, num_configs, {"Sr": 0.078, "Ti": 0.078, "O": 0.078}
        )
        z_cell = float(srtio3.cell.lengths()[2])
        full_thickness = z_cell * REPEAT[2]
        exit_planes = None
        if args.thickness is not None:
            exit_planes = (int(round(args.thickness / SLICE_THICKNESS)) + 1,)
        potential = abtem.Potential(
            frozen_phonons, parametrization=PARAMETRIZATION, sampling=SAMPLING,
            slice_thickness=SLICE_THICKNESS, projection="finite",
            exit_planes=exit_planes, device=device,
        )
        probe = abtem.Probe(energy=args.energy * 1e3, semiangle_cutoff=SEMIANGLE_CUTOFF, device=device)
        probe.grid.match(potential)
        cell_x, cell_y, _ = srtio3.cell.lengths()
        scan = [[cell_x / 2, cell_y / 2]]
        full_thickness = float(potential.num_slices) * SLICE_THICKNESS

    print("=" * 78)
    print(f" 能量={args.energy} keV, 声子态={num_configs}, compute={compute}")
    print("=" * 78)

    # ---- 各类型：现算或加载 ----
    times = {}
    cb = {}

    if "ref" in compute:
        print("\n[*] 现算 ref: 常规 FourierMultislice（无背散射） ...", flush=True)
        ew, t = run_multislice(potential, probe, scan, FourierMultislice())
        cb["ref"], times["ref"] = extract_exit(ew.diffraction_patterns(max_angle=MAX_ANGLE).array), t
    else:
        cb["ref"] = load_cb(DEFAULT_REF_NPZ, "cbed_without", LOAD_THICKNESS_INDEX)
        print(f"\n[*] 加载 ref（{DEFAULT_REF_NPZ} 的 cbed_without[{LOAD_THICKNESS_INDEX}]）")

    if "fd" in compute:
        print(f"\n[*] 现算 fd: RealSpace full + FD accuracy={args.fd_accuracy} ...", flush=True)
        ew, t = run_multislice(potential, probe, scan,
                               RealSpaceMultislice(expansion_scope="full",
                                                   derivative_accuracy=args.fd_accuracy))
        cb["fd"], times["fd"] = extract_exit(ew.diffraction_patterns(max_angle=MAX_ANGLE).array), t
    else:
        cb["fd"] = load_cb(DEFAULT_REF_NPZ, "cbed_with", LOAD_THICKNESS_INDEX)
        print(f"\n[*] 加载 fd（{DEFAULT_REF_NPZ} 的 cbed_with[{LOAD_THICKNESS_INDEX}]，"
              f"为已有多态默认精度结果）")

    if "fft" in compute:
        print("\n[*] 现算 fft: RealSpace full + laplace_method='fft' ...", flush=True)
        ew, t = run_multislice(potential, probe, scan,
                               RealSpaceMultislice(expansion_scope="full", laplace_method="fft"))
        cb["fft"], times["fft"] = extract_exit(ew.diffraction_patterns(max_angle=MAX_ANGLE).array), t
    else:
        raise RuntimeError("fft 必须现算")

    # ---- 元数据（优先用已有多态 npz 的，保证与加载/现算的网格一致）----
    meta = None
    try:
        meta = load_meta(DEFAULT_REF_NPZ)
    except Exception:
        pass
    if meta is None:
        meas = ew.diffraction_patterns(max_angle=MAX_ANGLE)
        meta = {"sampling": np.asarray(meas.sampling, float),
                "wavelength": float(probe.accelerator.wavelength),
                "extent": np.abs(np.asarray(meas.extent, float))}

    # ---- 指标（FFT vs FD）+ 计时 ----
    diff = np.abs(cb["fft"] - cb["fd"])
    max_abs = float(diff.max())
    max_fd = float(cb["fd"].max())
    rel = max_abs / max(max_fd, 1e-30)
    int_cons = (float(cb["fft"].sum()) - float(cb["fd"].sum())) / max(float(cb["fd"].sum()), 1e-30)

    print("\n" + "=" * 78)
    print(" 指标（FFT-Laplacian 背散射 vs FD 背散射）:")
    print(f"   max|I_FFT - I_FD|                = {max_abs:.6e}")
    print(f"   max|I_FFT - I_FD| / max(I_FD)    = {rel:.6e}")
    print(f"   (ΣI_FFT - ΣI_FD) / ΣI_FD         = {int_cons:+.6e}")
    if times:
        parts = [f"{k}={v:.2f}s" for k, v in times.items() if v is not None]
        if "fft" in times and "fd" in times and times["fft"] and times["fd"]:
            parts.append(f"FFT/FD = {times['fft'] / max(times['fd'], 1e-9):.3f}x")
        print("-" * 78)
        print(" 计时: " + ", ".join(parts))
    print("=" * 78)

    # ---- 保存数据（供后处理出图）----
    np.savez(
        OUT_DIR / f"cbed_bse_multi_phonons_16_{TAG}.npz",
        energy_kev=args.energy, n_configs=num_configs, fd_accuracy=args.fd_accuracy,
        thickness_angstrom=float(args.thickness) if args.thickness else float(full_thickness),
        cbed_reference=cb["ref"].astype(np.float32),
        cbed_fd=cb["fd"].astype(np.float32),
        cbed_fft=cb["fft"].astype(np.float32),
        diff_fd_fft=diff.astype(np.float32),
        sampling=meta["sampling"], wavelength=meta["wavelength"], extent=meta["extent"],
        max_abs=max_abs, rel=rel, intensity_conservation=int_cons,
        **{f"time_{k}": v for k, v in times.items()},
    )
    print(f"saved data: {OUT_DIR / f'cbed_bse_multi_phonons_16_{TAG}.npz'}")
    print("\n下一步：运行 cbed_bse_multi_phonons_16_postprocess.py 从该 npz 出图与出指标。")


if __name__ == "__main__":
    main()
