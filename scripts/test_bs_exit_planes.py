# =============================================================================
# 回归测试：take_exit_plane 在不同厚度系列 (exit_planes) 配置下的正确性
#
#   背景：abTEM 的 return_backscattered=True 返回 (forward, backward)，
#   每个都是 Waves，形状 (n_config, n_exit, ny, nx)，出射面轴是 ThicknessAxis(label='z')。
#   官方教程用 forward[-1]/backward[0] 取出口/入口面，但那作用在 config 轴（第一个轴），
#   当 NUM_CONFIGS>1 时会静默取错轴（选中某个声子态的所有出射面）。本脚本验证改用
#   语义化定位 (label=='z') 的 take_exit_plane 在任意厚度系列/出声子态下都正确。
#
#   用法：
#     python test_bs_exit_planes.py                # 默认：3 种 exit_planes + 2 声子态
#     python test_bs_exit_planes.py --configs 1     # 单声子态
#     python test_bs_exit_planes.py --cpu           # 用 CPU（慢，但免 GPU）
#
#   通过标准：每种配置下 exit_axis 由 label=='z' 正确定位；取出口/入口面后为 2D；
#             diffraction_patterns 也为 2D。任一不满足则抛 AssertionError。
# =============================================================================
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import ase.io
import abtem
from abtem.multislice import RealSpaceMultislice

REPEAT = (2, 2, 8)                 # 小晶胞（快）
ENERGY_KEV = 5
SLICE_THICKNESS = 0.05
SEMIANGLE = 9.4
CIF_PATH = Path(__file__).resolve().parents[1] / "atomic_model" / "SrTiO3.cif"

# 被测的厚度系列（exit_planes）
DEFAULT_CASES = {
    "A: (100,200,400,626) 4出射面": (100, 200, 400, 626),
    "B: (-1,100,200,400,626) 带入口标记": (-1, 100, 200, 400, 626),
    "C: (50,150,300,450,626) 5出射面": (50, 150, 300, 450, 626),
}


def exit_plane_axis(waves) -> int:
    """按元数据定位出射面轴（ThicknessAxis label=='z'）。不依赖硬编码轴位置。"""
    for i, ax in enumerate(waves.ensemble_axes_metadata):
        if getattr(ax, "label", None) == "z":
            return i
    raise RuntimeError("找不到 ThicknessAxis(label='z') 出射面轴")


def take_exit_plane(waves, which: str = "exit"):
    """语义化取 'exit'（出口面/最后一个出射面）或 'entrance'（入口面/第一个出射面）。"""
    ax = exit_plane_axis(waves)
    idx = (waves.shape[ax] - 1) if which == "exit" else 0
    items = [0] * len(waves.shape[:-2])     # 前面 ensemble 轴取第 0 个 config
    items[ax] = idx
    return waves[tuple(items)]


def to_np(arr) -> np.ndarray:
    return arr.get() if hasattr(arr, "get") else np.asarray(arr)


def run_case(name, exit_planes, frozen_phonons, num_configs, device) -> bool:
    print(f"\n===== {name} =====", flush=True)
    potential = abtem.Potential(
        frozen_phonons, parametrization="lobato", sampling=0.05,
        slice_thickness=SLICE_THICKNESS, projection="finite",
        exit_planes=exit_planes, device=device,
    )
    probe = abtem.Probe(energy=ENERGY_KEV * 1e3, semiangle_cutoff=SEMIANGLE, device=device)
    probe.grid.match(potential)
    cell_x, cell_y, _ = frozen_phonons.atoms.cell.lengths()

    t0 = time.perf_counter()
    out = probe.multislice(
        potential, scan=[[cell_x / 2, cell_y / 2]],
        algorithm=RealSpaceMultislice(expansion_scope="full"),
        return_backscattered=True, lazy=False,
    )
    dt = time.perf_counter() - t0
    fwd, bwd = out[0], out[1]
    print(f"  [{dt:.1f}s] fwd.shape={fwd.shape}  bwd.shape={bwd.shape}", flush=True)

    ax = exit_plane_axis(bwd)
    n_exit_expected = len(exit_planes)
    n_exit_actual = bwd.shape[ax]

    fw = take_exit_plane(fwd, "exit")
    bw = take_exit_plane(bwd, "entrance")
    print(f"  exit_axis={ax} (n_exit={n_exit_actual} vs expected {n_exit_expected})", flush=True)
    print(f"  exit(出口面)->{fw.shape}  entrance(入口面)->{bw.shape}", flush=True)

    # --- 断言 ---
    assert ax == (bwd.array.ndim - 3), f"exit_axis={ax} != ndim-3={bwd.array.ndim-3}"
    assert n_exit_actual == n_exit_expected, (
        f"n_exit {n_exit_actual} != expected {n_exit_expected}"
    )
    # 取面后应为 2D 单出射面
    assert len(fw.shape) == 2, f"exit 出口面不是 2D: {fw.shape}"
    assert len(bw.shape) == 2, f"entrance 入口面不是 2D: {bw.shape}"
    # diffraction_patterns 也为 2D
    fdp = to_np(fw.diffraction_patterns(max_angle="cutoff").array)
    bdp = to_np(bw.diffraction_patterns(max_angle="cutoff").array)
    fdp = fdp[0] if fdp.ndim == 3 else fdp
    bdp = bdp[0] if bdp.ndim == 3 else bdp
    assert fdp.ndim == 2 and bdp.ndim == 2, (
        f"dp 非 2D: exit={fdp.shape} entrance={bdp.shape}"
    )
    print(f"  exit dp={fdp.shape}  entrance dp={bdp.shape}  OK", flush=True)
    print(f"  PASS", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser(description="回归测试 take_exit_plane 厚度系列")
    ap.add_argument("--configs", type=int, default=2, help="冷冻声子态数")
    ap.add_argument("--cpu", action="store_true", help="用 CPU")
    ap.add_argument("--repeat", type=str, default="2,2,8", help="REPEAT 三元组")
    args = ap.parse_args()

    device = "cpu" if args.cpu else "gpu"
    repeat = tuple(int(x) for x in args.repeat.split(","))

    srtio3 = ase.io.read(CIF_PATH).repeat(repeat)
    frozen_phonons = abtem.FrozenPhonons(
        srtio3, args.configs, {"Sr": 0.078, "Ti": 0.078, "O": 0.078}
    )
    print(f"REPEAT={repeat}  NUM_CONFIGS={args.configs}  device={device}", flush=True)

    passed = 0
    for name, eps in DEFAULT_CASES.items():
        if run_case(name, eps, frozen_phonons, args.configs, device):
            passed += 1

    ok = passed == len(DEFAULT_CASES)
    print(f"\n{'='*60}\n结果: {passed}/{len(DEFAULT_CASES)} 通过", flush=True)
    if ok:
        print("ALL PASS ✅", flush=True)
    else:
        print("SOME FAILED ❌", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
