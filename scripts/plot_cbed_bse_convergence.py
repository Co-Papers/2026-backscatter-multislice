# =============================================================================
# 收敛性测试汇总绘图：net%(t) 按参数轴分组（4 面板）
#   面板 (a) slice_thickness 扫描
#   面板 (b) sampling 扫描
#   面板 (c) derivative_accuracy（FD 阶数）扫描
#   面板 (d) max_terms（级数截断）扫描
# 判据：随参数细化，net%(t) 曲线变化小 → 非单调特征是物理效应而非数值伪影。
#
# 注意：y 轴为线性轴（net% 有正有负且量级接近），不取对数。
#
# 用法：
#   python plot_cbed_bse_convergence.py                 # 读 paper/figures/convergence/
# =============================================================================
import glob
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "figures" / "convergence"
OUT_PNG = DATA_DIR / "cbed_bse_convergence_summary.png"

AXES_META = {
    "slice": ("slice_thickness", "slice thickness [Å]", "(a) slice thickness scan"),
    "sampling": ("sampling", "sampling [Å]", "(b) real-space sampling scan"),
    "fdacc": ("derivative_accuracy", "FD accuracy order", "(c) FD stencil order scan"),
    "terms": ("max_terms", "max terms in series", "(d) series truncation scan"),
}


def load_runs(data_dir):
    runs = []
    for f in sorted(glob.glob(str(data_dir / "conv_*.npz"))):
        m = re.match(r"conv_([a-z]+)_(.+)\.npz", Path(f).name)
        if not m:
            continue
        axis, valstr = m.group(1), m.group(2)
        d = np.load(f)
        runs.append(dict(
            axis=axis,
            value=float(valstr.replace("p", ".")),
            thickness=np.asarray(d["thickness_angstrom"]),
            net=np.asarray(d["net_percent"]),
        ))
    # 失败配置
    failed = {}
    for f in glob.glob(str(data_dir / "conv_*_FAILED.json")):
        m = re.match(r"conv_([a-z]+)_(.+)_FAILED\.json", Path(f).name)
        if m:
            with open(f, encoding="utf-8") as fh:
                failed[f"{m.group(1)}_{m.group(2)}"] = json.load(fh)["error"]
    return runs, failed


def main(data_dir=DATA_DIR):
    runs, failed = load_runs(data_dir)
    if not runs:
        print("no data found"); sys.exit(1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (axis, (key, xlabel, title)) in zip(axes.flat, AXES_META.items()):
        group = sorted([r for r in runs if r["axis"] == axis], key=lambda r: r["value"])
        cmap = plt.get_cmap("viridis")
        for i, r in enumerate(group):
            c = cmap(i / max(len(group) - 1, 1))
            label = f"{r['value']:g}"
            ax.plot(r["thickness"], r["net"], marker="o", ms=4, color=c, label=label)
        # 标注该轴上的失败配置
        for name, err in failed.items():
            if name.startswith(f"{axis}_"):
                ax.text(0.02, 0.02 + 0.06 * list(failed).index(name),
                        f"{name}: FAILED ({err.split(':')[0]})",
                        transform=ax.transAxes, fontsize=7, color="red")
        ax.set_xlabel("thickness [Å]")
        ax.set_ylabel("net intensity change [%]")
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(title=xlabel, fontsize=8, title_fontsize=8)

    fig.suptitle(
        "Convergence test: net intensity change (with vs without backscattering) at 5 keV\n"
        "If curves within each panel collapse onto each other as the parameter is refined, "
        "the non-monotonic / net-gain feature is physical, not a numerical artifact.",
        fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {OUT_PNG}", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_DIR)
