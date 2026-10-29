# =============================================================================
# 幺正性分析图（5 keV）：FD6 vs FFT Laplacian 的前向积分强度与物理上界
#   面板 (a)：衍射花样积分强度绝对值（入射归一化=1），>1 即违反幺正性
#   面板 (b)：净变化 net% 与物理上界 (1-sum_without)/sum_without 对比
# 数据：conv_slice_0p05.npz（FD 6 阶基准）、conv_fft_laplace_5keV.npz（FFT 拉普拉斯）
# =============================================================================
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATA = Path(__file__).resolve().parent.parent / "figures" / "convergence"
OUT = DATA / "cbed_bse_unitarity_5keV.png"

fd = np.load(DATA / "conv_slice_0p05.npz")
fft = np.load(DATA / "conv_fft_laplace_5keV.npz")
prop = np.load(DATA / "conv_unit_src_prop_only.npz")

t_fd = fd["thickness_angstrom"]
t_fft = fft["thickness_angstrom"]
t_prop = prop["thickness_angstrom"]
sw_fd, swo_fd = fd["sum_with"], fd["sum_without"]
sw_fft, swo_fft = fft["sum_with"], fft["sum_without"]
bound = (1 - swo_fd) / swo_fd * 100

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

ax = axes[0]
ax.axhline(1.0, color="k", ls="--", lw=1, label="incident intensity (=1)")
ax.plot(t_fd, swo_fd, "o-", color="tab:blue", ms=4, label="without backscattering (Fourier)")
ax.plot(t_prop, prop["sum_rs"], "v-", color="tab:green", ms=4,
        label="propagator-only RRS (correction off)")
ax.plot(t_fd, sw_fd, "s-", color="tab:red", ms=4, label="with correction (FD order 6)")
ax.plot(t_fft, sw_fft, "^--", color="tab:orange", ms=4, label="with correction (FFT Laplacian)")
ax.set_xlabel("thickness [Å]")
ax.set_ylabel("integrated CBED intensity")
ax.set_title("(a) absolute intensity vs unitarity limit")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

ax = axes[1]
ax.plot(t_fd, bound, "k--", lw=1.5, label="physical bound (1-sum_without)/sum_without")
ax.plot(t_fd, (sw_fd - swo_fd) / swo_fd * 100, "s-", color="tab:red", ms=4, label="net change (FD order 6)")
ax.plot(t_fft, fft["net_percent"], "^--", color="tab:orange", ms=4, label="net change (FFT Laplacian)")
ax.set_xlabel("thickness [Å]")
ax.set_ylabel("net intensity change [%]")
ax.set_title("(b) net change vs physical bound")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

fig.suptitle("Unitarity check at 5 keV: the propagator-only RRS wave stays below the incident intensity "
             "(green), while the backscatter-corrected forward wave exceeds it (red/orange) and its net gain "
             "exceeds the physical bound — the excess (~2-3%) is an algorithm-inherent norm drift of the "
             "correction term, independent of the Laplacian scheme.", fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(OUT, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"saved: {OUT}")
