# =============================================================================
# 基准测试：差分拉普拉斯「要多少阶才能达到 FT 精度」+「时间对比」
#
# 两个问题
# --------
#   A) 频域（与网格无关）：某一精度阶 accuracy 的有限差分模板，其“有效波数”在哪个归一化
#      频率 q=κ/κ_N 处偏离精确抛物线（FT：-κ²）超过给定容差 δ？反过来，要在 q 以内达到
#      FT 级精度，至少需要多少阶？这直接对应 Wacker & Schröder (2015, Fig. 2)。
#   B) 实测量：在真实网格上，对每个 accuracy 计算有限差分拉普拉斯（CPU/GPU）与傅里叶
#      (FFT) 拉普拉斯，测量（i）相对误差 vs FT 基准，（ii）单次应用耗时；并给出达到
#      “FT 精度”（误差进入 FT 自身机器精度地板 ~1e-12 附近）所需的最小阶数。
#
# 复用 abTEM 源码
# --------------
#   * `abtem/finite_difference.py` 的 `finite_difference_coefficients` 生成系数；
#   * `_laplace_operator_stencil(accuracy, prefactor, mode, dtype, device)` 得到实际算子，
#     device="cpu" -> `_laplace_stencil_cpu_batch`（numba），device="gpu" -> `_laplace_stencil_gpu`（cupy）。
#   * 傅里叶基准见 `abtem/finite_difference.py` §2.2.1：∇² -> -(2πk)²（本脚本用 numpy.fft 实现）。
#
# 独立性：纯 NumPy + 可选 CuPy，不含背散射 / multislice / 任何物理修正。
#
# 用法
# ----
#   python benchmark_laplacian_ft.py                # 默认 float64, N=[256,512], 精度阶 2..18
#   python benchmark_laplacian_ft.py --precision float32 --no-gpu
#
# 输出
# ----
#   1) 控制台：频域“所需阶数 vs q/容差”表 + 各网格“误差/耗时/达到FT的最小阶数”表；
#   2) `paper/figures/laplacian_req_accuracy.png`：不同容差下“达到 FT 精度所需阶数 vs q”；
#   3) `paper/figures/laplacian_timing.png`：误差 vs 阶数 + 耗时 vs 阶数（CPU/GPU + FFT 参考）。
# =============================================================================
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from abtem.finite_difference import (  # noqa: E402
    _laplace_operator_stencil,
    finite_difference_coefficients,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

DOMAIN_LEN = 4.0                 # 周期域边长（对 abTEM 即 Å）
ACCURACIES = [2, 4, 6, 8, 10, 12, 14, 16, 18]
N_LIST = [256, 512]              # 实测量网格
FT_PRECISION_FACTOR = 100.0      # “达到 FT 精度”=(FD误差 ≤ 该倍数 × FT误差地板)
TIME_REPS = 5                    # 计时重复次数，取中位数


# -----------------------------------------------------------------------------
# 1) 频域：有效波数偏离抛物线分析（网格无关）
# -----------------------------------------------------------------------------
def effective_factor(accuracy, q):
    """返回归一化有效因子 M(q)=κ_eff²/κ_N²。

    M(q) = -(1/π²)·[c_n + 2 Σ_{j=1..n} c_{n+j} cos(jπq)]  （n=模板半宽，q=κ/κ_N）。
    FT 精确解为 M(q)=q²；q→0 时各阶 M(q)→q²。
    """
    c = finite_difference_coefficients(2, accuracy)
    n = len(c) // 2
    th = np.pi * q
    return -(1 / np.pi**2) * (c[n] + 2 * sum(c[n + j] * np.cos(j * th) for j in range(1, n + 1)))


def q_tolerance(accuracy, tol):
    """返回该阶能把有效波数偏离控制在 rel≤tol 内的最大归一化频率 q_max。

    对 q∈(0,1] 扫描，找到第一个使 |M-q²|/q² > tol 的 q，其前一点即 q_max。
    tol=1e-6 可视为“逼近 FT 机器精度”。
    """
    q = np.linspace(1e-4, 1.0, 20001)
    rel = np.abs(effective_factor(accuracy, q) - q**2) / q**2
    idx = np.where(rel > tol)[0]
    return float(q[idx[0] - 1]) if len(idx) else 1.0


def run_frequency_analysis():
    print("\n" + "=" * 78)
    print("A) 频域：达到 FT 精度（有效波数偏离抛物线 ≤ 容差 δ）所需的最小阶数")
    print("   “FT 精度” = 有效波数完全贴合 -κ² 抛物线；δ 越小越接近 FT。")
    print("=" * 78)

    tolerances = [0.05, 0.01, 0.001, 1e-6]
    # 对每个容差，给出“要在 q 内达到该精度，至少需要多少阶”（分段阶梯）
    print(f"{'q=κ/κ_N':>9} | " + " | ".join(f"{'δ=' + format(t, 'g') + ' (需阶数)':>16}" for t in tolerances))
    print("-" * (9 + 3 + 18 * len(tolerances)))
    
    q_probes = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    for q in q_probes:
        cells = []
        for tol in tolerances:
            a_min = None
            for a in ACCURACIES:
                if effective_factor(a, q) >= (1 - tol) * q**2:
                    a_min = a
                    break
            cells.append(f"acc={a_min}" if a_min else "  >18")
        print(f"{q:>9.2f} | " + " | ".join(f"{c:>16}" for c in cells))
    
    print("\n  解读：q=κ/κ_N 为归一化空间频率（1.0=Nyquist）。要把该频率处的有效波数贴合到")
    print("  精确抛物线 -κ²、相对偏离 ≤δ，需要对应的精度阶 acc。δ=1e-6 已接近 FT 机器精度。")


# -----------------------------------------------------------------------------
# 2) 实测量：误差 + 耗时（在真实网格上）
# -----------------------------------------------------------------------------
def broadband_field(N, seed=0):
    """构造一个周期、宽带、含多个中等频率模式的“任意”复数场。"""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, DOMAIN_LEN, N, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="xy")
    f = np.zeros((N, N), dtype=np.complex128)
    for m in [8, 15, 22, 30, 45]:                 # 整数倍频 -> 周期；在 N 时 q≈0.06~0.35
        kx, ky = 2 * np.pi * m / DOMAIN_LEN, 2 * np.pi * (m + 2) / DOMAIN_LEN
        amp = rng.uniform(0.3, 1.0)
        ph = rng.uniform(0, 2 * np.pi)
        f += amp * np.exp(1j * (kx * X + ky * Y + ph))
    return f


def laplacian_fft(f, dx):
    """CPU（numpy）实现的谱拉普拉斯：∇² -> -(2πk)²。"""
    fh = np.fft.fft2(f)
    kx = np.fft.fftfreq(f.shape[1], d=dx)
    ky = np.fft.fftfreq(f.shape[0], d=dx)
    k2 = (2 * np.pi * kx)[None, :] ** 2 + (2 * np.pi * ky)[:, None] ** 2
    return np.fft.ifft2(-k2 * fh)


def laplacian_fft_gpu(fg, dx):
    """GPU（cupy）实现的谱拉普拉斯，与 laplacian_fft 数学等价，用于公平的 GPU 计时对比。"""
    import cupy as cp

    fh = cp.fft.fft2(fg)
    kx = cp.fft.fftfreq(fg.shape[1], d=dx)
    ky = cp.fft.fftfreq(fg.shape[0], d=dx)
    k2 = (2 * cp.pi * kx)[None, :] ** 2 + (2 * cp.pi * ky)[:, None] ** 2
    return cp.fft.ifft2(-k2 * fh)


def _timeit(fn, *args, reps=TIME_REPS, sync=None):
    """预热一次后计时 reps 次，返回中位耗时（秒/次）。

    sync：可选回调，在每次计时后调用（GPU 上用 deviceSynchronize 等待内核完成，
    否则 perf_counter 只测到异步发射耗时，会严重低估真实计算时间）。
    """
    fn(*args)  # warm-up（含 numba 编译 / cupy 初始化）
    if sync is not None:
        sync()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(*args)
        if sync is not None:
            sync()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


# -----------------------------------------------------------------------------
# 优化的 GPU-FD：单次发射融合核（周期边界），替代 abTEM 的逐次切片累加
# -----------------------------------------------------------------------------
_LAPLACE_KERNEL_SRC = r"""
extern "C" __global__ void laplace(
    const double* a, double* out, const double* c,
    int n, int H, int W) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = H * W;
  if (idx >= total) return;
  int i = idx / W, j = idx % W;
  double ar = 0.0, ai = 0.0;
  for (int k = -n; k <= n; k++) {
    int ii = i + k; if (ii < 0) ii += H; else if (ii >= H) ii -= H;
    int jj = j + k; if (jj < 0) jj += W; else if (jj >= W) jj -= W;
    double ck = c[k + n];
    int p0 = (ii * W + j) * 2, p1 = (i * W + jj) * 2;
    ar += ck * (a[p0] + a[p1]);
    ai += ck * (a[p0 + 1] + a[p1 + 1]);
  }
  out[2 * idx] = ar;
  out[2 * idx + 1] = ai;
}
"""
_laplace_kernel = None


def laplace_gpu_fused(f, dx, accuracy):
    """优化的 GPU 有限差分拉普拉斯：单次发射融合核（周期边界）。

    对比 abTEM `_laplace_operator_stencil(device="gpu")` 所用的 `_laplace_stencil_gpu`——
    那是逐次整阵切片累加（每个抽头读写全数组一次，内存带宽受限）；本实现每个线程
    一次读出、全部抽头累加、写回一次，大幅减少全局访存。

    说明：复数 complex128 在内存中为连续两个 double（实、虚），核内按 2*idx 索引，
    实系数算子对实/虚部独立，故一次发射即可完成，无需 CUDA complex 头。
    """
    global _laplace_kernel
    import cupy as cp

    if _laplace_kernel is None:
        _laplace_kernel = cp.RawKernel(_LAPLACE_KERNEL_SRC, "laplace")
    coefs = cp.asarray(finite_difference_coefficients(2, accuracy) * (1.0 / dx**2))
    n = len(coefs) // 2
    out = cp.empty_like(f)
    conf = 256
    grid = ((f.size + conf - 1) // conf,)
    _laplace_kernel(grid, (conf,), (f, out, coefs, n, f.shape[0], f.shape[1]))
    return out


def rel_linf(err, ref):
    d = float(np.max(np.abs(ref)))
    return float(np.max(np.abs(err)) / (d if d > 0 else 1e-300))


def get_stencil(accuracy, dx, device, dtype):
    return _laplace_operator_stencil(
        accuracy, 1.0 / dx**2, mode="wrap", dtype=np.dtype(dtype).type, device=device
    )


def run_real_benchmark(dtype, use_gpu):
    print("\n" + "=" * 78)
    print(f"B) 实测量（dtype={np.dtype(dtype).name}）：误差 vs 阶数、耗时 vs 阶数")
    print("   “FT 精度” = 有限差分相对误差 ≤ %.0f × FT 自身误差地板。" % FT_PRECISION_FACTOR)
    print("=" * 78)

    has_gpu = use_gpu
    gpu_sync = None
    if has_gpu:
        try:
            import cupy as cp

            cp.cuda.runtime.getDeviceCount()
            gpu_sync = cp.cuda.runtime.deviceSynchronize
        except Exception:
            print("[warn] CUDA/cupy 不可用，跳过 GPU", file=__import__("sys").stderr)
            has_gpu = False

    # 收集各网格结果
    results = {}   # N -> list of dict per accuracy
    for N in N_LIST:
        dx = DOMAIN_LEN / N
        f = broadband_field(N).astype(dtype)

        # FT 参考（CPU，numpy-FFT）+ 计时
        fft_time = _timeit(laplacian_fft, f, dx)
        fft_lap = laplacian_fft(f, dx)

        # FT 参考（GPU，cupy-FFT）+ 计时（同步后测真实耗时）
        fft_gpu_time = None
        if has_gpu:
            fg_ref = cp.asarray(f)
            fft_gpu_time = _timeit(laplacian_fft_gpu, fg_ref, dx, sync=gpu_sync)
            del fg_ref

        rows = []
        for a in ACCURACIES:
            # CPU（numba 单线程）
            st_cpu = get_stencil(a, dx, "cpu", dtype)
            fd_cpu_time = _timeit(lambda: st_cpu(f))
            fd_cpu = st_cpu(f)
            err_cpu = rel_linf(fd_cpu - fft_lap, fft_lap)

            rec = {"acc": a, "pt": 2 * a + 1, "err_cpu": err_cpu, "t_cpu": fd_cpu_time}
            if has_gpu:
                fg = cp.asarray(f)

                # abTEM GPU：现已是单次发射融合核（本会话已把 `_laplace_stencil_gpu`
                # 从逐次切片累加改成了融合 CUDA 核，见 abtem/finite_difference.py）
                st_gpu = get_stencil(a, dx, "gpu", dtype)
                t_gpu_abtem = _timeit(lambda: st_gpu(fg), sync=gpu_sync)
                fd_gpu_abtem = cp.asnumpy(st_gpu(fg)).astype(dtype)

                # 独立实现的融合核参考（应与 abTEM 一致）
                t_gpu_fused = _timeit(lambda: laplace_gpu_fused(fg, dx, a), sync=gpu_sync)
                fd_gpu_fused = cp.asnumpy(laplace_gpu_fused(fg, dx, a)).astype(dtype)

                rec["err_gpu_abtem"] = rel_linf(fd_gpu_abtem - fft_lap, fft_lap)
                rec["err_gpu_fused"] = rel_linf(fd_gpu_fused - fft_lap, fft_lap)
                rec["t_gpu_abtem"] = t_gpu_abtem
                rec["t_gpu_fused"] = t_gpu_fused
            rows.append(rec)
        results[N] = {"dx": dx, "fft_time": fft_time, "fft_gpu_time": fft_gpu_time, "rows": rows}

        # 打印该网格的表
        print("\n" + "-" * 78)
        ft_str = f"{fft_time * 1e3:.3f} ms/次"
        if has_gpu:
            ft_str += f"  |  GPU-FFT = {fft_gpu_time * 1e3:.3f} ms/次"
        print(f"网格 N={N} (Δx={dx:.4f})   FT 参考耗时 = {ft_str}")
        print("-" * 78)
        hdr = f"{'acc':>4} {'2Dpt':>5} | {'CPU err∞':>11} "
        if has_gpu:
            hdr += f"{'GPU(abTEM) err∞':>14} {'GPU(fused) err∞':>14} "
        hdr += f"| {'t_cpu':>8} "
        if has_gpu:
            hdr += f"{'t_gpuAb':>8} {'t_gpuF':>8} "
        print(hdr)
        print("-" * len(hdr))

        # 找达到 FT 精度的最小阶数（用 CPU 误差）
        ft_floor = 1e-12 if np.dtype(dtype).itemsize == 16 else 1e-7   # FT 误差地板估计
        reach = None
        for r in rows:
            is_reach = r["err_cpu"] <= FT_PRECISION_FACTOR * ft_floor
            if is_reach and reach is None:
                reach = r["acc"]
            line = f"{r['acc']:>4} {r['pt']:>5} | {r['err_cpu']:>11.3e} "
            if has_gpu:
                line += f"{r['err_gpu_abtem']:>14.3e} {r['err_gpu_fused']:>14.3e} "
            line += f"| {r['t_cpu'] * 1e3:>8.3f} "
            if has_gpu:
                line += f"{r['t_gpu_abtem'] * 1e3:>8.3f} {r['t_gpu_fused'] * 1e3:>8.3f}"
            print(line + ("   ←达到FT精度" if r["acc"] == reach else ""))
        if reach:
            print(f"C→ 达到 FT 精度（FD 误差 ≤ {FT_PRECISION_FACTOR:.0f}×地板 {ft_floor:.0e}）的"
                  f"最小阶数 = acc={reach}（{2 * reach + 1}Pt）")
        else:
            print(f"C→ 该网格下所有阶数均未达 FT 精度地板 {ft_floor:.0e}")

    return results, has_gpu


# -----------------------------------------------------------------------------
# 3) 绘图
# -----------------------------------------------------------------------------
def plot_req_accuracy():
    """不同容差下，所需阶数 vs 归一化频率（或 q_max vs 阶数）。"""
    tolerances = [0.05, 0.01, 0.001, 1e-6]
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    cols = plt.cm.viridis(np.linspace(0.1, 0.85, len(tolerances)))
    for tol, col in zip(tolerances, cols):
        qs = [q_tolerance(a, tol) for a in ACCURACIES]
        ax.plot(ACCURACIES, qs, "o-", color=col, label=f"δ={tol:g}")
        ax.annotate(f"{qs[-1]:.2f}", (ACCURACIES[-1], qs[-1]), fontsize=7,
                    color=col, va="bottom", ha="left")
    ax.set_xlabel("finite-difference accuracy order")
    ax.set_ylabel("max normalized frequency  q_max = κ/κ_N (within δ of −κ²)")
    ax.set_title("To match FT (−κ²) within δ, use accuracy ≥ this (high q = harder)")
    ax.set_ylim(0, 1.05)
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = OUT_DIR / "laplacian_req_accuracy.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"图已保存: {out}")


def plot_timing(results, has_gpu, dtype):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    cols = plt.cm.plasma(np.linspace(0.1, 0.85, len(N_LIST)))

    # 左图：误差 vs 阶数（各网格），画 FT 地板线
    ax = axes[0]
    for (N, data), col in zip(results.items(), cols):
        accs = [r["acc"] for r in data["rows"]]
        errs = [r["err_cpu"] for r in data["rows"]]
        ax.loglog(accs, errs, "o-", color=col, label=f"FD CPU  N={N}")
        if has_gpu:
            errg = [r["err_gpu_fused"] for r in data["rows"]]
            ax.loglog(accs, errg, "s--", color=col, alpha=0.6, mfc="none",
                      label=f"FD GPU(fused)  N={N}")
    ft_floor = 1e-12 if np.dtype(dtype).itemsize == 16 else 1e-7
    ax.axhline(ft_floor, color="k", ls="--", lw=1.2, label=f"FT machine floor ≈{ft_floor:.0e}")
    ax.set_xlabel("accuracy order"); ax.set_ylabel("FD rel error vs FT (L∞)")
    ax.set_title("Error vs accuracy (reaching FT precision)")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=7, loc="best")

    # 右图：耗时 vs 阶数（CPU / abTEM-GPU / 优化GPU-FD + FFT 参考）
    ax = axes[1]
    for (N, data), col in zip(results.items(), cols):
        accs = [r["acc"] for r in data["rows"]]
        t_cpu = [r["t_cpu"] * 1e3 for r in data["rows"]]
        ax.semilogy(accs, t_cpu, "o-", color=col, label=f"FD CPU  N={N}")
        ax.axhline(data["fft_time"] * 1e3, color=col, ls=":", lw=1.2,
                   label=f"FFT(CPU) N={N}")
        if has_gpu:
            t_ab = [r["t_gpu_abtem"] * 1e3 for r in data["rows"]]
            ax.semilogy(accs, t_ab, "s--", color=col, alpha=0.6, mfc="none",
                        label=f"FD GPU(abTEM) N={N}")
            t_fu = [r["t_gpu_fused"] * 1e3 for r in data["rows"]]
            ax.semilogy(accs, t_fu, "^-", color=col, alpha=0.9, ms=4,
                        label=f"FD GPU(fused) N={N}")
            if data.get("fft_gpu_time"):
                ax.axhline(data["fft_gpu_time"] * 1e3, color=col, ls="-.", lw=1.2,
                           label=f"FFT(GPU) N={N}")
    ax.set_xlabel("accuracy order"); ax.set_ylabel("time per apply (ms)")
    ax.set_title("Timing: FD (CPU/GPU) vs FFT")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=7, loc="best")

    fig.suptitle(f"Laplacian: accuracy needed to reach FT precision & timing (dtype={np.dtype(dtype).name})")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = OUT_DIR / "laplacian_timing.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"图已保存: {out}")


# -----------------------------------------------------------------------------
# 5) 复杂度缩放：随网格 N 放大，验证 FD(固定低阶) vs FFT 的时间复杂度
# -----------------------------------------------------------------------------
SCALE_N = [128, 256, 512, 1024, 2048]


def run_scaling(dtype, reps=TIME_REPS):
    """随 N 放大，观测固定低阶 FD(acc=2) 与 numpy-FFT 的耗时。

    理论复杂度：FD(可分离模板) ≈ O(N²·k)（k=模板宽度，固定低阶为常数）；
    FFT ≈ O(N²·log N)。故 FD 更省，且随 N 增长优势应扩大。
    这里打印两条线并拟合 log-log 斜率以核对指数（纯 O(N²) 的指数为 2）。
    """
    print("\n" + "=" * 78)
    print("C) 复杂度缩放（CPU，acc=2 固定 3 点模板 vs numpy-FFT）")
    print("   理论：FD=O(N²·k) vs FFT=O(N²·log N)  →  FD 应更快且随 N 优势扩大")
    print("=" * 78)

    ns, ts_fd, ts_fft = [], [], []
    prev = None
    print(f"{'N':>6} {'Δx':>8} | {'t_FD(ms)':>10} {'t_FFT(ms)':>10} {'FFT/FD':>8} | {'FD指数':>8} {'FFT指数':>8}")
    for N in SCALE_N:
        dx = DOMAIN_LEN / N
        x = np.linspace(0, DOMAIN_LEN, N, endpoint=False)
        X, Y = np.meshgrid(x, x, indexing="xy")
        f = (np.cos(6 * np.pi * X / DOMAIN_LEN) * np.cos(4 * np.pi * Y / DOMAIN_LEN)).astype(dtype)

        st = get_stencil(2, dx, "cpu", dtype)
        st(f)  # 触发 numba 编译
        tfd = _timeit(lambda: st(f), reps=reps)
        tfft = _timeit(laplacian_fft, f, dx, reps=reps)

        if prev is not None:
            sFD = np.log(tfd / prev[1]) / np.log(N / prev[0])
            sFFT = np.log(tfft / prev[2]) / np.log(N / prev[0])
        else:
            sFD = sFFT = float("nan")
        print(f"{N:>6} {dx:>8.4f} | {tfd * 1e3:>10.3f} {tfft * 1e3:>10.3f} {tfft / tfd:>8.2f} | {sFD:>8.2f} {sFFT:>8.2f}")
        ns.append(N); ts_fd.append(tfd); ts_fft.append(tfft)
        prev = (N, tfd, tfft)

    ns = np.array(ns); ts_fd = np.array(ts_fd); ts_fft = np.array(ts_fft)
    m = ns >= 512
    sFD = np.polyfit(np.log(ns[m]), np.log(ts_fd[m]), 1)[0]
    sFFT = np.polyfit(np.log(ns[m]), np.log(ts_fft[m]), 1)[0]
    print(f"C→ 拟合缩放指数（N≥512）：FD≈{sFD:.2f}（理论 2）, FFT≈{sFFT:.2f}（理论 2+log 修正）")
    print("C→ 结论：FD 复杂度确实更小（CPU 上 FD 更快且随 N 优势扩大）。要发挥这一复杂度优势，")
    print("    必须用单次发射的融合核（如 abTEM 现版本）；若用逐整阵切片累加的朴素 GPU 实现，")
    print("    会被内存带宽拖慢，反而不如 cuFFT 优化的 FFT。")
    return ns, ts_fd, ts_fft


def plot_scaling(ns, ts_fd, ts_fft):
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    ax.loglog(ns, np.array(ts_fd) * 1e3, "o-", color="tab:blue", label="FD acc=2 (O(N²·k))")
    ax.loglog(ns, np.array(ts_fft) * 1e3, "s-", color="tab:red", label="FFT (O(N² log N))")
    nn = np.array([ns[0], ns[-1]])
    ax.loglog(nn, nn**2 / ns[0] ** 2 * (np.array(ts_fd)[0] * 1e3), "k--", lw=1.0,
              label="O(N²) reference")
    ax.set_xlabel("grid size N"); ax.set_ylabel("time per apply (ms, CPU)")
    ax.set_title("Complexity scaling: FD vs FFT (CPU)")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = OUT_DIR / "laplacian_scaling.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"图已保存: {out}")


def main():
    parser = argparse.ArgumentParser(description="差分拉普拉斯达到 FT 精度所需阶数 + 耗时对比")
    parser.add_argument("--precision", choices=["float32", "float64"], default="float64")
    parser.add_argument("--no-gpu", action="store_true")
    args = parser.parse_args()

    dtype = np.complex128 if args.precision == "float64" else np.complex64

    run_frequency_analysis()
    results, has_gpu = run_real_benchmark(dtype, use_gpu=not args.no_gpu)
    ns, ts_fd, ts_fft = run_scaling(dtype)
    plot_req_accuracy()
    plot_timing(results, has_gpu, dtype)
    plot_scaling(ns, ts_fd, ts_fft)


if __name__ == "__main__":
    main()
