# =============================================================================
# 拉普拉斯算符（Laplacian operator）数值精度验证
#
# 目的
# ----
# abTEM 实空间多切片在背散射 / 高阶展开里用 `abtem/finite_difference.py` 中的
#   可分离 2D 拉普拉斯模板。本脚本独立验证这套模板本身的数值精度，**不涉及
#   multislice 传播、背散射修正项或任何物理测量**。
#
# 验证内容
# --------
#   1) 不同差分阶数的收敛阶：abTEM 的 `finite_difference_coefficients(2, accuracy)`
#      提供 accuracy = 2/4/6/8/... 的中心差分二阶导权重，对应理论截断误差
#      O(Δx^accuracy)。本脚本对 accuracy ∈ {2,4,6,8} 逐階扫描网格采样 Δx，
#      记录有限差分与解析解/谱解析解的误差，拟合双对数斜率，验证其是否等于该阶数。
#   2) 傅里叶谱解析方案（benchmark / ground truth）：倒空间 ∇^2 -> -(2πk)^2，
#      对同一函数做 FFT -> 乘 -k^2 -> IFFT 得谱拉普拉斯。周期性测试函数下其误差
#      停留在机器精度水平，可作为评定有限差分精度的基准。
#   3) CPU 与 GPU 实现一致性：差异应仅为浮点舍入，数量级在机器精度附近。
#   4) 机器精度地板：高阶数在细网格处受舍入限制而停止下降（见 --dtype-check）。
#
# 复用 abTEM 的 stencil 生成逻辑
# ----------------------------
#   见 `abtem/finite_difference.py`（F:\papers\backscatter\abTEM\abtem\finite_difference.py）
#     * `finite_difference_coefficients(2, accuracy)`    : 1D 二阶中心差分系数（随阶数变宽）
#     * `_laplace_operator_stencil(accuracy, prefactor, mode, dtype, device)`:
#         返回可直接调用的 stencil 函数，内部按 accuracy 生成系数、按 mode 做边界填充、
#         再按 device 分派：
#           - device="cpu"  -> `_laplace_stencil_cpu_batch`（@njit(parallel,fastmath)）
#           - device="gpu"  -> `_laplace_stencil_gpu`（cupy 平移切片等价实现）
#     * `prefactor` = 1/∏sampling = 1/Δx^2（各向同性网格，与 `Waves.sampling` 一致）。
#
# 独立性
# ------
#   纯 NumPy + 可选 CuPy，不依赖 paper/scripts 下其它脚本的输出文件，可在 CI/本地运行。
#   matplotlib 使用 Agg 后端，仅保存图片（headless 友好）。
#
# 用法
# ----
#   python test_laplacian_precision.py                 # 默认 float64，跑全部阶数与测试函数
#   python test_laplacian_precision.py --precision float32
#   python test_laplacian_precision.py --no-gpu
#   python test_laplacian_precision.py --dtype-check --csv data.csv
#
# 输出
# ----
#   1) 控制台打印每个采样/每组阶数下的误差数据表；
#   2) `paper/figures/laplacian_precision.png`：log-log 误差曲线（每测试函数一张子图，
#      内含各阶数 FD 曲线 + 傅里叶基准线）；
#   3) `paper/figures/laplacian_curves.png`：原始函数 vs 拉普拉斯计算曲线的 1D 切片对比；
#   4) `paper/figures/laplacian_spectral.png`：倒空间有效波数对比（复现 Wacker 2015, Fig. 2）。
# =============================================================================
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# headless 友好：先切后端再 import pyplot
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# abTEM 的实际拉普拉斯模板（可分离 stencil 生成 + CPU/GPU 分派 + 周期性边界）
from abtem.finite_difference import (  # noqa: E402
    _laplace_operator_stencil,
    finite_difference_coefficients,
)

# 输出目录与其它论文脚本一致：figure 存到 paper/figures
OUT_DIR = Path(__file__).resolve().parent.parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

# 网格各向同性：L 为周期域边长（长度单位，对 abTEM 即 Å），dx = dy = L / N
DOMAIN_LEN = 4.0
N_LIST = [32, 64, 128, 256, 512]   # 每步翻倍，Δx 减半 -> 用以观察收敛阶
ACCURACIES = [2, 4, 6, 8]          # 中心差分精度阶（理论截断误差 O(Δx^accuracy)）默认值

# 各阶数在曲线中用到的标记符号（足够多，以覆盖 acc=18 等更高阶的 9~10 个阶数）
_MARKERS = ["o", "s", "^", "D", "v", "P", "*", "X", "p", "h", "H", "+", "x", "<", ">"]

# 说明：差分阶数 accuracy 的 1D 模板点数 = accuracy+1（如 acc=18 为 19 点）；
# 2D 分离式拉普拉斯模板点数 = 2*accuracy+1（对应 Wacker & Schröder (2015) Table 2
# 的 5Pt/9Pt/13Pt/17Pt... 记法，acc=18 即 37Pt）。
# abTEM 预计算表精确支持 accuracy ∈ {2,4,6,8,10,12,14,16,18}；accuracy>18 时
# `finite_difference_coefficients` 会用 sympy 动态生成系数（需已安装 sympy）。
# 实际测试的阶数列表再由 main() 依据 --accs / --max-acc 动态确定，默认 [2,4,6,8]。


# -----------------------------------------------------------------------------
# 1) 测试函数注册表：返回 (f, laplacian_exact)，均可解析写出二阶导
# -----------------------------------------------------------------------------
def _cosine_plane(domain_len, mx, my):
    """周期余弦平面波 f=cos(kx x)cos(ky y)，∇^2 f = -(kx^2+ky^2) f。"""
    kx, ky = 2 * np.pi * mx / domain_len, 2 * np.pi * my / domain_len

    def fn(X, Y):
        f = np.cos(kx * X) * np.cos(ky * Y)
        lap = -(kx * kx + ky * ky) * f
        return f, lap

    return fn


def _complex_plane_wave(domain_len, mx, my):
    """周期复平面波 f=exp(i k·r)，∇^2 f = -(kx^2+ky^2) f。"""
    kx, ky = 2 * np.pi * mx / domain_len, 2 * np.pi * my / domain_len

    def fn(X, Y):
        f = np.exp(1j * (kx * X + ky * Y))
        lap = -(kx * kx + ky * ky) * f
        return f, lap

    return fn


def _sum_of_modes(domain_len):
    """多个不同频率周期模式叠加（整数倍频 -> 周期），得到“任意样”场。"""
    kx1, ky1 = 2 * np.pi * 2 / domain_len, 2 * np.pi * 5 / domain_len
    kx2, ky2 = 2 * np.pi * 7 / domain_len, 2 * np.pi * 3 / domain_len
    kx3, ky3 = 2 * np.pi * 11 / domain_len, 2 * np.pi * 13 / domain_len

    def fn(X, Y):
        t1 = np.cos(kx1 * X) * np.cos(ky1 * Y)
        t2 = np.sin(kx2 * X) * np.cos(ky2 * Y)
        t3 = np.cos(kx3 * X) * np.cos(ky3 * Y)
        f = 0.5 * t1 + 0.3 * t2 + 0.2 * t3
        lap = (
            -0.5 * (kx1 * kx1 + ky1 * ky1) * t1
            - 0.3 * (kx2 * kx2 + ky2 * ky2) * t2
            - 0.2 * (kx3 * kx3 + ky3 * ky3) * t3
        )
        return f, lap

    return fn


def _gaussian_wavepacket(domain_len, sigma_ratio):
    """高斯波包 f=exp(-r^2/(2σ^2))，∇^2 f = (r^2/σ^4 - 2/σ^2) f。
    中心置于域中心；当 σ << L 时边界处函数值趋于 0，周期填充带来的误差可忽略，
    因而傅里叶谱解析法可作基准（误差仅由周期折叠与 FFT 数值精度决定，非周期函数
    精度不如周期函数，本脚本将其视作补充测试）。"""
    sigma = domain_len * sigma_ratio
    x0 = y0 = domain_len / 2.0

    def fn(X, Y):
        r2 = (X - x0) ** 2 + (Y - y0) ** 2
        f = np.exp(-r2 / (2 * sigma * sigma))
        lap = (r2 / sigma**4 - 2.0 / sigma**2) * f
        return f, lap

    return fn


# name -> (factory, is_periodic, description)
TEST_FUNCTIONS = [
    ("cosine_plane_m3_2", _cosine_plane(DOMAIN_LEN, 3, 2), True, "cos(3·2πx/L)cos(2·2πy/L)，周期、单频"),
    ("complex_plane_wave_m2_5", _complex_plane_wave(DOMAIN_LEN, 2, 5), True, "exp(i·(2,5)·2π r/L)，周期、复值"),
    ("sum_of_modes", _sum_of_modes(DOMAIN_LEN), True, "多个整数倍频模式叠加，周期"),
    ("gaussian_wavepacket", _gaussian_wavepacket(DOMAIN_LEN, 0.08), False, "高斯波包，局部化但非周期"),
]


# -----------------------------------------------------------------------------
# 2) 傅里叶谱解析拉普拉斯：∇^2 -> -(2πk)^2（周期性边界，无截断误差）
# -----------------------------------------------------------------------------
def laplacian_fft(f, dx):
    """对周期函数 f 做谱拉普拉斯，返回实空间逆变换结果。

    fftfreq(N, d=dx) 给出 1/(长度单位) 的循环频率，乘以 2π 得角波数 k；
    -(2πk)^2 即傅里叶域中 ∇^2 的特征值（负半定）。
    """
    fh = np.fft.fft2(f)
    kx = np.fft.fftfreq(f.shape[1], d=dx)
    ky = np.fft.fftfreq(f.shape[0], d=dx)
    k2 = (2 * np.pi * kx)[None, :] ** 2 + (2 * np.pi * ky)[:, None] ** 2
    return np.fft.ifft2(-k2 * fh)


# -----------------------------------------------------------------------------
# 3) 复用 abTEM 的有限差分拉普拉斯模板（带缓存，避免每函数重复 numba 编译）
# -----------------------------------------------------------------------------
_stencil_cache = {}


def get_stencil(accuracy, dx, device, dtype):
    """按 (accuracy, dx, device, dtype) 缓存 abTEM stencil 函数。

    说明：abTEM 把 prefactor（=1/∏sampling=1/Δx^2）乘进差分系数，各向同性网格下
    为 1/Δx^2，与 `Waves.sampling` 一致，这里用同约定完全复用其系数生成与分派。

    一个坑：CPU 的 numba 内核在 `_laplace_stencil_cpu_batch` 里写 `cumul = dtype(0.0)`
    （finite_difference.py:216），它要求 `dtype` 是【可调用的标量类型类】（如
    np.complex128），而数组的 `f.dtype` 返回的是 `numpy.dtype` 实例，nopython 下不可
    调用（报 "No type info available ... as a callable"）。故这里统一用标量类型类。
    """
    key = (accuracy, dx, device, dtype)
    if key not in _stencil_cache:
        dtype = np.dtype(dtype).type  # 标量类型类（numba 可用），勿直接传 numpy.dtype 实例
        prefactor = 1.0 / dx**2
        _stencil_cache[key] = _laplace_operator_stencil(
            accuracy, prefactor, mode="wrap", dtype=dtype, device=device
        )
    return _stencil_cache[key]


# -----------------------------------------------------------------------------
# 4) 误差度量：L∞ 与 L2 相对误差
# -----------------------------------------------------------------------------
def rel_linf(err, ref):
    """L∞ 相对误差：max|err| / max|ref|。（err、ref 均可为复数，模取 max）"""
    denom = float(np.max(np.abs(ref)))
    return float(np.max(np.abs(err)) / (denom if denom > 0 else 1e-300))


def rel_l2(err, ref):
    """L2 相对误差：||err||_2 / ||ref||_2。"""
    denom = float(np.linalg.norm(ref))
    return float(np.linalg.norm(err) / (denom if denom > 0 else 1e-300))


# -----------------------------------------------------------------------------
# 5) 在某个 (accuracy, N) 网格上对所有测试函数做一次采样（stencil 复用）
# -----------------------------------------------------------------------------
def evaluate_on_grid(accuracy, N, dtype, use_gpu):
    """对全部测试函数在同一 (accuracy, N) 网格上采样，返回每函数的一步误差字典。

    网格与 abTEM stencil 只构建一次，再应用到各个测试函数上，从而避免每函数
    重复触发 numba 编译（编译开销由 (accuracy, Δx, dtype, device) 决定）。
    """
    dx = DOMAIN_LEN / N
    x = np.linspace(0, DOMAIN_LEN, N, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="xy")

    stencil_cpu = get_stencil(accuracy, dx, "cpu", dtype)
    stencil_gpu = get_stencil(accuracy, dx, "gpu", dtype) if use_gpu else None

    out = []
    for name, fn, is_periodic, _desc in TEST_FUNCTIONS:
        f, exact = fn(X, Y)
        f = f.astype(dtype)
        exact = exact.astype(dtype)

        fft_lap = laplacian_fft(f, dx)                 # 傅里叶谱解析（ground truth）
        fd_cpu = stencil_cpu(f)                        # 有限差分 CPU

        fd_gpu = None
        if stencil_gpu is not None:
            import cupy as cp

            fd_gpu = cp.asnumpy(stencil_gpu(cp.asarray(f))).astype(dtype)

        rec = {
            "func": name,
            "periodic": is_periodic,
            "acc": accuracy,
            "acc_width": accuracy + 1,  # 1D 模板点数
            "N": N,
            "dx": dx,
            "err_cpu_vs_exact": rel_linf(fd_cpu - exact, exact),   # 有限差分 vs 解析解
            "err_fft_vs_exact": rel_linf(fft_lap - exact, exact),  # 傅里叶方法 vs 解析解
            "err_cpu_vs_fft": rel_linf(fd_cpu - fft_lap, fft_lap), # 有限差分 vs 傅里叶基准
            # L2 相对误差（表格/绘图可选用 L2，这里一并记录）
            "l2_cpu_vs_exact": rel_l2(fd_cpu - exact, exact),
        }
        if fd_gpu is not None:
            rec["err_gpu_vs_exact"] = rel_linf(fd_gpu - exact, exact)
            rec["cpu_gpu_diff"] = rel_linf(fd_cpu - fd_gpu, fft_lap)
        out.append(rec)
    return out


# -----------------------------------------------------------------------------
# 6) 收敛阶拟合：只在“截断误差主导”（明显高于机器地板）的采样点上做 log-log 线性拟合
# -----------------------------------------------------------------------------
def fit_slope(dx, errs, dtype):
    """返回 (slope, n_fit, floor_opts)。slope≈accuracy 即符合理论收敛阶。

    高阶数在细网格处误差会被舍入地板封顶（地板的数值量级：float64≈1e-12、
    float32≈1e-7），此时误差不再按 O(Δx^accuracy) 下降。为得到真实的收敛阶，
    这里只用明显高于地板（>阈值）的采样点拟合；阈值按精度取经验值。
    """
    dx = np.asarray(dx, float)
    errs = np.asarray(errs, float)
    threshold = 1e-11 if np.dtype(dtype).itemsize == 16 else 1e-6  # complex128/64
    mask = errs > threshold
    if mask.sum() < 2:
        mask = errs > 0
    if mask.sum() < 2:
        return float("nan"), 0, mask.sum()
    slope = float(np.polyfit(np.log(dx[mask]), np.log(errs[mask]), 1)[0])
    return slope, int(mask.sum()), int((~mask).sum())  # slope, 拟合点数, 被地板点


# -----------------------------------------------------------------------------
# 7) 主流程
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="验证 abTEM 有限差分拉普拉斯算符的数值精度（不同阶数收敛阶 / 傅里叶基准 / CPU-GPU 一致性）"
    )
    parser.add_argument("--precision", choices=["float32", "float64"], default="float64",
                        help="计算精度（默认 float64 以隔离舍入误差、得到干净的 O(Δx^accuracy) 收敛）")
    parser.add_argument("--no-gpu", action="store_true", help="跳过 GPU（cupy）路径")
    parser.add_argument("--dtype-check", action="store_true",
                        help="额外对比 float32 与 float64 在各阶数下的误差下限（舍入地板）")
    parser.add_argument("--csv", type=str, default=None, help="可选：把数据表导出为 CSV 路径")
    parser.add_argument("--max-acc", type=int, default=None,
                        help=f"测试所有 ≤ 该值的偶数阶（如 --max-acc 18，支持到 18 及以上）")
    parser.add_argument("--accs", type=int, nargs="+", default=None,
                        help="自定义差分阶数列表（如 --accs 2 4 8 18），优先级高于 --max-acc")
    args = parser.parse_args()

    dtype = np.complex128 if args.precision == "float64" else np.complex64
    if args.accs is not None:
        accs = [a for a in args.accs if a > 0 and a % 2 == 0]
        if not accs:
            print("[error] --accs 需要给出正整数偶数阶", file=sys.stderr)
            sys.exit(1)
    else:
        max_acc = args.max_acc if args.max_acc is not None else max(ACCURACIES)
        accs = [a for a in range(2, max_acc + 1, 2)]

    use_gpu = (not args.no_gpu)
    if use_gpu:
        try:
            import cupy  # noqa: F401

            use_gpu = True
        except ImportError:
            print("[warn] cupy 不可用，跳过 GPU 路径", file=sys.stderr)
            use_gpu = False

    print("=" * 78)
    print("拉普拉斯算符数值精度验证")
    print(f"  域边长 L = {DOMAIN_LEN}, 采样 = {N_LIST}")
    print(f"  精度 = {args.precision} ({np.dtype(dtype).name}), 差分阶数 = {accs}")
    print(f"  各阶模板宽度(点数) = "
          f"{', '.join(f'acc={a}:{a + 1}pt' for a in accs)}")
    print(f"  理论收敛阶 = O(Δx^accuracy)")
    print(f"  GPU 路径: {'启用' if use_gpu else '关闭'}")
    print("=" * 78)

    # 计算：对所有 (accuracy, N) 采样，逐函数收集误差
    all_rows = []
    for acc in accs:
        for N in N_LIST:
            all_rows.extend(evaluate_on_grid(acc, N, dtype, use_gpu))

    # ---- 打印：按测试函数分组，给出 N×Δx 的误差矩阵（行=N，列=各阶数 + 傅里叶基准） ----
    func_names = [f[0] for f in TEST_FUNCTIONS]
    slopes = {}  # (func, acc) -> slope

    for fname in func_names:
        sub = [r for r in all_rows if r["func"] == fname]
        is_periodic = sub[0]["periodic"]
        desc = next(f[3] for f in TEST_FUNCTIONS if f[0] == fname)

        print("\n" + "-" * 78)
        print(f"测试函数: {fname}  [{desc}]")
        print("-" * 78)

        # 表头
        head = f"{'N':>5} {'Δx':>9} |"
        for a in accs:
            head += f" {'acc=' + str(a) + ' CPU/exact(∞)':>17} |"
        head += f" {'FFT/exact(∞)':>13}"
        print(head)
        print("-" * len(head))

        for N in N_LIST:
            row = next(r for r in sub if r["N"] == N)
            line = f"{N:>5} {row['dx']:>9.4f} |"
            for a in accs:
                rr = next(r for r in sub if r["N"] == N and r["acc"] == a)
                line += f" {rr['err_cpu_vs_exact']:>17.3e} |"
            line += f" {row['err_fft_vs_exact']:>13.3e}"
            print(line)

        # 收敛阶（对每个阶数用“截断主导”点拟合）
        print("收敛阶（log-log 拟合, 理论=阶数）:")
        for a in accs:
            r_a = [r for r in sub if r["acc"] == a]
            dxs = [r["dx"] for r in r_a]
            errs_cpu = [r["err_cpu_vs_exact"] for r in r_a]
            slope, n_fit, n_flr = fit_slope(dxs, errs_cpu, dtype)
            slopes[(fname, a)] = slope
            # CPU⇄GPU 一致性（该阶数下所有网格点中的最大相对差）
            diffs = [r["cpu_gpu_diff"] for r in r_a if "cpu_gpu_diff" in r]
            cg = f"{max(diffs):.2e}" if diffs else "n/a"
            print(
                f"   acc={a} (模板{a + 1}点): 斜率≈{slope:.2f}"
                f"  拟合点 {n_fit}/{len(r_a)}"
                f"  被地板掩蔽 {n_flr}" + (f"  max|CPU-GPU|={cg}" if use_gpu else "")
            )

        if is_periodic:
            fft_errs = np.array([r["err_fft_vs_exact"] for r in sub])
            print(
                f"C→ 傅里叶解析基准误差范围 [{fft_errs.min():.2e}, {fft_errs.max():.2e}]"
                f"（周期函数应为机器精度量级）"
            )
        else:
            print(
                "C→ 非周期函数受“周期折叠”地板限制：高阶差分到细网格后误差不再随 O(Δx^accuracy)"
                "下降（此处高阶斜率偏低属正常）。收敛阶应在周期函数上严格成立。"
            )

    # ---- 全局 CPU⇄GPU 一致性汇总 ----
    if use_gpu:
        diffs = [r["cpu_gpu_diff"] for r in all_rows if "cpu_gpu_diff" in r]
        best = min(diffs)
        worst = max(diffs)
        bits = np.dtype(dtype).itemsize * 8
        print("\n" + "=" * 78)
        print(
            f"CPU ⇄ GPU 一致性: 相对差范围 [{best:.2e}, {worst:.2e}]"
            f"（complex{bits} 机器精度 ≈ {np.finfo(dtype).eps:.1e}）"
        )
        print(
            "  应仅来自浮点舍入：CPU 用 numba fastmath、GPU 用 cupy 切片，"
            "运算顺序略有差异，故差异位于机器精度量级。"
        )
        print("=" * 78)

    # ---- float32 vs float64 误差地板对比（可选）----
    if args.dtype_check:
        print("\n" + "-" * 78)
        print("精度地板对比（代表函数: cosine_plane_m3_2，完整扫描，仅 CPU）")
        base = next(f[1] for f in TEST_FUNCTIONS if f[0] == "cosine_plane_m3_2")
        print(f"{'acc':>4} {'N':>5} {'Δx':>9} | {'float32 FD/exact(∞)':>18} {'float64 FD/exact(∞)':>18}")
        for a in accs:
            for N in N_LIST:
                r32 = evaluate_on_grid(a, N, np.complex64, False)
                r64 = evaluate_on_grid(a, N, np.complex128, False)
                v32 = next(r for r in r32 if r["func"] == "cosine_plane_m3_2")
                v64 = next(r for r in r64 if r["func"] == "cosine_plane_m3_2")
                print(
                    f"{a:>4} {N:>5} {DOMAIN_LEN / N:>9.4f} | "
                    f"{v32['err_cpu_vs_exact']:>18.3e} {v64['err_cpu_vs_exact']:>18.3e}"
                )
        print("-" * 78)
        print(
            "  预期：细网格处 float32 误差受舍入限制（~1e-7）而不再按 O(Δx^accuracy) 下降，"
            "且阶数越高越早触及地板；float64 可继续下降（地板 ~1e-12）。"
        )

    # ---- 数据导出（可选）----
    if args.csv:
        import csv

        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            if all_rows:
                w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
                w.writeheader()
                w.writerows(all_rows)
        print(f"\n数据表已导出: {args.csv}")

    # ---- 绘图 ----
    plot_result(all_rows, dtype, accs, slopes, use_gpu)   # 相对误差 vs Δx（多阶收敛）
    plot_curves(dtype, accs)                              # 原始函数 vs 拉普拉斯计算曲线
    plot_spectral(accs)                                   # 倒空间有效波数（复现 Wacker 2015 Fig.2）


# -----------------------------------------------------------------------------
# 8) 绘图：每测试函数一张子图，展示各阶数 FD 收敛曲线 + 傅里叶基准
# -----------------------------------------------------------------------------
def plot_result(all_rows, dtype, accs, slopes, use_gpu):
    func_names = list(dict.fromkeys(r["func"] for r in all_rows))
    n = len(func_names)

    # 为各阶数分配互异的颜色与标记
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(accs)))
    markers = _MARKERS[: len(accs)]

    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.6), squeeze=False)
    axes = axes[0]

    for ax, fname in zip(axes, func_names):
        sub = [r for r in all_rows if r["func"] == fname]
        is_periodic = sub[0]["periodic"]

        # 各阶数有限差分（CPU，与解析解比）
        for i, a in enumerate(accs):
            r_a = [r for r in sub if r["acc"] == a]
            dxs = np.array([r["dx"] for r in r_a])
            errs = np.array([r["err_cpu_vs_exact"] for r in r_a])
            slope = slopes.get((fname, a), float("nan"))
            ax.loglog(
                dxs, errs, marker=markers[i], color=colors[i], lw=1.6, ms=5,
                label=f"FD acc={a} ({a + 1}pt, slope≈{slope:.2f})",
            )

        # 傅里叶谱解析基准（vs 解析解）
        fft_errs = [r["err_fft_vs_exact"] for r in sub]
        dxs = np.array([r["dx"] for r in sub])
        ax.loglog(dxs, fft_errs, "k--", lw=1.0, alpha=0.7, label="FFT spectral (vs exact)")

        # CPU⇄GPU 一致性标注（若启用 GPU）
        if use_gpu:
            cg = [r["cpu_gpu_diff"] for r in sub if "cpu_gpu_diff" in r]
            if cg:
                ax.text(
                    0.02, 0.02, f"max|CPU−GPU| = {max(cg):.1e}",
                    transform=ax.transAxes, fontsize=7, color="0.3",
                    va="bottom", ha="left",
                )

        ax.set_xlabel("sampling Δx")
        ax.set_ylabel("relative error (L∞)")
        ax.set_title(f"{fname}" + ("" if is_periodic else "\n(non-periodic)"))
        ax.grid(True, which="both", ls=":", alpha=0.5)
        ax.legend(fontsize=7, loc="best")

    fig.suptitle(
        "Laplacian precision: FD accuracy orders vs FFT-spectral "
        f"(dtype={np.dtype(dtype).name}, accuracies={accs})",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = OUT_DIR / "laplacian_precision.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n图已保存: {out}")


# -----------------------------------------------------------------------------
# 9) 原始函数 vs 拉普拉斯计算曲线的对比图（一维中行切片）
# -----------------------------------------------------------------------------
def plot_curves(dtype, accs):
    """沿网格中行 (y=L/2) 取一维切片，对比原始函数与各方法算出的 ∇²f。

    左轴：原始测试函数 f(x)（复值取实部）；
    右轴：∇²f(x) 曲线——精确解析解(黑实线)、傅里叶谱解析(黑虚线)、
          以及各阶数有限差分（按阶着色）。
    选用最粗网格 N_LIST[0]，让低阶差分与真值的偏移足以被肉眼看出，
    直观呈现“阶数越高、曲线越贴合真值”的收敛效果。
    """
    N = N_LIST[0]
    dx = DOMAIN_LEN / N
    x = np.linspace(0, DOMAIN_LEN, N, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="xy")
    i = N // 2  # 中行切片

    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(accs)))
    markers = _MARKERS[: len(accs)]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    for ax, (fname, fn, is_periodic, desc) in zip(axes, TEST_FUNCTIONS):
        f, exact = fn(X, Y)
        f = f.astype(dtype)
        exact = exact.astype(dtype)
        fft = laplacian_fft(f, dx)

        # 左轴：原始函数（复值取实部）
        ax.plot(x, np.real(f[i]), color="0.35", lw=1.8, label="f(x) [original]")
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)  (original)", color="0.35")
        ax.tick_params(axis="y", labelcolor="0.35")
        ax.grid(True, ls=":", alpha=0.4)

        # 右轴：拉普拉斯曲线（精确/傅里叶/各阶有限差分）
        ax2 = ax.twinx()
        ax2.plot(x, np.real(exact[i]), "k-", lw=1.7, label="∇²f exact")
        ax2.plot(x, np.real(fft[i]), "k--", lw=1.0, alpha=0.7, label="∇²f FFT")
        for j, a in enumerate(accs):
            fd = get_stencil(a, dx, "cpu", dtype)(f)
            ax2.plot(
                x, np.real(fd[i]), marker=markers[j], ms=3.2, lw=1.0,
                color=colors[j], alpha=0.85, label=f"∇²f FD acc={a}",
            )
        ax2.set_ylabel("∇²f(x)  (computed)", color="0.15")
        ax2.tick_params(axis="y", labelcolor="0.15")

        ax.set_title(f"{fname}" + ("" if is_periodic else "\n(non-periodic)"))

        # 合并双侧图例
        h, l = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h + h2, l + l2, fontsize=6.5, loc="best", ncol=2)

    fig.suptitle(
        "Original function vs computed Laplacian (1D slice at y=L/2, "
        f"N={N}, dtype={np.dtype(dtype).name})",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = OUT_DIR / "laplacian_curves.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"图已保存: {out}")


# -----------------------------------------------------------------------------
# 10) 倒空间有效波数对比（复现 Wacker & Schröder 2015, Fig. 2）
# -----------------------------------------------------------------------------
def plot_spectral(accs):
    """在倒空间比较不同拉普拉斯模板的“有效波数”与精确 -κ² 抛物线。

    Wacker & Schröder (2015) Fig. 2：倒空间中拉普拉斯等价于乘 -κ²（FT 精确为抛物线），
    而实空间 XPt 模板只在低频接近抛物线，高频（κ→κ_N）处显露明显偏差；
    阶数越高、偏离越小。本函数用归一化频率 q=κ/κ_N 与归一化有效因子 M(q)=κ_eff²/κ_N²
    复现该图：M(q) 是让 有限差分拉普拉斯 ≈ -M(q)·(像素倒数尺度) 的系数。

    公式：对中心差分系数 c_j（j=-n..n，来自 finite_difference_coefficients(2, accuracy)），
    作用于平面波 e^{i κ x} 的特征值为 (1/Δx²)·Σ_j c_j e^{ijκΔx} = -κ_eff²，故
      κ_eff² = -(1/Δx²)·[c_0 + 2 Σ_{j≥1} c_j cos(jκΔx)]，而 κ_N=π/Δx、κΔx=πq，
      归一化得  M(q) = κ_eff²/κ_N² = -(1/π²)·[c_0 + 2 Σ_{j≥1} c_j cos(jπq)]。
    FT 精确解即 M(q)=q²（抛物线）；q→0 时各模板 M(q)→q²。
    """
    q = np.linspace(0, 1, 2001)
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(accs)))
    markers = _MARKERS[: len(accs)]

    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    ax.plot(q, q**2, "k-", lw=2.2, label="FT exact: κ_eff²/κ_N² = q² (parabola)")

    print("\n倒空间有效波数（复现 Wacker & Schröder (2015) Fig. 2）")
    print(f"{'acc':>4} {'2D点数':>6} {'M(q=1)':>10} {'max|M-q²|':>12} {'5%偏离q的位置q':>18}")
    for i, a in enumerate(accs):
        c = finite_difference_coefficients(2, a)
        n = len(c) // 2
        th = np.pi * q
        # M = -(1/π²)·[c_n + 2 Σ_{j=1..n} c_{n+j} cos(jπq)]
        M = -(1 / np.pi**2) * (c[n] + 2 * sum(c[n + j] * np.cos(j * th) for j in range(1, n + 1)))

        ax.plot(q, M, color=colors[i], lw=1.8, label=f"FD acc={a} ({2 * a + 1}Pt)")
        ax.plot(q[::50], M[::50], marker=markers[i], ms=3.2, ls="", color=colors[i], alpha=0.7)

        dev = np.abs(M - q**2)
        # 有效因子先偏离真实抛物线 5% 的归一化频率 q（衡量“可信分辨率上限”
        rel = np.abs(M - q**2) / np.maximum(q**2, 1e-12)
        idx = np.where(rel > 0.05)[0]
        q05 = float(q[idx[0]]) if len(idx) else float("nan")
        print(
            f"{a:>4} {2 * a + 1:>6} {M[-1]:>10.4f} "
            f"{dev.max():>12.4f} {('%.3f' % q05) if np.isfinite(q05) else '  >1':>18}"
        )

    ax.set_xlabel("normalized spatial frequency  κ/κ_N")
    ax.set_ylabel("effective wavenumber factor  κ_eff²/κ_N²")
    ax.set_title("Laplacian approximations in reciprocal space (cf. Wacker 2015, Fig. 2)")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(True, ls=":", alpha=0.5)
    fig.tight_layout()
    out = OUT_DIR / "laplacian_spectral.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"图已保存: {out}")


if __name__ == "__main__":
    main()
