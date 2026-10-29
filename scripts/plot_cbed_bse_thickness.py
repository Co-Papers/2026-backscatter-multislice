# =============================================================================
# 从 cbed_bse_{E}keV.npz 原始数据生成"厚度系列 × 4 列"对比图。
#   列 = 无背散射(log) / 有背散射(log) / |差值|(log) / 差值径向平均(log, mrad)
#   行 = 厚度系列。
# 这样改图（列数、配色、坐标）无需重跑多切片计算。用法：
#   python plot_cbed_bse_thickness.py [5] [10] [30] ...   # 指定能量；不给则所有 npz
# =============================================================================
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

out_dir = Path(__file__).resolve().parent.parent / "figures"


def apply_log(I, C=1.5e5):
    imax = max(float(np.max(np.abs(I))), 1e-30)
    return np.log(1.0 + C * np.abs(I) / imax)


def radial_annular_average(intensity, sampling, wavelength, bin_width=None):
    """径向（环形）平均：按不同半径分环取平均，再重 bin 到均匀角网格，空 bin 插值。
    theta_mrad = 1000 * wavelength * |k|。输入应是已 log 变换的强度。"""
    intensity = np.asarray(intensity, dtype=float)
    Ny, Nx = intensity.shape
    ky = (np.arange(Ny) - (Ny - 1) / 2) * sampling[0]
    kx = (np.arange(Nx) - (Nx - 1) / 2) * sampling[1]
    K = np.sqrt(kx[None, :] ** 2 + ky[:, None] ** 2)
    theta = (1000.0 * wavelength * K).ravel()
    I = intensity.ravel()
    key = np.round(theta, 3)
    uniq = np.unique(key)
    idx = np.searchsorted(uniq, key)
    cnt = np.bincount(idx, minlength=uniq.size)
    tot = np.bincount(idx, weights=I, minlength=uniq.size)
    prof_uniq = tot / np.maximum(cnt, 1)
    if bin_width is None:
        bin_width = 1000.0 * wavelength * float(sampling[0])
    edges = np.arange(uniq.min(), uniq.max() + bin_width, bin_width)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bidx = np.clip(np.digitize(uniq, edges) - 1, 0, centers.size - 1)
    bcnt = np.bincount(bidx, minlength=centers.size)
    btot = np.bincount(bidx, weights=prof_uniq, minlength=centers.size)
    profile = btot / np.maximum(bcnt, 1)
    empty = bcnt == 0
    if empty.any() and (~empty).any():
        profile[empty] = np.interp(centers[empty], centers[~empty], profile[~empty])
    return centers, profile


def plot_from_npz(npz_path):
    d = np.load(npz_path)
    energy = int(np.asarray(d["energy_kev"]))
    thickness = np.asarray(d["thickness_angstrom"])
    cb_ref = np.asarray(d["cbed_without"])
    cb_bse = np.asarray(d["cbed_with"])
    diff = np.asarray(d["cbed_diff"])
    sampling = np.asarray(d["sampling"])
    wl = float(np.asarray(d["wavelength"]))
    ext = np.abs(np.asarray(d["extent"]))
    dx, dy = ext[0], ext[1]
    n_exit = thickness.shape[0]

    log_ref = apply_log(cb_ref)
    log_bse = apply_log(cb_bse)
    log_diff = apply_log(diff)

    cols = ["no backscattering (log)", "with backscattering effect (log)",
            "|with - without| (log)", "radial avg of |diff| (log)"]
    fig, axes = plt.subplots(n_exit, 4, figsize=(16, n_exit * 2.6))
    for i in range(n_exit):
        for j, (arr, t) in enumerate([(log_ref[i], cols[0]),
                                      (log_bse[i], cols[1]),
                                      (log_diff[i], cols[2])]):
            ax = axes[i, j]
            im = ax.imshow(arr, extent=(-dx, dx, -dy, dy), origin="lower", cmap="inferno")
            ax.set_xlabel("kx [1/Å]"); ax.set_ylabel("ky [1/Å]")
            if i == 0:
                ax.set_title(t, fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="log intensity")
        rax = axes[i, 3]
        r_diff, ra_diff = radial_annular_average(log_diff[i], sampling, wl)
        rax.plot(r_diff, ra_diff)
        rax.set_xlabel("scattering angle [mrad]"); rax.set_ylabel("log intensity")
        rax.grid(alpha=0.3)
        if i == 0:
            rax.set_title(cols[3], fontsize=10)
        rax.set_ylabel(f"t={thickness[i]:.0f} Å\nlog intensity", fontsize=9)

    fig.suptitle(f"{energy} keV CBED thickness series (1 phonon config)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_png = out_dir / f"cbed_bse_{energy}keV.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_png}  (rows={n_exit}, max|diff| per row="
          f"{[f'{float(np.asarray(d['cbed_diff'])[i].max()):.1e}' for i in range(n_exit)]})", flush=True)
    return out_png


if __name__ == "__main__":
    energies = [int(a) for a in sys.argv[1:]]
    if energies:
        files = [out_dir / f"cbed_bse_{e}keV.npz" for e in energies]
    else:
        files = sorted(glob.glob(str(out_dir / "cbed_bse_*keV.npz")))
    for f in files:
        if Path(f).exists():
            plot_from_npz(str(f))
        else:
            print(f"  skip (missing): {f}", flush=True)
    print("done", flush=True)
