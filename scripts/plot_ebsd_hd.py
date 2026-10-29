# =============================================================================
# 高清版本：Mo 全参数 EBSD 菊池带 (4096x4096, 8a x 64c, ppc=512, slice=0.05A)
#   从 npz 数据重画，dpi=300，标注布拉格反射 + 径向峰位定量
# =============================================================================
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm

d = np.load(r'F:\papers\backscatter\paper\figures\ebsd_fourier_fullparam.npz')
Fc = d['Fc'].astype(np.float32)
kc = d['kc']; wl = float(d['wl']); a = float(d['a'])
sampling = float(d['sampling']); inplane = int(d['inplane'])
z_repeat = int(d['z_repeat']); ppc = int(d['ppc'])
print(f"Fc[{Fc.shape}]  kc[±{abs(kc[0]):.2f}]  wl={wl:.5f}  a={a:.4f}")

def apply_log(I, C=1.5e5):
    imax = max(float(np.max(np.abs(I))), 1e-30)
    return np.log(1.0 + C * np.abs(I) / imax)

IM = apply_log(Fc)

# ---- 径向剖面 & 峰位 ----
Xg, Yg = np.meshgrid(kc, kc); R = np.sqrt(Xg ** 2 + Yg ** 2)
nb = 300; edges = np.linspace(0, R.max(), nb + 1)
dig = np.digitize(R.ravel(), edges)
prof = np.array([Fc.ravel()[dig == b].mean() if (dig == b).any() else 0 for b in range(1, nb + 1)])
rb = 0.5 * (edges[1:] + edges[:-1])
th = np.degrees(np.arcsin(np.clip(wl * rb, 0, 0.999)))
# 局部极大峰
pk = np.where((prof[1:-1] > prof[:-2]) & (prof[1:-1] > prof[2:]))[0] + 1
pk = pk[prof[pk] > 0]  # 只保留有意义处
# 只取前若干个峰，按强度排序取显著
order = pk[np.argsort(-prof[pk])][:14]
order = order[np.argsort(rb[order])]  # 按半径排序

# ---- bcc Mo [001] 理论布拉格反射 g=(h,k) 沿象限 ----
# 允许 h,k 全偶或全奇(体心立方)；|g|(1/A) 用倒格矢 1/a(单位? abTEM k 是 cycles/A)
# bcc [001] 允许反射: h+k 偶数。最低: (110)->|g|=sqrt(2)/a, (200)->2/a, (211)->sqrt(6)/a, (220)->sqrt(8)/a
allowed = [(1,1),(2,0),(2,2),(3,1),(4,0),(4,2),(3,3)]
theo = [np.sqrt(h*h+k*k)/a for h,k in allowed]
theo_th = np.degrees(np.arcsin(np.clip(np.array(theo)*wl, 0, 0.999)))
print("理论布拉格反射 (bcc [001]):")
for (h,k),g,tt in zip(allowed, theo, theo_th):
    print(f"  ({h}{k})  |g|={g:.4f} 1/A  theta={tt:.2f} deg")

# ---- 高清图：3 面板【菊池带总览 | 中心布拉格环放大 | 径向剖面】----
fig = plt.figure(figsize=(15, 6.5))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.0], wspace=0.32)
ax1 = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1]); ax3 = fig.add_subplot(gs[2])

cmap = 'inferno'
# 左：菊池带总览 |k|<12
im = ax1.imshow(IM, extent=(kc[0], kc[-1], kc[0], kc[-1]), origin='lower', cmap=cmap)
ax1.set_xlabel(r'$k_x$ [1/$\mathrm{\AA}$]', fontsize=12)
ax1.set_ylabel(r'$k_y$ [1/$\mathrm{\AA}$]', fontsize=12)
ax1.set_title('EBSD Kikuchi (full, |k|<12 1/AA)', fontsize=12)
ax1.set_aspect('equal')
cb = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
cb.set_label(r'log$_{10}$ intensity', fontsize=11)

# 中：中心放大 |k|<3，标注理论布拉格环
scale = 3.0
mask3 = np.abs(kc) <= scale
ix = np.where(mask3)[0]
Fc3 = Fc[ix[0]:ix[-1]+1, ix[0]:ix[-1]+1]; kc3 = kc[ix]
IM3 = apply_log(Fc3)
ax2.imshow(IM3, extent=(kc3[0], kc3[-1], kc3[0], kc3[-1]), origin='lower', cmap=cmap)
for (h,k), g in zip(allowed, theo):
    if g <= scale:
        ax2.add_patch(plt.Circle((0,0), g, fill=False, color='cyan', lw=1.1, ls='--', alpha=0.9))
        ax2.annotate(f'({h}{k})', (g/np.sqrt(2), g/np.sqrt(2)), color='cyan', fontsize=9,
                     ha='left', va='bottom')
ax2.set_xlabel(r'$k_x$ [1/$\mathrm{\AA}$]', fontsize=12)
ax2.set_title('Center, |k|<3 (Bragg rings marked)', fontsize=12)
ax2.set_aspect('equal')

# 右：径向剖面
ax3.semilogy(th, prof + 1e-12, lw=1.6, color='tab:blue')
for i in order:
    ax3.axvline(th[i], color='tab:red', ls=':', lw=0.6)
ax3.plot(th[order], prof[order] + 1e-12, 'o', ms=4, color='tab:red', label='measured peaks')
for (h,k), tt in zip(allowed, theo_th):
    ax3.axvline(tt, color='green', ls='--', lw=0.9)
ax3.set_xlabel(r'$\theta=\mathrm{asin}(\lambda|k|)$ [deg]', fontsize=12)
ax3.set_ylabel('mean intensity (log)', fontsize=12)
ax3.set_title('Radial profile (red=measured, green=Bragg)', fontsize=12)
ax3.grid(True, which='both', ls=':', alpha=0.4)
ax3.set_xlim(0, 16)

fig.suptitle('Mo [001] EBSD Kikuchi - Fourier, point source z=12.6 A  '
             '( 8a x 64c, ppc=512, slice=0.05 A, 20 kV )', fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = r'F:\papers\backscatter\paper\figures\ebsd_fourier_fullparam_hd.png'
fig.savefig(out, dpi=300, bbox_inches='tight')
print('saved', out)

# 保存径向剖面
np.savez(r'F:\papers\backscatter\paper\figures\ebsd_fourier_fullparam_hd.npz',
         Fc=Fc, kc=kc, prof=prof, rb=rb, th=th, wl=wl, a=a,
         theo_g=np.array(theo), theo_th=theo_th)
print("峰位(deg):", [f'{th[i]:.2f}' for i in order[:10]])
