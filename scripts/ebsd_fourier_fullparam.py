# =============================================================================
# Mo [001] EBSD 菊池带：用户指定全参数 + Fourier(Fresnel) 传播子（稳定）
#   参数：8a x 8a 面内、64c 厚度、ppc=512（网格 4096x4096）、片层 0.05 A
#   传播：单深度点源 -> conventional_multislice_step(Fresnel/FFT) -> 出口倒空间
#   输出：|k|<12 1/A 窗口高清菊池带图 + 径向剖面
# =============================================================================
import numpy as np, ase.io, abtem, time
from abtem import Waves
from abtem.multislice import conventional_multislice_step, FresnelPropagator, AntialiasAperture

CELL = ase.io.read(r'F:\papers\backscatter\paper\atomic_model\Mo.cif')
a = float(CELL.cell.lengths()[0])

IN_PLANE = 8      # 面内 8 个晶胞
Z_REPEAT = 64     # 厚度 64c
PPC = 128         # 每晶胞采样点
SLICE_THICKNESS = 0.1 # in A
SIGMA_ANG = 0.01
VOLTAGE = 20e3
KMAX_OUT = 12.0
Z_SOURCE_ANG = 10    # 点源深度（近表面交互体积，保持之前一致）

N = IN_PLANE * PPC
extent = IN_PLANE * a
sampling = a / PPC
thickness = Z_REPEAT * a
supercell = CELL.repeat((IN_PLANE, IN_PLANE, Z_REPEAT))
supercell.set_array('debye_waller', np.full(len(supercell), 0.5))
print(f"grid {N}x{N}  extent {extent:.2f}A  sampling {sampling:.5f}A  "
      f"thick {thickness:.1f}A({Z_REPEAT}c)  slice {SLICE_THICKNESS}A", flush=True)

pot = abtem.Potential(supercell, parametrization='lobato', sampling=sampling,
                      slice_thickness=SLICE_THICKNESS, projection='finite', device='gpu')
ns = pot.num_slices
wl = abtem.PlaneWave(energy=VOLTAGE, sampling=sampling, extent=(extent, extent)).accelerator.wavelength
print(f"num_slices={ns}  wl={wl:.5f}A", flush=True)

prop = FresnelPropagator()
antialias = AntialiasAperture()
z_slice = max(0, int(round(Z_SOURCE_ANG / SLICE_THICKNESS)))

import cupy as cp
x = np.linspace(0, extent, N, endpoint=False)
X, Y = np.meshgrid(x.astype(np.float32), x.astype(np.float32), indexing='xy')
r2 = (X - extent / 2) ** 2 + (Y - extent / 2) ** 2
psi0 = cp.exp(-cp.asarray(r2) / (2 * SIGMA_ANG ** 2)).astype(cp.complex128)
psi0 /= cp.sqrt(cp.abs(psi0) ** 2).sum()
del X, Y, r2
print(f"point-Src z={Z_SOURCE_ANG}A (slice {z_slice})  Fresnel/Fourier  (流式切片)", flush=True)

w = Waves(psi0, sampling=sampling, extent=(extent, extent), energy=VOLTAGE)
t0 = time.perf_counter()
# 流式迭代生成器：逐切片传播，用后即释放，避免 list 物化导致 OOM
for i, sl in enumerate(pot.generate_slices()):
    if i < z_slice:
        continue
    w = conventional_multislice_step(w, sl, propagator=prop,
                                     antialias_aperture=antialias, order='exact')
    del sl
    if (i - z_slice) % 400 == 0:
        print(f"  slice {i}/{ns}  {time.perf_counter()-t0:.0f}s", flush=True)
print(f"propagate done {time.perf_counter()-t0:.0f}s", flush=True)
arr = np.asarray(w.array.get())
F = np.abs(np.fft.fftshift(np.fft.fft2(arr))) ** 2
F /= max(float(F.max()), 1e-30)

dk = 1.0 / extent
kk = (np.arange(N) - (N - 1) / 2) * dk
mask = np.abs(kk) <= KMAX_OUT
idx = np.where(mask)[0]
Fc = F[idx[0]:idx[-1] + 1, idx[0]:idx[-1] + 1]
kc = kk[idx]

np.savez(r'F:\papers\backscatter\paper\figures\ebsd_fourier_fullparam.npz',
         F=F.astype(np.float16), Fc=Fc.astype(np.float16), kk=kk, kc=kc,
         wl=wl, extent=extent, sampling=sampling, a=a,
         inplane=IN_PLANE, z_repeat=Z_REPEAT, ppc=PPC)
def apply_log(I, C=1.5e5):
    """仅用于显示的 log 压缩：ln(1+C·I/I_max)。
    【口径】这只用于下面 n_exit×4 网格图的 imshow/径向平均展示，
    **从不写进 npz**——np.savez 存的 cbed_without / cbed_with / cbed_diff
    全是**原始强度**（|with−without| 在原始强度上取）。
    度量背散射效应必须用原始强度：若在 log 域做差，
    log(with)−log(without) 不再对应物理强度变化，百分比会失真。
    面板 a–d 那种"log 轴"只是展示层（set_yscale("log")），曲线形状等于把
    度量值取 log 后画在线性轴上，但计算度量的原始强度未做变换。"""
    imax = max(float(np.max(np.abs(I))), 1e-30)
    return np.log(1.0 + C * np.abs(I) / imax)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
im = axes[0].imshow(apply_log(F), extent=(kk[0], kk[-1], kk[0], kk[-1]),
                    origin='lower', cmap='inferno')
axes[0].set_xlabel('kx [1/A]'); axes[0].set_ylabel('ky [1/A]')
axes[0].set_title(f'Full {N}x{N}', fontsize=10)
fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label='log I')
im = axes[1].imshow(apply_log(Fc), extent=(kc[0], kc[-1], kc[0], kc[-1]),
                    origin='lower', cmap='inferno')
axes[1].set_xlabel('kx [1/A]'); axes[1].set_ylabel('ky [1/A]')
axes[1].set_title(f'|k|<{KMAX_OUT} 1/A', fontsize=10)
fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label='log I')
fig.suptitle(f'Mo [001] EBSD Kikuchi: {IN_PLANE}a x {Z_REPEAT}c, ppc={PPC}, {N}x{N}, '
             f'slice={SLICE_THICKNESS}A, 20kV, Fourier', fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
out = r'F:\papers\backscatter\paper\figures\ebsd_fourier_fullparam.png'
fig.savefig(out, dpi=200, bbox_inches='tight')
print('saved', out)
