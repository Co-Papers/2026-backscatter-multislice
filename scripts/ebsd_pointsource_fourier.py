# =============================================================================
# EBSD 菊池带（方案B：常规 FourierMultislice，稳定）—— 单深度点源 512x512
#   用 Fourier 传播子（FFT，无条件稳定）替代会发散的 real-space Laplace RRS
#   * 单深度点源(z~12.6 A) -> conventional_multislice_step(FFT) -> 出口倒空间 -> 高清菊池带
#   * 512x512 网格；|k| 窗口裁剪
# =============================================================================
import numpy as np, ase.io, abtem, time
from abtem import Waves
from abtem.multislice import conventional_multislice_step, FresnelPropagator, AntialiasAperture

CELL = ase.io.read(r'F:\papers\backscatter\paper\atomic_model\Mo.cif')
a = float(CELL.cell.lengths()[0])

SIGMA_ANG = 0.06
VOLTAGE = 20e3
Z_REPEAT = 16
SLICE_THICKNESS = 0.5
IN_PLANE = 2
PPC = 256                # 512x512
KMAX_OUT = 5.0
Z_SOURCE_ANG = 12.6

N = IN_PLANE * PPC
supercell = CELL.repeat((IN_PLANE, IN_PLANE, Z_REPEAT))
extent = IN_PLANE * a
sampling = a / PPC
supercell.set_array('debye_waller', np.full(len(supercell), 0.4))
pot = abtem.Potential(supercell, parametrization='lobato', sampling=sampling,
                      slice_thickness=SLICE_THICKNESS, projection='finite', device='gpu')
slices = list(pot.generate_slices()); ns = len(slices)
wl = abtem.PlaneWave(energy=VOLTAGE, sampling=sampling, extent=(extent, extent)).accelerator.wavelength
print(f'grid {N}x{N}  extent {extent:.2f}A  sampling {sampling:.5f}A  ns={ns}', flush=True)

propagator = FresnelPropagator()
antialias = AntialiasAperture()
z_slice = max(0, int(round(Z_SOURCE_ANG / SLICE_THICKNESS)))

import cupy as cp
x = np.linspace(0, extent, N, endpoint=False)
X, Y = np.meshgrid(x.astype(np.float32), x.astype(np.float32), indexing='xy')
r2 = (X - extent / 2) ** 2 + (Y - extent / 2) ** 2
psi0 = cp.exp(-cp.asarray(r2) / (2 * SIGMA_ANG ** 2)).astype(cp.complex128)
psi0 /= cp.sqrt(cp.abs(psi0) ** 2).sum()
del X, Y, r2
w = Waves(psi0, sampling=sampling, extent=(extent, extent), energy=VOLTAGE)
print(f'point-Src z={Z_SOURCE_ANG:.1f}A (slice {z_slice})  Fourier(Fresnel) propagator', flush=True)

t0 = time.perf_counter()
for i in range(z_slice, ns):
    w = conventional_multislice_step(
        w, slices[i], propagator=propagator, antialias_aperture=antialias,
        conjugate=False, transpose=False, order='exact')
    if (i - z_slice) % 25 == 0:
        print(f'  slice {i}/{ns}  {time.perf_counter()-t0:.0f}s', flush=True)
arr = np.asarray(w.array.get())
print(f'propagate done {time.perf_counter()-t0:.0f}s', flush=True)
F = np.abs(np.fft.fftshift(np.fft.fft2(arr))) ** 2
F /= max(float(F.max()), 1e-30)

dk = 1.0 / extent
kk = (np.arange(N) - (N - 1) / 2) * dk
mask = np.abs(kk) <= KMAX_OUT
idx = np.where(mask)[0]
Fc = F[idx[0]:idx[-1] + 1, idx[0]:idx[-1] + 1]
kc = kk[idx]

np.savez(r'F:\papers\backscatter\paper\figures\ebsd_pointsource_fourier_512.npz',
         F=F, Fc=Fc, kk=kk, kc=kc, wl=wl, extent=extent, sampling=sampling)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

def apply_log(I, C=1.5e5):
    imax = max(float(np.max(np.abs(I))), 1e-30)
    return np.log(1.0 + C * np.abs(I) / imax)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
im = axes[0].imshow(apply_log(F), extent=(kk[0], kk[-1], kk[0], kk[-1]),
                    origin='lower', cmap='inferno')
axes[0].set_xlabel('kx [1/A]'); axes[0].set_ylabel('ky [1/A]')
axes[0].set_title(f'Full {N}x{N}', fontsize=10)
fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label='log I')
im = axes[1].imshow(apply_log(Fc), extent=(kc[0], kc[-1], kc[0], kc[-1]),
                    origin='lower', cmap='inferno')
axes[1].set_xlabel('kx [1/A]'); axes[1].set_ylabel('ky [1/A]')
axes[1].set_title(f'|k|<{KMAX_OUT} 1/A (Kikuchi)', fontsize=10)
fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label='log I')
fig.suptitle(f'Mo [001] EBSD Kikuchi, point-source z={Z_SOURCE_ANG}A + Fourier propagator, '
             f'{N}x{N}, 20kV', fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
out = r'F:\papers\backscatter\paper\figures\ebsd_pointsource_fourier_512.png'
fig.savefig(out, dpi=200, bbox_inches='tight')
print('saved', out)
