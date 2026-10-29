# =============================================================================
# 方案X EBSD：用 abTEM Probe（=高斯/聚焦点源）构造样品内部点源
# 全厚多深度叠加 + RRS(RealSpace full) 传播到出射面 + 出口倒空间非相干叠加
# 验证产生连续菊池带（而非相干平面波的离散斑点）
#
# 物理：EBSD 电子 = 样品内非相干点源（Probe 聚焦波包）向出射方向 + 动力学散射。
# 交互体积内多个深度各放一点源，分别 RRS 传播，出口倒空间强度非相干叠加深度的模方。
# 一次点源传播覆盖所有方向（Probe 倒空间含全 k），无需逐方向扫描。
# =============================================================================
import numpy as np, ase.io, abtem, time
from abtem import Waves
from abtem.finite_difference import LaplaceOperator, multislice_step

BASE=r'F:\papers\backscatter\paper'
cell=ase.io.read(BASE+r'\atomic_model\Mo.cif'); a=float(cell.cell.lengths()[0])
inplane,z_repeat,ppc=2,16,32
supercell=cell.repeat((inplane,inplane,z_repeat)); extent=inplane*a; sampling=a/ppc; N=inplane*ppc
supercell.set_array('debye_waller',np.full(len(supercell),0.4))
pot=abtem.Potential(supercell,parametrization='lobato',sampling=sampling,
                    slice_thickness=0.1,projection='finite',device='gpu')
slices=list(pot.generate_slices()); ns=len(slices)
laplace=LaplaceOperator(6,method='finite_difference')
wl=abtem.PlaneWave(energy=20e3,sampling=sampling,extent=(extent,extent)).accelerator.wavelength
print(f'grid {N}x{N}  thick {z_repeat*a:.1f}A  ns={ns}  a={a:.4f}')

def apply_log(I, C=1.5e5):
    imax = max(float(np.max(np.abs(I))), 1e-30)
    return np.log(1.0 + C * np.abs(I) / imax)

# ---- 用 abTEM Probe 构造点源（聚焦波包），放到样品内深度 z_s，向出射面(+z)传播 ----
def propagate_pointsource_from(z_s):
    """在切片 z_s 处放一点源（Probe 聚焦波包），向 +z 传播到出射面，返回出口倒空间强度。"""
    import cupy as cp
    # 会聚半角足够大(>=30 mrad) → 点源倒空间展宽充分(能覆盖全方向) → 产生连续菊池带
    # 太小(<=10 mrad) → 倒空间几乎无展宽 → 退化为相干平面波 → 只有离散斑点
    pr=abtem.Probe(semiangle_cutoff=30.0, sampling=sampling, extent=(extent,extent), energy=20e3)
    psi0=np.asarray(pr.build().array).astype(np.complex128)
    psi0=cp.asarray(psi0)
    psi0/=cp.sqrt(cp.abs(psi0)**2).sum()   # 归一化
    w=Waves(psi0,sampling=sampling,extent=(extent,extent),energy=20e3)
    for i in range(z_s, ns):
        if i<ns-1:
            o=multislice_step(w,slices[i],next_slice=slices[i+1],laplace=laplace,max_terms=80,order=1,fully_corrected=True); w=o[0]
        else:
            o=multislice_step(w,slices[i],None,laplace=laplace,max_terms=80,order=1,fully_corrected=True); w=o[0] if isinstance(o,tuple) else o
    arr=np.asarray(w.array.get())
    F=np.abs(np.fft.fftshift(np.fft.fft2(arr)))  # 出口倒空间强度
    return F, arr

# ---- 全厚多深度非相干叠加 ----
n_depths=6
depth_idx=np.linspace(0, ns-1, n_depths).astype(int)
S=np.zeros((N,N))
print('\n全厚点源叠加 (深度从 0 到 全厚):')
t0=time.perf_counter()
for z in depth_idx:
    F,_=propagate_pointsource_from(z)
    S+=F/F.max()   # 各深度归一化后非相干叠加（模方和）
    print(f'  z={z:3d} ({z*0.1:.1f}A)  done ({time.perf_counter()-t0:.0f}s)',flush=True)
print(f'叠加完成 {time.perf_counter()-t0:.0f}s')

# ---- 保存 & 画图 ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

dk=1/extent; kx=(np.arange(N)-(N-1)/2)*dk; ky=kx
fig,axes=plt.subplots(1,2,figsize=(13,5.6))
logS=apply_log(S)
im=axes[0].imshow(logS,extent=(kx[0],kx[-1],ky[0],ky[-1]),origin='lower',cmap='inferno')
axes[0].set_xlabel('kx [1/A]'); axes[0].set_ylabel('ky [1/A]')
axes[0].set_title(f'Mo [001] EBSD Kikuchi (point-source+RRS, {n_depths} depths incoherent)',fontsize=10)
fig.colorbar(im,ax=axes[0],fraction=0.046,pad=0.04,label='log intensity')
# 径向剖面
Xg,Yg=np.meshgrid(kx,ky); R=np.sqrt(Xg**2+Yg**2)
nb=80; edges=np.linspace(0,R.max(),nb+1); idx=np.digitize(R.ravel(),edges)
prof=np.array([S.ravel()[idx==b].mean() if (idx==b).any() else 0 for b in range(1,nb+1)])
rb=0.5*(edges[1:]+edges[:-1])
th=np.degrees(np.arcsin(np.clip(wl*rb,0,0.999)))
axes[1].semilogy(th,prof+1e-12)
axes[1].set_xlabel('scattering angle [deg]'); axes[1].set_ylabel('mean intensity (log)')
axes[1].set_title('radial profile'); axes[1].grid(True,which='both',ls=':',alpha=0.5)
fig.tight_layout()
out=r'F:\papers\backscatter\paper\figures\ebsd_kikuchi_pointsrcline.png'
fig.savefig(out,dpi=200,bbox_inches='tight')
np.savez(r'F:\papers\backscatter\paper\figures\ebsd_kikuchi_pointsrcline.npz',S=S,kx=kx,ky=ky,prof=prof,rb=rb,th=th)
print('saved',out)
