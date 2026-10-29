# EBSD 的计算原理，及如何用 TEM 多切片（abTEM）做 EBSD 仿真

> 领域知识笔记 · 电子显微学 / 衍射动力学 · 更新 2026-08-30
> 依据：`paper\` 下 3 篇文献（LIU 2016 J.Microsc.；李兴林 2023 孪晶；李相成 2023 Micron 169）及其引用。

## 一、EBSD 的物理模型（为什么不能用"纯背散射"直接算）

EBSD 图案来自**背散射**电子在晶格上的**动力衍射**。直接模拟背散射电子很难。Winkelmann（2009）把过程分成两部分：

1. **入射**：电子打在被倾斜的样品表面，产生非弹性散射（形成"内部电子源"）——**细节可忽略**。
2. **出射**：这些内部源电子与样品作用，产生弹性 + 非弹性 + 非相干散射，形成屏上的 EBSD 花样——**这才是要算的**。

关键：第 2 部分可用 **互易定理（reciprocity，Pogany & Turner 1968）** 反过来处理——
> 屏方向上的平面波**入射**到样品，与"样品内部源电子**出射**到屏"等价（互易）。
> 于是 EBSD 的背散射问题 **等价于一个前向透射衍射问题**，可用 TEM 的多切片（multislice）来解。

非相干散射 → 通过**叠加各深度的衍射强度**（逐层求和）体现；
非弹性/能散 → 用一定能宽（主要使图案变模糊，可先忽略）。

## 二、多切片法（TEM 动力衍射）的选择：RS vs RRS

EBSD 用**动力衍射**（运动学只给带边缘信息，强度不准）。动力法主要有：

| 方法 | 说明 | 近似 | 适缺陷 |
|---|---|---|---|
| **Bloch wave (BW)** | Winkelmann 2007 首创 EBSD；严格但不适缺陷 | 高能近似（抛物 Ewald + 忽略背散射） | 难 |
| **常规多切片 CMS**（=abTEM FourierMultislice） | Cowley&Moodie；物理光学 | 高能近似；仅前向 | 可 |
| **实空间法 RS** | Van Dyck 1980 | 高能近似；仅前向 | 可 |
| **修正实空间法 RRS** | Chen & Van Dyck 1997；Cai&Chen 2012 数值解 | **无抛物近似**（Ewald 球精确）；忽略背散射 | 可 |

- RRS 是 RS/CMS 的高阶（严格来说 RS 是 RRS 的一阶近似）——**低加速电压时更准**。
- EBSD 加速电压多在 **10–40 kV**（远低于常规 TEM 的 ≥100 kV），此时抛物近似误差大 → **RRS 更优**，故这几篇论文均用 RRS。
- **abTEM 对应**：`FourierMultislice()`≈CMS；`RealSpaceMultislice(expansion_scope="full")` 即 Chen & Van Dyck 的 RRS 系（含无抛物近似的处理，本仓库还加入了背散射修正项）。所以**用 abTEM 的 RealSpace-full 做低能 EBSD 最合适**。

## 三、EBSD——多切片仿真的具体做法（参数来自李相成 2023）

- 入射电子束**垂直样品表面**，且**沿某晶带轴**入射（等效于样品取向固定、束沿晶带轴）；这样交互体积无各向异性。
- 平面内网格 1024×1024，**切片厚约 0.01 nm**。
- **孔径半径**（aperture radius，倒空间像素）决定视场；**样品厚度**取代表性值（20 kV 下 EBSD 主贡献来自近表面 **10–40 nm**，取 20 nm）。
- Debye–Waller 因子（如 0.4 Å²）、原子散射因子用 **Peng et al. 1996** 拟合。
- **出射面归一化衍射强度**。
- **投影**：模拟图案是**立体投影（stereographic）**（Kikuchi 带会弯），实验是**透视投影（gnomonic）**（Baba-Kishi 1998；Nolze&Winkelmann 2017；Vos&Winkelmann 2017）。

隐含的**互易 + 逐层非相干**模型（LIU 2016）：
```
EBSD(I_screen) = Σ_{depth z} | 正演多切片在深度 z 的出射波衍射强度 |²
```
（Winkelmann/LIU 模型里"叠加各层"；李相成 2023 则用一个 20 nm 代表性厚度 + 出射面归一化，属简化。）

## 四、EBSD 仿真的物理公式推导（互易 + RRS 多切片 + 非相干叠加 + gnomonic 投影）

这一节给出可直接落地的数学公式，结合 Liu 2016 / Winkelmann 2009 / 李相成 2023 的模型。

### 4.1 电子散射的薛定谔方程与坐标分解

对加速电压为 \(E\)（相对论修正 \(\gamma = 1 + E/mc^2\)，\(k^2 = 2m\gamma E/\hbar^2\)）的电子，取光学轴为 \(z\)，波函数写成调制平面波

\[
\psi(\mathbf r) = A(\mathbf r)\,e^{i k z} ,
\]

则修正的薛定谔方程（式 5）为

\[
\partial_z^2 A + 2i k\,\partial_z A = \Delta_\perp A + \frac{2m\gamma}{\hbar^2} V(\mathbf r)\,A .
\]

其中 \(\Delta_\perp = \partial_x^2 + \partial_y^2\) 是横向拉普拉斯算子，\(V\) 是电子势能。

### 4.2 高能（抛物）近似与 RRS（无抛物近似）

- **高能近似（parabolic/CMS）**：略去 \(\partial_z^2 A\)，得
  \[
  2i k\,\partial_z A = \Delta_\perp A + \frac{2m\gamma}{\hbar^2}V\,A .
  \]
  这是关于 \(z\) 的一阶方程，等同于一维含时薛定谔方程，Ewald 球被近似成抛物面。对应 abTEM 的 `FourierMultislice`。

- **RRS（修正实空间法，Chen & Van Dyck 1997）**：保留二阶导，改写为算子平方根形式（式 35），
  \[
  \partial_z A = i k\left[-1 + \sqrt{1 + \frac{1}{k^2}\left(\Delta_\perp + \frac{2m\gamma}{\hbar^2}V\right)}\right]A .
  \]
  只取正号（忽略背散射 \(A_{\text{back}}\)），即前向散射。因平方根不含 \(\partial_z^2\)，Ewald 球为精确球面，无抛物近似。用二项式展开（式 37）\(\sqrt{1+x} = \sum_n \binom{1/2}{n}x^n\) 得
  \[
  U(z_j,z_{j+1}) = \exp\!\left[i k\,\Delta z\sum_{n=1}^{\infty}\binom{1/2}{n}\frac{1}{k^{2n}}\left(\Delta_\perp + \frac{2m\gamma}{\hbar^2}V_j\right)^{n}\right] .
  \]
  由于要展开拉普拉斯的高次幂，RRS 必须在**实空间**做（\(\Delta_\perp\) 为局部微分算子），这正是 `RealSpaceMultislice(expansion_scope="full")` 的实现。

### 4.3 拉普拉斯算子的离散化：有限差分 vs 谱（FFT）

\(\Delta_\perp\) 作用于每个切片。

- **有限差分（FD）**：可分离中心差分模板（一维二阶导，accuracy 阶），系数由 Taylor 展开得，例如最常见 5 点：
  \[
  f''(x) \approx \frac{1}{\Delta x^2}\left(-\tfrac{5}{4},\, \tfrac{4}{3},\, -\tfrac{1}{12},\, \tfrac{4}{3},\, -\tfrac{5}{4}\right) \quad(\text{acc}=4) .
  \]
  abTEM 在 `abtem/finite_difference.py` 用 `finite_difference_coefficients(2, accuracy)` 生成这些系数，模板经 `_laplace_operator_stencil` 以可分离方式对 \((M,H,W)\) 数组作用。

- **谱（FFT）**：对周期函数，倒空间 \(\Delta_\perp\to-(2\pi k)^2\)，
  \[
  \Delta_\perp f = \mathcal F^{-1}\big[-\kappa^2\, \mathcal F[f]\big], \qquad \kappa = 2\pi(k_x,k_y) ,
  \]
  其中 \(k_x,k_y\) 由 `fftfreq`（cycles/Å）得到。谱法**无截断误差**（仅 FFT 数值精度），是背散射厚样品下最接近真值的拉普拉斯。本仓库在 `LaplaceOperator(method="fft")` 提供该选项。

### 4.4 互易定理（reciprocity）：把背散射出射转成前向透射

Pogany & Turner (1968)：电子散射矩阵元满足互易性，

\[
\mathcal M(\mathbf k_s \leftarrow \mathbf k_p;\mathbf r) = \mathcal M(-\mathbf k_p \leftarrow -\mathbf k_s;\mathbf r) .
\]

用于 EBSD：**从探测屏方向 \(\hat{\mathbf k}_s\) 出射** 的振幅，等于**从该方向反向入射的平面波**在样品内部的振幅。于是

> EBSD 亮场像素 \(I(\hat{\mathbf k}_s)\) = 平面波沿 \(-\hat{\mathbf k}_s\) 入射时，样品内部出一波（前向透射）的衍射强度。

Winkelmann 2009 据此把 EBSD 建模为：束 \(-\hat{\mathbf k}_s\) 沿晶体倒格矢入射，做**平面波透射动力衍射**。实际操作中令平面波沿晶带轴（如 Mo [001]）垂直入射即可获得花样（中心晶带轴近似）。

### 4.5 逐层非相干强度叠加（interaction volume）

EBSD 电子来自近表面 \(10\,\text{–}\,40\,\text{nm}\) 交互体积、多个深度 \(z_j\) 的**非相干**贡献，故对每个深度取强度再求和：

\[
I_{\text{EBSD}}(k_x,k_y) = \sum_{j} \left|\mathcal F\big[\psi_{z_j}(x,y)\big](k_x,k_y)\right|^2 ,
\]

即

```
I_sum = Σ_{exit_planes z_j} | diffraction_patterns(ψ_z_j) |²
```

这里 \(\psi_{z_j}\) 由 `exit_planes` 第 \(j\) 个出口平面给出，`diffraction_patterns` 取该深度的倒空间强度（取 \(|\cdot|^2\)），逐层累加。这是 abTEM 用 `exit_planes` 实现的核心。

### 4.6 gnomonic 投影（球面 → 探测屏）

倒空间坐标 \((k_x,k_y)\) 对应散射角

\[
\theta = \arcsin\!\left(\lambda \sqrt{k_x^2+k_y^2}\right),\qquad \theta_x = \arcsin(\lambda k_x),\ \theta_y = \arcsin(\lambda k_y) ,
\]

其中 \(\lambda\) 为电子波长。**gnomonic（透视/切平面）投影**把散射角映射到距样品 \(L\) 的平面屏：

\[
X = L\,\tan\theta_x,\qquad Y = L\,\tan\theta_y .
\]

（**stereographic** 则为 \(X = 2L\tan(\theta_x/2)\)，二者在小角时趋同、大角时 gnomonic 让菊池带更弯。）

### 4.7 孔径视场与归一化

- **孔径**：用 `diffraction_patterns(max_angle=...)`（mrad）或直接在 \((k_x,k_y)\) 上做圆形掩膜，限制视场到探测器实际范围（一般 \(\lesssim \pm 30^\circ\)）。
- **归一化**：
  \[
  I_{\text{norm}} = \frac{I_{\text{EBSD}}}{\sum I_{\text{EBSD}}} \cdot N_{\text{pix}} ,\qquad \text{或按} \max(I_{\text{EBSD}}) \text{归一} .
  \]

### 4.8 验证：Kikuchi 带/HOLZ 环 

- **带间距**：相邻菊池带的角间距满足 Bragg 条件 \(2d\sin\theta\approx\lambda\)。
- **HOLZ（高阶劳厄带）环半径**：单个倒格矢 \(g\) 对应的环位于
  \[
  g_{\perp}^2 + \Big(\frac{1}{\lambda} - \sqrt{\frac{1}{\lambda^2}-g_z^2}\Big)^2 = \frac{1}{\lambda^2}\quad\Rightarrow\quad \theta_{\text{HOLZ}} = \arcsin\!\big(\lambda\,g_{\perp}\big) .
  \]
  其中 \(g_\perp\) 为横向倒格矢模长。可用第一 HOLZ 环半径反推晶格常数：\(g_{\perp}^{(1)} = \frac{\sin\theta}{\lambda}\).

## 五、如果要用 abTEM 实现 EBSD

abTEM 提供了**动力前向多切片**（最重的部分），但**缺 EBSD 专属的几何/投影/孔径包装**。需要自建：

1. **正向多切片**：构造样品（取向使晶带轴垂直于表面），用
   - 低能（10–40 kV，本项目 5 keV 更低）：`Probe(energy=..)` 或平面波，算法选 `RealSpaceMultislice(expansion_scope="full")`（≈RRS）。
   - grid / slice_thickness 按上面参数；`sampling` 由超胞与网格决定。
2. **取衍射花样**：`exit_wave.diffraction_patterns(max_angle=..)` 得到倒空间强度（本质是 SAED/倒空间像）。
3. **EBSD 化**：
   - 把倒空间坐标 (kx, ky) 转成**散射角**（θ = asin(k λ)），再按 **gnomonic/stereographic 投影**映射到探测器屏坐标（实验为 gnomonic，屏位置 = tan(θ)·相机长度）。
   - 施加**孔径**（视场）、做**强度归一化**。
4. **（可选）逐层非相干**：用 `exit_planes` 取多个深度，对每个 depth 取 `|diffraction|²` 再逐层求和（体现 EBSD 交互体积的非相干性）。
5. **对比/验证**：与实验图案或 Winkelmann 的 BW 结果比对 Kikuchi 带位置、HOLZ 环半径、晶格参数（如由 HOLZ 环半径反推晶格常数）。

一句话：**abTEM 已能做"正向动力多切片"这个核心；EBSD 的增量是"互易 + 屏投影(gnomonic) + 孔径 + 逐层非相干求和"这一层包装。**

## 六、关键参考文献（含其引用）

- Winkelmann et al. 2007；Winkelmann 2009, 2010 —— **Bloch wave 首创 EBSD 仿真**（方法源头）。
- Pogany & Turner 1968 —— **互易定理**。
- Zaefferer 2007 —— EBSD 花样的形成机制。
- Chen & Van Dyck 1997 —— RRS 理论；Cai & Chen 2012 —— RRS 数值解。
- Liu et al. 2016, J. Microsc. 264:71 —— **RS/RRS 两种多切片对比（Mo）**。
- Li Xinglin et al. 2023 —— 多层孪晶 EBSD（RRS）。
- Li Xiangcheng et al. 2023, Micron 169:103461 —— 位错结构 EBSD（RRS，参数详见上）。
- Peng et al. 1996 —— 原子散射因子；Dingley 2004 —— EBSD 有效深度 10–40 nm。
- Baba-Kishi 1998；Nolze & Winkelmann 2017；Vos & Winkelmann 2017 —— gnomonic / stereographic 投影。

## 七、与本项目（backscatter）的关系
本项目在做 **5 keV 低能 TEM 的透射 CBED + 背散射修正**（背散射修正为 **abTEM 自带**的 `RealSpaceMultislice(expansion_scope="full")` 修正项；Wacker 2015 仅作相关文献参考）。EBSD 与之共享同一套**动力多切片**与 **RRS/背散射**物理，但几何不同：CBED 看透射、把背散射当修正项；EBSD 看背散射、用互易降维成前向透射。因此**abTEM 的多切片核心可以直接复用**，只需加 EBSD 的投影/互易/孔径/逐层求和外壳。
