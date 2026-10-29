# 背散射修正多切片算法的系统性研究（复现基准仓库）

本仓库为论文《背散射修正多切片算法的系统性研究：从数值验证到低能电子衍射应用》
（目标期刊 *Ultramicroscopy*）的官方复现基准（reproduction baseline）。
包含全部模拟脚本、绘图脚本、论文图、关键数据与中文论文稿。

## 仓库结构

```
scripts/          全部计算与绘图脚本（Python，GPU/CuPy）
atomic_model/     SrTiO3.cif 原子模型
figures/          论文全部 16 张图（300 dpi，与正文编号对应见下）
data/             （预留；完整 npz 数据见 Release 资产）
docs/             设计文档、文献调查、CBED/EBSD 原理笔记
```

## 环境（复现前提）

- Windows / Linux，NVIDIA GPU（≥ 8 GB 显存），CUDA 驱动 ≥ 12.x
- Python 3.11+，CuPy（CUDA 12.x 系列）
- **abTEM 定制版**：`Co-STEM-Lab/abTEM`，分支 `dev`，commit `ed8173a5`
  （在官方 abTEM 基础上包含：FFT 拉普拉斯算子选项、GPU 融合 CUDA 拉普拉斯核、
  背散射零振幅早退与反向传播修复）
  ```bash
  git clone https://github.com/Co-STEM-Lab/abTEM.git
  cd abTEM && git checkout ed8173a5
  pip install -e .        # 配合 CuPy 环境
  ```

## 核心脚本与论文图的对应关系

| 图 | 文件 | 生成脚本 |
|---|---|---|
| 图 1 | fig01_summary_curve.png | `plot_cbed_bse_summary_curve.py`（数据：`cbed_bse_energy_sweep.py`） |
| 图 2 | fig02_fd_vs_fft.png | `cbed_bse_multi_phonons_16_fft.py` + postprocess |
| 图 3 | fig03_convergence.png | `plot_cbed_bse_convergence.py`（数据：`cbed_bse_convergence.py`） |
| 图 4 | fig04_unitarity.png | `plot_cbed_bse_unitarity.py`（数据：`cbed_bse_fft_laplace_5keV.py`、`cbed_bse_unitarity_source.py`、`flux_budget_5keV.py`） |
| 图 5-6 | fig05/fig06 | `test_laplacian_precision.py`、`benchmark_laplacian_ft.py` |
| 图 7-14 | cbed_bse_{5..300}keV.png | `cbed_bse_energy_sweep.py` |
| 图 15 | fig15_timing.png | 同上（计时输出） |
| 图 16 | fig16_ebsd_gnomonic.png | `ebsd_kikuchi_pointsource.py` 系列 |

完整 npz 数据（约 370 MB）与论文 Word 稿（约 103 MB）见 **Release 资产**
（git 仓库仅保留图片以控制体积）。

## 快速复现（关键结果）

```bash
# 电压/厚度扫描（图 1、7-14；GPU 约 40 min/8 电压）
python scripts/cbed_bse_energy_sweep.py
python scripts/plot_cbed_bse_summary_curve.py

# 收敛性测试（图 3；约 40 min）
python scripts/cbed_bse_convergence.py --smoke   # 冒烟（小晶胞）
python scripts/cbed_bse_convergence.py           # 全尺寸
python scripts/plot_cbed_bse_convergence.py

# 幺正性分析（图 4）
python scripts/cbed_bse_fft_laplace_5keV.py
python scripts/cbed_bse_unitarity_source.py
python scripts/flux_budget_5keV.py
python scripts/plot_cbed_bse_unitarity.py
```

## 主要结果速览

- 5 keV 下背散射修正使前向 CBED 积分强度净增约 13%（基准采样 0.05 Å），
  破坏传统多切片的单调下降；≥100 keV 时净变化 < 0.2%（纯重分布）
- 幺正性分析：物理净增益上界 ≈ 10.8%（195 Å）；超出部分（约 2-3%）为
  背散射修正项的算法固有范数漂移（与 Laplacian 方案无关，随切片数累积；
  关闭修正项的前向级数展开严格幺正）
- 切片厚度存在"合法窗口"（5 keV：0.1 Å 合法，0.05 Å 以下违反，0.2 Å 发散）——
  **不是越薄越好**
- FD 6 阶与 FFT 拉普拉斯结果一致（相对差异 3.58%）；计算时间代价 1.7-2.4 倍

## 许可

代码：MIT License（见 LICENSE）。论文稿与图片版权归原作者所有（投稿中）。
