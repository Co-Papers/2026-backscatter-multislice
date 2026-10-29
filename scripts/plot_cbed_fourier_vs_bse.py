# 读取傅里叶变换计算的CBED和考虑背散射效应的CBED的结果，并进行比较
import numpy as np
import matplotlib.pyplot as plt
import apply_log as al
files = ["paper/figures_multi_phonons_16/cbed_bse_phonons_16_300keV.npz", "paper/figures_multi_phonons_16/cbed_bse_phonons_16_200keV.npz",
         "paper/figures_multi_phonons_16/cbed_bse_phonons_16_120keV.npz", "paper/figures_multi_phonons_16/cbed_bse_phonons_16_80keV.npz",
         "paper/figures_multi_phonons_16/cbed_bse_phonons_16_60keV.npz", "paper/figures_multi_phonons_16/cbed_bse_phonons_16_30keV.npz",
         "paper/figures_multi_phonons_16/cbed_bse_phonons_16_10keV.npz", "paper/figures_multi_phonons_16/cbed_bse_phonons_16_5keV.npz"]
for ifile in files:
    npz = np.load(ifile)
    print(npz.keys())
    print(npz["cbed_without"].shape)  # 打印数组的形状，确保数据正确加载
    print(npz["cbed_with"].shape)  # 打印数组的形状，确保数据正确加载
    cbed_fft = npz["cbed_without"][-1]
    cbed_bse = npz["cbed_with"][-1]

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    _cmap = "viridis"
    # 绘制第一个子图，并保存返回的 ScalarMappable
    im0 = ax[0].imshow(cbed_fft, cmap=_cmap)
    # 通过 fig.colorbar 添加颜色条，指定所属坐标轴
    fig.colorbar(im0, ax=ax[0], label="Intensity")  # Apply log transformation to the colorbar
    ax[0].set_title("CBED by Fourier Transform Method")
    ax[0].set_xlabel("Pixel")
    ax[0].set_ylabel("Pixel")

    # 同理处理第二个子图
    im1 = ax[1].imshow(cbed_bse, cmap=_cmap)
    fig.colorbar(im1, ax=ax[1], label="Intensity")  # Apply log transformation to the colorbar
    ax[1].set_title("CBED by Backscattering Electron microscopy")
    ax[1].set_xlabel("Pixel")
    ax[1].set_ylabel("Pixel")

    diff = cbed_fft - cbed_bse
    im2 = ax[2].imshow(diff, cmap=_cmap)
    fig.colorbar(im2, ax=ax[2], label="Intensity")  # Apply log transformation to the colorbar
    ax[2].set_title("Difference")
    ax[2].set_xlabel("Pixel")
    ax[2].set_ylabel("Pixel")

    plt.tight_layout()  # 避免布局拥挤
    plt.savefig("paper/figures_multi_phonons_16/cbed_fft_vs_bse_phonons_16_" + ifile.split("_")[-1].split(".")[0] + ".png", dpi=300, bbox_inches="tight")
    #plt.show()
