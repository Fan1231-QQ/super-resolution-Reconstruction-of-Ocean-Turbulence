#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 12:14:31 2025

@author: s2656823
"""
import matplotlib.ticker as ticker
import h5py
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})

# ====== Configuration ======
file_path = "prediction_results.h5"
batch_index = 0  # You can change this to any index within range

# ====== Load data lazily (batch-wise) ======
with h5py.File(file_path, "r") as h5f:
    # Lazy load single sample from each dataset
    y_pred = h5f["y_pred"][batch_index:batch_index+1]  # shape (1, H, W, C)
    x_val= h5f["x_val"][batch_index:batch_index+1]
    raw = h5f["raw"][batch_index:batch_index+1]

# ====== Compute true residual ======
y_true_residual = raw - x_val
reconstructed = x_val + y_pred

beta, L_y = 2.0e-11, 1.8e6
scale = beta * L_y
# raw=raw[0,:,:,0]*scale; reconstructed=reconstructed[0, :, :, 0]*scale
# y_true_residual=y_true_residual[0, :, :, 0]*scale; y_pred=y_pred[0, :, :, 0]*scale
# min1,max1=raw.min(),raw.max()
# min2,max2=y_true_residual.min(),y_true_residual.max()
# ====== Visualize ======
fig, axs = plt.subplots(2, 3, figsize=(14, 9))  # 稍微压缩横向比例
axs = axs.flatten()

# 图像列表
images = [
    raw[0, :, :, 0],
    reconstructed[0, :, :, 0],
    np.abs(reconstructed[0, :, :, 0] - raw[0, :, :, 0]),
    y_true_residual[0, :, :, 0],
    y_pred[0, :, :, 0]
]

# 配色参数
cmaps = ["coolwarm", "coolwarm", "Reds", "coolwarm", "coolwarm"]
titles = ["(a) Original Field", "(b) Reconstructed Field", "(c) Absolute Error",
          "(d) True Residual", "(e) Predicted Residual"]
vmins = [-1, -1, None, -0.2, -0.2]
vmaxs = [1, 1, None, 0.2, 0.2]

# 绘图循环
for i in range(5):
    im = axs[i].imshow(images[i], cmap=cmaps[i], origin='lower', 
                       vmin=vmins[i], vmax=vmaxs[i])
    axs[i].set_title(titles[i])
    axs[i].axis('off')
    fig.colorbar(im, ax=axs[i], shrink=0.7)

axs[5].axis('off')  # 最后一格空白

plt.tight_layout()
plt.show()

#==========spectral=================
from scaling import compute_spectrum

# 计算谱
k_bin_centers, E_k = compute_spectrum(raw[0,:,:,0]*scale, raw[0,:,:,1]*scale)
k_train, Ek_train = compute_spectrum(reconstructed[0,:,:,0]*scale, reconstructed[0,:,:,1]*scale)

# 选择拟合区间范围（可根据实际数据微调）
fit_range = (k_bin_centers > 3) & (k_bin_centers < 30)
k_fit = k_bin_centers[fit_range]
E_fit = E_k[fit_range]

# 对 log-log 空间拟合直线：log(E) = a * log(k) + b
log_k = np.log10(k_fit)
log_E = np.log10(E_fit)
slope, intercept = np.polyfit(log_k, log_E, 1)

# 生成拟合线用于绘图
k_line = np.array([k_fit.min(), k_fit.max()])
E_line = 10**(intercept) * k_line**slope

# 绘图
plt.figure(figsize=(6, 4))
plt.loglog(k_bin_centers, E_k, label="DNS")
plt.loglog(k_train, Ek_train, label="Reconstructed")
plt.loglog(k_line, E_line, 'k--', label=fr'Fit: $k^{{{slope:.2f}}}$')  # 拟合斜率

plt.xlabel("Wavenumber k")
plt.ylabel("Spectrum E(k)")
plt.ylim(1e-9, 1e-2)
plt.grid(False, which='both')
plt.legend(fontsize='small')
plt.tight_layout()
plt.show()



# 计算谱
k_bin_centers, E_k = compute_spectrum(raw[0,:,:,0]*scale, raw[0,:,:,1]*scale)
k_train, Ek_train = compute_spectrum(reconstructed[0,:,:,0]*scale, reconstructed[0,:,:,1]*scale)

# 选择参考线段的起点与终点
# k_ref = np.array([2, 20])
# E_ref = 1e-3 * (k_ref / k_ref[0])**(-5)  
k_ref = np.linspace(6, 90, 100)  # 延长黑线并让它更平滑
E0 = E_k[np.argmin(np.abs(k_bin_centers - k_ref[0]))] * 10
E_ref = E0 * (k_ref / k_ref[0])**(-5)



# 绘图
plt.figure(figsize=(6, 4))
plt.loglog(k_bin_centers, E_k, label="DNS")
plt.loglog(k_train, Ek_train, label="Reconstructed")
plt.loglog(k_ref, E_ref, 'k--', label=r'$k^{-5}$')  
plt.xlabel("Wavenumber k")
plt.ylabel("Spectrum E(k)")
plt.ylim(1e-9, 1e-2)
plt.grid(False, which='both')
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()



#==========loss===============

# 设置文件路径
file_path = "training_results.h5"

plt.figure(figsize=(10, 6))

with h5py.File(file_path, "r") as h5f:
    for lr_group in h5f.keys():
        grp = h5f[lr_group]

        train_loss = grp["loss"][:]
        val_loss = grp["val_loss"][:]

        lr_str = lr_group.replace("lr_", "")  # e.g. '1e-03'
        plt.semilogy(train_loss, label=f"Train Loss lr={lr_str}")
        plt.semilogy(val_loss, linestyle='--', label=f"Val Loss lr={lr_str}")

# 图像设置
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Training & Validation Loss (Semilog)")
plt.legend()
plt.grid(True, which="both")
plt.tight_layout()
plt.show()





fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))  # 横向两个子图

best_lr_key = None
best_val_loss = float('inf')

with h5py.File(file_path, "r") as h5f:
    for lr_key in h5f.keys():  # 遍历每个学习率组
        grp = h5f[lr_key]
        train_loss = grp["loss"][:]
        val_loss = grp["val_loss"][:]

        lr_str = lr_key.replace("lr_", "")
        ax1.semilogy(train_loss, label=f"lr={lr_str}")
        ax2.semilogy(val_loss, label=f"lr={lr_str}")

        # ✅ 使用 val_loss 中的最小值作为 baseline 判据
        min_val = np.min(val_loss)
        if min_val < best_val_loss:
            best_val_loss = min_val
            best_lr_key = lr_str

    # baseline 横线（画在 val loss 图中）
    ax2.axhline(y=best_val_loss, color='red', linestyle='--',
                label=f"Baseline (lr={best_lr_key}, {best_val_loss:.2e})")

# 图像设置
ax1.set_title("Training Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss (MSE)")
ax1.grid(True, which='both')
ax1.legend()

ax2.set_title("Validation Loss")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss (MSE)")
ax2.grid(True, which='both')

# 对数坐标和刻度美化
ax2.set_yscale('log')
ax2.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=[1.0, 2.0, 5.0]))
ax2.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0)*0.1))
ax2.yaxis.set_minor_formatter(ticker.NullFormatter())

ax2.legend()

plt.tight_layout()
plt.show()

