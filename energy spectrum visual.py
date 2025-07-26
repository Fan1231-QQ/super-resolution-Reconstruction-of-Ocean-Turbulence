#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 16:47:56 2025

@author: s2656823
"""

import h5py
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from Spectral_Jacob import A, compute_energy_spectrum_for_two_layer,compute_average_energy_spectrum_Jacob
plt.rcParams.update({'font.size': 16})

file_path = "data_generation_new.h5"

with h5py.File(file_path, "r") as h5f:
    # 查看文件中的数据集
    print("Available datasets:")
    for name in h5f:
        print(name)

    # 读取时间数据
    time = h5f["time"][:]
    print(f"Time shape: {time.shape}")
    total_rows = time.shape[0]

# read data
with h5py.File(file_path, "r") as h5f:

    ui = jnp.array(h5f["u"][:]).mean(axis=0)  # mean over time
    vi = jnp.array(h5f["v"][:]).mean(axis=0)

    # 分离上下层并添加时间维度 (Nt = 1)
u1 = ui[0:1]  # shape: (1, 512, 512)
v1 = vi[0:1]
u2 = ui[1:2]
v2 = vi[1:2]

# 构造 vel_batch 形状为 (Nt, Nx, Ny, 2)
vel_batch = jnp.stack([u1[0], v1[0],u2[0],v2[0]], axis=-1)  # shape: (512, 512, 2)
# vel_batch = vel_batch[jnp.newaxis, ...]        # shape: (1, 512, 512, 2)


k_grid, E_k= A(vel_batch)


# visual
plt.figure(figsize=(8, 6))
plt.plot(k_grid, E_k, label='Energy Spectrum')
plt.xlabel('Wavenumber k')
plt.ylabel('E(k)')
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.title('Initial Energy Spectrum')
plt.grid(True)
plt.tight_layout()
plt.show()
