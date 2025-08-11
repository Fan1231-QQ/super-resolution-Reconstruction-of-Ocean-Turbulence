#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 12:52:02 2025

@author: s2656823
"""

import numpy as np
from scipy.interpolate import interp1d


def downscale(q, cgf):
    for n in q.shape:
        if n % cgf != 0:
            raise ValueError("Invalid coarse graining factor")
    return q[::cgf, ::cgf]


def upscale(q, cgf):
    # Extend and wrap
    M, N = q.shape
    q_e = np.zeros_like(q, shape=(M + 1, N + 1))
    q_e[:-1, :-1] = q
    q_e[-1, :] = q_e[0, :]
    q_e[:, -1] = q_e[:, 0]

    points = (np.arange(0, M + 1, dtype=int),
              np.arange(0, N + 1, dtype=int))
    query_points = (np.linspace(0, cgf * M - 1, cgf * M) / cgf,
                    np.linspace(0, cgf * N - 1, cgf * N) / cgf)

    q_u = q_e.copy()
    q_u = interp1d(points[0], q_u, axis=0)(query_points[0])
    q_u = interp1d(points[1], q_u, axis=1)(query_points[1])
    return q_u



def downscale_4D(q, cgf):
    if q.ndim != 4:
        raise ValueError("Expected 4D array: (batch, channel, height, width)")
    B, C, H, W = q.shape
    if H % cgf != 0 or W % cgf != 0:
        raise ValueError("Height and Width must be divisible by cgf")
    
    return q[:, :, ::cgf, ::cgf]


def upscale_4D(q, cgf):
    if q.ndim != 4:
        raise ValueError("Expected 4D array: (batch, channel, height, width)")
    
    B, C, H, W = q.shape
    upscaled = np.zeros((B, C, H * cgf, W * cgf), dtype=q.dtype)
    
    for b in range(B):
        for c in range(C):
            q2d = q[b, c, :, :]
            # Wrap and interpolate as before
            q_e = np.zeros((H + 1, W + 1), dtype=q.dtype)
            q_e[:-1, :-1] = q2d
            q_e[-1, :] = q_e[0, :]
            q_e[:, -1] = q_e[:, 0]

            points = (np.arange(0, H + 1), np.arange(0, W + 1))
            query_points = (
                np.linspace(0, cgf * H - 1, cgf * H) / cgf,
                np.linspace(0, cgf * W - 1, cgf * W) / cgf
            )

            q_u = interp1d(points[0], q_e, axis=0)(query_points[0])
            q_u = interp1d(points[1], q_u, axis=1)(query_points[1])

            upscaled[b, c, :, :] = q_u
    return upscaled

import numpy as np

def compute_spectrum(q1, q2):    
    q1_hat = np.fft.fft2(q1) 
    q2_hat = np.fft.fft2(q2) 
    # 创建波数网格
    nx, ny = q1_hat.shape
    kx = np.fft.fftfreq(nx) * nx
    ky = np.fft.fftfreq(ny) * ny
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')
    k_squared = kx_grid**2 + ky_grid**2
    k_squared[k_squared==0] = 1e-20  # 防止除以零
    
    # 计算每一层的能谱密度
    E1 = np.abs(q1_hat)**2 / k_squared
    E2 = np.abs(q2_hat)**2 / k_squared
    E_2D = 0.5 * (E1 + E2)  # 总谱能量密度
    
    # 求波数模长，用于径向平均
    k_mag = np.sqrt(k_squared)
    k_max = int(np.max(k_mag))
    k_bins = np.arange(0.5, k_max, 1.0)
    k_indices = np.digitize(k_mag.flat, k_bins)
    E_k = np.zeros(len(k_bins))
    
    for i in range(1, len(k_bins)):
        mask = (k_indices == i)
        if np.any(mask):
            E_k[i-1] = np.mean(E_2D.flat[mask])
    return k_bins, E_k


'''

a = np.arange(16).reshape(4, 4)

print(a)

print('downscale', downscale(a,2))
b=downscale(a,2)

print('upscale', upscale(a,2))

cgf = 2
M, N = a.shape
q_e = np.zeros_like(a, shape=(M + 1, N + 1))
q_e[:-1, :-1] = a
q_e[-1, :] = q_e[0, :]
q_e[:, -1] = q_e[:, 0]

points = (np.arange(0, M + 1, dtype=int),
          np.arange(0, N + 1, dtype=int))
query_points = (np.linspace(0, cgf * M - 1, cgf * M) / cgf,
                np.linspace(0, cgf * N - 1, cgf * N) / cgf)

q_u = q_e.copy()
q_u = interp1d(points[0], q_u, axis=0)(query_points[0])
q_u = interp1d(points[1], q_u, axis=1)(query_points[1])




import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# 已知点
x = np.array([0, 1, 2, 3])
y = np.array([0, 1, 0, 1])

# 构造插值函数
f = interp1d(x, y, kind='linear')

# 插值查询点
x_new = np.linspace(0, 3, 100)
y_new = f(x_new)

# 画图
plt.plot(x, y, 'o', label='data points')
plt.plot(x_new, y_new, '-', label='linear interpolation')
plt.legend()
plt.show()
'''



