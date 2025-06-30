#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 12:59:18 2025

@author: s2656823
"""

import numpy as np
import h5py
import pyqg
from pyqg import diagnostic_tools as tools

year = 24 * 60 * 60 * 360.

beta = 2e-11
epsilon = 1e-5

nx, ny = 512, 512
Lx, Ly = 1.8e6, 1.8e6
dx = Lx / nx
dy = Ly / ny

n_hour = 6
dt = 3600 / n_hour
n_day = 24 * n_hour
t0_year = 100
t1_year = t0_year + 10

characteristic_vorticity = epsilon * beta * Ly / 2

# Step 1: Run Spinup (100 years) and store only restart data
# TODO: cite the Berloff paper for these numbers
m = pyqg.QGModel(
    tavestart=0, tmax=t0_year * year, dt=600, beta=beta, H1=1000, delta=1/3,
    rd=25000, nx=nx, ny=ny, L=Lx, W=Ly, U1=0.06, U2=0., rek=2e-8)

q1 = characteristic_vorticity * np.random.randn(ny, nx)
q2 = characteristic_vorticity * np.random.randn(ny, nx)

q_integral = m.Hi[0] * np.sum(q1) * dx * dy + m.Hi[1] * np.sum(q2) * dx * dy
print(f"{q_integral=}")

q_correction = -(m.Hi[0] * np.sum(q1) + m.Hi[1] * np.sum(q2)) / (m.H * np.prod(q1.shape))
q1 += q_correction
q2 += q_correction
del q_correction

q_integral = m.Hi[0] * np.sum(q1) * dx * dy + m.Hi[1] * np.sum(q2) * dx * dy
print(f"{q_integral=}")

initial_q = np.array([q1, q2])
m.set_q(initial_q)
del q1, q2



for _ in m.run_with_snapshots(tsnapstart=0, tsnapint=1000 * m.dt):
    pass

m.tmax = t1_year * year



with h5py.File("data_generation_uv.h5", "w") as h5f:
    # 初始化时间、速度字段的数据集
    time_dset = h5f.create_dataset("time", shape=(0,), maxshape=(None,), dtype=np.float64)
    u_dset = h5f.create_dataset("u", shape=(0, 2, nx, ny), maxshape=(None, 2, nx, ny), dtype=np.float64)
    v_dset = h5f.create_dataset("v", shape=(0, 2, nx, ny), maxshape=(None, 2, nx, ny), dtype=np.float64)

    # 占位符：将在第一次快照时根据 kr 的长度动态初始化
    kr_dset = None
    kespec_upper_dset = None
    kespec_lower_dset = None

    for snapshot in m.run_with_snapshots(tsnapstart=t0_year * year, tsnapint=n_day * m.dt):
        current_time = np.array([m.t])
        m_ds = m.to_dataset().isel(time=-1)

        # 计算谱数据
        kr, kespec_upper = tools.calc_ispec(m, m_ds.KEspec.isel(lev=0).data)
        _, kespec_lower = tools.calc_ispec(m, m_ds.KEspec.isel(lev=1).data)

        # 如果尚未初始化 kr 和谱能量数据集，则在此初始化
        if kr_dset is None:
            kr_dset = h5f.create_dataset("kr", shape=(len(kr),), dtype=np.float64)
            kespec_upper_dset = h5f.create_dataset(
                "kespec_upper", shape=(0, len(kr)), maxshape=(None, len(kr)), dtype=np.float64
            )
            kespec_lower_dset = h5f.create_dataset(
                "kespec_lower", shape=(0, len(kr)), maxshape=(None, len(kr)), dtype=np.float64
            )
            kr_dset[:] = kr  # kr 通常在整个模拟中不变，直接存一次即可

        # 获取速度场（2个层级）
        u = m.u.copy()
        v = m.v.copy()

        # Resize 所有数据集以存储新一帧数据
        idx = time_dset.shape[0]  # 当前帧索引
        time_dset.resize((idx + 1,))
        u_dset.resize((idx + 1, 2, nx, ny))
        v_dset.resize((idx + 1, 2, nx, ny))
        kespec_upper_dset.resize((idx + 1, len(kr)))
        kespec_lower_dset.resize((idx + 1, len(kr)))

        # 存入数据
        time_dset[idx] = current_time
        u_dset[idx] = u
        v_dset[idx] = v
        kespec_upper_dset[idx] = kespec_upper
        kespec_lower_dset[idx] = kespec_lower

print("Data collection completed and saved to 'data_generation.h5'.")
