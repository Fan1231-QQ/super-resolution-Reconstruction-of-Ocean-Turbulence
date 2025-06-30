#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 11:54:11 2025

@author: s2656823
"""

import numpy as np
import h5py
import pyqg

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

with h5py.File("data_generation.h5", "w") as h5f:
    time_dset = h5f.create_dataset("time", shape=(0,), maxshape=(None,), dtype=np.float64)
    q_dset = h5f.create_dataset("q", shape=(0, 2, ny, nx), maxshape=(None, 2, ny, nx), dtype=np.float64)

    for snapshot in m.run_with_snapshots(tsnapstart=t0_year * year, tsnapint=n_day * m.dt):
        current_time = np.array([m.t])
        q_snapshot = m.q.copy()  # Store potential vorticity
        # Resize datasets to append new data
        time_dset.resize((time_dset.shape[0] + 1,))
        q_dset.resize((q_dset.shape[0] + 1, 2, ny, nx))
        # Store data
        time_dset[-1] = current_time
        q_dset[-1] = q_snapshot


print("Data collection completed.")
