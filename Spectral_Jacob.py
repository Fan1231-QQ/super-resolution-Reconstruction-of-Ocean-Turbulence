#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 14:26:45 2025

@author: s2656823
"""
import jax.numpy as jnp
import numpy as np


def compute_average_energy_spectrum_Jacob(vel_batch: jnp.ndarray, dk = 1.,k_max = None):
    if len(vel_batch.shape) == 3:
        vel_batch = vel_batch[jnp.newaxis, ...] # vel_batch is assumed to have a shape (Nt, Nx, Ny, 2)
        
    _, Nx, Ny = vel_batch.shape    
    # setup Fourier grid
    all_kx = 2 * jnp.pi * jnp.fft.fftfreq(Nx, 2 * jnp.pi / Nx)
    all_ky = 2 * jnp.pi * jnp.fft.fftfreq(Ny, 2 * jnp.pi / Ny)
    kx_mesh, ky_mesh = jnp.meshgrid(all_kx, all_ky)
    
    if k_max == None:
        k_max = int(1.5 * jnp.max(all_kx))
        
    abs_wavenumbers = jnp.sqrt(kx_mesh ** 2 + ky_mesh ** 2)
        
    # construct spectral energy
    vel_batch_ft = jnp.fft.fftn(vel_batch, axes=(1,2))
    
    ke_in_kxky = 0.5 * jnp.sum(
    jnp.abs(vel_batch_ft * vel_batch_ft.conj()),
    axis=-1
    ) / (jnp.array(Nx * Ny, dtype=jnp.float64) ** 2) 
    
    k_grid = []
    E_k = []
    for k in np.arange(0, k_max, dk):
        k_grid.append(k)
        kx_ky_indices = (abs_wavenumbers >= k) & (abs_wavenumbers < k + dk)
        E_k.append(jnp.sum(
        jnp.mean(ke_in_kxky[:, kx_ky_indices], axis=0)
        ))
        
    return k_grid, E_k

def A(vel_batch, dk=1,k_max=None):   
    if len(vel_batch.shape) == 3:
        vel_batch = vel_batch[jnp.newaxis, ...]
    
    _, Nx, Ny, _ = vel_batch.shape
    
    dk = 1.
    
    all_kx = 2 * jnp.pi * jnp.fft.fftfreq(Nx, 2 * jnp.pi / Nx)
    all_ky = 2 * jnp.pi * jnp.fft.fftfreq(Ny, 2 * jnp.pi / Ny)
    kx_mesh, ky_mesh = jnp.meshgrid(all_kx, all_ky)
    
    if k_max == None:
        k_max = int(1.5 * jnp.max(all_kx))
        
    abs_wavenumbers = jnp.sqrt(kx_mesh ** 2 + ky_mesh ** 2)
        
    # construct spectral energy
    vel_batch_ft = jnp.fft.fftn(vel_batch, axes=(1,2))
    
    ke_in_kxky = 0.5 * jnp.sum(
    jnp.abs(vel_batch_ft * vel_batch_ft.conj()),
    axis=-1
    ) / (jnp.array(Nx * Ny, dtype=jnp.float64) ** 2) 
    
    k_grid = []
    E_k = []
    for k in np.arange(0, k_max, dk):
        k_grid.append(k)
        kx_ky_indices = (abs_wavenumbers >= k) & (abs_wavenumbers < k + dk)
        E_k.append(jnp.sum(
        jnp.mean(ke_in_kxky[:, kx_ky_indices], axis=0)
        ))
    return k_grid, E_k


