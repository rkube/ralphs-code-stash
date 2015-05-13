#!/usr/bin/env python
#-*- Encoding: UTF-8 -*-

"""
Spectral routine prototypes used in 2dads
"""

import numpy as np

def d_dxi_dy(U, dx=dx, dy=dy):
    """
    Spectral derivation

    axis0 -> y
    axis1 -> x
    """

    kxrg = np.fft.fftfreq(U.shape[0], d=dx)
    kyrg = np.fft.fftfreq(U.shape[0], d=dy)

    kyrg = kyrg[:U.shape[0] / 2 + 1]



def solve_laplace(U, Lx):
    """
    Solves the laplace equation: U = del^2 A in spectral space
    Set zero mode to zero
    """

    N = np.shape(U)[0]
    # k_n = 2 pi n / L, n = 0, 1..., N/2-1, -N/2, -N/2-1, ..., -1
    kx = np.fft.fftfreq(N, d=Lx / (2. * np.pi * N))
    ky = kx[:N / 2 + 1]

    ky2, kx2 = np.meshgrid(ky ** 2.0, kx ** 2.0)
    k_grid = kx2 + ky2
    U_ft = np.fft.rfft2(U)

    # Set k_grid[0, 0] = 1.0 to avoid RuntimeWarning. Set A_ft[0,0] = 0 later
    # on
    k_grid[0, 0] = 1.0
    # Solve laplace equation
    A_ft = U_ft / k_grid
    # Set zero mode to zero
    A_ft[0, 0] = 0.0
    A = np.fft.irfft2(A_ft)
    return(A)



# End of file spectral.py
