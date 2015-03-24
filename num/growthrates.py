#!/usr/bin/env python
#-*- Encoding: UTF-8 -*-

from numpy import sqrt

# Hasegawa Wakatani model
# assuming no Diffusion
# Dispersion relation:
# w^2 k^2 + i w C (1 + k^2) - i k_y C = 0
# => w = -i C (1 + k^2) / 2 k^2 +- sqrt( -C^2 (1+k^2)^2 /4k^2 + i C ky/k^2)
def gr_hw1(kx, ky, C):
    k2 = kx * kx + ky * ky

    omega_plus = -0.5j * C * (1. + k2) / k2 + \
            sqrt(-0.25 * C * C * (1. + k2) * (1. + k2) / k2 +
                 1.j * C * ky / k2 )

    omega_minus = -0.5j * C * (1. + k2) / k2 - \
            sqrt(-0.25 * C * C * (1. + k2) * (1. + k2) / k2 +
                 1.j * C * ky / k2 )

    return (omega_plus, omega_minus)


# Hasegawa Wakatani model,
# Camargo, Biskamp, Scott (1995)
def gr_hw_cm(kx, ky, C, D):
    k2 = kx * kx + ky * ky

    gamma = -0.5 * (C * (1. + k2) / k2 + 2. * D * k2 * k2 * k2) +\
        (C * (1. + k2) / (2. * sqrt(2.) * k2)) *\
        sqrt(1. + sqrt(1. + 16. * ky * ky * k2 * k2 / (C * C * (1. + k2) ** 4.0 )))

    return gamma


# Growthrate for Hasegawa-Wakatani model without diffusion
def gr_hw(kx, ky, c):
    k2 = kx ** 2 + ky ** 2
    g = -0.5j * c * c * (1. + k2) / k2
    rest = -1j * ky
    return (g + 1j * sqrt(g * g + rest), g - 1.j * sqrt(g * g + rest))


# Growthrate for HW model with diffusion
def gr_hw_diff(kx, ky, c, kappa, mu):
    k2 = kx * kx + ky * ky
    g = -0.5j * ((1. + k2) / k2 + (mu + kappa) * k2)
    rest = 1.j * ky / k2 - (mu * k2 - kappa) + kappa *mu * k2 * k2
    return (g + 1.j * sqrt(g * g / 4. + rest), g - 1.j * sqrt(g * g / 4. + rest))


# Growthrate for interchange model
def ic_growth(kx, ky, kappa, mu):
    k2 = kx * kx + ky * ky
    g = -0.5j * k2 * (kappa + mu)
    return (g + 1.j * sqrt(g * g / 4. + ky * ky / k2 - kappa * mu * k2 * k2),
            g - 1.j * sqrt(g * g / 4. + ky *ky / k2 - kappa * mu * k2 * k2))


# Rayleigh Bernard Model
# d n / dt + d phi / d y = D del^2 n
# d O / dt + d n / dy = D del^2 O
# O = del^2 phi
def gr_rb(kx, ky, D):
    k2 = kx * kx + ky * ky
    omega_plus = D * k2 + np.sqrt(-D * k2 * (k2 - 1.0) - ky * ky / k2)
    omega_minus = D * k2 - np.sqrt(-D * k2 * (k2 - 1.0) - ky * ky / k2)

    return (omega_plus, omega_minus)

# Growthrate for diffusion equation
def gr_diff(kx, ky, kappa):
    k2 = kx * kx + ky * ky
    return -1.j * kappa * k2


def gr_hv4(kx, ky, D_hv4):
    k2 = (kx * kx + ky * ky)
    return -1.j * D_hv4 * k2 * k2


def gr_hv6(kx, ky, D_hv6):
    k2 = (kx * kx + ky * ky)
    return -1.j * D_hv6 * k2 * k2 * k2


# End of file growthrates.py
