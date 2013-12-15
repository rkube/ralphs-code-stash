#!/usr/bin/env python
#-*- Encoding: UTF-8 -*-

import numpy as np

"""
Finite volume schemes for conservation laws

Schemes are of the type
U^{n+1}_{j} = U^{n}_{j} - C(F^{n}_{j-1/2} - F^{n}_{j+1/2})
with C = delta_t / delta_x
"""


def F_lxf(U, f_flux, C):
    """
    Lax-Friedrich scheme for non-linear hyperbolic PDEs:
    F^{n}_{j+1/2} = (f(U^{n}_{j} + f(U^{n}_{j+1}))/2 -
                        dx/dt (U^{n}_{j+1} - U^{n}_{j})/2
    F^{n}_{j-1/2} = (f(U^{n}_{j-1}) + f(^{n}_{j}))/2 -
                        dx/dt (U^{n}_{j} - U^{n}_{j-1})/2

    U^{n+1}_{j} = U^{n}_{j} - dt/dx[ (f(U^{n}_{j+1}) - f(U^{n}_{j-1}))/2 +
                  dx/dt (U^{n}_{j-1} - 2 U^{n}_{j} + U^{n}_{j-1})/2 ]
    """

    unew = np.zeros(U.size - 2, dtype='float64')
    unew[:] = C * 0.5 * (f_flux(U[2:]) - f_flux(U[:-2])) - \
        0.5 * (U[:-2] - 2. * U[1:-1] + U[2:])

    return unew


def F_godunov(U, f_flux, fmin=0.0):
    """
    Godunov scheme for conservation laws. Assumes that fmin is the global
    minimum of f_flux

    F_{j+1/2}^{n} = max{f(max(U_{j}^{n}, fmin)), f(min(U_{j+1}^{n}, fmin))}
    F_{j-1/2}^{n} = max{f(max(U_{j-1}^{n}, fmin)), f(min(U_{j}^{n}, fmin))}
    See Mishra, Finite volume schemes for scalar conservation laws

    Input:
    ======
    U:      array like, cell averages of old solution
    f_flux: callable, flux function
    fmin:   float, global minimum of f_flux

    U contains the cell averages of the weak solution including ghost points
    U[0]  is at x_l - 1/2 * delta_x
    U[-1] is at x_r + 1/2 * delta_x
    """

    F_jplus = np.zeros(U.size - 2, dtype='float64')
    F_jminus = np.zeros(U.size - 2, dtype='float64')

    # Compute F_{j+1/2} - F_{j-1/2} for j = 1...N-1
    # Compute F_{j+1/2}
    max1 = np.maximum(U[1:-1], fmin)
    #print 'max1=', max1
    min1 = np.minimum(U[2:], fmin)
    #print 'min1=', min1
    #print 'max1.size = %d, min1.size = %d, F_jplus.size = %d' % (max1.size,
    #                                                             min1.size,
    #                                                             F_jplus.size)
    F_jplus[:] = np.maximum(f_flux(max1), f_flux(min1))

    #print 'F_jplus = ', F_jplus
    # Compute F_{j-1/2}
    max1 = np.maximum(U[:-2], fmin)
    min1 = np.minimum(U[1:-1], fmin)
    F_jminus[:] = np.maximum(f_flux(max1), f_flux(min1))

    #print 'F_jminus = ', F_jminus
    #print 'F_godunov, F_jminus.size = %d, F_jplus.size=%d' % (F_jminus.size,
    #                                                          F_jplus.size)
    return (F_jplus - F_jminus)


def F_lax_wendroff(U, f_flux, f_prime, C):
    """
    Second order Lax-Wendroff scheme:
    F^{n}_{j+1/2} = 1/2(f(U^{n}_{j}) + f(U^{n}_{j+1})) -
        a^{n}_{j+1/2} dt / 2 dx (f(U^{n}_{j+1}) - f(U^{n}_{j}))

    Input:
    ======
    U:      array_like, cell averages of old solution
    f_flux: callable, flux function of the conservation law
    f_prime:callable, derivative of the flux function
    C: delta_t / delta_x
    """

    unew = np.zeros(U.size - 2, dtype='float64')
    f_fluxU = f_flux(U)
    a_jplus = f_prime(0.5 * (U[1:-1] + U[2:]))
    a_jminus = f_prime(0.5 * (U[:-2] + U[1:-1]))

    unew[:] = C * 0.5 * (f_fluxU[2:] - f_fluxU[:-2]) -\
        0.5 * (a_jplus * (f_fluxU[2:] - f_fluxU[1:-1]) -
            a_jminus * (f_fluxU[1:-1] - f_fluxU[:-2]))

    return unew


def minmod_N(X):
    """
    Minmod function
    Let X = [x_1, x_2, ..., x_N]. Then

                 / signum(x) min(|x|) if signum(x_1) = signum(x_2) = ...
    minmod(X) :=|
                 \  0                 else

    Input:
    ======
    X: array_like
    """

    pass


def minmod_2(a1, a2):
    """
    Same as minmod(X), but for 2 arguments.

    Input:
    ======
    a1, a2: float
    """
    if (np.sign(a1) != np.sign(a2)):
        return 0.0

    return np.sign(a1) * np.min([np.abs(a1), np.abs(a2)])


def minmod_3(a1, a2, a3):
    """
    minmod(X) funciton for 3 arguments

    Input:
    ======
    a1, a2, a3: float
    """

    if (np.sign(a1) != np.sign(a2)):
        return 0.0
    if (np.sign(a1) != np.sign(a3)):
        return 0.0
    if (np.sign(a2) != np.sign(a3)):
        return 0.0

    return np.sign(a1) * np.min([np.abs(a1), np.abs(a2), np.abs(a3)])


def F_transport(U, a, dt, dx, limiter='minmod'):
    """
    Update step for the advection equation U_t + a U_x = 0
    See Mishra, (5.18)
    U^{n+1}_{j} = U^{n}_{j} - a dt/dx(U^{n}_{j} - U^{n}_{j-1}) -
        a dt/2 dx (dx - a dt)(sigma^{n}_{j} - sigma^{n}_{j-1})

    Input:
    ======
    U:          array_like, cell averages of old solution
    a:          coefficient in the transport equation
    dx:         grid spacing
    dt:         time step
    limiter:    the slope limiter to use. One of:
                ['minmod', 'superbee', 'mc']
    """

    assert(limiter in ['minmod', 'superbee', 'mc'])
    unew = np.zeros(U.size - 2, dtype='float64')

    sigma_j = np.zeros(U.size - 2, dtype='float64')
    sigma_j1 = np.zeros(U.size - 2, dtype='float64')

    if limiter == 'minmod':
        sigma_j[:] = minmod_2((U[2:] - U[1:-1]) / dx,
                              (U[1:-1] - U[:-2]) / dx)
        sigma_j1[1:] = minmod_2((U[2:-1] - U[1:-2]) / dx,
                                (U[1:-2] - U[:-3]) / dx)
        sigma_j1[0] = 0.

    elif limiter == 'superbee':
        pass

    elif limiter == 'mc':
        pass

    part1 = a * dt * (U[1:-1] - U[:-2]) / dx
    part2 = 0.5 * dt * (dx - a * dt) * (sigma_j - sigma_j1) / dx

    unew[1:-1] = part1 + part2

    return unew

# End of file schemes.py
