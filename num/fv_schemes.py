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


# End of file schemes.py
