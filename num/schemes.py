#!/usr/bin/env python
#-*- Encoding: UTF-8 -*-

import numpy as np


"""
Numerical schemes, mostly taken from
Parallel Scientific Computing in C++ and MPI
G.Em Karniadakis, R.M. Kirby II, Cambridge University Press (2007)
"""


def ef_cd(CFL, uold, unew):
    """
    Euler-Forward/Centered-Difference Scheme for advection problem:
    dF/dt + U*dF/dx = 0
    p. 419
    """
    assert(CFL >= 0)
    assert(uold.size == unew.size)

    unew[1:-1] = uold[1:-1] - 0.5 * CFL * (uold[2:] - uold[:-2])
    unew[0] = uold[0] - 0.5 * CFL * (uold[1] - uold[-1])
    unew[-1] = uold[-1] - 0.5 * CFL * (uold[0] - uold[-2])

    return unew


def ef_ud(CFL, uold, unew):
    """
    Euler-Forward/First order upward scheme for advection problem.
    See p. 421
    """
    assert (CFL > 0.)
    assert(uold.size == unew.size)

    unew[1:] = uold[1:] - CFL * (uold[1:] - uold[:-1])
    unew[0] = uold[0] - CFL * (uold[0] - uold[-1])
    return unew


def ef_ud2(CFL, uold, unew):
    """
    Euler-Forward/Second order upward scheme for advection problem.
    See p. 426
    """

    assert(CFL > 0.)
    assert(uold.size == unew.size)

    unew[2:] = (1. - CFL * (1.5 - 0.5 * CFL)) * uold[2:] + \
        CFL * (2. - CFL) * uold[1:-1] +\
        (0.5 * CFL) * (CFL - 1.) * uold[:-2]

    unew[0] = (1. - CFL * (1.5 - 0.5 * CFL)) * uold[0] + \
        CFL * (2. - CFL) * uold[-1] +\
        (0.5 * CFL) * (CFL - 1.) * uold[-2]
    unew[1] = (1. - CFL * (1.5 - 0.5 * CFL)) * uold[1] + \
        CFL * (2. - CFL) * uold[0] +\
        (0.5 * CFL) * (CFL - 1.) * uold[-1]

    return unew


def lax_wendroff(CFL, uold, unew):
    """
    Lax-Wendroff scheme.
    Second order in space.
    """

    unew[1:-1] = uold[1:-1] - (CFL * 0.5) * (uold[2:] - uold[:-2]) +\
        (CFL * CFL * 0.5) * (uold[2:] - 2.0 * uold[1:-1] + uold[:-2])

    unew[0] = uold[0] - (CFL * 0.5) * (uold[1] - uold[-1]) +\
        (CFL * CFL * 0.5) * (uold[1] - 2.0 * uold[0] + uold[-1])

    unew[-1] = uold[-1] - (CFL * 0.5) * (uold[0] - uold[-2]) +\
        (CFL * CFL * 0.5) * (uold[0] - 2.0 * uold[-1] + uold[-2])

    return unew


def ab2_cd(CFL, uold1, uold2, unew):
    """
    Second-Order Adams-Bashforth time integrator, centered
    difference scheme.
    uold1: tlev-1
    uold2: tlev-2
    """

    unew[1:-1] = uold1[1:-1] - 0.75 * CFL * (uold1[2:] - uold1[:-2]) +\
        0.25 * CFL * (uold2[2:] - uold2[:-2])
    unew[-1] = uold1[-1] - 1.5 * CFL * (uold1[-1] - uold1[-2]) +\
        0.5 * CFL * (uold2[-1] - uold2[-2])

    return unew

#End of file schemes.py
