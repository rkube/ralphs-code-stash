#!/usr/bin/env python
#-*- Encoding:UTF-8 -*-

import numpy as np


"""
Second order finite-difference schemes
Assume dim0 = y, dim1 = x
"""


def d_dx(src, dx=1.0, BC='per', axis=0):
    """
    Second order finit difference scheme for d/dx
    f'(x[i,:]) = (x[i + 1] - x[i - 1]) / (2dx)
    """

    assert(BC in ['per', 'dir', 'neu'])

    inv2dx = 1. / (2. * dx)
    result = np.zeros_like(src)
    if (axis == 0):
        result[1:-1, :] = src[2:, :] - src[:-2, :]
        if BC is 'per':
            result[0, :] = src[1, :] - src[-1, :]
            result[-1, :] = src[0, :] - src[-2, :]
        else:
            result[0, :] = result[1, :]
            result[-1, :] = result[-2, :]


    elif (axis == 1):
        result[:, 1:-1] = src[:, 2:] - src[:, :-2]
        if BC is 'per':
            result[:, 0] = src[:, 1] - src[:, -1]
            result[:, -1] = src[:, 0] - src[:, -2]
        else:
            result[:, 0] = result[:, 1]
            result[:, -1] = result[:, -2]


    result = result * inv2dx

    return result


def d2_dx2(src, dx=1.0, axis=0):
    """
    Second order finite difference scheme for d^2/dx^2
    f''(x[i]) = (x[i - 1] - 2 x[i] + x[i + 1]) / dx^2

    Takes derivative along first axis

    Periodic boundary conditions
    """

    invdx2 = 1. / (dx * dx)

    result = np.zeros_like(src)
    if(axis == 0):
        result[0, :] = src[-1, :] - 2. * src[0, :] + src[1, :]
        result[1:-1, :] = src[:-2, :] - 2. * src[1:-1, :] + src[2:, :]
        result[-1, :] = src[-2, :] - 2. * src[-1, :] + src[0, :]

    elif(axis == 1):
        result[:, 0] = src[:, -1] - 2. * src[:, 0] + src[:, 1]
        result[:, 1:-1] = src[:, :-2] - 2. * src[:, 1:-1] + src[:, 2:]
        result[:, -1] = src[:, -2] - 2. * src[:, -1] + src[:, 0]

    result = result * invdx2
    return result



# End of file fd.py
