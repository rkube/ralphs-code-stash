#!/usr/bin/env python
#-*- Encoding: UTF-8 -*-

import numpy as np


"""
Create a gaussian profile and its y-derivative on a specified grid.
Used to test the LU solver for div(n div(phi))
"""


def init_fields(L, N, n0=1.0, amp=1.0, x0=0.0, y0=0.0,
                sigma_x=1.0, sigma_y=1.0):
    """
    Create a Gaussian profile:

        n(x,y) = np.exp( -(x-x_0)^2/2 sigma_x^2 - (y - y_0)^2 / 2 sigma_y^2)
        phi(x,y) = d n / d y
                 =  -(y - y_0)/sigma_y * exp( (x-x0)^2/2 sigma_x^2 - (y-y0)^2/2 sigma_y^2)
    """

    dx = 2.0 * L / float(N)

    x = np.arange(-L, L, dx)
    xx, yy = np.meshgrid(x, x)

    n = n0 + amp * np.exp(-(xx - x0) ** 2.0 / (2. * sigma_x * sigma_x) -
                          (yy - y0) ** 2.0 / (2. * sigma_y * sigma_y))

    phi = -((yy - y0) / sigma_y ** 2.0) * (n - n0) / amp

    return n, phi

# End of file init_fields.py
