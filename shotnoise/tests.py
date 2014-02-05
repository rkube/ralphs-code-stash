#!/usr/bin/env python
#-*- Encoding: UTF-8 -*-


"""
Implementation of statistical goodness
of fit statistics for distributions whose
parameters were determined by log-likelihood
method.
"""

import numpy as np


def ad(x, cdf):
    """
    Anderson-Darling statistics for data coming from a particular distribution.

    Parameters
    ----------
    x : array_like
        array of sample data
    cdf : callable
    """

    #y = np.sort(x)
    N = float(x.size)
    z = cdf(x)
    z.sort()
    z1 = z[::-1]

    i = np.arange(1.0, N + 1.0, 1.0)
    S = ((2.0 * i - 1.0) * (np.log(z) + np.log(1.0 - z1))).sum(axis=0) / N
    A2 = -N - S
    return(A2)


def cvm(x, cdf):
    """
    Cramer-von Mises statistics for data coming from a
    particular distribution.

    Parameters
    ----------
    x: array_like
       array of sample data
    cdf: callable
       CDF of the distribution at to be tested
    """

    #y = np.sort(x)
    N = float(x.size)
    z = cdf(x)
    z.sort()
    i = np.arange(1., N + 1.0, 1.0)
    S = ((z - (i - 0.5) / N) ** 2.0).sum(axis=0)
    W2 = S + 1.0 / (12.0 * N)
    # Modified test statistic for upper tail percentage
    W2up = (W2 - 0.4 / N + 0.6 * N ** -2.0) * (1.0 + 1.0 / N)
    W2lo = (W2 - 0.03 / N) * (1. + 0.35 / N)
    return (W2up)
# End of file tests.py
