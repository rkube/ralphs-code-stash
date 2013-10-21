#!/usr/bin/env python
#-*- Encoding: UTF-8 -*-


import numpy as np
from scipy.special import gamma as gamma_func

# Define commonly used PDFs


def sattin(n, p):
    """ Sattin Distribution, Sattin et al. Phys. of Plasmas 11, 5032(2004) """
    F0, sigma, n0, K = p
    L = 1. - np.log(n / n0) / K
    retval = F0 * np.exp(- 0.5 * np.log(L) ** 2 / (sigma * sigma)) * \
        n0 / (n * L)
    return retval


def lognorm(n, p):
    """ Lognormal distribution. """
    F0, n0, sigma = p
    retval = F0 * np.exp(-0.5 * (np.log(n / n0) / sigma) ** 2.) / n
    return (retval)


def gamma(Phi, p):
    """Gamma distribution for shot-noise process in terms of gamma and <A>. """
    Amean, g = p
    retval = 1. / (Amean * gamma_func(g)) * (Phi / Amean) ** (g - 1.) *\
        np.exp(-Phi / Amean)
    return retval


def gamma_ss(Phi, shape, scale):
    """ Gamma distribution for shot-noise process in terms of
        shape and scale parameter.
        shape: gamma = <Phi>/Phi_rms^2
        scale: Phi_rms^2 / <Phi>
    """
    print 'gamma_ss: shape = %f, scale = %f' % (shape, scale)
    retval = 1. / (gamma_func(shape) * scale) *\
        (Phi / scale) ** (shape - 1.) *\
        np.exp(-Phi / scale)
    return retval




# End of file pdfs.py
