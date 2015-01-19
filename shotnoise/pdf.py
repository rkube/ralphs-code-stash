#!/usr/bin/env python
#-*- Encoding: UTF-8 -*-


import numpy as np
from scipy.special import gamma as gamma_func

# Define commonly used PDFs


def sattin(n, params):
    """ Sattin Distribution, Sattin et al. Phys. of Plasmas 11, 5032(2004) """
    H, sigma, n0, K = params
    L = 1. - np.log(n / n0) / K
    retval = H * np.exp(- 0.5 * np.log(L) ** 2 / (sigma * sigma)) * \
        n0 / (n * L)
    return retval


def sattin_derivs(n, params):
    """
    Returns derivatives of the sattin distribution.
    See ~/uni/cmod/sheath_pdf_derivs.nb

    Returns:
        dP/dH
        dP/dsigma
        dP/dn0
        dP/dK


    """
    H, sigma, n0, K = params

    n_over_n0 = n / n0
    log_n_n0 = np.log(n_over_n0)

    # dp/dH:
    term1 = np.log((1.0 - log_n_n0 / K)) ** 2.0 / (2. * sigma * sigma)
    dPdH = np.exp(-term1) * K / (n_over_n0 * (K - log_n_n0))

    # dP/dsigma
    dPdn0 = np.exp(-term1) * H * K * (sigma ** 2.0 * (K - 1.0 -
                                                      np.log(n_over_n0))
                                      - np.log(1. - np.log(n_over_n0) / K))

    # dP/dsigma
    dPdsigma = np.exp(-term1) * H * K * n0 * np.log(1. - log_n_n0 / K) ** 2.0
    dPdsigma /= (sigma ** 3.0 * n * (K - log_n_n0))

    # dP/dK
    dPdK = np.exp(-term1) * H * n0 * log_n_n0 * (sigma ** 2.0 +
                                                 np.log(1. - log_n_n0 / K))
    dPdK /= (sigma ** 2.0 * n * (K - log_n_n0) ** 2.0)
    return np.array([dPdH, dPdsigma, dPdn0, dPdK]).T


def lognorm_s(n, params):
    """
    Lognormal distribution, definition from Sattin.
        Uses a superfluous parameter F0.
    """
    F0, n0, sigma = params
    retval = F0 * np.exp(-0.5 * (np.log(n / n0) / sigma) ** 2.) / n
    return (retval)


def lognorm(x, params):
    """
    Lognormal distribution, see:
    https://en.wikipedia.org/wiki/Log-normal_distribution.
    Connection to lognorm_s:
    F0 = 1 / (sqrt(2. * np.pi) * sigma)
    Insted of using this one, use the lognormal distribution in scipy.stats
    """
    shape, scale = params
    part1 = 1. / (np.sqrt(2. * np.pi) * x * shape)
    part2 = np.exp(-0.5 * np.log(x / scale) * np.log(x / scale)
                   / (shape * shape))
    return (part1 * part2)


#def gamma(Phi, p):
#    """Gamma distribution for shot-noise process in terms of gamma and <A>."""
#    Amean, g = p
#    retval = 1. / (Amean * gamma_func(g)) * (Phi / Amean) ** (g - 1.)*\
#        np.exp(-Phi / Amean)
#    return retval


#def gamma_ss(Phi, shape, scale):
def gamma_sn(x, params):
    """
    Gamma distribution for shot-noise process in terms of
    shape and scale parameter.
    shape: gamma = <Phi>^2/Phi_rms^2
    scale: Phi_rms^2 / <Phi>
    PDF(Phi) = 1 / (scale * Gamma(shape)) * (x/scale) ** (shape - 1) *
                exp(-x / scale)
    """
    shape, scale = params
    #print 'gamma_ss: shape = %f, scale = %f' % (shape, scale)
    retval = 1. / (gamma_func(shape) * scale) *\
        (x / scale) ** (shape - 1.) *\
        np.exp(-x / scale)
    return retval

# End of file pdfs.py
