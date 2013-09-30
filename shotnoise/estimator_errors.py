#!/usr/bin/env python
#-*- Encoding: UTF-8 -*-

import numpy as np


#def mse_mu_sn(A, delta_t, tau_d, tau_w, N):
#    """
#    Compute absolute and relative error on mean square estimator
#    of a shot noise process.
#
#    MSE(mu)_abs = 1/N <A>^2 tau_d / tau_w ( 1 + ... )
#    MSE(mu)_rel = MSE(mu)_abs / <Phi>^2 with <Phi> = <A> tau_d / tau_w
#    """
#    alpha = delta_t / tau_d
#
#    part1 = (A * A * tau_d) / (N * tau_w)
#
#    mse_mu_abs = part1 * (1. + sum_ij(tau_d, tau_w, alpha, N) / N)
#    mse_mu_rel = (1. + sum_ij(tau_d, tau_w, alpha, N) / N) *\
#        tau_w / (N * tau_d)
#
#    return mse_mu_abs, mse_mu_rel

def mse_mu_sn(A, alpha, gamma, n_rg):
    """
    Compute error on the mean of a shot-noise signal
    MSE(mu) = N^(-1) <A>^2 gamma (1. + ...)
    """

    part1 = 1.
    part2 = (np.exp(-n_rg * alpha) - 1. + n_rg * (1. - np.exp(-alpha))) /\
            (2. * np.sinh(0.5 * alpha) ** 2)
    result = A * A * gamma * (part1 + part2 / n_rg) / n_rg
    return(result)


#def mse_var_sn(A, delta_t, tau_d, tau_w, nrg):
def mse_var_sn(A, alpha, gamma, n_rg):
    """
    Compute the absolute error on the variance estimator for a shot
    noise process.

    MSE(sigma^2) = ...
    """

    alpha2 = alpha * alpha
    #gamma = tau_d / tau_w
    gamma2 = gamma * gamma
    n_rg2 = n_rg * n_rg

    #print 'tau_d = %f, tau_w = %f, delta_t = %f, alpha = %f' % (tau_d, tau_w,
    #                                                            delta_t,
    #                                                            alpha)

    part1 = gamma2 * (2.0 / alpha) + gamma * 6.0 / alpha
    part2 = gamma * (3. * np.exp(-2.0 * n_rg * alpha) - 27.) / alpha2 + \
            gamma2 * (-1. +
                      2. * np.exp(-2.0 * n_rg * alpha) -
                      12. * np.exp(-1.0 * n_rg * alpha))

    part1 = part1 / n_rg
    part2 = part2 / n_rg2

    result = A * A * A * A * (part1 + part2)
    return result


def sum_ij(tau_d, tau_w, alpha, N):
    result = (np.exp(-N * alpha) - 1. + N * (1. - np.exp(-alpha))) / \
             (2.0 * np.sinh(0.5 * alpha) ** 2.0)

    return (result)



# End of file estimator_errors.py
