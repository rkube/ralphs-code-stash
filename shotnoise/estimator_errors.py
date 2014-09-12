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
    gamma2 = gamma * gamma
    n_rg2 = n_rg * n_rg

    part1 = (2. / (n_rg * alpha)) + (-5. + 8. * np.exp(-n_rg * alpha)
                                     + np.exp(-2. * alpha * n_rg)) / (n_rg2 * alpha2)
    part2 = (6. / (n_rg * alpha)) + (-27. + 3. * np.exp(-n_rg * alpha)) / (n_rg2 * alpha2)

    part1 = part1 * gamma2
    part2 = part2 * gamma

    #part1 = gamma2 * (2.0 / alpha) + gamma * 6.0 / alpha
    #part2 = gamma * (3. * np.exp(-2.0 * n_rg * alpha) - 27.) / alpha2 + \
    #        gamma2 * (-1. +
    #                  2. * np.exp(-2.0 * n_rg * alpha) -
    #                  12. * np.exp(-1.0 * n_rg * alpha))

    #part1 = part1 / n_rg
    #part2 = part2 / n_rg2

    result = A * A * A * A * (part1 + part2)
    return result


def mse_corr_mu_sigma(A, alpha, gamma, n_rg):
    """
    Compute the correlation between \hat{mu} and \hat{\sigma^2} for a shot
    noise process.
    See eval_threepoint_ijk.nb for calculations
    """
    alpha2 = alpha * alpha
    alpha3 = alpha2 * alpha
    gamma2 = gamma * gamma
    exp_alpha = np.exp(-alpha * n_rg)
    exp_two_alpha = np.exp(-2.0 * alpha * n_rg)

    part1 = gamma2 * (4. * (1. - exp_alpha) /
                      (alpha2 * n_rg * n_rg))
    part2 = gamma * (3. / (n_rg * alpha)
                     + (-17. + 4. * exp_alpha + exp_two_alpha)
                     / (2. * alpha2 * n_rg * n_rg)
                     + (9. - 12. * exp_alpha + 3. * exp_two_alpha)
                     / (alpha3 * n_rg * n_rg * n_rg))

    result = A * A * A * (part1 + part2)
    return result


def sum_ij(tau_d, tau_w, alpha, N):
    result = (np.exp(-N * alpha) - 1. + N * (1. - np.exp(-alpha))) / \
             (2.0 * np.sinh(0.5 * alpha) ** 2.0)

    return (result)



# End of file estimator_errors.py
