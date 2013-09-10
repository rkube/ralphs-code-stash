#!/usr/bin/env python
#-*- Encoding: UTF-8 -*-

import numpy as np


def mse_mu_sn(A, delta_t, tau_d, tau_w, N):
    """
    Compute absolute and relative error on mean square estimator
    of a shot noise process.

    MSE(mu)_abs = 1/N <A>^2 tau_d / tau_w ( 1 + ... )
    MSE(mu)_rel = MSE(mu)_abs / <Phi>^2 with <Phi> = <A> tau_d / tau_w
    """
    alpha = delta_t / tau_d
    part1 = (A * A * tau_d) / (N * tau_w)
    #part2 = (np.exp(-N * delta_t / tau_d) - 1. +
    #         N * (1. - np.exp(-delta_t / tau_d))) / \
    #        (2.0 * N * np.sinh(delta_t / (2. * tau_d)) ** 2.0)
    #mse_mu_abs = part1 * (1. + part2)
    #mse_mu_rel = (1. + part2) * tau_w / (N * tau_d)
    mse_mu_abs = part1 * (1. + sum_ij(tau_d, tau_w, alpha, N) / N)
    mse_mu_rel = (1. + sum_ij(tau_d, tau_w, alpha, N) / N) *\
        tau_w / (N * tau_d)

    return mse_mu_abs, mse_mu_rel


def mse_var_sn(A, delta_t, tau_d, tau_w, nrg):
    """
    Compute the absolute error on the variance estimator for a shot
    noise process.

    MSE(sigma^2) = ...
    """

    alpha = delta_t / tau_d
    alpha2 = alpha * alpha
    gamma = tau_d / tau_w
    gamma2 = gamma * gamma
    nrg2 = nrg * nrg

    print 'tau_d = %f, tau_w = %f, delta_t = %f, alpha = %f' % (tau_d, tau_w,
                                                                delta_t,
                                                                alpha)

    part1 = gamma * 6.0 * alpha + gamma2 * (+4.0 / alpha - 2.0 * alpha)

    part2 = gamma * alpha2 * (3. * np.exp(-2.0 * nrg / alpha) - 27.) + \
            gamma2 * ((4.0 * (np.exp(-nrg * alpha) - 1.0)) / alpha2 +
                      alpha2 * (1.0 +
                              12.0 * np.exp(-1.0 * nrg / alpha) +
                              - np.exp(-2.0 * nrg / alpha)))

    print 'tau_d = %f, tau_w = %f, alpha = %f, A = %f, part1 = %f, part2 = %f' %\
          (tau_d, tau_w, alpha, A, part1, part2[5])

    part1 = part1 / nrg
    part2 = part2 / nrg2
    #part1 = 0.0
    #part2 = 0.0

    result = A * A * A * A * (part1 + part2)
    return result


def sum_ij(tau_d, tau_w, alpha, N):
    result = (np.exp(-N * alpha) - 1. +
              N * (1. - np.exp(-alpha))) / \
             (2.0 * np.sinh(0.5 * alpha) ** 2.0)

    return (result)


def sum_i2j2(A, tau_d, tau_w, alpha, N):
    """
    terms from \sum_\limits_{i,j=1}^{N} <x_i x_i x_j x_j>
    see corr_fourpoint_integral.nb
    """

    expm1N = np.exp(-1.0 * N / alpha)
    expm2N = np.exp(-2.0 * N / alpha)
    N2 = N * N
    gamma = tau_d / tau_w
    # terms (tau_d / tau_w)^4
    tau_d4 = N2 * gamma * gamma * gamma * gamma

    # terms (tau_d / tau_w)^3
    tau_d3 = 2.0 * N2 + 8.0 * N * alpha +\
        8.0 * alpha * alpha * (expm1N - 1.0)
    tau_d3 = tau_d3 * gamma * gamma * gamma

    # terms (tau_d / tau_w)^2
    tau_d2 = N2 + 26.0 * N * alpha + \
        alpha * alpha * (-21.0 + 5.0 * expm2N + 16.0 * expm1N)
    tau_d2 = 0.5 * tau_d2 * gamma * gamma

    # terms (tau_d / tau_w)^1
    tau_d1 = 3.0 * alpha * (2.0 * N + (expm2N - 1.0) * alpha) * gamma

    #result = (A * A * A * A) * (tau_d4 + tau_d3 + tau_d2 + tau_d1)
    result = (A * A * A * A) * (tau_d4 + tau_d3 + tau_d2 + tau_d1)

    return (result)


def sum_ij2k(A, tau_d, tau_w, alpha, N):
    """
    terms from \sum_\limits{i,j,k=1}^{N} <x_i x_j x_j x_k>
    see eval_fourpoint_ijjk_integral.nb
    """

    expm1N = np.exp(-1.0 * N / alpha)
    expm2N = np.exp(-2.0 * N / alpha)
    expm3N = np.exp(-3.0 * N / alpha)
    N3 = N * N * N
    N2 = N * N
    gamma = tau_d / tau_w
    alpha2 = alpha * alpha
    alpha3 = alpha * alpha * alpha
    # terms (tau_d / tau_w)^4
    tau_d4 = 5.0 * gamma * gamma * gamma * gamma

    # terms (tau_d / tau_w)^3
    tau_d3 = ((5.0 / 6.0) * N3) + (8.5 * N2 * alpha) - 9.0 * N * alpha2 +\
        expm1N * (8.0 * N - alpha) * alpha2 + alpha3
    tau_d3 = tau_d3 * gamma * gamma * gamma

    # Terms (tau_d / tau_w)^2
    tau_d2 = (1. / 12.) * (4.0 * N3 +
                           90.0 * N2 * alpha +
                           12 * N * alpha2 -
                           159. * alpha3 -
                           45.0 * expm2N * alpha3 +
                           102. * expm1N * alpha2 * (N + 2 * alpha)
                           )
    tau_d2 = tau_d2 * gamma * gamma
    # Terms (tau_d / tau_W)
    tau_d1 = (alpha / 18.) * (36.0 * N2 + 90. * N * alpha -
                              alpha2 * (np.exp(-6.0 * N / alpha) +
                                        4.0 * expm3N +
                                        54.0 * expm2N -
                                        216.0 * expm1N +
                                        157.0))
    tau_d1 = tau_d1 * gamma

    result = (A * A * A * A) * (tau_d4 + tau_d3 + tau_d2 + tau_d1)

    return (result)


def sum_ijkl(A, tau_d, tau_w, alpha, N):
    """
    terms from \sum \limits_{i,j,k,l=1}^{N} <x_i x_j x_k x_l>
    see eval_fourpoint_ijkl_integral.nb
    """
    N2 = N * N
#    expp1N = np.exp(-1.0 * N / alpha)
    expm1N = np.exp(-1.0 * N / alpha)
    expm2N = np.exp(-2.0 * N / alpha)

    alpha2 = alpha * alpha
    alpha3 = alpha2 * alpha

    gamma = tau_d / tau_w

    # Terms ~ (tau_d / tau_w)^3
    tau_d3 = 12.5 * N2 * alpha * (N + (expm1N - 1.0) * alpha)
    tau_d3 = tau_d3 * gamma * gamma * gamma

    # Terms ~ (tau_d / tau_w)^2
    tau_d2 = 6.25 * alpha2 * (-2.0 * expm1N * alpha * (alpha - 5.0 * N) +
                              expm2N * alpha * (alpha - 2.0 * N) +
                              (5.0 * N2 - 8.0 * N * alpha + alpha2))
    tau_d2 = tau_d2 * gamma * gamma

    # Terms ~ (tau_d / tau_w)^1
    tau_d1 = (1. / 150.) * alpha * (75. * N2 * N +
                                    alpha * (75. * N * expm1N +
                                             375. * N2) +
                                    alpha2 * N * (300. * expm2N + 870.) +
                                    alpha3 * (20. * np.exp(-6. * N / alpha) -
                                              26. * np.exp(-5. * N / alpha) +
                                              820. * np.exp(-3. * N / alpha) -
                                              3175 * expm2N +
                                              6030 * expm1N -
                                              1920. * np.exp(-0.5 * N / alpha) -
                                              1749.))
    tau_d1 = tau_d1 * gamma

    result = (A * A * A * A) * (tau_d3 + tau_d2 + tau_d1)
    return (result)


# End of file estimator_errors.py
