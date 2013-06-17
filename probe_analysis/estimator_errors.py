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
    part1 = (A * A * tau_d) / (N * tau_w)
    part2 = (np.exp(-N * delta_t / tau_d) - 1. +
             N * (1. - np.exp(-delta_t / tau_d))) / \
            (2.0 * N * np.sinh(delta_t / (2. * tau_d) ** 2))
    mse_mu_abs = part1 * (1. + part2)
    mse_mu_rel = (1. + part2) * tau_w / (N * tau_d)

    return mse_mu_abs, mse_mu_rel

# End of file estimator_errors.py
