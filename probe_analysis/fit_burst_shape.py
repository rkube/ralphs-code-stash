#!/usr/bin/env python
#-*- Encoding: UTF-8 -*-

"""
Fit exponential wave forms on a burst
"""

import numpy as np
from scipy.optimize import leastsq


def fit_burst_shape(waveform, dt):
    """
    Fit an exponential rise and decay on a waveform:
        Phi(t) = theta(-t) exp(t / tau_rise) + theta(t) exp(-t / tau_decay)

    Input:
    ======
    waveform:   array_like, size: 2*N+1. Waveform with burst peak at element N
    dt:         timebase

    Output:
    =======
    tau_rise:   Rise time
    tau_decay:  Decay time
    """

    # Define fit function for rise / decay
    exp_func = lambda p, t: p[0] * np.exp(t / p[1])
    err_func = lambda p, y, t: np.abs(y - exp_func(p, t))

    burst_length = (waveform.size - 1) / 2
    burst_tau = np.arange(-burst_length, burst_length + 1) * dt

    burst_rise = waveform[:burst_length + 1]
    tau_rise = burst_tau[:burst_length + 1]
    min_rise = burst_rise.min()
    p0_rise = [1.0, 1.0]

    burst_fall = waveform[burst_length:]
    tau_fall = burst_tau[burst_length:]
    min_fall = burst_fall.min()
    p0_fall = [1.0, -1.0]

    # leastsq returns 1,2,3 or 4 if the fit was successfull
    rv_success = [1, 2, 3, 4]

    p_rise, success = leastsq(err_func, p0_rise,
                              args=(burst_rise - min_rise, tau_rise),
                              maxfev = 100)
    if success not in rv_success:
        err_str = 'Least squares fit on rise returned success=%d' % success
        raise ValueError(err_str)

    p_fall, success = leastsq(err_func, p0_fall,
                              args=(burst_fall - min_fall, tau_fall),
                              maxfev = 100)

    if success not in rv_success:
        err_str = 'Least squares fit on fall returned success=%d' % success
        raise ValueError(err_str)

    return [p_rise[1], p_fall[1]]


# End of file fit_burst_shape.py
