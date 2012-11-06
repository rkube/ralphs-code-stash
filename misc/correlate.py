#!/opt/local/bin/python
#-*- Encoding: UTF-8 -*-

"""
===============
cross correlate

..codeauthor.. Ralph Kube <ralphkube@googlemail.com>

Compute the cross-correlation C of two timeseries X(t), Y(t), defined as:

C_XY(tau) = E[ (X_t - mu_x)*(Y_t+tau - mu_y) ] / sigma_x sigma_y
C is even: C_XY(tau) = C_XY(-tau)

Where 
mu_t = E[ X_t ]
mu_y = E[ Y_t ]
are the mean of the timeseries X(t), Y(t) and the expectation value is defined as:

E[ ] = Sum_i=1^N t_i p_i

where t_i is the value of the timeseries at t=i.

"""

import numpy as np

def correlate( signal1, signal2, window_length, mode = 1 ):

    if ( np.size(signal1) != np.size(signal2) ):
        raise ValueError('signal1 and signal2 have to be the same length')
    if ( window_length > np.size(signal1) ):
        raise ValueError('window_length must be smaller than the signal')

    corr = np.zeros(2 * window_length + 1)
    
    # Compute correlation only for positive time lags., tau is the timelag
    corr[ window_length ] = ( (signal1-signal1.mean()) * (signal2-signal2.mean())).mean() / ( signal1.std() * signal2.std() )
    for tau in np.arange(1, window_length+1):
        if mode == 1:
            corr[ window_length + tau ]     = ( (signal1[:-tau] -signal1[:-tau].mean())  * (signal2[tau:] - signal2[tau:].mean())).mean() / ( signal1[:-tau].std() * signal2[tau:].std() )
        elif mode == 2:
            corr[ window_length + tau ]     = ( (signal1[:-tau] -signal1[:-tau].mean())  * (signal2[tau:] - signal2[tau:].mean())).mean() / ( signal1.std() * signal2.std() )
    corr[:window_length] = corr[-1:window_length:-1]
    
    return corr


def fwhm_estimate( corr, dt ):
    n = np.size(corr)

    for n_up in np.arange(n/2+1, n):
        if ( corr[n_up] < 0.5  ):
            break

    for n_down in np.arange( (n-1)/2, 0, -1 ):
        if ( corr[n_down] < 0.5 ):
            break

    return [n_down, n_up], dt*(n_up-n_down)










