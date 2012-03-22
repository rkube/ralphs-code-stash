#!/opt/local/bin/python
#-*- Encoding: UTF-8 -*-

"""
===============
cross correlate

..codeauthor.. Ralph Kube <ralphkube@googlemail.com>

Compute the cross-correlation C of two timeseries X(t), Y(t), defined as:

C_XY(tau) = E[ (X_t - mu_t)*(Y_t+tau - mu_y) ]

Where 
mu_t = E[ X_t ]
mu_y = E[ Y_t ]
are the mean of the timeseries X(t), Y(t) and the expectation value is defined as:

E[ ] = Sum_i=1^N t_i p_i

where t_i is the value of the timeseries at t=i and p_i the probablity.



"""

import numpy as np


def correlate( signal1, signal2, window_length ):

    if ( np.size(signal1) != np.size(signal2) ):
        raise ValueError('Both signals have to be the same size')

    if ( window_length > np.size(signal1) ):
        raise ValueError('Correlation window must be smaller than signal lengths')
        
    corr = np.zeros(2 * window_length + 1)
    
    # Normalize both signals to zero mean value
    signal1 = signal1 - signal1.mean()
    signal2 = signal2 - signal2.mean()

    # Compute correlation, tau is the timelag
    corr[ window_length ] = (signal1 * signal2).mean()
    for tau in np.arange(1, window_length+1):
        # tau < 0
        corr[ window_length - tau ]     = (signal1[tau:]  * signal2[:-tau]).mean()
        # tau > 0
        corr[ window_length + tau ]     = (signal1[:-tau] * signal2[tau:]).mean()
    
    return corr
