#!/opt/local/bin/python
#-*- Encoding: UTF-8 -*-

import numpy as np

def r_over_s(signal, mode='increments', npts_min=1):
    """
    Compute rescaled range analysis on the signal.
    Cut the signal in 1, 2, 4, ..., equally long sub arrays.
    Compute the average Range over Std 
   
    Input:
    ======
    signal:     ndarray, float. Signal for which to compute R/S
    mode:       string. Compute R/S for either
                        increments:  z_i = y_i - y_{i-1}
                        signal:      z_i = y_i
                        cumulative:  z_i = sum_{j=1}^{i} y_i
    npts_min:   int. Gives the smallest interval size on which to compute R/S via
                     min_num_pts = 2 ** npts_min. Must be positive as to compute standard deviation

    Output:
    =======
    N_range:    ndarray, float. Number of points for which average R/S has been computed
    RS:         ndarray, float.
    """

    assert mode in ['increments', 'signal', 'cumulative']
    assert npts_min > 0
    # Cut the signal down to 2^N number of points
    N_max = int(np.floor(np.log2(signal.size)))
#    print 'N_max = %d' % N_max
    # Number of intervals the time series is divided up in
    N_range = np.arange(0, N_max + 1 - npts_min, 1)
    # Number of points in each interval
    num_pts_range = 2 ** (N_range[::-1] + npts_min)
    RS = np.zeros(N_max + 1- npts_min, dtype='float')

    signal = signal[:2**N_max]

    # Remove the mean
    signal = signal - signal.mean()
    # Use either cumulativ, increments or raw signal
    if mode is 'cumulative':
        z = signal.cumsum()
    elif mode is 'signal':
        pass
    elif mode is 'increments':
        z = np.zeros_like(signal)
        z[1:] = np.diff(signal)
        z[0] = 0.0

    # Iterate over number of sub intervals
    for N in N_range:
        # Number of intervals
        num_intervals = 2 ** N
        # Number of points in each interval
        num_pts = 2 ** (N_max - N)
        # Average R/S over the sub-intervals
        RS_sum = 0.0
        for nn in np.arange(num_intervals):
            #print 'interval %d: from %d..%d' % (nn, nn * num_pts, (nn + 1) * num_pts)
            interval = signal[nn * num_pts: (nn + 1) * num_pts]
            RS_sum += (interval.max() - interval.min()) / interval.std()
        RS[N] = RS_sum / float(num_intervals)
#        print 'N = %d, num_intervals = %d, num_pts = %d, R/S = %f' % (N, num_intervals, num_pts, RS[N])

#    print 'num_pts_range = ', num_pts_range
    print 'N_range = ', 2 ** N_range[::-1]
    print 'num_pts = ', num_pts_range[::-1]
    print 'RS      = ', RS[::-1]
#    print 'R over S = ', RS
    return num_pts_range[::-1], RS[::-1]

# End of file rs_analysis.py
