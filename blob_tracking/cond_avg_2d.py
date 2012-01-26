#!/opt/local/bin/python
#-*- Encoding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def cond_avg_top_peak_surface(signal, px, minmax, t0, lag, rel_idx=True):

    """
    Identify peaks in a time series of a given pixel in a series of 2d arrays. 
    Compute the conditional average surfaces in a time-lag around the events.

    Input:
        signal  : ndarray, 3d array of surfaces
        px      : array-like, [x,y] pixel of the time series to analyze
        minmax  : array-like, [min, max] range in which the peak must lie
        t0      : integer, ignore all peaks before this offset
        lags    : integer, window size for conditional averaging
        rel_idx : If true, return indices of events relative to t0. Return absolute indices otherwise

    Output
        avg_surf    : Averaged surfaces    
    """

    print 'Detecting peaks within [%f:%f] * rms at px (%d,%d)' % ( minmax[0], minmax[1], px[0], px[1] )
    
    idx_events = []                 # Indices where peaks are located
    
    # Convert input data to float and compute RMS
    tseries = signal[t0:, px[0], px[1]]

    # Copy and renormalize the original time series
    #tseries = np.zeros( np.shape(signal)[0] - t0 )
    #tseries[:] = ( tseries_original - tseries_original.mean() ) / tseries_original.std()
    
    # Identify position of the peaks in the timeseries falling in the specified range
    # We ignore the following peaks:
    # 1.) Early and late peaks in the time series where the specified window cannot be fitted around
    # 2.) Peaks that fall in the averaging window of another peak

    # Return value from np.argwhere are indices from tseries[lag:-lag]. To convert these indices
    # to indices for tseries, we have to add the offset lag
    peak_indices = np.squeeze (np.argwhere( (tseries[lag:-lag] > minmax[0]) & (tseries[lag:-lag] <  minmax[1]) )) + lag
        
    # Sort the peaks by amplitude, largest first: traverse the return from argsort from back to front
    peak_indices = peak_indices[np.argsort( tseries[peak_indices] )[::-1]]
        
    # Iterate over peaks sorted by magnitude, blank out those within the cond. avg. windows
    # of a previous peak
    for idx, peak in enumerate(peak_indices):
        if ( peak == 0 ):   # Peak was blanked out, go to next one
            continue
        # Blank out all other peaks within the specified cond. avg. window
        peak_indices[ np.abs(peak_indices - peak) < lag] = 0        
        idx_events.append(peak)        
        
    idx_events = np.array(idx_events)
    
    cavg_window = np.zeros([np.size(idx_events), 2*lag])
    for ctr, idx in enumerate(idx_events):
        cavg_window[ctr, :] = tseries[ idx-lag : idx+lag ]
        
    if ( rel_idx == True ):
        return cavg_window, idx_events
    else :
        # Return average windows and move from relative to absolute indices
        return cavg_window, idx_events + t0
