#!/opt/local/bin/python
#-*- Encoding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def detect_peak_3d(signal, px, minmax, t0, lag, rel_idx=False):

    """
    Given a timeseries of 2d surfaces, identify peaks within a specified intervall that occur
    in a trigger subarray in the timeseries. Sort the peaks by amplitude and succesively eliminate 
    all peaks occuring within a deadtime after a peak.
    
    Input:
        signal  : ndarray, 3d array of surfaces
        px      : array-like, [Rl, Ru, zl, zu] bounding box of the trigger array
        minmax  : array-like, [min, max] range in which the peak must lie
        t0      : integer, ignore all peaks before this offset
        lag     : integer, deadtime for peak detection
        rel_idx : If true, return tidx, ridx, zidx as absolute indices to signal, not as relative
                  indices to the provided cutoffs

    Output
        peak_list   : Structured array ( Amplitude, tidx, ridx, zidx ) of detected peaks
    """
    
    # Crop the trigger window from the input signal
    tseries = signal[t0:, px[2] : px[3], px[0] : px[1]]

    # Identify position of the peaks in the timeseries falling in the specified range
    # We ignore the following peaks:
    # 1.) Peaks with lower amplitude that fall in the deadtime intervall after a peak was detected

    # Return value from np.argwhere are indices from tseries[lag:-lag]. 
    peak_idx_list = np.squeeze( np.argwhere( tseries > minmax[0] ) )
    
    # Create a structured array to sort the tuples (Amplitude(t_i, z_i, R_i), t_i, z_i, R_i 
    # by amplitude.
    pl = [(tseries[peak_idx[0], peak_idx[1], peak_idx[2]], peak_idx[0], peak_idx[1], peak_idx[2]) for peak_idx in peak_idx_list]
    
    dtype = [('value', 'f4'), ('tidx', 'i4'), ('zidx', 'i4'), ('ridx', 'i4')]
    peak_list = np.array(pl, dtype=dtype)
    # Sort the peaks by amplitude, largest first
    peak_list = np.sort(peak_list, order='value')[::-1]
    
    
# Debug, show the found peaks
#    print '%d peaks, sorted after amplitude:' % ( np.shape(peak_list)[0] )
#    for p in peak_list:
#        print p
    
    # We now have the sorted amplitudes of all detected peaks in the trigger box.
    # Iterate over peaks sorted by magnitude, blank out those within the deadtime window
    # of a previous peak

    for idx, peak in enumerate(peak_list):

        if ( peak['value'] == -1.0 ):   # Peak was blanked out, go to next one
            continue
        # Mark all indices within the specified lag to be blanked out and ignore the current index
        blank_indices = np.abs( peak_list['tidx'] - peak['tidx'] ) < lag
        blank_indices[idx] = False

        # Ignore the peak with index zero, as this is the current peak. Also, add the relative
        # array index idx to the peak indices that need to be blanked out
        peak_list['value'][blank_indices] = -1.0
        if ( rel_idx == False ):
            peak['ridx'] = peak['ridx'] + px[0]
            peak['zidx'] = peak['zidx'] + px[2]

# Debug, plot what we have found
#    for peak in peak_list:
#        if ( peak['value'] > 0 ) :
#            plt.figure()
#            plt.title('tidx=%d, ridx=%d zidx=%d' % (peak['tidx'], peak['ridx'], peak['zidx']) )
#            plt.contourf( tseries[peak['tidx'], :, : ])
#            plt.plot( peak['ridx'], peak['zidx'], 'ko')
#            plt.colorbar()
    
    # Return only peaks that have not been blanked out
    return ( peak_list[peak_list['value'] > 0.0] )

    
def detect_peak_2d(signal, px, minmax, t0, lag, rel_idx=True):

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

