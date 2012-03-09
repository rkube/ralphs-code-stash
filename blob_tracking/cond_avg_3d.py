#!/opt/local/bin/python
#-*- Encoding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def cond_avg_top_peak_3d(signal, px, minmax, t0, lag, rel_idx=True):

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
    peak_idx_list = np.squeeze( np.argwhere( (tseries > minmax[0]) & (tseries <  minmax[1]) ) )

    # Create a structured array to sort the tuples (Amplitude(t_i, z_i, R_i), t_i, z_i, R_i 
    # by amplitude.
    pl = [(tseries[peak_idx[0], peak_idx[1], peak_idx[2]], peak_idx[0], peak_idx[1], peak_idx[2]) for peak_idx in peak_idx_list]
    
    dtype = [('value', 'f4'), ('tidx', 'i4'), ('zidx', 'i4'), ('ridx', 'i4')]
    peak_list = np.array(pl, dtype=dtype)
    # Sort the peaks by amplitude, largest first
    np.sort(peak_list, order='value')
    peak_list = peak_list[::-1]

    
# Debug, show the found peaks
    print '%d peaks, sorted after amplitude:' % ( np.shape(peak_list)[0] )
    for p in peak_list:
        print p
    
    # We now have the sorted amplitudes of all detected peaks in the trigger box.
    # Iterate over peaks sorted by magnitude, blank out those within the deadtime window
    # of a previous peak
    print peak_list['tidx'][:]
    for idx, peak in enumerate(peak_list):
        if ( peak['value'] == -1.0 ):   # Peak was blanked out, go to next one
            continue
        # Mark all indices within the specified lag to be blanked out
        blank_indices = np.argwhere(peak_list['tidx'][np.abs(peak_list['tidx'] - peak['tidx']) < lag])
#  Debug, write out which indices to blank
#        print 'tidx %d, peaks within lag:' % peak['tidx'] + ', ', peak_list['tidx'][blank_indices+idx]

        # Ignore the peak with index zero, as this is the current peak. Also, add the relative
        # array index idx to the peak indices that need to be blanked out
        peak_list['value'][blank_indices[blank_indices > 0]+idx] = -1.0

# Debug, plot what we have found
#    for peak in peak_list:
#        if ( peak['value'] > 0 ) :
#            plt.figure()
#            plt.title('tidx=%d, ridx=%d zidx=%d' % (peak['tidx'], peak['ridx'], peak['zidx']) )
#            plt.contourf( tseries[peak['tidx'], :, : ])
#            plt.plot( peak['ridx'], peak['zidx'], 'ko')
#            plt.colorbar()
    
    # Return only peaks that have not been blanked out
    return peak_list[peak_list['value'] > 0.0]
