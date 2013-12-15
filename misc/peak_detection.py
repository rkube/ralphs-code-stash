#!/usr/bin/python
#-*- Encoding: UTF-8 -*-

import numpy as np
#import numpy.ma as ma

"""
==============
peak_detection
==============

..codeauthor :: Ralph Kube <ralphkube@gmail.com>


Set of functions for peak detection in time series
"""


def detect_peaks_1d(timeseries, delta_peak, threshold, peak_width=5):
    """
    detect_peaks_1d

    Find all local maxima in timeseries, exceeding threshold,
    separated by delta_peak

    Input:
    ========
    timeseries:     Timeseries to scan for peaks, np.ma.MaskedArray
    delta_peak:     Minimum separation of peaks, integer
    threshold:      Peaks have to exceed this value, integer
    peak_width:     Number of neighbouring elements a peak has exceed

    Output:
    ========
    peak_idx_list:  Indices of peaks matching the input criteria, np.ndarray



    """

    # Sort time series by magnitude.
    max_idx = np.squeeze(timeseries.argsort())[::-1]

    # Cut off peaks too close to the boundary
    max_idx = max_idx[max_idx > delta_peak]
    max_idx = max_idx[max_idx < np.size(timeseries) - delta_peak]

    max_values = np.zeros_like(timeseries[max_idx])
    max_values[:] = np.squeeze(timeseries[max_idx])

    # Number of peaks exceeding threshold
    num_big_ones = np.sum(timeseries > threshold)
    #print 'Total: %d elements, %d over threshold' %
    try:
        max_values = max_values[:num_big_ones]
        max_idx = max_idx[:num_big_ones]
    except:
        print 'detect_peaks_1d: No peaks in the unmasked part of the array.'
        return np.array([])

    # Mark the indices we need to skip here
    max_idx_copy = np.zeros_like(max_idx)
    max_idx_copy[:] = max_idx

    # Eliminate values exceeding the threshold within delta_peak of another
    #for idx, mv in enumerate(max_values):
    #print 'iterating over %d peaks' % ( np.size(max_idx))
    for i, idx in enumerate(max_idx):
        current_idx = max_idx_copy[i]
        if (max_idx_copy[i] == -1):
        #    print 'idx %d is zeroed out' % (idx)
            continue

        # Check if this value is larger than the surrounding values of the
        # timeseries. If it is, continue with the next index
        if (timeseries[current_idx] < timeseries[current_idx - peak_width:
                                                 current_idx + peak_width]).any():
            max_idx_copy[i] = -1
            continue

        # Zero out all peaks closer than delta_peak
        close_idx = np.abs(max_idx_copy - idx)
        close_ones = np.squeeze(np.where(close_idx < delta_peak)[0])
        max_idx_copy[close_ones] = -1
        # Copy back current value
        max_idx_copy[i] = max_idx[i]

    # Remove all entries equal to -1
    max_idx_copy = max_idx_copy[max_idx_copy != -1]
    max_idx_copy = max_idx_copy[max_idx_copy < np.size(timeseries)]
    return max_idx_copy



def detect_peaks_1d_old(timeseries, delta_peak, threshold, peak_width=5):
    """
    detect_peaks_1d

    Find all local maxima in timeseries, exceeding threshold,
    separated by delta_peak

    Input:
    ========
    timeseries:     Timeseries to scan for peaks, np.ma.MaskedArray
    delta_peak:     Minimum separation of peaks, integer
    threshold:      Peaks have to exceed this value, integer
    peak_width:     Number of neighbouring elements a peak has exceed

    Output:
    ========
    peak_idx_list:  Indices of peaks matching the input criteria, np.ndarray



    """

    # Make timeseries a masked array
    #if ( ma.isMA(timeseries) == False):
    #    timeseries = ma.MaskedArray(timeseries)
    # Sort time series by magnitude.
    # Use fill_value to sort masked values lowest
    #max_idx = np.squeeze((timeseries.argsort( fill_value = -1.0))[::-1])
    max_idx = np.squeeze(timeseries.argsort())[::-1]

    # Cut off peaks too close to the boundary
    max_idx = max_idx[max_idx > delta_peak]
    max_idx = max_idx[max_idx < np.size(timeseries) - delta_peak]

    max_values = np.zeros_like(timeseries[max_idx])
    max_values[:] = np.squeeze(timeseries[max_idx])
    #(timeseries.argsort( fill_value = -1.0))[::-1]

    # Number of peaks exceeding threshold
    num_big_ones = np.sum(timeseries > threshold)
    #print 'Total: %d elements, %d over threshold' %
    # (np.size(timeseries), num_big_ones)
    # Cut off fluctuations not exceeding the threshold. This fails
    # if the whole array is masked out.
    # In this case, return no peaks
    try:
        max_values = max_values[:num_big_ones]
        max_idx = max_idx[:num_big_ones]
    except:
        print 'detect_peaks_1d: No peaks in the unmasked part of the array.'
        return np.array([])

    # Mark the indices we need to skip here
    max_idx_copy = np.zeros_like(max_idx)
    max_idx_copy[:] = max_idx

    # Eliminate values exceeding the threshold within delta_peak of another
    #for idx, mv in enumerate(max_values):
    #print 'iterating over %d peaks' % ( np.size(max_idx))
    for i, idx in enumerate(max_idx):
        current_idx = max_idx_copy[i]
        if (max_idx_copy[i] == -1):
        #    print 'idx %d is zeroed out' % (idx)
            continue

        #print 'Treating peak  %d, value %f' % (idx, timeseries[idx])
        #if (mv < peak_width + 1):
        #    # This accounts for peaks too close to the boundary as
        #    # well as peaks
        #    # discarded in previous iterations. Continue with next item
        #    continue
        #if (np.abs(np.size(timeseries) - idx) < peak_width + 1):
        #    # Peak is too close to the upper boundary, skip
        #    continue
        # Discard current peak if it is not a local maximum
        # Using < > operators on masked elements yields a float for
        # which the | operator is undefined.

        #try:
        #    if (timeseries[idx] < timeseries[idx - peak_width:idx + peak_width + 1]).any():
        #        max_idx_copy[i] = -1
        #except TypeError:
        #    pass # Do nothing, one of the neighbouring values is masked.
        #         # Assume this is a local maximum
        #except IndexError:
        #    # This imlpies that one of mv-+1 are out of bounds. The peak is at
        #    # the bounds of the domain and not treated as a local maximum
        #    print 'Out of bounds, skipping peak'
        #    #max_values[idx] = -1
        #    max_idx_copy[i] = -1

        # Check if this value is larger than the surrounding values of the
        # timeseries. If it is, continue with the next index
        if (timeseries[current_idx] < timeseries[current_idx - peak_width:current_idx + peak_width]).any():
            max_idx_copy[i] = -1
            continue

        # Zero out all peaks closer than delta_peak
        close_ones = np.squeeze(np.where(np.abs(max_idx_copy - idx) < delta_peak)[0])
        #print 'found %d peaks close by' % ( np.size(close_ones))
        max_idx_copy[close_ones] = -1
        # Copy back current value
        max_idx_copy[i] = max_idx[i]

        #try:
        #    for co in close_ones:
        #        if (max_idx[co] == timeseries[idx] ):
        #            # The current peak is also identified when iterating
        #            # over all peaks. skip.
        #            continue
        #        else:
        #            max_idx_copy[co+idx] = -1
        #except TypeError:
        #    # TypeError is thrown when we cannot iterate over close_ones
        #    # This implies there are no peaks close by.
        #    continue
        # Also cut off values close to the boundary
    #max_values[ max_values < delta_peak ] = -1
    #max_values[ max_values > (np.size(timeseries) - delta_peak) ] = -1
    # Remove all entries equal to -1
    max_idx_copy = max_idx_copy[max_idx_copy != -1]
    max_idx_copy = max_idx_copy[max_idx_copy < np.size(timeseries)]
    return max_idx_copy

# End of file peak_detection.py
