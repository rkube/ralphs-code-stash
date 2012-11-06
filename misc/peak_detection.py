#!/usr/bin/python
#-*- Encoding: UTF-8 -*-

import numpy as np
import numpy.ma as ma

"""
==============
peak_detection
==============

..codeauthor :: Ralph Kube <ralphkube@gmail.com>


Set of functions for peak detection in time series
"""



def detect_peaks_1d( timeseries, timebase, delta_peak, threshold ):
    """
    detect_peaks_1d

    Find all local maxima in timeseries, exceeding threshold, separated by delta_peak
    
    Input:
    ========
    timeseries:     Timeseries to scan for peaks, np.ma.MaskedArray
    timebase:       Timebase of the timeseries, np.ndarray
    delta_peak:     Minimum separation of peaks, integer
    threshold:      Peaks have to exceed this value, integer
    

    Output:
    ========
    peak_idx_list:  Indices of peaks matching the input criteria, np.ndarray
    """

    # Make timeseries a masked array
    if ( ma.isMA(timeseries) == False):
        timeseries = ma.MaskedArray( timeseries )
    # Sort time series by magnitude. Use fill_value to sort masked values lowest
    max_values = ( timeseries.argsort( fill_value = -1.0) )[::-1]
    # Number of peaks exceeding threshold
    num_big_ones = np.sum( timeseries > threshold )
    # Cut off fluctuations not exceeding the threshold. This fails, if the whole array is masked out.
    # In this case, return no peaks
    try:
        max_values = max_values[ :num_big_ones ]
    except ValueError:
        print 'detect_peaks_1d: No peaks in the unmasked part of the array.'
        return np.array([])
       
    # Eliminate values exceeding the threshold within delta_peak of another

    for idx, mv in enumerate(max_values):
        if ( mv == -1 ):
            # Peak was discarded in previous iteration, continue with next item
            continue
        # Discard current peak if it is not a local maximum
        # Using < > operators on masked elements yields a float for which the | operator is undefined.
        try: 
            if ( (timeseries[mv-1] > timeseries[mv]) | (timeseries[mv+1] > timeseries[mv]) ):
                max_values[idx] = -1    
        except TypeError:
            pass # Do nothing, one of the neighbouring values is masked. Assume this is a local maximum
        except IndexError:  
            # This imlpies that one of mv-+1 are out of bounds. The peak is at the bounds of the domain and not treated as a local maximum
            print 'Out of bounds, skipping peak'
            max_values[idx] = -1
            
        # Identify peaks within delta_peak
        close_ones = np.squeeze( np.argwhere( np.abs( mv - max_values ) < delta_peak ) )
        try:
#           The line below should do the same as the for loop. Test this out later.
#            max_values[close_ones[close_ones != 0]] = -1
            for co in close_ones:
                if (max_values[co] == mv ):
                    # The current peak is also identified when iterating over all peaks. skip.
                    continue
                else:
                    max_values[co] = -1 
        except TypeError:
            # TypeError is thrown when we cannot iterate over close_ones
            # This implies there are no peaks close by.
            continue 
        # Also cut off values close to the boundary
    max_values[ max_values < delta_peak ] = -1
    max_values[ max_values > (np.size(timeseries) - delta_peak) ] = -1
    # Remove all entries equal to -1
    max_values = max_values[ max_values != -1 ]

    return max_values


