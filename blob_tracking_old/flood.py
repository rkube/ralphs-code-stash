#!/opt/local/bin/python
#-*- Encoding: UTF-8 -*-

import numpy as np
import numpy.ma as ma
import scipy.ndimage.measurements as snm


"""
=========
flood
=========

.. codeauthor :: Ralph Kube <ralphkube@gmail.com>

Find connecte regions(blobs) and stuff

"""


def cut_blob(array, peak, thresh):
    """
    Cut out the area where the blob is 
    
    Input:
        array:  ndarray, GPI frame
        peak:   Index where blob is detected
        thresh: Fraction of peak above which connected pixels must be to constitute blob

    Output: 
        blob:   Array with cut out blob
    """
    print 'Regions containting peak at ', peak
    print peak[0], peak[1]   
 
    # Zero out all pixels of lower value than thresh * times 
    array[array < thresh] = 0    
 
    # Find connected regions
    regions, count = snm.label(array)
 
    # Select connected regions containing the peak
    select_region = regions[peak[0], peak[1]]
    print 'Peak at %d,%d is in region %d' % (peak[0], peak[1], select_region)

    # Build masked array 
    #print np.where(regions == select_region) 
    print regions == select_region

    
    result = ma.MaskedArray(array, mask = regions != select_region)

    return result


