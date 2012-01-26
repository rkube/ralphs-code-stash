#1/opt/local/bin/python
#-*- Encoding: UTF-8 -*-

"""
==========
box_detect
==========

.. codeauthor :: Ralph Kube <ralphkube@gmail.com>

Detect peaks(blobs) in a box of the domain
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import maximum_filter

def peak_list(coords, profile, peak_width = 4, smoothed = True):
    """
    detects all local maxima in a rectangular subdomain of profile.
    
    input:
        coords      : ndarray, [x0, y0, x1, y1] : Lower x, lower y, upper x, upper y coordinate of bounding box
        profile     : 64x64 profile
        size        : Size of the peak
        smoothed    : True, if profile is smoothed
    output:
        maxima  : list of local maxima
    """

    if ( np.max(coords) > np.max(np.shape(profile)) ):
        raise ValueError('Detection box may not be larger than profile') 

    foo = np.zeros_like( profile )
    result = maximum_filter( profile, size=peak_width   )

    return result

