#/opt/local/bin/python
#-*- Encoding: UTF-8 -*-

"""
=========
Kernels
========

.. codeauthor :: Ralph Kube <ralphkube@gmail.com>
2d smoothing kernels
"""

import numpy as np

def kern_gaussian(size, sizey = None):
    """ Normalized Gaussian Kernel for convolutions / smoothing """
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)

    x, y = np.mgrid[-size:size-1, -size:size-1]
    g = np.exp( -(x*x / float(size) + y*y / float(size) ) )
    return g 



