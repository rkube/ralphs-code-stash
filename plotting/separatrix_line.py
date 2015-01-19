#!/opt/local/bin/python
#-*- Encoding: UTF-8 -*-

import numpy as np
import numpy.ma as ma


def surface_line(sep_pixels, mode='max'):
    """
    Given the pixels which are mapped to the closed field line region,
    return the pixel with the largest radial coordinate for each
    poloidal coordinate.
    """

    # Index all pixels radially
    lin_array = np.repeat(np.arange(64), 64).reshape(64, 64).T

    # Apply the mask
    la_masked = ma.array(lin_array, mask=sep_pixels)

    # Return the maximum radial indices for each poloidal position
    if (mode == 'max'):
        return np.argmax(la_masked, axis=1)
    elif (mode == 'min'):
        return np.argmin(la_masked, axis=1)

# End of file separatrix_line.py
