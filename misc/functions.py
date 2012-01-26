#/opt/local/bin/python
#-*- Encoding: UTF-8 -*-

"""
=========
functions
========

.. codeauthor :: Ralph Kube <ralphkube@gmail.com>

Definitions of mathematical functinos often used
"""

import numpy as np

def gauss2d( xx, yy, params ):
    """
    input:
        xx, yy  : ndarray, Coordinate grid
        A0      : float, Amplitude of Gaussion
        x0, y0  : float, x-, y-coordinate of the peak
        sigma_x, sigma_y    : float, width of Gaussion
        rot     : float, Rotation angle

    output:
        res     : ndarray, resulting Gaussian
    """
    A0, x0, y0, sigma_x, sigma_y, rot = params[1:]
    # Convert rot to degrees
    rot = 2.0 * 2.0 * np.pi * rot / 360.
    

    a = 0.5 * ( (np.cos(rot) / sigma_x )**2 + ( np.sin(rot) / sigma_y )**2 )
    b = -0.25 * np.sin( 2. * rot ) * ( 1./(sigma_x*sigma_x) + 1./(sigma_y*sigma_y)  ) 
    c = 0.5 * ( (np.sin(rot) / sigma_x)**2 + (np.cos(rot) / sigma_y)**2 )

    # The Matrix [a b, b c] must be positive definite, which is equal to having only pos.
    # Eigenvalues

    ev  = np.linalg.eigvalsh( np.array( [[a,b], [b,c]] ) )
    if ( np.min(ev)  < 0 ):
        raise NumberError('Eigenvalues of coefficient matrix are not positive')

    return( A0 * np.exp(-a*(xx-x0)*(xx-x0) - 2.*b*(xx-x0)*(yy-y0) - c*(yy-y0)*(yy-y0) ) )



def com( xx, yy, profile ):
    """
    Computes center-of-mass coordinates for the structure given in profile
    
    input:
        xx, yy  : 2d ndarray, Coordinate meshgrids
        profile : 2d ndarray, structure
    output:
        x_com   : float, x center-of-mass coordinate
        y_com   : float, y-center-of-mass coordinate
    """

    if ( np.ndim(xx) != 2 ):
        raise ValueError('Dimension of x input must be smaller than 2')
    if ( np.ndim(yy) != 2 ):
        raise ValueError('Dimension of x input must be smaller than 2')
    if ( np.ndim(profile) != 2 ):
        raise ValueError('Dimension of x input must be smaller than 2')

    x_com = np.sum( profile * xx ) / np.sum( profile )
    y_com = np.sum( profile * yy ) / np.sum( profile )

    return x_com, y_com









