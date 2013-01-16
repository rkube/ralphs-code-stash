#!/usr/bin/python
#-*- Encoding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def zero_crossing_ip( x, y, epsilon = 1e-6, title=None, show_plots = False):
    """ 
    Find the zero crossing of the function y given at interpolation points x.
    Assume the the function y has only one zero crossing
    """

    # If a function value is less then eps, return this
    if ( np.size(np.argwhere( np.abs(y - 0.5*epsilon) < epsilon )) == 1 ):
        print 'A function value is close to zero'
        return np.squeeze( (np.argwhere( np.abs(y - 0.5*epsilon) < epsilon ) ) )

    # When every value is larger or less then zero we don't have a crossing
    if ( (y>0).all() ):
        raise ValueError('Cannot determine a zero crossing for all positive values')
    elif ( (y<0).all() ):
        raise ValueError('Cannot determine a zero crossing for all negative values')

    # Find the indices left and right to the zero crossing
    x_idx = np.zeros(2, dtype = 'int')
    if (y[0] > 0):
        # Array goes from pos. to neg. 
        # Find index of first negative value
        x_idx[1] = (y < 0).argmax()
    elif ( y[0] < 0 ):
        # Array goes from neg. to pos.
        # Find index of first positive value
        x_idx[1] = (y > 0).argmax()
    x_idx[0] = x_idx[1] - 1

    # Solve for zero between indices crossing_idx[0,1] by linear interpolation
    y0 = y[x_idx[0]]
    y1 = y[x_idx[1]]
    n = (y1*x_idx[0] - y0*x_idx[1])/(x_idx[0] - x_idx[1]) 
    m = (y0 - n)/x_idx[0]
    x_zero = - n / m

    if show_plots:
        plt.figure()
        plt.title('Zero at %f, between %d and %d' % (x_zero, x_idx[0], x_idx[1]) )
        plt.plot( x, y, '.-')
        plt.plot( x_zero*(x[1]-x[0]), m*x_zero+n, 'rx', markersize=12)
        plt.show()

    return x_zero 
    

if __name__ == "__main__":

    y = np.arange(-5., 5.)
#    y[5] = 0.562
    print y

    zero_x = zero_crossing_ip( np.arange(10), y )
    print 'value at zero: ', zero_x
    plt.plot(np.arange(10), y, '.-')
    plt.plot(zero_x, [0.0], 'kx')
    plt.grid()
    plt.show()

