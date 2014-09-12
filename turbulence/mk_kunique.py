#!/usr/bin/python

import numpy as np


def mk_kunique(Nx, My, Lx=1.0, output=False):
    """
    Given an index pair (n,m) describing a frequency
    k = k_x[n]**2 + k_y[m]**2, which index
    in the array f2_grid_unique does this correspond to?
    """

    f_x = np.fft.fftfreq(Nx) * Nx
    f_y = np.fft.fftfreq(My) * My

    # Meshgrid of frequencies squared
    f_xx, f_yy = np.meshgrid(f_x, f_y)
    f2_grid = f_xx ** 2 + f_yy ** 2
    # Unique frequencies in the grid
    f2_grid_unique = np.unique(f2_grid)

    # Given an index pair (n,m) describing a frequency
    # k = k_x[n]**2 + k_y[m]**2, which index
    # in the array f2_grid_unique does this correspond to?

    k_idx = np.zeros([Nx, Nx])

    for n in np.arange(0, Nx):
        for m in np.arange(0, My):
            res = np.where(np.abs(f2_grid_unique -
                                  f_x[n] ** 2 - f_y[m] ** 2) < 1e-10)
            #print np.squeeze(res)
            k_idx[n, m] = np.squeeze(res)

    if output:
        print 'f2_grid:'
        print f2_grid
        print '======================================'
        print 'f2_grid_unique:'
        print f2_grid_unique
        print '======================================'
        print 'k_idx:'
        print k_idx

    return k_idx
#np.savez('k_unique_1024.npz', k_idx=k_idx)


if __name__ == '__main__':
    np.set_printoptions(linewidth=9999999)
    mk_kunique(256)

# End of file mk_kunique.py
