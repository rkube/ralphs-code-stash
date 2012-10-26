#!/usr/bin/python


import numpy as np

def mk_kunique(Nx):
    #N = 1024

    #f_x = np.fft.fftfreq(Nx, 1./float(Nx))
    f_x = 2.0*np.pi*np.fft.fftfreq(Nx, 40./256.)
    #f_x = np.arange(Nx)

    # Meshgrid of frequencies squared
    f_xx, f_yy = np.meshgrid(f_x, f_x)
    f2_grid = f_xx**2 + f_yy**2
    # Unique frequencies in the grid
    f2_grid_unique = np.unique(f2_grid)
    
    # Given an index pair n,m describing a frequency k = k_x[n]**2 + k_y[m]**2, which index
    # in the array f2_grid_unique does this correspond to?

    k_idx = np.zeros([Nx,Nx]) 

    for n in np.arange(0,Nx):
        k_x = f_x[n]
        for m in np.arange(0,Nx):
            k_y = f_x[m]
            res =  np.where(f2_grid_unique == k_x**2 + k_y**2)
            k_idx[n,m] = np.squeeze(res)


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


