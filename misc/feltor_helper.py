#!/opt/local/bin/python
#-*- Encoding: UTF-8 -*-

import numpy as np

def make_rz_array(geom):
    """
    Input:
        geom:   dictionary with simulation details
        geom['xrg']:  Radial coordinate   (r)
        geom['yrg']:  Poloidal coordinate (Z)

    Output:
        res:          ndarray, dim0: r-coordinate. dim1: z-coordinate
    """
    # Radial position is along x
    rrg = geom['xrg']
    # Poloidal position is along y
    Zrg = geom['yrg']

    rr, zz = np.meshgrid(rrg, Zrg)

    res = np.concatenate([rr[:, :, np.newaxis], zz[:, :, np.newaxis]], axis=2)
    # res is a Ny * Nx * 2 matrix
    # axis0 is the poloidal pixel coordinate, 0 = bottom, Ny-1 = top
    # axis1 is the radial pixel coordinate, 0 = left, Nx-1 = right
    # axis2 is the (R, z) coordinate at the given pixel

    return res 


def find_sol_pixels(geom):
    """
    Returns all pixels in between the LCFS and the right
    boundary of the simulation domain.

    Input:   Geometry object form load_frames


    Output:  sol_px: ndarray, list of tuples identifying pixels in the SOL.
                     axis0: pixel, axis1 [pol_px, rad_px]
    """

    x_sol = geom['x_sol']
    xrg = geom['xrg']
    yrg = geom['yrg']
    xx, yy = np.meshgrid(xrg, yrg)

    gap_idx_mask = xrg > x_sol

    return np.argwhere(gap_idx_mask)


def find_sol_mask(geom):
    """
    Returns all pixels in between the LCFS and the right
    boundary of the simulation domain.

    Input:   Geometry object form load_frames


    Output:  sol_px: ndarray, list of tuples identifying pixels in the SOL.
                     axis0: pixel, axis1 [pol_px, rad_px]
    """

    x_sol = geom['x_sol']
    xrg = geom['xrg']
    yrg = geom['yrg']
    xx, yy = np.meshgrid(xrg, yrg)

    return(xrg > x_sol)


# End of file feltor_helper.py
