#!/opt/local/bin/python
#-*- Encoding: UTF-8 -*-

import numpy as np
from matplotlib.axes import Axes
from matplotlib.transforms import Transform, BboxTransformTo, Affine2D

"""
==============
phantom_frames
==============

.. codeauthor :: Ralph Kube <ralphkube@gmail.com>

Functions to work with phantom data

    make_rz_array       Compute r,z coordinates of GPI FOV
"""


def make_rz_array(frame_info):
    """
    Input:
        frame_info: dictionary with the following items:
        frame_info['tr_corner'] =   R,z position of the top right corner
        frame_info['br_corner'] =   R,z position of the bottom right corner
        frame_info['bl_corner'] =   R,z position of the bottom left corner
        frame_info['tl_corner'] =   R,z position of the top left corner
        frame_info['view_rot']  =   Rotation angle (degrees) through which
                                    the image should be rotated in order that
                                    the image corresponds to the orientation of
                                    the real object
        frame_info['ang']       =   Angle by which the frame needs to be
                                    rotated to make image horizontal and
                                    vertical
        frame_info['bot_pix']   =   Pixel value of the top edge after
                                    rotation by ANG
        frame_info['top_pix']   =   Pixel value of the bottom edge after
                                    rotation by ANG
        frame_info['rt_pix']    =   Pixel value of the right side after
                                    rotation by ANG
        frame_info['lt_pix']    =   Pixel value of the left side after
                                    rotation by ANG
        frame_info['x_px_size'] =   Number of pixels in x direction
        frame_info['y_px_size'] =   Number of pixels in y direction
        frame_info['view']      =   Name of the camera view
        frame_info['image_op']  =   Rotation that needs to be applied to
                                    frames. See MDS tree for more ino

    output:
        res             : The resulting array with R- and z-coordinates
        transform_data  : The offset vector and the computed transformation
                          matrix
    """

    # We assume that the frame is correctly rotated
    # Define a linear transformation from pixel space into machine space
    # See /home/terry/gpi/phantom/retrieve_phantom_RZ_array.pro
    #
    # 1.) Choose an offset pixel P_x,0, P_y,0, for example the lower left
    # corner of the phantom image
    # Then define the linear mapping M = [ [m1, m2], [m3, m4] ]
    # ( R )   ( m1  m2 )    (P_x - P_x,0)   ( R_0 )
    # ( z ) = ( m3  m4 )  * (P_y - P_y,0) + ( z_0 )
    # where R,z are the coordinates in machine space
    # P_x is the pixel value relative to an offset pixel P_x,0, P_y,0 and
    # R_0, z_0 are the offset R,z coordinates for the chosen offset pixel
    #
    # Solve for m1..m4 by plugging in the top right and top left pixel for
    # P_x, P_y. Those have known coordinates R1,z1 and R2, z2
    # From the resulting 4 equations, compute m1..m4

    print 'Computing rotation matrix'
    Rz0  = frame_info['bl_corner'][0]
    R0, z0 = frame_info['bl_corner'][0]     # Offset point
    R1, z1 = frame_info['tl_corner'][0]     # P1
    R2, z2 = frame_info['tr_corner'][0]     # P2
    R3, z3 = frame_info['br_corner'][0]     # P3
    px10 = 0            # distance along x-dimension from P1 to offset, in px
    px20 = 63           # distance along x-dimension from P2 to offset, in px
    py10 = 63           # distance along y-dimension from P1 to offset, in px
    py20 = 63           # distance along y-dimension from P2 to offset, in px

#   Debug, print corner positions of GPI camera FOV
#    print 'Bottom left corner at R,z = (%f,%f)' % ( R0, z0 )
#    print 'Bottom right corner at R,z = (%f,%f)' % ( R3, z3 )
#    print 'Top left corner at R,z = (%f,%f)' % ( R1, z1 )
#    print 'Top right corner at R,z = (%f,%f)' % ( R2, z2 )

    m1 = (py20 * (R1 - R0) - py10 * (R2 - R0)) / (px10 * py20 - px20 * py10)
    m2 = (px20 * (R1 - R0) - px10 * (R2 - R0)) / (py10 * px20 - py20 * px10)
    m3 = (py20 * (z1 - z0) - py10 * (z2 - z0)) / (px10 * py20 - px20 * py10)
    m4 = (px20 * (z1 - z0) - px10 * (z2 - z0)) / (py10 * px20 - px10 * py20)
    M = np.array([[m1, m3], [m2, m4]])

#   Debug, print coefficients
#    print 'm1 = %f' % m1
#    print 'm2 = %f' % m2
#    print 'm3 = %f' % m3
#    print 'm4 = %f' % m4

    # Define pixel indices
    x = np.arange(0, 64)
    y = np.arange(0, 64)
    xx, yy = np.meshgrid(x, y)

    # Concatenate the arrays storing the x- and y- pixel coordinate of each
    # array
    # The last index of px_idx_array gives the x- and y- pixel tuple
    px_idx_array = np.concatenate((xx[:, :, np.newaxis], yy[:, :, np.newaxis]),
                                  axis=2)
    #rz_idx_array = np.zeros_like(px_idx_array.astype('float32'))

    # Apply the rotation matrix m to each pixel
    res = np.dot(px_idx_array, M) + Rz0
    # res is a 64x64x2 matrix
    # axis0 is the poloidal coordinate, 0 = bottom, 63 = top
    # axis1 is the radial coordinate, 0 = left, 64 = right
    # axis2 is the (R,z) coordinate at the given pixel field0: R, field1: z


#   Debug, print corner positions of GPI camera FOV
#    assert ( R0 == res[0,0,0] )
#    assert ( z0 == res[0,0,1] )
#
    print 'Bottom left corner: From MDS R,z = (%f,%f), computed: (%f,%f)' % (R0, z0, res[0, 0, 0], res[0, 0, 1])
    print 'Bottom right corner at R,z = (%f,%f), computed: (%f,%f)' % (R3, z3, res[0, 63, 0], res[0, 63, 1])
    print 'Top left corner at R,z = (%f,%f), computed: (%f,%f)' % (R1, z1, res[63, 0, 0], res[63, 0, 1])
    print 'Top right corner at R,z = (%f,%f), computed: (%f,%f)' % (R2, z2, res[63, 63, 0], res[63, 63, 1])

    transform_data = [M, Rz0]
    return res, transform_data


def find_sol_pixels(s):
    """
    Returns the indices of the pixels in between the separatrix and the LCFS.

    s:    Processed separatrix data from IDL
          i.e. s = readsav('%s/separatrix.sav' % (datadir), verbose=False)
          see /usr/local/cmod/codes/efit/idl/efit_rz2rmid.pro
              /home/terry/gpi/phantom/retrieve_phantom_RZ_array.pro
              /home/rkube/IDL/separatrix.pro,

          s['rmid'] is a vector whose entries are the R-coordinate for each pixel

          The pixels which are in the SOL have R > R_sep and R < R_limiter
    """
    
    gap_idx_mask = ((s['rmid'].reshape(64, 64) > s['rmid_sepx']) &
                    (s['rmid'].reshape(64, 64) < s['rmid_lim']))

    return np.argwhere(gap_idx_mask)


def find_sol_mask(shotnr, frame_info=None, rz_array=None,
                  datadir='/Users/ralph/source/blob_tracking/test_data'):
    """
    Returns a mask for the pixels in between the separatrix and the LCFS.
    """
    s = readsav('%s/separatrix.sav' % (datadir), verbose=False)

    return ((s['rmid'].reshape(64, 64) > s['rmid_sepx']) &
            (s['rmid'].reshape(64, 64) < s['rmid_lim']))




# End of file phantom_helper.py
