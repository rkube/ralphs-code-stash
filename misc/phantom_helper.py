#!/opt/local/bin/python
#-*- Encoding: UTF-8 -*-

import numpy as np


"""
==============
phantom_frames
==============

.. codeauthor :: Ralph Kube <ralphkube@gmail.com>

Functions to work with phantom data
"""


def make_rz_array(frame_info):
    """
    Input:
        frame_info: dictionary with the following items:
        frame_info['tr_corner'] =   R,z position of the top right corner
        frame_info['br_corner'] =   R,z position of the bottom right corner
        frame_info['bl_corner'] =   R,z position of the bottom left corner
        frame_info['tl_corner'] =   R,z position of the top left corner
        frame_info['view_rot']  =   Rotation angle (degrees) through which the image should be 
                                    rotated in order that the image corresponds to the orientation 
                                    of the real object
        frame_info['ang']       =   Angle by which the frame needs to be rotated to make image horizontal
                                    and vertical
        frame_info['bot_pix']   =   Pixel value of the top edge after rotation by ANG
        frame_info['top_pix']   =   Pixel value of the bottom edge after rotation by ANG
        frame_info['rt_pix']    =   Pixel value of the right side after rotation by ANG
        frame_info['lt_pix']    =   Pixel value of the left side after rotation by ANG
        frame_info['x_px_size'] =   Number of pixels in x direction
        frame_info['y_px_size'] =   Number of pixels in y direction
        frame_info['view']      =   Name of the camera view
        frame_info['image_op']  =   Rotation that needs to be applied to frames. See MDS tree for more ino
    
    output:
        R_array, z_array    : ndarray with R and z values for each pixel
    """
    
    
    # We assume that the frame is correctly rotated
    # Define a linear transformation from pixel space into machine space
    # See /home/terry/gpi/phantom/retrieve_phantom_RZ_array.pro
    #
    # 1.) Choose an offset pixel P_x,0, P_y,0, for example the lower left corner of the phantom
    # image
    # Then define the linear mapping M = [ [m1, m2], [m3, m4] ]
    # ( R )   ( m1  m2 )    (P_x - P_x,0) + R_0
    # ( z ) = ( m3  m4 )  * (P_y - P_y,0) + z_0
    # where R,z are the coordinates in machine space 
    # P_x is the pixel value relative to an offset pixel P_x,0, P_y,0 and
    # R_0, z_0 are the offset R,z coordinates for the chosen offset pixel
    #
    # Solve for m1..m4 by plugging in the top right and top left pixel for P_x, P_y. Those have
    # known coordinates R1,z1 and R2, z2
    # From the resulting 4 equations, compute m1..m4
    
    
    



