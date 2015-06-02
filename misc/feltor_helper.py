#!/opt/local/bin/python
#-*- Encoding: UTF-8 -*-

import numpy as np

def make_rz_array(frame_info):
    """
    Input:
        frame_info:   dictionary with simulation details
        frame_info['xrg']:  Radial coordinate   (r)
        frame_info['yrg']:  Poloidal coordinate (Z)

    Output:
        res:          ndarray, dim0: r-coordinate. dim1: z-coordinate
    """

    rrg = frame_info['xrg']
    Zrg = frame_info['yrg']

    rr, zz = np.meshgrid(rrg, Zrg)

    res = np.concatenate([rr[:, :, np.newaxis], zz[:, :, np.newaxis]], axis=2)
    frame_info = 0

    return res, frame_info

# End of file feltor_helper.py
