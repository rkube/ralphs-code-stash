#!/opt/local/bin/python
#-*- Encoding: UTF-8 -*-

import numpy as np
from matplotlib.axes import Axes
from matplotlib.transforms import Transform, BboxTransformTo, Affine2D, IdentityTransform

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
        res             : The resulting array with R- and z-coordinates
        transform_data  : The offset vector and the computed transformation matrix
    """
    
    
    # We assume that the frame is correctly rotated
    # Define a linear transformation from pixel space into machine space
    # See /home/terry/gpi/phantom/retrieve_phantom_RZ_array.pro
    #
    # 1.) Choose an offset pixel P_x,0, P_y,0, for example the lower left corner of the phantom
    # image
    # Then define the linear mapping M = [ [m1, m2], [m3, m4] ]
    # ( R )   ( m1  m2 )    (P_x - P_x,0)   ( R_0 )
    # ( z ) = ( m3  m4 )  * (P_y - P_y,0) + ( z_0 )
    # where R,z are the coordinates in machine space 
    # P_x is the pixel value relative to an offset pixel P_x,0, P_y,0 and
    # R_0, z_0 are the offset R,z coordinates for the chosen offset pixel
    #
    # Solve for m1..m4 by plugging in the top right and top left pixel for P_x, P_y. Those have
    # known coordinates R1,z1 and R2, z2
    # From the resulting 4 equations, compute m1..m4
        
    print 'Computing rotation matrix'
    Rz0    = frame_info['bl_corner'][0]
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
    
    m1 = ( py20 * (R1 - R0) - py10 * (R2 - R0) ) / (px10 * py20 - px20 * py10)
    m2 = ( px20 * (R1 - R0) - px10 * (R2 - R0) ) / (py10 * px20 - py20 * px10)
    m3 = ( py20 * (z1 - z0) - py10 * (z2 - z0) ) / (px10 * py20 - px20 * py10)
    m4 = ( px20 * (z1 - z0) - px10 * (z2 - z0) ) / (py10 * px20 - px10 * py20)
    M = np.array([ [m1, m3], [m2, m4] ])

#   Debug, print coefficients
#    print 'm1 = %f' % m1
#    print 'm2 = %f' % m2
#    print 'm3 = %f' % m3
#    print 'm4 = %f' % m4

    # Define pixel indices
    x = np.arange(0, 64)
    y = np.arange(0, 64)
    xx, yy = np.meshgrid(x, y)

    # Concatenate the arrays storing the x- and y- pixel coordinate of each array
    # The last index of px_idx_array gives the x- and y- pixel tuple 
    px_idx_array = np.concatenate( (xx[:,:,np.newaxis], yy[:,:,np.newaxis]), axis=2 )
    rz_idx_array = np.zeros_like(px_idx_array.astype('float32'))
    
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
    print 'Bottom left corner: From MDS R,z = (%f,%f), computed: (%f,%f)' % ( R0, z0, res[0,0,0], res[0,0,1] )
    print 'Bottom right corner at R,z = (%f,%f), computed: (%f,%f)' % ( R3, z3, res[0,63,0], res[0,63,1] )
    print 'Top left corner at R,z = (%f,%f), computed: (%f,%f)' % ( R1, z1, res[63,0,0], res[63,0,1] )
    print 'Top right corner at R,z = (%f,%f), computed: (%f,%f)' % ( R2, z2, res[63,63,0], res[63,63,1] )

    transform_data = [M, Rz0]
    return res, transform_data
    
    
class GPI_projection(Axes):
    """
    Scale data from phantom camera from pixel space to object space
    See http://matplotlib.sourceforge.net/examples/api/custom_scale_example.html
    
    Scale function:
    ( R )   ( m1 m2 )   ( P_x - P_x,0 )   ( R0 )
    ( z ) = ( m3 m4 ) + ( P_y - P_y,0 ) + ( z0 )
    
    Inverse scale function:
    
    ( P_x - P_x,0 )   ( m1 m2 )^-1 ( R - R0 )
    ( P_y - P_y,0 ) = ( m3 m4 )    ( z - z0 )
    
    """

    name = 'GPI_projection'
    
    def __init__(self, fig, rect=None, *args, **kwargs):
        """
        To use this transformation, the 2x2 array M has to be passed. 
        """
        if rect == None:
            rect = [0.0, 0.0, 1.0, 1.0] 
        self.fig = fig
        
        if kwargs.has_key('M') == False:
            self.M = np.array([ [0.09365082, 0.00476183], [-0.0015873, 0.09920635] ])
        else:
            self.M = kwargs.pop('M')            
            
        if kwargs.has_key('Rz0') == False:
            self.Rz0 = np.array([85.90000153, -6.05000019])
        else:
            self.Rz0 = kwargs.pop('Rz0')
        
        Axes.__init__(self, fig, rect)
        self.set_aspect(1.0, adjustable='box', anchor='C')
        self.cla()
    
    def cla(self):
        Axes.cla(self)
        Axes.set_xlim(self, 0., 1. )
        Axes.set_ylim(self, 0., 1. )

    def _set_lim_and_transforms(self):
        print 'GPI_projection: setting limit and transforms'
        
        self.transProjection = Affine2D().translate(self.Rz0[0], 20.0)

        self.affine = Affine2D().scale(1./200., 1./200.)
        self.transAxes = BboxTransformTo(self.bbox)

        self.transData = self.transProjection + self.affine + self.transAxes
               
        xaxis_stretch = Affine2D().scale(200., 200.)
        yaxis_stretch = Affine2D().scale(200., 200.)
        self._xaxis_transform =  xaxis_stretch + self.affine + self.transAxes #self.transData 
        self._yaxis_transform =  yaxis_stretch + self.affine + self.transAxes #self.transData 
        
        self._xaxis_text1_transform = self._xaxis_transform
        self._xaxis_text2_transform = self._xaxis_transform
        self._yaxis_text_transform = self._yaxis_transform

    # Disallow interactive zooming and panning
    def can_zoom(self):
        return False
    
    def start_pan(self, x, y, button):
        pass

    def end_pan(self):
        pass
        
    def drag_pan(self, button, key, x, y):
        pass

    class GPI_projection_transform(Transform):
        input_dims = 2
        output_dims = 2
        is_separable = False

        def __init__(self, M, Rz0):
            Transform.__init__(self)
            self.M = M
            self.Rz0 = Rz0

        def transform(self, a):
            """
            This transformation takes a NxN ``numpy`` array and returns a
            transformed copy of the array.
            """           
#            print 'GPI_projection_transform: ', np.shape(a), a     
#            a = np.dot(a, self.M) + self.Rz0
#            print 'Transformed a:', np.shape(a), a

            return a
            
        def inverted(self):
            """
            The inverse transform
            """
            return GPI_projection.GPI_projection_inverse_transform(self.M, self.Rz0)
 
    class GPI_projection_inverse_transform(Transform):
        intput_dims = 2
        output_dims = 2
        is_separable = False
        
        def __init__(self, M, Rz0):
            Transform.__init__(self)
            
            self.M = M
            self.M_inv = np.linalg.inv(M)
            self.Rz0 = Rz0
#            print 'GPI_inverse_transform, M = ', self.M
#            print 'GPI_inverse_transform, M_inv = ', self.M_inv
#            print 'GPI_inverse_transform, Rz0 = ', self.Rz0
#            print '----'
                
        def transform(self, a):
            """
            Takes a NxN ``numpy`` array and returns its inverse transformation.
            That is, go from object to pixel space
            """
            
#            print 'GPI_projection_inverse_transform: ', np.shape(a), a
#            print 'GPI_projection_inverse_transform: ', np.shape(self.Rz0), self.Rz0
# 
#            return (np.dot( self.M_inv * a  ) )
#           print np.shape( a - self.Rz0 )
#           for jy in np.arange(np.shape(a)[0]):
#               for ix in np.arange(np.shape(a)[1]):
#                   a[jy, ix, :] = a[jy, ix, :] - self.Rz0[:]
#           a = np.dot( self.M_inv * ( a - self.Rz0 ))
#                   a[jy, ix, :] = np.sum( self.M_inv * ( a[jy, ix, :] - self.Rz0 ), axis = 1) 
            return a
            
        def inverted(self):
            """
            The inverse of this transformation is given by the rotation matrix M
            """
            return GPI_projection.GPI_projection_transform(self.M, self.Rz0)