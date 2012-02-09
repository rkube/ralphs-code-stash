#!/opt/local/bin/python
#-*- Encoding: UTF-8 -*-

"""
=========
blobtrail
=========

.. codeauthor :: Ralph Kube <ralphkube@gmail.com>

A class that defines a blob event in a sequence of frames for phantom camera data

"""

import numpy as np
import matplotlib.pyplot as plt
from helper_functions import tracker, fwhm, com
import time


class blobtrail:
    """
    A realization of a blob event and its associated trail
    """
    
    
    def __init__(self, frames, event, frame0, tau_max = 7, thresh_dis = 8., fwhm_max_idx = 10, blob_ext = 8, thresh_amp = 0.6, doplots = False):

        # Store all the parameters this blob has been tracked with.
        self.event = event                  # Time-, x-, and y-index of recorded peak
        self.tau_max = tau_max              # Maximum frames used for blob tracking
        self.thresh_dist = 8.               # Maximum distance a peak is allowed to travel per frame
        self.fwhm_max_idx = 10              # Maximum width allowed
        self.blob_ext = 8                   # Size of the blob: blob_center +- blob_ext pixel
        self.thresh_amp = thresh_amp        # Threshold in percentage of original amplitude before a blob is considered lost
        self.frame0 = frame0                # Offset frame for frames array passed
        self.dt = 2.5e-6                    # Sampling time
        
        
        # Error flags that signal something went wrong with blob tracking
        self.invalid_fw_tracking = False
        self.invalid_bw_tracking = False
        
        # Track blob forwards and backwards, combine results
        self.track_backward(frames, doplots)         
        self.track_forward(frames, doplots)
        
        # If the blob cannot be tracked forward and backwards, abort
        if ( self.invalid_fw_tracking and self.invalid_bw_tracking ):
            raise ValueError('Could not track blob, invalid event')

        # Combine results from forward and backward tracking
        # Values about the blob path:        
        # Frames tracked forward and backward
        self.tau = np.arange( -self.tau_b, self.tau_f )         
        # Amplitude of the blob
        self.amp = np.concatenate( (self.amp_b[self.tau_b:0:-1], self.amp_f[:self.tau_f]) , axis=0 )
        # x- and y- position of the blob
        self.xycom = np.concatenate( (self.xycom_b[self.tau_b:0:-1,:], self.xycom_f[:self.tau_f,:]) , axis=0 )
        self.xymax = np.concatenate( (self.xymax_b[self.tau_b:0:-1,:], self.xymax_f[:self.tau_f,:]) , axis=0 )
        # The shape of the blob
        self.blob_shape = ( self.blob_shape_b + self.blob_shape_f ) / ( self.tau_b + self.tau_f )
        # Radial and poloidal width of the blob
        self.fwhm_ell_rad = np.zeros_like(self.amp)
        self.fwhm_ell_pol = np.zeros_like(self.amp)
        
    def track_backward(self, frames, doplots = False):
        """
        Track blob backward frame0 to beginning of frames
        """     
        
        try:
            self.tau_b, self.amp_b, self.xycom_b, self.xymax_b, fwhm_rad_idx_b, fwhm_pol_idx_b, self.blob_shape_b = \
                tracker(frames[:self.tau_max,:,:], self.event, self.thresh_amp, self.thresh_dist, self.fwhm_max_idx, self.blob_ext, 'backward', plots = doplots, verbose = False)
        except:
            print 'Could not track backward'
            self.invalid_bw_tracking = True
            
        # Delete FWHM, this class has routines to compute it again.
        

    def track_forward(self, frames, doplots = False):
        """
        Track blob forward from frame0
        """
        try:
            self.tau_f, self.amp_f, self.xycom_f, self.xymax_f, fwhm_rad_idx_f, fwhm_pol_idx_f, self.blob_shape_f = \
                tracker(frames[self.tau_max:,:,:], self.event, self.thresh_amp, self.thresh_dist, self.fwhm_max_idx, self.blob_ext, 'forward', plots = doplots, verbose = False)
        except:
            print 'Could not track forward'
            self.invalid_fw_tracking = True


    def plot_trail(self, frames, plot_com = False, plot_max = False, plot_shape = False, save_frames = False):
        """
        Plot the motion of the blob. The GPI frames are to be supplied externally
        
        Input:
            frames:         GPI data
            plot_com:       Mark the center of mass of the blob
            plot_max:       Mark the maximum of the blob
            plot_shape:     If available, mark the FWHM of the blob
            save_frames:    Save the frames
        
        """
        print 'Plotting the blob event from frame %d-%d' % ( self.event[1] + self.frame0 - self.tau_b, self.event[1] + self.frame0 + self.tau_f )

        for f_idx, tau in enumerate( np.arange( -self.tau_b, self.tau_f) ):
            plt.figure()
            plt.title('frame %05d' % ( self.event[1] + self.frame0 + tau) )  
            plt.contour (frames[ self.event[1] + self.frame0 + tau, :, :], 15, linewidths=0.5, colors='k')
            plt.contourf(frames[ self.event[1] + self.frame0 + tau, :, :], 15, cmap = plt.cm.hot, levels=np.linspace(0.0,3.5,32))
            plt.colorbar(ticks=np.arange(0.0, 3.5, 0.5), format='%3.1f')
    
            if ( plot_com == True ):
                plt.plot(self.xycom[f_idx, 1], self.xycom[f_idx, 0], 'wo')
                
            if ( plot_max == True ):
                plt.plot(self.xymax[f_idx, 1], self.xymax[f_idx, 0], 'w^')

        plt.show()
        
    def compute_fwhm(self, frames, rz_array = None, position = 'COM', norm = False, plots = False):
        """
        Computes the FWHM of the detected blob at its maximum
        
        Input:
            frames:         GPI data
            rz_array:       2d array with (R,z) value for each pixel. If omitted, computes FWHM in pixels
            position:       Compute FWHM at center of mass 'COM' or maximum 'MAX'
            norm:           Normalize intensity to maximum
        """
        
        assert ( position in ['COM', 'MAX'] )
        
        fwhm_rad_idx = np.zeros([self.tau_b + self.tau_f, 2], dtype='int')
        fwhm_pol_idx = np.zeros([self.tau_b + self.tau_f, 2], dtype='int')
        
        if ( position == 'COM' ):
            xy_off = self.xycom.astype('int')
            self.fwhm_computed = 'COM'
        elif ( position == 'MAX' ):
            xy_off = self.xymax.astype('int')
            self.fwhm_computed = 'MAX'
            
        # Compute the FWHM for each frame if the blob has sufficiently large distance from the
        # frame boundaries.
        for t, ttau in enumerate( self.tau ):
            t_idx = self.event[1] + self.frame0 + ttau

            if ( xy_off[t,:].min() < self.fwhm_max_idx or (64 - xy_off[t,:].max() ) < self.fwhm_max_idx):
                continue
            
            slice_pol = frames[t_idx, xy_off[t,0] - self.fwhm_max_idx : xy_off[t,0] + self.fwhm_max_idx, xy_off[t,1] ]
            slice_rad = frames[t_idx, xy_off[t,0], xy_off[t,1] - self.fwhm_max_idx : xy_off[t,1] + self.fwhm_max_idx ]
            if ( norm ):
                slice_pol /= slice_pol.max()
                slice_rad /= slice_rad.max()
                

            fwhm_rad_idx[t,:] = fwhm( slice_rad ) + xy_off[t,1] - self.fwhm_max_idx
            fwhm_pol_idx[t,:] = fwhm( slice_pol ) + xy_off[t,0] - self.fwhm_max_idx

            if ( rz_array == None ):
                self.fwhm_ell_rad[t] = fwhm_rad_idx[t,1] - fwhm_rad_idx[t,0]
                self.fwhm_ell_pol[t] = fwmh_pol_idx[t,1] - fwhm_pol_idx[t,0]
            else:
                self.fwhm_ell_rad[t] = (rz_array[xy_off[t,0], fwhm_rad_idx[t,1], 0] - rz_array[xy_off[t,0], fwhm_rad_idx[t,0], 0]) / 2.355
                self.fwhm_ell_pol[t] = (rz_array[fwhm_pol_idx[t,1], xy_off[t,1], 1] - rz_array[fwhm_pol_idx[t,0], xy_off[t,1], 1]) / 2.355

# Debugging of the expressions above
#                print 'poloidal:  x_off = ', xy_off[t,1], ' from r_idx = ', fwhm_pol_idx[t,1] ,' to ', fwhm_pol_idx[t,0]
#                print ' is ', rz_array[fwhm_pol_idx[t,1], xy_off[t,1], 1] , ' to ', rz_array[fwhm_pol_idx[t,0], xy_off[t,1], 1]

            if ( plots ) :
                plt.figure()
                plt.title('Cross sections at %s' % position)
                plt.plot( frames[ t_idx, xy_off[t,0], :], '.-', label='radial xsection')
                plt.plot( frames[ t_idx, :, xy_off[t,1]], '.-', label='poloidal xsection')
                plt.plot( fwhm_rad_idx[t,:], frames[t_idx, xy_off[t,0], fwhm_rad_idx[t,:]] , 'b--' )
                plt.plot( fwhm_pol_idx[t,:], frames[t_idx, fwhm_pol_idx[t,:], xy_off[t,1] ], 'g--' )
                plt.axvline( xy_off[t, 1], color='red')
                plt.axvline( xy_off[t, 0], color='red')
                plt.legend(loc='upper right')
        
        
    def get_frame0(self):
        """
        The index for the frame where the blob was detected
        """
        return self.tau_b
    # If a rz_array is passed, compute positions and velocities in R-Z space. Otherwise return
    # positions and velocities in pixel space
    def get_trail_com(self, rz_array = None):
        """
        Return the position of the blob COM. Either in pixel or in (R,Z) coordinates if rz_array
        is passed.
        """

        if ( rz_array == None ):
            return self.xycom
            
        return rz_array[ self.xymax[:,1].astype('int'), self.xymax[:,0].astype('int'), :]
        
        
    def get_trail_max(self, rz_array = None):
        """
        Return the position of the blob maximum. Either in pixel or in (R,Z) coordinates if rz_array
        is passed.
        """
        if ( rz_array == None ):
            return self.xymax
        
        # Remember xycom[:,1] is the radial (X) index which corresponds to R
        return rz_array[ self.xycom[:,1].astype('int'), self.xycom[:,0].astype('int'), :]
            
            
    def get_velocity_max(self, rz_array = None):
        """
        Return the velocity of the blob maximum. Either in pixel / frame of m/s when rz_array is given
        """
        assert (np.size(self.tau) > 1), 'Cannot compute blob velocity with only one frame recognized'
        if ( rz_array == None ):        
            return self.xymax[1:, :] - self.xymax[:-1, :]
            
        trail = self.get_trail_max( rz_array )
        return 100*( trail[1:, :] - trail[:-1, :] ) / self.dt
        
    
    def get_velocity_com(self, rz_array = None):
        """
        Return the velocity of the blob COM. Either in pixel / frame of m/s when rz_array is given
        """

        assert (np.size(self.tau) > 1), 'Cannot compute blob velocity with only one frame recognized'
        if ( rz_array == None ):
            return self.xycom[1:, :] - self.xycom[:-1, :]
        
        trail = self.get_trail_com( rz_array )
        return 100*( trail[1:, :] - trail[:-1, :] ) / self.dt
        
        
    def get_ell_pol(self):
        """
        Return the previously computed poloidal width of the blob
        """
        return self.fwhm_ell_pol
        
    def get_ell_rad(self):
        """
        Return the previously computed radial width of the blob
        """
        return self.fwhm_ell_rad
        
    def get_amp(self):
        """
        Return the amplitude (maximum intensity) of the blob
        """
        return self.amp
    
    def get_blob_shape(self, frames, frameno = None, position = 'COM'):
        """
        Return a the shape of the blob centered around its COM position
        position:   Return blob position at center of mass ('COM') or maximum ('MAX')
        frameno:    Returns the blob shape in the specified range, this range must be within [-tau_b : tau_f]
        """
        
        assert( position in ['COM', 'MAX'] )
        
        if ( frameno != None ):
            assert ( isinstance( frameno, np.ndarray ) )
            assert ( frameno.max() <= self.tau_f )
            assert ( frameno.min() >= -self.tau_b )
            
            blob_shape = np.zeros([ np.size(frameno), 2*self.blob_ext, 2*self.blob_ext])
            t_off = frameno

        else:
            blob_shape = np.zeros([self.tau_b + self.tau_f, 2*self.blob_ext, 2*self.blob_ext])
            t_off = self.tau        

        if ( position == 'COM' ):
            x_off = self.xycom[self.tau_b + t_off, 1].astype('int')
            y_off = self.xycom[self.tau_b + t_off, 0].astype('int')
        elif ( position == 'MAX' ):
            x_off = self.xymax[self.tau_b + t_off, 1].astype('int')
            y_off = self.xymax[self.tau_b + t_off, 0].astype('int')
        
        for t_idx, t in enumerate(t_off):
            blob_shape[t_idx, :, :] = frames[t + self.event[1] + self.frame0, y_off[t_idx] - self.blob_ext : y_off[t_idx] + self.blob_ext, x_off[t_idx] - self.blob_ext : x_off[t_idx] + self.blob_ext]
    
        return blob_shape
 
    