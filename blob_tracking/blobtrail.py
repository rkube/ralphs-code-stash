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
import matplotlib.patches as mpatches
from scipy.interpolate import griddata
from scipy.optimize import leastsq
from misc.helper_functions import tracker, fwhm, com
from plotting.separatrix_line import surface_line
#from helper_functions import tracker, fwhm, com
#from separatrix_line import surface_line


class blobtrail:
    """
    A realization of a blob event and its associated trail
    """
    
    
    def __init__(self, frames, event, frame0, shotnr, tau_max = 7, thresh_dist = 8., fwhm_max_idx = 10, blob_ext = 8, thresh_amp = 0.6, doplots = False):

        # Store all the parameters this blob has been tracked with.
        self.event = event                  # Time-, x-, and y-index of recorded peak
        self.tau_max = tau_max              # Maximum frames used for blob tracking
        self.thresh_dist = thresh_dist      # Maximum distance a peak is allowed to travel per frame
        self.fwhm_max_idx = fwhm_max_idx    # Maximum width allowed
        self.blob_ext = blob_ext            # Size of the blob: blob_center +- blob_ext pixel
        self.thresh_amp = thresh_amp        # Threshold in percentage of original amplitude before a blob is considered lost
        self.frame0 = frame0                # Offset frame for frames array passed
        self.dt = 2.5e-6                    # Sampling time
        self.shotnr = shotnr
        
#        print 'Creating blobtrail event, t0 = %d' % (self.event[1]) , 'lalala', self.event[1]
        
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
        # Frames tracked forward and backward. If no frames are tracked forwards, self.tau_f = 0, but
        # we need to include the frame where the blob is detected. Thus, use max(1, self.tau_f) for 
        # indexing
        # Use np.max... to make sure, 0 is always included in self.tau
        self.tau = np.arange( -self.tau_b, max(self.tau_f, 1) )   
        # Amplitude of the blob
        self.amp = np.concatenate( (self.amp_b[self.tau_b:0:-1], self.amp_f[:max(1, self.tau_f)]) , axis=0 )
        # x- and y- position of the blob
        self.xycom = np.concatenate( (self.xycom_b[self.tau_b:0:-1,:], self.xycom_f[:max(1, self.tau_f),:]) , axis=0 )
        self.xymax = np.concatenate( (self.xymax_b[self.tau_b:0:-1,:], self.xymax_f[:max(1, self.tau_f),:]) , axis=0 ).astype('int')
        # The shape of the blob
        self.blob_shape = ( self.blob_shape_b + self.blob_shape_f ) / ( self.tau_b + max(1, self.tau_f) )
        # Radial and poloidal width of the blob
        self.fwhm_ell_rad = np.zeros_like(self.amp)
        self.fwhm_ell_pol = np.zeros_like(self.amp)
        self.fwhm_err_ell_rad = np.zeros_like(self.amp)
        self.fwhm_err_ell_pol = np.zeros_like(self.amp)
        
        
    def track_backward(self, frames, doplots = False):
        """
        Track blob backward frame0 to beginning of frames
        """     
        
        if ( True ):
            self.tau_b, self.amp_b, self.xycom_b, self.xymax_b, fwhm_rad_idx_b, fwhm_pol_idx_b, self.blob_shape_b = \
                tracker(frames[:self.tau_max,:,:], self.event, self.thresh_amp, self.thresh_dist, self.blob_ext, 'backward', plots = doplots, verbose = False)
        else:
            print 'Could not track backward'
            self.invalid_bw_tracking = True


    def track_forward(self, frames, doplots = False):
        """
        Track blob forward from frame0
        """
        if ( True ):
            self.tau_f, self.amp_f, self.xycom_f, self.xymax_f, fwhm_rad_idx_f, fwhm_pol_idx_f, self.blob_shape_f = \
                tracker(frames[self.tau_max:,:,:], self.event, self.thresh_amp, self.thresh_dist, self.blob_ext, 'forward', plots = doplots, verbose = False)
        else:
            print 'Could not track forward'
            self.invalid_fw_tracking = True


    def plot_trail(self, frames, rz_array = None, xyi = None, trigger_box = None, sep_data = None, plot_com = False, plot_max = False, plot_shape = False, plot_geom = False, save_frames = False):
        """
        Plot the motion of the blob. The GPI frames are to be supplied externally
        
        Input:
            frames:         GPI data
            plot_com:       Mark the center of mass of the blob
            plot_max:       Mark the maximum of the blob
            plot_shape:     If available, mark the FWHM of the blob
            plot_geom:      Overplot triggering blox, limiter shadow and separatrix
            save_frames:    Save the frames
        
        """
        print 'Plotting the blob event from frame %d-%d' % ( self.event[1] + self.frame0 - self.tau_b, self.event[1] + self.frame0 + self.tau_f )
        minval = np.min( frames[ self.event[1] + self.frame0 - self.tau_b : self.event[1] + self.frame0 + self.tau_f, :, :]) 
        maxval = np.max( frames[ self.event[1] + self.frame0 - self.tau_b : self.event[1] + self.frame0 + self.tau_f, :, :]) 
        frames[ self.event[1] + self.frame0 - self.tau_b : self.event[1] + self.frame0 + self.tau_f, 0,0] = minval 
        frames[ self.event[1] + self.frame0 - self.tau_b : self.event[1] + self.frame0 + self.tau_f, 0,1] = maxval
        print 'min = %f, max = %f' % ( minval, maxval )

        for f_idx, tau in enumerate( np.arange( -self.tau_b, self.tau_f) ):
            plt.figure()
            plt.title('frame %05d' % ( self.event[1] + self.frame0 + tau) )  
            plt.xlabel('R / cm')
            plt.ylabel('Z / cm')
            
            try:    # Try plotting everythin in machine coordinates. If it fails, draw in pixels
                zi = griddata(rz_array.reshape(64*64, 2), frames[self.event[1] + self.frame0 + tau, :, :].reshape( 64*64 ), xyi.reshape( 64*64, 2 ), method='linear' )
#                zi[0] = 5.0#np.max(frames)
#                zi[1] = 5.0#np.max(frames)
                plt.contour(xyi[:,:,0], xyi[:,:,1], zi.reshape(64,64), 32, linewidths = 0.5, colors = 'k')
                plt.contourf(xyi[:,:,0], xyi[:,:,1], zi.reshape(64,64), 32, cmap = plt.cm.hot, levels=np.linspace(0.0,maxval,32))

            except:
                plt.contour (frames[ self.event[1] + self.frame0 + tau, :, :], 32, linewidths=0.5, colors='k')
                plt.contourf(frames[ self.event[1] + self.frame0 + tau, :, :], 32, cmap = plt.cm.hot, levels=np.linspace(0.0,maxval,32))

            plt.colorbar(ticks=np.arange(0.0, 3.5, 0.5), format='%3.1f')
    

            if ( plot_com == True ):
            
                try:
                    if ( plot_shape == False ):
                        plt.plot( xyi[ self.xycom[:f_idx+1, 0].astype('int'), self.xycom[:f_idx+1,1].astype('int'), 0], \
                            xyi[ self.xycom[:f_idx+1, 0].astype('int'), self.xycom[:f_idx+1,1].astype('int'), 1], '-ws')
                    
                    elif ( plot_shape == True ):
                        frame_xerr = self.fwhm_ell_rad[:f_idx+1]
                        frame_xerr[:-1] = 0.
                        frame_yerr = self.fwhm_ell_pol[:f_idx+1]
                        frame_yerr[:-1] = 0.
                        plt.errorbar( xyi[self.xycom[:f_idx+1, 0].astype('int'), self.xycom[:f_idx+1,1].astype('int'), 0], \
                            xyi[self.xycom[:f_idx+1, 0].astype('int'), self.xycom[:f_idx+1,1].astype('int'), 1], \
                            xerr = frame_xerr, yerr = frame_yerr, ecolor='w', linestyle = 'None', mfc='white', mec='green', marker = 's')

                    # Set the coordinates for plotting the text field
                    text_x, text_y = 86.2, -6.
                except TypeError:
                    plt.plot(self.xycom[:f_idx+1, 1], self.xycom[:f_idx+1, 0], '-bs')        
                    text_x, text_y = 5., 2.
                    
                if ( tau < self.tau_f-1 ):
                    plt.text( text_x, text_y, '$V_{COM} = (%4.1f, %4.1f)$' % \
                        (self.get_velocity_com(rz_array)[f_idx,0], self.get_velocity_com(rz_array)[f_idx,1] ), \
                        fontdict = dict(size = 16., color='white', weight='bold' ) )

            
                
            if ( plot_max == True ):
                try:
                    plt.plot( xyi[self.xymax[:f_idx+1, 0], self.xymax[:f_idx+1, 1], 0], xyi[self.xymax[:f_idx+1, 0], self.xymax[:f_idx+1, 1], 1], '-.wo')
                    text_x, text_y = 86.2, -6.
                except TypeError:
                    plt.plot(self.xymax[:f_idx+1, 1], self.xymax[:f_idx+1, 0], '-.wo')
                    text_x, text_y = 5., 2.
                    
                if ( tau < self.tau_f-1 ):
                    plt.text(text_x, text_y, '$V_{max} = (%4.1f, %4.1f)$' % \
                        (self.get_velocity_max(rz_array)[f_idx,0], self.get_velocity_max(rz_array)[f_idx,1] ) , \
                        fontdict = dict(size = 16., color='white', weight='bold' ) )
            if ( plot_geom == True ):
                try:
                    # Get the position of the pixels for the separatrix and limiter
                    separatrix_pxs  = surface_line( sep_data['rmid'].reshape(64,64) > sep_data['rmid_sepx'], mode='max' )
                    limiter_pxs  = surface_line( sep_data['rmid'].reshape(64,64) < sep_data['rmid_lim'], mode='min' )
                    # Compute position, width and height of the triggering box
                    tb_lower_left = (xyi[ trigger_box[2],trigger_box[0],0], xyi[trigger_box[2],trigger_box[0],1])
                    tb_width =  (xyi[trigger_box[2],trigger_box[1],0] - xyi[trigger_box[2],trigger_box[0],0])
                    tb_height = (xyi[trigger_box[3],trigger_box[0],1] - xyi[trigger_box[2],trigger_box[0],1] )
                    
                    # Plot the triggering domain. Position, height and width are not automatically determined
                    # but static values.
                    
                    triggering_box = mpatches.Rectangle( (89.9,-4.5), width = 1.0, height = 3.2, fill = False, ls = 'dashdot', ec = 'w', lw = 3 )
                    fig = plt.gcf()
                    ax = fig.gca()
                    ax.add_patch( triggering_box )
                    
                    # Plot the separatrix
                    sep_x = [xyi[i, separatrix_pxs[i], 0] for i in np.arange(64) ]
                    sep_y = [xyi[i, separatrix_pxs[i], 1] for i in np.arange(64) ]
                    plt.plot( sep_x, sep_y, 'w--', linewidth=4)
                    
                    lim_x = [xyi[i, limiter_pxs[i], 0] for i in np.arange(64) ]
                    lim_y = [xyi[i, limiter_pxs[i], 1] for i in np.arange(64) ]
                    plt.plot( lim_x, lim_y, 'w-.', linewidth=4)
            
                except:
                    print 'Error plotting geometry :('

            if ( save_frames == True ):
                F = plt.gcf()
                F.savefig('%d/frames/frame_%05d.eps' % (self.shotnr, self.event[1] + self.frame0 + tau) )
                plt.close()

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
            
            slice_pol = frames[t_idx, max(0, xy_off[t,0] - self.fwhm_max_idx) : min(63, xy_off[t,0] + self.fwhm_max_idx), xy_off[t,1] ]
            slice_rad = frames[t_idx, xy_off[t,0], max(xy_off[t,1] - self.fwhm_max_idx, 0) : min(xy_off[t,1] + self.fwhm_max_idx, 63) ]
                
            fwhm_rad_idx[t,:] = fwhm( slice_rad/slice_rad.max() ) + xy_off[t,1] - self.fwhm_max_idx
            fwhm_pol_idx[t,:] = fwhm( slice_pol/slice_pol.max() ) + xy_off[t,0] - self.fwhm_max_idx

            try:
                self.fwhm_ell_rad[t] = (rz_array[xy_off[t,0], fwhm_rad_idx[t,1], 0] - rz_array[xy_off[t,0], fwhm_rad_idx[t,0], 0]) / 2.355
                self.fwhm_ell_pol[t] = (rz_array[fwhm_pol_idx[t,1], xy_off[t,1], 1] - rz_array[fwhm_pol_idx[t,0], xy_off[t,1], 1]) / 2.355
            except NameError:
                self.fwhm_ell_rad[t] = fwhm_rad_idx[t,1] - fwhm_rad_idx[t,0]
                self.fwhm_ell_pol[t] = fwmh_pol_idx[t,1] - fwhm_pol_idx[t,0]

# Debugging of the expressions above
#                print 'poloidal:  x_off = ', xy_off[t,1], ' from r_idx = ', fwhm_pol_idx[t,1] ,' to ', fwhm_pol_idx[t,0]
#                print ' is ', rz_array[fwhm_pol_idx[t,1], xy_off[t,1], 1] , ' to ', rz_array[fwhm_pol_idx[t,0], xy_off[t,1], 1]

            if ( plots  == True) :
                plt.figure()
                plt.title('Cross sections at %s' % position)
                plt.plot( frames[ t_idx, xy_off[t,0], :], '.-', label='radial xsection')
                plt.plot( frames[ t_idx, :, xy_off[t,1]], '.-', label='poloidal xsection')
                plt.plot( fwhm_rad_idx[t,:], frames[t_idx, xy_off[t,0], fwhm_rad_idx[t,:]] , 'b--' )
                plt.plot( fwhm_pol_idx[t,:], frames[t_idx, fwhm_pol_idx[t,:], xy_off[t,1] ], 'g--' )
                plt.axvline( xy_off[t, 1], color='red')
                plt.axvline( xy_off[t, 0], color='red')
                plt.legend(loc='lower left')
                plt.show()
                

        
    def compute_width_gaussian(self, frames, rz_array = None, position = 'MAX', i_size_max = 12, plots = False ):
        """
        Attempts to fit a Gaussian to the peak
        
        Input:
            frames:         GPI data
            rz_array:       2d array with (R,z) value for each pixel. If omitted, computes FWHM in pixels
            position:       Compute FWHM at center of mass 'COM' or maximum 'MAX'
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

        i_size_0 = 4

        gaussian_fun = lambda p, x: np.exp( -(x-p[0])**2 / (2. * p[1]**2.) ) 
        err_fun = lambda p, y, x: y - gaussian_fun(p, x)
        fit_errs = np.zeros( int( (i_size_max - i_size_0)/2) )
        i_range = np.arange(i_size_0, i_size_max, 2)

        for t, ttau in enumerate( self.tau ):
        # Fit a Gaussian in the radial and poloidal direction        
            t_idx = self.event[1] + self.frame0 + ttau
            for size_idx, i_size in enumerate( i_range ):
                # Test different sizes of the fit domain. Take the fit which minimizes the residual squared                
                slice_pol = frames[t_idx, max(0, xy_off[t,0] - i_size) : min(63, xy_off[t,0] + i_size), xy_off[t,1] ]
                slice_rad = frames[t_idx, xy_off[t,0], max(0, xy_off[t,1] - i_size) : min(63, xy_off[t,1] + i_size) ]
                
                try:
                    X_pol = rz_array[ max(0, xy_off[t,0] - i_size) : min(63, xy_off[t,0] + i_size), xy_off[t,1],  1]
                    X_rad = rz_array[ xy_off[t,0], max(0, xy_off[t,1] - i_size) : min(63, xy_off[t,1] + i_size), 0] 
    
                except NameError:
                    X_pol = np.arange( np.size(slice_pol) )
                    X_rad = np.arange( np.size(slice_rad) )
                
                p0_pol = [ rz_array[xy_off[t,0], xy_off[t,1], 1], 0.1 ]
                p0_rad = [ rz_array[xy_off[t,0], xy_off[t,1], 0], 0.1 ]
                
                [p_pol, success_pol] = leastsq( err_fun, p0_pol, args = (slice_pol/slice_pol.max(), X_pol ), maxfev = 10000 )
                [p_rad, success_rad] = leastsq( err_fun, p0_rad, args = (slice_rad/slice_rad.max(), X_rad ), maxfev = 10000 )                
                            
                error_pol = np.sqrt( np.sum( (slice_pol - gaussian_fun(p_pol, X_pol))**2  ) / (np.size(slice_pol) - 1.)  )
                error_rad = np.sqrt( np.sum( (slice_rad - gaussian_fun(p_rad, X_rad))**2  ) / (np.size(slice_rad) - 1.)  )        
                fit_errs[size_idx] = max( error_pol, error_rad)

            # Repeat the fit with domain that minimizes error
            i_size = i_range[fit_errs.argmin()]
            slice_pol = frames[t_idx, max(0, xy_off[t,0] - i_size) : min(63, xy_off[t,0] + i_size), xy_off[t,1] ]
            slice_rad = frames[t_idx, xy_off[t,0], max(0, xy_off[t,1] - i_size) : min(63, xy_off[t,1] + i_size) ]
            
            try:
                X_pol = rz_array[ max(0, xy_off[t,0] - i_size) : min(63, xy_off[t,0] + i_size), xy_off[t,1],  1]
                X_rad = rz_array[ xy_off[t,0], max(0, xy_off[t,1] - i_size) : min(63, xy_off[t,1] + i_size), 0] 

            except NameError:
                X_pol = np.arange( np.size(slice_pol) )
                X_rad = np.arange( np.size(slice_rad) )
            
            p0_pol = [ rz_array[xy_off[t,0], xy_off[t,1], 1], 0.1 ]
            p0_rad = [ rz_array[xy_off[t,0], xy_off[t,1], 0], 0.1 ]
            [p_pol, success_pol] = leastsq( err_fun, p0_pol, args = (slice_pol/slice_pol.max(), X_pol ), maxfev = 10000 )
            [p_rad, success_rad] = leastsq( err_fun, p0_rad, args = (slice_rad/slice_rad.max(), X_rad ), maxfev = 10000 )                
                        
            error_pol = np.sqrt( np.sum( (slice_pol - gaussian_fun(p_pol, X_pol))**2  ) / (np.size(slice_pol) - 1.)  )
            error_rad = np.sqrt( np.sum( (slice_rad - gaussian_fun(p_rad, X_rad))**2  ) / (np.size(slice_rad) - 1.)  )

            self.fwhm_ell_rad[t] = p_rad[1]
            self.fwhm_ell_pol[t] = p_pol[1]
            self.fwhm_err_ell_rad[t] = error_rad
            self.fwhm_err_ell_pol[t] = error_pol
 
            if ( plots == True ):
                F = plt.figure()
                plt.title('Gaussian fit, i_size = %d' % i_size)
                plt.plot( X_pol, slice_pol/slice_pol.max() )
                plt.plot( X_pol, gaussian_fun(p_pol, X_pol), label = 'width =%f, Error = %f' % (p_pol[1], error_pol) )
                plt.legend()
                F.savefig('%d/fits/pol_fit_%d.eps' % (self.shotnr, t_idx ) )
#                plt.show()
                plt.close()
                
                
            

        
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
            
        return rz_array[ self.xycom[:,0].astype('int'), self.xycom[:,1].astype('int'), :]
        
        
    def get_trail_max(self, rz_array = None):
        """
        Return the position of the blob maximum. Either in pixel or in (R,Z) coordinates if rz_array
        is passed.
        """
        if ( rz_array == None ):
            return self.xymax
        
        # Remember xycom[:,1] is the radial (X) index which corresponds to R
        return rz_array[ self.xymax[:,0].astype('int'), self.xymax[:,1].astype('int'), :]
            
            
    def get_velocity_max(self, rz_array = None):
        """
        Return the velocity of the blob maximum. Either in pixel / frame of m/s when rz_array is given
        """
        assert (np.size(self.tau) > 1), 'Cannot compute blob velocity with only one frame recognized'      

        try:
            trail = self.get_trail_max().astype('int')                
            return 1e-2*( rz_array[ trail[1:, 0], trail[1:, 1], :]  - rz_array[ trail[:-1, 0], trail[:-1, 1], :] ) / self.dt
        except TypeError:
            return self.xymax[1:, :] - self.xymax[:-1, :]    
    
    
    def get_velocity_com(self, rz_array = None):
        """
        Return the velocity of the blob COM. Either in pixel / frame of m/s when rz_array is given
        """

        assert (np.size(self.tau) > 1), 'Cannot compute blob velocity with only one frame recognized'
        
        try:
            trail = self.get_trail_com().astype('int')
            return 1e-2*( rz_array[ trail[1:, 0], trail[1:, 1], :]  - rz_array[ trail[:-1, 0], trail[:-1, 1], :] ) / self.dt
        except TypeError:
            return self.xycom[1:, :] - self.xycom[:-1, :]
        
        
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
        

    def get_err_ell_pol(self):
        """
        Return error from length fitting
        """
        return self.fwhm_err_ell_pol
        
    def get_err_ell_rad(self):
        """
        Return error from length fitting
        """
        return self.fwhm_err_ell_rad



    def get_amp(self):
        """
        Return the amplitude (maximum intensity) of the blob
        """
        return self.amp
        
    
    def get_tau(self):
        """
        Return the frames in blob trail relative to the frame number where the blob was detected
        """
        return self.tau
    
    def get_event_frames(self):
        """
        Return the frames in which the blob event occurs
        """
        return self.tau + self.event[1]

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
            blob_shape = np.zeros([np.size(self.tau), 2*self.blob_ext, 2*self.blob_ext])
            t_off = np.arange(np.size(self.tau))
    

        if ( position == 'COM' ):
            x_off, y_off = self.xycom[:,0].astype('int'), self.xycom[:,1].astype('int')
        elif ( position == 'MAX' ):
            x_off, y_off = self.xymax[:,0].astype('int'), self.xymax[:,1].astype('int')

        for t_idx, t in enumerate(t_off):
            blob_shape[t_idx, :, :] = frames[t + self.event[1] + self.frame0, y_off[t_idx] - self.blob_ext : y_off[t_idx] + self.blob_ext, x_off[t_idx] - self.blob_ext : x_off[t_idx] + self.blob_ext]
    
        print 'blob_shape finished'
        return blob_shape

    
