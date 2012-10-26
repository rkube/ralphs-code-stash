#!/opt/local/bin/python
#-*- Encoding: UTF-8 -*-

"""
================
Helper functions
================

.. codeauthor :: Ralph Kube <ralphkube@gmail.com>
Helper functions to make blob analysis and tracking easier

* com......Computes the center of mass position for a 2d array
* com_Rz...Computes the center of mass position for a 2d array with R and z coordinates
* fwhm.....Compute the full width half maximum of a 1d array

"""

import numpy as np
import pymorph as pm
import matplotlib.pyplot as plt
from phantom_helper import make_rz_array
from scipy.interpolate import griddata
import idlsave


def frac_to_idx(frac, min, max, nbins):
    """
    Given a float within in the interval [min,max], return the index in which of nbins equidistant
    bins it belongs
    """
    return np.floor((frac-min) * float(nbins) / (max-min)).astype('int')
    
def com(array, xx = None, yy = None):
    """
    Return the center of mass position of the array x_com and y_com
    x_com = int( x * array ) / int( array ),
    y_com = int( y * array ) / int( array )
    If xx and yy are not specified, use x = 0,1,...,np.shape(array)[0], and analog for y
    Returns 
        x_com, y_com
    """
    array = array.astype('float')
    try:
        dx, dy = np.max( xx[:,1:] - xx[:,:-1] ), np.max( yy[1:,:] - yy[:-1,:] )
        return np.sum( xx * array ) / np.sum ( array ), np.sum( yy * array ) / np.sum ( array )
    except TypeError:
        xx, yy = np.meshgrid( np.arange(0, np.shape(array)[0]), np.arange(0, np.shape(array)[1]) )
        dx, dy = np.max( xx[:,1:] - xx[:,:-1] ), np.max( yy[1:,:] - yy[:-1,:] )
        return np.sum( xx * array * dx * dy) / np.sum ( array * dx * dy), np.sum( yy * array * dx * dy) / np.sum ( array * dx * dy)


def com_rz(array, RR, zz):
    """
    Return the center of mass position on the irregulary spaced RR, zz array:
    R_com = int ( R * n * dA ) / int ( n * dA ), along second dimension
    z_com = int ( z * n * dA ) / int ( n * dA ), along first dimension
    """
    array = array.astype("float")
    dR, dz = np.zeros_like(RR), np.zeros_like(zz)
    dR[:,:-1] = RR[:,1:] - RR[:,:-1]
    dR[:,-1] = dR[:,-2]
    dz[:-1,:] = zz[1:,:] - zz[:-1,:]
    dz[-1,:] = dz[:,-2]
    
    # COM along second dimension, COM along first dimension
    return np.sum( RR * array * dR * dz ) / np.sum (array * dR * dz ) ,\
        np.sum( zz * array * dR * dz ) / np.sum (array * dR * dz )


def fwhm(array):
    """
    Computes the full width half maximum of a 1-d array
    Returns the indices of the array elements left and right closest to the maximum that cross below
    half the maximum value
    """
    
    assert ( type(array) == type(np.ndarray([])) )

    # Find the maximum in the interior of the array
    fwhm = 0.5 * array[1:-1].max()
    max_idx = array[1:-1].argmax() + 1
    # Divide the intervall in halfs at the peak and find the index of the value in the left half
    # of the intervall before it increases over 0.5 * max

    # The FWHM is between the indices closest to the maximum, whose values are below 0.5 times the max
    try:
        left_idx  = np.argwhere( array[1: max_idx] < fwhm )[-1] + 1
    except IndexError:  
        # This can occur, if there is no value in the array smaller than 0.5*max. In this case, return
        # the left limit of the array as the FWHM.
        left_idx = 0
    try:
        right_idx = np.argwhere( array[max_idx:-1] < fwhm )[0 ] + max_idx
    except IndexError:
        # Subtract one because the index of an array with size n is within 0..n-1
        right_idx = np.size(array) - 1

    return  np.array([ int(left_idx), int(right_idx) ])
    
    
    
def tracker(frames, event, thresh_amp, thresh_dis, blob_ext, direction='forward', plots = False, verbose = False):
    """
    Track the blob in a dynamic intervall forward or backward in time, as long as its
    amplitude is over a given threshold and the peak has detected less than a given
    threshold over consecutive frames.
    The maximum number of frames the blob is tracked for, is given by dim0 of frames
    
    Input:
        tau:        Maximum number of frames to track blob
        event:      ndarray, [I0, t0, R0, z0] Index of original feature to track
        direction:  Traverse frames 'forward' or 'backward' in dimension 0
        thresh_amp: Threshold for amplitude decay relative to frame0
        thresh_dis: Threshold for blob movement relative to previous frame
        blob_ext:   Extend of the blob used for determining its average shape
    
    Returns:
        numframes:      Number of frames the blob was tracked
        xycom:          COM position of the blob in each frame
        amp:            Amplitude at COM in each frame
        fwhm_rad_idx:   Indices that mark left and right FWHM of the blob   
        fwhm_pol_idx:   Indices that mark the lower and upper FWHM of the blob
        blob:           Array that stores the blob extend
    """
    if ( verbose == True ): 
        print 'Called tracker with '
        print 'event = ', event
        print 'thresh_amp = ', thresh_amp
        print 'thresh_dis = ', thresh_dis
        print 'blob_ext = ', blob_ext
        print 'plots = ', plots
    
    assert ( direction in ['forward', 'backward'] )
    assert ( blob_ext % 2 == 0 )
    
    tau_max             = np.shape(frames)[0]   # Maximum number of frames the blob is tracked for is given by dimension 0 of frames
    good_blob           = True
    z0_last, R0_last    = event[2], event[3] 

    if ( direction == 'forward' ):          # Include the current frame when determining fwhm, com coordinates, etc
        I0 = frames[0,z0_last,R0_last]
        f_idx = 0                           # Index used to access frames 
        tau = 0                             # Start with zero offset
    elif ( direction == 'backward' ):
        I0 = frames[-1,z0_last,R0_last]
        f_idx = -1                          # Start at the second to last frame, 0 based indexing
        tau = 1                             # Start with one frame offset

    if ( verbose ):
        print 'Tracking blob %s, t_idx %d x = %d, y = %d, I0 = %f' % (direction, tau, R0_last, z0_last, I0 )
        print 'thresh_amp = %f, thresh_dis = %f' % (thresh_amp * I0, thresh_dis)
    xycom   = np.zeros([tau_max, 2])            # Return values: COM position of peak
    xymax   = np.zeros([tau_max, 2])            # Position of the blob peak
    fwhm_pol_idx = np.zeros([tau_max, 2], dtype='int')       # Poloidal FWHM 
    fwhm_rad_idx = np.zeros([tau_max, 2], dtype='int')       # Radial FWHM
    amp     = np.zeros([tau_max])               # Amplitude at COM position
    blob    = np.zeros([tau_max, blob_ext, blob_ext])
   
    while ( good_blob == True and tau < tau_max ):      
        if ( verbose  ):
            print 'f_idx %d, blob from x = %d, y = %d, I0 = %f' % (f_idx, R0_last, z0_last, frames[f_idx, z0_last, R0_last] )    
               
        event_frame = frames[f_idx, :, :]
        labels = pm.label( event_frame > thresh_amp * I0 )          # Label the regions with more than 60% 
                                                                    # of the original intensity
        blob_area = pm.blob( labels, 'area', output = 'data')       # Get blob areas
        blob_cent = pm.blob( labels, 'centroid', output = 'data')   # Get blob centers
        #print 'Centroid analysis', blob_cent, np.shape(blob_cent), np.size(blob_cent)
        
        if ( np.size(blob_cent) < 1 ):      # No peak here, quit tracking
            good_blob = False
            print 'Frame %d, %ss: lost track of blob' % (f_idx, direction)
            break

        min_dist_frame = np.sqrt(64.*64.+64.*64.)       # Minimal distance to last peak in current frame
        # We got a peak, see if it is within the maximum distance a peak may travel over one frame
        for d_idx, i in enumerate(np.where( blob_area > 0.1 * max(blob_area) )[0]):
            # TODO: Vectorize loop
            dist = np.sqrt( (blob_cent[i,1]-R0_last)**2 + (blob_cent[i,0]-z0_last)**2 )
            #if ( verbose ):
            #    print 'Region %d, distance to last peak: %f' % (d_idx, dist)
            if ( dist < min_dist_frame and dist < thresh_dis):
                min_dist_frame = dist
                min_idx = i
                #if ( verbose ):
                #    print 'Accepted'
        
        if ( min_dist_frame == np.sqrt( 8192. ) ):
            print 'No peak satisfying criteria found: dist = %f, Stopping %s tracking after %d frames' % ( min_dist_frame, direction, tau )
            break
        
        # Compute the x and y COM coordinates of the blob, store
        blob_mask = labels!=(min_idx+1)
        event_masked = np.ma.MaskedArray(event_frame , mask = blob_mask, fill_value = 0)
        
        # When used to index frames[:,:,:]: xymax[tau,:] = [ index for axis 2, index for axis 1 ] 
        xymax[tau,:] = np.unravel_index( event_masked.argmax(), np.shape(labels) )  # Maximum in the blob mask
        # When used to index frames[:,:,:]: xycom[tau,:] = [ index for axis 1, index for axis 2 ]
        # To be consistent with indexing from xymax, flip this array
        xycom[tau,::-1] = com(event_masked)                 # COM returns com along second dimension at index 0
        
        ycom_off, xcom_off = np.round(xycom[tau,:]).astype('int')   
#        print 'xycom', xycom[tau,:], 'ycom_off = ', ycom_off, ', xcom_off = ', xcom_off

        if ( verbose ):
            print 'Peak at (%d,%d), COM at (%d,%d)' % (xymax[tau,0], xymax[tau,1], xycom[tau,0], xycom[tau,1])


#        try:
        blob_shape = event_frame [ycom_off - blob_ext/2 : ycom_off + blob_ext/2, xcom_off - blob_ext/2 : xcom_off + blob_ext/2]    
        blob[:np.shape(blob_shape)[0], :np.shape(blob_shape)[1]] += blob_shape
#        except ValueError:
#            print 'Error adding blob shape to average'
        
        amp[tau] = event_frame[z0_last, R0_last]      
        # Follow the peak
        z0_last, R0_last = xymax[tau, :].astype('int')

        if ( plots ) :
            plt.figure()
            plt.title('%s, frame %d' % (direction, f_idx ))
            plt.contour (event_frame, 16, colors = 'k', linewidth=0.5 )
            plt.contourf(event_frame, 16, cmap = plt.cm.hot )
            plt.plot(xycom[tau,1], xycom[tau,0], 'wo')
            plt.plot(xymax[tau,1], xymax[tau,0], 'w^')
            plt.colorbar()
            plt.xlabel('x / px')
            plt.ylabel('y / px')       

        tau += 1                              # Frame accepted, advance indices
        if ( direction == 'forward' ):
            f_idx += 1
        elif ( direction == 'backward' ):
            f_idx -= 1

    if ( direction == 'backward' ):
        tau -= 1    # We started at tau=1, subtract one to return correct number of frame tracked
                    # in one direction, ignoring the starting frame
                    # Ignore this for forward frame as we count the original frame here

    if ( plots ) :
        plt.show()

    return tau, amp, xycom, xymax, fwhm_rad_idx, fwhm_pol_idx, blob


def find_sol_pixels(shotnr, frame_info = None, rz_array = None, path = '/Users/ralph/source/blob_tracking'):
    """
    Returns the indices of the pixels in between the separatrix and the LCFS.    
    """
    
    s = idlsave.read('%s/test_data/separatrix.sav' % ( path ), verbose = False)
    
    gap_idx_mask = (s['rmid'].reshape(64,64) > s['rmid_sepx']) & (s['rmid'].reshape(64,64) < s['rmid_lim'])
                
    return np.argwhere(gap_idx_mask)
    
    
def find_sol_mask(shotnr, frame_info = None, rz_array = None, path = '/Users/ralph/source/blob_tracking' ):
    """
    Returns a mask for the pixels in between the separatrix and the LCFS.
    """
    s = idlsave.read('%s/test_data/separatrix.sav' % ( path ), verbose = False)
    
    return (s['rmid'].reshape(64,64) > s['rmid_sepx']) & (s['rmid'].reshape(64,64) < s['rmid_lim'])
    
    
def blob_in_sol(trail, good_domain, logger = None):
    """
    Returns a bool array of the indices, in which the COM of a blob is in the SOL (good_domain)
    """
    
    try:
        # Consider only the positions, where the blob is in the good domain
        blob_pos = trail.get_trail_com()
        good_pos_idx = np.array([i in good_domain for i in blob_pos.round().astype('int').tolist()])     
    
    except:
        good_pos_idx = np.ones_like(trail.get_tau())
        if ( logger != None ):
            logger.info('This should not happen. Ignoring trigger domain')

    good_pos_idx = good_pos_idx[:-1]
    return good_pos_idx

