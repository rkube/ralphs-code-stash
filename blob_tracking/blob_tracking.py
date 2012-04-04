#!/opt/local/bin/python
#-*- Encoding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from detect_peak import detect_peak_3d
from blobtrail import blobtrail
from phantom_helper import make_rz_array
from scipy.interpolate import griddata
import cPickle as pickle


"""
Run blob detection on a set of GPI frames and store information about the blob in a blobtrail
object. Return the list of blobtrail objects
"""

def blob_tracking(shotnr, frames, frame_info, frame0 = 0, minmax = [2.0, 10.0], logger = None, nframes = 30000):
    
    np.set_printoptions(linewidth=999999)
    frame0 = 0                              # Begin analysis at this frame. 
    minmax  = np.array(minmax)              # Peaks within 2.5 and 10.0 times the rms
    lag     = 20                            # Deadtime after a peak in which no blob is detected
    time_arr= np.arange(0, nframes)         # Just the time, in frame numbers of the time series
    trigger = np.array([40, 50, 16, 48])    # Triggerbox r_low, r_up, z_low, z_up
    toffset = frame0 + lag                  # Total frame offset used in this script.
    tau_max   = 7                           # Maximal time intervall for which we track a blob
    # 1 frame is 2.5µs
    dt = 2.5e-6
    
    try:
        logger.info('frame0 = %d, nframes = %d' % (frame0, nframes) )
    except:
        print 'frame0 = ', frame0, 'nframes = ', nframes
    
    # Detect peaks
    # The detect_peak_3d returns time indices of blob events relative for the array passed to it.
    # Remember to add frame0 to idx_event[t0] to translate to the frame indexing used in this script.
    idx_events = detect_peak_3d(frames[frame0:frame0+nframes,:,:], trigger, minmax, 0, lag, rel_idx=False)
    num_events = np.shape(idx_events)[0]
    event_ctr = np.ones([num_events])
    
    try:
        logger.info('%d blob events detected' % ( num_events) )
    except:
        print '%d blob events detected' % ( num_events )
    
    # Define the events we will analyze
    event_range = np.arange( num_events )
#    event_range = np.arange( 550, 560)
    num_events  = np.size(event_range)
    
    # Get R,z projection, grid data
    rz_array, transform_data = make_rz_array(frame_info)
    xxi, yyi = np.meshgrid( np.linspace( np.min(rz_array[:,:,0] ), np.max( rz_array[:,:,0] ),64 ), np.linspace( np.min( rz_array[:,:,1] ), np.max( rz_array[:,:,1] ),64 ) )
    xyi = np.concatenate( (xxi[:,:,np.newaxis], yyi[:,:,np.newaxis]), axis=2 )
       
    trails = []
    fail_list = []
    failcount = 0
    
    for idx, event in enumerate(idx_events[event_range]):
        I0 = event[0] 
        t0 = event[1] + frame0
        z0 = event[2]
        R0 = event[3]    
        
        print 'peak %d / %d, frame %d' % ( idx, num_events, t0)
        try:
            newtrail = blobtrail( frames[t0 - tau_max : t0 + tau_max, :, :], event, frame0, shotnr, thresh_amp=0.7, blob_ext = 14, thresh_dist = 8., fwhm_max_idx = 18, doplots = False )
            if ( np.size(newtrail.get_tau()) < 4 ):
                fail_list.append(idx)
                failcount += 1
                
                if ( logger != None ):
                    logger.info('Peak %d: Trail too short: %d frames' % ( idx, np.size(newtrail.get_tau) ) )
                else:
                    print 'Peak %d: Trail too short: %d frames' % ( idx, np.size(newtrail.get_tau) )

                continue

            
        except ValueError, IndexError:
            fail_list.append(idx)
            failcount += 1
            if ( logger != None ):
                logger.info('Failed to track blob %d / %d' % ( idx, num_events) )
            else:
                print 'Failed to track blob %d / %d' % (idx, num_events)
            continue
            
#        try:
        if ( True ):
#            newtrail.compute_fwhm(frames, rz_array, position = 'MAX', norm = True,  plots = True)
            newtrail.compute_width_gaussian(frames, rz_array, position = 'MAX', i_size_max = 10, plots = True )
            
        if ( False ):
#        except:
            fail_list.append(idx)
            failcount += 1
            
            if ( logger != None ):
                logger.info('Peak %d: Unable to compute FWHM' % (idx ) )
            else:
                print 'Peak %d: Unable to compute FWHM' % ( idx )
            continue
            

#        newtrail.plot_trail(frames, rz_array = rz_array, xyi = xyi, plot_com = True, plot_shape = True, save_frames = True)
        trails.append(newtrail)
        
    return trails
    
   