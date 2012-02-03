#!/opt/local/bin/python
#-*- Encoding: UTF-8 -*-

import numpy as np
import pymorph as pm
import matplotlib.pyplot as plt
from detect_peak import detect_peak_3d
from helper_functions import com, com_rz, fwhm
from phantom_helper import make_rz_array
from scipy.interpolate import griddata


"""
Check out how image segmenting works
"""
np.set_printoptions(linewidth=999999)
shotnr  = 1100803015
frame0 = 00000                          # Begin analysis at this frame. 
nframes = 1000                         # Number of frames to analyze
minmax  = np.array([2.5, 10.0])         # Peaks within 1.5 and 2.5 times above rms
lag     = 20                            # Deadtime after a peak in which no blob is detected
time_arr= np.arange(0, nframes)         # Just the time, in frame numbers of the time series
trigger = np.array([40, 50, 10, 53])    # Triggerbox r_low, r_up, z_low, z_up
blobbox = np.array([8,8])               # Box used to determine blob size around single blob events
toffset = frame0 + lag                  # Total frame offset used in this script.
tau_b_max   = 7                         # Maximal time intervall for which we track a blob
tau_a_max   = 7
blob_dist_max = 8.                      # Maximal number of pixels a blob may travel in one time
fwhm_max_idx    = 10                    # Required minimum pixel to boundary for FWHM analysis 

# 1 frame is 2.5mus
dt = 1./400000.

try:
    datafile = np.load('../../../blob_tracking/%d/%d_frames.npz' % (shotnr, shotnr), mmap_mode = 'c')
    frames = datafile['frames_normalized_mean']
    print 'Loaded frames for shot %d' % shotnr
except IOError:
    print 'Could not open file %d/%d_frames.npz' % (shotnr, shotnr)
    datafile = np.load('../../test/test_frames_200.npz')
    frames = datafile['frames']
frame_info = datafile['frame_info']

# Detect peaks
# The detect_peak_3d returns time indices of blob events relative for the array passed to it.
# Remember to add frame0 to idx_event[t0] to translate to the frame indexing used in this script.
idx_events = detect_peak_3d(frames[frame0:frame0+nframes,:,:], trigger, minmax, 0, lag, rel_idx=False)
num_events = np.shape(idx_events)[0]
event_ctr = np.ones([num_events])
print '%d blob events detected' % ( num_events )

# Get R,z projection, grid data
rz_array, transform_data = make_rz_array(frame_info)
RRi, zzi = np.meshgrid( np.linspace( np.min(rz_array[:,:,0] ), np.max( rz_array[:,:,0] ),64 ), np.linspace( np.min( rz_array[:,:,1] ), np.max( rz_array[:,:,1] ),64 ) )
Rzi = np.concatenate( (RRi[:,:,np.newaxis], zzi[:,:,np.newaxis]), axis=2 )
zi = griddata(rz_array.reshape(64*64, 2), frames[666,:,:].reshape( 64*64 ), Rzi.reshape( 64*64, 2 ), method='linear' )


# Average amplitude, velocity, and length for each blob event
blob_amp = np.zeros([num_events])
blob_vel = np.zeros([num_events])
blob_lp  = np.zeros([num_events])
blob_lr  = np.zeros([num_events])

for idx, event in enumerate(idx_events[8:9]):
#    try:
    for bob in np.arange(0,1):
        I0 = event[0] 
        t0 = event[1] + frame0
        R0 = event[2]
        z0 = event[3]    
        event_frame = frames[t0,:,:]
        r0_last = R0
        z0_last = z0
        xycom   = np.zeros([2, tau_b_max + tau_a_max])
        rzcom   = np.zeros([2, tau_b_max + tau_a_max])            # Rz com position at time t0 +- tau
        fwhm_Rz = np.zeros([2, tau_b_max + tau_a_max])
        amp     = np.zeros([tau_b_max + tau_a_max])

        print 'Frame %d, peak at (%d,%d): %f' % (t0, R0, z0, frames[t0,R0,z0])
        plt.figure()
        plt.contour (frames[t0, :, :], 16, colors = 'k', linewidth=0.5 )
        plt.contourf(frames[t0, :, :], 16, cmap = plt.cm.hot )
        plt.title('Frame %d, peak at (%d,%d): %f' % (t0, R0, z0, frames[t0,R0,z0]) )
        plt.colorbar()
        plt.plot( z0, R0, 'ko' )
        
        # Find time intervall for which we can track the blob.
        # Step one frame back in time from where the event was detected
        # Detect and label regions that have more than 60% the intensity of the detected peak
        # If the center of the region is less than max_dist away from where the blob was in the
        # previous frame, identify this as the same blob. Proceed to the next frame.
        #        
        good_blob = True
        tau_b = 0               # Frame before blob event
        t_idx = tau_b_max       # Index of frame, relative to t0+tau_b_max
        while ( good_blob == True and tau_b < tau_b_max ):      
            print 'Tracking blob backwards, frame %d tau_b = %d tau_idx = %d' % (t0 - tau_b, tau_b, t_idx )
            event_frame = frames[t0 - tau_b, :, :]
            labels = pm.label( event_frame > 0.6 * I0 )         # Label the regions with more than 60% 
                                                                # of the original intensity
            blob_area = pm.blob( labels, 'area', output = 'data')       # Get blob areas
            blob_cent = pm.blob( labels, 'centroid', output = 'data')   # Get blob centers
#            print 'Centroid analysis', blob_cent, np.shape(blob_cent), np.size(blob_cent)
            if ( np.size(blob_cent) < 1 ):      # No peak here, quit tracking
                good_blob = False
#                print 'Lost track of blob'
                break

            min_dist_frame = np.sqrt(64.*64.+64.*64.)       # Minimal distance to last peak in current frame
            # We got a peak, see if it is within the maximum distance a peak may travel over one frame
            for d_idx, i in enumerate(np.where( blob_area > 0.1 * max(blob_area) )[0]):
                # TODO: Vectorize loop
                dist = np.sqrt( (blob_cent[i,1]-z0_last)**2 + (blob_cent[i,0]-r0_last)**2 )
#                print 'Region %d, distance to last peak: %f' % (d_idx, dist)
                if ( dist < min_dist_frame and dist < blob_dist_max):
                    min_dist_frame = dist
                    min_idx = i
#                    print 'Accepted'
            
            if ( min_dist_frame == np.sqrt( 8192. ) ):
#                print 'No peak satisfying criteria found. Stopping backward tracking'
                break
            
            # Compute the x and y COM coordinates of the blob, store
            blob_mask = labels!=(min_idx+1)
            event_masked = np.ma.MaskedArray(labels, mask = blob_mask, fill_value=0)
            xycom[:,t_idx] = com(event_masked)

            fig = plt.figure()
            xcom_off, ycom_off = np.round(xycom[:,t_idx]).astype('int')                
            # Skip blob if it is too close to the boundary for good FWHM analysis
            if ( xycom[:,t_idx].min() > fwhm_max_idx and ( 64. - xycom[:,t_idx].max() ) > fwhm_max_idx ):
                fwhm_rad_idx = fwhm(event_frame[ ycom_off, xcom_off - fwhm_max_idx : xcom_off + fwhm_max_idx ])
                fwhm_pol_idx = fwhm(event_frame[ ycom_off - fwhm_max_idx : ycom_off + fwhm_max_idx, xcom_off ])    
                fwhm_Rz[0,t_idx] = RRi[ ycom_off, xcom_off - fwhm_max_idx + fwhm_rad_idx[1][1] ] - RRi[ ycom_off, xcom_off - fwhm_max_idx + fwhm_rad_idx[1][0] ]
                fwhm_Rz[1,t_idx] = zzi[ ycom_off - fwhm_max_idx + fwhm_pol_idx[1][1] , xcom_off] - zzi[ ycom_off - fwhm_max_idx + fwhm_pol_idx[1][0] , xcom_off ]       

                fig.add_subplot(212)
                plt.title('Cross sections')
                plt.plot(event_frame[ycom_off, xcom_off - fwhm_max_idx : xcom_off + fwhm_max_idx ], 'b-o', label='Radial cross, FWHM=%3.1f' % fwhm_Rz[0,t_idx])
                plt.plot( fwhm_rad_idx[1], event_frame[ ycom_off, (fwhm_rad_idx[1] + xcom_off - fwhm_max_idx).astype('int') ], 'b--' )
                plt.plot(event_frame[ ycom_off - fwhm_max_idx : ycom_off + fwhm_max_idx, xcom_off], 'g-o', label='Poloidal cross, FWHM=%3.1f' % fwhm_Rz[0,t_idx] )
                plt.plot( fwhm_pol_idx[1], event_frame[ (fwhm_pol_idx[1] + ycom_off - fwhm_max_idx).astype('int'), xcom_off ], 'g--' )
                plt.legend(loc='upper right')

            
            rzcom[:,t_idx] = com_rz(event_masked, rz_array[:,:,0], rz_array[:,:,1])
            print 't_idx %d, region center at (%d,%d), com coordinates (x,y) = (%f,%f), (R,z) = (%f,%f)' %\
                (t_idx, blob_cent[i,0], blob_cent[i,1], xycom[0,t_idx], xycom[1,t_idx], rzcom[0,t_idx], rzcom[1,t_idx])       

            
            # Store FWHM, amplitude and blob position
            amp[t_idx] = event_frame[r0_last, z0_last]      
            r0_last, z0_last = blob_cent[min_idx, :]


            plt.subplot(211)
            plt.title('frame %d' % (t0 - tau_b))
            plt.contour (RRi[0,:], zzi[:,0], event_frame, 16, colors = 'k', linewidth=0.5 )
            plt.contourf(RRi[0,:], zzi[:,0], event_frame, 16, cmap = plt.cm.hot )
            plt.colorbar()
            plt.plot(rzcom[0,t_idx], rzcom[1,t_idx], 'ko')
            plt.xlabel('R / cm')
            plt.ylabel('z / cm')

 
            tau_b += 1                              # Frame accepted, advance indices
            t_idx -= 1
 
 
        # Track blob forwards in time
        good_blob = True
        tau_a = 1               # Frame after detected blob event
        t_idx = tau_b_max + 1       # Index of frame, relative to t0
        r0_last = R0
        z0_last = z0

        while ( good_blob == True and tau_a < tau_a_max ):      
            print 'Tracking blob forward, frame %d tau_a = %d t_idx = %d' % (t0 + tau_a, tau_a, t_idx )
            event_frame = frames[t0 + tau_a, :, :]
            labels = pm.label( event_frame > 0.6 * I0 )         # Label the regions with more than 60% 
                                                                # of the original intensity
            blob_area = pm.blob( labels, 'area', output = 'data')       # Get blob areas
            blob_cent = pm.blob( labels, 'centroid', output = 'data')   # Get blob centers
#            print 'Centroid analysis', blob_cent, np.shape(blob_cent), np.size(blob_cent)
            if ( np.size(blob_cent) < 1 ):      # No peak here, quit tracking
                good_blob = False
#                print 'Lost track of blob'
                break

            min_dist_frame = np.sqrt(64.*64.+64.*64.)       # Minimal distance to last peak in current frame
            # We got a peak, see if it is within the maximum distance a peak may travel over one frame
            d_idx = 0
            for i in np.where( blob_area > 0.1 * max(blob_area) )[0]:
                # TODO: Vectorize loop
                dist = np.sqrt( (blob_cent[i,1]-z0_last)**2 + (blob_cent[i,0]-r0_last)**2 )
#                print 'Region %d, distance to last peak: %f' % (d_idx, dist)
                if ( dist < min_dist_frame and dist < blob_dist_max):
                    min_dist_frame = dist
                    min_idx = i
                d_idx += 1
            
            if ( min_dist_frame == np.sqrt( 8192. ) ):
#                print 'No peak satisfying criteria found. Stopping backward tracking'
                break
            
            # Compute the x and y COM coordinates of the blob, store
            blob_mask = labels!=(min_idx+1)
            event_masked = np.ma.MaskedArray(labels, mask = blob_mask, fill_value=0)
            xycom[:,t_idx] = com(event_masked)
            rzcom[:,t_idx] = com_rz(event_masked, rz_array[:,:,0], rz_array[:,:,1])
            print 't_idx %d, region center at (%d,%d), com coordinates (x,y) = (%f,%f), (R,z) = (%f,%f)' %\
                (t_idx, blob_cent[i,0], blob_cent[i,1], xycom[0,t_idx], xycom[1,t_idx], rzcom[0,t_idx], rzcom[1,t_idx])       

            # Round COM coordinates to be used as indices for FWHM analysis
            xcom_off, ycom_off = np.round(xycom[:,t_idx]).astype('int')    
            fig = plt.figure()

            # Do FWHM analysis only if blob is far enough away from the border of FOV
            if ( xycom[:,t_idx].min() > fwhm_max_idx and (64. - xycom[:,t_idx].max()) > fwhm_max_idx ):
                fwhm_rad_idx = fwhm(event_frame[ ycom_off, xcom_off - fwhm_max_idx : xcom_off + fwhm_max_idx ])
                fwhm_pol_idx = fwhm(event_frame[ ycom_off - fwhm_max_idx : ycom_off + fwhm_max_idx, xcom_off ])        
                fwhm_Rz[0,t_idx] = RRi[ ycom_off, xcom_off - fwhm_max_idx + fwhm_rad_idx[1][1] ] - RRi[ ycom_off, xcom_off - fwhm_max_idx + fwhm_rad_idx[1][0] ]
                fwhm_Rz[1,t_idx] = zzi[ ycom_off - fwhm_max_idx + fwhm_pol_idx[1][1] , xcom_off] - zzi[ ycom_off - fwhm_max_idx + fwhm_pol_idx[1][0] , xcom_off ]       
 
                fig.add_subplot(212)   
                plt.title('Cross sections')
                plt.plot(event_frame[ycom_off, xcom_off - fwhm_max_idx : xcom_off + fwhm_max_idx ], 'b-o', label='Radial cross, FWHM=%3.1f' % fwhm_Rz[0,t_idx])
                plt.plot( fwhm_rad_idx[1], event_frame[ ycom_off, (fwhm_rad_idx[1] + xcom_off - fwhm_max_idx).astype('int') ], 'b--' )
                plt.plot(event_frame[ ycom_off - fwhm_max_idx : ycom_off + fwhm_max_idx, xcom_off], 'g-o', label='Poloidal cross, FWHM=%3.1f' % fwhm_Rz[0,t_idx] )
                plt.plot( fwhm_pol_idx[1], event_frame[ (fwhm_pol_idx[1] + ycom_off - fwhm_max_idx).astype('int'), xcom_off ], 'g--' )
                plt.legend(loc='upper right')

 
            # Store FWHM, amplitude and blob position
            amp[t_idx] = event_frame[r0_last, z0_last]
            r0_last, z0_last = blob_cent[min_idx, :]

            plt.subplot(211)
            plt.title('frame %d' % (t0 + tau_a))
            plt.contour (RRi[0,:], zzi[:,0], event_frame, 16, colors = 'k', linewidth=0.5 )
            plt.contourf(RRi[0,:], zzi[:,0], event_frame, 16, cmap = plt.cm.hot )
            plt.colorbar()
            plt.plot(rzcom[0,t_idx], rzcom[1,t_idx], 'ko')
            plt.xlabel('R / cm')
            plt.ylabel('z / cm')


            tau_a += 1  # Frame accepted
            t_idx += 1
  

        print 'Blob event %d, frames %d:%d' % (idx, t0 - tau_b, t0 + tau_a )
    
        print 'Amplitude: %f Radial velocity: %f m/s' % ( amp.mean(), 0.01*(rzcom[0,-1] - rzcom[0,0]) / float(tau_a+tau_b)/dt )
        # Plot poloidal vs. radial position
        blob_t = np.arange(t0-tau_b_max, t0+tau_a_max)
        
        
        plt.figure()
        plt.subplot(121)
        plt.plot(rzcom[0,:], rzcom[1,:])
        plt.xlabel('R / cm')
        plt.ylabel('z / cm')
    
    
        # Plot blob size as a function of time    
        plt.subplot(122)
        plt.plot(blob_t, fwhm_Rz[0,:], 'x-', label='rad')
        plt.plot(blob_t, fwhm_Rz[1,:], 'x-', label='pol')
        plt.plot(blob_t, amp, 'x-', label='Intensity')
        plt.xlabel('time / frames')
        plt.ylabel('size / cm')
        plt.legend()
#        
#        blob_amp[idx] = amp.mean()
#        blob_vel[idx] = 0.01*(rzcom[0,-1] - rzcom[0,0]) / float(tau_a+tau_b)/dt
#        blob_lp[idx]  = fwhm_Rz[0,:].mean()
#        blob_lr[idx]  = fwhm_Rz[1,:].mean()
#
##    except:
##        event_ctr[idx] = -1
##        print 'Oooops'
#    
#plt.figure()
#plt.plot(blob_amp, blob_vel, 'o')
#plt.title('%d blob events' % (event_ctr == 1).sum() )
#plt.xlabel('Amplitude / a.u.')
#plt.ylabel('Velocity / ms^-1')
#
#plt.figure()
#plt.plot(blob_lp, blob_vel, 'o')
#plt.title('%d blob events' % (event_ctr == 1).sum() )
#plt.xlabel('Poloidal length / cm')
#plt.ylabel('Velocity / ms^-1')
#
#plt.figure()
#plt.plot(blob_lr, blob_vel, 'o')
#plt.title('%d blob events' % (event_ctr == 1).sum() )
#plt.xlabel('Radial length / cm')
#plt.ylabel('Velocity / ms^-1')
#
#
#print 'Analyzed shot %d, averaged particle density %f' % ( shotnr, 0.1 )#shotparams['n_avg'] )
#print '%d blobs analyzed, %d blobs ignored' % ( (event_ctr == 1).sum(), (event_ctr == -1).sum() )


plt.show()
