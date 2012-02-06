#!/opt/local/bin/python
#-*- Encoding: UTF-8 -*-

import numpy as np
#import pymorph as pm
import matplotlib.pyplot as plt
from detect_peak import detect_peak_3d
#from helper_functions import com, com_rz, fwhm
from tracker import tracker
from phantom_helper import make_rz_array
from scipy.interpolate import griddata


"""
Check out how image segmenting works
"""
np.set_printoptions(linewidth=999999)
shotnr  = 1100803015
frame0 = 20000                          # Begin analysis at this frame. 
nframes = 30000                         # Number of frames to analyze
minmax  = np.array([2.5, 10.0])         # Peaks within 1.5 and 2.5 times above rms
lag     = 20                            # Deadtime after a peak in which no blob is detected
time_arr= np.arange(0, nframes)         # Just the time, in frame numbers of the time series
trigger = np.array([40, 50, 10, 53])    # Triggerbox r_low, r_up, z_low, z_up
toffset = frame0 + lag                  # Total frame offset used in this script.
tau_b_max   = 7                         # Maximal time intervall for which we track a blob
tau_a_max   = 7
thresh_amp = 0.6
thresh_dis = 8.                         # Maximal number of pixels a blob may travel in one time
fwhm_max_idx    = 10                    # Required minimum pixel to boundary for FWHM analysis 
blob_ext = 12                            # Use +- 8 px to average the blob shape
# 1 frame is 2.5µs
dt = 2.5e-6

try:
    datafile = np.load('../../blob_tracking/%d/%d_frames.npz' % (shotnr, shotnr), mmap_mode = 'c')
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

# Define the events we will analyze
event_range = np.arange(0,num_events)
num_events  = np.size(event_range)

# Allocate arrays for size, extend and velocity of blobs
blob_amp = np.zeros([num_events, tau_b_max + tau_a_max])            # Amplitude
blob_pos = np.zeros([num_events, tau_b_max + tau_a_max, 2])
blob_lr  = np.zeros([num_events, tau_b_max + tau_a_max, 2])         # Radial length at COM pos. in pixel 
blob_lp  = np.zeros([num_events, tau_b_max + tau_a_max, 2])         # Poloidal length at COM pos. in pixel
blob_vel_avg = np.zeros([num_events, 2])                            # Radial velocity in m / s, (radial , poloidal)
blob_ell_avg = np.zeros([num_events, 2])                            # Physical blob size, in mm (radial, poloidal)
blob_amp_avg = np.zeros([num_events])
blob_vel_avg = np.zeros([num_events, 2])                            # Blob velocity (radial, poloidal)        
blob_t   = np.zeros([num_events, tau_b_max + tau_a_max])

# Get R,z projection, grid data
rz_array, transform_data = make_rz_array(frame_info)
RRi, zzi = np.meshgrid( np.linspace( np.min(rz_array[:,:,0] ), np.max( rz_array[:,:,0] ),64 ), np.linspace( np.min( rz_array[:,:,1] ), np.max( rz_array[:,:,1] ),64 ) )
Rzi = np.concatenate( (RRi[:,:,np.newaxis], zzi[:,:,np.newaxis]), axis=2 )
zi = griddata(rz_array.reshape(64*64, 2), frames[666,:,:].reshape( 64*64 ), Rzi.reshape( 64*64, 2 ), method='linear' )

failcount = 0
fail_list = []

for idx, event in enumerate(idx_events[event_range]):
    I0 = event[0] 
    t0 = event[1] + frame0
    z0 = event[2]
    R0 = event[3]    

    print 'Frame %d, peak at (%d,%d): %f' % (t0, R0, z0, frames[t0,z0,R0])
#    plt.figure()
#    plt.contour (frames[t0, :, :], 16, colors = 'k', linewidth=0.5 )
#    plt.contourf(frames[t0, :, :], 16, cmap = plt.cm.hot )
#    plt.title('Frame %d, peak at (%d,%d): %f' % (t0, R0, z0, frames[t0,z0,R0]) )
#    plt.colorbar()
#    plt.plot( z0, R0, 'ko' )

    # Do backwards tracking
    # If the blob is within fwhm_max_idx of the boundary, there will be no fwhm data from the 
    # tracking algorithm. Instead there is a 0 in the return value
    try:
        tau_b, amp_b, xycom_b, fwhm_rad_idx_b, fwhm_pol_idx_b, blob_shape_b = tracker(frames[t0-tau_b_max:t0,:,:], event, thresh_amp, thresh_dis, fwhm_max_idx, blob_ext, 'backward', plots = False)
        # Do forwards tracking
        tau_f, amp_f, xycom_f, fwhm_rad_idx_f, fwhm_pol_idx_f, blob_shape_f = tracker(frames[t0:t0+tau_a_max,:,:], event, thresh_amp, thresh_dis, fwhm_max_idx, blob_ext, 'forward', plots = False)
    except:
        print 'Something went wront. Blob tracking failed'
        failcount += 1
        fail_list.append(t0)
        continue
    if ( tau_b + tau_f == 1 ):
        print 'Too few frames. Blob tracking failed'
        failcount += 1
        fail_list.append(t0)
        continue
    
    tau = np.arange( -tau_b, tau_f )          
    amp = np.concatenate( (amp_b[tau_b:0:-1], amp_f[:tau_f]) , axis=0 )
    xycom = np.concatenate( (xycom_b[tau_b:0:-1,:], xycom_f[:tau_f,:]) , axis=0 )
    fwhm_rad_idx = np.concatenate( (fwhm_rad_idx_b[tau_b:0:-1,:], fwhm_rad_idx_f[:tau_f,:]), axis=0 )
    fwhm_pol_idx = np.concatenate( (fwhm_pol_idx_b[tau_b:0:-1,:], fwhm_pol_idx_f[:tau_f,:]), axis=0 )
    blob_shape = ( blob_shape_b + blob_shape_f ) / ( tau_b + tau_f )
    
    
    # Concatenate blob tracking results
    #print 'single', amp_b, ' and', amp_f
    #print 'concatenated', amp
    #print 'single', xycom_b, xycom_f
    #print 'concatenated', xycom
    #print 'single', fwhm_rad_idx_b, 'and ', fwhm_rad_idx_f
    #print 'concatenated', fwhm_rad_idx
    #print 'single', fwhm_pol_idx_b, ' and ', fwhm_pol_idx_f
    #print 'concatenated', fwhm_pol_idx

    # Remember, tau_b and tau_f are the number of frames a blob is tracked backwards and
    # forwards. tau_b includes the frame where the blob was detected. Subtract 1 to scope the
    # array over whole time intervall where the blob is detected.
    blob_amp[idx,: tau_b + tau_f ] = amp[:]
    blob_pos[idx,: tau_b + tau_f ] = xycom[:]
#    blob_lr[idx,: tau_b + tau_f ] = fwhm_rad_idx[:]
#    blob_lp[idx,: tau_b + tau_f ] = fwhm_pol_idx[:]
    blob_t[idx, : tau_b + tau_f ] = np.arange(t0 - tau_b, t0 + tau_f )
    
    if ( np.isnan( amp.mean() ) ):
        print 'Amplitude is infinity. Blob tracking failed'
        fail_list.append(t0)
        continue

    blob_amp_avg[idx] = amp.mean()
    print 'Average amplitude at COM position', amp.mean()
    
    ###
    ### DO NOT FUCK WITH THE INDICES BELOW!!!!!!!
    ###
    
    # Convert blob sizes from indices to physical units
    xycom = xycom.astype('int')
    blob_pos_rz = Rzi[xycom[:,1], xycom[:,0] ,:]
    # Compute physical blob velocity. Divide 
    blob_vel_avg[idx,:] = 1e-2*( blob_pos_rz[-1,:] - blob_pos_rz[0,:] ) / (dt * (tau_b+tau_f-1.0) )  

    ###
    ### SERIOUSLY, DON'T TOUCH THE INDICES!!!!
    ###

    # Compute the average length of the blob while tracked
    # Ignore frames where not FWHM has been determined. 
    # If FWHM hasn't been computed for any index this leads to a 2d array.
    lr_idx = fwhm_rad_idx[fwhm_rad_idx[:,0] > 0, :].round().astype('int')
    lp_idx = fwhm_pol_idx[fwhm_pol_idx[:,0] > 0, :].round().astype('int')
    
    # for fwhm_[rp]_idx :
    # 1st index is lower(0) and upper(1) fwhm position, 2nd index is radial(0) and poloidal(1) COM coordinate 
    # 3rd index is time 0..tau_b+tau_f
    fwhm_r_idx = np.array([ [lr_idx[:,0], xycom[:,1]], [lr_idx[:,1], xycom[:,1]] ])
    fwhm_p_idx = np.array([ [xycom[:,0], lp_idx[:,0]], [xycom[:,0], lp_idx[:,1]] ])
    
    if ( fwhm_r_idx.ndim == 3 ):   
    #   Compute the radial position in cm of the lower and upper computed FWHM at the poloidal COM position of the blob
    #   for each frame. Then tak the mean
    #    print 'left radial fwhm position in px', fwhm_r_idx[0,:,:]
    #    print 'left radial fwhm position in cm', Rzi[fwhm_r_idx[0, 1, :], fwhm_r_idx[0, 0, :], :]
    #    
    #    print 'right radial fwhm positoin in px', fwhm_r_idx[1,:,:]
    #    print 'right radial fwhm position in cm', Rzi[fwhm_r_idx[1, 1, :], fwhm_r_idx[1, 0, :], :]
    #    print 'mean radial fwhm: %3.2fcm ' % ( (Rzi[fwhm_r_idx[1, 1, :], fwhm_r_idx[1, 0, :], :] - Rzi[fwhm_r_idx[0, 1, :], fwhm_r_idx[0, 0, :], :]).mean(axis=0)[0] )
        
    #   Do the same for the poloidal FWHM of the blob at the radial COM position of the blob in each frame
    #    print 'lower poloidal fwhm position in px', fwhm_p_idx[0,:,:]
    #    print 'lower poloidal fwhm position in cm', Rzi[fwhm_p_idx[0, 1, :], fwhm_p_idx[0, 0, :], :]#
    #
    #    print 'upper poloidal fwhm position in px', fwhm_p_idx[1,:,:]
    #    print 'upper poloidal fwhm position in cm', Rzi[fwhm_p_idx[1, 1, :], fwhm_p_idx[1, 0, :], :]
    #
    #    print 'mean poloidal fwhm: %3.2fcm' % ( (Rzi[fwhm_p_idx[1, 1, :], fwhm_p_idx[1, 0, :], :] - Rzi[fwhm_p_idx[0, 1, :], fwhm_p_idx[0, 0, :], :]).mean(axis=0)[1] )
        
        blob_ell_avg[idx, 0] = ( (Rzi[fwhm_r_idx[1, 1, :], fwhm_r_idx[1, 0, :], :] - Rzi[fwhm_r_idx[0, 1, :], fwhm_r_idx[0, 0, :], :]).mean(axis=0)[0] ) / 2.355
        blob_ell_avg[idx, 1] = ( (Rzi[fwhm_p_idx[1, 1, :], fwhm_p_idx[1, 0, :], :] - Rzi[fwhm_p_idx[0, 1, :], fwhm_p_idx[0, 0, :], :]).mean(axis=0)[1] ) / 2.355
    
    else:
        blob_ell_avg[idx,:] = np.array([0., 0.])
    
    print 'Tracked blob %d/%d for %d frames backwards and %d frames forward: ' % (idx, num_events, tau_b, tau_f)
    print 'Amplitude: %4.3f, velocity: rad: %5.2f m/s, pol: %5.2f m/s' % (blob_amp_avg[idx], blob_vel_avg[idx,0], blob_vel_avg[idx,1])
    
#    plt.figure()
#    plt.title('Event %d, blob shape' % idx)
#    plt.contour (blob_shape, 16, colors = 'k', linewidth=0.5 )
#    plt.contourf(blob_shape, 16, cmap = plt.cm.hot )
#    plt.colorbar()

plt.figure()
plt.title('Average blob size')
plt.hist( blob_ell_avg[:, 0], label='radial', bins=20)
plt.hist( blob_ell_avg[:, 1], label='poloidal', histtype='step', bins=20)
plt.xlabel('Size / cm')
plt.ylabel('Counts')
plt.legend()

#plt.figure()
#plt.title('Average blob amplitude')
#plt.hist( blob_amp_avg, bins=20 )
#plt.xlabel('Amplitude a.u.')
#plt.ylabel('Counts')

plt.figure()
plt.title('Blob velocity')
plt.hist( blob_vel_avg[:,0], label='radial', bins=20 )
plt.hist( blob_vel_avg[:,1], label='poloidal', histtype='step', bins=20)
plt.xlabel('Velocity m/s')
plt.ylabel('Counts')
plt.legend()


np.savez('../../blob_tracking/%d/%d_results.npz' % (shotnr, shotnr), blob_ell_avg = blob_ell_avg, blob_amp_avg = blob_amp_avg,\
    blob_vel_avg = blob_vel_avg, fail_list = fail_list )

plt.show()
