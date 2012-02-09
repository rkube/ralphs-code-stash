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
Do blob tracking, but use the blobtrail class
"""
np.set_printoptions(linewidth=999999)
shotnr  = 1100803008
frame0 = 20000                          # Begin analysis at this frame. 
nframes = 30000                         # Number of frames to analyze
minmax  = np.array([2.5, 10.0])         # Peaks within 1.5 and 2.5 times above rms
lag     = 20                            # Deadtime after a peak in which no blob is detected
time_arr= np.arange(0, nframes)         # Just the time, in frame numbers of the time series
trigger = np.array([40, 50, 10, 53])    # Triggerbox r_low, r_up, z_low, z_up
toffset = frame0 + lag                  # Total frame offset used in this script.
tau_max   = 7                         # Maximal time intervall for which we track a blob
# 1 frame is 2.5µs
dt = 2.5e-6

try:
    datafile = np.load('../../blob_tracking/%d/%d_frames.npz' % (shotnr, shotnr), mmap_mode = 'c')
    frames = datafile['frames_normalized_mean']
    print 'Loaded frames for shot %d' % shotnr
except IOError:
    print 'Could not open file %d/%d_frames.npz' % (shotnr, shotnr)
    datafile = np.load('../../blob_tracking/1100803005/1100803005_testframes.npz')
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
event_range = np.arange(num_events)
num_events  = np.size(event_range)

# Get R,z projection, grid data
rz_array, transform_data = make_rz_array(frame_info)
RRi, zzi = np.meshgrid( np.linspace( np.min(rz_array[:,:,0] ), np.max( rz_array[:,:,0] ),64 ), np.linspace( np.min( rz_array[:,:,1] ), np.max( rz_array[:,:,1] ),64 ) )
Rzi = np.concatenate( (RRi[:,:,np.newaxis], zzi[:,:,np.newaxis]), axis=2 )
zi = griddata(rz_array.reshape(64*64, 2), frames[666,:,:].reshape( 64*64 ), Rzi.reshape( 64*64, 2 ), method='linear' )

failcount = 0
fail_list = []

trails = []

blob_amps = np.zeros([num_events])
blob_ell  = np.zeros([num_events, 2])
blob_vel  = np.zeros([num_events, 2])
blob_shape= np.zeros([16,16])


for idx, event in enumerate(idx_events[event_range]):
    I0 = event[0] 
    t0 = event[1] + frame0
    z0 = event[2]
    R0 = event[3]    
    
    print 'peak %d / %d' % ( idx, num_events)
    try:
        newtrail = blobtrail( frames[t0 - tau_max : t0 + tau_max, :, :], event, frame0, doplots = False )
    except ValueError:
        fail_list.append(idx)
        failcount += 1
        continue
    
    newtrail.compute_fwhm(frames, rz_array, position = 'MAX', norm = True, plots = False)
    
    try:
        blob_vel[idx]   = newtrail.get_velocity_com().max( axis=0 )
        blob_amps[idx] = newtrail.get_amp()[newtrail.get_frame0()]
        blob_ell[idx,0] = newtrail.get_ell_rad()[newtrail.get_frame0()]
        blob_ell[idx,1] = newtrail.get_ell_pol()[newtrail.get_frame0()]
        
    except AssertionError:
        fail_list.append(idx)
        failcount += 1
        continue

    # Collect blob shapes at time where the maximum was recorded
    blob_shape += newtrail.get_blob_shape(frames, position = 'MAX', frameno = np.arange(1)).mean(axis=0)
    trails.append(newtrail)
 
 
blob_shape /= float(num_events - failcount) 

F = plt.figure()
F.text(0.5, 0.95, 'shot #%d, %d blob events' % (shotnr, num_events), ha='center')
plt.subplot(221)
plt.xlabel('$\\bar{I}$ / a.u.')
plt.ylabel('Counts')
plt.hist( blob_amps, bins = 25 )

plt.subplot(222)
plt.title('Length at normalized amplitude' )
plt.hist( blob_ell[:, 0], bins = 25, histtype = 'step', label='radial', linewidth = 2 )
plt.hist( blob_ell[:, 1], bins = 25, histtype = 'step', label='poloidal', linewidth = 2)
plt.xlabel('Length / cm')
plt.ylabel('Counts')
plt.legend(loc = 'upper left')

plt.subplot(223)
plt.hist( blob_vel[:, 0], bins = 25, histtype = 'step', label='radial', linewidth = 2 )
plt.hist( blob_vel[:, 1], bins = 25, histtype = 'step', label='poloidal', linewidth = 2 )
plt.xlabel('Velocity / m/s')
plt.ylabel('Counts')
plt.legend(loc = 'upper left')

plt.subplot(224)
plt.title('Mean structure at $\\max{\\bar{I}}$')
plt.contour (blob_shape, 15, colors='k')
plt.contourf(blob_shape, 15, cmap = plt.cm.hot)
plt.colorbar()


plt.figure()
plt.subplot(121)
plt.title('Amplitude vs. blob length')
plt.scatter(blob_amps, blob_ell[:,0], label='radial')
plt.scatter(blob_amps, blob_ell[:,1], label='poloidal')
plt.legend()
plt.xlabel('Intensity / a.u.')
plt.ylabel('Length / cm')

plt.subplot(122)
plt.title('Amplitude vs. blob velocity')
plt.scatter(blob_amps, blob_vel[:, 0], marker = '>', label='radial')
plt.scatter(blob_amps, blob_vel[:, 1], marker = '^', c='g', label='poloidal')
plt.legend()
plt.xlabel('Intensity / a.u.')
plt.ylabel('Velocity / m/s')
# Store the blob events for later analysis
pickle.dump(trails, open('../../blob_tracking/%d/%d_blobs.pkl' % (shotnr, shotnr), 'wb'))

plt.show()    