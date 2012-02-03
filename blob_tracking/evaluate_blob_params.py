#!/opt/local/bin/python
#-*- Encoding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import time
import gaussfitter as gf
from detect_peak import detect_peak_3d
from helper_functions import frac_to_idx
from functions import gauss2d




"""
Test conditional averaging routine
"""

np.set_printoptions(linewidth=999999)
basedir = '/Users/ralph/source/blob_tracking'
shotnr = 1100803006
try:
    datafile = np.load('%s/%d/%d_frames.npz' % (basedir, shotnr, shotnr), mmap_mode = 'c')
    frames = datafile['frames_normalized_mean']
    print 'Loaded frames for shot %d' % shotnr
except IOError:
    print 'Could not open file %d/%d_frames.npz' % (shotnr, shotnr)

#datafile = np.load('../test/test_frames_200.npz')
#frames = datafile['frames']

frame0 = 20000                          # Begin analysis at this frame
nframes = 30000                         # Number of frames to analyze
minmax  = np.array([2.5, 4.5])          # Peaks within 1.5 and 2.5 times above rms
lag     = 20                            # Deadtime after a peak in which no blob is detected
time_arr= np.arange(0, nframes)         # Just the time, in frame numbers of the time series
trigger = np.array([40, 55, 8, 56])     # Triggerbox r_low, r_up, z_low, z_up
blobbox = np.array([8,8])               # Box used to determine blob size around single blob events
toffset = frame0 + lag                  # Total frame offset used in this script.
nbins   = 10                            # Bins for amplitude sorting


plt.figure()
plt.plot(frames[frame0:frame0+nframes, trigger[2]+8, trigger[0]+5])

# Detect peaks
idx_events = detect_peak_3d(frames[frame0:frame0+nframes,:,:], trigger, minmax, 0, lag, rel_idx=False)

#for event in idx_events:
#    print event

num_events = np.shape(idx_events)[0]
print '%d events' % num_events

print 'Removing events from the edge of the bounding box...'
idx_events = idx_events[idx_events['ridx'] > trigger[0] + 2]
idx_events = idx_events[idx_events['ridx'] < trigger[1] - 2]

num_events = np.shape(idx_events)[0]
print '%d events' % num_events

plt.figure()
plt.hist( idx_events['value'], bins=50)
plt.title('Blob amplitude occurence')
plt.ylabel('Occurence')
plt.xlabel('Amplitude')
amp_max = np.max(idx_events['value'])+0.0001
amp_min = np.min(idx_events['value'])

print 'max amplitude: %f min amplitude: %f' % (amp_max, amp_min)

# Field to average blobs
# first index describes percentage of max amplitude
# second and third index are z,R
avg_blob = np.zeros([nbins,16,16])
avg_num = np.zeros(nbins)

# Blob parameters for each amplitude bin
amp_bin = np.zeros(nbins)
size_x  = np.zeros(nbins)
size_y  = np.zeros(nbins)


Rr = np.arange(0,16)
zr = np.arange(0,16)

all_plot = False
for event in idx_events:
    t0 = event['tidx'] + frame0
    R0 = event['ridx'] # + trigger[0]
    z0 = event['zidx'] # + trigger[2]
    amp = event['value']
    avg_idx = frac_to_idx(amp, amp_min, amp_max, nbins)      # Which average blob amplitude this one goes to
#    print 'Event has amplitude %f, binning to idx %d' % (amp, avg_idx)
    avg_num[avg_idx] = avg_num[avg_idx] + 1         # Increase count of this event bin by one
    avg_blob[avg_idx, :,:] = avg_blob[avg_idx, :,:] + frames[t0, z0-8:z0+8, R0-8:R0+8]
    
    if all_plot:
        plt.figure()
        plt.subplot(221)
        plt.title('tidx=%d, ridx=%d zidx=%d' % (event['tidx']+frame0, event['ridx'], event['zidx']) )
        plt.contourf( frames[t0, :, : ], 32)
        plt.plot( R0 , z0 , 'ko')
        plt.colorbar()
        
        # Plot trigger box
        plt.plot( (trigger[1], trigger[0]), (trigger[2], trigger[2]), 'w')
        plt.plot( (trigger[1], trigger[0]), (trigger[3], trigger[3]), 'w')
        plt.plot( (trigger[0], trigger[0]), (trigger[2], trigger[3]), 'w')
        plt.plot( (trigger[1], trigger[1]), (trigger[2], trigger[3]), 'w')   

          
        plt.subplot(222)
        plt.contourf(Rr, zr, frames[t0, z0-8:z0+8, R0-8:R0+8], 32 )
        plt.colorbar()
    
        plt.subplot(223)
        plt.plot( frames[t0, z0, :], label='cut r' )
        plt.plot( frames[t0, :, R0], label='cut z')    
        plt.legend()


# Plot average form of blob if there is 1 or more count
for i in np.squeeze(np.argwhere(avg_num>0)):
    avg_blob[i,:,:] = avg_blob[i,:,:] / float(avg_num[i])

    plt.figure()
    plt.subplot(121)
    plt.title('Bin %d, %d events, average shape' % (i, avg_num[i]) )
    plt.contourf(avg_blob[i,:,:], 32)
    plt.colorbar()

    # Fit a gaussian to the average blob structure
    
    print 'Fitting gaussian to average blob shape'
    # Offset, maximum, center_x, center_y, sigma_x, sigma_y, rotation
    p0 = np.array([0., np.max(avg_blob[i,:,:]), 8, 8, 1., 1., 0.])
    tic = time.clock()
    result = gf.gaussfit( avg_blob[i,:,:], params = p0, return_all = 1 )
    toc = time.clock()
    xx, yy = np.meshgrid( np.arange(16), np.arange(16) )
    plt.contour(gauss2d(xx, yy, result[0]), 8)

    plt.subplot(122)
    plt.plot(Rr, avg_blob[i,8,:], label='z cut')
    plt.plot(zr, avg_blob[i,:,8], label='R cut')

    print 'Offset: %f Amplitude: %f, x_0: %f, y_0: %f, sigma_x: %f, sigma_y: %f, rot: %f' %\
        (result[0][0], result[0][1], result[0][2], result[0][3], result[0][4], result[0][5], result[0][6] )

    amp_bin[i] = result[0][1]
    size_x[i] = result[0][4]
    size_y[i] = result[0][5]


plt.figure()
plt.plot(amp_bin, size_x, 'bo', label='size_x')
plt.plot(amp_bin, size_y, 'go', label='size_y')
plt.xlabel('Amplitude')
plt.ylabel('size')
plt.title('Average blob sizes')
plt.legend(loc='upper left')
plt.show()
