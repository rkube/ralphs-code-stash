#!/opt/local/bin/python
#-*- Encoding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from gaussfitter import gf

from cond_avg_2d import cond_avg_top_peak_surface


"""
Use a trigger pixel of the phantom data to detect blobs. Then cut out an area behind the
separatrix with almost constant rms and mean to detect blobs.

"""


frame0 = 15000
nframes = 50000
shotnr = 1100803006
datadir = '/Users/ralph/source/blob_tracking'


try:
    datafile = np.load('%s/%d/%d_frames.npz' % (datadir, shotnr, shotnr), mmap_mode = 'c')
    frames = datafile['frames_normalized_mean']
    print 'Loaded frames for shot %d' % shotnr
except IOError:
    print 'Could not open file %d/%d_frames.npz' % (shotnr, shotnr)


px      = np.array([32,50])         # Choose a pixel well after the separatrix
minmax  = np.array([3., 6.])        # Peaks within 1.5 and 2.5 times above rms
lag     = 20
t       = np.arange(-lag, lag)
tseries = frames[:, px[0], px[1]]
time    = np.arange(0, nframes)
bb_zl   = 20
bb_zu   = 44
bb_rl   = 35
bb_ru   = 55


plt.figure()
plt.plot(time[:frame0], tseries[:frame0], 'r--')
plt.plot(time[frame0:], tseries[frame0:], 'k')

# Detect peaks
cavg_window, idx_events = cond_avg_top_peak_surface(frames, px, minmax, frame0, lag, rel_idx=True)
num_events = np.size(idx_events)

blob_box = np.zeros([num_events, bb_zu-bb_zl, bb_ru-bb_rl ])
print np.shape(blob_box)

for num, event in enumerate(idx_events):
    blob_box[num, :, :] = frames[frame0+event : frame0+event+lag, bb_zl : bb_zu, bb_rl : bb_ru].mean(axis=0)
    
    plt.figure()
    plt.contourf(blob_box[num, :, :])
    plt.title('#%d, time avg [%d:%d]x[%d:%d]' % (shotnr, bb_zl, bb_zu, bb_rl, bb_ru) )
    plt.xlabel('x / px')
    plt.ylabel('y / px')
    plt.colorbar()

    # Fit a gaussian to the identified blobs



plt.show()