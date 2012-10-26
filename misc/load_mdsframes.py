#!/opt/local/bin/python
#-*- Encoding: UTF-8 -*-

import numpy as np

"""
Loads frames grabbed from MDS tree in memory
"""

def load_mdsframes(shotnr, test = False, path = '/Users/ralph/source/blob_tracking'):

    if ( test == False ):
        # Open file in copy-on-write mode
        print 'Loading frames from %d/%d_frames.npz' % (shotnr, shotnr)
        datafile = np.load('%s/%d/%d_frames.npz' % (path, shotnr, shotnr), mmap_mode = 'c')
        frames = datafile['frames_normalized_mean']
        print 'Loaded frames for shot %d' % shotnr
    else:
        print 'Loading frames from %d/%d_testframes.npz' % (shotnr, shotnr)
        datafile = np.load('%s/%d/%d_testframes.npz' % (path, shotnr, shotnr) )
        frames = datafile['frames']

    frame_info = datafile['frame_info']
    
    # Make frames read-only
    frames.flags.writeable = True

    return frames, frame_info #, datafile['frames_mean'], datafile['frames_rms']
