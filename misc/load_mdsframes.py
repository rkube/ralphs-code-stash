#!/opt/local/bin/python
#-*- Encoding: UTF-8 -*-

import numpy as np

"""
Loads frames grabbed from MDS tree in memory
"""

def load_mdsframes(shotnr, test = False):

    if ( test == False ):
        # Open file in copy-on-write mode
        datafile = np.load('%d/%d_frames.npz' % (shotnr, shotnr), mmap_mode = 'c')
        frames = datafile['frames_normalized_mean']
        print 'Loaded frames for shot %d' % shotnr
    else:
        print 'Could not open file %d/%d_frames.npz' % (shotnr, shotnr)
        datafile = np.load('../../blob_tracking/%d/%d_testframes.npz' % (shotnr, shotnr) )
        frames = datafile['frames']

    frame_info = datafile['frame_info']
    
    # Make frames read-only
    frames.flags.writeable = True

    return frames, frame_info, datafile['frames_mean'], datafile['frames_rms']
