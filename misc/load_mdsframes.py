#!/opt/local/bin/python
#-*- Encoding: UTF-8 -*-

import numpy as np

"""
Loads frames grabbed from MDS tree in memory
"""

#def load_mdsframes(shotnr, test=False, path='/Users/ralph/source/blob_tracking'):
def load_mdsframes(shotnr, test=False, path=None):

    # Set the default path
    if path is None:
        path = '/Users/ralph/source/blob_tracking/%10d' % (shotnr)

    if (test is False):
        # Open file in copy-on-write mode
        print 'Loading frames from %s/%d_frames.npz' % (path, shotnr)
        datafile = np.load('%s/%10d_frames.npz' % (path, shotnr), mmap_mode = 'c')
        frames = datafile['frames_normalized_mean']
        print 'Loaded frames for shot %d' % shotnr
    else:
        print 'Loading frames from %s/%d_testframes.npz' % (shotnr, shotnr)
        datafile = np.load('%s/%10d_testframes.npz' % (path, shotnr) )
        frames = datafile['frames']

    frame_info = datafile['frame_info']
    
    # Make frames read-only
    frames.flags.writeable = True

    return frames, frame_info #, datafile['frames_mean'], datafile['frames_rms']


def load_mdsframes_mean_rms(shotnr, test=False, path='/Users/ralph/source/blob_tracking'):

    if (test is False):
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

    print datafile.keys()

    return frames, frame_info, datafile['frames_mean'], datafile['frames_rms']

#End of file load_mdsframes.py
