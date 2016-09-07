#!/opt/local/bin/python
#-*- Encoding: UTF-8 -*-

import numpy as np

"""
Loads frames grabbed from MDS tree in memory
"""

def load_mdsframes(shotnr, test=False, path=None):
    """
    Load phantom frames preprocessed from
    cmodws:~rkube/misc_scripts/save_phantomr_frames.py

    The data is assumed to be in a directory structure as
    basedir/shotnr/GPI/phantom

    Input:
    ======
    shotnr:     int, the shot for which we attempt to get the frames
    test:       bool, When True, load a smaller test file (if it is there). If not, load all frames.
    path:       string, The basedir where to look for the data file

    Output:
    =======

    frames:      ndarray, float: The frames normalized to the RMS
    frame_info:  dictionary: Information about the view angle of the phantom camera
    frames_mean: ndarray, float: The mean subtracted from the raw frames for normalization
    frames_rms:  ndarray, float: The rms the raw frames were divided by for normalization
    """

    # Set the default path
    if path is None:
        #path = '/Users/ralph/source/blob_tracking/%10d' % (shotnr)
        path = '/Users/ralph/uni/cmod/data'

    if test: 
        fname_df = '%s/%10d/GPI/phantom/%10d_testframes.npz'
    else:
        # Open file in copy-on-write mode
        fname_df = '%s/%10d/GPI/phantom/%10d_frames_normalized.npz' % (path, shotnr, shotnr)

        print 'Loading frames from %s' % (fname_df)
        df = np.load(fname_df, mmap_mode='c')
        frames = df['frames_normalized_rms']

    frame_info = df['frame_info']
    
    # Make frames read-only
    frames.flags.writeable = True

    print df.keys()

    return frames, frame_info, df['frames_mean'], df['frames_rms']


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
