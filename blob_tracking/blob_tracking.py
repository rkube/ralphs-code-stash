#!/opt/local/bin/python
# -*- Encoding: UTF-8 -*-

import numpy as np
from misc.phantom_helper import make_rz_array
# from blob_tracking.detect_peak import detect_peak_3d
import blobtrail
from detect_peak import detect_peak_3d
# from scipy.interpolate import griddata
# import cPickle as pickle
from scipy.io import readsav


"""
Run blob detection on a set of GPI frames and store information about the
blob in a blobtrail
object. Return the list of blobtrail objects
"""


def blob_tracking(shotnr, frames, frame_info, frame0=0, minmax=[2.0, 10.0],
                  logger=None, nframes=30000):

    np.set_printoptions(linewidth=999999)
    # Begin analysis at this frame.
    frame0 = 0
    # Peaks within 2.5 and 10.0 times the rms
    minmax = np.array(minmax)
    # Deadtime after a peak in which no blob is detected
    lag = 20
    # Triggerbox r_low, r_up, z_low, z_up
    trigger = np.array([40, 50, 16, 48])
    # Total frame offset used in this script.
    # toffset = frame0 + lag
    # Maximal frames for which we track a blob
    tau_max = 7
    # 1 frame is 2.5µs
    # dt = 2.5e-6

    try:
        logger.info('frame0 = %d, nframes = %d' % (frame0, nframes))
    except:
        print 'frame0 = ', frame0, 'nframes = ', nframes

    # Load separatrix data for shot
    s = readsav('%d/%d_separatrix.sav' % (shotnr, shotnr), verbose=False)

    # Detect peaks
    # The detect_peak_3d returns time indices of blob events relative for
    # the array passed to it. Remember to add frame0 to idx_event[t0] to
    # translate to the frame indexing used in this script.
    idx_events = detect_peak_3d(frames[frame0:frame0+nframes, :, :],
                                trigger, minmax, 0, lag, rel_idx=False)
    num_events = np.shape(idx_events)[0]
    # event_ctr = np.ones([num_events])

    try:
        logger.info('%d blob events detected' % (num_events))
    except:
        print '%d blob events detected' % (num_events)

    # Define the events we will analyze
    event_range = np.arange(num_events)
#    event_range = np.arange( 5,10 )
#    num_events  = np.size(event_range)

    # Get R,z projection, grid data
    rz_array, transform_data = make_rz_array(frame_info)
    xxi, yyi = np.meshgrid(np.linspace(rz_array[:, :, 0].min(),
                                       rz_array[:, :, 0].max(), 64),
                           np.linspace(rz_array[:, :, 1].min(),
                                       rz_array[:, :, 1].max(), 64))
    xyi = np.concatenate((xxi[:, :, np.newaxis],
                          yyi[:, :, np.newaxis]), axis=2)
    trails = []
    fail_list = []
    failcount = 0

    for idx, event in enumerate(idx_events[event_range]):
        # I0 = event[0]
        t0 = event[1] + frame0
        # z0 = event[2]
        # R0 = event[3]

        print 'Tracking peak %d / %d, frame %d' % (idx, num_events, t0)
        # try:
        newtrail = blobtrail.blobtrail(frames[t0 - tau_max:
                                              t0 + tau_max, :, :],
                                       event, frame0, shotnr,
                                       thresh_amp=0.7, blob_ext=14,
                                       thresh_dist=8.,
                                       fwhm_max_idx=18,
                                       doplots=False)
        if (np.size(newtrail.get_tau()) < 4):
            fail_list.append(idx)
            failcount += 1

            log_str = 'Peak %d: Trail too short: %d frames' %\
                (idx, newtrail.get_tau.size)

            try:
                logger.info(log_str)
            except:
                print log_str

            continue

        #except ValueError, IndexError:
        #    fail_list.append(idx)
        #    failcount += 1
        #    log_str = 'Failed to track blob %d / %d' % (idx, num_events)
        #    try:
        #        logger.info(log_str)
        #    except:
        #        print log_str
        #    continue

        print 'Computing blob width'
        try:
            newtrail.compute_width_gaussian(frames, rz_array, position='MAX',
                                            i_size_max=10, plots=False)

        except:
            fail_list.append(idx)
            failcount += 1
            log_str = 'Peak %d: Unable to compute FWHM' % (idx)
            try:
                logger.info(log_str)
            except:
                print log_str

        newtrail.plot_trail(frames, rz_array=rz_array, xyi=xyi,
                            trigger_box=trigger, sep_data=s,
                            plot_com=True, plot_shape=True, plot_geom=True,
                            save_frames=True)
        trails.append(newtrail)

    return trails

# End of file blob_tracking.py
