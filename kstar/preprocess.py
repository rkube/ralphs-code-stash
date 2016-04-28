#!/opt/local/bin/python
#-*- Encoding: UTF-8 -*-

import numpy as np

def find_good_tidx(tb, bad_times_tuple):
    """
    Removes part of the signal, as specified by time intervals listed in bad_times.

    Input:
    ======
    tb:     ndarray, float: Time base of the signal. Must have size of signal.
    bad_times: tuple of 2-tuples. Each 2-tuple specifies a time interval which we wish to remove

    Output:
    =======
    good_tidx: Good time indices that are not marked by bad_times

    """

    good_tidx = np.ones(tb.size, dtype='bool')

    for bad_times in bad_times_tuple:
        #print bad_times, (tb > bad_times[0]).sum(), (tb < bad_times[1]).sum()
        good_tidx[(tb > bad_times[0]) & (tb < bad_times[1])] = False
        print bad_times, 'good_tidx.sum = %d' % (good_tidx.sum())

        #signal = signal[good_tidx]
        #tb = tb[good_tidx]
        #print 'signal.size = %d' % (signal.size)

    #print '----------------------------------'
    #print 'signal.size = %d' % (signal.size)
    return good_tidx

# End of file preprocess.py
