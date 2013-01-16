#!/opt/local/bin/python
#-*- Encoding: UTF-8 -*-


"""
=======================
Smoothing of dataseries
=======================


Taken from http://www.scipy.org/Cookbook/SignalSmooth

* smooth....smooth a dataseries using a window with requested size
* moving_average: an alternative implementation
* moving_rms: Compute the RMS within a sliding windows: RMS(x_i) = np.sum(

"""

import numpy as np
import ctypes
import matplotlib.pyplot as plt
from multiprocessing import Pool

def smooth( x, window_len = 5, window = 'hanning' ):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string   
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]

    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y



def moving_average(x, window_len = 11):
    """
    Another moving average function.
    <X>_i = sum_j=-k^k x_i+j

    """

    one_over_size = 1./float(window_len)
    window = np.ones(window_len) * one_over_size
    result = np.convolve(x, window, 'valid')
    # Cut off the clutter at the boundaries
    #return result[ (window_len-1)/2 : -(window_len-1)/2]
    return result



def moving_rms(x, window_len = 11, smp=False, nthreads=4):
    """
    Compute the standard deviation on a running window 

    RMS(X)_i = sum_j=-k^k sqrt{ (X_i - <X>_k)^2/(2k+1) }

    Assume that the running mean has already been subtracted from the signal
    """
    #if ( np.size(running_mean) != np.size(x)):
    #    raise ValueError, "array and running mean of the array must be the same size."
    # Assure that window_len is even. If not, decrease window_len by one
    if window_len%2 == 0:
        window_len = window_len-1
    i0 = (window_len-1)/2
    result = np.zeros_like(x)
    if smp:
        print 'smp Keyword does not do anything yet' 
  
#    if smp:
#        p = Pool(nthreads)
#        f = lambda x, i, i0: x[i-i0:i+i0].std()
#        
#    else:
    for i in np.arange(i0, np.size(x)-i0):
        #result[i] = x[i-i0:i+i0].std()
        result[i] = np.sqrt( np.sum(x[i-i0:i+i0+1]*x[i-i0:i+i0+1])/(window_len))

    return result[i0:-i0]



def normalize_timeseries(timeseries, radius=16384, blocksize=128, maxelem = 262144, show_plots = False):
    """
    Normalize a timeseries via:

    \tilde{A}_i = ( A_i-<A>_i ) / RMS(A)_i

    where <A>_i is the running mean defined via:

    <A>_i = 1/(2K+1) \sum_{i=-K}^{K} A_i

    and RMS(A)_i is the running RMS, defined via

    RMS(A)_i = \sqrt( 1/(2K+1) \sum_{i=-K}^{K} (A_i -<A>_i)^2 ).

    Load a c++ shared library that wraps a CUDA implementation of the
    running average and running RMS.
    """ 


    if( np.mod(blocksize, 2) == 1):
        raise ValueError('The blocksize %d must be a power of 2.' % blocksize)

    if ( np.mod(radius, 2) == 1):
        raise ValueError('The radius %d must be a power of 2.' % radius)

    if ( np.mod(maxelem, 2) == 1):
        raise ValueError('The maximal number of elements must be a power of 2.' % maxelem)

    module_name = '/Users/ralph/source/cuda/lib/running_marms_cuda_bs%d_r%d.so' % (blocksize, radius)
    try:
        ma_lib = ctypes.cdll.LoadLibrary(module_name)
    except:
        raise NameError('Could not load CUDA module for blocksize: %d, radius: %d: %s' % (blocksize, radius, module_name) )


    timeseries_normalized = np.zeros_like(timeseries)

    # Setup input data for the wrapper functions
    num_points_avg = maxelem + 2*radius
    num_points_rms = maxelem

    indata_rms = np.zeros(num_points_rms + 2*radius, dtype='float64')

    # Arrays that store the result from the module routine calls
    result_ma = np.zeros( num_points_avg, dtype='float64')
    result_rms = np.zeros( num_points_rms, dtype='float64')

    # ctypes of local variables passed to module routines
    c_numpoints_avg = ctypes.c_int(num_points_avg)
    c_numpoints_rms = ctypes.c_int(num_points_rms)
    c_result_ma = ctypes.c_void_p(result_ma.ctypes.data)
    c_result_rms = ctypes.c_void_p(result_rms.ctypes.data)

    # Crop timeseries to a multiple of the blocksize
    numel = np.size(timeseries)

    # The CUDA module is a lit picky about the max size it accepts. If the array is longer than
    # 2^19 = 524288 elements, the machine crashes :(
    # Cut the whole timeseries in chunks of size S= 2^19 + 4*radius, where consecutive chunks have an overlap
    # of 2*radius elements. This is, because both, application of running average and application of running RMS
    # need to cut off radius elements at the boundaries of the interval.

    maxelem = maxelem + 4*radius
    
    elements_left = np.size(timeseries)
    loop_pass = 0
    num_pass = np.ceil( float(numel) / float(maxelem) )
    idx_start = 2*radius

    while(loop_pass < num_pass):
        if ( (numel - idx_start) < 4*radius):
            break
        #print 'Smoothing, pass %d/%d. Starting at %d, %d remaining' % (loop_pass, num_pass, idx_start, numel-idx_start)
        #print 'Processing %d elements' % ( np.size(timeseries[idx_start : idx_start + maxelem ]))

        print 'Computing moving average of:', timeseries[idx_start:idx_start+5] , 'storing to:', result_ma[:5]
        c_indata_avg = ctypes.c_void_p(timeseries[idx_start:idx_start + maxelem].ctypes.data)
        result = ma_lib.ma_cuda(c_indata_avg, c_result_ma, c_numpoints_avg)
        print 'Done. Size of returned array: %d' % (np.size(result_ma)) 

        #print 'Computing moving RMS...'
        indata_rms[:] = timeseries[idx_start + radius : idx_start + maxelem - radius] - result_ma
        c_indata_rms = ctypes.c_void_p(indata_rms.ctypes.data)
        result = ma_lib.rms_cuda(c_indata_rms, c_result_rms, c_numpoints_rms)
        #print 'Done.' 

        if show_plots:
            fig = plt.figure()
            ax_1 = fig.add_subplot(411)
            ax_1.plot( np.arange(idx_start, idx_start + maxelem), timeseries[idx_start : idx_start + maxelem])

            ax_2 = fig.add_subplot(412, sharex = ax_1)
            ax_2.plot( np.arange(idx_start+radius, idx_start+maxelem-radius), result_ma)

            ax_3 = fig.add_subplot(413, sharex = ax_1)
            ax_3.plot( np.arange(idx_start+2*radius, idx_start+maxelem-2*radius), result_rms)

            ax_4 = fig.add_subplot(414, sharex = ax_1)
            ax_4.plot( np.arange(idx_start + 2*radius, idx_start + maxelem - 2*radius), (timeseries[idx_start + 2*radius : idx_start + maxelem - 2*radius]- result_ma[radius:-radius])/result_rms )

            ax_1.set_ylabel('Isat')
            ax_2.set_ylabel('Moving avg.')
            ax_3.set_ylabel('Moving RMS')
            ax_4.set_ylabel('Normalized Isat')

            plt.setp(ax_1.get_xticklabels(), visible=False)
            plt.setp(ax_2.get_xticklabels(), visible=False)

            plt.show()
        timeseries_normalized[idx_start + 2*radius : idx_start + maxelem - 2*radius] = (timeseries[idx_start+2*radius : idx_start + maxelem - 2*radius] - result_ma[radius:-radius])/result_rms
        idx_start = idx_start + maxelem - 4*radius
        loop_pass += 1

    return timeseries_normalized[2*radius:-2*radius]

