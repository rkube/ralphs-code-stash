#!/usr/bin/env python
#-*- Encoding: UTF-8 -*-

import ctypes
import numpy as np
import matplotlib.pyplot as plt

def detrend(timeseries, radius=16384, blocksize=128):
    """
    Detrend a time series with fast CUDA routines

    1.) Compute moving average/RMS
    2.) output = ( input - moving_average)/moving_rms

    Input parameters:
        timeseries: numpy ndarray
        radius: Filter radius
        blocksize: CUDA blocksize

    Output:
        detrended: detrended timeseries
        cropped: number of items cropped at the end
    """

    # Assert that the time series is in float64 format
    assert(timeseries.dtype == np.core.numerictypes.float64)

    # Filter radius has to be a multiple of the blocksize
    assert(radius % blocksize == 0)
    #ma_lib = ctypes.cdll.LoadLibrary('/Users/ralph/source/running_ma/stencil_threaded.so')
    ma_lib = ctypes.cdll.LoadLibrary('/Users/ralph/source/running_ma/stencil_threaded_module.so')

    # Crop input of the MA filter
    len_input = timeseries.size
    cropped = len_input % blocksize
    # Crop MA input data to a multiple of the blocksize
    len_ma_input = len_input - cropped
    # output of MA time series is equal to length of len_rms_input
    # Length of the RMS input data is the length of MA output data
    len_rms_input = len_ma_input - 2 * radius
    # Length of the RMS output data is cropped by 2*blocksize
    len_rms_output = len_rms_input - 2 * radius

    #print 'Input length: %d' % (len_input)
    #print 'Input length after cropping: %d' % (len_ma_input)
    #print 'MA output length: %d' % (len_rms_input)
    #print 'RMS output length: %d' % (len_rms_output)

    # Crop timeseries to suit input of MA
    ts_cropped = timeseries[:len_ma_input]

    # Declare input and output arrays for the filter
    ma_in = np.zeros(len_ma_input, dtype='float64')
    rms_in = np.zeros(len_rms_input, dtype='float64')
    ma_out = np.zeros(len_rms_input, dtype='float64')
    rms_out = np.zeros(len_rms_output, dtype='float64')

    # Convert to ctypes
    # Length of the input data of the MA filter (output len = input length of
    # rms filter)
    c_len_ma_input = ctypes.c_int(len_ma_input)
    # Length of input and output data of the RMS filter
    # Length of the RMS input equals length of the MA output
    c_len_rms_input = ctypes.c_int(len_rms_input)
    c_len_rms_output = ctypes.c_int(len_rms_output)
    # Input data to MA filter
    c_ma_in = ctypes.c_void_p(ts_cropped.ctypes.data)
    # Output data of the MA filter
    c_ma_out = ctypes.c_void_p(ma_out.ctypes.data)
    # Output data of the RMS filter
    c_rms_out = ctypes.c_void_p(rms_out.ctypes.data)

    # Compute moving average
    result_ma = ma_lib.ma_cuda(c_ma_in, c_ma_out, c_len_rms_input)

    # Declar input data for RMS
    indata_rms = np.zeros_like(ma_out)
    indata_rms[:] = timeseries[radius:len_ma_input - radius] - ma_out
    c_rms_in = ctypes.c_void_p(indata_rms.ctypes.data)
    result_rms = ma_lib.rms_cuda(c_rms_in, c_rms_out, c_len_rms_output)

    detrended = (ts_cropped[2 * radius:-2 * radius] - ma_out[radius:-1 * radius]) / rms_out

    return detrended, cropped
 #End of file detrend.py
