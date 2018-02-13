#!/usr/bin/env python
#-*- Encoding: UTF-8 -*-

import ctypes
import numpy as np
import matplotlib.pyplot as plt

def detrend(timeseries, radius=16384, blocksize=128):
    """
    Detrend a time series with fast CUDA routines for moving average, moving rms

    1.) Compute moving average/RMS
    2.) detrended = ( input - moving_average)/moving_rms

    Input parameters:
        timeseries: ndarray, input time series
        radius:     int, Filter radius
        blocksize:  int, CUDA blocksize

    Output:
        detrended: ndarray, detrended timeseries
        cropped:   int, number of items cropped at the end

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
    # Compute moving tms
    result_rms = ma_lib.rms_cuda(c_rms_in, c_rms_out, c_len_rms_output)

    detrended = (ts_cropped[2 * radius:-2 * radius] - ma_out[radius:-1 * radius]) / rms_out

    return detrended, cropped, ma_out, rms_out


def detrend_omp(timeseries, radius=16384, module_dir="/home/rkube/source/running_ma"):
    """
    Detrend a time series using openmp moving average and moving rms:

    detrend = (input - moving_avg(input)) / moving_rms(input)


    Input:
    ======
        timeseries: ndarray, (float64!) input time series
        radius: int, Radius for moving average and moving rms


    Output:
    =======
        norm: ndarray, normalized time series
    """

    # Assert that the time series is float64
    assert(timeseries.dtype == np.core.numerictypes.float64)

    # Load the moving stat library
    #ma_lib = ctypes.cdll.LoadLibrary("/Users/ralph/source/running_ma/module_moving_stat.so")
    ma_lib = ctypes.cdll.LoadLibrary("%s/module_moving_stat.so" % module_dir)

    numel = timeseries.size
    norm_out = np.ones(numel, dtype='float64')
    moving_avg = np.ones(numel, dtype="float64")
    moving_rms = np.ones(numel, dtype="float64")

    # Convert to ctypes
    c_numel = ctypes.c_size_t(numel)
    c_radius= ctypes.c_size_t(radius)

    # Input time series to moving average
    c_norm_in_ptr = ctypes.c_void_p(timeseries.ctypes.data)
    # Normalized time series
    c_norm_out_ptr = ctypes.c_void_p(norm_out.ctypes.data)
    # Moving average
    c_moving_avg_out_ptr = ctypes.c_void_p(moving_avg.ctypes.data)
    # Moving rms
    c_moving_rms_out_ptr = ctypes.c_void_p(moving_rms.ctypes.data)

    ma_lib.running_norm_omp(c_norm_in_ptr, c_norm_out_ptr,
                            c_moving_avg_out_ptr,
                            c_moving_rms_out_ptr, c_radius, c_numel)

    return norm_out, moving_avg, moving_rms


def running_avg(timeseries, radius=16384, module_dir="/home/rkube/source/running_ma"):
    """
    Compute the moving average of a time series using the openmp module


    Input:
    ======
         timeeseries: ndarray(float64): input time series
         radius: int, Radius for moving average filter

    Output:
    ======
        mavg: ndarray, moving average
    """

    assert(timeseries.dtype == np.core.numerictypes.float64)

    ma_lib = ctypes.cdll.LoadLibrary("%s/module_moving_stat.so" % module_dir)

    numel = timeseries.shape[0]
    moving_avg = np.zeros(numel, dtype='float64')
    
    # convert to ctypes
    c_numel = ctypes.c_size_t(numel)
    c_radius = ctypes.c_size_t(radius)

    c_ma_in_ptr = ctypes.c_void_p(timeseries.ctypes.data)
    c_ma_out_ptr = ctypes.c_void_p(moving_avg.ctypes.data)

    ma_lib.moving_average_omp(c_ma_in_ptr, c_ma_out_ptr, c_radius, c_numel)

    return moving_avg

#End of file detrend.py
