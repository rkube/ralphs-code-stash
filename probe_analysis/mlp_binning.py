#!/opt/local/bin/python
# -*- Encoding: UTF-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import skew, kurtosis



def binning_moments_sweep(probe_signal, tb_signal, probe_rho, tb_rho, rho_min, rho_max, delta_rho, interp=True):
    """
    Compute statistical moments of probe_signal on intervals, where probe is in
    rho in [rho_min + n * delta_rho : rho_min + (n + 1) * delta_rho]

    Input:
    ======   
        probe_signal:     ndarray, Timeseries to analyze (ne, Isat, Vp, Vf, Te)
        tb_signal:        ndarray, Timebase of signal
        probe_rho:        ndarray, Rho position of probe
        tb_rho:           ndarray, Timebase of rho position
        rho_min:          float, minimum plunge depth
        rho_max:          float, maximum plunge depth
        delta_rho:        float, spatial bin size
        interp:           bool, interpolate rho on fast signal timebase

    Output:
    =======
        rho_bin_arr:    ndarray, rho value in the middle of the interval on which 
                                 statistics have been computer. 
        mean_arr:       ndarray, mean of the series where probe is within [rho - 0.5 delta_rho:rho + 0.5 delta_rho] 
        rms_arr:        ndarray, rms of the time sereis where probe is ...
        skew_arr:       ndarray, skewness...
        flat_arr:       ndarray, flatness...
        nelem_arr:      ndarray, number of elements in probe_signal 
        hist_list:      ndarray, histogram and edges of probe_signal 
    """

    rho_bin_arr = np.arange(rho_min, rho_max, delta_rho)
    # Interpolate rho on fast timebase
    if(interp):
        rho_ip_fast = interp1d(tb_rho, probe_rho)
        rho_fast = rho_ip_fast(tb_signal)
    else:
        rho_fast = probe_rho

    nelem_profiles = rho_bin_arr.shape[0] - 1
    mean_arr = np.zeros(nelem_profiles, dtype='float64')
    rms_arr = np.zeros(nelem_profiles, dtype='float64')
    skew_arr = np.zeros(nelem_profiles, dtype='float64')
    flat_arr = np.zeros(nelem_profiles, dtype='float64')
    nelem_arr = np.zeros(nelem_profiles, dtype='int')

    hist_list = []
    tidx_list = []
    # Compute statistics in the interval [rho:rho + delta]
    for rho_idx in np.arange(rho_bin_arr.size - 1):
        good_tidx = ((rho_fast > rho_bin_arr[rho_idx]) & (rho_fast < rho_bin_arr[rho_idx + 1]))
        tidx_list.append(good_tidx)
        
        if (good_tidx.sum() > 0):
            signal_cut = probe_signal[good_tidx]
            mean_arr[rho_idx] = signal_cut.mean()
            rms_arr[rho_idx] = signal_cut.std(ddof=1)
            skew_arr[rho_idx] = skew(signal_cut)
            flat_arr[rho_idx] = kurtosis(signal_cut)
            nelem_arr[rho_idx] = int(good_tidx.sum())

            nbins = int(np.sqrt(good_tidx.sum()))
            res = np.histogram(probe_signal[good_tidx], bins=nbins, density=True)
            hist_list.append(res)

        else:
            mean_arr[rho_idx] = np.NaN
            rms_arr[rho_idx] = np.NaN 
            skew_arr[rho_idx] = np.NaN
            flat_arr[rho_idx] = np.NaN
            nelem_arr[rho_idx] = 0
            hist_list.append(None)

    # rho <- rho + 0.5 * delta. This is now the mid-point of the interval we have
    # computed the statistics on
    rho_bin_arr = rho_bin_arr[:-1] + 0.5 * delta_rho
    print rho_bin_arr.size, mean_arr.size
    return rho_bin_arr, mean_arr, rms_arr, skew_arr, flat_arr, nelem_arr, hist_list, tidx_list


def binning_moments_sweep_fixnelem(probe_signal, tb_signal, probe_rho, tb_rho, nelem=2e4, nbins=150):
    """
    Compute statistical moments of time series, sampled by a plunging probe on 
    sub samples of the time series, on equal number of elements

    Input:
    ======
        probe_signal:     ndarray, Timeseries to analyze (ne, Isat, Vp, Vf, Te)
        tb_signal:        ndarray, Timebase of signal
        probe_rho:        ndarray, Rho position of probe
        tb_rho:           ndarray, Timebase of rho position
        nelem:            integer, Number of points to use for each bin

    Output:
    =======
        rho_bin_arr:    ndarray, average rho value of the interval on which statistics have been computed.
        mean_arr:       ndarray, mean of time series, per nelem bin
        rms_arr:        ndarray, rms of the time sereis where probe is ...
        skew_arr:       ndarray, skewness...
        flat_arr:       ndarray, flatness...
        nelem_arr:      ndarray, number of elements in probe_signal 
        hist_list:      ndarray, histogram and edges of probe_signal 
    """

    nbins = int(probe_signal.size / nelem)
    rho_bin_arr = np.zeros(nbins, dtype='float')
    mean_arr = np.zeros_like(rho_bin_arr)
    rms_arr = np.zeros_like(rho_bin_arr)
    skew_arr = np.zeros_like(rho_bin_arr)
    flat_arr = np.zeros_like(rho_bin_arr)
    nelem_arr = np.ones_like(rho_bin_arr) * nelem
    hist_list = []
    tidx_list = []

    for bin_idx in np.arange(nbins):
        signal_cut = probe_signal[bin_idx * nelem:(bin_idx + 1) * nelem] 
        if (signal_cut.size < 1e2):
            # Do not compute statistics with less than 100 elemnts
            rho_bin_arr[bin_idx] = np.nan
            mean_arr[bin_idx] = np.nan
            rms_arr[bin_idx] = np.nan
            skew_arr[bin_idx] = np.nan
            flat_arr[bin_idx] = np.nan
            hist_list.append([])
            tidx_list.append([])

        else:
            rho_bin_arr[bin_idx] = probe_rho[bin_idx * nelem:(bin_idx + 1) * nelem].mean()
            mean_arr[bin_idx] = signal_cut.mean()
            rms_arr[bin_idx] = signal_cut.std(ddof=1)
            skew_arr[bin_idx] = skew(signal_cut)
            flat_arr[bin_idx] = kurtosis(signal_cut)

            res = np.histogram(signal_cut, bins=nbins, density=True)
            hist_list.append(res)

            tidx_list.append(np.arange(bin_idx * nbins, (bin_idx + 1) * nbins, 1))

    # Make rho_bin_arr the mid points of the interval on which we compute the statistics
    rho_bin_arr = rho_bin_arr[:-1] + 0.5 * delta_rho
    return rho_bin_arr, mean_arr, rms_arr, skew_arr, flat_arr, nelem_arr, hist_list, tidx_list


def binning_hist_sweep(probe_signal, tb_signal, probe_rho, tb_rho, rho_min, rho_max, delta_rho, nbins, norm_ts=False):
    """
    Compute histogram of probe_signal on intervals, where probe is in
    rho in [rho_min + n * delta_rho : rho_min + (n + 1) * delta_rho]

    Input:
    =====   
        probe_signal:     ndarray, Timeseries to analyze (ne, Isat, Vp, Vf, Te)
        tb_signal:        ndarray, Timebase of signal
        probe_rho:        ndarray, Rho position of probe
        tb_rho:           ndarray, Timebase of rho position
        rho_min:          float, minimum plunge depth
        rho_max:          float, maximum plunge depth
        delta_rho:        float, spatial bin size
        nbins:            int, number of bins for histograms
        norm_ts:          bool, True: normalize time series to I' = (I - <I>) / I_rms, 
                                False: Do nothing

    Returns:
    ========
        hist_list:      list, list of histograms of signal PDF where probe is in the
                              respective bin of rho.
        nelem_list:     ndarray, number of elements used to compute the histogram at a given rho position
    """

    if norm_ts:
        probe_signal = (probe_signal - probe_signal.mean()) / probe_signal.std(ddof=1)

    rho_bin_arr = np.arange(rho_min, rho_max, delta_rho)

    # Interpolate rho on fast timebase
    rho_ip_fast = interp1d(tb_rho, probe_rho)
    rho_fast = rho_ip_fast(tb_signal)

    hist_list = []
    hist_mid_list = []
    nelem_list = np.zeros_like(rho_bin_arr)
    for rho_idx in np.arange(rho_bin_arr.size - 1):
        good_tidx = ((rho_fast > rho_bin_arr[rho_idx]) & (rho_fast < rho_bin_arr[rho_idx + 1]))

        #if rho_idx is 0:
        print 'delta_rho = %e -> %d elements' % (delta_rho, good_tidx.sum())
        nelem_list[rho_idx] = good_tidx.sum()
        signal_cut = probe_signal[good_tidx]

        x_hist = np.arange(signal_cut.min(), signal_cut.max(), 
                (signal_cut.max() - signal_cut.min()) / float(nbins))

        mid_hist = x_hist[:-1] + 0.5 * np.diff(x_hist)
        res = np.histogram(signal_cut, x_hist, density=True)
        hist_list.append(res[0])
        hist_mid_list.append(mid_hist)

    rho_bin_arr = rho_bin_arr[:-1]
    plt.show()
    return rho_bin_arr, hist_list, hist_mid_list, nelem_list



def binning_tidx(tb_signal, probe_rho, tb_rho, rho_min = 0.0, rho_max = 2e-2, delta_rho = 2e-3, ip_timebase=True):
    """
    Find the time indices, where the probe is in the interval rho \in [rho_min + n * delta_rho : rho_min + (n + 1) * delta_rho]

    Return a list with the indices of tb_signal, which correspon to the index, where
    probe is in rho_range

    Input:
    ======
        tb_signal:    ndarray, Fast time signal for which we want time indices corresponding to a rho
        probe_rho:    ndarray, Rho position of the probe
        tb_rho:       ndarray, Time base of probe's rho position
        rho_min:      ndarray, minimum value of probe_rho
        rho_max:      ndarray, maximum value of probe_rho
        delta_rho:    ndarray, used for spacing in rho array
        ip_timebase:  bool, if True, interpolate the probe's rho waveform onto the timebase of the signal

    Output:
    =======
        rho_tidx_list list of ndarrays, Indices of tb_signal, corresponding to where probe is in a rho intervall
    """

    rho_range = np.arange(rho_min, rho_max, delta_rho)
    print rho_range
    if ip_timebase:
        rho_ip_fast = interp1d(tb_rho, probe_rho, bounds_error=False, fill_value=1.0)
        rho_fast = rho_ip_fast(tb_signal)
    else:
        rho_fast = probe_rho

    tb_rho_idx = []
    for rho_idx, rho in enumerate(rho_range[:-1]):
        good_tidx = ((rho_fast > rho_range[rho_idx]) & (rho_fast < rho_range[rho_idx + 1]))
        # print '%d elements in [%e:%e]' % (good_tidx.sum(), rho_range[rho_idx], rho_range[rho_idx + 1])
        tb_rho_idx.append(np.squeeze(good_tidx))

    return rho_range, tb_rho_idx, rho_fast

# End of file mlp_binning.py
