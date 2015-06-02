#!/usr/bin/python
# -*- Encoding: UTF-8 -*-


import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from misc.correlate import correlate, fwhm_estimate
from misc.peak_detection import detect_peaks_1d
from scipy.interpolate import interp1d
from scipy.stats.mstats import skew as mskew, kurtosis as mkurtosis
from scipy.optimize import leastsq


# Linear function for detrending and error function
linfun = lambda p, x : p[0]*x+p[1]
errfun = lambda p, x, y :  y - linfun(p, x)
# Returs the detrended signal
signal_dt = lambda signal_raw, p, time : signal_raw - linfun(p, time)


def binning_moments_sweep(probe_signal, probe_vsweep, probe_rho, timebase, interval_size = 400, show_plots = False):
    """
    Compute statistical moments from parts of the time series where we assume the probe to be in
    Isat mode. Use an interval of numpoints points around each minima of a voltage sweep.
    Input:
        probe_signal:       Timeseries of the probe (Current)
        probe_vsweep:       Sweeping voltage of the probe
        timebase:           Timebase of the time series
        interval_size:      Size of the intervals for which to compute moments
        show_plots:         Show debugging ploots
    Output:
    """

    # Detect points where vsweep is minimal and the probe is in isat. Use -probe_vsweep because the routine detects maxima.
    sweep_peak_list = detect_peaks_1d((-1.0 * probe_vsweep)/(-1.0 * probe_vsweep).max(), timebase, 100, 0.97, peak_width=10)

    n_sweeps = np.size(sweep_peak_list) 
    i2 = int(round(interval_size/2))
    mean = np.zeros(n_sweeps)
    fluc = np.zeros(n_sweeps)
    skewness = np.zeros(n_sweeps)
    kurtosis = np.zeros(n_sweeps)
    rho = np.zeros(n_sweeps)
    t_mom = timebase[sweep_peak_list]
    
    # Iterate over the minima for the sweeps, compute moments
    if show_plots:
        fig1 = plt.figure()
        fig2 = plt.figure()
        ax_max = fig1.add_subplot(111)
        ax_ts = fig2.add_subplot(111) 

    for peak_idx, peak in enumerate(sweep_peak_list):
        if show_plots:
            ax_ts.plot( timebase[peak-i2:peak+i2], probe_signal[peak-i2:peak+i2], 'k')

        mean[peak_idx] = probe_signal[peak-i2:peak+i2].mean()
        fluc[peak_idx] = probe_signal[peak-i2:peak+i2].std() / mean[peak_idx]
        skewness[peak_idx] = mskew(probe_signal[peak-i2:peak+i2])
        kurtosis[peak_idx] = mkurtosis(probe_signal[peak-i2:peak+i2])
        rho[peak_idx] = probe_rho[peak-i2:peak+i2].mean()

    if show_plots:
        ax_max.plot(timebase, probe_vsweep)
        ax_max.plot(timebase[sweep_peak_list], probe_vsweep[sweep_peak_list], 'ko')
        plt.show()


    return mean, fluc, skewness, kurtosis, rho, t_mom

def binning_moments_2( probe_signal, rho_signal, probe_time, t_intervals, binned_list, te_signal = False, mode = 'I', probe_A = 9.662781e-07, show_plots = False, return_numel = False):
    """ 
    Compute statistical moments of the probe signal, given some binning

    Parameters:
        probe_signal:
        rho_signal:
        probe_time:
        t_intervals:
        binned_list:

    """

    num_reciprocations = len(t_intervals)/2
    n_zbins = np.max( [len(bin) for bin in binned_list] )

    # Ion mass:
    mi = 1.67e-27
    q = 1.602e-19    

    mean = np.zeros( [num_reciprocations*2 + 1, n_zbins] ) 
    rho = np.zeros( [num_reciprocations*2 + 1, n_zbins] ) 
    std = np.zeros( [num_reciprocations*2 + 1, n_zbins] ) 
    skewness = np.zeros( [num_reciprocations*2 + 1, n_zbins] ) 
    kurt = np.zeros( [num_reciprocations*2 + 1, n_zbins] ) 
    fluc = np.zeros( [num_reciprocations*2 + 1, n_zbins] ) 
    if return_numel:    
        numel = np.zeros( [num_reciprocations*2 + 1, n_zbins] ) 


    # Compute the moments of the probe signal for each partial reciprocation of the probe
    for idx, interval, bin_list in zip( np.arange(len(t_intervals)), t_intervals, binned_list ):
        t_int = probe_time[ interval[0] : interval[-1] ]
        s_int = probe_signal[ interval[0] : interval[-1] ]
        rho_int = rho_signal[ interval[0] : interval[-1] ]
        try:
            te_int = te_signal[ interval[0] : interval[-1] ]
        except:
            print 'binning_moments_2: No electron temperature available'

        # Convert to density if requested:
        if ( mode == 'n' ):
            te_int = te_signal[ interval[0] : interval[-1] ]
            #s_int = s_int * np.sqrt(mi) / ( np.sqrt(2 * te_int * q) * q * probe_A )         
            # Hutchinson, p.64:
            s_int = s_int * np.sqrt(mi / (q*te_int) ) / ( 0.607 * probe_A * q)

        if show_plots:
            plt.figure()
            plt.subplot(211)
            plt.plot( t_int, rho_signal[ interval[0] : interval[-1] ] )
            plt.subplot(212)
            for idx2, bl in enumerate(bin_list):
                plt.plot(t_int[bl], s_int[bl]  )

        #for blist in bin_list:
        #    rho_f = rho_int[blist].mean()
        #    try:
        #        numel = np.size(s_int[blist].data) - np.sum(s_int[blist].mask)
        #    except: 
        #        numel = np.size(s_int[blist].data)
        #    print 'Bin around %f with %d items' % ( rho_f, numel)
        if return_numel: 
            try:
                numel[idx+1, :len(bin_list)] = np.array([np.size(s_int[bl].data) - np.sum(s_int[bl].mask) for bl in bin_list])
            except AttributeError:
                numel[idx+1, :len(bin_list)] = np.array([np.size(s_int[bl].data) for bl in bin_list])
 
        mean[idx + 1, :len(bin_list)] = np.array( [s_int[bl].mean() for bl in bin_list] )
        std[idx + 1, :len(bin_list)] = np.array( [s_int[bl].std() for bl in bin_list] )
        # std computes the RMS (see scipy documentation). So this is the relative fluctuations
        #fluc[idx + 1, :len(bin_list)] = std[idx + 1, :len(bin_list)] / mean[idx + 1, :len(bin_list)]
        fluc[idx + 1, :len(bin_list)] = std[idx + 1, :len(bin_list)] / mean[idx + 1, :len(bin_list)]
        skewness[idx + 1, :len(bin_list)] = np.array( [ mskew( s_int[bl] ) for bl in bin_list ] )
        kurt[idx + 1, :len(bin_list)] = np.array( [ mkurtosis( s_int[bl], fisher=False ) for bl in bin_list ] )
        rho[idx + 1, :len(bin_list)] = np.array( [rho_int[bl].mean() for bl in bin_list] )

    if show_plots:
        plt.show()

    # Convert to masked arrays. Tested with ASP only
    mean = ma.MaskedArray( mean, mask = np.isnan(mean) )
    std = ma.MaskedArray( std, mask = np.isnan(std) )
    fluc = ma.MaskedArray( fluc, mask = np.isnan(fluc) )
    skewness = ma.MaskedArray( skewness, mask = np.isnan(skewness) )
    kurt = ma.MaskedArray( kurt, mask = np.isnan(kurt) )
    rho = ma.MaskedArray( rho, mask = np.isnan(rho) )

    mean[0,:] = mean[1:,:].mean( axis=0 )
    std[0,:] = std[1:,:].mean( axis=0 )
    fluc[0,:] = fluc[1:,:].mean( axis=0 )
    skewness[0,:] = skewness[1:,:].mean( axis=0 )
    kurt[0,:] = kurt[1:,:].mean( axis=0 )
    rho[0,:] = rho[1:,:].mean( axis=0 )

    if show_plots:
        plt.show()
    if return_numel:
        return mean, std, fluc, skewness, kurt, rho, numel
    return mean, std, fluc, skewness, kurt, rho 



def binning_time_vfloat( probe_vertical, probe_voltage, z_bins, time_signal, t_maxrc, t_down = 0.075, t_up = 0.05, delta_z = 1e-3, epsilon = 1e-6, vmin = 0.85):
    """
    Given a list of z positions for the probe, return two lists of time indices:
    t_intervals:        Time indices for each in/out reciprocration of the probe
    binned_list:        List of time indices for each reciprocation where the probe is in v_float, i.e. 
                        v_probe < v_min * probe_voltage.min()

    Use with FSP only!

    Parameters:
    probe_vertical:     Vertical position of the probe
    probe_voltage:      Potential signal of the probe
    z_bins:             List of probe positions for which we want the time intervals
    time_signal:        Time series of the probe
    t_down:             Time it takes the probe to move from max. reciprocation to baseline
    t_up:               Time it takes the probe to move from baseline to max. reciprocation
    delta_z:            Width of spatial interval
    epsilon:            A very small number
    vmin:               Multiplikator to determine where the probe is in floating mode
    """
   
    print 'Function binning_time_vfloat' 
    delta_t = time_signal[1] - time_signal[0]
    n_zbins = np.shape(z_bins)[0]
    vfloat = vmin * probe_voltage.min()

    print z_bins - delta_z, z_bins + delta_z

    # The time where the probe has maximum reciprocation in the plasma deviates
    # from the programmed maxima. Find these times exactly.

    # Find the time indices where the maximum probe reciprocation is programmed
    t_idx_maxrc = [ np.argwhere ( np.abs( time_signal - t ) < epsilon )[0][0] for t in t_maxrc ]

    # Now find the local maxima around these indices
    probe_vertical_max = [ probe_vertical[ idx - 50000 : idx + 50000 ].argmax() for idx in t_idx_maxrc ]
    # Compute the times when maximum reciprocation occurs
    t_maxrc = [ time_signal[ fvrt_max + idx - 50000 ] for fvrt_max, idx in zip( probe_vertical_max, t_idx_maxrc) ]

    # Now split the time array in intervals where the probe is reciprocating in and
    # out of the plasma

    # Find the indices for when the probe is reciprocating into and out of the plasma
    t_intervals_full = [ np.squeeze( np.argwhere((time_signal > (t - t_down)) & (time_signal < t )) ) for t in t_maxrc] \
        + [ np.squeeze( np.argwhere((time_signal > t ) & ( time_signal < t + t_up )) ) for t in t_maxrc]
    
    t_intervals = [ (t_int[0], t_int[-1]) for t_int in t_intervals_full ]
    t_intervals_full = []

    binned_list = []

    # Cut each probe trajectory (up, down) in sub-intervals and filter by magnitude of V
    for idx, interval in enumerate( t_intervals ):
        plt.figure()
        probe_vertical_interval = probe_vertical[ interval[0] : interval[-1] ]
        vfloat_interval = probe_voltage[ interval[0] : interval[-1] ]
        time_interval = time_signal[ interval[0] : interval[-1] ]

        plt.subplot(211)

        plt.plot( time_interval, probe_vertical_interval )
#        plt.plot( time_interval, vfloat_interval )

        # Find time indices corresponding to each interval [z - delta_z : z + delta_z]
        t_bins_full = [np.squeeze(np.argwhere( np.squeeze(( probe_vertical_interval > upper) & ( probe_vertical_interval < lower)  )) )for upper, lower in zip( z_bins-delta_z, z_bins+delta_z ) ]

        # Remove empty bins where the probe does not reciprocate into
        while( np.size(t_bins_full[-1]) < 1 ):
            t_bins_full.pop()
            z_bins = z_bins[:-1]

        # Apply the same binning to the probe voltage
        v_bins = [ vfloat_interval[ tbin[0] : tbin[-1] ] for tbin in t_bins_full ] 
        # Keep only the time indices where v < v_float
        v_bins_float = [ np.squeeze(np.argwhere(vbin < vfloat) + tbin[0]) for vbin, tbin in zip( v_bins, t_bins_full ) ]
        
        print v_bins_float[0][0], v_bins_float[0][-1]

        plt.subplot(212)
        for idx, bin in enumerate(v_bins_float):
            if len(bin) == 0:
                continue
            print idx, bin
            plt.plot( time_signal[ bin[0] : bin[-1] ], probe_vertical[ bin[0] : bin[-1] ], 'k' )
#            plt.plot( time_signal[ bin[0] : bin[-1] ], probe_voltage[ bin[0] : bin[-1] ] )
        binned_list.append( v_bins_float )

        plt.show()

    return  t_intervals, binned_list


def binning_time_asp(probe_position, r_bins, timebase, t_maxrc_programmed, t_down=0.05, t_up=0.05, delta_r=1e-3, epsilon=1e-6, min_rc_delta=50000, show_plots=False):
    """
    Given a list of R or rho positions for the probe return. 
    Use this routine of the probe's coordinate is minimal when fully plunged into the plasma. I.e. R or rho.
    t_intervals:    Time indices for each in/out reciprocation of the probe

    Parameters:
    probe_position:     Horizontal (r) position of the probe
    r_bins:             List of probe positions for which the time intervals are determined
    timebase:           Timebase of the probe
    t_maxrc_programmed: Time when maximum probe reciprocation is programmed
    t_down:             Time it takes the probe to move from max. reciprocation to baseline
    t_up:               Time it takes the probe to move from baseline to max. reciprocation
    delta_r:            Width of spatial interval
    epsilon:            A very small number
    min_rc_delta:       Interval in which to look for time of maximal probe reciprocation

    Output:
    t_recip_intervals: Time of the individual probe reciprocations
    binned_list:       List. Index0 gives the in/out reciproctaion, i.e. 0/6, 
                             Index1 gives the indices in timebase, for which the probe is in a given r_bin
    """
    delta_t = timebase[1] - timebase[0]
    n_rbins = np.shape(r_bins)[0]

    # The time where the probe has maximum reciprocation in the plasma deviates
    # from the programmed maxima. Find these times exactly.

    # Find the time indices where the maximum probe reciprocation is programmed
    t_idx_maxrc = [np.argwhere(np.abs(timebase - t) < epsilon)[0][0] for t in t_maxrc_programmed]
    # Now find the local minima around these indices as the probe reciprocates inwards
    probe_position_min = [probe_position[idx - min_rc_delta:idx + min_rc_delta].argmin() for idx in t_idx_maxrc]

    if show_plots:
        plt.figure()
        for idx in t_idx_maxrc:
            plt.plot(timebase[idx - min_rc_delta:idx + min_rc_delta], probe_position[idx - min_rc_delta:idx + min_rc_delta])
        plt.xlabel('time / s')
        plt.ylabel('Probe plunge / m')
        plt.show()

    # Compute the times when maximum reciprocation occurs
    t_maxrc = [ timebase[ fvrt_min + idx - min_rc_delta ] for fvrt_min, idx in zip( probe_position_min, t_idx_maxrc) ]

    # Now split the time array in intervals where the probe is reciprocating in and
    # out of the plasma

    # Find the indices for when the probe is reciprocating into and out of the plasma
    t_recip_intervals_full = [ np.squeeze( np.argwhere((timebase > (t - t_down)) & (timebase < t )) ) for t in t_maxrc] \
        + [ np.squeeze( np.argwhere((timebase > t ) & ( timebase < t + t_up )) ) for t in t_maxrc]
    
    t_recip_intervals = [ (t_int[0], t_int[-1]) for t_int in t_recip_intervals_full ]
    t_recip_intervals_full = []

    binned_list = []

    if show_plots:
        plt.figure()
        plt.title('Your intervals, sire!')
        for interval in t_recip_intervals:
            pv = probe_position[ interval[0] : interval[1] ]
            ti = timebase[ interval[0] : interval[1] ]
            plt.plot(ti, pv)
        plt.plot( t_maxrc, -0.02*np.ones_like(t_maxrc), 'ko')
        plt.show()

    # Cut each probe trajectory (up, down) in sub-intervals 
    for idx, interval in enumerate(t_recip_intervals):
        print idx, interval
        probe_position_interval = probe_position[interval[0] : interval[-1]]
        time_interval = timebase[interval[0] : interval[-1]]

        # Find time indices corresponding to each interval [z - delta_z : z + delta_z]
        t_bins_full = [np.squeeze(np.argwhere(np.squeeze((probe_position_interval > upper) & 
                                                         (probe_position_interval < lower)))) 
                                                         for upper, lower in zip(r_bins-delta_r, r_bins+delta_r)]
        # print len(r_bins)
        # print [(upper, lower) for upper, lower in zip(r_bins - delta_r, r_bins + delta_r)]
        # print t_bins_full

        # Remove empty bins where the probe does not reciprocate into
        while(t_bins_full[-1].size < 1):
            t_bins_full.pop()
        #    r_bins = r_bins[:-1]

        if show_plots:
            plt.figure()
            for idx, bin in enumerate(t_bins_full):
                if len(bin) == 0:
                    continue
                #print idx, bin
                plt.plot( time_interval[ bin[0] : bin[-1] ], probe_position_interval[ bin[0] : bin[-1] ] )
 
        binned_list.append( t_bins_full )
        plt.show()

    return  t_recip_intervals, binned_list


def binning_time_fsp(probe_position, z_bins, timebase, t_maxrc_programmed, t_down = 0.05, t_up = 0.05, delta_z = 1e-3, epsilon = 1e-6, max_rc_delta = 50000, show_plots = False):
    """
    Use this routine of the probe's coordinate is minimal when fully plunged into the plasma. I.e. R or rho.
    t_intervals:    Time indices for each in/out reciprocation of the probe

    Parameters:
    probe_position:     Horizontal (r) position of the probe
    z_bins:             List of probe positions for which the time intervals are determined
    timebase:        Timebase of the probe
    t_maxrc_programmed: Time when maximum probe reciprocation is programmed
    t_down:             Time it takes the probe to move from max. reciprocation to baseline
    t_up:               Time it takes the probe to move from baseline to max. reciprocation
    delta_z:            Width of spatial interval
    epsilon:            A very small number
    max_rc_delta:       Interval in which to look for time of maximal probe reciprocation

    Output:
    t_recip_intervals: Time of the individual probe reciprocations
    binned_list:       List. Index0 gives the in/out reciproctaion, i.e. 0/6, 
                             Index1 gives the indices in timebase, for which the probe is in a given z_bin

    """
    delta_t = timebase[1] - timebase[0]
    n_zbins = np.shape(z_bins)[0]

    # The time where the probe has maximum reciprocation in the plasma deviates
    # from the programmed maxima. Find these times exactly.

    # Find the time indices where the maximum probe reciprocation is programmed
    t_idx_maxrc = [ np.argwhere ( np.abs( timebase - t ) < epsilon )[0][0] for t in t_maxrc_programmed ]

#    print 'time indices where probe has programmed max rec.:', t_idx_maxrc
#    print [ np.shape(probe_position[idx - max_rc_delta : idx + max_rc_delta]) for idx in t_idx_maxrc ]

    # Now find the local maxima around these indices as the probe reciprocates inwards
    probe_position_max = [ probe_position[ idx - max_rc_delta : idx + max_rc_delta ].argmax() for idx in t_idx_maxrc ]

    if show_plots:
        plt.figure()
        for idx in t_idx_maxrc:
            plt.plot( timebase[ idx - max_rc_delta : idx + max_rc_delta], probe_position[ idx - max_rc_delta : idx + max_rc_delta ] )
        plt.xlabel('time / s')
        plt.ylabel('Probe plunge / m')
        plt.show()

    # Compute the times when maximum reciprocation occurs
    t_maxrc = [ timebase[ fvrt_min + idx - max_rc_delta ] for fvrt_min, idx in zip( probe_position_max, t_idx_maxrc) ]

    # Now split the time array in intervals where the probe is reciprocating in and
    # out of the plasma

    # Find the indices for when the probe is reciprocating into and out of the plasma
    t_recip_intervals_full = [ np.squeeze( np.argwhere((timebase > (t - t_down)) & (timebase < t )) ) for t in t_maxrc] \
        + [ np.squeeze( np.argwhere((timebase > t ) & ( timebase < t + t_up )) ) for t in t_maxrc]
    
    t_recip_intervals = [ (t_int[0], t_int[-1]) for t_int in t_recip_intervals_full ]
    t_recip_intervals_full = []

    binned_list = []

    if show_plots:
        plt.figure()
        plt.title('Your intervals, sire!')
        for interval in t_recip_intervals:
            pv = probe_position[ interval[0] : interval[1] ]
            ti = timebase[ interval[0] : interval[1] ]
            plt.plot(ti, pv)
        plt.plot( t_maxrc, -0.02*np.ones_like(t_maxrc), 'ko')
        plt.show()

    # Cut each probe trajectory (up, down) in sub-intervals 
    for idx, interval in enumerate( t_recip_intervals ):
        probe_position_interval = probe_position[ interval[0] : interval[-1] ]
        time_interval = timebase[interval[0] : interval[-1]]

        # Find time indices corresponding to each interval [z - delta_z : z + delta_z]
        t_bins_full = [np.squeeze(np.argwhere( np.squeeze((probe_position_interval > upper) & ( probe_position_interval < lower))))for upper, lower in zip(z_bins-delta_z, z_bins+delta_z)]
        #print len(z_bins)
        #print [ (upper, lower) for upper, lower in zip( z_bins - delta_z, z_bins + delta_z ) ]
        #print t_bins_full

        # Remove empty bins where the probe does not reciprocate into
        while(np.size(t_bins_full[-1]) < 1):
            t_bins_full.pop()
        #    z_bins = z_bins[:-1]

        if show_plots:
            plt.figure()
            for idx, bin in enumerate(t_bins_full):
                if len(bin) == 0:
                    continue
                #print idx, bin
                plt.plot(time_interval[bin[0] : bin[-1]], probe_position_interval[bin[0] : bin[-1]])
 
        binned_list.append(t_bins_full)
        plt.show()

    return  t_recip_intervals, binned_list

