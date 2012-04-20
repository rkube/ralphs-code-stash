#!/usr/bin/python
# -*- Encoding: UTF-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from correlate import correlate, fwhm_estimate
import MDSplus as mds
from scipy.interpolate import interp1d
from scipy.stats import skew, kurtosis
from scipy.optimize import leastsq


"""
Plot and analyze radial profiles from reciprocating scanning probes
"""

# Linear function for detrending and error function
linfun = lambda p, x : p[0]*x+p[1]
errfun = lambda p, x, y :  y - linfun(p, x)
# Returs the detrended signal
signal_dt = lambda signal_raw, p, time : signal_raw - linfun(p, time)


def binning_moments( probe_signal, probe_vertical, z_bins, time_signal, t_maxrc, probe_rho = None, ne = None, \
    t_low = 0.075, t_up = 0.05, t_overlap_idx = 15000, delta_z = 3e-3, tau_max = 1e-4, num_reciprocations = 3, epsilon = 1e-6 ):
    """
    Compute statistical moments for ``signal`` for the part of the timeseries
    when the probe position is in the interval defined by z_bins

    probe_signal:       Timeseries of the probe signal
    probe_vertical:     Timeseries of vertical probe position
    z_bins:             List of probe position on which the moments are to be computed
    time_signal:        Time signal for probe data
    t_maxrc:            Times where the maximum reciprocation is programmed to happen
    probe_rho:          Probe position mapped to rho at outboard-midplane
    ne:                 Particle density
    t_top:              times at which the probe is programmed to have max. reciprocation
    t_low = 0.075       Length of time interval before maximum probe reciprocation
    t_up = 0.05         Length of time interval after maximum probe reciprocation
    t_overlap_idx = 1e+4    Interval after maximum we include in each time interval, in index points
    delta_z = 3e-3      Width of a reciprocation bin
    tau_max             Maximum correlation time
    num_reciprocations  Number of probe reciprocations. Usually 3
    """


    delta_t = time_signal[1] - time_signal[0]

    n_zbins = np.shape(z_bins)[0]

    # List for linear fits used for detrending the signals
    lf_list = []

    # Allocate memory for the return values
    ne_mean  = np.zeros( [num_reciprocations*2 + 1 ] )
    mean = np.zeros( [num_reciprocations*2 + 1, n_zbins] ) 
    rho = np.zeros( [num_reciprocations*2 + 1, n_zbins] ) 
    std = np.zeros( [num_reciprocations*2 + 1, n_zbins] ) 
    skewness = np.zeros( [num_reciprocations*2 + 1, n_zbins] ) 
    kurt = np.zeros( [num_reciprocations*2 + 1, n_zbins] ) 
    fluc = np.zeros( [num_reciprocations*2 + 1, n_zbins] ) 
    acor = np.zeros( [num_reciprocations*2 + 1, n_zbins] )

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
    t_intervals_full = [ np.squeeze( np.argwhere((time_signal > (t - t_low)) & (time_signal < t )) ) for t in t_maxrc] \
        + [ np.squeeze( np.argwhere((time_signal > t ) & ( time_signal < t + t_up )) ) for t in t_maxrc]
    
    t_intervals = [ (t_int[0], t_int[-1]) for t_int in t_intervals_full ]
    t_intervals_full = []

    # For each half reciprocation of the probe, compute now the statistics of the
    # probe signal for the subintervals in z_bins 

    for interval, idx in zip( t_intervals, np.arange( len(t_intervals) ) ):
    
        print 'Interval %d/%d' % ( idx, len(t_intervals) )
   
        # Cut out the interval we are working 
        probe_vertical_interval = probe_vertical[ interval[0] : interval[-1] ]
        time_interval = time_signal[ interval[0] : interval[-1] ]
        # And store the index offset accompanied by cutting out the signal
        abs_offset = interval[0]

        # Create bins where the probe is in a given spatial position
        t_bins = [np.squeeze(np.argwhere( np.squeeze(( probe_vertical_interval > upper) & ( probe_vertical_interval < lower)  )) )for upper, lower in zip( z_bins-delta_z, z_bins+delta_z ) ]

        # Remove empty spatial bins where the probe does not reciprocate into
        while( np.size(t_bins[-1]) < 1 ):
            t_bins.pop()
            z_bins = z_bins[:-1]

        # Compute triplets containing the indices  (t_start, t_center_, t_end) for each
        # z_bin

        t_bin_triplet = [ np.array( [t_bin[0] + abs_offset, np.abs( probe_vertical[t_bin] - z_bin ).argmin() + t_bin[0] + abs_offset , t_bin[-1] + abs_offset] ) for t_bin, z_bin in zip(t_bins, z_bins)]

        # Replace the t_start for the first z_bin and t_end for the last z_bin with t_overlap
        # If the probe is going into ( out of ) the SOL, clip ( extend ) the first bin
        # and extend ( clip ) the last time bin to incorporate an overlap where the
        # probe reverses its direction
        if ( ( probe_vertical[ interval[-1] ] - probe_vertical[ interval[0] ] ) > 0 ):
            t_bin_triplet[0][0] -= t_overlap_idx
            t_bin_triplet[-1][-1] += t_overlap_idx
    
        elif ( ( probe_vertical[ interval[-1] ] - probe_vertical[ interval[0] ] ) < 0 ):
            t_bin_triplet[0][-1] += t_overlap_idx
            t_bin_triplet[-1][0] -= t_overlap_idx

        # Compute the mean values of signal from the original signal, not the detrended one
        mean[idx + 1, :len(t_bin_triplet)] = np.array( [ (probe_signal[t_bin[0] : t_bin[-1]]).mean() for t_bin in t_bin_triplet ] )
        
        # Compute rho, fail silently
        try:
            rho[idx + 1, :len(t_bin_triplet)]  = np.array( [ (probe_rho[t_bin[0] : t_bin[-1]]).mean() for t_bin in t_bin_triplet ] )
        except:
            rho[idx + 1, :len(t_bin_triplet)] = -1.0
            print 'Could not compute rho'

        # Compute ne, fail silently
        try:
            ne_mean[idx + 1] = ne[ interval[0] : interval[-1] ].mean()
        except:
            ne_mean[idx + 1] = -1.0
            print 'Could not compute ne'


        # Compute trends in the probe signal 
        p0 = [0., 0.]
        for t_bin in t_bin_triplet:
            p, success = leastsq( errfun, p0, args = (time_signal[ t_bin[0] : t_bin[-1]], probe_signal[ t_bin[0] : t_bin[-1] ]), maxfev = 5000 )
            lf_list.append(p)


        # Compute moments of the time series in each z_bin from the detrended signals
        print 'Computing moments'
        std[idx + 1, :len(t_bin_triplet)]  = np.array( [ signal_dt( probe_signal[t_bin[0] : t_bin[-1]] , lf_list[idx], time_signal[t_bin[0] : t_bin[-1]] ).std() for t_bin in t_bin_triplet ] )
        fluc[idx + 1, :len(t_bin_triplet)] = std[idx+1, :len(t_bin_triplet)] / mean[idx+1, :len(t_bin_triplet)]
        skewness[idx + 1, :len(t_bin_triplet)] = np.array( [ skew( signal_dt( probe_signal[t_bin[0] : t_bin[-1]] , lf_list[idx], time_signal[t_bin[0] : t_bin[-1]] ) ) for t_bin in t_bin_triplet ] )
        kurt[idx + 1, :len(t_bin_triplet)] = np.array( [ kurtosis( signal_dt( probe_signal[t_bin[0] : t_bin[-1]] , lf_list[idx], time_signal[t_bin[0] : t_bin[-1]] ) ) for t_bin in t_bin_triplet ] )

       
        t_lag = np.arange( -int(tau_max / delta_t), int(tau_max / delta_t) + 1) * delta_t / 1e-6


        plt.title('Interval %d/%d' % (idx,  len(t_intervals) ) )
        for idx2, t_bin in enumerate(t_bin_triplet):
            print 'plotting timeseries %d/%d' % ( idx, len(t_bin_triplet) )
            print idx, t_bin, np.shape(time_signal), np.shape(probe_signal)
            fig = plt.figure() 
            plt.plot( time_signal[ t_bin[0] : t_bin[-1]], probe_signal[ t_bin[0]: t_bin[-1] ] )
            fig.savefig('../plots/plot_tmp_%04d.png' % (100*idx+idx2) )
            plt.close()


#        plt.figure()
#        plt.title('Autocorrelation functions') 
#        # Compute autocorrelation function and autocorr. time for each z_bin
#        for idx2, t_bin in enumerate(t_bin_triplet):
#            print 'Computing correlations, bin %d/%d' % ( idx2, len(t_bin_triplet) )
#
#            ac = correlate( probe_signal[t_bin[0] : t_bin[-1]], probe_signal[ t_bin[0] : t_bin[-1] ], int(tau_max / delta_t)) 
#            ac_dt = correlate( signal_dt( probe_signal[t_bin[0] : t_bin[-1]] , lf_list[idx2], time_signal[t_bin[0] : t_bin[-1]] ), signal_dt( probe_signal[t_bin[0] : t_bin[-1]] , lf_list[idx2], time_signal[t_bin[0] : t_bin[-1]] ),   int(tau_max / delta_t) )
#
#            n, t_corr = fwhm_estimate( ac_dt, delta_t )
#
#
#            if ( (idx2+ 2 > 0 ) and (idx2 + 2 < 6) ):
#                plt.subplot(6,2,idx2*2 + 1)
#                plt.title('I_sat')
#                plt.plot( time_signal[ t_bin[0] : t_bin[-1] ], probe_signal[t_bin[0] : t_bin[-1] ])
#    
#                plt.plot( time_signal[ t_bin[0] : t_bin[-1] ], signal_dt( probe_signal[t_bin[0] : t_bin[-1]] , lf_list[idx2], time_signal[t_bin[0] : t_bin[-1]] ) )
#                plt.subplot(6,2,idx2*2+2)
#                plt.title('AC, t = %4.2f' % time_signal[t_bin[1]] ) 
#                plt.plot(t_lag,  ac )
#
#                plt.plot( t_lag[n[0] : n[1]], ac_dt[n[0]:n[1]], 'r', label = 't_corr = %f' % (t_corr))
#                plt.plot(t_lag,  ac_dt )
#    
#                plt.xlabel('time / mus')
#                plt.legend()
#        
#            acor[idx+1, idx2] = t_corr
#

    # Average over all reciprocations

    mean[0,:] = mean[1:,:].mean(axis=0)
    std[0,:] = std[1:,:].mean(axis=0)
    fluc[0,:] = fluc[1:,:].mean(axis=0)
    skewness[0,:] = skewness[1:,:].mean(axis=0)
    kurt[0,:] = kurt[1:,:].mean(axis=0)
    acor[0,:] = acor[1:,:].mean(axis=0)

    # Which doesn't make sense for rho and ne
    rho[0, :] = -1.0
    ne_mean[0] = -1.0
    

    return ne, rho, mean, std, fluc, skewness, kurt, acor



def binning_moments_2( probe_signal, ne_signal, rho_signal, probe_time, t_intervals, binned_list ):
    """ 
    Compute statistical moments of the probe signal, given some binning
    """

    num_reciprocations = len(t_intervals)/2
    n_zbins = len(binned_list[0])

    ne_mean  = np.zeros( [num_reciprocations*2 + 1, n_zbins ] )
    mean = np.zeros( [num_reciprocations*2 + 1, n_zbins] ) 
    rho = np.zeros( [num_reciprocations*2 + 1, n_zbins] ) 
    std = np.zeros( [num_reciprocations*2 + 1, n_zbins] ) 
    skewness = np.zeros( [num_reciprocations*2 + 1, n_zbins] ) 
    kurt = np.zeros( [num_reciprocations*2 + 1, n_zbins] ) 
    fluc = np.zeros( [num_reciprocations*2 + 1, n_zbins] ) 

    # Compute the moments of the probe signal for each partial reciprocation of the probe
    for idx, interval, bin_list in zip( np.arange(len(t_intervals)), t_intervals, binned_list ):
        t_int = probe_time[ interval[0] : interval[-1] ]
        s_int = probe_signal[ interval[0] : interval[-1] ]
        ne_int = ne_signal[ interval[0] : interval[-1] ]
        rho_int = rho_signal[ interval[0] : interval[-1] ]

        #for idx2, bl in enumerate(bin_list):
        #    plt.plot(t_int[bl], s_int[bl], '.' )
        #    continue

        mean[idx + 1, :len(bin_list)] = np.array( [s_int[bl].mean() for bl in bin_list] )
        std[idx + 1, :len(bin_list)] = np.array( [s_int[bl].std() for bl in bin_list] )
        fluc[idx + 1, :len(bin_list)] = std[idx + 1, :len(bin_list)] / mean[idx + 1, :len(bin_list)]
        skewness[idx + 1, :len(bin_list)] = np.array( [ skew( s_int[bl] ) for bl in bin_list ] )
        kurt[idx + 1, :len(bin_list)] = np.array( [ kurtosis( s_int[bl] ) for bl in bin_list ] )
        ne_mean[idx + 1, :len(bin_list)] = np.array( [ne_int[bl].mean() for bl in bin_list] )
        rho[idx + 1, :len(bin_list)] = np.array( [rho_int[bl].mean() for bl in bin_list] )

    mean[0,:] = mean[1:,:].mean( axis=0 )
    std[0,:] = std[1:,:].mean( axis=0 )
    fluc[0,:] = fluc[1:,:].mean( axis=0 )
    skewness[0,:] = skewness[1:,:].mean( axis=0 )
    kurt[0,:] = kurt[1:,:].mean( axis=0 )
    ne_mean[0,:] = ne_mean[1:,:].mean( axis=0 )
    rho[0,:] = rho[1:,:].mean( axis=0 )


    return mean, std, fluc, skewness, kurt, ne_mean, rho 



def binning_time_vfloat( probe_vertical, probe_voltage, z_bins, time_signal, t_maxrc, t_low = 0.075, t_up = 0.05, delta_z = 3e-3, epsilon = 1e-6, vmin = 0.65):
    """
    Given a list of z positions for the probe, return two lists of time indices:
    t_intervals:        Time indices for each in/out reciprocration of the probe
    binned_list:        List of time indices for each reciprocation where the probe is in v_float, i.e. 
                        v_probe < v_min * probe_voltage.min()

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

    
    delta_t = time_signal[1] - time_signal[0]
    n_zbins = np.shape(z_bins)[0]
    vfloat = vmin * probe_voltage.min()

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
    t_intervals_full = [ np.squeeze( np.argwhere((time_signal > (t - t_low)) & (time_signal < t )) ) for t in t_maxrc] \
        + [ np.squeeze( np.argwhere((time_signal > t ) & ( time_signal < t + t_up )) ) for t in t_maxrc]
    
    t_intervals = [ (t_int[0], t_int[-1]) for t_int in t_intervals_full ]
    t_intervals_full = []

    binned_list = []

    # Cut each probe trajectory (up, down) in sub-intervalls and filter by magnitude of V
    for idx, interval in enumerate( t_intervals ):
        probe_vertical_interval = probe_vertical[ interval[0] : interval[-1] ]
        vfloat_interval = probe_voltage[ interval[0] : interval[-1] ]
        time_interval = time_signal[ interval[0] : interval[-1] ]

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

        binned_list.append( v_bins_float )

    return  t_intervals, binned_list

