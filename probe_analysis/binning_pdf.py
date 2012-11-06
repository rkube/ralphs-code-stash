#!/usr/bin/python
#-*- Encoding: UTF-8 -*-

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from misc.correlate import correlate
from misc.peak_detection import detect_peaks_1d as peak_detection
from misc.zero_crossing import zero_crossing_ip as zero_crossing
from scipy.stats import skew, kurtosis


def binning_pdf( probe_signal, probe_time, probe_rho, t_intervals, binned_list, nbins = 100, plot = True):
    """
    Compute PDF of the probe signal within the specifed bins

    Input:
        probe_signal:
        t_intervals:        List of time intervals that correspond to a single radial position

    Output:
        pdf


    """
    num_reciprocations = len( t_intervals ) / 2
    num_zbins = np.max( [len(bin) for bin in binned_list] )

#    pdf   = np.zeros( [num_reciprocations*2 + 1, num_zbins, nbins] )
    rho       = np.zeros(  [num_reciprocations*2 + 1, num_zbins ] )
    sigma     = np.zeros(  [num_reciprocations*2 + 1, num_zbins ] )
    mu        = np.zeros(  [num_reciprocations*2 + 1, num_zbins ] )
    skewness  = np.zeros(  [num_reciprocations*2 + 1, num_zbins ] )
    kurt      = np.zeros(  [num_reciprocations*2 + 1, num_zbins ] )


    pdf = 0.0

    # Compute PDF at radial positions for each in/out reciprocation
    for idx, interval, bin_list in zip( np.arange(num_reciprocations*2), t_intervals, binned_list ):
        t_int = probe_time  [ interval[0] : interval[-1] ] 
        s_int = probe_signal[ interval[0] : interval[-1] ]
        r_int = probe_rho   [ interval[0] : interval[-1] ]
    
        print 'interval = ', interval, 'rho = ', r_int[0], ' - ', r_int[-1]
        idx2 = 1

        plt.figure()
        plt.subplot(211)
        plt.plot( t_int, r_int )
        plt.subplot(212)
        for bl in bin_list:
            plt.plot( t_int[bl], s_int[bl] )
        plt.show()


        for bl in bin_list:
            # Skip empty bin lists
            if len(bl) == 0:
                continue
            signal_short = np.array(s_int[ bl[0] : bl[-1] ])
            rho[idx+1, idx2-1] = r_int[ int(np.mean(bl)) ]
            rho_now = rho[idx+1, idx2-1]
            mu[idx+1, idx2-1] = signal_short.mean()
            # std() gives the RMS, i.e. the absolute fluctuations
            sigma[idx+1, idx2-1] = signal_short.std() 
            skewness[idx+1, idx2-1] = skew(signal_short)
            kurt[idx+1, idx2-1] = kurtosis(signal_short)           

 
            print 'rho = %f, timeseries is %d elements long' % ( rho_now, len(bl) )

            signal_short = np.array(s_int[ bl[0] : bl[-1] ])

            #plt.figure()

            # Compute PDF of signal
            hist, bin_edges = np.histogram( signal_short, bins = nbins, range = ( signal_short.min(), signal_short.max() ), normed= True )
            x = 0.5*( bin_edges[1:] + bin_edges[:-1] )
            # Compute mean, std for Gaussian
            #plt.plot( x, hist, 'ko')   
            #plt.plot( x, np.exp( -(x-mu[idx+1,idx2-1])*(x-mu[idx+1,idx2-1]) / (2.0 * sigma[idx+1,idx2-1]*sigma[idx+1,idx2-1]) ) / np.sqrt(2.0 * np.pi * sigma[idx+1,idx2-1] * sigma[idx+1,idx2-1]), 'r--' )
            #plt.title('rho = %4f: mean = %5f std=%f' % (rho_now, mu[idx,idx2], sigma[idx,idx2] ) )

            #fig = plt.gcf()
            #filename = '/home/rkube/FSP/plots/120217/1120217010_Vfloat_pdf_rho_recip%d_%d.png' % (idx, idx2)
            #print filename
            #fig.savefig(filename)
            #plt.close()
            idx2 = idx2 + 1

        plt.show()

    rho[0, :] = rho[1:,:].mean(axis=0)
    mu[0,:] = mu[1:,:].mean(axis=0)
    sigma[0,:] = sigma[1:,:].mean(axis=0)
    skewness[0,:] = skewness[1:,:].mean(axis=0)
    kurt[0,:] = kurt[1:,:].mean(axis=0)


    return rho,  mu, sigma, skewness, kurt




def binning_acorr_two( probe_signal_1, probe_signal_2, probe_time, probe_rho, t_intervals, binned_list, tau_wlen = 25, wmax_len = 8, plot = True ):

    """
    Compute autocorrelation function for large amplitude fluctuations

    Input:
    ============
    probe_signal:   Probe signal 1 (I_sat, V_float)...
    probe_signal:   Probe signal 2 (I_sat, V_float)...
    probe_time:     Time for the signal
    probe_rho:      Probe rho coordinate
    t_intervals:    Indices where probe is in in/out reciprocation
    binned_list:    Intervalls for rho binning
    tau_max:        Maximum autocorrelation time to compute autocorrelation for
    wmax_len:       Intervall for local meaximum detection in the timeseries
    plot:           Show plots of what we are doing?

    Output:
    ============

    autocorrelation times

    This routine detects large peaks on probe_signal one. It then crops a small intervall from both,
    signal_1 and signal_2 and computes the autocorrelation function and autocorrelation times of the intervalls.

    """


    num_reciprocations = len ( t_intervals ) / 2
    num_zbins = np.max( [len(bin) for bin in binned_list] )
    delta_t = probe_time[3] - probe_time[2]
    n_zbins = np.max( [len(bin) for bin in binned_list] )

    tau_acorr1 = np.zeros( [num_reciprocations*2 + 1, n_zbins] )
    tau_acorr2 = np.zeros( [num_reciprocations*2 + 1, n_zbins] )
    rho_int = np.zeros( [num_reciprocations*2 + 1, n_zbins] )

    print 'dt = %10.9f' % delta_t

    figure_number = 0

    for idx, interval, bin_list in zip( np.arange( num_reciprocations*2), t_intervals, binned_list ):
        # Iteration over in/out reciprocation of the probe
        t_int  = probe_time    [ interval[0] : interval[-1] ] 
        s1_int = probe_signal_1[ interval[0] : interval[-1] ]
        s2_int = probe_signal_2[ interval[0] : interval[-1] ]
        r_int  = probe_rho     [ interval[0] : interval[-1] ]

        avg_blob = np.zeros( 2*tau_wlen )
        avg_vfl = np.zeros( 2*tau_wlen )
        for bin_idx, bl in enumerate(bin_list):
            # Iteration over rho_bin of a single reciprocation
            t_int_bin  = t_int[bl]
            s1_int_bin = s1_int[bl]
            s2_int_bin = s2_int[bl]
            r_int_bin  = r_int[bl]

            # Normalize probe signal to relative fluctuation
            s_int_bin_rel = (s1_int_bin - s1_int_bin.mean() )/ s1_int_bin.std()
           
            # Find time indices of peaks within +- tau_wlen of the intervall where the probe is reciprocating
            avg_blob[:] = 0.
            avg_vfl[:] = 0.
            n_blobs = 0
            tau_ac = []

            # Sort array by value, by decreasing amplitude
            max_values = np.argsort( s_int_bin_rel )[::-1]
            # Number of values in the fluctuations exceeding 2.5
            num_big_ones = np.sum( s_int_bin_rel > 2.5 )
            # Cut off fluctuations not exceeding 2.5
            max_values = max_values[:num_big_ones]
            # Eliminate peaks within wmax_len around one another, keeping the largest peak only

            for mv in max_values:
                if mv == -1:
                    # Item was discarded in previous iteration, skip
                    continue
                # Identify peaks close by
                close_ones = np.squeeze( np.argwhere( np.abs(mv - max_values) < wmax_len ) )
                # Eliminate peaks close by
                try:
                    for co in close_ones:   
                        if ( max_values[co] == mv ):
                            # The current peak is identified as well, skip it
                            continue
                        max_values[co] = -1
                except TypeError:
                    # TypeError is thrown when we cannot iterate over close_ones
                    continue
            # Cut off values at the boundary
            max_values[ max_values < tau_wlen ] = -1
            max_values[ max_values > (np.size(s_int_bin_rel) - tau_wlen)] = -1
            # Cut off all filtered out values
            max_values = max_values[max_values != -1]

            print 'max_values = ', max_values
            if ( plot == True ):
                plt.figure(2)
                plt.plot( s_int_bin_rel, '.-' )
    #            plt.plot( (np.arange( np.size(s_int_bin_rel) ))[max_values], s_int_bin_rel[max_values], 'ko')
                plt.plot( max_values, s_int_bin_rel[max_values], 'ko')

            if ( plot == True ):
                fig1 = plt.figure(1)
                fig1.text(0.5, 0.95, 'rho = %4.3f' % r_int_bin.mean() )

            # Compute the average blob form
            for max_idx_it in max_values:
#                plt.figure(2)
#                plt.plot( max_idx_it, s_int_bin_rel[max_idx_it], 'rx' )
                avg_blob[:] += s_int_bin_rel[ max_idx_it - tau_wlen : max_idx_it + tau_wlen ]
                avg_vfl[:] += s2_int_bin[ max_idx_it - tau_wlen : max_idx_it + tau_wlen ]
                #acorr = correlate( s_int_bin_rel[ max_idx_it - tau_wlen : max_idx_it + tau_wlen ], s_int_bin_rel[ max_idx_it - tau_wlen : max_idx_it + tau_wlen ], tau_wlen ) 
                #tau_ac.append( (acorr[ tau_wlen :] > 0.5).argmin() * delta_t )
                if ( plot == True ):
                    plt.figure(2)
                    plt.plot( max_idx_it, s_int_bin_rel[max_idx_it], 'rx' )
                    plt.figure(1)
                    plt.subplot(321)
                    plt.plot( s_int_bin_rel[ max_idx_it - tau_wlen : max_idx_it + tau_wlen ], '.' )
                    plt.subplot(322)
                    plt.plot( s2_int_bin[ max_idx_it - tau_wlen : max_idx_it + tau_wlen ] )
                    #print 'Index %d, rho: %f, value: %3.2f, tau_ac = %e' % ( max_idx_it, r_int[bl].mean(), s_int_bin_rel[max_idx_it], tau_ac[-1] )

            # Compute autocorrelation time of the average blob
            #acorr = correlate( s_int_bin_rel[ max_idx_it - tau_wlen : max_idx_it + tau_wlen ], s_int_bin_rel[ max_idx_it - tau_wlen : max_idx_it + tau_wlen ], tau_wlen ) 
            acorr = correlate( avg_blob, avg_blob, tau_wlen )
            acorr_vfl = correlate( avg_vfl, avg_vfl, tau_wlen )
            tau_acorr1[idx, bin_idx] = (acorr[tau_wlen:] > 0.5).argmin() * delta_t
            #tau_ac.append( (acorr[ tau_wlen :] > 0.5).argmin() * delta_t )
            
            if ( plot == True ):
                plt.figure(1)
                plt.subplot(323)
                plt.plot( avg_blob / float(np.size(max_values)) )
                plt.subplot(324)
                plt.plot( avg_vfl / float(np.size(max_values)) )
                plt.subplot(325)
                plt.plot( np.arange(-tau_wlen, tau_wlen+1), acorr )
                plt.subplot(326)
                plt.plot( np.arange(-tau_wlen, tau_wlen+1), acorr_vfl )
                fig_filename = '../plots/ac_blob_isat_fig%d.png' % figure_number
                #fig1.savefig( fig_filename )
                #print 'saved figure to %s' % fig_filename 
                figure_number += 1
                plt.close(1)
                plt.close(2)

            
            plt.show()
                    

        rho_int[ idx+1, :len(bin_list) ] = np.array( [r_int[bl].mean()  for bl in bin_list])
        tau_acorr1[ idx+1, :len(tau_ac) ] = np.array( tau_ac )

        plt.show() 

    return rho_int, tau_acorr1, tau_acorr2



def binning_acorr( probe_signal, timebase, probe_rho, t_intervals, binned_list, min_sep = 25, tau_max = 8, threshold = 2.0, plot = True ):
    """
    Compute autocorrelation function for large amplitude fluctuations

    Input:
    ============
    probe_signal:   Probe signal (I_sat, V_float)...
    probe_signal:   Probe signal (I_sat, V_float)...
    probe_time:     Time for the signal
    probe_rho:      Probe rho coordinate
    t_intervals:    Indices where probe is in in/out reciprocation
    binned_list:    Intervalls for rho binning
    min_sep:        Minimum separation for peak detection
    tau_max:        Maximum autocorrelation time
    plot:           Show plots of what we are doing?

    Output:
    ============

    """

    num_reciprocations = len ( t_intervals ) / 2
    num_zbins = np.max( [len(bin) for bin in binned_list] )
    delta_t = timebase[300] - timebase[299]
    n_zbins = np.max( [len(bin) for bin in binned_list] )

    tau_acorr = np.zeros( [num_reciprocations*2 + 1, n_zbins] )
    rho_int = np.zeros( [num_reciprocations*2 + 1, n_zbins] )
    tau_fwhm = np.zeros( [num_reciprocations*2 + 1, n_zbins] )

    acorr_mask = np.zeros( np.shape(tau_acorr) )
    fwhm_mask = np.zeros( np.shape(tau_acorr) )

    figure_number = 0

    for idx, interval, bin_list in zip( np.arange( num_reciprocations*2), t_intervals, binned_list ):
        # Iteration over in/out reciprocation of the probe
        t_int = timebase  [ interval[0] : interval[-1] ] 
        s_int = probe_signal[ interval[0] : interval[-1] ]
        r_int = probe_rho   [ interval[0] : interval[-1] ]

        for bin_idx, bl in enumerate(bin_list):
            avg_blob = np.zeros( 2*tau_max )
            if (np.size(bl) == 0):
                # Skip rho bins where probe does not reciprocate into
                continue
            # Iteration over rho_bin of a single reciprocation
            r_int_bin = r_int[bl]
#            print 'reciprocation %d rho = %4.3f, s_int.mean() = ' %( idx +1, r_int[bl].mean() ), s_int[bl].mean(), 's_int.std() = ', s_int[bl].std() 
#            print 'Signal: ', s_int[bl][:5]
            t_int_bin = t_int[bl]
            s_int_bin = ( s_int[bl] - s_int[bl].mean() ) / s_int[bl].std()

#            print 'Signal from %d to %d' % ( bl[0], bl[-1] )
#            print 'Signal: ', s_int_bin[:5]

            # Locate local peaks
            max_idx = peak_detection( s_int_bin, t_int_bin, min_sep, threshold)
            num_blobs = np.size( max_idx )

#            print 'reciprocation %d rho = %4.3f, max_idx = ' % (idx+1, r_int_bin.mean()),  max_idx
            if ( plot == True ):
                fig1 = plt.figure(1)
                fig1.text(0.5, 0.95, 'rho = %4.3f, %d blobs' % (r_int_bin.mean(), num_blobs) )
                plt.subplot(411)
                plt.plot( t_int_bin, s_int_bin, '.-' )
                plt.plot( t_int_bin[max_idx], s_int_bin[max_idx], 'ko')

            # Compute the average blob form
            for max_idx_it in max_idx:
                avg_blob[:] = avg_blob[:] + s_int_bin[ max_idx_it - tau_max : max_idx_it + tau_max ]
                if ( plot == True ):
                    plt.figure(1)
                    plt.subplot(412)
                    plt.plot( t_int_bin[ max_idx_it - tau_max : max_idx_it + tau_max ], s_int_bin[ max_idx_it - tau_max : max_idx_it + tau_max], '.' )
                    #print 'Index %d, rho: %f, value: %3.2f, tau_ac = %e' % ( max_idx_it, r_int[bl].mean(), s_int_bin_rel[max_idx_it], tau_ac[-1] )

            # Normalize average blob size
            print '%d blobs detected' % ( float(num_blobs) )
            avg_blob[:] = avg_blob[:] / float(num_blobs)
            np.savez('/home/rkube/tmp/fsp_avg_blob_%03d_1120711020.npz' % figure_number, avg_blob = avg_blob)

            # Compute autocorrelation time of the average blob
            acorr = correlate( avg_blob, avg_blob, tau_max )
            #acorr = np.correlate( avg_blob, avg_blob, 'full' )
            #acorr = np.correlate( avg_blob, avg_blob, 'full')
            acorr = acorr[np.size(acorr)/2 - tau_max : np.size(acorr)/2 + tau_max] / acorr.max()

            # Ladida, compute le FWHM
            try:
                fwhm2 = zero_crossing( np.arange(tau_max), avg_blob[tau_max:] - 0.5 * avg_blob[tau_max:].max(), title='FWHM1' ) * delta_t
                fwhm1 = zero_crossing(np.arange(tau_max), avg_blob[tau_max:0:-1], title='FWHM2' ) * delta_t
                tau_fwhm[idx, bin_idx] = fwhm1 + fwhm2
                print 'FWHM1 = %e, FWHM2 = %e' % ( fwhm1, fwhm2 )
            except ValueError:
                fwhm_mask = True
                print 'Could not determine FWHM'

            # Ladida, compute le Autocorrelation function
            try:
                # The autocorrelation time is defined as the time it takes for the correlation amplitude to cross 0.5
                zero_x = zero_crossing( np.arange(tau_max), acorr[tau_max:] - 0.5, title='AC' ) 
                print 'Zero x at %f, delta_t = %e => tau_ac = %e' % ( zero_x, delta_t, zero_x * delta_t )
                tau_acorr[idx, bin_idx] = zero_x * delta_t
            except ValueError:
                print 'Could not determine autocorrelation time'
                acorr_mask[idx, bin_idx] = True
            
            if ( plot == True ):
                plt.figure(1)
                plt.subplot(413)
                plt.plot(np.arange(-tau_max, tau_max), avg_blob , '.-' )
                try:
                    # These might not exist
                    plt.plot( [0, fwhm1/delta_t], [0.5*avg_blob.max(), 0.5*avg_blob.max()], 'k' )
                    plt.plot( [-fwhm2/delta_t, 0], [0.5*avg_blob.max(), 0.5*avg_blob.max()], 'k--')
                except:
                    pass
                plt.subplot(414)
                plt.plot( np.arange(-tau_max, tau_max), acorr, '.-', label='zero at %f' % zero_x )
                plt.grid()
                plt.legend()
                plt.ylim( (-1.0, 1.0) )
                fig_filename = '../plots/ac/1120711020_ac_blob_isat_fig%d.png' % figure_number
                fig1.savefig( fig_filename )
                print 'saved figure to %s' % fig_filename 
                figure_number += 1
                #plt.close(1)
                plt.show() 
 
            plt.show()
        
        rho_int[ idx+1, :len(bin_list) ] = np.array( [r_int[bl].mean()  for bl in bin_list])

        plt.show() 

    tau_acorr = np.ma.MaskedArray( tau_acorr, mask = acorr_mask )
    tau_fwhm  = np.ma.MaskedArray( tau_fwhm, mask = fwhm_mask )

    return rho_int, tau_acorr, tau_fwhm



def binning_avg_blob(probe_signal, timebase, probe_R, probe_rho, t_intervals, binned_list, shotnr = 0, min_sep=25, tau_max=8, threshold=2.0, plot=True, verbose=False):
    """
    Compute average blob shape for each position of probe reciprocation

    Input:
    ==============
    probe_signal:       Unnormalized Isat signal (This will be automatically normalized for each radial bin)
    timebase:           Timebase for the isat signal
    probe_R:            Radial coordinate of the probe
    probe_rho:          Radial position of the probe
    t_intervals:        Indices where probe is in in/out reciprocation
    binned_list:        Intervalls for rho binning
    min_sep:            Minimum separation for peak detection
    tau_max:            Maximum autocorrelation time
    threshold:          Signal threshold for blob detection
    plot:               Show plots of what we are doing?

    Output:
    =========

    """

    num_reciprocations = len ( t_intervals ) / 2
    num_zbins = np.max( [len(bin) for bin in binned_list] )
    delta_t = timebase[300] - timebase[299]
    n_zbins = np.max( [len(bin) for bin in binned_list] )

    rho_int = np.zeros( [num_reciprocations*2 + 1, n_zbins] )

    figure_number = 0
    x_int = np.arange(-tau_max, tau_max+1)

    # Compute the average blob shape for each rho bin
    #avg_blob, axis0: [rho, num_blobs, avg_blob[0], avg_blob[1], avg_blob[2], ...] 
    avg_blob = np.zeros( [len(binned_list[0]), 3 + 2*tau_max+1])

    for idx, interval, bin_list in zip( np.arange( num_reciprocations*2), t_intervals, binned_list ):
        # Iteration over in/out reciprocation of the probe
        t_int = timebase  [ interval[0] : interval[-1] ] 
        s_int = probe_signal[ interval[0] : interval[-1] ]
        R_int = probe_R   [ interval[0] : interval[-1] ]
        rho_int = probe_rho   [ interval[0] : interval[-1] ]

        for bin_idx, bl in enumerate(bin_list):
            #avg_blob = np.zeros( 2*tau_max+1 )
            if (np.size(bl) == 0):
                # Skip rho bins where probe does not reciprocate into
                continue
            # Iteration over rho_bin of a single reciprocation
            R_int_bin = R_int[bl]
            rho_int_bin = rho_int[bl]
            t_int_bin = t_int[bl]
            s_int_bin = ( s_int[bl] - s_int[bl].mean() ) / s_int[bl].std()

            # Locate local peaks
            max_idx = peak_detection( s_int_bin, t_int_bin, min_sep, threshold)
            avg_blob[bin_idx, 0] = R_int_bin.mean()
            avg_blob[bin_idx, 1] = rho_int_bin.mean()
            avg_blob[bin_idx, 2] = avg_blob[bin_idx,2] + np.size(max_idx)

            for max_idx_it in max_idx:
                avg_blob[bin_idx, 3:] = avg_blob[bin_idx, 3:] + s_int_bin[max_idx_it-tau_max:max_idx_it+tau_max+1]
    
            if plot:
                fig = plt.figure()
                ax =fig.add_subplot(111)
                ax.plot(x_int, avg_blob)
                ax.set_title('R=%4.3fm: %d blobs' % (R_int_bin.mean(), num_blobs))
                filename = '../plots/111208/%d/%d_%d_r_%4.3f_avg_blob_pin0.png' % (shotnr, shotnr, idx, R_int_bin.mean())
                print 'Saving to %s' % filename
                fig.savefig(filename)
                  

    if verbose:
        for r_idx in np.arange( np.shape(avg_blob)[0] ): 
            print 'R=%4.3f: %d blobs' % ( avg_blob[r_idx,0], avg_blob[r_idx,2])

    return avg_blob



def binning_acorr_multiple( probe_signal_list, probe_time, probe_rho, t_intervals, binned_list, min_sep = 25, tau_max = 8, threshold = 2.0, plot = True ):
    """
    Compute autocorrelation function for large amplitude fluctuations

    Input:
    ============
    probe_signal:   List of probe signals (I_sat, V_float)...
    probe_time:     Timebase for the signals
    probe_rho:      Probe rho coordinate
    t_intervals:    Indices where probe is in in/out reciprocation
    binned_list:    Intervalls for rho binning
    min_sep:        Minimum separation for peak detection
    tau_max:        Maximum autocorrelation time
    plot:           Show plots of what we are doing?

    Output:
    ============

    """

    # Each probe signal has its timeseries, timebase, and rho list
    
    num_reciprocations = len( t_intervals ) / 2
    num_zbins = np.max( [len(bin) for bin in binned_list] )
    num_signals = len(probe_signal_list)
    delta_t = probe_time[3] - probe_time[2]
    n_zbins = np.max( [len(bin) for bin in binned_list] )

    tau_acorr = np.zeros( [num_signals, num_reciprocations*2 + 1, n_zbins] )
    rho_int = np.zeros( [num_reciprocations*2 + 1, n_zbins] )

    figure_number = 0

    for idx, interval, bin_list in zip( np.arange( num_reciprocations*2), t_intervals, binned_list ):
        # Iteration over in/out reciprocation of the probe
        t_int = probe_time[ interval[0] : interval[-1] ] 
        s_int_list = [probe_signal[ interval[0] : interval[-1] ] for probe_signal in probe_signal_list ]
        r_int = probe_rho[ interval[0] : interval[-1] ] 

        rho_int[idx+1, :len(bin_list) ] = np.array( [r_int[bl].mean() for bl in bin_list])
        avg_blob = np.zeros([ num_signals, 2*tau_max] )
        for bin_idx, bl in enumerate(bin_list):
            if (np.size(bl) == 0):
                # Skip rho bins where probe does not reciprocate into
                continue
            # Iteration over rho_bin of a single reciprocation
            r_int_bin = r_int[bl] 
            t_int_bin = t_int[bl]
            s_int_bin_list = [ (s_int[bl] - s_int[bl].mean())/s_int[bl].std() for s_int in s_int_list ]

            # Locate local peaks
            max_idx_list = [peak_detection( s_int_bin, t_int_bin, min_sep, threshold) for s_int_bin in s_int_bin_list  ]

            print 'reciprocation %d rho = %4.3f, max_idx = ' % (idx+1, r_int_bin.mean()),  max_idx_list
            if ( plot == True ):
                fig1 = plt.figure(1)
                fig1.text(0.5, 0.95, 'rho = %4.3f' % r_int_bin.mean() )
                plt.subplot(4, 1, 1)


            # Compute the average blob form
            for signal in np.arange( num_signals ):
                for max_idx_it in max_idx_list[signal]:
                    avg_blob[ signal, :] = avg_blob[ signal, :] + s_int_bin_list[signal][ max_idx_it - tau_max : max_idx_it + tau_max ]
                    if ( plot == True ):
                        plt.subplot(4, 2, 3 + signal )
                        plt.plot( s_int_bin_list[signal][ max_idx_it - tau_max : max_idx_it + tau_max], '.' )
                if ( plot ):
                    plt.subplot(4, 1, 1) 
                    plt.plot( t_int_bin, s_int_bin_list[signal] )
                    plt.plot( t_int_bin[max_idx_list[signal]], s_int_bin_list[signal][max_idx_list[signal]], 'kx')
                avg_blob[ signal, :] = avg_blob[signal, :] / np.size(max_idx_list[signal])

            # Compute autocorrelation time of the average blob
            for signal in np.arange( num_signals ):
                acorr = correlate( avg_blob[ signal ], avg_blob[ signal ], tau_max )
                # The autocorrelation time is defined as the time it takes for the correlation amplitude to cross 0.5
                zero_x = zero_crossing( np.arange(tau_max), acorr[tau_max:] - 0.5 ) 
                print 'Zero x at idx %d, delta_t = %f => tau_ac = %f' % ( zero_x, delta_t, zero_x * delta_t )
                tau_acorr[signal, idx, bin_idx] = zero_crossing( np.arange(tau_max), acorr[tau_max:] - 0.5 ) * delta_t
#                old_tau_ac = (acorr[tau_max:] > 0.5).argmin() * delta_t
                print 'Autocorrelation time: %e' % ( tau_acorr[signal, idx, bin_idx])
            
            if ( plot == True ):
                for signal in np.arange( num_signals ):
                    plt.figure(1)
                    plt.subplot(4, 2, 5 + signal)
                    plt.plot( avg_blob[signal, :] )
                    plt.subplot(4, 2, 7 + signal)
                    plt.plot( np.arange(-tau_max, tau_max+1), acorr, '.-' )
                    plt.grid()
                    plt.ylim( (-1.0, 1.0) )
                    fig_filename = '../plots/1120711020_ac_blob_isat_fig%d_compare.png' % figure_number
                fig1.savefig( fig_filename )
                print 'saved figure to %s' % fig_filename 
                figure_number += 1
                plt.close(1)
#                plt.close(2)

    return rho_int, tau_acorr





