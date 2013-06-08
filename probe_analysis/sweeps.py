#!/usr/bin/env python
#-*- Encoding: UTF-8 -*-

"""
======
sweeps
======

..codeauthor.. Ralph Kube

Procesing of sweeped probe timesereis

Contents:
get_sweep_idx

"""

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from uifit import uifit
from misc.peak_detection import detect_peaks_1d
from probe_analysis.uifit import uifit_minTE, FitException, ui_fun

def get_sweep_idx(probe_vsweep, probe_R, Rmin, Rmax, timebase, dt_sweep = 3000, show_plots = False):
    """
    Returns index tuples for each sweep of the probe when it is plunged in between Rmin and Rmax.

    Input:
    probe_vsweep:  Potential waveform
    probe_R:       Timeseries of radial probe position.
    Rmin:          Minimum radial probe position where sweeps are identified
    Rmax:          Maximum radial probe position where sweeps are indentified
    timebase:      Timebase for probe_vsweep and probe_R
    dt_sweep:      Discard all sweep tuples which are longer than dt_sweep

    Returns:
    sweep_idx:     List of tuples giving the start and end of each sweep (sweep_start_idx, sweep_end_idx)

    Last edited: Ralph Kube 12/05/12
    History:    12/05/12 Ralph Kube: Iniital version
    """

    # Cut timeseries down to parts where probe is in designated area
    #print 'Creating set of time indices where probe is in SOL'
    #where_idx = set( np.squeeze(np.argwhere( (probe_R > Rmin) & (probe_R < Rmax) )))
    #print 'done'
    #probe_vsweep = np.squeeze(probe_vsweep[where_idx])
    #probe_R = np.squeeze(probe_R[where_idx])
    #timebase = np.squeeze(timebase[where_idx])
    # Identify voltage minima
    sweep_max = np.sort(detect_peaks_1d(-probe_vsweep, timebase, 50, -0.9*probe_vsweep.min(), peak_width=100))

    # Built array of tuples
    print 'Creating probe sweep tuples'
    #sweep_tuples = [ (s_low, s_up) for s_low, s_up in zip( sweep_max[:-1], sweep_max[1:]) if (np.abs(s_low - s_up) < int(dt_sweep / (timebase[100] - timebase[99])) and s_low in where_idx and s_up in where_idx)  ]
    sweep_tuples = [ (s_low, s_up) for s_low, s_up in zip( sweep_max[:-1], sweep_max[1:]) if (np.abs(s_low - s_up) < int(dt_sweep / (timebase[100] - timebase[99])) )  ]
    print 'Done'
    # Remove tuples which are more than average apart. We assume th


    if show_plots:
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        #ax2 = fig.add_subplot(212)
        ax2 = ax1.twinx()
        ax1.plot( timebase, probe_R )
        ax1.set_xlabel('time/s')
        ax1.set_ylabel('R/m')
   
        ax2.plot(timebase, probe_vsweep)
        ax2.plot( timebase[sweep_max], probe_vsweep[sweep_max], 'k.')
        ax2.set_xlabel('time/s')
        ax2.set_ylabel('V/Volt')


        fig2 = plt.figure()
        ax3 = fig2.add_subplot(111)
        ax3.plot( timebase[sweep_tuples[0][0]:sweep_tuples[0][1]],  probe_vsweep[sweep_tuples[0][0]:sweep_tuples[0][1]]) 
        ax3.plot( timebase[sweep_tuples[1][0]:sweep_tuples[1][1]],  probe_vsweep[sweep_tuples[1][0]:sweep_tuples[1][1]])
        ax3.plot( timebase[sweep_tuples[2][0]:sweep_tuples[2][1]],  probe_vsweep[sweep_tuples[2][0]:sweep_tuples[2][1]])
        ax3.plot( timebase[sweep_tuples[3][0]:sweep_tuples[3][1]],  probe_vsweep[sweep_tuples[3][0]:sweep_tuples[3][1]])

        plt.show()

    print 'Returning %d sweeps' % ( len(sweep_tuples) )
    return sweep_tuples


def sweep_profile_mean(sweep_tuples, timeseries):
    """
    Return the mean over each tuple [t1:t2] of sweep_tuples
    """


    mean_list = np.array([ timeseries[t[0]:t[1]].mean() for t in sweep_tuples ])

    return mean_list


#def sweep_profile_fitui(timebase, jsat, vsweep, probe_R, sweep_tuple, show_plots = True, save_good_fits = True, save_basename = '/Users/ralph/tmp/mybook_backup/cmod_data/local/plots'):
def sweep_profile_fitui(timebase, jsat, vsweep, probe_R, sweep_tuple, show_plots = True, show_bad_fits = False, save_good_fits = False, save_basename = '/Volumes/My Book Thunderbolt Duo/cmod_data/local/plots', silent=True):
    """
    Fit a Langmuir characteristic on each U-I curve

    Input:
        timebase:       Common timebase for jsat, vsweep
        jsat:           Jsat signal of the probe
        vsweep:         Sweeping voltage applied to the probe
        sweep_tuple:    List of tuples [(sweep_start_idx, sweep_end_idx), ...]
    Returns:
    """

    #ui_fun = lambda U, V_float, T_e, I_sat : I_sat* ( np.exp( (U-V_float)/T_e) - 1.0)
    # Iterate over the tuples
    R = np.zeros( len(sweep_tuple), dtype='float64')
    isat_profile = np.zeros(len(sweep_tuple), dtype='float64')
    isat_rms_profile = np.zeros(len(sweep_tuple), dtype='float64')
    isat_fluc_profile = np.zeros(len(sweep_tuple), dtype='float64')
    isat_skew_profile = np.zeros(len(sweep_tuple), dtype='float64')
    isat_kurt_profile = np.zeros(len(sweep_tuple), dtype='float64')
    Te_profile = np.zeros(len(sweep_tuple), dtype='float64')
    Vf_profile = np.zeros(len(sweep_tuple), dtype='float64')
    num_points = np.zeros(len(sweep_tuple), dtype='float64')

    # Create a mask which indicates bad fits
    bad_fit_mask = np.ones(np.shape(isat_profile))
    bad_fit_counter = 0
    for idx, stuple in enumerate(sweep_tuple):
        if not(silent):
            print '===================== Processing tuple (%d,%d)' % stuple
        s0, s1 = stuple
        tl2 = int(round((s1-s0)/2))

        U = vsweep[s1-tl2:s1].astype('float64')
        I = jsat[s1-tl2:s1].astype('float64')
        t = timebase[s1-tl2:s1]
        R[idx] = probe_R[s1-tl2:s1].mean()

        U_work = np.zeros_like(U)
        I_work = np.zeros_like(I)

        if (np.size(U) < 100 ):
            isat_profile[idx] = np.nan
            isat_rms_profile[idx] = np.nan
            isat_fluc_profile[idx] = np.nan
            isat_skew_profile[idx] = np.nan
            isat_kurt_profile[idx] = np.nan
            Te_profile[idx] = np.nan
            Vf_profile[idx] = np.nan
            bad_fit_mask[idx] = True
            bad_fit_counter += 1
            continue            

        # Assure that the U-I sweep is ordered such that U is an array of increasing voltages
        if ( U[0] > U[-1] ):
            U_work[:] = U[::-1]
            I_work[:] = I[::-1]
        else:
            U_work[:] = U[:]
            I_work[:] = I[:]

        # Fit U-I characteristic on each tuple
        try:
            if not(silent):
                print '------- Trying U-I fit on U=[%f:%f]' % (U_work[0], U_work[-1])
            if save_good_fits:
                isat_profile[idx], isat_rms_profile[idx], isat_fluc_profile[idx], isat_skew_profile[idx], isat_kurt_profile[idx], Vf_profile[idx], sigma_vf, Te_profile[idx], sigma_te, num_points[idx], fit_result_fig = uifit_minTE(U_work, I_work, fit_min = 50, fit_max = 300, fit_step = 20, eps_isat_trend =5e-2, return_plot = True, silent=True)
                fit_filename = '%s/uifit_t%5.4fs_R%3.2f.eps' % (save_basename, t.mean(), R[idx])
                if not(silent):
                    print 'Saving good fit to %s' % (fit_filename)
                fit_result_fig.savefig(fit_filename)
                plt.close()

            else:
                isat_profile[idx], isat_rms_profile[idx], isat_fluc_profile[idx], isat_skew_profile[idx], isat_kurt_profile[idx], Vf_profile[idx], sigma_vf, Te_profile[idx], sigma_te, num_points[idx] = uifit_minTE(U_work, I_work, fit_min = 50, fit_max = 300, fit_step = 20, eps_isat_trend =5e-2, return_plot = False, silent=True)

            if show_plots:
                fit_fig = plt.figure()
                fit_axis = fit_fig.add_subplot(111)
                fit_axis.plot(U,I, label='Probe signal')
                fit_axis.plot(U, -ui_fun(U, isat_profile[idx], Vf_profile[idx], Te_profile[idx]), label='Fit: %4.3f*( exp((U-%5.2fV)/%4.2eV) -1.0)' % ( isat_profile[idx], Vf_profile[idx], Te_profile[idx]))
                fit_axis.legend(loc='best')
                fit_axis.set_xlabel('probe bias/V')
                fit_axis.set_xlabel('probe current/A')
                plt.show()

        except FitException, e:
            print e
            print 'Bad fit at t=%5.4fs, R=%4.2fm' % (t.mean(), R[idx])
            isat_profile[idx] =  np.nan
            isat_rms_profile[idx] = np.nan
            isat_fluc_profile[idx] = np.nan
            isat_skew_profile[idx] = np.nan
            isat_kurt_profile[idx] = np.nan
            Te_profile[idx] = np.nan
            Vf_profile[idx] = np.nan
            bad_fit_mask[idx] = True
            bad_fit_counter += 1
            if show_bad_fits:
                fail_fig = plt.figure()
                fail_axis = fail_fig.add_subplot(111)
                fail_axis.plot(U, I)
                fail_axis.set_title('Failed to fit')
                plt.show()

        isat_profile_ma = ma.MaskedArray(isat_profile, mask=bad_fit_mask)
        Te_profile_ma = ma.MaskedArray(Te_profile, mask=bad_fit_mask)
        Vf_profile_ma = ma.MaskedArray(Vf_profile, mask=bad_fit_mask)

        plt.show()

    print 'Total/Failed fits: %d/%d' % (np.size(sweep_tuple), bad_fit_counter)

    return R, isat_profile, isat_rms_profile, isat_fluc_profile, isat_skew_profile, isat_kurt_profile, Te_profile, Vf_profile, num_points






