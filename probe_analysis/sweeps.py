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




def sweep_profile_fitui(timebase, jsat, vsweep, probe_R, sweep_tuple, show_plots = True):
    """
    Fit a Langmuir characteristic on each U-I curve

    Input:
        timebase:       Common timebase for jsat, vsweep
        jsat:           Jsat signal of the probe
        vsweep:         Sweeping voltage applied to the probe
        sweep_tuple:    List of tuples [(sweep_start_idx, sweep_end_idx), ...]
    Returns:
    """

    ui_fun = lambda U, V_float, T_e, I_sat : I_sat* ( np.exp( (U-V_float)/T_e) - 1.0)
    # Iterate over the tuples
    R = np.zeros( len(sweep_tuple), dtype='float64')
    isat_profile = np.zeros(len(sweep_tuple), dtype='float64')
    Te_profile = np.zeros(len(sweep_tuple), dtype='float64')
    Vf_profile = np.zeros(len(sweep_tuple), dtype='float64')

    for idx, stuple in enumerate(sweep_tuple):
        print '===================== Processing tuple (%d,%d)' % stuple
        s0, s1 = stuple
        tl2 = int(round((s1-s0)/2))

        U = vsweep[s1-tl2:s1]
        I = jsat[s1-tl2:s1]
        t = timebase[s1-tl2:s1]

        R[idx] = probe_R[s1-tl2:s1].mean()

        # Fit U-I characteristic on each tuple
        Isat, err_Isat, Vf, err_Vf, Te, err_Te = uifit(U.astype('float64'),I.astype('float64'), show_plots = False)
        print 'Recieved: ISat=%4.3f pm %4.3f A, Vf= %4.3f pm %4.3f V, Te=%4.3f pm %4.3f eV' % (Isat, err_Isat, Vf, err_Vf, Te, err_Te)
        isat_profile[idx] = Isat
        Te_profile[idx] = Te
        Vf_profile[idx] = Vf


        if show_plots:
            fig = plt.figure()
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)
            ax1.plot(t, U)
            ax2.plot(U,I, 'b,', label='Probe')
            ax2.plot(U, -ui_fun(U, Vf, Te, Isat),'k', label='Fit: I = %4.3f * exp(V - %4.2fV/%4.2f)'% ( Isat, Vf, Te))
            ax2.legend(loc='lower left')

    plt.show()

    return R, isat_profile, Te_profile, Vf_profile






