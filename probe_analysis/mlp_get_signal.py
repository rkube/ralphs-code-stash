#!/opt/local/bin/python
#-*- Encoding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt

"""
Get signal from MLP.
Data taken from MDS tree according to ~/misc_scripts/save_iv_profile_mlp.py

* Radial velocity, V_rad, is averaged over both pairs of poloidally separated bins
* Radial particle flux, Gamma_n, is V_rad * average of all 4 particle densities
* Radial heat flux, Gamma_T, is V_rad * average of all 4 particle densities and all 4 temperatures
"""

mlp_var_list = ['ne', 'Te', 'Vp', 'Vf', 'Is', 'Vrad', 'Gamma_n', 'Gamma_T']
mlp_data_dir = '/Volumes/ekgroup/cmod_data/'

var_str_dict = {'ne': r"n_\mathrm{e}",
                'Te': r"T_\mathrm{e}",
                'Is': r"I_\mathrm{s}",
                'Vp': r"V_\mathrm{p}",
                'Vf': r"V_\mathrm{f}",
                'Vrad': r"V_\mathrm{rad}",
                'Gamma_n': r"\Gamma_n",
                'Gamma_T': r"\Gamma_T"}

pin_name_dict = {0: 'NE', 1: 'SE', 2: 'SW', 3: 'NW'}

def mlp_get_signal(varname, pin, shotnr, t_start, t_end, datadir=mlp_data_dir,
        pot_var='Vp', vr_mode='average', flux_var = 'ne'):
    """
    Construct signal from MLP datafile

    Arguments:
    ===========
    varname.....string, gives the signal to return:
                            ne: Particle density in 10^20 m^-3
                            Te: Electron temperature in eV
                            Vp: Plasma potential in Volt
                            Vf: Floating potential in Volt
                            Vrad: Radial velocity in ms^-1, as computed by Vplasma
                            Gamma_n: Radial particle flux in 10^20 m^-2 s^-1
                            Gamma_T: Radial heat flux in 10^20 m^-2 s^-1 eV
    pin.........int, pin from which we gather the signal. 0: Ne, 1: SE, 2: SW, 3: NW. Only
                     valid for ne, Te, Vp, Vf.
    shotnr......int, shot number
    t_start.....double, start time for signal
    t_end.......double, end time for signal
    datadir.....string, Directory with data files
    pot_var......string, The potential signal to compute the radial velocity from
                         "Vp": Use the plasma potential
                         "Vf": Use the floating potential
    vr_mode.....string, Which pin groups to use to compute the radial velocity
                        east: Vrad = (V^{SE} - V^{NE}) / B delta_z
                        west: Vrad = (V^{SW} - V^{NW}) / B delta_z
                        average: Vrad = 0.5 * (V^{SE} + V^{SW} - V^{NE} - V^{NW}) / B delta_z
    flux_var = string, Use either Isat or ne to compute the radial flux
                       "Is": Use ion saturation current signal
                       "ne": Use electron density signal

    Returns:
    ========
    tb........ndarray, float: Timebase for the waveform
    ts........ndarray, float: Signal of the waveform
    """
    assert varname in mlp_var_list
    assert pot_var in ['Vp', 'Vf']
    assert vr_mode in ['east', 'west', 'average']
    assert flux_var in ["Is", "ne"]

    invBdz = 1.0 / (4. * 2.24e-3)
    df_fname = '%s/%10d/ASP/MLP/%10d_ASP_MLP.npz' % (datadir, shotnr, shotnr)

    with np.load(df_fname) as df:
        if varname in ['Is', 'ne', 'Te', 'Vp', 'Vf']:
            varname_ts = '%s_p%1d' % (varname, pin)
            varname_tb = 'tb_' + varname_ts
            #####################################################
            # Load reference data
            ######################################################
            tb = df[varname_tb]
            ts = df[varname_ts]
            if varname is 'ne':
                ts = ts * 1e-20

            good_tidx = ((tb > t_start) & (tb < t_end))
            tb = tb[good_tidx]
            ts = ts[good_tidx]

        elif varname in ['Vrad', 'Gamma_n', 'Gamma_T']:
            ##################################################################################
            # Combine appropriate data signals as to generate the signal
            # All of these need the estimated radial velocity.
            ##################################################################################
            # Compute Vrad as average over both poloidally separated pairs:
            # Vrad = (Vf(SE, pin1) - Vf(NE, pin0) + Vf(SW, pin2) - Vf(NW, pin3)) / 2.0

            # Which voltage to use is set by vr_pot
            tb_NE = df['tb_%s_p0' % pot_var]
            ts_NE = df['%s_p0' % pot_var]

            good_tidx = ((tb_NE > t_start) & (tb_NE < t_end))
            tb_NE = tb_NE[good_tidx]
            ts_NE = ts_NE[good_tidx]
            #ts_1 = ts_1 - ts_1.mean()

            tb_SE = df['tb_%s_p1' % pot_var]
            ts_SE = df['%s_p1' % pot_var]

            good_tidx = ((tb_SE > t_start) & (tb_SE < t_end))
            tb_SE = tb_SE[good_tidx]
            ts_SE = ts_SE[good_tidx]
            #ts_2 = ts_2 - ts_2.mean()

            tb_SW = df['tb_%s_p2' % pot_var]
            ts_SW = df['%s_p2' % pot_var]

            good_tidx = ((tb_SW > t_start) & (tb_SW < t_end))
            tb_SW = tb_SW[good_tidx]
            ts_SW = ts_SW[good_tidx]
            #ts_3 = ts_3 - ts_3.mean()

            tb_NW = df['tb_%s_p3' % pot_var]
            ts_NW = df['%s_p3' % pot_var]

            good_tidx = ((tb_NW > t_start) & (tb_NW < t_end))
            tb_NW = tb_NW[good_tidx]
            ts_NW = ts_NW[good_tidx]
            #ts_4 = ts_4 - ts_4.mean()

            assert((tb_NE == tb_SE).all())
            assert((tb_NE == tb_SW).all())
            assert((tb_NE == tb_NW).all())

            assert(tb_NE.size == ts_NE.size)
            assert(tb_SE.size == ts_SE.size)
            assert(tb_SW.size == ts_SW.size)
            assert(tb_NW.size == ts_NW.size)

            tb = tb_SE
            #Vrad = 0.5 * ((ts_1 - ts_2) + (ts_3 - ts_4)) * invBdz
            #Vrad = (ts_1 - ts_2) * invBdz
            if vr_mode is 'east':
                Vrad = (ts_SE - ts_NE) * invBdz
            elif vr_mode is 'west':
                Vrad = (ts_SW - ts_NW) * invBdz
            elif vr_mode is 'average':
                Vrad = 0.5 * ((ts_SE - ts_NE) + (ts_SW - ts_NW)) * invBdz

            if varname is 'Vrad':
                ts = Vrad

            elif varname in ['Gamma_n', 'Gamma_T']:
            # Load particle density signal from all pins and average
                # Load and average particle density time series of all four pins
                ts_list = []
                for pin_idx in [0, 1, 2, 3]:
                    tb_f = df['tb_%s_p%1d' % (flux_var, pin_idx)]
                    ts = df['%s_p%1d' % (flux_var, pin_idx)] * 1e-20

                    good_tidx = ((tb_f > t_start) & (tb_f < t_end))
                    ts = ts[good_tidx]
                    ts_list.append(ts)

                    # Check if we cropped the time series to the same time as Vrad
                    assert((tb_f[good_tidx] == tb).all())
                    assert(ts.size == Vrad.size)

                # If we have the same time base, delete arrays
                avg_n = 0.25 * (ts_list[0] + ts_list[1] + ts_list[2] + ts_list[3])
                if varname is 'Gamma_n':
                    ts = avg_n * Vrad

                elif varname is 'Gamma_T':
                    # Load and average particle density time series of all four pins
                    ts_Te_list = []
                    for pin_idx in [0, 1, 2, 3]:
                        tb_T = df['tb_ne_p%1d' % pin_idx]
                        ts = df['Te_p%1d' % pin_idx]

                        good_tidx = ((tb_T > t_start) & (tb_T < t_end))
                        assert((tb_T[good_tidx] == tb).all())
                        ts = ts[good_tidx]
                        ts_Te_list.append(ts)

                        # Check if we cropped to the same time base as Vrad
                        assert(ts.size == Vrad.size)

                    avg_Te = 0.25 * (ts_Te_list[0] + ts_Te_list[1] + ts_Te_list[2] + ts_Te_list[3])
                    ts = avg_n * avg_Te * Vrad

    return tb, ts


def mlp_jdiv_get_signal(varname, electrode, shotnr, t_start, t_end, datadir=mlp_data_dir):
    """
    Construct signal from JDIV MLP datafiles.
    Stored locally by using cmodws73:misc_scripts/save_iv_jdiv_mlp.py

    Arguments:
    ==========
    varname.......string. Gives the signal to return:
                          ne: Particle density in 10^20m^-3
                          Te: Electron temperature
                          Vp: Plasma potential
                          Vf: Floating potential
    electrode.....int The electrode to read out. 0...7
    shotnr........int, shot number
    t_start.......float, start time for signal
    t_end.........float, end time for signal
    datadir.......string, Directory where data files are stored


    Returns:
    ========
    tb............ndarray, float. Timebase of the signal waveform
    ts............ndarray, float. The data time series sampled by the MLP
    """

    assert varname in mlp_var_list
    if varname in ["Vrad", "Gamma_n", "Gamma_T"]:
        print "Cannot compute %s from JDIV data" % (varname)
        return 0.0, 0.0


    df_fname = "%s/%10d/JDIV/MLP/%10d_JDIV_MLP_P%1d.npz" % (datadir, shotnr, shotnr, electrode)

    with np.load(df_fname) as df:
        varname_ts = "%s_fit" % (varname)
        varname_tb = "tb_%s_fit" % (varname)

        tb = df[varname_tb]
        ts = df[varname_ts]
        
        if varname is "ne":
            ts * ts * 1e-20


        good_tidx = ((tb > t_start) & (tb < t_end))
        tb = tb[good_tidx]
        ts = ts[good_tidx]

        df.close()

    return tb, ts


# End of file mlp_get_signal.py
