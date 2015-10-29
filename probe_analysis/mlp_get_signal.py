#!/opt/local/bin/python
#-*- Encoding: UTF-8 -*-

import numpy as np

"""
Get signal from MLP.
Data taken from MDS tree according to ~/misc_scripts/save_iv_profile_mlp.py

* Radial velocity, V_rad, is averaged over both pairs of poloidally separated bins
* Radial particle flux, Gamma_n, is V_rad * average of all 4 particle densities
* Radial heat flux, Gamma_T, is V_rad * average of all 4 particle densities and all 4 temperatures
"""

mlp_var_list = ['ne', 'Te', 'Vp', 'Vf', 'Is', 'Vrad', 'Gamma_n', 'Gamma_T']
mlp_data_dir = '/Volumes/ekgroup/cmod_data/'

def mlp_get_signal(varname, pin, shotnr, t_start, t_end, datadir=mlp_data_dir):
    """
        Construct signal from MLP datafile

        varname:    string, gives the signal to return: 
                                ne: Particle density in 10^20 m^-3
                                Te: Electron temperature in eV
                                Vp: Plasma potential in Volt
                                Vf: Floating potential in Volt
                                Vrad: Radial velocity in ms^-1
                                Gamma_n: Radial particle flux in 10^20 m^-2 s^-1
                                Gamma_T: Radial heat flux in 10^20 m^-2 s^-1 eV
        pin:        int, pin from which we gather the signal. 0: Ne, 1: SE, 2: SW, 3: NW. Only
                         valid for ne, Te, Vp, Vf.
        shotnr:     int, shot number
        t_start:    double, start time for signal
        t_end:      double, end time for signal
        datadir:    string, Directory with data files

    """
    assert varname in mlp_var_list

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
            # Compute Vrad as aerage over both poloidally separated pairs:
            # Vrad = (Vf(SE, pin1) - Vf(NE, pin0) + Vf(SW, pin2) - Vf(NW, pin3)) / 2.0
            #
            tb_1 = df['tb_Vf_p1']
            ts_1 = df['Vf_p1']

            good_tidx = ((tb_1 > t_start) & (tb_1 < t_end))
            tb_1 = tb_1[good_tidx]
            ts_1 = ts_1[good_tidx]

            tb_2 = df['tb_Vf_p0']
            ts_2 = df['Vf_p0']

            good_tidx = ((tb_2 > t_start) & (tb_2 < t_end))
            tb_2 = tb_2[good_tidx]
            ts_2 = ts_2[good_tidx]

            tb_3 = df['tb_Vf_p2']
            ts_3 = df['Vf_p2']

            good_tidx = ((tb_3 > t_start) & (tb_3 < t_end))
            tb_3 = tb_3[good_tidx]
            ts_3 = ts_3[good_tidx]

            tb_4 = df['tb_Vf_p3']
            ts_4 = df['Vf_p3']

            good_tidx = ((tb_4 > t_start) & (tb_4 < t_end))
            tb_4 = tb_4[good_tidx]
            ts_4 = ts_4[good_tidx]

            assert((tb_1 == tb_2).all())
            assert((tb_1 == tb_3).all())
            assert((tb_1 == tb_4).all())

            assert(tb_1.size == ts_1.size)
            assert(tb_2.size == ts_2.size)
            assert(tb_3.size == ts_3.size)
            assert(tb_4.size == ts_4.size)

            tb = tb_1
            Vrad = 0.5 * ((ts_1 - ts_2) + (ts_3 - ts_4)) * invBdz

            if varname is 'Vrad':
                ts = Vrad


            elif varname in ['Gamma_n', 'Gamma_T']:
            # Load particle density signal from all pins and average
                # Load and average particle density time series of all four pins
                ts_ne_list = []
                for pin_idx in [0, 1, 2, 3]:
                    tb_n = df['tb_ne_p%1d' % pin_idx] 
                    ts = df['ne_p%1d' % pin_idx] * 1e-20

                    good_tidx = ((tb_n > t_start) & (tb_n < t_end))
                    ts = ts[good_tidx]
                    ts_ne_list.append(ts)

                    # Check if we cropped the time series to the same time as Vrad
                    assert((tb_n[good_tidx] == tb).all())
                    assert(ts.size == Vrad.size)

                # If we have the same time base, delete arrays
                avg_ne = 0.25 * (ts_ne_list[0] + ts_ne_list[1] + ts_ne_list[2] + ts_ne_list[3])
                if varname is 'Gamma_n':
                    ts = avg_ne * Vrad

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
                    ts = avg_ne * avg_Te * Vrad

    return tb, ts



# End of file mlp_get_signal.py
