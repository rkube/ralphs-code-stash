#!/usr/bin/python
#-*- Encoding: UTF-8 -*-

import numpy as np
import MDSplus as mds

"""
Scripts to get probe data from MDS tree. Implemented for MP800 where
we take data from:

    1.) ASP MLP:      get_signal_asp_mlp
    2.) JDIV MLP:     get_signal_jdiv_mlp
    3.) JDIV LP:      get_signal_jdiv
    4.) FSP:          get_signal_fsp
    5.) Rail probes:  get_signal_rail

Probes were in 2 different setups (see logbook):

    Setup A
    Setup B

In that runday the probes were in the following configuration:

    1160616007
    1160616008
    1160616009
    1160616010
    1160616011
    1160616012
    1160616013
    1160616014
    1160616015
    1160616016
    1160616017
    1160616018
    1160616019
    1160616020
    1160616021
    1160616022
    1160616023
    1160616024
    1160616025
    1160616026
"""

# Node names for ASP MLP data, used by get_signal_asp_mlp
variables_dict_mlp = {'ne': 'DENSITY_FIT',
                      'Is': 'ISAT_FIT',
                      'Js': 'JSAT_FIT',
                      'Vp': 'PHI_FIT',
                      'Te': 'TE_FIT',
                      'Vf': 'VF_FIT'}

# Node names to read for JDIV mlp data. Used by get_signal_jdiv_mlp
variables_dict_mlp_jdiv = {'ne': 'ne_fast',
                           'Is': 'ISAT_FAST',
                           'Js': 'JSAT_FAST',
                           'Vp': 'PHI_FAST',
                           'Te': 'TE_FAST',
                           'Vf': 'VF_FAST'}

# Node names for slow sampling. Used by get_signal_rail
variables_dict_slow = {'Is': 'I_SLOW',
                       'Vf': 'V_SLOW'}

# Node names for fast sampled Isat and Vfloat. Used by get_signal_fsp
variables_dict_fast = {'Is': 'I_FAST',
                       'Vf': 'V_FAST'}


def get_signal_asp_mlp(varname, pin, shotnr, t_start, t_end,
                       pot_var='Vp', vr_mode='average', flux_var='ne'):
    """
    Reads ASP MLP data from MDS.
    Arguments:
    ==========

    varname....string, gives the signal to return.
                       ne: Electron density in 10^20/m^-3
                       Is: Ion saturation current
                       Js: Jsat (unsure what the difference to Isat is)
                       Vp: Plasma potential
                       Te: Electron temperature
                       Vf: Floating potential
    pin...........int, pin 0: NE electrode, pin 1: SE, 2: SW, 3: NW.
                       Only used for ne, Te, Vp, Vf
    shotnr........int, shot number
    t_start.....float, start time for signal.
    t_end.......float, end time for signal
    pot_var....string, The potential signal from which to compute the radial
                       velocity
                       "Vp": Use plasma potential to compute velocity
                       "Vf": Use floating potential to compute velocity
    vr_mode....string, Which pin groups to use to compute radial velocity
                       "east": Vrad = (V^{SE} - V^{NE}) / B delta_z
                       "west": Vrad = (V^{SW} - V^{NW}) / B delta_z
                       "average": Vrad = 0.5* (V^{SE} + V^{SW} - V^{NE} - V^{NW}) /
                                  B delta_z
    flux_var...string, Use either Isat or ne to compute the radial flux
                       "Is" use IS_FIT signal to compute radial flux
                       "ne" use NE_FIT signal to compute radial flux

    Returns:
    ========
    tb......ndarray, float: Timebase
    ts......ndarray, float: Signal
    """
    raise Warning("Please use get_signal_asp_mlp from uit_library.cmod.get_ts_mds")

    #tree_edge = mds.Tree('edge', shotnr)
    #node = tree_edge.getNode('\EDGE::TOP.PROBES.ASP.MLP.P%1d:%s' %
    #                         (pin, variables_dict_mlp[varname]))

    #ts = node.data().astype('float64').flatten()

    #if varname is 'ne':
    #    ts = ts * 1e-20

    #tb = node.dim_of().data().flatten()
    #good_tidx = ((tb > t_start) & (tb < t_end))

    return tb[good_tidx], ts[good_tidx]


def get_signal_jdiv_mlp(varname, pin, shotnr, t_start, t_end):
    raise Warning("Please use get_signal_jdiv_mlp from uit_library.cmod.get_ts_mds")
    #tree_edge = mds.Tree('edge', shotnr)

    ## Map pins to letters
    #jdiv_map_dict = {1: 'A',
    #                 3: 'B',
    #                 5: 'C',
    #                 6: 'D',
    #                 7: 'E'}

    ## We swapped cards A and B after run 015 becuase MLP on strike point looks
    ## funky
    #if(shotnr > 1160616015):
    #    jdiv_map_dict[1] = 'B'
    #    jdiv_map_dict[2] = 'A'
    ## We swapped cards C and E after run 016 because data on card C looks funky
    #if(shotnr > 1160616016):
    #    print 'switched config!'
    #    jdiv_map_dict[7] = 'C'
    #    jdiv_map_dict[5] = 'E'

    #letter = jdiv_map_dict[pin]

    #node_name = '\EDGE::TOP.PROBES.JDIV_MLP.G%s.P0:%s' %\
    #            (letter, variables_dict_mlp[varname])
    #print node_name
    #node = tree_edge.getNode(node_name)

    #ts = node.data().astype('float64').flatten()

    #if varname is 'ne':
    #    ts = ts * 1e-20

    #tb = node.dim_of().data().flatten()
    #good_tidx = ((tb > t_start) & (tb < t_end))

    #return tb[good_tidx], ts[good_tidx]


def get_signal_jdiv(varname, pin, shotnr, t_start, t_end):
    raise Warning("Please use get_signal_jdiv from uit_library.cmod.get_ts_mds")
    #tree_edge = mds.Tree('edge', shotnr)
    #node_name = '\EDGE::TOP.PROBES.JDIV.G%02d.P0:%s' %\
    #            (pin, variables_dict_slow[varname])
    #print node_name
    #node = tree_edge.getNode(node_name)

    #ts = node.data().astype('float64').flatten()
    #tb = node.dim_of().data().flatten()
    #good_tidx = ((tb > t_start) & (tb < t_end))

    #return tb[good_tidx], ts[good_tidx]


def get_signal_fsp(varname, pin, shotnr, t_start, t_end):
    raise Warning("Please use get_signal_fsp from uit_library.cmod.get_ts_mds")
    #tree_edge = mds.Tree('edge', shotnr)
    #node = tree_edge.getNode('\EDGE::TOP.PROBES.FSP.G_1.P%1d:%s' %
    #                         (pin, variables_dict_fast[varname]))

    #ts = node.data().astype('float64').flatten()
    #tb = node.dim_of().data().flatten()
    #good_tidx = ((tb > t_start) & (tb < t_end))

    #return tb[good_tidx], ts[good_tidx]


def get_signal_rail(varname, pin, shotnr, t_start, t_end):
    raise Warning("Please use get_signal_rail from uit_library.cmod.get_ts_mds")
    #tree_edge = mds.Tree('edge', shotnr)
    #node = tree_edge.getNode('\EDGE::TOP.PROBES.RAIL.TILE%1d.P0:%s' %
    #                         (pin, variables_dict_fast[varname]))

    #ts = node.data().astype('float64').flatten()
    #tb = node.dim_of().data().flatten()
    #good_tidx = ((tb > t_start) & (tb < t_end))

    #return tb[good_tidx], ts[good_tidx]



# End of file get_ts_mds.py
