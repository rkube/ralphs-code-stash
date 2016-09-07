#!/opt/local/bin/python
#-*- Encoding: UTF-8 -*-

"""
Probe timing for the KSTAR data
"""

# Use these colors for the shots
shot_color_dict = {13084: 'k',
                   13092: 'm',
                   13093: 'c',
                   13094: 'r',
                   13095: 'b',
                   13097: 'g'}

# Use these shot markers for the papaer, as discussed on 2016-04-26. 
# (low density) ‘kv', ‘mD', ‘bo', ‘gs', 'r^' (high density). Unused markers are assigned 'co'
shot_marker_dict = {13084: ('kv', 'kp'),
                    13092: ('mD', 'mp' ),
                    13093: ('co', 'cp' ),
                    13094: ('rs', 'r^' ),
                    13095: ('bo', 'bp'),
                    13097: ('gs', 'gp' )}


# Time for inward probe plunges in full time series
plunge_times_dict = {13084: ((3.85, 4.0), (6.85, 7.0)),
                     13092: ((3.85, 4.0), (6.85, 7.0)),
                     13093: ((3.85, 4.0), (6.85, 7.0)),
                     13094: ((3.85, 4.0), (6.85, 7.0)),
                     13095: ((3.85, 4.0), (6.85, 7.0)),
                     13097: ((3.85, 4.0), (0.0, ))}

plunge_times_out_dict = {13084: ((4.0, 4.15), (7.0, 7.15)),
                         13092: ((4.0, 4.15), (7.0, 7.15)),
                         13093: ((4.0, 4.15), (7.0, 7.15)),
                         13094: ((4.0, 4.15), (7.0, 7.15)),
                         13095: ((4.0, 4.15), (7.0, 7.15)),
                         13097: ((4.0, 4.15), (0.0))}

# Line-averaged density at mid point of probe plungeo
neng_dict = {13084: (0.17, 0.20),
             13092: (0.25, 0.24),
             13093: (0.22, 0.24),
             13094: (0.38, 0.55),
             13095: (0.34, 0.50),
             13097: (0.44, )}


# Mask for single plunge time series for arcing etc.
# Include dummy tuples because python somehow removes zero dimensions in iterations :/
mask_times_dict = {13084: ((0.0, 0.0), ),
                   13092: ((3.99525, 3.99534), ),
                   13093: ((6.9741, 6.97423), ),
                   13094: ((3.989728, 3.98731), (3.98898, 3.98903), (3.99109, 3.99205), (3.99323, 3.9939)),
                   13095: ((3.98754, 3.98763), (3.99194, 3.99215)),
                   13097: ((3.97161, 3.97562), )}

# Mask for single outwards plunges of the probe
mask_times_out_dict = {13084: ((0.0, 0.0), ),
                       13092: ((0.0, 0.0), ),
                       13093: ((0.0, 0.0), ),
                       13094: ((0.0, 0.0), ),
                       13095: ((4.0055, 4.0093), ),
                       13097: ((0.0, 0.0), )}


# Interval on which we compute profiledata 
radial_variable_list = ['R', 'rsep']
profile_R_min_max = (2.22, 2.3)
profile_rrsep_min_max = (0.0225, 0.075)

# End of file kstar_data.py
