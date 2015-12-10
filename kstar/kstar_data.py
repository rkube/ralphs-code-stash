#!/opt/local/bin/python
#-*- Encoding: UTF-8 -*-

"""
Probe timing for the KSTAR data we have
"""

shot_color_dict = {13084: 'b',
                   13092: 'g',
                   13093: 'r',
                   13094: 'c',
                   13095: 'm',
                   13097: 'k'}

shot_marker_dict = {13084: 'bv',
                    13092: 'go',
                    13093: 'rs',
                    13094: 'cp',
                    13095: 'mD',
                    13097: 'k^'}


# Time for inward probe plunges in full time series
plunge_times_dict = {13084: ((3.7, 4.0), (6.7, 7.0)),
                     13092: ((3.7, 4.0), (6.7, 7.0)),
                     13093: ((3.7, 4.0), (6.7, 7.0)),
                     13094: ((3.7, 4.0), (6.7, 7.0)),
                     13095: ((3.7, 4.0), (6.7, 7.0)),
                     13097: ((3.7, 4.0), (0.0, ))}

# Line-averaged density at mid point of probe plungeo
neng_dict = {13084: (0.17, 0.20),
             13092: (0.25, 0.24),
             13093: (0.22, 0.24),
             13094: (0.38, 0.55),
             13095: (0.34, 0.50),
             13097: (0.44, )}


# Mask for single plunge time series for arcing etc.
# Include dummy tuples because python somehow removes zero dimensions in iterations :/
mask_times_dict = {13084: ((0.0, 0.0), (0.0, 0.0)),
                   13092: ((3.99526, 3.99534), (0.0, 0.0)),
                   13093: ((0.0, 0.0), (0.0, 0.0)),
                   13094: ((3.98900, 3.98902), (3.99117, 3.99122)),
                   13095: ((3.98754, 3.98763), (3.99195, 3.9925)),
                   13097: ((3.97161, 3.97562), (0.0, 0.0))}

# End of file kstar_data.py
