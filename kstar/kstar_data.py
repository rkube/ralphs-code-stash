#!/opt/local/bin/python
#-*- Encoding: UTF-8 -*-

"""
Probe timing for the KSTAR data we have
"""


# Time for inward probe plunges in full time series
plunge_times_dict = {13084: ((3.7, 4.0), (6.7, 7.0)),
                     13092: ((3.7, 4.0), (6.7, 7.0)),
                     13093: ((3.7, 4.0), (6.7, 7.0)),
                     13094: ((3.7, 4.0), (6.7, 7.0)),
                     13095: ((3.7, 4.0), (6.7, 7.0)),
                     13097: ((3.7, 4.0), (0.0, ))}

# Mask for double plunge time series, arcing etc.
mask_time_dict = {13084: (), 
                  13092: ((3.99530, 3.99533), ),
                  13094: ((3.98900, 3.98902), (3.99118, 3.99204), (3.99341, 3.99384))}
                  


# Mask for single plunge time series for arcing etc.
# Include dummy tuples because python somehow removes zero dimensions in iterations :/
mask_times_dict = {13084: ((0.0, 0.0), (0.0, 0.0)),
                   13092: ((3.99526, 3.99531), (0.0, 0.0)),
                   13093: ((0.0, 0.0), (0.0, 0.0)),
                   13094: ((3.98900, 3.98902), (3.99117, 3.99122)),
                   13095: ((3.98754, 3.98763), (3.99196, 3.992)),
                   13097: ((3.97161, 3.97562), (0.0, 0.0))}

# End of file kstar_data.py
