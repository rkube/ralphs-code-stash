#!/usr/bin/env python
#-*- Encoding: UTF-8 -*-

from scipy.io import idlread



"""
Detrend ion saturation timeseries from langmuir probes

Load data stored in IDL file
Use filters on scales of ca. 100 times the autocorreltaion time of the signal
Detrend the signal by subtracting the running mean and dividing by the running rms
"""


shotnr = 1111208006
datadir = '/Volumes/Backup/cmod_data/ASP/data/'
filename = datadir + '%10d_raw_ASP.sav' % ( shotnr )








