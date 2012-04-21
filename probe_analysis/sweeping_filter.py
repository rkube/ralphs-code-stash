#!/usr/bin/python
#-*- Encoding: UTF-8 -*-

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt


def sweeping_filter( probe_voltage, v_max, smooth = False ):
    """
    Returns a mask for all bad voltages
    Use only values where the probe is negative enough, so that it draws the ion saturation current.
    """

    # Dummy, don't do anything yet
    if smooth == True:
        probe_voltage = probe_voltage


    return probe_voltage < v_max 
    




