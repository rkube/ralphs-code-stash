#!/usr/bin/env python
#-*- Encoding: UTF-8 -*-

import numpy as np
import h5py
import matplotlib.pyplot as plt


def plot_field(df, fname='/T/0'):
    """
    Plot field with colorbar, return field as ndarray
    """
    field = df[fname].value
    plt.figure()
    plt.contourf(field)
    plt.colorbar()

    return field


def plot_field_f(df, fname='/T/0'):
    """
    Plot DFT of field with colorbar, return field as ndarray
    """
    field = df[fname].value
    field_f = np.abs(np.fft.rfft2(field))
    plt.figure()
    plt.contourf(field_f)
    plt.colorbar()

    return field_f


# End of file helpers.py
