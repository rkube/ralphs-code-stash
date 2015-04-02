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



def parseval_rfft2(arr_c):
    """
    Compute sum_k |f_hat(k)|^2
    , taking data layout of real-to-complex dft into consideration:
    zero- and nyquist frequency are real.
    dft_r2c has hermitian symmetry: f_k = f_(N-k).conj() -> count f_k.abs() twice
    """

    sum = 0.0
    for m in np.arange(arr_c.shape[0]):
        for n in np.arange(arr_c.shape[0]):
            if n < arr_c.shape[1]:
                sum += np.abs(arr_c[m, n]) * np.abs(arr_c[m, n])
            else:
                idx0 = (arr_c.shape[0] - m) % arr_c.shape[0]
                idx1 = arr_c.shape[0] - n
                sum += np.abs(arr_c[idx0, idx1]) * np.abs(arr_c[idx0, idx1])
                # sum += np.abs(arr_c[arr_c.shape[0] - m, arr_c.shape[0] - n]) * np.abs(arr_c[arr_c.shape[0] - m, arr_c.shape[0] - n])

    sum /= float(arr_c.shape[0] * arr_c.shape[0])

    return sum


# End of file helpers.py
