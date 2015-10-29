#!/opt/local/bin python
# -*- Encoding: UTF-8 -*-
#
# Contains routines used to study the intermittency of  time series
#
# 1) Rescaled PDF analysis
#
# 2) Local intermittency measury
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cwt

def rescaled_pdf(signal, dt, dist, normed=True): 
    """
    Computes the histogram of the fluctuating part of the time series.

    for every d in dist plot the PDF of the signal[::d]

    Input:
    signal:   ndarray, Time series
    dt:       float, time base
    dist:     list of distances to take
    normed:   Whether signal is normalized to unity rms and zero mean
    """

    fig_hist = plt.figure()
    ax_hist = fig_hist.add_subplot(111)

    # Freeze normal distribution with unity rms and zero mean
    rv = norm()

    for d in dist:
        label_hist = r"$\tau = %4.2e$" % (d * dt)
        nelem = signal[::d].size
        print '%d: %d elements' % (d, nelem)
        hist, edges = np.histogram(signal[d::d] - signal[:-d:d], bins=int(np.sqrt(nelem)), density=True) 
        pdf_mid = edges[:-1] + 0.5 * np.diff(edges)

        ax_hist.semilogy(pdf_mid, hist, label=label_hist)

    ax_hist.semilogy(pdf_mid, norm.pdf(pdf_mid), 'k--')
    box = ax_hist.get_position()
    ax_hist.set_position([box.x0, box.y0, 0.8 * box.width, box.height])

    ax_hist.legend(loc='lower right', bbox_to_anchor=(1.4, 0.5))

    # Set the plot range
    xmin, xmax = -4.0, 4.0

    ax_hist.set_xlim((xmin, xmax))
    ymin = norm.pdf((pdf_mid > xmin) & (pdf_mid < xmax)).min()
    ymax = norm.pdf((pdf_mid > xmin) & (pdf_mid < xmax)).max()

    #ax_hist.set_ylim((ymin, ymax))
    ax_hist.set_ylim((1e-4, 1e1))

    ax_hist.set_ylabel("PDF(x(t + tau) - x(t))")
    ax_hist.set_xlabel("x")
   
    return fig_hist



def local_intermittency(signal)


# End of file intermittency.py
