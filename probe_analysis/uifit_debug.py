#!/usr/bin/env python
#-*- Encoding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from probe_analysis.uifit import call_uifit


class FitException(BaseException):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


fit_func = lambda U, a, b, c: a + b * np.exp(U / c)


def sort_IV(I_raw, U_raw):
    """
    Sort input data from positive to negative currents
    """

    I_data = np.zeros_like(I_raw)
    V_data = np.zeros_like(U_raw)

    sort_idx = I_raw.argsort()
    I_data[:] = I_raw[sort_idx][::-1]
    V_data[:] = V_raw[sort_idx][::-1]

    return I_data, V_data


def do_UI_fit(U_data, I_data, one_over_probeA, Isat_est,
              nVknee=3, Ie_ratio=2.0, min_Ie_ratio=0.5, it=0):

    """
    Given current and voltage on a langmuir probe, attempt a range
    of fits of the data on the model
    I(V) = Isat + Ib * exp(V/Te)

    The routine attempts a series of fits on the intervall
    [min(V) : V_knee], where min(V) is the minimal (negative!) value of V

    The knee voltage is given for each fit as the voltage which
    yields the maximal current Ie_ratio.
    I.e. we attempt nVknee fits on the intervall [I_min : -I_max]
    determined by
    I_max = min_ie_ratio : ie_ratio

    Return the parameters of the fit with the smallest error on
    the electron temperature

    """

    #
    #####################################################################
    # Step 2: Set up nVknee possible input data sets,
    # spanning from I_data = -min_Ie_ratio : -2 * Ie_ratio
    ####################################################################
    #

    nelem = I_data.size - 1
    nVknee = max(nVknee, 3)
    #iend = np.zeros(nVknee, dtype='i4')
    Ie_set = np.ones(nVknee, dtype='float64')

    # Set up Ie_ratio for first fit. Fit only on I values exceeding Ie_ratio
    Ie_ratio = min(Ie_ratio, -0.5 * I_data.min() / Isat_est)
    print "nelem = %d, nVknee = %d, Ie_ratio = %e" % (nelem, nVknee, Ie_ratio)

    print 'Fit boundaries (units of Isat):'
    Ie_set = np.linspace(-min_Ie_ratio, -2 * Ie_ratio, nVknee) * Isat_est
    print Ie_set

    # Check if the intervall defined by each Ie contains more points
    # than points were used to guess Isat
    num_fit_pts = np.zeros(nVknee, dtype='i4')
    #for idx, s in enumerate(Ie_set):
    #    num_fit_pts[idx] = int((I_data > s).sum())
    for idx in np.arange(nVknee):
        num_fit_pts[idx] = int((I_data > Ie_set[idx]).sum())
    print 'Number of points to fit:'
    print num_fit_pts

    # Abort if less then 3 fit ranges are identified that exceed th number of
    # points used to estimate the ion saturation current
    if ((num_fit_pts > npts_Isat).sum() < 4):
        npts_exceed = (num_fit_pts > npts_Isat).sum()
        err_str = "Number of points used to estimate isat = %d\n" % (npts_Isat)
        err_str = "Number of fit ranges exceeding npts_Isat: %d\n" %\
            (npts_exceed)
        err_str = "Increase Ie_ratio (passed: %f)\n" % (Ie_ratio)
        raise FitException(err_str)

    #
    #####################################################################
    # Step 3: Fit all possible datasets using FastExpRange0
    ####################################################################
    #
    #TeBounds = 149.
    #Te_guess = 10.0
    res = call_uifit(V_data, I_data, num_fit_pts)
    isat_set, sigma_is_set, ib_set, sigma_ib_set, Te_set, sigma_Te_set, chi2_set, err_set = res

    #for idx, npts in enumerate(num_fit_pts):
    for idx in np.arange(nVknee):
        npts = num_fit_pts[idx]
        plt.figure()
        plt.plot(V_data, I_data, 'k.')
        plt.plot(V_data[:npts], I_data[:npts], 'r.')
        plt.plot(V_data[:npts], fit_func(V_data[:npts], isat_set[idx],
                                         ib_set[idx], Te_set[idx]), 'g.')

    #
    #####################################################################
    # Step 4: Select best fit based on sigmaTe
    ####################################################################
    #

    # Good fits require err=0 and Isat > 0. Weed out bad fits
    # and exit if all fits are bad
    good_fit_idx = (err_set == 0) & (isat_set > 0.0)
    if(good_fit_idx.sum() == 0):
        err_str = 'No good fits (err != 0). aborting\n'
        raise FitException(err_str)

    best_Te_idx = sigma_Te_set.argmin()
    isat = isat_set[best_Te_idx]
    sigma_isat = sigma_is_set[best_Te_idx]
    ib = ib_set[best_Te_idx]
    sigma_ib = sigma_ib_set[best_Te_idx]
    Te = Te_set[best_Te_idx]
    sigma_Te = sigma_Te_set[best_Te_idx]
    RChi2 = chi2_set[best_Te_idx]
    npts = num_fit_pts[best_Te_idx]

    bad_fit_coeff = (isat < 0.0) | (ib > 0.0)

    if bad_fit_coeff:
        err_str = 'FastExpRange returned\n'
        err_str += 'Isat = %f (must be >= 0.0)\n' % (isat)
        err_str += 'Ib = %f (must be < 0.0\n' % (ib)
        err_str += 'Te = %f\n' % (Te)

        raise FitException(err_str)

    return (isat, sigma_isat, ib, sigma_ib, Te, sigma_Te, RChi2, npts)


def remove_linear_trend(raw_U, raw_I, isat_seg_idx, V_float, Te):

    """
    If exising, remove a linear trend on the current in the region
    where the probe is in ion saturation current.
    """

    # Compute I_fit fir isat_seg to see if it is good for trend removal
    isat_V = raw_U[isat_seg_idx]
    isat_I = raw_I[isat_seg_idx]
    isat_trend = isat + ib * np.exp(isat_V / Te)
    isat_norm = isat_I / isat_trend
    isat_RMS = np.sqrt(((isat_norm - 1.0) ** 2.0).sum()) / isat_norm.size

    print 'From exponential model:'
    print 'isat_rms = %f' % isat_RMS

    # Compute linear trend on isat region
    A = np.vstack([isat_V, np.ones_like(isat_V)]).T
    m, n = np.linalg.lstsq(A, isat_I)[0]

    print 'Linear fit: m = %f, n = %f' % (m, n)

    if m <= 0.0:
        #
        # Negative linear trend found, use it
        #
        isat_trend2 = m * isat_I + n
        isat_norm2 = isat_I / isat_trend2
        isat_RMS2 = (np.sqrt(((isat_norm2 - 1.0) ** 2.0).sum()) /
                     isat_norm2.size)
        good_fit = True
        print 'From linear model:'
        print 'isat_rms = %f' % isat_RMS2
    else:
        #
        # Positive trend found, use mean as trend
        #
        isat_trend2 = isat_I.mean()
        if(isat_trend2 < 0.0):
            err_str = "Using isat_trend2 = mean gives negative trend"
            raise FitException(err_str)
        isat_norm2 = isat_I / isat_trend2
        isat_RMS2 = (np.sqrt(((isat_norm2 - 1.0) ** 2.0).sum()) /
                     isat_norm2.size)
        print 'From linear model (using mean):'
        print 'isat_rms = %f' % isat_RMS2

    return (isat_RMS, isat_norm, isat_trend,
            isat_RMS2, isat_norm2, isat_trend2,
            good_fit)

#
#
#
#
#
#
#
npts_Isat = 100
Ie_ratio = 2.0
nVknee = 5
do_remove_linear_trend = True
loop_max = 0
if(do_remove_linear_trend):
    loop_max = 3
loop = 0

#df = np.load('/Users/ralph/source/py_fortran_tests/ui_sample_data_3.npz')
df = np.load('/Users/ralph/uni/cmod/tmp_data/ui_sample_data_11.npz')
V_raw = df['U']
I_raw = df['I']

plt.figure()
plt.plot(V_raw, I_raw, 'k.')
plt.show()

if (npts_Isat > I_raw.size):
    err_str = "npts_Isat is larger than the total data set: "
    err_str += "%d > %d" % (npts_Isat, I_raw.size)
    raise FitException(err_str)

one_over_probeA = np.ones_like(I_raw)

while True:
    #
    #####################################################################
    # Step -1: Remove linear Isat trend on subsequent passes
    ####################################################################
    #
    curr = I_raw * one_over_probeA

    #
    #####################################################################
    # Step 0: Estimate Isat
    ####################################################################
    #
    if (V_raw[0] > V_raw[-1]):
        Isat_est = curr[-npts_Isat:].mean()
    elif (V_raw[0] < V_raw[-1]):
        Isat_est = curr[:npts_Isat].mean()

    if(Isat_est < 1e-6):
        err_str = "Isat estimate = %e < 0.0. Isat has to be positive" %\
            (Isat_est)
        raise FitException(err_str)
    #
    ######################################################################
    # Step 2: Sort input data from positive to negative currents
    ######################################################################
    #
    curr_data, V_data = sort_IV(curr, V_raw)

    #
    ######################################################################
    # Step 3: Fit all possible datasets using FastExpRange0
    # Step 4: Select best fit based on smalles sigmaTe
    ######################################################################
    #

    res = do_UI_fit(V_raw, curr_data, np.ones_like(I_raw), Isat_est,
                    Ie_ratio=Ie_ratio, nVknee=nVknee)
    isat, sigma_isat, ib, sigma_ib, Te, sigma_Te, RChi2, npts = res

    idx_range = np.arange(npts)
    Vknee = V_data[idx_range].max()
    I_positive = (curr_data[idx_range] > 0.0).sum()
    I_negative = (curr_data[idx_range] < 0.0).sum()
    F_electron = float(I_negative) / idx_range.size
    Iknee = curr_data[idx_range].min()
    Ie_ratio_used = -Iknee / Isat_est

    #
    ##################################################################
    # Step 5: Determine ion saturation segment: voltages <= Vfloat - Te
    ##################################################################
    #

    V_float = Te * np.log(-1.0 * isat / ib)

    print '======================================================='
    print 'Trend removal'
    print ''

    if do_remove_linear_trend:
        V_cutoff = V_float - 0.5 * Te
    else:
        V_cutoff = V_float - Te
    V_cutoff = min(V_cutoff, V_float - 5)
    V_cutoff = max(V_cutoff, V_float - 30)
    print 'loop = %d, V_cutoff = %f' % (loop, V_cutoff)

    # find the segment where V < v_cutoff
    v_cut_idx = np.where(V_raw < V_cutoff)[0]
    if(v_cut_idx.size < 10):
        raise FitException("Less than 10 datapoins found with V < Vf_Te\n")

    isat_seg_idx = np.arange(v_cut_idx[0], v_cut_idx[-1])

    res = remove_linear_trend(V_raw, I_raw, isat_seg_idx, V_float, Te)
    isat_RMS = res[0]
    isat_norm = res[1]
    isat_trend = res[2]
    isat_RMS2 = res[3]
    isat_norm2 = res[4]
    isat_trend2 = res[5]
    good_fit = res[6]

    #
    # Comare trend2 to the trend from the exponential model.
    # If it is better, use it
    if (isat_RMS2 < 1.5 * isat_RMS):
        isat_RMS = isat_RMS2
        isat_norm = isat_norm2
        isat_trend = isat_trend2
        if good_fit:
            linear_fit_better = True
        one_over_probeA[isat_seg_idx] = isat_trend2.min() / isat_trend2

    loop += 1
    plt.figure()
    plt.plot(one_over_probeA)
    plt.title('loop = %d' % loop)

    if(loop >= loop_max):
        break


plt.show()
#End of file uifit_debug.py
