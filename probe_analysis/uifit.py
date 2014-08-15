#!/usr/bin/env python
#-*- Encoding: UTF-8 -*-

import numpy as np
import ctypes
import matplotlib.pyplot as plt
from misc.smoothing import smooth
from scipy.stats import skew, kurtosis


# Define an error class that is thrown if a fit fails:

class FitException(BaseException):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


def ui_fun(U, Isat, Vfloat, Te):
    """Theoretical U-I curve"""
    return Isat * (np.exp((U - Vfloat) / Te) - 1.0)


fit_func = lambda U, a, b, c: a + b * np.exp(U / c)


def sort_IV(I_raw, V_raw):
    """
    Sort input data from positive to negative currents
    """

    I_data = np.zeros_like(I_raw)
    V_data = np.zeros_like(V_raw)

    sort_idx = I_raw.argsort()
    I_data[:] = I_raw[sort_idx][::-1]
    V_data[:] = V_raw[sort_idx][::-1]

    return I_data, V_data


def do_UI_fit(volt, I_data, Isat_est, npts_Isat,
              nVknee=3, Ie_ratio=2.0, min_Ie_ratio=0.5, it=0,
              show_plot=False, debug=False):

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
    Ie_set = np.linspace(-min_Ie_ratio, -2 * Ie_ratio, nVknee) * Isat_est

    # Check if the intervall defined by each Ie contains more points
    # than points were used to guess Isat
    num_fit_pts = np.zeros(nVknee, dtype='i4')
    for idx in np.arange(nVknee):
        num_fit_pts[idx] = int((I_data > Ie_set[idx]).sum())

    if(debug):
        print "nelem = %d, nVknee = %d, Ie_ratio = %e" % (nelem, nVknee,
                                                          Ie_ratio)
        print 'Fit boundaries (units of Isat):', Ie_set
        print 'Number of points to fit: ', num_fit_pts

    # Abort if less then 3 fit ranges are identified that exceed th number of
    # points used to estimate the ion saturation current
    if ((num_fit_pts > npts_Isat).sum() < 4):
        npts_exceed = (num_fit_pts > npts_Isat).sum()
        err_str = "Number of points used to estimate Isat = %d\n" % (npts_Isat)
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
    res = call_uifit(volt, I_data, num_fit_pts)
    Isat_set = res[0]
    sigma_Is_set = res[1]
    Ib_set = res[2]
    sigma_Ib_set = res[3]
    Te_set = res[4]
    sigma_Te_set = res[5]
    chi2_set = res[6]
    err_set = res[7]

    #
    #####################################################################
    # Step 4: Select best fit based on sigmaTe
    ####################################################################
    #

    # Good fits require err=0 and Isat > 0. Weed out bad fits
    # and exit if all fits are bad
    good_fit_idx = (err_set == 0) & (Isat_set > 0.0)
    if(good_fit_idx.sum() == 0):
        err_str = 'No good fits (err != 0). aborting\n'
        raise FitException(err_str)

    best_Te_idx = sigma_Te_set.argmin()
    Isat = Isat_set[best_Te_idx]
    sigma_Isat = sigma_Is_set[best_Te_idx]
    Ib = Ib_set[best_Te_idx]
    sigma_Ib = sigma_Ib_set[best_Te_idx]
    Te = Te_set[best_Te_idx]
    sigma_Te = sigma_Te_set[best_Te_idx]
    RChi2 = chi2_set[best_Te_idx]
    npts = num_fit_pts[best_Te_idx]

    bad_fit_coeff = (Isat < 0.0) | (Ib > 0.0)

    if bad_fit_coeff:
        err_str = 'FastExpRange returned\n'
        err_str += 'Isat = %f (must be >= 0.0)\n' % (Isat)
        err_str += 'Ib = %f (must be < 0.0\n' % (Ib)
        err_str += 'Te = %f\n' % (Te)
        raise FitException(err_str)

    fig = None
    if show_plot:
        volt_sort = np.zeros_like(volt)
        volt_sort[:] = volt[:]
        volt_sort.sort()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(volt, I_data, 'k.')
        ax.set_xlabel('voltage / V')
        ax.set_ylabel('current / A')
        str_title = 'loop = %d, isat_est = %4.2e' % (it, Isat_est)
        ax.set_title(str_title)
        for idx in np.arange(nVknee):
            label_fit = 'npts=%d, Isat=%4.2e, Ib=%4.2e, Te=%4.2e' %\
                (npts, Isat_set[idx], Ib_set[idx], Te_set[idx])
            label_fit += ', sigma_Te = %4.2f' % sigma_Te_set[idx]
            npts = num_fit_pts[idx]

            if(idx == best_Te_idx):
                lw = 3
                label_fit += ' ***best'
            else:
                lw = 1
            npts = num_fit_pts[idx]

            ax.plot(volt_sort[:npts], fit_func(volt_sort[:npts], Isat_set[idx],
                                               Ib_set[idx], Te_set[idx]),
                    label=label_fit, lw=lw)
        ax.legend(loc='lower left')

    return (Isat, sigma_Isat, Ib, sigma_Ib, Te, sigma_Te, RChi2, npts, fig)


def remove_linear_trend(V_raw, I_raw, Isat_seg_idx, Isat, ib, V_float, Te,
                        loop=0, show_plot=False, debug=False):

    """
    Remove a linear trend in the current region
    If exising, remove a linear trend on the current in the region
    where the probe is in ion saturation mode
    """

    good_fit = False

    ##################################################################
    # Determine current and voltage range in which we are in ion saturation
    # regime
    ########################################################################
    #
    V_Isat = V_raw[Isat_seg_idx]
    I_Isat = I_raw[Isat_seg_idx]
    #print 'I_Isat.mean() = %f' % I_Isat.mean()
    Isat_trend = Isat + ib * np.exp(V_Isat / Te)
    Isat_norm = I_Isat / Isat_trend
    Isat_RMS = np.sqrt(((Isat_norm - 1.0) ** 2.0).sum()) / Isat_norm.size
    label_exp = 'Exponential: Isat_rms = %4.2e' % (Isat_RMS)

    #
    ##################################################################
    # Attempt a linear fit on the current in the ion sat. regime
    ##################################################################
    #
    try:
        A = np.vstack([V_Isat, np.ones_like(V_Isat)]).T
        m, n = np.linalg.lstsq(A, I_Isat)[0]
    except np.LinAlgError:
        if(debug):
            print '******** uifit.remove_linear_trend():'
            print '******** Linear fit in Isat regime failed.'
            print '******** Using mean to remove trend'
        m = 1.0
        n = 0.0

    if m < -1e-6:
        #
        ##################################################################
        # If we have a negative slope, assume it is a good fit and use it for
        # trend remove of the sheath expansion
        ##################################################################
        #
        Isat_trend2 = m * I_Isat + n
        Isat_norm2 = I_Isat / Isat_trend2
        Isat_RMS2 = (np.sqrt(((Isat_norm2 - 1.0) ** 2.0).sum()) /
                     Isat_norm2.size)
        label_lin = 'Linear (slope): Isat_rms = %4.2e' % (Isat_RMS2)
        good_fit = True
    else:
        #
        ##################################################################
        # If we have a positive slope, giv values to remove the trend with
        # the mean current in the ion saturation segment
        ####################################################################
        #
        Isat_trend2 = I_Isat.mean() * np.ones_like(I_Isat)
        if(Isat_trend2[0] < 0.0):
            # We should not have a negative ion saturation current!
            err_str = "Using Isat_trend2 = I_Isat.mean() gives negative trend"
            raise FitException(err_str)
        Isat_norm2 = I_Isat / Isat_trend2
        Isat_RMS2 = (np.sqrt(((Isat_norm2 - 1.0) ** 2.0).sum()) /
                     Isat_norm2.size)
        label_lin = 'Linear (mean): Isat_rms = %4.2e' % (Isat_RMS2)

    fig = None
    if(show_plot):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        str_title = 'loop: %d Linear trend removal' % (loop)
        str_title = str_title + ' V_cutoff = %5.1f' % (V_Isat.max())
        ax.set_title(str_title)
        ax.plot(V_raw, I_raw, 'k.')
        ax.plot(V_Isat, I_Isat, 'r.')
        ax.plot(V_Isat, Isat_trend, 'b', label=label_exp)
        ax.plot(V_Isat, Isat_trend2, 'g', label=label_lin)
        ax.legend(loc='lower left')

    return (Isat_RMS, Isat_norm, Isat_trend,
            Isat_RMS2, Isat_norm2, Isat_trend2,
            good_fit, fig)


#
#
#
#
#
#
def FitIV_MinSigTe(V_raw, I_raw, npts_Isat=100, Ie_ratio=2.0, nVknee=5,
                   do_remove_linear_trend=True,
                   Te_min=10.0, Te_max=150.0, TeAcc=1.0, MaxIterTe=1000,
                   show_plot=False, debug=False):
    """
    FitIV_MinSigTE
    ==============
    This procedure fits Isat and Te to the I-V probe characteristic
    I(V) = Isat + Ib * exp(V/Te)

    I: probe current
    V: probe voltage

    Input:
    ======
    V_raw      - Voltage array, np.ndarray(dtype='float64')
    I_raw      - Current array, np.ndarray(dtype='float64')
    npts_Isat  - Number of points for initial guess on Isat, int
    Ie_ratio   - Maximum -I/I_sat for fitting, float64
    nVknee     - Number of possible values for upper fit limit
    do_remove_linear_trend
    TeMin      - Minimum electron temperature
    TeMax      - Maximum electron temperature
    TeAcc      - Accuracy of electron temperature fit coefficient
    MaxIterTe  - Maximum iteration for fitting routine
    show_plot  - Plot I-V data set, all fits and trend removal

    Output:
    =======


    """

    # Loop three times if we are tasked to remove a linear trend in the
    # ion saturation regime
    loop_max = (0, 3)[do_remove_linear_trend]
    loop = 0

    # Estimate the ion saturation current on less data points than the
    # entire data set
    assert(npts_Isat < I_raw.size)

    one_over_probeA = np.ones_like(I_raw)

    figure_list = None
    if (show_plot):
        figure_list = []

    while True:
        #
        #####################################################################
        # Step -1: Remove linear Isat trend on subsequent passes
        ####################################################################
        #
        linear_fit_better = False
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

        # Assert that we have a sufficiently negative Isat estimate.
        assert(Isat_est > 1e-6)

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

        res = do_UI_fit(V_data, curr_data, Isat_est, npts_Isat, nVknee=nVknee,
                        Ie_ratio=Ie_ratio, it=loop, show_plot=show_plot,
                        debug=debug)
        Isat = res[0]
        sigma_Isat = res[1]
        Ib = res[2]
        sigma_Ib = res[3]
        Te = res[4]
        sigma_Te = res[5]
        RChi2 = res[6]
        npts = res[7]
        if (show_plot):
            figure_list.append(res[8])

        idx_range = np.arange(npts)
        Vknee = V_data[idx_range].max()
        #I_positive = (curr_data[idx_range] > 0.0).sum()
        I_negative = (curr_data[idx_range] < 0.0).sum()
        F_electron = float(I_negative) / idx_range.size
        Iknee = curr_data[idx_range].min()
        Ie_ratio_used = -Iknee / Isat_est

        #
        ##################################################################
        # Step 5: Determine ion saturation segment: voltages <= Vfloat - Te
        ##################################################################
        #

        V_float = Te * np.log(-1.0 * Isat / Ib)
        if(debug):
            print '======================================================='
            print 'U-I Fit: Isat = %4.2e Vfloat = %4.2e, Te = %4.2e' %\
                (Isat, V_float, Te)
            print '======================================================='
            print 'Trend removal'
            print ''

        # Take V_cutoff = V_float - 0.5 * Te : remove_linear_trend = True
        #      V_cutoff = V_float - 1.0 * Te : remove_linear_trend = False
        factor = (-1.0, -1.75)[do_remove_linear_trend]
        V_cutoff = V_float + factor * Te

        V_cutoff = min(V_cutoff, V_float - 5)
        V_cutoff = max(V_cutoff, V_float - 50)
        if(debug):
            print 'loop = %d, V_cutoff = %f' % (loop, V_cutoff)

        # find the segment where V < v_cutoff
        # Assume that this segment is longer than 10 data points, so that
        # statistics computed on it make sense. If not, we are out
        v_cut_idx = np.where(V_raw <= V_cutoff)[0]
        if (v_cut_idx.size < 10):
            break

        Isat_seg_idx = np.arange(v_cut_idx[0], v_cut_idx[-1])

        res = remove_linear_trend(V_raw, I_raw, Isat_seg_idx, Isat,
                                  Ib, V_float, Te, show_plot=show_plot,
                                  debug=debug)
        Isat_RMS, Isat_norm, Isat_trend = res[0:3]
        Isat_RMS2, Isat_norm2, Isat_trend2 = res[3:6]
        good_fit = res[6]

        #
        # Comare trend2 to the trend from the exponential model.
        # If it is better, use it
        if (Isat_RMS2 < 1.5 * Isat_RMS):
            Isat_RMS = Isat_RMS2
            Isat_norm = Isat_norm2
            Isat_trend = Isat_trend2
            linear_fit_better = (False, True)[good_fit]
            one_over_probeA[Isat_seg_idx] = Isat_trend2.min() / Isat_trend2

        loop += 1
        if(loop >= loop_max or not linear_fit_better):
            break

    # See if the fit is in good bounds
    if (Te + sigma_Te >= Te_max):
        err_str = "Te exceeds maximum allowed"
        err_str = "Te = %f, TeAcc= %f, TeMax = %f" % (Te, sigma_Te, Te_max)
        raise FitException(err_str)

    if (Te - sigma_Te <= Te_min):
        err_str = "Te under minimum allowed"
        err_str = "Te = %f, TeAcc= %f, TeMin = %f" % (Te, sigma_Te, Te_min)
        raise FitException(err_str)

    # Return all fit values
    return (Isat, sigma_Isat, V_float, Te, sigma_Te, Vknee, Isat_est, Ib,
            sigma_Ib, F_electron, Ie_ratio_used, RChi2, Isat_RMS, Isat_seg_idx,
            Isat_norm, Isat_trend, figure_list)


def uifit_minsigTE_stat(U, I, fit_min=100, fit_max=300, fit_step=10,
                        probe_a=9.6e-7, n0=1e13, eps_isat_trend=1e-4,
                        num_isat_shrink=10, return_plot=False,
                        silent=True):
    """
    Fit a U-I characteristic on the profile using the min Te method.

    1.) Estimate V_float by smoothing the profile
    2.) Starting from V_float, nfits fits on the U-I characteristic, finding
        the minimum T_e value
    3.) Using the T_e value, compensate for a varying sheath size with
        increasingly negative V_probe (Hutchinson, (3.2.27))
    3a) Alternatively, compensate for a linear trend in the Isat segment by
        subtracting a linear fit


    Input:
        U:                  Probe voltage
        I:                  Probe current
        nfit:               Number of points to fit
        interval:           Number of points to increase fit interval
                            to find Te
        eps_isat_trend:     Maximal trend in isat region
        num_isat_shrink:    Shrink isat region iteratively by num_isat_shrink
                            when compensating for trend

    Output:
        ne          Particle density
        Isat:       Ion saturation current
        Te:         Electron temperature

    Return lines:
        isat_mean:          Mean of isat on the stationary region
        isat_rms:           RMS of isat on the stationary region
        isat_fluc:          Normalized fluctuation of isat on the stationary
                            region
        isat_skew:          Skewness of isat on the stationary region
        isat_kurt:          Kurtosis of isat on the stationary region
        vfloat_best:        Best fit parameter of Vfloat
        sigma_vfloat:       Error on best Vfloat guess from fit
        Te_best:            Best fit parameter on T_e
        sigma_Te:           Error on best T_e guess from fit
        num_stationary:     Length of determined interval on which isat is
                            approximately steady
        fig_result:         Figure with fit results

    """
    smooth_length = 40
    x_sheath = lambda V0, Te: 1.02 * (np.sqrt(np.sqrt(-V0 / Te)
                                              - 2.0 ** (-0.5)) *
                                      (np.sqrt(-V0 / Te) + 2.0 ** 0.5))
    nlin_fitfun = lambda U, a, b, c: a + b * np.exp(U / c)
    fit_range = np.arange(fit_min, fit_max + 1, fit_step)
    nfits = np.size(fit_range)
    # Values from non-lin fit
    a_fit = np.zeros(nfits, dtype='float64')
    sigA_fit = np.zeros(nfits, dtype='float64')
    b_fit = np.zeros(nfits, dtype='float64')
    sigB_fit = np.zeros(nfits, dtype='float64')
    c_fit = np.zeros(nfits, dtype='float64')
    sigC_fit = np.zeros(nfits, dtype='float64')

    # Smooth the U-I curve and estimate the floating potential
    try:
        I_sm = smooth(I, smooth_length)
        I_sm = I_sm[smooth_length / 2:-(smooth_length / 2 - 1)]
    except:
        raise FitException('Failed to smooth')
    try:
        vfloat_guess_idx = np.argmin(np.abs(I_sm))
        vfloat_guess = U[vfloat_guess_idx]
    except:
        pass

    if not(silent):
        print 'Guessing vfloat: at idx %d: %f, I[%d] = %f' %\
            (vfloat_guess_idx, vfloat_guess, vfloat_guess_idx,
             I[vfloat_guess_idx])

    if return_plot:
        #fig_vfest = plt.figure()
        fig_result = plt.figure(figsize=(12, 12))
        ax_res1 = fig_result.add_subplot(311)
        ax_res1.plot(U, I, label='Input data')
        ax_res1.set_ylim((0.8 * I.min(), 1.2 * I.max()))
        #ax_vfest = fig_vfest.add_subplot(111)
        #ax_vfest.plot(U, I, label='Input data')
        #ax_vfest.plot(U, I_sm, label='smoothed data')
        #ax_vfest.plot(U[vfloat_guess_idx], I[vfloat_guess_idx],
        #              'ko', label='Estimated Vfloat')

    # Setup the fit interval
    # Assure the following conditions:
    # 1.) vfloat_guess_idx + fit_min < np.size(I) :
    #     Minimum length of the fit interval does not exceed array dimension
    # 2.) vfloat_guess_idx + fit_max < np.size(I) :
    #     Maximum length of the fit interval does not exceed array dimension
    # If any condition fails, adjust the fit interval appropriately

    if (vfloat_guess_idx + fit_min > np.size(I)):
        if not(silent):
            notify_str = "The minimum length of the fit interval cannot "
            notify_str += "exceed the array size. Adjist min_fit = 10"
            print notify_str
        fit_min = 10

    if(vfloat_guess_idx + fit_max > np.size(I)):
        #Cut fit interval down to U-I sweep size
        fit_max = np.size(I) - vfloat_guess_idx

    #Increase the fit areal and fit Isat, I_b and Te. Te has a global
    # minimum, so once c(Te) increases we can break
    if not(silent):
        print '====== Running first fit ========'
    for idx, npoints in enumerate(fit_range):
        res = call_uifit(U[vfloat_guess_idx:vfloat_guess_idx + npoints],
                         I[vfloat_guess_idx:vfloat_guess_idx + npoints],
                         show_plots=False)
        a, sA, b, sB, c, sC, chi2, err = res
        a_fit[idx] = a[0]
        sigA_fit[idx] = sA[0]
        b_fit[idx] = b[0]
        sigB_fit[idx] = sB[0]
        c_fit[idx] = c[0]
        sigC_fit[idx] = sC[0]
        RChi2 = chi2[0]
        if not(silent):
            output_str = '\t:d from lstsq-fit: a = %f pm %f' % (a_fit[idx],
                                                                sigA_fit[idx])
            output_str += '\tb=%f pm %f' % (b_fit[idx], sigB_fit[idx])
            output_str += '\tc=%f pm %f' % (c_fit[idx], sigC_fit[idx])
            output_str += 'Rchi2 = %f' % (RChi2)
            print output_str
    # Choose the fit with the minimum error on temperature
    best_fit_idx = sigC_fit.argmin()
    Te_first = c_fit[best_fit_idx]
    if not(silent):
        print '====== Results from first fit ======'
        print '       Minimum Te=%f, %d points' %\
            (Te_first, fit_range[best_fit_idx])
    # Compensate for varying sheath thickness
    # Sheath thickness as a function of voltage, compensate for this
    lambdad = 7.43e2 * np.sqrt(Te_first / n0)
    probea_bd = 2.0 ** (-0.5) - 5.
    # Effective area = A_probe * (1. + xs/(probe radius)).
    # This is only defined for U>0
    A_effective = np.ones_like(U)
    # x_sheath is not defined for volates larger than probea_bd, see
    # Hutchinson 3.2.25., the discriminant of the roots
    # would be negative. Compute the effective area only for voltages
    # smaller than this.
    # Set all voltages which turned out an effective area of NaN to unity.
    A_effective[U < probea_bd] = 1. +\
        x_sheath(U[U < probea_bd], Te_first) * lambdad / 0.75
    A_effective[np.isnan(A_effective)] = 1.0
    I = I / A_effective

    if not (silent):
        print '===== Compenstaed area for sheath thickness====='
        print '===== Second fit ======'

    # Do second round of fitting with U-I curve compensated for varying
    # sheath thickness
    fit_pts = np.arange(fit_min, fit_max + 1, fit_step)
    for idx, npoints in enumerate(fit_pts):
        res = call_uifit(U[:vfloat_guess_idx + npoints],
                         I[:vfloat_guess_idx + npoints],
                         show_plots=False)

        a, sA, b, sB, c, sC, chi2, err = res
        a_fit[idx] = a[0]
        sigA_fit[idx] = sA[0]
        b_fit[idx] = b[0]
        sigB_fit[idx] = sB[0]
        c_fit[idx] = c[0]
        sigC_fit[idx] = sC[0]
        RChi2 = chi2[0]

        if not(silent):
            output_str = '\t:d from lstsq-fit: a = %f pm %f' % (a_fit[idx],
                                                                sigA_fit[idx])
            output_str += '\tb=%f pm %f' % (b_fit[idx], sigB_fit[idx])
            output_str += '\tc=%f pm %f' % (c_fit[idx], sigC_fit[idx])
            output_str += 'Rchi2 = %f' % (RChi2)
            print output_str

    # Choose the best fit: Select the fit with minimal temperature on
    # all fits, where I_sat > 0 and I_b < 0
    # 1.) Isat > 0
    neg_isat_idx = np.argwhere(a_fit > 0)
    # 2.) Ib < 0
    neg_ib_idx = np.argwhere(b_fit < 0)
    # 3.) 5 < Te < 100
    large_te_idx = np.argwhere((c_fit > 10.) & (c_fit < 130.))
    # The fits we consider are in the intersection of these arrays
    good_fit_idx = np.intersect1d(large_te_idx,
                                  np.intersect1d(neg_isat_idx, neg_ib_idx))

    # The fit can of course go wrong. If this is the case, raise an error
    # Conditions for a bad fit: (found empirical)
    if (good_fit_idx.size < 1):
        if not(silent):
            print 'No fit qualifies, good_fit_idx=', good_fit_idx
        plt.close()
        exc_str = "Bad fit: no fit with Isat > 0, Ib < 0, Te < 100eV found"
        raise FitException(exc_str)

    best_fit_idx = sigC_fit[good_fit_idx].argmin()
    if not(silent):
        print 'Considering fits at only indices:', good_fit_idx
        print 'best fit at idx=%d' % best_fit_idx

    a_best = (a_fit[good_fit_idx])[best_fit_idx]
    sigA_best = (sigA_fit[good_fit_idx])[best_fit_idx]
    b_best = (b_fit[good_fit_idx])[best_fit_idx]
    sigB_best = (sigB_fit[good_fit_idx])[best_fit_idx]
    c_best = (c_fit[good_fit_idx])[best_fit_idx]
    sigC_best = (sigC_fit[good_fit_idx])[best_fit_idx]

    Isat_best = a_best
    sig_Isat = sigA_best

    Te_best = c_best
    sig_Te = sigC_best

    vfloat_best = Te_best * np.log(-Isat_best / b_best)
    sig_vfloat = np.sqrt(np.log(-Isat_best / b_best) ** 2 * sig_Te ** 2 +
                         Te_best * Te_best * sig_Isat * sig_Isat /
                         (Isat_best * Isat_best * np.abs(b_best)) +
                         Te_best ** 2 * sigB_best ** 2 / b_best ** 2)

    if not(silent):
        print '====Best fit parameters:'
        res_str = "a = %f\tsigmaA = %f\t" % (a_best, sigA_best)
        res_str += "b = %f\tsigmaB = %f\t" % (b_best, sigB_best)
        res_str += "c = %f\tsigmab = %f" % (c_best, sigC_best)
        print res_str
        res_str = "Isat = %f (+-) %f\t" % (Isat_best, sig_Isat)
        res_str += "Vfloat = %f (+-) %f\t" % (vfloat_best, sig_vfloat)
        res_str += "Te = %f (+-) %f" % (Te_best, sig_Te)
        print res_str

    if return_plot:
        ax_res1.plot(U, I, '.g', label='Probe data, probe_A compensated')
        ax_res1.plot(U, nlin_fitfun(U, a_best, b_best, c_best),
                     label='Fit: a=%f, b=%f, c=%f' % (a_best, b_best, c_best))
        ax_res1.plot(U, -ui_fun(U, Isat_best, vfloat_best, Te_best), 'k',
                     label='Fit, Isat=%4.3fA Vfloat=%4.2fV, Te=%4.2feV' %
                     (Isat_best, vfloat_best, Te_best))
        ax_res1.legend(loc='lower left')

    # Last part. Compute statistics for I in the Isat region. We need a
    # detrended Isat region. The criterion for a detrended Isat region is:
    # Let the trend in this region be given by T = m*U + n
    # Then the trend m is much smaller than the mean on this interval:
    # m / <I> < epsilon
    # Compute mean, rms, skewness and kurtosis on minimum 100 points.
    # If the isat region  still shows a significant trend, this is a bad fit.

    # Define the Isat region. First, take an educated guess: between
    # 1-150 or index of (U < vfloat-3TE)
    try:
        # The line below raises a value error for vfloat -3Te < min(U).
        isat_indices = np.arange(1, max(np.argwhere(U[U < vfloat_best
                                                      - 3.0 * Te_best]).max(),
                                        201))
    except ValueError:
        isat_indices = np.arange(1, 201)

    lin_trend_norm = 100.
    good_isat_region = False
    while ((lin_trend_norm > eps_isat_trend)
           and (np.size(isat_indices) > 100)):
        # Find the trend of the signal by computing linear least squares fit
        A = np.vstack([np.squeeze(U[isat_indices]),
                       np.ones(np.size(U[isat_indices]))]).T
        try:
            lin_trend, offset = np.linalg.lstsq(A, I[isat_indices])[0]
        except ValueError:
            print 'Least squares fit on isat region failed: %d points.' %\
                (np.size(isat_indices))
            isat_indices = isat_indices[:-num_isat_shrink]
            good_isat_region = False
            continue
        good_isat_region = True
        lin_trend_norm = np.abs(lin_trend / I[isat_indices].mean())
        if not(silent):
            res_str = "Isat region: %d items\t" % (isat_indices.size)
            res_str += "<I> = %f\t" % (I[isat_indices].mean())
            res_str += "trend / <I> = %f" % (lin_trend_norm)
            print res_str
        isat_indices = isat_indices[:-num_isat_shrink]

    if good_isat_region:
        if not(silent):
            print 'Good isat region on voltages %d:%d' %\
                (U[isat_indices[0]], U[isat_indices[-1]])
        # This block is executed if we found part of the Isat signal showing
        # only a small trend.
        # Remove the remaining linear trend on the Isat region
        isat_region = np.zeros_like(I[isat_indices])
        isat_region[:] = I[isat_indices] - U[isat_indices] * lin_trend

        # Compute statistics in the isat signal
        if (np.size(isat_indices) > 100):
            #  Large enough region, compute moments
            isat_mean = isat_region.mean()
            isat_rms = isat_region.std()
            isat_fluc = isat_rms / isat_mean
            isat_skew = skew(isat_region - isat_mean)
            isat_kurt = kurtosis(isat_region - isat_mean)
            num_stationary = np.size(isat_indices)
        else:
            # Region too small, statistics are meaningless
            isat_mean = Isat_best
            isat_rms = -999.9
            isat_fluc = -999.9
            isat_skew = -999.9
            isat_kurt = -999.9
            num_stationary = -1
    else:
        # No subintervall in the isat region without a
        # trend could be identified
        isat_mean = Isat_best
        isat_rms = -999.9
        isat_fluc = -999.9
        isat_skew = -999.9
        isat_kurt = -999.9
        num_stationary = -1

    if return_plot:
        ax_sig = fig_result.add_subplot(312)
        ax_hist = fig_result.add_subplot(313)
        ax_sig.set_xlabel('probe bias/V')
        ax_sig.set_ylabel('probe current/A')
        ax_sig.plot(U[isat_indices], I[isat_indices], label='Probe')
        ax_sig.plot(U[isat_indices], U[isat_indices] * lin_trend + offset, 'k',
                    label='linear trend')
        ax_sig.plot(U[isat_indices], isat_region, 'g', label='Detrended')
        ax_sig.legend(loc='best')

        nbins = isat_region / 20
        hist, edges = np.histogram(isat_region, bins=nbins)

        x_mid = np.diff(edges) * 0.5 + edges[:-1]
        label_hist = 'detrended. Mean=%3.2f, rms=%3.2f, S=%3.2f, F=%3.2f' %\
            (isat_mean, isat_rms, isat_skew, isat_kurt)
        ax_hist.plot(x_mid, hist, 'b.', label=label_hist)
        ax_hist.set_xlabel('detrended Isat /A')
        ax_hist.set_ylabel('counts')
        ax_hist.legend(loc='best')

    if return_plot:
        return (isat_mean, isat_rms, isat_fluc, isat_skew, isat_kurt,
                vfloat_best, sig_vfloat, Te_best,
                sig_Te, num_stationary, fig_result)
    else:
        return (isat_mean, isat_rms, isat_fluc, isat_skew, isat_kurt,
                vfloat_best, sig_vfloat, Te_best, sig_Te, num_stationary)


def call_uifit(U, I_fit, npts_fit=None, show_plots=False, debug=False):
    """
    Wrapper function for fastexprange_work routine
    Fit the function y(x) = a + b * exp(x/c) on the U-U curve passed.
    Return I_sat, V_float and T_e of the fit on the interval that yields the
    smallest error on T_e.

    Input:
        U:         Probe voltage, np.ndarray(dtype='float64')
        I:         Probe current, np.ndarray(dtype='float64')
        npts_fit:  Use I[:npts_fit[i]] U[:npts_fit[i]] points for fitting.
                   np.ndarray(dtype='int')

    Output:
       I_sat, T_e and V_float including errors of the best fit

    """

    # Load fastexprange library
    fname_lib = "/Users/ralph/source/py_fortran_tests/fastexprange_work.so"
    fexp_lib = ctypes.cdll.LoadLibrary(fname_lib)

    if npts_fit is None:
        npts_fit = np.array([U.size])

    assert(U.size == I_fit.size)
    assert(npts_fit.max() <= U.size)

    #npoints = U.size
    # Prepare input for U-I fit in python. These need to be created as
    # ctypes objects valid references to them can be created later
    m = ctypes.c_int(U.size)
    npts_end = ctypes.c_int(npts_fit.size)
    #iend = ctypes.c_int(npoints)
    #a = ctypes.c_double(0.0)
    #b = ctypes.c_double(0.0)
    #c = ctypes.c_double(0.0)
    #sigmaA = ctypes.c_double(0.0)
    #sigmaB = ctypes.c_double(0.0)
    #sigmaC = ctypes.c_double(0.0)
    #RChi2 = ctypes.c_double(0.0)
    c_guess = ctypes.c_double(20.0)
    cMin = ctypes.c_double(1.0)
    cMax = ctypes.c_double(150.0)
    Xacc = ctypes.c_double(0.001)
    MaxIter = ctypes.c_int(1000)
    #Iter = ctypes.c_int(0)
    #Error = ctypes.c_int(0)

    a = np.zeros(npts_fit.size, dtype='float64')
    sigmaA = np.zeros(npts_fit.size, dtype='float64')
    b = np.zeros(npts_fit.size, dtype='float64')
    sigmaB = np.zeros(npts_fit.size, dtype='float64')
    c = np.zeros(npts_fit.size, dtype='float64')
    sigmaC = np.zeros(npts_fit.size, dtype='float64')
    RChi2 = np.zeros(npts_fit.size, dtype='float64')
    Iter = np.zeros(npts_fit.size, dtype='int')
    Error = np.zeros(npts_fit.size, dtype='int')

    # Create references to fit parameters
    # integer, intent(in) :: m
    # Number of data points in x,y array
    f_m = ctypes.byref(m)
    # integer, intent(in) :: npts_end
    #Number of end points for different fit intervalls
    f_npts_end = ctypes.byref(npts_end)
    # integer, intent(in) :: iend
    # Endpoints of the fit intervals
    f_iend = ctypes.c_void_p(npts_fit.ctypes.data)
    # real(kind=8), intent(in) :: x   x_values
    f_x = ctypes.c_void_p(U.ctypes.data)
    # real(kind=8), intent(in) :: y   y values
    f_y = ctypes.c_void_p(I_fit.ctypes.data)
    # real(kind=8), intent(out) :: a  fit parameter
    #f_a = ctypes.byref(a)
    f_a = ctypes.c_void_p(a.ctypes.data)
    # real(kind=8), intent(out) :: sigmaA  error on fit parameter
    #f_sigmaA = ctypes.byref(sigmaA)
    f_sigmaA = ctypes.c_void_p(sigmaA.ctypes.data)
    # real(kind=8), intent(out) :: b  fit parameter
    #f_b = ctypes.byref(b)
    f_b = ctypes.c_void_p(b.ctypes.data)
    # real(kind=8), intent(out) :: sigmaB  error on fit parameter
    #f_sigmaB = ctypes.byref(sigmaB)
    f_sigmaB = ctypes.c_void_p(sigmaB.ctypes.data)
    # real(kind=8), intent(out) :: c  fit parameter
    #f_c = ctypes.byref(c)
    f_c = ctypes.c_void_p(c.ctypes.data)
    # real(kind=8), intent(out) :: sigmaC  error on fit parameter
    #f_sigmaC = ctypes.byref(sigmaC)
    f_sigmaC = ctypes.c_void_p(sigmaC.ctypes.data)
    #  real(kind=8), intent(out) :: RChi2  reduced chi^2 from fit
    #f_RChi2 = ctypes.byref(RChi2)
    f_RChi2 = ctypes.c_void_p(RChi2.ctypes.data)
    # real(kind=8), intent(in) :: c_guess  initial guess for c
    f_c_guess = ctypes.byref(c_guess)
    # real(kind=8), intent(in) :: cMin   min value for c
    f_cMin = ctypes.byref(cMin)
    # real(kind=8), intent(in) :: cMax   max value for c
    f_cMax = ctypes.byref(cMax)
    # real(kind=8), intent(in) :: Xacc  desired accuracy for c
    f_Xacc = ctypes.byref(Xacc)
    # integer, intent(inout) :: MaxIter
    # maximum number of function evaluations allowed / f. evals done
    f_MaxIter = ctypes.byref(MaxIter)
    # integer, intent(inout) :: Iter
    # max number of function evaluations allowed / f. evals done
    #f_Iter = ctypes.byref(Iter)
    f_Iter = ctypes.c_void_p(Iter.ctypes.data)
    # integer, intent(out) :: Error  : Did something go wrong?
    #f_Error = ctypes.byref(Error)
    f_Error = ctypes.c_void_p(Error.ctypes.data)

    fit_error = fexp_lib.fexpr_py_(f_m, f_npts_end, f_iend, f_x, f_y, f_a,
                                   f_sigmaA, f_b, f_sigmaB, f_c, f_sigmaC,
                                   f_RChi2, f_c_guess, f_cMin, f_cMax,
                                   f_Xacc, f_MaxIter, f_Iter, f_Error)
    if(debug):
        print '\tcall_uifit: a = ', a
        print '\tcall_uifit: sigmaA = ', sigmaA
        print '\tcall_uifit: b = ', b
        print '\tcall_uifit: sigmaB = ', sigmaB
        print '\tcall_uifit: c = ', c
        print '\tcall_uifit: sigmaC = ', sigmaC
        print '\tcall_uifit: f_cMin = ', cMin
        print '\tcall_uifit: f_cMax = ', cMax
        print '\tcall_uifit: f_Xacc = ', Xacc
        print '\tcall_uifit: f_RChi2 = ', RChi2
        print '\tcall_uifit: f_Error = ', Error

    #if(fit_error.max() == 1):
    #    print 'Bad fit: fit_error = %d' % fit_error
    return a, sigmaA, b, sigmaB, c, sigmaC, RChi2, Error

# end of file uifit.py
