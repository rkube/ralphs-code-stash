#!/usr/bin/env python
#-*- Encoding: UTF-8 -*-

import numpy as np
import ctypes
import matplotlib.pyplot as plt
from misc.smoothing import smooth
from scipy.stats import skew, kurtosis

def uifit_minTE(U, I, fit_min=100, fit_max=300, fit_step=10, probe_a = 9.6e-7, n0=1e13, show_plots = True, save_plots = False, plotname_base = ''): 
    """
    Fit a U-I characteristic on the profile using the min Te method.

    1.) Estimate V_float by smoothing the profile
    2.) Starting from V_float, nfits fits on the U-I characteristic, finding the minimum
        T_e value
    3.) Using the T_e value, compensat for a varying sheath size with increasingly negative V_probe (Hutchinson, (3.2.27))
    3a) Alternatively, compensate for a linear trend in the Isat segment by subtracting a linear fit


    Input:
        U:          Probe voltage
        I:          Probe current
        nfit:       Number of points to fit
        interval:   Number of points to increase fit interval to find Te


    Output:
        ne          Particle density
        Isat:       Ion saturation current
        Te:         Electron temperature

    """
    smooth_length = 40
    x_sheath = lambda V0,Te: 1.02 * ( ( (-V0/Te)**0.5 - 2.0**-0.5)**0.5 * ( (-V0/Te)**0.5 + 2.0**0.5) )
    I_fun = lambda U, Isat, Vfloat, Te: Isat * ( np.exp( (U-Vfloat)/Te) - 1.0)
    nlin_fitfun = lambda U, a, b, c: a + b*np.exp(U/c)
    fit_range = np.arange( fit_min, fit_max+1, fit_step)
    nfits = np.size(fit_range)
    # Values from non-lin fit
    a_fit = np.zeros(nfits, dtype='float64')
    sigmaA_fit = np.zeros(nfits, dtype='float64')
    b_fit = np.zeros(nfits, dtype='float64')
    sigmaB_fit = np.zeros(nfits, dtype='float64')
    c_fit = np.zeros(nfits, dtype='float64')
    sigmaC_fit = np.zeros(nfits, dtype='float64')
    # Epsilon parameter for defining Isat region
    eps_isat_trend = 1e-4
    # Number of points by which the isat interval is shrunk per iteration
    num_isat_shrink = 10


    # Smooth the U-U curve and estimate the floating potential
    I_sm = smooth(I, smooth_length)
    I_sm = I_sm[smooth_length/2:-(smooth_length/2-1)]
    try:
        vfloat_guess_idx = np.argmin(np.abs(I_sm))
        vfloat_guess = U[vfloat_guess_idx]
    except:
        pass

    print 'Guessing vfloat: at idx %d: %f, I[%d] = %f' % ( vfloat_guess_idx, vfloat_guess, vfloat_guess_idx, I[vfloat_guess_idx])

    if show_plots:
        fig_res = plt.figure()
        ax_res = fig_res.add_subplot(111)
        ax_res.plot(U, I, label='Input data')
        plt.figure()
        plt.plot(U,I_sm)
        plt.plot(U[vfloat_guess_idx], I[vfloat_guess_idx], 'ko')
    #    plt.show()

    # Setup the fit interval
    # Assure the following conditions:
    # 1.) vfloat_guess_idx + fit_min < np.size(I) : Minimum length of the fit interval does not exceed array dimension
    # 2.) vfloat_guess_idx + fit_max < np.size(I) : Maximum length of the fit interval does not exceed array dimension
    # If any condition fails, adjust the fit interval appropriately

    if (vfloat_guess_idx + fit_min > np.size(I)):
        print 'The minimum length of the fit interval cannot exceed the array size'
        print 'Adjusting fit_min = 10'
        fit_min = 10

    if(vfloat_guess_idx + fit_max > np.size(I)):
        fit_max = np.size(I) - vfloat_guess_idx     #Cut fit interval down to U-I sweep size

    #Increase the fit areal and fit Isat, I_b and Te. Te has a global minimum, so once c(Te) increases we can break
    print '====== Running first fit ========'
    for idx, npoints in enumerate(fit_range):
        a_fit[idx], sigmaA_fit[idx], b_fit[idx], sigmaB_fit[idx], c_fit[idx], sigmaC_fit[idx], RChi2 = uifit(U[vfloat_guess_idx:vfloat_guess_idx+npoints], I[vfloat_guess_idx:vfloat_guess_idx+npoints], nfits=1, offset=0, interval=0, show_plots=show_plots)
        print '   from lstsq-fit: a=%f pm %f\tb=%f pm %f\tc=%f pm %f, RChi2=%f' % ( a_fit[idx], sigmaA_fit[idx], b_fit[idx], sigmaB_fit[idx], c_fit[idx], sigmaC_fit[idx], RChi2)
    # Choose the fit with the minimum error on temperature
    best_fit_idx = sigmaC_fit.argmin() 
    Te_first = c_fit[best_fit_idx]
    print '====== Results from first fit ======'
    print '       Minimum Te=%f, %d points' % ( Te_first, fit_range[best_fit_idx])
    # Compensate for varying sheath thickness
    # Sheath thickness as a function of voltage, compensate for this
    lambdad = 7.43e2*(Te_first/n0)**0.5
    probea_bd = 2.0**-0.5 - 5.
    # Effective area = A_probe * (1. + xs/(probe radius)). This is only defined for U>0
    A_effective = np.ones_like(U)
    # x_sheath is not defined for volates larger than probea_bd, see Hutchinson 3.2.25., the discriminant of the roots
    # would be negative. Compute the effective area only for voltages smaller than this.
    # Set all voltages which turned out an effective area of NaN to unity.
    A_effective[U < probea_bd] = 1. + x_sheath(U[ U < probea_bd], Te_first)*lambdad / 0.75
    A_effective[np.isnan(A_effective)] = 1.0
    I = I / A_effective

    # Plot the effective area   
    #if show_plots:
    #    fig = plt.figure()
    #    ax = fig.add_subplot(111)
    #    ax.plot(U, A_effective)
    #    ax.set_xlabel('Prove Bias/V')
    #    ax.set_ylabel('Effective area [a.u.]')

    # Do second round if fitting with U-I curve compensated for varying sheath thickness
    for idx, npoints in enumerate(np.arange(fit_min, fit_max+1, fit_step)):
        a_fit[idx], sigmaA_fit[idx], b_fit[idx], sigmaB_fit[idx], c_fit[idx], sigmaC_fit[idx], RChi2 = uifit(U[:vfloat_guess_idx+npoints], I[:vfloat_guess_idx+npoints], nfits=1, offset=0, interval=0, show_plots=show_plots)
        print '   from lstsq-fit: a=%f pm %f\tb=%f pm %f\tc=%f pm %f, RChi2=%f' % ( a_fit[idx], sigmaA_fit[idx], b_fit[idx], sigmaB_fit[idx], c_fit[idx], sigmaC_fit[idx], RChi2)
    # Choose the best fit: Select the fit with minimal temperature on all fits, where I_sat > 0 and I_b < 0
    neg_isat_idx = np.argwhere(a_fit > 0)       # 1.) Isat > 0
    neg_ib_idx = np.argwhere(b_fit < 0)         # 2.) Ib < 0
    large_te_idx = np.argwhere(c_fit > 10.)     # 3.) Te > 10.
    # The fits we consider are in the intersection of these arrays
    good_fit_idx = np.intersect1d(large_te_idx, np.intersect1d( neg_isat_idx, neg_ib_idx))
    print 'Considering only indices:', good_fit_idx
    
    # The fit can of course go wrong. If this is the case, raise an error
    # Conditions for a bad fit: (found empirical)
    if ( np.size(good_fit_idx) < 1 ):
        print 'good_fit_idx = ', good_fit_idx, ' are too few to choose from'
        #raise ValueError('Bad fit, no fit with positve Isat, negative Ib and Te < 100eV found')

    best_fit_idx = sigmaC_fit[good_fit_idx].argmin()
    print 'best_fit_idx=%d' % best_fit_idx

    a_best = (a_fit[good_fit_idx])[best_fit_idx]
    sigmaA_best = (sigmaA_fit[good_fit_idx])[best_fit_idx]
    b_best = (b_fit[good_fit_idx])[best_fit_idx]
    sigmaB_best = (sigmaB_fit[good_fit_idx])[best_fit_idx]
    c_best = (c_fit[good_fit_idx])[best_fit_idx]
    sigmaC_best = (sigmaC_fit[good_fit_idx])[best_fit_idx]

    Isat_best = a_best
    sigma_Isat = sigmaA_best

    Te_best = c_best
    sigma_Te = sigmaC_best

    vfloat_best = Te_best * np.log(-Isat_best / b_best)
    sigma_vfloat = np.sqrt( np.log(-Isat_best/b_best)**2*sigma_Te**2 + Te_best*Te_best*sigma_Isat*sigma_Isat/(Isat_best*Isat_best * np.abs(b_best)) + Te_best**2*sigmaB_best**2/b_best**2)

 
    print '====Best fit parameters:'
    print 'a = %f, sigmaA = %f, b = %f, sigmaB = %f, c = %f, sigmaC = %f' % ( a_best, sigmaA_best, b_best, sigmaB_best, c_best, sigmaC_best)
    print 'Isat = %f pm %f\tVfloat = %f pm %f\tTe = %f pm %f' % (Isat_best, sigma_Isat, vfloat_best, sigma_vfloat, Te_best, sigma_Te)

   
    if show_plots:
        ax_res.plot(U, I, 'g', label='Probe data, probe_A compensated')
        ax_res.plot(U, nlin_fitfun(U, a_best, b_best, c_best), label='Fit: a=%f, b=%f, c=%f' % (a_best, b_best, c_best) )
        ax_res.plot(U, -I_fun(U, Isat_best, vfloat_best, Te_best), 'k', label='Fit, Isat=%4.3fA Vfloat=%4.2fV, Te=%4.2feV' % (Isat_best, vfloat_best, Te_best))
        ax_res.legend(loc='lower left')

    # Last part. Compute statistics for I for the Isat region. We need a detrended Isat region.
    # The criterion for a detrended Isat region is: Let the trend in this region be given by I = m*I + n 
    # Then the trend m is much smaller than the mean on this interval: m / <I> < epsilon
    # Compute mean, rms, skewness and kurtosis on minimum 100 points. If the isat region
    # still shows a significant trend, this is a bad fit.

    # Define the Isat region. First, take an educated guess:
    isat_indices = np.argwhere( U[U< vfloat_best - 3.0*Te_best] )
    lin_trend_norm = 100.
    while (lin_trend_norm > eps_isat_trend):
        #print 'size of isat interval: %d' % (np.size(isat_indices)), 'shape(I[isat_indices]) = ', np.shape(I[isat_indices]) , 'shape(U[isat_indices]) = ', np.shape(U[isat_indices])
        # Find the trend of the signal by computing linear least squares fit
        A = np.vstack([ np.squeeze(U[isat_indices]), np.ones(np.size(U[isat_indices]))]).T
        lin_trend, offset = np.linalg.lstsq(A,I[isat_indices])[0]
        lin_trend_norm = np.abs(lin_trend / I[isat_indices].mean())
        #print 'Linear trend in isat region: %f, trend/<Isat> = %f' % (lin_trend, lin_trend_norm)
        isat_indices = isat_indices[:-num_isat_shrink]
             
    # Remove the remaining linear trend on the Isat region
    isat_region = np.zeros_like(I[isat_indices])
    isat_region[:] = I[isat_indices] - U[isat_indices] * lin_trend 

    # Compute statistics in the isat signal
    if ( np.size(isat_indices) > 100 ):
        #  Large enough region, compute moments
        isat_mean = isat_region.mean()
        isat_rms = isat_region.std() / isat_region.mean()
        isat_skew = skew(isat_region - isat_mean)
        isat_kurt = kurtosis(isat_region - isat_mean) 
    else:
        # Region too small, statistics are meaningless
        isat_mean = Isat_best
        isat_rms = -999.9
        isat_skew = -999.9
        isat_kurt = -999.9

    if show_plots:
        fig_stat = plt.figure()
        ax_sig = fig_stat.add_subplot(211)
        ax_hist = fig_stat.add_subplot(212)
        ax_sig.plot(U[isat_indices], I[isat_indices], label='Probe')
        ax_sig.plot(U[isat_indices], U[isat_indices]*lin_trend + offset, 'k', label='linear trend')
        ax_sig.plot(U[isat_indices], isat_region, 'g', label='Detrended') 
        ax_sig.legend(loc='best') 
    
        hist, edges = np.histogram(isat_region, bins = np.round(float(np.size(isat_region))/20.)) 
        ax_hist.plot( edges[:-1] + (edges[1:] - edges[:-1])*0.5, hist, 'b.', label='detrended, mean=%3.2f, rms=%3.2f, s=%3.2f, k=%3.2f' % ( isat_mean, isat_rms, isat_skew, isat_kurt) )
        ax_hist.set_xlabel('detrended Isat /A')
        ax_hist.set_ylabel('counts')
        ax_hist.legend(loc='best')

    return isat_mean, isat_rms, isat_skew, isat_kurt, vfloat_best, sigma_vfloat, Te_best, sigma_Te


def uifit(U, I_fit, nfits=4, offset = 1, interval = 20, show_plots = False):
    """
    Fit a U-I characteristic on the profile passed.
    Make a number of fit, ignoring the last i*interval points, where i = 1...nfits
    Return I_sat, V_float and T_e of the fit on the interval that yields the smallest error
    on T_e.

    Input:
        U:      Probe voltage
        I:      Probe current
        nfits:  Number of fits, has to be smaller than 4
        offset:
        interval:

    Output:
       I_sat, T_e and V_float including errors of the best fit

    """


    # Load fastexprange library
    fexp_lib = ctypes.cdll.LoadLibrary('/Users/ralph/source/py_fortran_tests/fastexprange_work.so')

    # Theoretica U-I characteristic
    I_fun = lambda U, V_float, T_e, I_sat : I_sat* ( np.exp( (U-V_float)/T_e) - 1.0)
    # Create an artificial U-I characteristic with random noise superposed
    npoints = np.size(U)

    print 'U-I curve %d points' % (npoints)

    # Prepare input for U-I fit in python. These need to be created as
    # ctypes objects valid references to them can be created later
    m        = ctypes.c_int(np.size(U))
    npts_end = ctypes.c_int(1)
    iend     = ctypes.c_int(npoints)# - np.arange( int( (npts_end.value+offset) * interval), int(offset*interval), -interval, dtype='i4')
    a        = ctypes.c_double(0.0)
    b        = ctypes.c_double(0.0)
    c        = ctypes.c_double(0.0)
    sigmaA   = ctypes.c_double(0.0)
    sigmaB   = ctypes.c_double(0.0)
    sigmaC   = ctypes.c_double(0.0)
    RChi2    = ctypes.c_double(0.0)
    #a        = np.zeros(npts_end.value, dtype='float64')
    #b        = np.zeros(npts_end.value, dtype='float64')
    #c        = np.zeros(npts_end.value, dtype='float64')
    #sigmaA   = np.zeros(npts_end.value, dtype='float64')
    #sigmaB   = np.zeros(npts_end.value, dtype='float64')
    #sigmaC   = np.zeros(npts_end.value, dtype='float64')
    #RChi2    = np.zeros(npts_end.value, dtype='float64')
    c_guess  = ctypes.c_double(20.0)
    cMin     = ctypes.c_double(1.0)
    cMax     = ctypes.c_double(150.0)
    Xacc     = ctypes.c_double(0.001)
    MaxIter  = ctypes.c_int(1000) 
    Iter     = ctypes.c_int(0)
    Error    = ctypes.c_int(0)

    
    # Create references to fit parameters
    f_m = ctypes.byref(m)                             #      integer, intent(in) :: m                      Number of data points in x,y array
    f_npts_end = ctypes.byref(npts_end)               #      integer, intent(in) :: npts_end               Number of end points for different fit intervalls
    f_iend = ctypes.byref(iend)                       #      integer, intent(in) :: iend                   Endpoints of the fit intervals
    f_x = ctypes.c_void_p(U.ctypes.data)              #      real(kind=8), intent(in) :: x                 x values
    f_y = ctypes.c_void_p(I_fit.ctypes.data)          #      real(kind=8), intent(in) :: y                 y values
    f_a = ctypes.byref(a)              #      real(kind=8), intent(out) :: a                fitting parameter returned
    f_sigmaA = ctypes.byref(sigmaA)    #      real(kind=8), intent(out) :: sigmaA           error in fitting parameter returned
    f_b = ctypes.byref(b)              #      real(kind=8), intent(out) :: b                fitting parameter
    f_sigmaB = ctypes.byref(sigmaB)    #      real(kind=8), intent(out) :: sigmaB           error on fitting parameter
    f_c = ctypes.byref(c)              #      real(kind=8), intent(out) :: c                fitting parameter
    f_sigmaC = ctypes.byref(sigmaC)    #      real(kind=8), intent(out) :: sigmaC           error on fitting parameter
    f_RChi2 = ctypes.byref(RChi2)      #      real(kind=8), intent(out) :: RChi2            reduced chi^2 from fit
    #f_a = ctypes.c_void_p(a.ctypes.data)              #      real(kind=8), intent(out) :: a                fitting parameter returned
    #f_sigmaA = ctypes.c_void_p(sigmaA.ctypes.data)    #      real(kind=8), intent(out) :: sigmaA           error in fitting parameter returned
    #f_b = ctypes.c_void_p(b.ctypes.data)              #      real(kind=8), intent(out) :: b                fitting parameter
    #f_sigmaB = ctypes.c_void_p(sigmaB.ctypes.data)    #      real(kind=8), intent(out) :: sigmaB           error on fitting parameter
    #f_c = ctypes.c_void_p(c.ctypes.data)              #      real(kind=8), intent(out) :: c                fitting parameter
    #f_sigmaC = ctypes.c_void_p(sigmaC.ctypes.data)    #      real(kind=8), intent(out) :: sigmaC           error on fitting parameter
    #f_RChi2 = ctypes.c_void_p(RChi2.ctypes.data)      #      real(kind=8), intent(out) :: RChi2            reduced chi^2 from fit
    f_c_guess = ctypes.byref(c_guess)                 #      real(kind=8), intent(in) :: c_guess           initial guess for c
    f_cMin = ctypes.byref(cMin)                       #      real(kind=8), intent(in) :: cMin              min value for c
    f_cMax = ctypes.byref(cMax)                       #      real(kind=8), intent(in) :: cMax              max value for c
    f_Xacc = ctypes.byref(Xacc)                       #      real(kind=8), intent(in) :: Xacc              desired accuracy for c
    f_MaxIter = ctypes.byref(MaxIter)                 #      integer, intent(inout) :: MaxIter             maximum number of function evaluations allowed / f. evals done
    f_Iter = ctypes.byref(Iter)                       #      integer, intent(inout) :: Iter                max number of function evaluations allowed / f. evals done
    f_Error = ctypes.byref(Error)                     #      integer, intent(out) :: Error                 Error parameter


    #print '============================ Start Fortran magic ==================================='
    fit_result = fexp_lib.fexpr_py_( f_m, f_npts_end, f_iend, f_x, f_y, f_a, f_sigmaA, f_b, f_sigmaB, f_c, f_sigmaC, f_RChi2, f_c_guess, f_cMin, f_cMax, f_Xacc, f_MaxIter, f_Iter, f_Error)
    #print '============================ End Fortran magic ==================================='


    return a.value, sigmaA.value, b.value, sigmaB.value, c.value, sigmaC.value, RChi2.value
#
#    v_float_fit = np.zeros( npts_end.value, dtype='float64')
#    #print '=============================================================================== Fitting U-I curve'
#    #print '    Returned from fit_result:'
#    for i in np.arange(npts_end.value):
#        #print '    I_sat = %f pm %f,\tI_b, = %f pm %f,\tT_e = %f pm %f,\tRChi2 = %e' % (a[i], sigmaA[i], b[i], sigmaB[i], c[i], sigmaC[i], RChi2[i])
#        v_float_fit[i] = c[i] * np.log(-a[i]/b[i])
#        #print '    Computing: V_float = %f' % ( v_float_fit[i] )
#
#    if show_plots:
#        fig = plt.figure()
#        ax_list = [fig.add_subplot(npts_end.value, 1, i) for i in np.arange(1, npts_end.value+1)]
#        for i in np.arange(npts_end.value):
#            ax_list[i].plot(U[:iend[i]], I_fit[:iend[i]] - a[i], 'k.')
#            ax_list[i].plot(U[:iend[i]], -I_fun(U[:iend[i]], v_float_fit[i], c[i], a[i]) - a[i], 'k')
#        
#        plt.show()
#    bidx = np.argmin(sigmaC)
#
#    return a[bidx], sigmaA[bidx], c[bidx]*np.log(-a[bidx]/b[bidx]), sigmaB[bidx], c[bidx], sigmaC[bidx]







