#!/usr/bin/env python
#-*- Encoding: UTF-8 -*-

import numpy as np
import ctypes
import matplotlib.pyplot as plt




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
    m        = np.size(U)
    npts_end = ctypes.c_int(4)
    iend     = npoints - np.arange( int( (npts_end.value+offset) * interval), int(offset*interval), -interval, dtype='i4')
    a        = np.zeros(npts_end.value, dtype='float64')
    b        = np.zeros(npts_end.value, dtype='float64')
    c        = np.zeros(npts_end.value, dtype='float64')
    sigmaA   = np.zeros(npts_end.value, dtype='float64')
    sigmaB   = np.zeros(npts_end.value, dtype='float64')
    sigmaC   = np.zeros(npts_end.value, dtype='float64')
    RChi2    = np.zeros(npts_end.value, dtype='float64')
    c_guess  = ctypes.c_double(20.0)
    cMin     = ctypes.c_double(1.0)
    cMax     = ctypes.c_double(150.0)
    Xacc     = ctypes.c_double(0.001)
    MaxIter  = ctypes.c_int(1000) 
    Iter     = ctypes.c_int(0)
    Error    = ctypes.c_int(0)


    # Create references to fit parameters
    f_m = ctypes.byref(ctypes.c_int(m))               #      integer, intent(in) :: m                      Number of data points in x,y array
    f_npts_end = ctypes.byref(npts_end)               #      integer, intent(in) :: npts_end               Number of end points for different fit intervalls
    f_iend = ctypes.c_void_p(iend.ctypes.data)        #      integer, intent(in) :: iend                   Endpoints of the fit intervals
    f_x = ctypes.c_void_p(U.ctypes.data)              #      real(kind=8), intent(in) :: x                 x values
    f_y = ctypes.c_void_p(I_fit.ctypes.data)          #      real(kind=8), intent(in) :: y                 y values
    f_a = ctypes.c_void_p(a.ctypes.data)              #      real(kind=8), intent(out) :: a                fitting parameter returned
    f_sigmaA = ctypes.c_void_p(sigmaA.ctypes.data)    #      real(kind=8), intent(out) :: sigmaA           error in fitting parameter returned
    f_b = ctypes.c_void_p(b.ctypes.data)              #      real(kind=8), intent(out) :: b                fitting parameter
    f_sigmaB = ctypes.c_void_p(sigmaB.ctypes.data)    #      real(kind=8), intent(out) :: sigmaB           error on fitting parameter
    f_c = ctypes.c_void_p(c.ctypes.data)              #      real(kind=8), intent(out) :: c                fitting parameter
    f_sigmaC = ctypes.c_void_p(sigmaC.ctypes.data)    #      real(kind=8), intent(out) :: sigmaC           error on fitting parameter
    f_RChi2 = ctypes.c_void_p(RChi2.ctypes.data)      #      real(kind=8), intent(out) :: RChi2            reduced chi^2 from fit
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

    v_float_fit = np.zeros( npts_end.value, dtype='float64')
    #print '=============================================================================== Fitting U-I curve'
    #print '    Returned from fit_result:'
    for i in np.arange(npts_end.value):
        #print '    I_sat = %f pm %f,\tI_b, = %f pm %f,\tT_e = %f pm %f,\tRChi2 = %e' % (a[i], sigmaA[i], b[i], sigmaB[i], c[i], sigmaC[i], RChi2[i])
        v_float_fit[i] = c[i] * np.log(-a[i]/b[i])
        #print '    Computing: V_float = %f' % ( v_float_fit[i] )

    if show_plots:
        fig = plt.figure()
        ax_list = [fig.add_subplot(npts_end.value, 1, i) for i in np.arange(1, npts_end.value+1)]
        for i in np.arange(npts_end.value):
            ax_list[i].plot(U[:iend[i]], I_fit[:iend[i]] - a[i], 'k.')
            ax_list[i].plot(U[:iend[i]], -I_fun(U[:iend[i]], v_float_fit[i], c[i], a[i]) - a[i], 'k')
        
        plt.show()
    bidx = np.argmin(sigmaC)

    return a[bidx], sigmaA[bidx], c[bidx]*np.log(-a[bidx]/b[bidx]), sigmaB[bidx], c[bidx], sigmaC[bidx]







