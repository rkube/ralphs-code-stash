#!/opt/local/bin python
# -*- Encoding: UTF-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq 

# Define the functors used in rho_curve_fitting
class functor_gauss(object):
    """ 
    Functor of a gaussian function

    returns f(t) = A * exp(-(t - t0)^2 / 2 sigma^2)
    """
    def __init__(self, p):
        # p[0] = A
        # p[1] = t0
        # p[2] = sigma
        self.p = p
        self.r_min = 0.0
        self.r_max = 1.0

        print 'Created gaussian functor. A = %f, t0 = %f, sigma = %f' % (self.p[0], self.p[1], self.p[2])

    def __call__(self, t):
        return self.p[0] * np.exp(-1.0 * ((t - self.p[1]) * (t - self.p[1])) / (2.0 * self.p[2] * self.p[2]))

    def set_range(self, trange):
        """
        Sets max and min values of the function as evaluated on trange
        """
        yvals = self.__call__(trange)
        self.r_min = yvals.min()
        self.r_max = yvals.max()

    def invert(self, rho):
        """
        Inverts Gaussian function:
        t(rho) = sqrt(-2 B^2 log(rho/A)) + t0
        """
        assert(rho < self.p[0])
        assert(rho >= self.r_min)
        assert(rho <= self.r_max)
        return np.sqrt(-2.0 * self.p[2] * self.p[2] * np.log(rho / self.p[0])) + self.p[1]

    def get_p(self):
        return p

    def get_min_rho(self):
        return self.r_min

    def get_max_rho(self):
        return self.r_max


class functor_parabola(object):
    """
    Functor for a parabola

    f(t) = A + B * (t - t0)^2
    """

    def __init__(self, p):
        # p[0] = A
        # p[1] = t0
        # p[2] = B
        self.p = p
        self.r_min = 0.0
        self.r_max = 1.0
        print 'Created parabola functor. A = %f, t0 = %f, B = %f' % (self.p[0], self.p[1], self.p[2])

    def __call__(self, t):
        return self.p[0] + self.p[2] * (t - self.p[1]) * (t - self.p[1]) 

    def set_range(self, trange):
        """
        Sets max and min values of the function as evaluated on trange
        """
        yvals = self.__call__(trange)
        self.r_min = yvals.min()
        self.r_max = yvals.max()

    def invert(self, rho):
        """
        Returns sqrt((rho - A) / B) + t0
        """
        assert(rho >= self.r_min)
        assert(rho <= self.r_max)
        return -1.0 * np.sqrt((rho - self.p[0]) / self.p[2]) + self.p[1]

    def get_p(self):
        return p

    def get_min_rho(self):
        return self.r_min

    def get_max_rho(self):
        return self.r_max


def rho_curve_fitting(probe_rho, probe_tb, t_start, t_end, mode='parabola'): 
    """
    Returns a functor that gives rho(t) for the inward plunge

    Input:
    ======
        probe_rho:  ndarray, rho waveform 
        probe_tb:   ndarray, timebase of rho waveform
        t_start:    float, start time for fit
        t_end:      float, end time of fit
        mode:       string, Determines which function to fit on the plunge. Either "gaussian" or "parabola"

    Output:
    =======
        rho_func:   callable functor that gives rho(t)
    """

    assert(mode in ['parabola', 'gaussian'])

    def plunge_func_gauss(p, t):
        return p[0] * np.exp(-1.0 * ((t - p[1]) * (t - p[1])) / (2.0 * p[2] * p[2]))

    def plunge_func_parabola(p, t):
        return p[0] + p[2] * (t - p[1]) * (t - p[1]) 

    if mode is 'parabola':
        def err_func(p, y, t):
            return np.abs(y - plunge_func_parabola(p, t))
    elif mode is 'gaussian':
        def err_func(p, y, t):
            return np.abs(y - plunge_func_gauss(p, t))

    good_tidx = ((probe_tb > t_start) & (probe_tb < t_end))
    rho_fit = probe_rho[good_tidx]
    tb_fit = probe_tb[good_tidx]

    # Initial fit parameters
    p0 = [rho_fit.min(), t_start + 0.5 * (t_end - t_start), 0.15]
    p_recip, success_recip = leastsq(err_func, p0, args=(rho_fit, tb_fit), maxfev=1000)

    if mode is 'parabola':
        functor_rho =  functor_parabola(p_recip)
    elif mode is 'gaussian':
        functor_rho =  functor_gauss(p_recip)
    
    # Set the range for which the functor is valid
    functor_rho.set_range(tb_fit)

    print 'created functor for range %e - %e' % (functor_rho.get_min_rho(), functor_rho.get_max_rho())

    return functor_rho


def binning_moments_kstar(probe_signal, tb_signal, rho_mid, rho_func, delta_rho_0=1e-3, eps_err=1e-1):
    """
    The idea is to have a routine that computes radial profiles with
    * fixed radial points
    * over an approximately equal number of points in each bin

    Input:
    ======
        probe_signal:   ndarray, Raw Isat data from the probe
        tb_signal:      ndarray, timebase of the probe signal
        rho_mid:        ndarray, mid points for the profile points
        rho_func:       callable, gives rho at current time
        delta_rho_0:    float, initial guess for rho bin width
        rel_err:        float, relative error for iterative interval

    Output:
    =======
        nelem_arr:      ndarray, number of points in each rho bin
        stat_arr:       ndarray, statistic computed on the grid

    Selection of elements in a bin:

    For each point in rho_mid, choose an initial guess for the interval:
    [rho_mid - delta_rho_0 : rho_mid + delta_rho_0] 

    -> Do a linear fit on Isat in this interval
    while I(rho_mid - delta_rho) - I(rho_mid + delta_rho) / mean(I) <= rel_err
        -> increase delta_rho

    """

    for rho_pt in rho_mid:
        mean_err = 1e3
        rho_up = rho_pt
        rho_dn = rho_pt
        print '*****Iteration for rho=%6.4fm' % (rho_pt)
        while(mean_err >= eps_err):
            # ind the rho interval to work on
            rho_up += delta_rho_0
            rho_dn -= delta_rho_0

            # Invert to time domain and find time indices to work on
            # OBS! inwards sweep, that is rho_up is reached at an earlier time than rho_dn
            t_up = rho_func.invert(rho_dn)
            t_dn = rho_func.invert(rho_up)

            print '    -> increasing t = %4.2f - %4.2fs -> rho = %5.3f - %5.3fmm' % (t_dn, t_up, rho_up, rho_dn)

            good_tidx = ((tb_signal > t_dn) & (tb_signal < t_up))

            # compute linear fit on isat
            tb_fit = tb_signal[good_tidx]
            signal_fit = probe_signal[good_tidx]
            A = np.vstack([tb_fit, np.ones_like(tb_fit)]).T
            #print A.shape, signal_fit.shape
            m, n = np.linalg.lstsq(A, signal_fit)[0]

            mean_err = np.abs(m * (t_dn - t_up)) / signal_fit.mean()
            print '    mean_err = %e' % (mean_err)

        plt.figure()
        plt.plot(tb_fit, signal_fit)
        plt.plot(tb_fit, m * tb_fit + n, 'r--')
        plt.plot([t_dn], np.array([t_dn]) * m + n, 'kv')
        plt.plot([t_up], np.array([t_up]) * m + n, 'k^')


# End of file kstar_binning.py
