#!/usr/bin/env python
#-*- Encoding: UTF-8 -*-

import numpy as np

"""
Thomas Algorithm for tridiagonal matrix systems

See Karniadakis, p.322
"""


def thomas1d(N, b, a, c, q):
    """
    Solve A x = q

              a0    c0   0   ...   0     0
              b1    a1   c1  ...   0     0
              0     b2   a1  ...   0     0
                       ...
    where A =          ...
              0     0    0         a_N-2 c_N-2
              0     0    0         b_N-1 a_N-1



    Storage of matrix elements in vectors as
    b = [b1, b2, ...b_N-1, 0]
    a = [a0, a1, ...a_N-2, a_N-1]
    c = [c0, c1, ...c_N-1, 0]
    """

    l = np.zeros(N, dtype='float64')
    u = np.zeros(N, dtype='float64')
    d = np.zeros(N, dtype='float64')
    y = np.zeros(N, dtype='float64')
    x = np.zeros(N, dtype='float64')

    #print 'b = ', b
    #print 'a = ', a
    #print 'c = ', c

    d[0] = a[0]
    u[0] = c[0]
    for i in np.arange(N - 2):
        l[i] = b[i] / d[i]
        #print 'b[i] = %f, d[i] = %f, l[i] = %f' % (b[i], d[i], l[i])
        d[i + 1] = a[i + 1] - l[i] * u[i]
        #print 'a[i+1] = %f, l[i] = %f, u[i] = %f => d[i+1 ] %f' % (a[i + 1],
        #                                                           l[i], u[i],
        #                                                           d[i + 1])
        u[i + 1] = c[i + 1]

    l[N - 2] = b[N - 2] / d[N - 2]
    d[N - 1] = a[N - 1] - l[N - 2] * u[N - 2]

    # Forward substitution
    y[0] = q[0]
    for i in np.arange(1, N):
        y[i] = q[i] - l[i - 1] * y[i - 1]
        #print 'q[i] = %f, l[i-1] = %f, y[i-1] = %f => y[i] = q - l[i-1] * y[i-1] = %f' %\
        #    (q[i], l[i - 1], y[i - 1], y[i])

    # Backward substitution
    x[N - 1] = y[N - 1] / d[N - 1]
    for i in np.arange(N - 2, -1, -1):
        x[i] = (y[i] - u[i] * x[i + 1]) / d[i]
        #print 'y[i] = %f, u[i] * x[i + 1] = %f, d[i] = %f => x=(y-ux)/d = %f' %\
        #    (y[i], u[i] * x[i + 1], d[i], x[i])

    return x, (l, u, d, y)


def thomas1d_2(N, b, a, c, q):
    """
    Solve A x = q

              a0    c0   0   ...   0     0
              b1    a1   c1  ...   0     0
              0     b2   a1  ...   0     0
                       ...
    where A =          ...
              0     0    0         a_N-2 c_N-2
              0     0    0         b_N-1 a_N-1



    Storage of matrix elements in vectors as
    b = [b0, b1, ...b_N-1]
    a = [a0, a1, ...a_N-1]
    c = [c0, c1, ...c_N-1]
    where b0 =0, c_N-1 = 2
    """

    l = np.zeros(N, dtype='float64')
    u = np.zeros(N, dtype='float64')
    d = np.zeros(N, dtype='float64')
    y = np.zeros(N, dtype='float64')
    x = np.zeros(N, dtype='float64')

    # LU decomposition
    d[0] = a[0]
    u[0] = c[0]
    for i in np.arange(1, N - 1):
        l[i] = b[i] / d[i - 1]
        d[i] = a[i] - l[i] * u[i - 1]
        u[i] = c[i]
    l[-1] = b[-1] / d[-2]
    d[-1] = a[-1] - l[-1] * u[-2]

    # Forward substitution, Ly = q
    y[0] = q[0]
    for i in np.arange(1, N):
        y[i] = q[i] - l[i] * y[i - 1]

    # Forward substitution Ux = y
    x[-1] = y[-1] / d[-1]
    for i in np.arange(N - 2, -1, -1):
        x[i] = (y[i] - u[i] * x[i + 1]) / d[i]

    return x, (l, u, d, y)

# End of file thomas.py
