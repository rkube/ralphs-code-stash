#!/usr/bin/env python
#-*- Encoding: UTF-8 -*-

import numpy as np


def float_exp_mant(f):

    """
    Return the exponent and the mantissa for a floating point number,
    i.e. given f we write it as
    f = mant * 10^{exp}, where mant = x.abcd... with x in {1..9}
                         and a,b,c,d,... in {0,...,9}

    input:
    =======
    f         floating point number

    output:
    =======

    exp       Smallest exponent so that first digit of f is non-zero
    mant      Mantissa of f
    """

    if (f < 0):
        expon = np.floor(np.log10(-f))
    else:
        expon = np.floor(np.log10(f))
    mant = f / 10 ** expon
    return expon, mant


def formatter_sd(f, n):
    """
    Returns a float of the number formatted to significant digits

    Input:
    ======
    f...... float, number to format
    n...... int, number of significant digits

    Output:
    =======
    fstr... string, formatted string
    """

    assert(n > 1)

    exp, mant = float_exp_mant(f)

    mant_r = np.around(mant, n - 1)

    fstr = "{0:g}".format(mant_r * 10**exp)
    return(fstr)

# End of file scientific_notation.py
