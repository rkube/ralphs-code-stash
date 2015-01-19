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
        expon = np.ceil(np.log10(-f))
    else:
        expon = np.ceil(np.log10(f))
    mant = f / 10 ** (expon - 1)
    return expon, mant

# End of file scientific_notation.py
