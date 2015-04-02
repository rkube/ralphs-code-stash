#!/usr/bin/env python
#-*- Encoding: UTF-8 -*-

import re
import numpy as np

"""
Parse cucumplx.h ascii output as to reconstruct a np.cmplx128 ndarray from ascii
output
"""

def read_cmplx_arr(fname):

    #fname = '/home/rku000/cuda-workspace/cuda_array2/run/run_ns/lamb/run4/dt1e-4/128/arr_c.dat'
    df = open(fname)

    # Get line count
    num_lines = sum(1 for line in df) - 2
    result_array = np.zeros([num_lines, num_lines / 2 + 1], dtype=np.complex128)

    # skip first and last line
    df.seek(0)
    l1 = df.readline()

    i = 0
    for l in df.readlines():
        # Break, if we are at the last line, which is empty
        if(len(l) < 10):
            break
        res = re.finditer(r'\((.*?)\)', l)

        clist = []
        for r in res:
            c1 = float(r.group(1).split(',')[0]) + 1.j * float(r.group(1).split(',')[1])
            clist.append(c1)

        result_array[i, :] = np.array(clist)
        i += 1

    return (result_array)



# End of file read_cmplx.py
