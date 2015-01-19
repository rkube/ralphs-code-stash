#!/usr/bin/env python
#-*- Encoding: UTF-8 -*-

import numpy as np

"""
Generates a numpy ndarray from a regular text file.
Use this when np.loadtxt throws a memory error

See:
http://stackoverflow.com/questions/8956832/python-out-of-memory-on-large-csv-file-numpy
"""


def iter_loadtxt(filename, skiprows = 0, delimiter = ' ', dtype=float):
    def iter_fun():
        with open(filename, 'r') as infile:
            for _ in range(skiprows):
                next(infile)
            for line in infile:
                line = line.rstrip().split(delimiter)
                for item in line:
                    yield dtype(item)
        iter_loadtxt.rowlength = len(line)

    data = np.fromiter(iter_fun(), dtype=dtype)
    data.reshape((-1, iter_loadtxt.rowlength))
    return data
#
