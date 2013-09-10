#!/usr/bin/env python
#-*- Encoding: UTF-8 -*-

import numpy as np

"""
Load APD text files stored in
/Volumes/My Book Thunderbolt Duo/cmod_data/Odd Eriks CMOD disk/APD/

APD is a 9x10 array, values stored in format
%f %f %f %f %f %f
%f %f %f
... repeat 8 times
%f %f %f %f %f %f
%f %f %f

Next block


Iterator yields single blocks, formatted as a 9x10 np.ndarray

"""


def apd_read_two_lines(ffile):
    """
    Read two lines formatted as
    %f %f %f %f %f %f
    %f %f %f
    from file object
    """

    line1 = ffile.readline().split()
    if (len(line1) == 0):
        raise ValueError
    line2 = ffile.readline().split()
    if (len(line2) != 3):
        raise ValueError('%d: too few elements in line 2' % (len(line2)))

    return np.array([float(l) for l in line1 + line2])


def apd_block_from_text(fname):
    """
    Read 9 by 10 blocks from the ascii file, return each block
    """
    with open(fname, 'r') as ffile:
        while True:
            # Allocate a ndarray for the current block
            block_arr = np.zeros([9, 10])
            try:
                # Fill it with values. If the line size mismatches, we assume
                # we are at the end of the file
                for i in np.arange(10):
                    block_arr[:, i] = apd_read_two_lines(ffile)
            except ValueError:
                # Could not read a block, assume that we are done
                raise StopIteration
            # Skip a line
            ffile.readline()
            # Yield the current block
            yield(block_arr)


def apd_block_count(fname):
    """
    Count the lines in the given file. A block is 21 lines
    Returns the block count, i.e. num_lines / 21
    """
    block_count = sum(1 for line in open(fname)) / 21

    return block_count


# End file read_apd_text.py
