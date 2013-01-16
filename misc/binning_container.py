#!/usr/bin/env python

"""
Define a binning container class.
The idea bin intervals according to a filter function.

Example:

blob_bc = binning_container(num_bins, bin_length, bin_edges, bin_function)
num_bins: Number of bins
bin_length: Length of each bin (each bin is a np.ndarray)
bin_function: Function operating on an interval


When calling blob_bc( array ), call bin_function(array) to determine which bin it adds to.

For example
# Initialize a binning container for Arrays of length 20 with 5 bins:
0: 0.0 < x.max() <= 2.5
1: 2.5 < x.max() <= 3.0
2: 3.0 < x.max() <= 3.5
3: 3.5 < x.max() <= 4.0
4: 4.0 < x.max() <= 4.5
5: 4.5 < x.max() <= 5.0

blob_bc = binning_containter(6, 20, np.array([0.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]), lambda x: x.max() )

# Add an array to the binning container
blob_bc.bin( np.random.random(20) )

# Return bin #0
blob_bc[0]

# Return number of arrays added to bin #0
blob_bc.num(0)
"""

import numpy as np


class binning_container:
    def __init__(self, num_bins, bin_length, bin_edges, bin_function):
        self.num_bins = num_bins
        self.bin_length = bin_length
        self.bin_edges = np.array(bin_edges)
        self.bin_function = bin_function

        # Create list of bins
        self.bins = []
        for ibin in np.arange(num_bins):
            self.bins.append( np.zeros(bin_length, dtype='float64') )


        self.count = np.zeros(num_bins, dtype='int')

    
    def bin(self, array):
        assert( np.size(array) == self.bin_length)
   
        # Find the bin where we add to
        rv = self.bin_function(array)
        print 'Bin functions returned: ', rv
        print zip(self.bin_edges[:-1], self.bin_edges[1:])
        idx = np.where( np.array([(rv > t1) & (rv <= t2) for t1, t2 in zip( self.bin_edges[:-1], self.bin_edges[1:])]) )[0]
        print 'Binning to idx=%d' % (idx)

        # Add to the appropriate bin
        (self.bins[idx])[:] = (self.bins[idx])[:]  + array
        # Increase bin counter
        self.count[idx] = self.count[idx] + 1


    def count(self, bin_idx=-1):
        if (bin_idx > 0):
            return self.count(bin_idx)
        return self.count


    def __getitem__(self, idx):
        return self.bins[idx]

