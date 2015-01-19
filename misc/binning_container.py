#!/usr/bin/env python

"""
Define a binning container class.
The idea is to bin arrays according to a filter function.

Example:

blob_bc = binning_container(num_bins, bin_length, bin_edges,
                            bin_function, mode)
num_bins: Number of bins
bin_length: Length of each bin (each bin is a np.ndarray)
bin_function: Function operating on an interval
mode: Either add or append. If add, the instances adds the
      argument interval to the interval in each bin
      If mode=='append', the instance appends the argument
      interval to the list for each bin


When calling blob_bc( array ), call bin_function(array) to
determine which bin it adds to.

For example
# Initialize a binning container for Arrays of length 20 with 5 bins:
0: 0.0 < x.max() <= 2.5
1: 2.5 < x.max() <= 3.0
2: 3.0 < x.max() <= 3.5
3: 3.5 < x.max() <= 4.0
4: 4.0 < x.max() <= 4.5
5: 4.5 < x.max() <= 5.0

blob_bc = binning_containter(6, 20,
                             np.array([0.0, 2.5, 3.0, 3.5, 4.0, 4.5,
                             5.0]), lambda x: x.max() )

Add an array to the binning container, evaluate bin_function
with the same array:

blob_bc.bin( np.random.random(20) )

Add an array, evaluate bin_function with a different array:

blob_bc.bin(np.random.random(20), feval_array = np.random.random(20) )


# Return bin #0
blob_bc[0]

# Return number of arrays added to bin #0
blob_bc.num(0)
"""

import numpy as np


class binning_container(object):
    #def __init__(self, num_bins, bin_length, bin_edges, bin_function,
    #             mode='add'):
    def __init__(self, num_bins, bin_length, bin_edges, bin_function,
                 mode='add'):
        assert(mode in ['add', 'append'])

        self.num_bins = num_bins
        self.bin_length = bin_length
        self.bin_edges = zip(bin_edges[:-1], bin_edges[1:])
        self.bin_function = bin_function
        #self.bin_function = bin_function
        self.mode = mode

        self.bin_max = bin_edges.max()
        self.bin_min = bin_edges.min()
        # Create list of bins
        self.bins = []
        # Fill the bins
        for ibin in np.arange(num_bins):
            # If we add to the bins, insert an intervall we keep adding to
            if (self.mode == 'add'):
                self.bins.append(np.zeros(bin_length, dtype='float64'))
            elif (self.mode == 'append'):
                self.bins.append([])

        self.count = np.zeros(num_bins, dtype='int')

    def max(self, array):
        return array.max()

    def bin(self, array, feval_array=None):
        # Bin the data in array into the according bin
        # If supplied, use feval_array to determine the bin
        # array is binned into.
        # If feval_array == None, use array to determine the
        # bin used

        assert(array.size == self.bin_length)

        # Find the bin we bin ''array'' in
        if feval_array is None:
            # If feval_array is unspecified, pass ''array'' to bin_function
            rv = self.bin_function(array)
        else:
            # if feval_array is specified, pass ''feval_array'' to bin_function
            rv = self.bin_function(feval_array)

        # Perform boundary checks of rv against the upper and lower bin
        # boundary
        if (rv > self.bin_max):
            raise ValueError('Could not bin array: %f > max(bin_edges)' % rv)

        if (rv < self.bin_min):
            #raise ValueError('Could not bin array: %f < min(bin_edges)' % rv)
            raise ValueError('Could not bin array: %f < %f' % (rv, self.bin_min))
        #try:
        #    # If feval_array is not specified, the line below raises an
        #    # AttributeError
        #    rv = self.bin_function(feval_array)
        #    if (rv > self.bin_max):
        #        raise ValueError('Could not bin array: %f > max(bin_edges)' %
        #                          feval_array.max())
        #    if (rv < self.bin_min):
        #        raise ValueError('Could not bin array: %f < min(bin_edges)' %
        #                          feval_array.min())
        #except AttributeError:
        #    #print 'Did not use feval_array'
        #    rv = self.bin_function(array)
        #    if (rv > self.bin_max):
        #        raise ValueError('Could not bin array: %f > max(bin_edges)' %
        #        array.max())
        #    if (rv < self.bin_min):
        #        raise ValueError('Could not bin array: %f < min(bin_edges)' %
        #        array.min())

        idx = np.where(np.array([(rv > t1) & (rv <= t2) for t1, t2 in
                                 self.bin_edges]))[0]
        # Add to the appropriate bin
        if (self.mode == 'add'):
            (self.bins[idx])[:] = (self.bins[idx])[:] + array
        elif(self.mode == 'append'):
            self.bins[idx].append(array)
        # Increase bin counter
        self.count[idx] = self.count[idx] + 1

    def count(self, bin_idx=None):
        # return the count of each bin
        if bin_idx is None:
            return self.count
        else:
            return self.count[bin_idx]

    def get_num_bins(self):
        return len(self.num_bins)

    def cond_var(self, bin_idx=None):
        # Compute the conditional variance, see Oynes et al. PRL 75, 81
        # (1995)

        pass

    def __getitem__(self, idx):
        if (self.mode == 'add'):
            return self.bins[idx]
        elif (self.mode == 'append'):
            return np.array(self.bins[idx])

# End of file binning_container.py
