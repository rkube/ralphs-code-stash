#!/usr/bin/env python
# -*- coding: utf-8 -*-


__author__  = "André Bergner"

__version__ = "0.1"


#import scipy
import numpy
import math



def closest_anti_prime( n ):
    """
    computes the to n closest integer m > n, which can be factored into the primes 2,5,7.
    Will be used by cwt in order to speed up the fft, which is fastes if the data length
    can be factored into many small primes.
    """

    l2 = math.log( 2.0 )
    l3 = math.log( 3.0 )
    l5 = math.log( 5.0 )
    ln = math.log(  n  )

    x_max = math.ceil( ln / l2 )
    m     = math.pow( 2.0 , x_max )	# first guess

    for  x in range(0 , int(x_max) + 1 ):
      y_max = math.ceil( (ln - l2*x) / l3 );
      for y in range(0 , int(y_max) + 1 ):
        z = math.ceil( (ln-l2*x-l3*y)/l5 );
        m_ = math.pow( 2.0 , x ) * math.pow( 3.0 , y ) * math.pow( 5.0 , z )	# FIXME can be exponentialized
        if m_ < m : m = m_;

    return int(m)


#  _________________________________________________________
#  now come some standard wavelets

# TODO nomralize Q parameter via standard deviation

def cauchy( w , Q = 30. ):
    """
    Cauchy-Paul Wavelet
    """
    return numpy.exp( Q*( numpy.log(w) - w + 1. ) )


# alternative name for cauchy wavelet
def paul( w , Q = 30. ):
    """
    Cauchy-Paul Wavelet
    """
    return cauchy( w , Q )


def morlet( w , Q = 5 ):
    """
    Morlet Wavelet
    """
    return numpy.exp( -(Q*(w-1.))**2 )


def log_morlet( w , Q = 5 ):
    """
    Log-Morlet Wavelet -- has a gaussian shaped kernel on a scale/logarithmic frequancy axis
    """
    return numpy.exp( -(Q*numpy.log(w))**2 )




def cwt(
  x ,
  frequencies = numpy.exp(numpy.arange( -5.5 , 0.0 , 0.01 )) ,
  wavelet = cauchy,
  dev = 100
):
    """

    Computes a continuous wavelet transform



    @param x : input data

    @type  x : array of real or complex type


    @param frequencies : Frequencies/Scales at which the CWT is computed, normalized to 1 = Nyquist frequency

    @type  frequencies : array of reals


    @param wavelet : Wavelet

    @type  wavelet : callback function to wavelet



    Examples:

    cwt( data ) -- CWT of data with default parameters

    cwt( data , numpy.arange( 0.01 , 1.0 , 0.)
    """



    N_x   = len(x)
    N_pad = closest_anti_prime( N_x + 120 ) - N_x
    N     = N_x + N_pad		# data length including padding

    X = numpy.fft.fft( numpy.concatenate(( x , numpy.zeros(N_pad) )) )	# fft of padded input data
    w = numpy.arange( 0 , N/2 ) * 2./N 
    # TODO check if frequency scaling is correct ( either Nyquist or zero included or both ? )

    WT = [] 	# the resulting transform


    for f in frequencies:

        a = 1.0 / f
        WT.append( numpy.fft.ifft( numpy.concatenate((X[:N/2] * wavelet(a*w,Q=dev) ,  numpy.zeros(N/2))) )[:N_x] )	# <-- this makes real w'lets progressive, FIXME

    return  ( numpy.array(WT) , frequencies )		# make this a class behaving like the actual transform with freq and wlet as memebers



def icwt(	# TODO include padded zeros in inverse transorm
  WT , 		# as returned by cwt
  wavelet = cauchy
):		# FIXME how to compute scale measure for summation ?
    N_x   = len( WT[0][0,:] )
    N_pad = closest_anti_prime( N_x + 120 ) - N_x
    N     = N_x + N_pad		# data length including padding

    w = numpy.arange( 0 , N/2 ) * 2./N 

    x = zeros( N )
    for n in range(len(WT[1])):   # TODO use some form of struct: w.frequencies:

        Vox = numpy.fft.fft( numpy.concatenate(( WT[0][n,:] , numpy.zeros(N_pad) )) )	# fft of padded input data

        a  =  1.0 / WT[1][n]
        x  +=  numpy.concatenate(( Vox[:N/2] * wavelet(a*w) ,  numpy.zeros(N/2) ))	# FIXME, see cwt()
        # TODO x += scipy... * C(a_,{a_n})

    return  numpy.fft.ifft(x)[:N_x]








from pylab import *



def plotcwt( x , fignum = None ):

    W = cwt( x )[0]


    figure( num = fignum , figsize = (16,8) )

    a1 = axes([0.1, 0.3, 0.8, 0.7])
    ratio = (float(W.shape[1]) / float(W.shape[0]))
    imshow( abs(W) , aspect = 0.4*ratio , origin='lower')

    a2 = axes([0.1, 0.1, 0.8, 0.2])
    plot( x , 'k' )
    xlim([ 0 , len(x) ])

    show()

    return W



if __name__ == '__main__':

    import sys

    if len(sys.argv) < 2:
        print "usage: cwt <filename>"
        sys.exit( -1 )

    try:  x = numpy.loadtxt( sys.argv[1] )
    except IOError:
        print "error: could not open file '" + sys.argv[1] + "'"
        sys.exit( -1 )

    plotcwt( x )


