#! /usr/bin/env python
# -*- Coding: UTF-8 -*-

"""
matplotlib parameters to produce publishing quality figures
"""

from math import sqrt
import matplotlib as mpl

# Figure size with 300dpi
fig_dpi = 300.
fig_ipp = 1. / 72.72
golden_ratio = 0.5*(1+sqrt(5))


def set_rcparams_poster(myParams):
    fig_height = 1100 # in points
    fig_width = fig_height * golden_ratio
    print 'Figure size: %d x %d' % (fig_width, fig_height)
    print 'Figure resolution: %d dpi' % fig_dpi

    myParams['font.family'] = 'sans-serif'
    myParams['font.size'] = 14
    myParams['axes.linewidth'] = 3
    myParams['axes.labelsize'] = 14
    myParams['legend.fontsize'] = 14
    myParams['lines.linewidth'] = 4
    myParams['lines.markersize'] = 8
    myParams['figure.dpi'] = 300
    myParams['figure.figsize'] = fig_width * fig_ipp, fig_height*fig_ipp
    myParams['text.usetex'] = True
    myParams['savefig.dpi'] = 300
    myParams['pdf.fonttype'] = 42
    myParams['ps.fonttype'] = 42

# Figure in RevTex4 is 246 points wide


# Full column width figure for revtex articles
def set_rcparams_article_full(myParams):
    fig_width = 246.0 # in points
    fig_height = fig_width / golden_ratio

    fig_width_in = fig_width / 72.72
    fig_height_in = fig_height / 72.72
    print 'Figure size: %d x %d' % (fig_width, fig_height)
    print 'Figure resolution: %d dpi' % fig_dpi

    myParams['font.family'] = 'sans-serif'
    myParams['font.size'] = 12
    myParams['axes.linewidth'] = 1
    myParams['axes.labelsize'] = 12
    myParams['legend.fontsize'] = 8 
    myParams['lines.linewidth'] = 2
    myParams['lines.markersize'] = 5
    myParams['figure.dpi'] = 300
    myParams['figure.figsize'] = [fig_width_in, fig_height_in]
    myParams['text.usetex'] = True
    myParams['savefig.dpi'] = 300
    myParams['pdf.fonttype'] = 42

    return fig_width, fig_height

# Full column, square, width figure for revtex articles
def set_rcparams_article_fullsq(myParams):
    fig_width = 246.0 # in points
    fig_height = fig_width #/ golden_ratio

    fig_width_in = fig_width / 72.72
    fig_height_in = fig_height / 72.72
    print 'Figure size: %d x %d' % (fig_width, fig_height)
    print 'Figure resolution: %d dpi' % fig_dpi

    myParams['font.family'] = 'sans-serif'
    myParams['font.size'] = 12
    myParams['axes.linewidth'] = 1
    myParams['axes.labelsize'] = 12
    myParams['legend.fontsize'] = 8 
    myParams['lines.linewidth'] = 2
    myParams['lines.markersize'] = 5
    myParams['figure.dpi'] = 300
    myParams['figure.figsize'] = [fig_width_in, fig_height_in]
    myParams['text.usetex'] = True
    myParams['savefig.dpi'] = 300
    myParams['pdf.fonttype'] = 42

    return fig_width, fig_height


# Half column width figure for revtex articles
def set_rcparams_article_half(myParams):
    fig_width = 123.0 # in points
    fig_height = fig_width / golden_ratio
    fig_width_in = fig_width / 72.72
    fig_height_in = fig_height / 72.72
    print 'Figure size: %d x %d' % (fig_width, fig_height)
    print 'Figure resolution: %d dpi' % fig_dpi

    myParams['font.family'] = 'sans-serif'
    myParams['font.size'] = 6
    myParams['axes.linewidth'] = 0.5
    myParams['axes.labelsize'] = 6
    myParams['legend.fontsize'] = 6 
    myParams['lines.markersize'] = 2
    myParams['lines.linewidth'] = 0.5
    myParams['figure.dpi'] = 300
    myParams['figure.figsize'] = [fig_width_in, fig_height_in]
    myParams['text.usetex'] = True
    myParams['savefig.dpi'] = 300
    myParams['pdf.fonttype'] = 42

    return fig_width, fig_height



