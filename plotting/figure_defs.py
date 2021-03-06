#! /opt/local/bin/python
# -*- Coding: UTF-8 -*-

"""
matplotlib parameters to produce publishing quality figures
"""

from math import sqrt
import matplotlib as mpl

# Figure size with 300dpi
fig_dpi = 300.
fig_ipp = 1. / 72.72
golden_ratio = 0.5 * (1. + sqrt(5.0))


def set_mplrcparams(rcparams):
    rcparams['text.usetex'] = True
    rcparams['text.latex.preamble'] = r"\usepackage{lmodern}"
    rcparams['font.family'] = 'lmodern'
    rcparams['font.size'] = 20
    rcparams['axes.labelsize'] = 20
    rcparams['axes.titlesize'] = 20
    rcparams['lines.linewidth'] = 3
    rcparams['lines.markersize'] = 6
    rcparams['xtick.labelsize'] = 20
    rcparams['ytick.labelsize'] = 20
    rcparams['legend.fontsize'] = 20

    return rcparams


def set_rcparams_poster(myParams):
    fig_width = 300 # in points
    fig_height = fig_width / golden_ratio
    
    print 'Figure size: %d x %d' % (fig_width, fig_height)
    print 'Figure resolution: %d dpi' % fig_dpi

    myParams['font.family'] = 'sans-serif'
    myParams['font.size'] = 18
    myParams['axes.linewidth'] = 3
    myParams['axes.labelsize'] = 18
    myParams['legend.fontsize'] = 14
    myParams['lines.linewidth'] = 3
    myParams['lines.markersize'] = 6
    myParams['figure.dpi'] = 300
    myParams['figure.figsize'] = fig_width * fig_ipp, fig_height*fig_ipp
    myParams['text.usetex'] = True
    myParams['savefig.dpi'] = 300
    myParams['pdf.fonttype'] = 42
    myParams['ps.fonttype'] = 42

# Figure in RevTex4 is 246 points wide

def set_rcparams_article_full(myParams):
    """
    Full column width figure for revtex
    """
    fig_width = 246.0 # in points
    fig_height = fig_width / golden_ratio

    fig_width_in = fig_width / 72.72
    fig_height_in = fig_height / 72.72
    print 'Figure size: %d x %d' % (fig_width, fig_height)
    print 'Figure resolution: %d dpi' % fig_dpi

    myParams['font.family'] = 'sans-serif'
    myParams['font.size'] = 10
    myParams['axes.linewidth'] = 1
    # myParams['axes.labelsize'] = 12
    myParams['legend.fontsize'] = 8 
    myParams['lines.linewidth'] = 1
    myParams['lines.markersize'] = 3
    myParams['figure.dpi'] = 300
    myParams['figure.figsize'] = [fig_width_in, fig_height_in]
    myParams['text.usetex'] = True
    myParams['savefig.dpi'] = 300
    myParams['pdf.fonttype'] = 42

    return fig_width, fig_height

def set_rcparams_article_full_macos(myParams):
    """
    Full column width figure, for macbook
    """
    fig_width = 350 # in points
    fig_height = fig_width / golden_ratio

    fig_width_in = fig_width / 72.72
    fig_height_in = fig_height / 72.72
    print 'Figure size: %d x %d' % (fig_width, fig_height)
    print 'Figure resolution: %d dpi' % fig_dpi

    myParams['font.family'] = 'sans-serif'
    myParams['font.size'] = 12
    myParams['axes.linewidth'] = 1
    myParams['axes.labelsize'] = 12
    myParams['legend.fontsize'] = 10
    myParams['lines.linewidth'] = 2
    myParams['lines.markersize'] = 6
    myParams['figure.dpi'] = 300
    myParams['figure.figsize'] = [fig_width_in, fig_height_in]
    myParams['text.usetex'] = True
    myParams['savefig.dpi'] = 300
    myParams['pdf.fonttype'] = 42

    return fig_width, fig_height

def set_rcparams_paper(myParams):
    """
    One 8cm column for the TCV paper
    """
    fig_width = 227 # in points
    fig_height = fig_width / golden_ratio
    fig_width_in = fig_width / 72.72
    fig_height_in = fig_height / 72.72
    #print 'Figure size: %d x %d' % (fig_width, fig_height)
    #print 'Figure resolution: %d dpi' % fig_dpi

    myParams['font.family'] = 'Time'
    myParams['font.size'] = 6
    myParams['axes.linewidth'] = 0.5
    myParams['axes.labelsize'] = 6
    myParams['legend.fontsize'] = 6
    myParams['legend.handlelength'] = 4
    myParams['lines.markersize'] = 2
    myParams['lines.linewidth'] = 0.5
    myParams['figure.dpi'] = 300
    myParams['figure.figsize'] = [fig_width_in, fig_height_in]
    myParams['text.usetex'] = True
    myParams['savefig.dpi'] = 300
    myParams['pdf.fonttype'] = 42
    myParams['patch.linewidth'] = 0.5 #For legend box borders

    return fig_width, fig_height

def set_rcparams_paper_thickline(myParams):
    """
    One 8cm column for the TCV paper
    """
    fig_width = 227 # in points
    fig_height = fig_width / golden_ratio
    fig_width_in = fig_width / 72.72
    fig_height_in = fig_height / 72.72
    #print 'Figure size: %d x %d' % (fig_width, fig_height)
    #print 'Figure resolution: %d dpi' % fig_dpi

    myParams['font.family'] = 'Time'
    myParams['font.size'] = 8
    myParams['axes.linewidth'] = 0.5
    myParams['axes.labelsize'] = 8
    myParams['legend.fontsize'] = 8
    myParams['legend.handlelength'] = 3
    myParams['lines.markersize'] = 2
    myParams['lines.linewidth'] = 1.5
    myParams['figure.dpi'] = 300
    myParams['figure.figsize'] = [fig_width_in, fig_height_in]
    myParams['text.usetex'] = True
    myParams['savefig.dpi'] = 300
    myParams['pdf.fonttype'] = 42
    myParams['patch.linewidth'] = 0.5 #For legend box borders

    return fig_width, fig_height

def set_rcparams_article_fullsq(myParams):
    """
    Full column, square with figure for revtex
    """
    fig_width = 369.0 # 246.0 # in points
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

def set_rcparams_article_half(myParams):
    """
    Half column width figures for revtex, s
    http://publishing.aip.org/authors/preparing-graphics
    """
    # Figure size in inch
    fig_width_in = 3.37
    fig_height_in = fig_width_in / golden_ratio
    # figure size in pts
    fig_width_pt = fig_width_in * fig_dpi 
    fig_height_pt = fig_height_in * fig_dpi 
    print 'Figure size: %4.2f"" x %4.2f""' % (fig_width_in, fig_height_in)
    print 'Figure resolution: %d dpi' % fig_dpi

    myParams['font.family'] = 'sans-serif'
    myParams['font.size'] = 6
    myParams['axes.linewidth'] = 0.5
    myParams['axes.labelsize'] = 6
    myParams['legend.fontsize'] = 4 
    myParams['lines.markersize'] = 2
    myParams['lines.linewidth'] = 0.5
    myParams['figure.dpi'] = 300
    myParams['figure.figsize'] = [fig_width_in, fig_height_in]
    myParams['text.usetex'] = True
    myParams['savefig.dpi'] = 300
    myParams['pdf.fonttype'] = 42

    return fig_width_in, fig_height_in


def set_rcparams_beamer(myParams):
    """
    Full screen figure in latex beamer
    Size of beamer slide: 128mm by 96mm
    Default font size in beamer is 11pt
    http://userpages.umbc.edu/~rostamia/beamer/quickstart-Z-H-24.html#node_sec_24

    Creates a figure with 
    width = 9.6 / 2.54                   = 3.78"
    height = width / 1.61 (golden ratio) = 2.34"
    """

    fig_width_in = 9.6 / 2.54
    fig_height_in =fig_width_in / golden_ratio 
    fig_width = fig_width_in * 72.72
    fig_height = fig_height_in * 72.72


    myParams['font.family'] = 'sans-serif'
    myParams['font.size'] = 11
    myParams['axes.linewidth'] = 0.5
    myParams['axes.labelsize'] = 11
    myParams['legend.fontsize'] = 11 
    myParams['lines.markersize'] = 3
    myParams['lines.linewidth'] = 1
    myParams['figure.dpi'] = 300
    myParams['figure.figsize'] = [fig_width_in, fig_height_in]
    myParams['text.usetex'] = True
    myParams['savefig.dpi'] = 300
    myParams['pdf.fonttype'] = 42




def set_rcparams_pres_small(myParams):
    """
    Use large axis labels and thick lines for small plots in presentations
    """
#    myParams['font.family'] = 'monospace'
#    myParams['text.usetex'] = False
    myParams['font.size'] = 18
    #myParams['axes.labelsize'] = 16
    
    myParams['lines.linewidth'] = 2.0
    #myParams['font.size'] = 20
    #myParams['legend.fontsize'] = 14
    myParams['xtick.labelsize'] = 'large'
    myParams['ytick.labelsize'] = 'large'    




# End of file figure_defs.py
