import os,sys,glob
import numpy as np
import time
from freetype import *
import scipy.stats as sstats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker

##############################################################################
# general functions
##############################################################################
def print_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

##############################################################################
# signal processing
##############################################################################
def histogram(X,density=True):
    valmax = np.max(X)
    valmin = np.min(X)
    iqrval = sstats.iqr(X)
    nbins_fd = (valmax-valmin)*np.float_(len(X))**(1./3)/(2.*iqrval)
    if (nbins_fd < 1.0e4):
        return np.histogram(X,bins='auto',density=density)
    else:
        return np.histogram(X,bins='sturges',density=density)

##############################################################################
# Movie Maker
##############################################################################
def make_label(text, face, size=12, angle=0):
    '''Uses freetype to make a time label.

    Parameters:
    -----------
    text : string
        Text to be displayed
    filename : string
        Path to a font
    size : int
        Font size in 1/64th points
    angle : float
        Text angle in degrees
    '''
    face.set_char_size( size*64 )
    angle = (angle/180.0)*np.pi
    matrix = FT_Matrix( (int)( np.cos( angle ) * 0x10000 ),
                         (int)(-np.sin( angle ) * 0x10000 ),
                         (int)( np.sin( angle ) * 0x10000 ),
                         (int)( np.cos( angle ) * 0x10000 ))
    flags = FT_LOAD_RENDER
    pen = FT_Vector(0,0)
    FT_Set_Transform( face._FT_Face, byref(matrix), byref(pen) )
    previous = 0
    xmin, xmax = 0, 0
    ymin, ymax = 0, 0
    for c in text:
        face.load_char(c, flags)
        kerning = face.get_kerning(previous, c)
        previous = c
        bitmap = face.glyph.bitmap
        pitch  = face.glyph.bitmap.pitch
        width  = face.glyph.bitmap.width
        rows   = face.glyph.bitmap.rows
        top    = face.glyph.bitmap_top
        left   = face.glyph.bitmap_left
        pen.x += kerning.x
        x0 = (pen.x >> 6) + left
        x1 = x0 + width
        y0 = (pen.y >> 6) - (rows - top)
        y1 = y0 + rows
        xmin, xmax = min(xmin, x0),  max(xmax, x1)
        ymin, ymax = min(ymin, y0), max(ymax, y1)
        pen.x += face.glyph.advance.x
        pen.y += face.glyph.advance.y

    L = np.zeros((ymax-ymin, xmax-xmin),dtype=np.ubyte)
    previous = 0
    pen.x, pen.y = 0, 0
    for c in text:
        face.load_char(c, flags)
        kerning = face.get_kerning(previous, c)
        previous = c
        bitmap = face.glyph.bitmap
        pitch  = face.glyph.bitmap.pitch
        width  = face.glyph.bitmap.width
        rows   = face.glyph.bitmap.rows
        top    = face.glyph.bitmap_top
        left   = face.glyph.bitmap_left
        pen.x += kerning.x
        x = (pen.x >> 6) - xmin + left
        y = (pen.y >> 6) - ymin - (rows - top)
        data = []
        for j in range(rows):
            data.extend(bitmap.buffer[j*pitch:j*pitch+width])
        if len(data):
            Z = np.array(data,dtype=np.ubyte).reshape(rows, width)
            L[y:y+rows,x:x+width] |= Z[::-1,::1]
        pen.x += face.glyph.advance.x
        pen.y += face.glyph.advance.y

    return L

def array_bin(array, p=0):
    """
    Smooth the input image by averaging
    squares of size 2p+1.
    """

    # no binning
    if (p == 0):
        return array

    # start binning
    array_new = np.empty(array.shape, dtype=array.dtype)

    nrow, ncol = array.shape[:2]
    for r in range(nrow):
        for c in range(ncol):
            r0 = max(0,r-p)
            r1 = min(nrow-1,r+p)
            c0 = max(0,c-p)
            c1 = min(ncol-1,c+p)
            array_new[r,c] = np.mean(array[r0:r1+1, c0:c1+1])
    # end binning
    return array_new

def get_background(data, delta=1.5):
    """
    Return background value for data.
    """

    median = np.median(data)
    iqr = sstats.iqr(data)
    bg = median+delta*iqr
    return bg

def plot_histogram(data, fileout, nbinsx_max=8, color='darkblue', lw=0.5):
    """
    Plot the histogram of the input file
    """

    fig = plt.figure(num='none',facecolor='w', figsize=(4,3))
    ax = fig.gca()

    hist,edges=histogram(data,density=False)
    left = edges[:-1]
    idx = hist > 0

    # add plot
    #ax.bar(edges[:-1], hist, width=np.diff(edges), color=color, lw=0)
    ax.plot(left[idx], hist[idx], '-', color=color, lw=lw)

    # statistics
    median = np.median(data)
    iqr = sstats.iqr(data)
    bg = get_background(data)
    ax.axvline(x=median, linestyle='-', color='k', lw=lw, label="median = {:.0f}, IQR={:.0f}".format(median,iqr))
    ax.axvline(x=bg, linestyle='--', color='k', lw=lw, label="background = {:.0f}".format(bg))

    ax.legend(loc='best', fontsize='x-small')

    ax.set_xlabel('pixel value', fontsize='medium')
    ax.set_ylabel('histogram', fontsize='medium')
    #ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=nbins_max))
    #ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=nbins_max))
    #ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:.2g}'))
    #ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:.2g}'))
    ax.tick_params(axis='both', labelsize='xx-small', pad=2)
    ax.tick_params(axis='x', which='both', bottom='on', top='off')
    ax.tick_params(axis='y', which='both', left='on', right='off')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    rect = [0.,0.,1.,0.98]
    fig.tight_layout(rect=rect)
    print "{:<20s}{:<s}".format('fileout',fileout)
    fig.savefig(fileout,bbox_inches='tight',pad_inches=0)
    plt.close('all')
    return

