import os,sys,glob
import numpy as np
import time
from freetype import *
import scipy.stats as sstats

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
    iqrval = iqr(X)
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



