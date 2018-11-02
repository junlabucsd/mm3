#!/usr/bin/python
from __future__ import print_function

# import modules
import sys
import os
import time
import inspect
import argparse
import yaml
import traceback
import glob
import math
import subprocess as sp
import numpy as np
from freetype import *
import warnings

# user modules
# realpath() will make your script run, even if you symlink it
cmd_folder = os.path.realpath(os.path.abspath(
                              os.path.split(inspect.getfile(inspect.currentframe()))[0]))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

# This makes python look for modules in ./external_lib
cmd_subfolder = os.path.realpath(os.path.abspath(
                                 os.path.join(os.path.split(inspect.getfile(
                                 inspect.currentframe()))[0], "external_lib")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

# supress the warning this always gives
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import tifffile as tiff

# this is the mm3 module with all the useful functions and classes
import mm3_helpers as mm3

### functions ##################################################################
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
    angle = (angle/180.0)*math.pi
    matrix = FT_Matrix( (int)( math.cos( angle ) * 0x10000 ),
                         (int)(-math.sin( angle ) * 0x10000 ),
                         (int)( math.sin( angle ) * 0x10000 ),
                         (int)( math.cos( angle ) * 0x10000 ))
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

def find_img_min_max(image_filepaths):
    '''find_img_max_min returns the average minimum and average maximum
    intensity for a set of tiff images.

    Parameters
    ----------
    image_filepaths : list
        list of image file path (strings)

    Returns
    -------
    avg_min : float
        average minimum intensity value
    avg_max : float
        average maximum intensity value
    '''
    min_list = []
    max_list = []
    for image_filepath in image_filepaths:
        image = tiff.imread(image_filepath)
        min_list.append(np.min(image))
        max_list.append(np.max(image))
    avg_min = np.mean(min_list)
    avg_max = np.min(max_list)
    return avg_min, avg_max

### main #######################################################################
if __name__ == "__main__":
    '''You must have ffmpeg installed, which you can get using homebrew:
    https://trac.ffmpeg.org/wiki/CompilationGuide/MacOSX
    '''

    # hard parameters
    # path to FFMPEG
    FFMPEG_BIN = sp.check_output("which ffmpeg", shell=True).replace('\n','')
    #FFMPEG_BIN = "/usr/local/bin/ffmpeg" # location where FFMPEG is installed

    # path to font for label
    fontfile = "/Library/Fonts/Andale Mono.ttf"    # Mac OS location
    if not os.path.isfile(fontfile):
        # fontfile = "/usr/share/fonts/truetype/freefont/FreeMono.ttf"  # Linux Ubuntu 16.04 location
        fontfile = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf" # Another font
        if not os.path.isfile(fontfile):
            sys.exit("You need to install some fonts and specify the correct path to the .ttf file!")
    fontface = Face(fontfile)

    # set switches and parameters
    parser = argparse.ArgumentParser(prog='python mm3_MovieMaker.py',
                                     description='Make .mpg movies from TIFF images.')
    parser.add_argument('-f', '--paramfile',  type=file,
                        required=True, help='Yaml file containing parameters.')
    parser.add_argument('-o', '--fov',  type=str,
                        required=False, help='List of fields of view to analyze. Input "1", "1,2,3", etc. ')
    parser.add_argument('-j', '--nproc',  type=int,
                        required=False, help='Number of processors to use.')
    namespace = parser.parse_args()

    # Load the project parameters file
    mm3.information('Loading experiment parameters.')
    if namespace.paramfile.name:
        param_file_path = namespace.paramfile.name
    else:
        mm3.warning('No param file specified. Using 100X template.')
        param_file_path = 'yaml_templates/params_SJ110_100X.yaml'
    p = mm3.init_mm3_helpers(param_file_path) # initialized the helper library

    if namespace.fov:
        user_spec_fovs = [int(val) for val in namespace.fov.split(",")]
    else:
        user_spec_fovs = []

    # assign shorthand directory names
    TIFF_dir = os.path.join(p['experiment_directory'], p['image_directory']) # source of images
    movie_dir = os.path.join(p['experiment_directory'], p['moviemaker']['movie_directory'])

    # set up movie folder if it does not already exist
    if not os.path.exists(movie_dir):
        os.makedirs(movie_dir)

    # port over labels from yaml file
    show_time_stamp = p['moviemaker']['show_time_stamp']
    show_label = p['moviemaker']['show_label']
    label1_text = p['moviemaker']['label1_text']
    shift_time = p['moviemaker']['shift_time']
    label2_text = p['moviemaker']['label2_text']
    show_scalebar = p['moviemaker']['show_scalebar']
    scalebar_length_um = p['moviemaker']['scalebar_length_um']
    # color management
    show_phase = p['moviemaker']['show_phase']
    phase_plane_index = p['moviemaker']['phase_plane_index']
    show_green = p['moviemaker']['show_green']
    fl_green_index = p['moviemaker']['fl_green_index']
    fl_green_interval = p['moviemaker']['fl_green_interval']
    show_red = p['moviemaker']['show_red']
    fl_red_index =p['moviemaker']['fl_red_index']
    fl_red_interval = p['moviemaker']['fl_red_interval']

    # min and max pixel intensity for scaling the data
    auto_phase_levels = p['moviemaker']['auto_phase_levels']
    imin = {'phase': p['moviemaker']['ph_min'],
            'green': p['moviemaker']['gr_min'],
            'red': p['moviemaker']['rd_min']}
    imax = {'phase': p['moviemaker']['ph_max'],
            'green': p['moviemaker']['gr_max'],
            'red': p['moviemaker']['rd_max']}

    # find FOV list
    fov_list = [] # list will hold integers which correspond to FOV ids
    fov_images = glob.glob(os.path.join(TIFF_dir,'*.tif'))
    for image_name in fov_images:
        # add FOV number from filename to list
        fov_list.append(int(image_name.split('xy')[1].split('.tif')[0]))

    # sort and remove duplicates
    fov_list = sorted(list(set(fov_list)))
    mm3.information('Found %d FOVs to process.' % len(fov_list))

    # start the movie making
    for fov in fov_list: # for every FOV
        # skip fov if not in the group
        if user_spec_fovs and fov not in user_spec_fovs:
            continue

        # grab the images for this fov
        images = glob.glob(os.path.join(TIFF_dir, '*xy%03d*.tif' % (fov)))
        if len(images) == 0:
            images = glob.glob(os.path.join(TIFF_dir, '*xy%02d*.tif' % (fov))) # for filenames with 2 digit FOV
        if len(images) == 0:
            raise ValueError("No images found to export for FOV %d." % fov)
        mm3.information("Found %d files to export." % len(images))

        if auto_phase_levels:
            # automatically scale images
            imin['phase'], imax['phase'] = find_img_min_max(images[::100])

        # use first image to set size of frame
        image = tiff.imread(images[0]) # pull out an image
        size_x, size_y = image.shape[-1], image.shape[-2]
        size_x = (size_x / 2) * 2 # fixes bug if images don't have even dimensions with ffmpeg
        size_y = (size_y / 2) * 2

        # set command to give to ffmpeg
        command = [FFMPEG_BIN,
                '-y', # (optional) overwrite output file if it exists
                '-f', 'rawvideo',
                '-vcodec','rawvideo',
                '-s', '%dx%d' % (size_x, size_y), # size of one frame
                '-pix_fmt', 'rgb48le',
                '-r', '%d' % p['moviemaker']['fps'], # frames per second
                '-i', '-', # The imput comes from a pipe
                '-an', # Tells FFMPEG not to expect any audio
                # options for the h264 codec
                '-vcodec', 'h264',
                '-pix_fmt', 'yuv420p',

                # options for mpeg4 codec
                #'-vcodec', 'mpeg4',
                #'-qscale:v', '4', # set quality scale from 1 (high) to 31 (low)
                #'-b:v', '1024k', # set output bitrate
                #'-vf', 'scale=iw*0.5:ih*0.5', # rescale output
                #'-bufsize', '300k',

                # set the movie name
                os.path.join(movie_dir, p['experiment_name'] + '_xy%03d.mp4' % fov)]

        mm3.information('Writing movie for FOV %d.' % fov)

        pipe = sp.Popen(command, stdin=sp.PIPE)

        # display a frame and send it to write
        for img in images:
            # skip images not specified by param file.
            t = mm3.get_time(img)
            if ((p['moviemaker']['image_start'] and t < p['moviemaker']['image_start']) or
                (p['moviemaker']['image_end'] and t > p['moviemaker']['image_end'])):
                continue

            image_data = tiff.imread(img) # get the image

            # make phase stack
            if show_phase:
                if len(image_data.shape) > 2:
                    phase = image_data[phase_plane_index] # get phase plane
                else:
                    phase = image_data

                # process phase image
                phase = phase.astype('float64')
                # normalize
                phase -= imin['phase']
                phase[phase < 0] = 0
                phase /= (imax['phase'] - imin['phase'])
                phase[phase > 1] = 1

                if show_green or show_red:
                    # dim phase if putting on fluorescence
                    phase = phase * 0.75

                # three color stack
                phase = np.dstack((phase, phase, phase))

            # make green stack
            if show_green:
                if (t - 1) % fl_green_interval == 0:
                    if len(image_data.shape) > 2:
                        flgreen = image_data[fl_green_index].astype('float64') # pick red image
                    else:
                        flgreen = image_data

                    # normalize
                    flgreen -= imin['green']
                    flgreen[flgreen < 0] = 0
                    flgreen /= (imax['green'] - imin['green'])
                    flgreen[flgreen > 1] = 1

                    # three color stack
                    flgreen = np.dstack((np.zeros_like(flgreen), flgreen, np.zeros_like(flgreen)))

            # make red stack
            if show_red:
                if (t - 1) % fl_red_interval == 0:
                    if len(image_data.shape) > 2:
                        flred = image_data[fl_red_index].astype('float64') # pick red image
                    else:
                        flred = image_data

                    # normalize
                    flred -= imin['red']
                    flred[flred < 0] = 0
                    flred /= (imax['red'] - imin['red'])
                    flred[flred > 1] = 1

                    # three color stack
                    flred = np.dstack((flred, np.zeros_like(flred), np.zeros_like(flred)))

            # combine images as appropriate
            if show_phase and show_green and show_red:
                image = 1 - ((1 - flgreen) * (1 - flred) * (1 - phase))

            elif show_phase and show_green:
                image = 1 - ((1 - flgreen) * (1 - phase))

            elif show_phase and show_red:
                image = 1 - ((1 - flred) * (1 - phase))

            elif show_green and show_red:
                image = 1 - ((1 - flgreen) * (1 - flred))

            elif show_phase:
                # just send the phase image forward
                image = phase

            elif show_green:
                image = flgreen

            elif show_red:
                image = flred

            if show_time_stamp:
                # put in time stamp
                seconds = float((t-1) * p['moviemaker']['seconds_per_time_index'])
                mins = seconds / 60
                hours = mins / 60
                timedata = "%dhrs %02dmin" % (hours, mins % 60)
                timestamp = np.fliplr(make_label(timedata, fontface, size=20,
                                                   angle=180)).astype('float64')
                timestamp = np.pad(timestamp, ((size_y - 10 - timestamp.shape[0], 10),
                                                   (size_x - 10 - timestamp.shape[1], 10)),
                                                   mode = 'constant')
                timestamp /= 255.0

                # create label
                timestamp = np.dstack((timestamp, timestamp, timestamp))

                image = 1 - ((1 - image) * (1 - timestamp))

            if show_label:
                label1 = np.fliplr(make_label(label1_text, fontface, size=15,
                                              angle=180)).astype('float64')
                label1 = np.pad(label1, ((10, size_y - 10 - label1.shape[0]),
                                         (10, size_x - 10 - label1.shape[1])),
                                         mode='constant')
                label1 /= 255.0
                label1 = np.dstack((label1, label1, label1))

                if shift_time:
                    label2 = np.fliplr(make_label(label2_text, fontface, size=15,
                                                  angle=180)).astype('float64')
                    label2 = np.pad(label2, ((10, size_y - 10 - label2.shape[0]),
                                             (10, size_x - 10 - label2.shape[1])),
                                                    mode='constant')
                    label2 /= 255.0
                    label2 = np.dstack((label2, label2, label2))

                if shift_time and t >= shift_time:
                    image = 1 - ((1 - image) * (1 - label2))
                else:
                    image = 1 - ((1 - image) * (1 - label1))

            if show_scalebar:
                scalebar_height = 10
                scalebar_length = np.around(scalebar_length_um / p['pxl2um']).astype(int)
                scalebar = np.zeros((size_y, size_x), dtype='float64')
                scalebar[size_y - 10 - scalebar_height:size_y - 10,
                         10:10 + scalebar_length] = 1

                # scalebar legend
                scale_text = '{} um'.format(scalebar_length_um)
                scale_legend = np.fliplr(make_label(scale_text, fontface, size=20,
                                                    angle=180)).astype('float64')
                scale_legend = np.pad(scale_legend, ((size_y - 10 - scale_legend.shape[0], 10),
                    (20 + scalebar_length, size_x - 20 - scalebar_length - scale_legend.shape[1])),
                                      mode='constant')
                scale_legend /= 255.0
                scalebar = np.add(scalebar, scale_legend) # put em together
                scalebar = np.dstack((scalebar, scalebar, scalebar))
                image = 1 - ((1 - image) * (1 - scalebar))

            # shoot the image to the ffmpeg subprocess
            pipe.stdin.write((image * 65535).astype('uint16').tostring())

        pipe.stdin.close()
