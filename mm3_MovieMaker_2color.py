#!/usr/bin/python
from __future__ import print_function
def warning(*objs):
    print(time.strftime("%H:%M:%S Error:", time.localtime()), *objs, file=sys.stderr)
def information(*objs):
    print(time.strftime("%H:%M:%S", time.localtime()), *objs, file=sys.stdout)

# import modules
import sys
import os
import time
import inspect
import getopt
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

# debug
# import matplotlib as mpl
# mpl.rcParams['figure.figsize'] = 15, 15
# mpl.rcParams['pdf.fonttype'] = 42
# mpl.rcParams['xtick.direction'] = 'out'
# mpl.rcParams['ytick.direction'] = 'out'
# import matplotlib.pyplot as plt

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

    By Steven
    Edited 20151128 jt
    Edited 20160830 jt
    '''

    if sys.platform == 'darwin':
        FFMPEG_BIN = "/usr/local/bin/ffmpeg" # location where FFMPEG is installed
        fontface = Face("/Library/Fonts/Andale Mono.ttf")
    elif sys.platform == 'linux2': # Ubuntu (Docker install)
        FFMPEG_BIN = '/usr/bin/ffmpeg'
        fontface = Face("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")

    # soft defaults, overridden by command line parameters if specified
    param_file = ""
    specify_fovs = []
    start_fov = -1

    # hard parameters
    shift_time = 1081 # put in a timepoint to indicate the timing of a shift (colors the text)
    phase_plane_index = 0 # index of the phase plane
    fl_plane_index = 1 # index of the fluorescent plane
    fl_interval = 4 # how often the fluorescent image is taken. will hold image over rather than strobe

    # switches
    try:
        opts, args = getopt.getopt(sys.argv[1:], "f:o:s:")
    except getopt.GetoptError:
        print('No or wrong arguments detected (-f -o -s).')
    for opt, arg in opts:
        if opt == '-f':
            param_file = arg
        if opt == '-o':
            arg.replace(" ", "")
            [specify_fovs.append(int(argsplit)) for argsplit in arg.split(",")]
        if opt == '-s':
            try:
                start_fov = int(arg)
            except:
                warning("Could not convert start parameter (%s) to an integer." % arg)
                raise ValueError

    # Load the project parameters file into a dictionary named p
    if len(param_file) == 0:
        raise ValueError("A parameter file must be specified (-f <filename>).")
    information('Loading experiment parameters...')
    with open(param_file) as pfile:
        p = yaml.load(pfile)

    # assign shorthand directory names
    TIFF_dir = p['experiment_directory'] + p['image_directory'] # source of images
    movie_dir = p['experiment_directory'] + p['movie_directory']

    # set up movie folder if it does not already exist
    if not os.path.exists(movie_dir):
        os.makedirs(movie_dir)

    # find FOV list
    fov_list = [] # list will hold integers which correspond to FOV ids
    fov_images = glob.glob(TIFF_dir + '*.tif')
    for image_name in fov_images:
        # add FOV number from filename to list
        fov_list.append(int(image_name.split('xy')[1].split('.tif')[0]))

    # sort and remove duplicates
    fov_list = sorted(list(set(fov_list)))
    information('Found %d FOVs to process.' % len(fov_list))

    # start the movie making
    for fov in fov_list: # for every FOV
        # skip FOVs as specified above
        if len(specify_fovs) > 0 and not (fov) in specify_fovs:
            continue
        if start_fov > -1 and fov < start_fov:
            continue

        # grab the images for this fov
        images = glob.glob(TIFF_dir + '*xy%03d*.tif' % (fov))
        if len(images) == 0:
            images = glob.glob(TIFF_dir + '*xy%02d*.tif' % (fov)) # for filenames with 2 digit FOV
        if len(images) == 0:
            raise ValueError("No images found to export for FOV %d." % fov)
        information("Found %d files to export." % len(images))

        # get min max pixel intensity for scaling the data
        imin = {}
        imax = {}
        # imin['phase'], imax['phase'] = find_img_min_max(images[::100])
        imin['phase'], imax['phase'] = 500, 5000
        imin['488'], imax['488'] = 130, 250

        # use first image to set size of frame
        image = tiff.imread(images[0])
        if image.shape[0] < 10:
            image = image[phase_plane_index] # get phase plane
        size_x, size_y = image.shape[1], image.shape[0] # does not worked for stacked tiff

        # set command to give to ffmpeg
        command = [FFMPEG_BIN,
                '-y', # (optional) overwrite output file if it exists
                '-f', 'rawvideo',
                '-vcodec','rawvideo',
                '-s', '%dx%d' % (size_x, size_y), # size of one frame
                '-pix_fmt', 'rgb48le',
                '-r', '%d' % p['fps'], # frames per second
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
                movie_dir + p['experiment_name'] + '_xy%03d.mp4' % fov]

        information('Writing movie for FOV %d.' % fov)

        pipe = sp.Popen(command, stdin=sp.PIPE)

        # display a frame and send it to write
        for t, img in enumerate(images, start=1):
            # skip images not specified by param file.
            if t < p['image_start'] or t > p['image_end']:
                continue

            image = tiff.imread(img) # get the image

            # print('image shape', np.shape(image))

            phase = image[phase_plane_index ] # get phase plane

            # process phase image
            phase = phase.astype('float64')
            # normalize
            phase -= imin['phase']
            phase[phase < 0] = 0
            phase /= (imax['phase'] - imin['phase'])
            phase[phase > 1] = 1
            phase = phase * 0.75
            # three color stack
            # print('phase shape', np.shape(phase))
            phase = np.dstack((phase, phase, phase))

            if (t - 1) % fl_interval == 0:
                fl488 = image[fl_plane_index].astype('float64') # pick red image
                # normalize
                fl488 -= imin['488']
                fl488[fl488 < 0] = 0
                fl488 /= (imax['488'] - imin['488'])
                fl488[fl488 > 1] = 1
                # print('fl shape', np.shape(fl488))
                # three color stack
                fl488 = np.dstack((np.zeros_like(fl488), fl488, np.zeros_like(fl488)))

            image = 1 - ((1 - fl488) * (1 - phase))

            # put in time stamp
            seconds = float(t * p['seconds_per_time_index'])
            mins = seconds / 60
            hours = mins / 60
            timedata = "%dhrs %02dmin" % (hours, mins % 60)
            r_timestamp = np.fliplr(make_label(timedata, fontface, size=48,
                                               angle=180)).astype('float64')
            r_timestamp = np.pad(r_timestamp, ((size_y - 10 - r_timestamp.shape[0], 10),
                                               (size_x - 10 - r_timestamp.shape[1], 10)),
                                               mode = 'constant')
            r_timestamp /= 255.0

            # create label
            if shift_time and t >= shift_time:
                r_timestamp = np.dstack((r_timestamp, r_timestamp, np.zeros_like(r_timestamp)))
            else:
                r_timestamp = np.dstack((r_timestamp, r_timestamp, r_timestamp))

            image = 1 - ((1 - image) * (1 - r_timestamp))

            # Plot image for debug
            # fig = plt.figure(figsize = (10,5))
            # ax = fig.add_subplot(1,1,1)
            # ax.imshow((image * 65535).astype('uint16'))
            # plt.show()

            # shoot the image to the ffmpeg subprocess
            pipe.stdin.write((image * 65535).astype('uint16').tostring())
        pipe.stdin.close()
