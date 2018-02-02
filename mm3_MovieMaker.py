#!/usr/bin/python
from __future__ import print_function

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

### functions ##################################################################
def warning(*objs):
    print(time.strftime("%H:%M:%S Error:", time.localtime()), *objs, file=sys.stderr)

def information(*objs):
    print(time.strftime("%H:%M:%S", time.localtime()), *objs, file=sys.stdout)

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

    # put in a timepoint to indicate the timing of a shift (colors the text)
    shift_time = 157

    # Fluorescent image parameters (two color movies)
    two_colors = True # set to true if you want to do two color movies.
    phase_plane_index = 0 # index of the phase plane
    fl_plane_index = 1 # index of the fluorescent plane
    fl_interval = 1 # how often the fluorescent image is taken. will hold image over rather than strobe

    # soft defaults, overridden by command line parameters if specified
    param_file = ""
    specify_fovs = []
    start_fov = -1

    # switches
    try:
        unixoptions='f:o:s:'
        gnuoptions=['paramfile=','fov=','start-fov=']
        opts, args = getopt.getopt(sys.argv[1:], unixoptions, gnuoptions)
    except getopt.GetoptError:
        print('No or wrong arguments detected (-f -o -s).')
    for opt, arg in opts:
        if opt in ['-f','--paramfile']:
            param_file = arg
        if opt in ['-o','--fov']:
            arg.replace(" ", "")
            [specify_fovs.append(int(argsplit)) for argsplit in arg.split(",")]
        if opt in ['-s','--start-fov']:
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
    TIFF_dir = os.path.join(p['experiment_directory'], p['image_directory']) # source of images
    movie_dir = os.path.join(p['experiment_directory'], p['movie_directory'])

    # set up movie folder if it does not already exist
    if not os.path.exists(movie_dir):
        os.makedirs(movie_dir)

    # find FOV list
    fov_list = [] # list will hold integers which correspond to FOV ids
    fov_images = glob.glob(os.path.join(TIFF_dir,'*.tif'))
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
        images = glob.glob(os.path.join(TIFF_dir, '*xy%03d*.tif' % (fov)))
        if len(images) == 0:
            images = glob.glob(os.path.join(TIFF_dir, '*xy%02d*.tif' % (fov))) # for filenames with 2 digit FOV
        if len(images) == 0:
            raise ValueError("No images found to export for FOV %d." % fov)
        information("Found %d files to export." % len(images))

        # get min max pixel intensity for scaling the data
        imin = {}
        imax = {}

        if not two_colors:
            # automatically scale images
            imin['phase'], imax['phase'] = find_img_min_max(images[::100])
        else:
            # it is hardcoded.
            imin['phase'], imax['phase'] = 100, 1750
            imin['488'], imax['488'] = 100, 350

        # use first image to set size of frame
        image = tiff.imread(images[0])
        if two_colors or image.shape[0] < 10:
            image = image[phase_plane_index] # get phase plane
        size_x, size_y = image.shape[1], image.shape[0] # does not work for stacked tiff

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
                os.path.join(movie_dir,p['experiment_name']+'_xy%03d.mp4' % fov)]

        information('Writing movie for FOV %d.' % fov)

        pipe = sp.Popen(command, stdin=sp.PIPE)

        # display a frame and send it to write
        for t, img in enumerate(images, start=1):
            # skip images not specified by param file.
            if (t < p['image_start']) or (t > p['image_end']):
                continue

            image_data = tiff.imread(img) # get the image

            if two_colors or image_data.shape[0] < 10 or len(image_data.shape[0]) > 1:
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

            if two_colors:
                # dim phase
                phase = phase * 0.75

            # three color stack
            phase = np.dstack((phase, phase, phase))

            """
            # test
            import matplotlib.pyplot as plt
            fl488 = image_data[fl_plane_index].astype('float64') # pick red image
            hist,edges = np.histogram(np.ravel(fl488),bins='auto')
            #for e0,e1,h in zip(edges[:-1],edges[1:],hist):
            #    print ("{:<20.6g}{:<20.6g}{:<20.6g}".format(e0,e1,h))
            fig = plt.figure()
            idx = hist > 0.
            ax1=fig.add_subplot(2,1,1)
            ax1.plot(edges[:-1][idx],hist[idx],'b-')
            ax1.set_xlabel("I")
            ax1.set_ylabel("pdf")
            ax2=fig.add_subplot(2,1,2)
            myimg = ax2.imshow(fl488)
            ax2.axis('off')
            plt.colorbar(myimg, ax=ax2)
            fig.tight_layout()
            fdir = "debug"
            fileout = os.path.join(fdir,"fov{:03d}_time{:04d}.pdf".format(fov,t))
            fig.savefig(fileout,bbox_inches='tight',pad_inches=0)
            plt.close('all')
            # test
            #"""

            if two_colors:
                # combine with fluorescent image
                if (t - 1) % fl_interval == 0:
                    fl488 = image_data[fl_plane_index].astype('float64') # pick red image
                    # normalize
                    fl488 -= imin['488']
                    fl488[fl488 < 0] = 0
                    fl488 /= (imax['488'] - imin['488'])
                    fl488[fl488 > 1] = 1
                    # print('fl shape', np.shape(fl488))
                    # three color stack
                    fl488 = np.dstack((np.zeros_like(fl488), fl488, np.zeros_like(fl488)))

                image = 1 - ((1 - fl488) * (1 - phase))
            else:
                # just send the phase image forward
                image = phase

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

            # shoot the image to the ffmpeg subprocess
            pipe.stdin.write((image * 65535).astype('uint16').tostring())
        pipe.stdin.close()
