'''movie_from_tiffs does what the title says. You must have ffmpeg installed,
which you can get using homebrew:
https://trac.ffmpeg.org/wiki/CompilationGuide/MacOSX

By Steven
Edited 20151128 jt
'''

import subprocess as sp
import jdcal, datetime, math, copy, gc, getopt, fnmatch, yaml, os
from freetype import *
import matplotlib as mpl
import tifffile as tiff
mpl.rcParams['figure.figsize'] = 15, 15
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['xtick.direction'] = 'out'
mpl.rcParams['ytick.direction'] = 'out'
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cm as cm
#%matplotlib inline
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageMath

# hard coded variables
debug = False # debug mode
FFMPEG_BIN = "/usr/local/bin/ffmpeg"

# soft defaults, can be overridden by parameters .yaml file loading
fps = 24
seconds_per_time_index = 60

# soft defaults, overridden by command line parameters if specified
specify_fovs = False
user_spec_fovs = []

tifdir = '/Volumes/Latour/MMExperiments/20151223_BEC8_streptomycin/tiff/'

# switches
try:
    opts, args = getopt.getopt(sys.argv[1:],"o:")
except getopt.GetoptError:
    print('No arguments detected (-o <fov(s)>).')
for opt, arg in opts:
    if opt == '-o':
        try:
            specify_fovs = True
            for fov_to_proc in arg.replace(" ", "").split(","):
                user_spec_fovs.append(int(fov_to_proc))
        except:
            print("Couldn't convert argument to an integer:",arg)
            raise ValueError

### functions ##################################################################
def make_label(text, face, size=12, angle=0):
    '''
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
    #face = Face(filename)
    face.set_char_size( size*64 )
    angle = (angle/180.0)*math.pi
    matrix  = FT_Matrix( (int)( math.cos( angle ) * 0x10000 ),
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

def distance2(a, b):
    return (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2])

def makeColorTransparent(image, color, thresh2=0):
    image = image.convert("RGBA")
    red, green, blue, alpha = image.split()
    image.putalpha(ImageMath.eval("""convert(((((t - d(c, (r, g, b))) >> 31) + 1) ^ 1) * a, 'L')""",
        t=thresh2, d=distance2, c=color, r=red, g=green, b=blue, a=alpha))
    return image

def days_to_hmsm(days):
    hours = days * 24.
    hours, hour = math.modf(hours)
    mins = hours * 60.
    mins, min = math.modf(mins)
    secs = mins * 60.
    secs, sec = math.modf(secs)
    micro = round(secs * 1.e6)
    return int(hour), int(min), int(sec), int(micro)

def hmsm_to_days(hour=0,min=0,sec=0,micro=0):
    days = sec + (micro / 1.e6)
    days = min + (days / 60.)
    days = hour + (days / 60.)
    return days / 24.

def datetime_to_jd(date):
    days = date.day + hmsm_to_days(date.hour,date.minute,date.second,date.microsecond)
    return date_to_jd(date.year,date.month,days)

def date_to_jd(year,month,day):
    if month == 1 or month == 2:
        yearp = year - 1
        monthp = month + 12
    else:
        yearp = year
        monthp = month
    # this checks where we are in relation to October 15, 1582, the beginning
    # of the Gregorian calendar.
    if ((year < 1582) or
        (year == 1582 and month < 10) or
        (year == 1582 and month == 10 and day < 15)):
        # before start of Gregorian calendar
        B = 0
    else:
        # after start of Gregorian calendar
        A = math.trunc(yearp / 100.)
        B = 2 - A + math.trunc(A / 4.)
    if yearp < 0:
        C = math.trunc((365.25 * yearp) - 0.75)
    else:
        C = math.trunc(365.25 * yearp)
    D = math.trunc(30.6001 * (monthp + 1))
    jd = B + C + D + day + 1720994.5
    return jd

def find_img_min_max(image_names):
    '''find_img_max_min returns the average minimum and average maximum
    intensity for a set of tiff images.
    Parameters
    ----------
    image_names : list
        list of image names
    Returns
    -------
    avg_min : float
        average minimum intensity value
    avg_max : float
        average maximum intensity value
    '''
    min_list = []
    max_list = []
    for i, image_name in enumerate(image_names):
        print(image_name)
        print("loading prescan %d" % i)
        image = tiff.imread(experiment_directory + image_directory + image_name)
        min_list.append(np.min(image))
        max_list.append(np.max(image))
    avg_min = np.mean(min_list)
    avg_max = np.min(max_list)
    return avg_min, avg_max


### main #######################################################################
if __name__ == "__main__":
    seconds_per_time_index = 60
    if not specify_fovs:
        raise

    fov = user_spec_fovs[0]

    # grab the images
    images = fnmatch.filter(os.listdir(tifdir), "*Point%04d*.tif" % fov)
    if len(images) == 0:
        raise ValueError("No images found to export for fov %d." % fov)
    print("-------------------------------------------------")
    print("Found %d files to export." % len(images))

    # get a rough range for scaling the data, phase then fluorescence
    ph_min, ph_max = 5000, 40000
    fl_min, fl_max = 65, 170
    print 'phase min/max:', ph_min, ph_max

    # use first image to set size of frame
    image = tiff.imread(tifdir + images[0])
    size_x, size_y = image[1].shape[1], image[1].shape[0]
    fontface = Face("/Library/Fonts/Andale Mono.ttf")

    # set command to give to ffmpeg
    command = [ FFMPEG_BIN,
            '-y', # (optional) overwrite output file if it exists
            '-f', 'rawvideo',
            '-vcodec','rawvideo',
            '-s', '%dx%d' % (size_x, size_y), # size of one frame
            '-pix_fmt', 'rgb48le',
            '-r', '%d' % fps, # frames per second
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
            '/Users/sdbrown/Dropbox/20151223_xy%02d_late.mp4' % fov ]
    pipe = sp.Popen(command, stdin=sp.PIPE)

    # display a frame
    for n, i in enumerate(images):
        if n < 750:
            continue
        statinfo = os.stat(tifdir + i)
        if statinfo.st_size < 3500000:
            continue
        
        image = tiff.imread(tifdir + i)
        phase = image[0].astype('float64')
        # normalize
        phase -= ph_min
        phase[phase < 0] = 0
        phase /= ph_max
        phase[phase > 1] = 1
        # three color stack
        phase = np.dstack((phase, phase, phase))
        
        fl561 = image[1].astype('float64')
        # normalize
        fl561 -= fl_min
        fl561[fl561 < 0] = 0
        fl561 /= fl_max
        fl561[fl561 > 1] = 1
        # three color stack
        fl561 = np.dstack((fl561, np.zeros_like(fl561), np.zeros_like(fl561)))

        image = 1 - ((1 - fl561) * (1 - phase))

        image = np.flipud(image)

        # put in time stamp
        #seconds = copy.deepcopy(nd2n2[i].metadata['t_ms']) / 1000.
        #minutes = seconds / 60.
        seconds = float(int(i.split("ime")[1].split("_Poin")[0]) * seconds_per_time_index)
        mins = seconds / 60
        hours = mins / 60
        #days = hours / 24.
        #acq_time = datetime_to_jd(datetime.datetime.strptime(starttime, "%m/%d/%Y %I:%M:%S %p")) + days
        #ayear, amonth, aday, afraction = jdcal.jd2gcal(acq_time, 0.0)
        #ahour, aminute, asecond, amicro = days_to_hmsm(afraction)
        #timedata = "%04d/%02d/%02d %02d:%02d:%02d" % (ayear, amonth, aday, ahour, aminute, asecond)
        timedata = "%dhrs %02dmin" % (hours, mins % 60)
        r_timestamp = np.fliplr(make_label(timedata, fontface, size = 48, angle = 180)).astype('float64')
        r_timestamp = np.pad(r_timestamp, ((size_y - 10 - r_timestamp.shape[0], 10), (size_x - 10 - r_timestamp.shape[1], 10)), mode = 'constant')
        r_timestamp /= 255.
        #r_timestamp = 1 - r_timestamp
        r_timestamp = np.dstack((r_timestamp, r_timestamp, r_timestamp))

        image = 1 - ((1 - image) * (1 - r_timestamp))
        #plt.imshow(pix)

        # fig = plt.figure(figsize = (10,5))
        # ax = fig.add_subplot(1,1,1)
        # ax.imshow((image * 65535).astype('uint16'))
        # plt.show()

        # shoot the image to the ffmpeg subprocess
        pipe.stdin.write((image * 65535).astype('uint16').tostring())
        if n % 10 == 0:
            print n
    pipe.stdin.close()
