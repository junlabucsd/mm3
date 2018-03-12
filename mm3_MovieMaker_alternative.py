import os,sys,glob
import cPickle as pkl
import argparse
import yaml
import numpy as np
import time
import shutil
import scipy.io as spio
import re
import subprocess as sp
from freetype import *
import inspect
import warnings
from PIL import Image

# This makes python look for modules in ./external_lib
cmd_subfolder = os.path.realpath(os.path.abspath(
                                 os.path.join(os.path.split(inspect.getfile(
                                 inspect.currentframe()))[0], "external_lib")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import tifffile as tiff

import mm3_helpers
from mm3_helpers import get_fov, get_time, information
from mm3_utils import print_time, make_label

# yaml formats
npfloat_representer = lambda dumper,value: dumper.represent_float(float(value))
nparray_representer = lambda dumper,value: dumper.represent_list(value.tolist())
float_representer = lambda dumper,value: dumper.represent_scalar(u'tag:yaml.org,2002:float', "{:<.6g}".format(value))
yaml.add_representer(float,float_representer)
yaml.add_representer(np.float_,npfloat_representer)
yaml.add_representer(np.ndarray,nparray_representer)

################################################
# main
################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Making movie from .tif files.")
    parser.add_argument('-f', '--paramfile',  type=file, required=True, help='Yaml file containing parameters.')
    parser.add_argument('-o', '--fov',  type=int, required=True, help='Field of view with which make the movie.')
    namespace = parser.parse_args(sys.argv[1:])
    paramfile = namespace.paramfile.name
    allparams = yaml.load(namespace.paramfile)
    fov = namespace.fov

    # first initialization of parameters
    params = allparams['movie']

################################################
# make movie directory
################################################
    print print_time(), "Making directory..."
    movie_dir = params['directory']

    if not movie_dir.startswith('/'):
        movie_dir = os.path.join('.', movie_dir)
    if not os.path.isdir(movie_dir):
        os.makedirs(movie_dir)

################################################
# make list of images
################################################
    exp_name = allparams['experiment_name']
    tiffs = allparams['image_directory']
    pattern = exp_name + '_t(\d+)xy\w+.tif'
    filelist = []
    for root, dirs, files in os.walk(tiffs):
        for f in files:
            res = re.match(pattern, f)
            if not (res is None):
                # determine fov
                if (fov == get_fov(f)):
                    filelist.append(os.path.join(root,f))

        # do not go beyond first level
        break

    # open one image to get dimensions
    fimg=filelist[0]
    img = tiff.imread(fimg) # read the image
    if (len(img.shape) == 2):
        img = np.array([img])
    if (len(img.shape) != 3):
        sys.exit('wrong image format/dimensions!')
    img = np.moveaxis(img, 0, 2)
    size_y_ref, size_x_ref = img.shape[:2]
    # make sure the output image has dimensions multiple of 2
    # in addition, ffmpeg will issue a warning of 'data is not aligned' if dimensions
    # are not multiple of 8, 16 or 32
    # so let's choose 8 as basis instead.
    size_y_ref = (size_y_ref / 8) * 8
    size_x_ref = (size_x_ref / 8) * 8

################################################
# make movie
################################################
    # set command to give to ffmpeg
    # path to FFMPEG
    FFMPEG_BIN = sp.check_output("which ffmpeg", shell=True).replace('\n','')

    # path to font for label
    fontfile = "/Library/Fonts/Andale Mono.ttf"    # Mac OS location
    if not os.path.isfile(fontfile):
        # fontfile = "/usr/share/fonts/truetype/freefont/FreeMono.ttf"  # Linux Ubuntu 16.04 location
        fontfile = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf" # Another font
        if not os.path.isfile(fontfile):
            sys.exit("You need to install some fonts and specify the correct path to the .ttf file!")
    fontface = Face(fontfile)

    # ffmpeg command
    # 'ffmpeg -f image2 -pix_fmt gray16le -r 2 -i test/16b/20180223_SEM4158_col1_mopsgluc_dnp_t%04dxy01.tif -an -c:v h264 -pix_fmt yuv420p movies/test.mp4'
    command = [FFMPEG_BIN,
            '-y', # (optional) overwrite output file if it exists
            '-f', 'rawvideo',
            '-c:v','rawvideo',
            '-s', '%dx%d' % (size_x_ref, size_y_ref), # size of one frame
            '-pix_fmt', 'rgb24',
            '-r', '%d' % params['fps'], # frames per second
            '-i', '-', # The imput comes from a pipe
            '-an', # Tells FFMPEG not to expect any audio
            # options for the h264 codec
            '-c:v', 'h264',
            '-pix_fmt', 'yuv420p',

            # set the movie name
            os.path.join(movie_dir,allparams['experiment_name']+'_xy%03d.mp4' % fov)]

    information('Writing movie for FOV %d.' % fov)

    #print " ".join(command)
    # comment following for debug
    pipe = sp.Popen(command, stdin=sp.PIPE)

    for fimg in filelist:
        #print fimg
        t = get_time(fimg)
        if not (params['t0'] is None) and (t < params['t0']):
            continue
        if not (params['tN'] is None) and (t > params['tN']):
            continue

        img = tiff.imread(fimg) # read the image
        if (len(img.shape) == 2):
            img = np.array([img])
        if (len(img.shape) != 3):
            sys.exit('wrong image format/dimesions!')
        img = np.moveaxis(img, 0, 2)
        size_y, size_x = img.shape[:2]
        if not (size_y == size_y_ref and size_x == size_x_ref):
            img = img[:size_y_ref, :size_x_ref]
        size_y, size_x = img.shape[:2]
        nchannels = img.shape[2]
        stack=[]
        masks={}
        for i in range(nchannels):
            img_tp = img[:,:,i]
            mask =  None
            # masking operations (used when building overlay)
            try:
                xlo = np.uint16(params['channels'][i]['xlo'])
            except KeyError:
                xlo = None

            mask = (img_tp > xlo) # get binary mask
            masks[i] = mask

            # rescale dynamic range
            try:
                pmin = np.uint16(params['channels'][i]['min'])
            except (KeyError,TypeError):
                pmin = np.min(img_tp)
            try:
                pmax = np.uint16(params['channels'][i]['max'])
            except (KeyError,TypeError):
                pmax = np.max(img_tp)
            img_tp = (np.array(img_tp, dtype=np.float_) - pmin)/float(pmax-pmin)
            idx = img_tp > 1
            img_tp [img_tp < 0] = 0.
            img_tp [img_tp > 1] = 1.

            # color
            try:
                color = params['channels'][i]['rgb']
            except KeyError:
                color = [255,255,255]

            #img_tp = (1. - img_tp)
            norm = float(2**8 - 1)
            img_tp *= norm
            #rgba = np.dstack([img_tp*color[0]/255., img_tp*color[1]/255., img_tp*color[2]/255., np.ones(img_tp.shape)*255.])
            #rgba = np.array(rgba,dtype=np.uint8)
            #stack.append(rgba)
            #rgb = np.dstack([img_tp*color[0]/255., img_tp*color[1]/255., img_tp*color[2]/255.])
            rgb = np.dstack([img_tp*color[0]/norm, img_tp*color[1]/norm, img_tp*color[2]/norm])
            rgb = np.array(rgb,dtype=np.uint8)
            stack.append(rgb)

        # construct final image
        bg = params['background']
        img_bg = stack[bg]

        # add overlays
        overlay = []
        try:
            overlay = params['overlay']
        except KeyError:
            pass

        if not ( (overlay is None) or (overlay == []) ):
            img = np.zeros(img_bg.shape, dtype=np.float_)
            tot_coeffs = np.zeros(img_bg.shape, dtype=np.float_)
            for i in overlay:
                img_tp = stack[i]
                w = params['channels'][i]['alpha']
                mask = masks[i]
                coeffs = np.ones(img_tp.shape,dtype=np.float_)
                if not (mask is None):
                    coeffs[~mask] = 0.

                coeffs *= w
                img += coeffs*img_tp.astype('float')
                tot_coeffs += coeffs

            img += (1-tot_coeffs)*img_bg.astype('float')
            img = np.array(img, dtype=np.uint8)

        else:
            img = img_bg

        # add time stamp
        size_y,size_x = img.shape[:2]
        seconds = float((t-1) * allparams['seconds_per_time_index']) # t=001 is the first capture
        mins = seconds / 60
        hours = mins / 60
        timedata = "%dhrs %02dmin" % (hours, mins % 60)
        r_timestamp = np.fliplr(make_label(timedata, fontface, size=48,
                                           angle=180)).astype('float64')
        r_timestamp = np.pad(r_timestamp, ((size_y - 10 - r_timestamp.shape[0], 10),
                                           (size_x - 10 - r_timestamp.shape[1], 10)),
                                           mode = 'constant')

        mask = (r_timestamp > 0)
        r_timestamp = np.dstack((r_timestamp, r_timestamp, r_timestamp)).astype(np.uint8)
        coeffs = np.zeros(img.shape,dtype=np.float_)
        img[mask] = r_timestamp[mask]

        # decomment for debug
#        img = Image.fromarray(img, mode='RGB')
#        img.save('test.png')
#        break

        # write the image to the ffmpeg subprocess
        # comment for debug
        pipe.stdin.write(img.tostring())

    # end of loop
    # comment for debug
    pipe.terminate()
