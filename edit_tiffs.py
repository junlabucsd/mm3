#!/usr/bin/env python3

# import modules
from __future__ import print_function
import sys, os, time, shutil, glob, re
from pprint import pprint
import numpy as np
# import json
import warnings
from skimage import io
from skimage.external import tifffile as tiff

# takes images with name format "20181011_JDW3308_StagePosition01_Frame0205_Second61500_Channel1.tif"
#    and saves them as a multichannel stack in a subdirectory called "TIFF"
#    with name format "20181011_JDW3308_t0205xy01.tif"

# SO, I need to update my alignImages.ipynb in motherMachineSegger
#     and mm3_Compile.py to accept these saved multichannel stacks

# functions for printing
def warning(*objs):
    print("%6.2f Error:" % time.clock(), *objs, file=sys.stderr)
def information(*objs):
    print(time.strftime("%H:%M:%S", time.localtime()), *objs, file=sys.stdout)

# runs if script is run from terminal
if __name__ == '__main__':
    '''Edit TIFFs from Jeremy's format to the one expected by mm3.'''

    # define variables here
    source_dir = sys.argv[1]
    dest_dir = os.path.join(source_dir,'TIFF')
    file_prefix = '20181011_JDW3308' # prefix for output images

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # get all .tif files
    source_files = glob.glob(os.path.join(source_dir, '*.tif'))
    source_files = [file_path.split('/')[-1] for file_path in source_files]

    # cropping
    t_crop = [None, None] # in the first 30 frames the image moves by more than 1 channel spacing
    y_crop = [None, None]
    x_crop = [None, None]

    # grab unique names excluding channels
    files_noch = [name[:-5] for name in source_files]
    files_noch.sort()
    # files_noch = sorted(set(files_noch))
    pprint(files_noch)

    # go through these images and save em out after processing
    for name in files_noch:
        # open all three images
        imgnames = glob.glob(os.path.join(source_dir, name + '*.tif'))
        imgnames.sort()
        pprint(imgnames)

        fileparts = name.split('_')
        fov = int(fileparts[2][-2:])
        t = int(fileparts[3][-4:])

        # skip if we aren't using this time point
        if (t_crop[0] and t < t_crop[0]) or (t_crop[1] and t > t_crop[1]):
            continue

        imgs = []
        for imgname in imgnames:
            with tiff.TiffFile(imgname) as tif:
                imgs.append(tif.asarray())

        img_data = np.stack(imgs, axis=0) # combine channels into a stacked tiff
        # img_data = np.rollaxis(img_data, 0, 3) # channels go to back

        # crop image. Set defaults incase there are Nones
        if not y_crop[0]: y_crop[0] = 0
        if not y_crop[1]: y_crop[1] = img_data.shape[1]
        if not x_crop[0]: x_crop[0] = 0
        if not x_crop[1]: x_crop[1] = img_data.shape[2]

        img_data = img_data[:, y_crop[0]:y_crop[1], x_crop[0]:x_crop[1]]
        # img_data = np.expand_dims(img_data, 0)

        # print(img_data.shape)

        # save out images
        tif_filename = file_prefix + "_t%04dxy%02d.tif" % (t, fov)
        information('Saving %s.' % tif_filename)
        print(os.path.join(dest_dir, tif_filename))
        io.imsave(os.path.join(dest_dir, tif_filename), img_data)
