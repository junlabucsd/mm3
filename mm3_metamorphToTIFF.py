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
import argparse
import inspect

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

# this is the mm3 module with all the useful functions and classes
import mm3_helpers as mm3

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

    # set switches and parameters
    parser = argparse.ArgumentParser(prog='python mm3_Compile.py',
                                     description='Identifies and slices out channels into individual TIFF stacks through time.')
    parser.add_argument('-f', '--paramfile',  type=str,
                        required=True, help='Yaml file containing parameters.')
    namespace = parser.parse_args()

    # Load the project parameters file
    mm3.information('Loading experiment parameters.')
    if namespace.paramfile:
        param_file_path = namespace.paramfile
    else:
        mm3.warning('No param file specified. Using 100X template.')
        param_file_path = 'yaml_templates/params_SJ110_100X.yaml'
    p = mm3.init_mm3_helpers(param_file_path) # initialized the helper library

    # define variables here
    source_dir = p['experiment_directory']
    dest_dir = os.path.join(source_dir,'TIFF')
    file_prefix = p['experiment_name'] # prefix for output images
    file_name_filters = p['metamorphToTIFF']['file_name_filters']

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    pat = re.compile(r'.+s(\d+)_t(\d+)\.TIF')
    time_between_frames = p['metamorphToTIFF']['seconds_between_frames']
    strain_name = p['metamorphToTIFF']['strain_name']

    file_name_dict = {}
    for i in file_name_filters:
        file_name_dict[i] = glob.glob(os.path.join(source_dir, '*{}*.TIF'.format(i)))
        #file_name_dict[i] = [file_path.split('/')[-1] for file_path in file_name_dict[i]]
        file_name_dict[i].sort()

    # cropping
    t_crop = p['metamorphToTIFF']['t_crop']
    y_crop = p['metamorphToTIFF']['x_crop']
    x_crop = p['metamorphToTIFF']['y_crop']
    for i in range(len(t_crop)):
        if t_crop[i] == "None":
            t_crop[i] = None
        if y_crop[i] == "None":
            y_crop[i] = None
        if x_crop[i] == "None":
            x_crop[i] = None

    new_name_list = []

    for i in range(len(file_name_dict[file_name_filters[0]])):

        file_name_list = [file_name_dict[file_name_filter][i] for file_name_filter in file_name_filters]
        #pprint(file_name_list) # uncomment for debugging
        init_file_name = file_name_list[0]

        nameStr = init_file_name.replace(' ','_')
        #print(nameStr) # uncomment for debugging
        mat = pat.match(nameStr)
        stagePosition,frame = mat.group(1,2)
        stagePosition = int(stagePosition)
        frame = int(frame)

        # skip if we aren't using this time point
        if (t_crop[0] and frame < t_crop[0]) or (t_crop[1] and frame > t_crop[1]):
            continue

        new_name = os.path.join(dest_dir, '{}_t{:0=4}xy{:0=2}.tif'.format(file_prefix,
                                                                          frame,
                                                                          stagePosition))

        imgs = []
        for imgname in file_name_list:
            with tiff.TiffFile(imgname) as tif:
                imgs.append(tif.asarray())

        img_data = np.stack(imgs, axis=0) # combine channels into a stacked tiff

        # crop image. Set defaults incase there are Nones
        if not y_crop[0]: y_crop[0] = 0
        if not y_crop[1]: y_crop[1] = img_data.shape[1]
        if not x_crop[0]: x_crop[0] = 0
        if not x_crop[1]: x_crop[1] = img_data.shape[2]

        img_data = img_data[:, y_crop[0]:y_crop[1], x_crop[0]:x_crop[1]]

        # save out image
        information("Saving {}.".format(new_name))
        # print(os.path.join(dest_dir,new_name))
        io.imsave(os.path.join(dest_dir, new_name), img_data)
