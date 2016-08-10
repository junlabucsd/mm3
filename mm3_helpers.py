#!/usr/bin/python
from __future__ import print_function
def warning(*objs):
    print(time.strftime("%H:%M:%S mm3_helpers:", time.localtime()), *objs, file=sys.stderr)
def information(*objs):
    print(time.strftime("%H:%M:%S mm3_helpers:", time.localtime()), *objs, file=sys.stdout)

# import modules
import sys
import os
import time
import inspect
import yaml
try:
    import cPickle as pickle
except:
    import pickle
import numpy as np

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

import tifffile as tiff



### functions ###########################################################
# load the parameters file into a global dictionary for this module
def init_mm3_helpers(param_file_path):
    # load all the parameters into a global dictionary
    global params
    with open(param_file_path, 'r') as param_file:
        params = yaml.safe_load(param_file)
    return

### functions about loading files
# loading the channel id of cell containing peaks
def load_cell_peaks(fov_id):
    '''Returns and array of the cell peaks from the spec file for a given fov number'''
    exp_dir = params['experiment_directory']
    ana_dir = params['analysis_directory']
    if not os.path.exists(exp_dir + ana_dir + 'specs/specs_%03d.pkl' % fov_id):
        warning("Spec file missing for " + fov_id)
        return -1
    else:
        with open(exp_dir + ana_dir + 'specs/specs_%03d.pkl' % fov_id, 'rb') as pkl_file:
            user_picks = pickle.load(pkl_file) # tuple = (drop_peaks, cell_peaks, bgrd_peaks)
    # get the cell-containing peaks
    cell_peaks = user_picks[1]
    cell_peaks = np.array(cell_peaks) # it is currently just a list of ints
    return cell_peaks

# load empty tif for an fov_id
def load_empty_tif(fov_id):
    exp_dir = params['experiment_directory']
    ana_dir = params['analysis_directory']
    if not os.path.exists(exp_dir + ana_dir + "empties/fov_%03d_emptymean.tif" % fov_id):
        warning("Empty mean .tif file missing for " + fov_id)
        return -1
    else:
        empty_mean = tiff.imread(exp_dir + ana_dir + "empties/fov_%03d_emptymean.tif" % fov_id)
    return empty_mean

### functions about trimming and padding images
# cuts out a channel from an tiff image (that has been processed)
def cut_slice(image_pixel_data, channel_loc):
    '''Takes an image and cuts out the channel based on the slice location
    slice location is the list with the peak information, in the form
    [peak_id, [[y1, y2],[x1, x2]]]. returns the channel slice as a numpy array.
    The numpy array will be a stack if there are multiple planes.

    if you want to slice all the channels from a picture with the channel_masks
    dictionary use a loop like this:

    for channel_loc in channel_masks[fov_id]: # fov_id is the fov of the image
        channel_slice = cut_slice[image_pixel_data, channel_loc]
        # ... do something with the slice

    NOTE: this function is for images that have gone through the
          rotation in process_tif
    '''
    channel_id = channel_loc[0] # the id is the peak location and is the first element
    channel_slice = np.zeros([image_pixel_data.shape[0],
                              channel_loc[1][0][1]-channel_loc[1][0][0],
                              channel_loc[1][1][1]-channel_loc[1][1][0]])
    #print(channel_id, channel_slice.shape)
    channel_slicer = np.s_[channel_loc[1][0][0]:channel_loc[1][0][1],
                           channel_loc[1][1][0]:channel_loc[1][1][1],:] # slice obj
    channel_slice = image_pixel_data[channel_slicer]
    if np.any([a < 1 for a in channel_slice.shape]):
        raise ValueError('channel_slice shapes must be positive (%s, %s)' % (str(channel_loc[0]), str(channel_slice.shape)))
    return channel_id, channel_slice

# remove margins of zeros from 2d numpy array
def trim_zeros_2d(array):
    # make the array equal to the sub array which has columns of all zeros removed
    # "all" looks along an axis and says if all of the valuse are such and such for each row or column
    # ~ is the inverse operator
    # using logical indexing
    array = array[~np.all(array == 0, axis = 1)]
    # transpose the array
    array = array.T
    # make the array equal to the sub array which has columns of all zeros removed
    array = array[~np.all(array == 0, axis = 1)]
    # transpose the array again
    array = array.T
    # return the array
    return array
