#!/usr/bin/python
from __future__ import print_function
def warning(*objs):
    print(time.strftime("%H:%M:%S Warning:", time.localtime()), *objs, file=sys.stderr)
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
from pprint import pprint # for human readable file output
try:
    import cPickle as pickle
except:
    import pickle
import multiprocessing
from multiprocessing import Pool #, Lock
import numpy as np
import warnings
import h5py

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

# when using this script as a function and not as a library the following will execute
if __name__ == "__main__":
    # hardcoded parameters
    load_metadata = False
    load_channel_masks = False

    # get switches and parameters
    try:
        opts, args = getopt.getopt(sys.argv[1:],"f:o:")
        # switches which may be overwritten
        specify_fovs = False
        user_spec_fovs = []
        param_file_path = ''
    except getopt.GetoptError:
        print('No arguments detected (-f -o).')

    # set parameters
    for opt, arg in opts:
        if opt == '-f':
            param_file_path = arg # parameter file path
        if opt == '-o':
            try:
                specify_fovs = True
                for fov_to_proc in arg.split(","):
                    user_spec_fovs.append(int(fov_to_proc))
            except:
                warning("Couldn't convert argument to an integer:",arg)
                raise ValueError

    # Load the project parameters file
    # if the paramfile string has no length ie it has not been specified, ERROR
    if len(param_file_path) == 0:
        raise ValueError("a parameter file must be specified (-f <filename>).")
    information ('Loading experiment parameters.')
    p = mm3.init_mm3_helpers(param_file_path) # initialized the helper library

    # set up how to manage cores for multiprocessing
    cpu_count = multiprocessing.cpu_count()
    num_analyzers = cpu_count*2 - 2

    # assign shorthand directory names
    TIFF_dir = p['experiment_directory'] + p['image_directory'] # source of images
    ana_dir = p['experiment_directory'] + p['analysis_directory']
    chnl_dir = p['experiment_directory'] + p['analysis_directory'] + 'channels/'
    hdf5_dir = p['experiment_directory'] + p['analysis_directory'] + 'hdf5/'

    # create the subfolders if they don't
    if not os.path.exists(ana_dir):
        os.makedirs(ana_dir)

    if p['output'] == 'TIFF':
        if not os.path.exists(chnl_dir):
            os.makedirs(chnl_dir)
    elif p['output'] == 'HDF5':
        if not os.path.exists(hdf5_dir):
            os.makedirs(hdf5_dir)

    # declare information variables
    analyzed_imgs = {} # for storing get_params pool results.

    ### process TIFFs for metadata #################################################################
    if load_metadata:
        information("Loading image parameters dictionary.")

        with open(ana_dir + '/TIFF_metadata.pkl', 'r') as tiff_metadata:
            analyzed_imgs = pickle.load(tiff_metadata)

    else:
        information("Finding image parameters.")

        # get all the TIFFs in the folder
        found_files = glob.glob(TIFF_dir + '*.tif') # get all tiffs
        found_files = [filepath.split('/')[-1] for filepath in found_files] # remove pre-path
        found_files = sorted(found_files) # should sort by timepoint

        # if user has specified only certain FOVs, filter for those
        if specify_fovs:
            information('Filetering TIFFs by FOV.')
            fitered_files = []
            for fov_id in user_spec_fovs:
                fov_string = 'xy%02d' % fov_id # xy01
                fitered_files += [ifile for ifile in found_files if fov_string in ifile]

            found_files = fitered_files[:]
            del fitered_files

        # get information for all these starting tiffs
        if len(found_files) > 0:
            information("Found %d image files." % len(found_files))
        else:
            warning('No TIFF files found')

        # initialize pool for analyzing image metadata
        pool = Pool(num_analyzers)

        # loop over images and get information
        for fn in found_files:
            # get_params gets the image metadata and puts it in analyzed_imgs dictionary
            # for each file name. True means look for channels

            # This is the non-parallelized version (useful for debug)
            # analyzed_imgs[fn] = mm3.get_tif_params(fn, True)

            # Parallelized
            analyzed_imgs[fn] = pool.apply_async(mm3.get_tif_params, args=(fn, True))

        information('Waiting for image analysis pool to be finished.')

        pool.close() # tells the process nothing more will be added.
        pool.join() # blocks script until everything has been processed and workers exit

        information('Image analyses pool finished, getting results.')

        # get results from the pool and put them in a dictionary
        for fn, result in analyzed_imgs.iteritems():
            if result.successful():
                analyzed_imgs[fn] = result.get() # put the metadata in the dict if it's good
            else:
                analyzed_imgs[fn] = False # put a false there if it's bad

        information('Got results from analyzed images.')

        # save metadata to a .pkl and a human readable txt file
        information('Saving metadata from analyzed images...')
        with open(ana_dir + '/TIFF_metadata.pkl', 'wb') as tiff_metadata:
            pickle.dump(analyzed_imgs, tiff_metadata)
        with open(ana_dir + '/TIFF_metadata.txt', 'w') as tiff_metadata:
            pprint(analyzed_imgs, stream=tiff_metadata)
        information('Saved metadata from analyzed images.')

    ### Make consensus channel masks and get other shared metadata #################################
    if load_channel_masks:
        information("Loading channel masks dictionary.")

        with open(ana_dir + '/channel_masks.pkl', 'r') as cmask_file:
            channel_masks = pickle.load(cmask_file)

    else:
        information("Calculating channel masks.")

        # Uses channel information from the already processed image data
        channel_masks = mm3.make_masks(analyzed_imgs)

        #save the channel mask dictionary to a pickle and a text file
        with open(ana_dir + '/channel_masks.pkl', 'wb') as cmask_file:
            pickle.dump(channel_masks, cmask_file)
        with open(ana_dir + '/channel_masks.txt', 'w') as cmask_file:
            pprint(channel_masks, stream=cmask_file)

        information("Channel masks saved.")

    ### Slice and write TIFF files into channels ###################################################
    information("Saving channel slices.")

    # do it by FOV. Not set up for multiprocessing
    for fov, peaks in channel_masks.iteritems():
        information("Loading images for FOV %03d." % fov)

        # get filenames just for this fov along with the julian date of acquistion
        send_to_write = [[k, v['jd']] for k, v in analyzed_imgs.items() if v['fov'] == fov]

        # sort the filenames by jdn
        send_to_write = sorted(send_to_write, key=lambda time: time[1])

        if p['output'] == 'TIFF':
            #This is for loading the whole raw tiff stack and then slicing through it
            mm3.tiff_stack_slice_and_write(send_to_write, channel_masks, analyzed_imgs)

        elif p['output'] == 'HDF5':
            # Or write it to hdf5
            mm3.hdf5_stack_slice_and_write(send_to_write, channel_masks, analyzed_imgs)

    information("Channel slices saved.")
