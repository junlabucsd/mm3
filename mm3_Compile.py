#!/usr/bin/python
from __future__ import print_function

# import modules
import sys
import os
import time
import inspect
import argparse
import yaml
import glob
import re
from pprint import pprint # for human readable file output
try:
    import cPickle as pickle
except:
    import pickle
import multiprocessing
from multiprocessing import Pool
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

# supress the mm3.warning this always gives
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import tifffile as tiff

# this is the mm3 module with all the useful functions and classes
import mm3_helpers as mm3

# when using this script as a function and not as a library the following will execute
if __name__ == "__main__":
    '''mm3_Compile.py locates and slices out mother machine channels into image stacks.
    '''

    # set switches and parameters
    parser = argparse.ArgumentParser(prog='python mm3_Compile.py',
                                     description='Identifies and slices out channels into individual TIFF stacks through time.')
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

    # number of threads for multiprocessing
    if namespace.nproc:
        p['num_analyzers'] = namespace.nproc

    # only analyze images up until this t point. Put in None otherwise
    t_end = p['compile']['t_end']

    # create the subfolders if they don't
    if not os.path.exists(p['ana_dir']):
        os.makedirs(p['ana_dir'])
    if p['output'] == 'TIFF':
        if not os.path.exists(p['chnl_dir']):
            os.makedirs(p['chnl_dir'])
    elif p['output'] == 'HDF5':
        if not os.path.exists(p['hdf5_dir']):
            os.makedirs(p['hdf5_dir'])

    # declare information variables
    analyzed_imgs = {} # for storing get_params pool results.

    ### process TIFFs for metadata #################################################################
    if not p['compile']['do_metadata']:
        mm3.information("Loading image parameters dictionary.")

        with open(os.path.join(p['ana_dir'], 'TIFF_metadata.pkl'), 'r') as tiff_metadata:
            analyzed_imgs = pickle.load(tiff_metadata)

    else:
        mm3.information("Finding image parameters.")

        # get all the TIFFs in the folder
        found_files = glob.glob(os.path.join(p['TIFF_dir'],'*.tif')) # get all tiffs
        found_files = [filepath.split('/')[-1] for filepath in found_files] # remove pre-path
        found_files = sorted(found_files) # should sort by timepoint

        # remove images after this timepoint
        if t_end:
            # go through list and find first place where timepoint is equivalent to t_end
            for n, ifile in enumerate(found_files):
                string = re.compile('t%03dxy|t%04dxy' % (t_end, t_end)) # account for 3 and 4 digit
                # if re.search == True then a match was found
                if re.search(string, ifile):
                    # cut off found files
                    found_files = found_files[:n]
                    break # get out of the loop

        # if user has specified only certain FOVs, filter for those
        if (len(user_spec_fovs) > 0):
            mm3.information('Filtering TIFFs by FOV.')
            fitered_files = []
            for fov_id in user_spec_fovs:
                fov_string = 'xy%02d' % fov_id # xy01
                fitered_files += [ifile for ifile in found_files if fov_string in ifile]

            found_files = fitered_files[:]

        # get information for all these starting tiffs
        if len(found_files) > 0:
            mm3.information("Found %d image files." % len(found_files))
        else:
            mm3.warning('No TIFF files found')

        # initialize pool for analyzing image metadata
        pool = Pool(p['num_analyzers'])

        # loop over images and get information
        for fn in found_files:
            # get_params gets the image metadata and puts it in analyzed_imgs dictionary
            # for each file name. True means look for channels

            # This is the non-parallelized version (useful for debug)
            # analyzed_imgs[fn] = mm3.get_tif_params(fn, True)

            # Parallelized
            analyzed_imgs[fn] = pool.apply_async(mm3.get_tif_params, args=(fn, True))

        mm3.information('Waiting for image analysis pool to be finished.')

        pool.close() # tells the process nothing more will be added.
        pool.join() # blocks script until everything has been processed and workers exit

        mm3.information('Image analyses pool finished, getting results.')

        # get results from the pool and put them in a dictionary
        for fn, result in analyzed_imgs.iteritems():
            if result.successful():
                analyzed_imgs[fn] = result.get() # put the metadata in the dict if it's good
            else:
                analyzed_imgs[fn] = False # put a false there if it's bad

        mm3.information('Got results from analyzed images.')

        # save metadata to a .pkl and a human readable txt file
        mm3.information('Saving metadata from analyzed images...')
        with open(os.path.join(p['ana_dir'], 'TIFF_metadata.pkl'), 'wb') as tiff_metadata:
            pickle.dump(analyzed_imgs, tiff_metadata, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(p['ana_dir'], 'TIFF_metadata.txt'), 'w') as tiff_metadata:
            pprint(analyzed_imgs, stream=tiff_metadata)
        mm3.information('Saved metadata from analyzed images.')

    ### Make table for jd time to FOV and time point
    if not p['compile']['do_time_table']:
        mm3.information('Skipping time table creation.')

    else:
        # do it
        time_table = mm3.make_time_table(analyzed_imgs)

        # save to .pkl. This pkl will be loaded into the params
        mm3.information('Saving time table...')
        with open(os.path.join(p['ana_dir'], 'time_table.pkl'), 'wb') as time_table_file:
            pickle.dump(time_table, time_table_file, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(p['ana_dir'], 'time_table.txt'), 'w') as time_table_file:
            pprint(time_table, stream=time_table_file)
        mm3.information('Saved time table.')

    ### Make consensus channel masks and get other shared metadata #################################
    if not p['compile']['do_channel_masks'] and p['compile']['do_slicing']:
        mm3.information("Loading channel masks dictionary.")

        with open(os.path.join(p['ana_dir'],'channel_masks.pkl'), 'r') as cmask_file:
            channel_masks = pickle.load(cmask_file)

    elif p['compile']['do_channel_masks']:
        mm3.information("Calculating channel masks.")

        # only calculate channels masks from images before t_end in case it is specified
        if t_end:
            analyzed_imgs = {fn : i_metadata for fn, i_metadata in analyzed_imgs.items() if
                             i_metadata['t'] <= t_end}

        # Uses channel mm3.information from the already processed image data
        channel_masks = mm3.make_masks(analyzed_imgs)

        #save the channel mask dictionary to a pickle and a text file
        with open(os.path.join(p['ana_dir'],'channel_masks.pkl'), 'wb') as cmask_file:
            pickle.dump(channel_masks, cmask_file, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(p['ana_dir'],'channel_masks.txt'), 'w') as cmask_file:
            pprint(channel_masks, stream=cmask_file)

        mm3.information("Channel masks saved.")

    ### Slice and write TIFF files into channels ###################################################
    if p['compile']['do_slicing']:

        mm3.information("Saving channel slices.")

        # do it by FOV. Not set up for multiprocessing
        for fov, peaks in channel_masks.iteritems():

            # skip fov if not in the group
            if user_spec_fovs and fov not in user_spec_fovs:
                continue

            mm3.information("Loading images for FOV %03d." % fov)

            # get filenames just for this fov along with the julian date of acquistion
            send_to_write = [[k, v['t']] for k, v in analyzed_imgs.items() if v['fov'] == fov]

            # sort the filenames by jdn
            send_to_write = sorted(send_to_write, key=lambda time: time[1])

            if p['output'] == 'TIFF':
                #This is for loading the whole raw tiff stack and then slicing through it
                mm3.tiff_stack_slice_and_write(send_to_write, channel_masks, analyzed_imgs)

            elif p['output'] == 'HDF5':
                # Or write it to hdf5
                mm3.hdf5_stack_slice_and_write(send_to_write, channel_masks, analyzed_imgs)

        mm3.information("Channel slices saved.")
