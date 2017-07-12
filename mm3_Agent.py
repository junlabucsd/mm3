#!/usr/bin/python
from __future__ import print_function

# import modules
import sys
import os
import time
import inspect
import getopt
import yaml
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
import signal

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

# this returns the files yet to be processed
def find_unknown_files(processed_files):
    '''
    Given a list of processed files, return a sorted list of the files that
    are in the TIFF directory but have not been processed.
    '''

    # get all the TIFFs in the folder
    found_files = glob.glob(p['TIFF_dir'] + '*.tif') # get all tiffs
    found_files = [filepath.split('/')[-1] for filepath in found_files] # remove pre-path
    found_files = set(found_files) # make a set so we can do comparisons

    unknown_files = sorted(found_files.difference(processed_files))

    return unknown_files


# define function for exting the loop
def signal_handler(signal, frame):
    global interrupted
    interrupted = True

# Function for processing the images in one FOV
def process_FOV_images(fov_id, filenames, channel_masks, specs):
    '''
    Process images from one FOV, from opening to segmentation.
    '''

    # make an array of images and then concatenate them into one big stack
    image_fov_stack = []

    # make arrays for filenames and times
    image_filenames = []
    image_times = [] # times is still an integer but may be indexed arbitrarily
    image_jds = [] # jds = julian dates (times)

    # go through images and get raw and metadata.
    mm3.information('Loading images and collecting metadata for FOV %d' % fov_id)
    for filename in filenames:
        # load image
        with tiff.TiffFile(p['TIFF_dir'] + filename) as tif:
            image_data = tif.asarray()

        # channel finding was also done on images after orientation was fixed
        image_data = mm3.fix_orientation(image_data)

        # add additional axis if the image is flat
        if len(image_data.shape) == 2:
            image_data = np.expand_dims(image_data, 0)

        #change axis so it goes X, Y, Plane
        image_data = np.rollaxis(image_data, 0, 3)

        # add it to list. The images should be in time order
        image_fov_stack.append(image_data)

        # Get the metadata (but skip finding channels)
        if p['TIFF_source'] == 'elements':
            image_metadata = mm3.get_tif_metadata_elements(tif)
        elif p['TIFF_source'] == 'nd2ToTIFF':
            image_metadata = mm3.get_tif_metadata_nd2ToTIFF(tif)

        # add information to metadata arrays
        image_filenames.append(filename)
        image_times.append(image_metadata['t'])
        image_jds.append(image_metadata['jd'])

    # concatenate the list into one big stack.
    image_fov_stack = np.stack(image_fov_stack, axis=0)

    # slice out different channels
    mm3.information('Slicing channels for FOV %d' % fov_id)
    channel_stacks = {} # dictionary with keys as peak_id, values as image stacks
    for peak_id, channel_loc in channel_masks[fov_id].iteritems():
        # slice out channel and put in dictionary
        channel_stacks[peak_id] = mm3.cut_slice(image_fov_stack, channel_loc)

    # delete the image_fov_stack here to free up memory.
    del image_fov_stack

    # go through specs file and find empty and analysis channels.
    empty_peak_ids = []
    ana_peak_ids = []
    for peak_id, spec in specs[fov_id].items():
        if spec == 0: # 0 means it should be used for empty
            empty_peak_ids.append(peak_id)
        if spec == 1: # 0 means it should be used for empty, -1 is ignore
            ana_peak_ids.append(peak_id)
    empty_peak_ids = sorted(empty_peak_ids) # sort for repeatability
    ana_peak_ids = sorted(ana_peak_ids) # sort for repeatability

    # average the empties into
    mm3.information('Averaging empties for FOV %d' % fov_id)
    # make list of just the stacks that will be used for averaging empties
    empty_stacks = []
    for peak_id in empty_peak_ids:
        image_data = channel_stacks[peak_id]
        # just get phase data and put it in list
        if len(image_data.shape) > 3:
            image_data = image_data[:,:,:,0]

        empty_stacks.append(image_data)

    if len(empty_peak_ids) == 1:
        avg_empty_stack = empty_stacks[0]

    else:
        # go through time points and create list of averaged empties
        avg_empty_stack = [] # list will be later concatentated into numpy array
        time_points = range(empty_stacks[0].shape[0]) # index is time
        for t in time_points:
            # get images from one timepoint at a time and send to alignment and averaging
            imgs = [stack[t] for stack in empty_stacks]
            avg_empty = mm3.average_empties(imgs) # function is in mm3
            avg_empty_stack.append(avg_empty)

        # concatenate list and then save out to tiff stack
        avg_empty_stack = np.stack(avg_empty_stack, axis=0)

        del empty_stacks # free up memory

    # subtract and segment
    mm3.information('Subtracting and segmenting for FOV %d' % fov_id)
    subtracted_stacks = {} # dict which will hold the subtracted stacks, similar to channel_stacks
    segmented_stacks = {} # same for segmented images
    for peak_id in ana_peak_ids:
        image_data = channel_stacks[peak_id]

        # get just the phase data
        if len(image_data.shape) > 3:
            image_data = image_data[:,:,:,0] # just get phase data and put it in list

        subtract_pairs = zip(image_data, avg_empty_stack)
        subtracted_imgs = []

        for pair in subtract_pairs:
            subtracted_imgs.append(mm3.subtract_phase(pair))

        # put the concatenated list into the dictionary
        subtracted_stacks[peak_id] = np.stack(subtracted_imgs, axis=0)

        # can segmented now
        segmented_imgs = []
        for sub_image in subtracted_imgs:
            segmented_imgs.append(mm3.segment_image(sub_image))

        # stack them up along a time axis
        segmented_imgs = np.stack(segmented_imgs, axis=0)
        segmented_imgs = segmented_imgs.astype('uint16')

        segmented_stacks[peak_id] = segmented_imgs


    # Save everything to HDF5
    mm3.information('Saving to HDF5 for FOV %d' % fov_id)
    h5f = h5py.File(p['hdf5_dir'] + 'xy%03d.hdf5' % fov_id, 'r+')

    # new length, current length plus how many images we are adding now
    old_length = h5f[u'filenames'].shape[0]
    new_length = h5f[u'filenames'].shape[0] + len(filenames)

    # save the items general to the FOV
    h5ds = h5f[u'filenames']
    h5ds.resize(new_length, axis=0)
    h5ds[old_length:new_length] = np.expand_dims(image_filenames, 1)

    h5ds = h5f[u'times']
    h5ds.resize(new_length, axis=0)
    h5ds[old_length:new_length] = np.expand_dims(image_times, 1)

    h5ds = h5f[u'times_jd']
    h5ds.resize(new_length, axis=0)
    h5ds[old_length:new_length] = np.expand_dims(image_jds, 1)

    h5ds = h5f[u'empty_channel']
    h5ds.resize(new_length, axis=0)
    h5ds[old_length:new_length] = avg_empty_stack

    # save the information for each channel
    for peak_id, spec in specs[fov_id].items():
        # put get the channel group
        h5g = h5f[u'channel_%04d' % peak_id]

        # put in raw channel information
        for color_index in range(channel_stacks[peak_id].shape[3]):
            h5ds = h5g[u'p%04d_c%1d' % (peak_id, color_index+1)]
            h5ds.resize(new_length, axis=0)
            h5ds[old_length:new_length] = channel_stacks[peak_id][:,:,:,color_index]

        # Put in subtracted and segmented images if this is an analysis peak
        if peak_id in ana_peak_ids:
            h5ds = h5g[u'p%04d_sub' % (peak_id)]
            h5ds.resize(new_length, axis=0)
            h5ds[old_length:new_length] = subtracted_stacks[peak_id]

            h5ds = h5g[u'p%04d_seg' % (peak_id)]
            h5ds.resize(new_length, axis=0)
            h5ds[old_length:new_length] = segmented_stacks[peak_id]

    # We're done here
    h5f.close()

    # return the filenames if the process was successful
    return filenames

# __main__ executes when running the script from the shell
if __name__ == "__main__":

    # get switches and parameters
    try:
        opts, args = getopt.getopt(sys.argv[1:],"f:")
        param_file_path = ''
    except getopt.GetoptError:
        print('No arguments detected (-f).')

    # set parameters
    for opt, arg in opts:
        if opt == '-f':
            param_file_path = arg # parameter file path

    # Load the project parameters file
    if len(param_file_path) == 0:
        raise ValueError("A parameter file must be specified (-f <filename>).")
    mm3.information('Loading experiment parameters.')
    p = mm3.init_mm3_helpers(param_file_path) # initialized the helper library

    # Load the channel_masks file
    with open(p['ana_dir'] + '/channel_masks.pkl', 'r') as cmask_file:
        channel_masks = pickle.load(cmask_file)

    # Load specs file
    with open(p['ana_dir'] + '/specs.pkl', 'r') as specs_file:
        specs = pickle.load(specs_file)

    # make list of FOVs to process (keys of specs file)
    fov_id_list = sorted([fov_id for fov_id in specs.keys()])

    # start looking for interupts
    signal.signal(signal.SIGINT, signal_handler)
    interrupted = False

    # first determine processed files
    processed_files = []
    # you can do this by looping through the HDF5 files and looking at the list 'filenames'
    for fov_id in fov_id_list:
        with h5py.File(p['hdf5_dir'] + 'xy%03d.hdf5' % fov_id, 'r') as h5f:
            # add all processed files to a big list
            for filename in h5f[u'filenames']:
                processed_files.append(str(filename[0]))

    # now get the files that are new
    unknown_files = find_unknown_files(processed_files)

    ### Now begin watching loop
    while True:
        # Organize images by FOV.
        images_by_fov = {}
        for fov_id in fov_id_list:
            fov_string = 'xy%02d' % fov_id # xy01
            images_by_fov[fov_id] = [filename for filename in unknown_files
                                     if fov_string in filename]

        # pool for analyzing each FOV image list
        pool = Pool(p['num_analyzers'])

        # loop over fovs and send to processing
        process_results = {}
        for fov_id, filenames in images_by_fov.items():
            # only send files to processing if there are more than x images
            if len(filenames) >= 1:
                mm3.information('Analyzing %d images for FOV %d.' % (len(filenames), fov_id))
                # send to multiprocessing
                process_results[fov_id] = pool.apply_async(process_FOV_images,
                                          args=(fov_id, filenames, channel_masks, specs))
                #process_FOV_images(fov_id, filenames, channel_masks, specs)

        pool.close()
        pool.join() # wait until analysis for every FOV is finished.

        # move the analyzed files, if the results were successful, to the analyzed list.
        for fov_id, result in process_results.iteritems():
            # if result was good, add files to known files and remove them from unknown files
            if result.successful():
                mm3.information('Processing successful for FOV %d.' % fov_id)
                for filename in result.get():
                    processed_files.append(filename)
            else:
                mm3.warning('Processing failed for FOV %d.' % fov_id)

        # Check to see if new files have been added to the TIFF folder
        unknown_files = find_unknown_files(processed_files)

        mm3.information('Found %d more files.' % len(unknown_files))

        # Check if we should stop the loop.
        if interrupted:
            mm3.information('Killing loop.')
            break

        # wait for 10 seconds before starting anew.
        time.sleep(10)
