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

if sys.platform == "linux" or sys.platform == "linux2":
    # linux
    import gevent_inotifyx as inotify
    from gevent.queue import Queue as gQueue
    use_inotify = True
elif sys.platform == "darwin":
    # OS X
    from watchdog.observers import Observer
    from watchdog.events import PatternMatchingEventHandler
    use_watchdog = True

# class which responds to file addition events and saves their file names
class tif_sentinal(PatternMatchingEventHandler):
    '''tif_sentinal will react to file additions that have .tif at the end.
    It only looks at file creations, not modifications or deletions.
    It inherits from the pattern match event handler in watchdog.
    There are two class attributes, events_filenames and events_buffer.
    events_filenames is the name of all added .tif files. Use the
    self.get_filenames method to return all events.
    events_buffer are just the filenames since last time the buffer was returned
    use self.get_buffer to return the buffer (which also clears the buffer).
    '''
    patterns = ["*.tif"]

    def __init__(self):
        # Makes sure to do the init function from the parent (super) class first
        super(tif_sentinal, self).__init__()
        self.events_filenames = [] # keeps all event filenames
        self.events_buffer = [] # keeps just event filenames since last call to buf

    def process(self, event):
        """
        called when an event (file change) occurs

        event.event_type
            'modified' | 'created' | 'moved' | 'deleted'
        event.is_directory
            True | False
        event.src_pquitath
            path/to/observed/file
        """
        filename = event.src_path.split('/')[-1] # get just the file name
        # append the file name to the class attributes for later retrieving
        self.events_filenames.append(filename)
        self.events_buffer.append(filename)

    def get_filenames(self):
        '''returns all event file names'''
        return self.events_filenames

    def get_buffer(self):
        '''returns all events since last call to print buffer'''
        # make a temporary buffer for printing
        buf = [x for x in self.events_buffer]
        self.events_buffer = [] # reset buffer
        return buf

    # could use this to do something on file modifications
    # def on_modified(self, event):
    #     self.process(event)

    def on_created(self, event):
        self.process(event)


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
    for filename in filenames:
        # load image
        with tiff.TiffFile(params['TIFF_dir'] + filename) as tif:
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
        if params['TIFF_source'] == 'elements':
            image_metadata = mm3.get_tif_metadata_elements(tif)
        elif params['TIFF_source'] == 'nd2ToTIFF':
            image_metadata = mm3.get_tif_metadata_nd2ToTIFF(tif)

        # add information to metadata arrays
        image_filenames.append(image_name)
        image_times.append(image_metadata['t'])
        image_jds.append(image_metadata['jd'])

    # concatenate the list into one big stack.
    image_fov_stack = np.stack(image_fov_stack, axis=0)

    # slice out different channels
    channel_stacks = {} # dictionary with keys as peak_id, values as image stacks
    for peak_id, channel_loc in channel_masks[fov_id].iteritems():
        # slice out channel and put in dictionary
        channel_stacks[peak_id] = cut_slice(image_fov_stack, channel_loc)

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



    # open up the HDF5 for this FOV, keep it open till the end.
    hf5 = h5py.File(p['hdf5_dir'] + 'xy%03d.hdf5' % fov_id, 'r+')

    # # put segmented channel in correct group
    # h5g = h5f['channel_%04d' % peak_id]

    # Slice out the channels from the TIFFs
    # must have been sent the channel_masks file.
    # could also use the specs file to only analyze channels that have cells in them...

    h5f['channel_%04d/p%04d_%s' % (peak_id, peak_id, color)]

    # you have to resize the hdf5 dataset before adding data
    h5ds.resize(h5si.shape[0] + 1, axis = 0) # add a space fow new images
    # h5ds.flush()
    h5ds[-1] = subtracted_image
    # h5ds.flush()


    # Send them emtpy channels for averaging

    # Subtract the averaged empty from the others

    # Segment them.

    # We're done here
    hf5.close()

    return

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

    ### Pre watching loop

    # intialize known files list and files to be analyzed list
    # get all the TIFFs in the folder
    found_files = glob.glob(p['TIFF_dir'] + '*.tif') # get all tiffs
    found_files = [filepath.split('/')[-1] for filepath in found_files] # remove pre-path
    found_files = set(found_files) # make a set so we can do comparisons

    # first determine known files
    processed_files = []
    # you can do this by looping through the HDF5 files and looking at the list 'filenames'
    for fov_id in fov_id_list:
        with h5py.File(p['hdf5_dir'] + 'xy%03d.hdf5' % fov_id, 'r') as h5f:
            # add all processed files to a big list
            processed_files += [str(filename) for filename in h5f[u'filenames']]

    # Filter for known and unknown files. found_files should be > processed_files
    # known files are both in found_files and processed_files. Sort by time
    known_files = sorted(found_files.intersection(processed_files))

    # unknown files
    unknown_files = sorted(found_files.difference(processed_files))

    del processed_files
    del found_files # clear up some memories

    ### Now begin loop
    # Organize images by FOV.
    images_by_fov = {}
    for fov_id in fov_id_list:
        fov_string = 'xy%02d' % fov_id # xy01
        images_by_fov[fov_id] = [filename for filename in unknown_files
                                 if fov_string in filename]

    # # pool for analyzing each FOV image list
    # pool = Pool(p['num_analyzers'])

    # loop over images and get information
    for fov_id, filenames in images_by_fov.items():




    mm3.information('Analyzing images per FOV.')

    # pool.close()
    # pool.join() # wait until analysis for every FOV is finished.




    # All of that should have been done in a pool for all FOVs, wait for that pool to finish.

    # move the analyzed files, if the results were successful, to the analyzed list.

    # Check to see if new files have been added to the TIFF folder and put them in the known files
    # list
