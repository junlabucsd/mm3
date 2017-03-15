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
import fnmatch
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
    load_empties = False # use precomputed empties
    do_subtraction = True

    # get switches and parameters
    try:
        opts, args = getopt.getopt(sys.argv[1:],"f:o:s:")
        # switches which may be overwritten
        specify_fovs = False
        user_spec_fovs = []
        start_with_fov = -1
        param_file = ""
    except getopt.GetoptError:
        warning('No arguments detected (-f -s -o).')

    for opt, arg in opts:
        if opt == '-o':
            try:
                specify_fovs = True
                for fov_to_proc in arg.split(","):
                    user_spec_fovs.append(int(fov_to_proc))
            except:
                warning("Couldn't convert argument to an integer:",arg)
                raise ValueError
        if opt == '-s':
            try:
                start_with_fov = int(arg)
            except:
                warning("Couldn't convert argument to an integer:",arg)
                raise ValueError
        if opt == '-f':
            param_file_path = arg # parameter file path

    # Load the project parameters file
    if len(param_file_path) == 0:
        raise ValueError("a parameter file must be specified (-f <filename>).")
    information ('Loading experiment parameters.')
    with open(param_file_path, 'r') as param_file:
        p = yaml.safe_load(param_file) # load parameters into dictionary

    mm3.init_mm3_helpers(param_file_path) # initialized the helper library

    # assign shorthand directory names and create folders if they do not exist
    ana_dir = p['experiment_directory'] + p['analysis_directory']

    if p['output'] == 'TIFF':
        chnl_dir = p['experiment_directory'] + p['analysis_directory'] + 'channels/'
        empty_dir = p['experiment_directory'] + p['analysis_directory'] + 'empties/'
        sub_dir = p['experiment_directory'] + p['analysis_directory'] + 'subtracted/'
        if not os.path.exists(empty_dir):
            os.makedirs(empty_dir)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
    elif p['output'] == 'HDF5':
        hdf5_dir = p['experiment_directory'] + p['analysis_directory'] + 'hdf5/'

    # load specs file
    try:
        with open(ana_dir + '/specs.pkl', 'r') as specs_file:
            specs = pickle.load(specs_file)
    except:
        warning('Could not load specs file.')
        raise ValueError

    # make list of FOVs to process (keys of channel_mask file)
    fov_id_list = sorted([fov_id for fov_id in specs.keys()])

    # remove fovs if the user specified so
    if specify_fovs:
        fov_id_list[:] = [fov for fov in fov_id_list if fov in user_spec_fovs]
    if start_with_fov > 0:
        fov_id_list[:] = [fov for fov in fov_id_list if fov_id >= start_with_fov]

    information("Found %d FOVs to process." % len(fov_id_list))

    ### Make average empty channels ###############################################################
    if load_empties:
        information("Loading precalculated empties.")
        pass # just skip this part and go to subtraction

    else:
        information("Calculated averaged empties.")
        for fov_id in fov_id_list:
            # send to function which will create empty stack for each fov.
            averaging_result = mm3.average_empties_stack(fov_id, specs)

    ### Subtract ##################################################################################
    if do_subtraction:
        information("Subtracting channels.")
        for fov_id in fov_id_list:
            # send to function which will create empty stack for each fov.
            subtraction_result = mm3.subtract_fov_stack(fov_id, specs)

    # Else just end, they only wanted to do empty averaging.
    else:
        information("Skipping subtraction.")
        pass

    information("Finished.")
