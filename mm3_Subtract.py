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
    do_empties = True # calculate empties. Otherwise expect them to be there.
    do_subtraction = True

    # get switches and parameters
    try:
        opts, args = getopt.getopt(sys.argv[1:],"f:o:")
        # switches which may be overwritten
        specify_fovs = False
        user_spec_fovs = []
        start_with_fov = -1
        param_file = ""
    except getopt.GetoptError:
        mm3.warning('No arguments detected (-f -o).')

    for opt, arg in opts:
        if opt == '-f':
            param_file_path = arg # parameter file path
        if opt == '-o':
            try:
                specify_fovs = True
                for fov_to_proc in arg.split(","):
                    user_spec_fovs.append(int(fov_to_proc))
            except:
                mm3.warning("Couldn't convert argument to an integer:",arg)
                raise ValueError

    # Load the project parameters file
    if len(param_file_path) == 0:
        raise ValueError("A parameter file must be specified (-f <filename>).")
    mm3.information ('Loading experiment parameters.')
    p = mm3.init_mm3_helpers(param_file_path) # initialized the helper library

    # Create folders for subtracted info if they don't exist
    if p['output'] == 'TIFF':
        if not os.path.exists(p['empty_dir']):
            os.makedirs(p['empty_dir'])
        if not os.path.exists(p['sub_dir']):
            os.makedirs(p['sub_dir'])

    # load specs file
    with open(os.path.join(p['ana_dir'],'specs.pkl'), 'r') as specs_file:
        specs = pickle.load(specs_file)

    # make list of FOVs to process (keys of specs file)
    fov_id_list = sorted([fov_id for fov_id in specs.keys()])

    # remove fovs if the user specified so
    if specify_fovs:
        fov_id_list[:] = [fov for fov in fov_id_list if fov in user_spec_fovs]

    mm3.information("Found %d FOVs to process." % len(fov_id_list))

    ### Make average empty channels ###############################################################
    if not do_empties:
        mm3.information("Loading precalculated empties.")
        pass # just skip this part and go to subtraction

    else:
        mm3.information("Calculating phase averaged empties.")
        for fov_id in fov_id_list:
            # send to function which will create empty stack for each fov.
            averaging_result = mm3.average_empties_stack(fov_id, specs, color=p['phase_plane'])

    ### Subtract ##################################################################################
    if do_subtraction:
        mm3.information("Subtracting channels.")
        for fov_id in fov_id_list:
            # send to function which will create empty stack for each fov.
            subtraction_result = mm3.subtract_fov_stack(fov_id, specs, color=p['phase_plane'])
        mm3.information("Finished subtraction.")

    # Else just end, they only wanted to do empty averaging.
    else:
        mm3.information("Skipping subtraction.")
        pass
