#!/usr/bin/python
from __future__ import print_function

# import modules
import sys
import os
import time
import inspect
import getopt
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

    # switches which may be overwritten
    param_file_path = 'yaml_templates/params_SJ110_100X.yaml'
    user_spec_fovs = []
    sub_plane = 'c1'

    # get switches and parameters
    try:
        unixoptions="f:o:c:"
        gnuoptions=["paramfile=","fov=","phase-plane="]
        opts, args = getopt.getopt(sys.argv[1:],unixoptions,gnuoptions)
    except getopt.GetoptError:
        mm3.warning('No arguments detected (-f -o -c), using hardcoded parameters.')

    for opt, arg in opts:
        if opt in ['-f',"--paramfile"]:
            param_file_path = arg # parameter file path
        if opt in ['-o',"--fov"]:
            try:
                for fov_to_proc in arg.split(","):
                    user_spec_fovs.append(int(fov_to_proc))
            except:
                mm3.warning("Couldn't convert -o argument to an integer:",arg)
                raise ValueError
        if opt in ['-c',"--phase-plane"]:
            sub_plane = arg # this should be a postfix c1, c2, c3, etc.

    # Load the project parameters file
    mm3.information ('Loading experiment parameters.')
    p = mm3.init_mm3_helpers(param_file_path) # initialized the helper library

    # Create folders for subtracted info if they don't exist
    if p['output'] == 'TIFF':
        if not os.path.exists(p['empty_dir']):
            os.makedirs(p['empty_dir'])
        if not os.path.exists(p['sub_dir']):
            os.makedirs(p['sub_dir'])

    # load specs file
    try:
        with open(os.path.join(p['ana_dir'],'specs.pkl'), 'r') as specs_file:
            specs = pickle.load(specs_file)
    except:
        mm3.warning('Could not load specs file.')
        raise ValueError

    # make list of FOVs to process (keys of specs file)
    fov_id_list = sorted([fov_id for fov_id in specs.keys()])

    # remove fovs if the user specified so
    if user_spec_fovs:
        fov_id_list[:] = [fov for fov in fov_id_list if fov in user_spec_fovs]

    mm3.information("Found %d FOVs to process." % len(fov_id_list))

    # determine if we are doing fluorescence or phase subtraction, and set flags
    if sub_plane == p['phase_plane']:
        align = True # used when averaging empties
        sub_method = 'phase' # used in subtract_fov_stack
    else:
        align = False
        sub_method = 'fluor'

    ### Make average empty channels ###############################################################
    if not do_empties:
        mm3.information("Loading precalculated empties.")
        pass # just skip this part and go to subtraction

    else:
        mm3.information("Calculating averaged empties for channel {}.".format(sub_plane))

        need_empty = [] # list holds fov_ids of fov's that did not have empties
        for fov_id in fov_id_list:
            # send to function which will create empty stack for each fov.
            averaging_result = mm3.average_empties_stack(fov_id, specs,
                                                         color=sub_plane, align=align)
            # add to list for FOVs that need to be given empties from other FOvs
            if not averaging_result:
                need_empty.append(fov_id)

        # deal with those problem FOVs without empties
        have_empty = list(set(fov_id_list).difference(set(need_empty))) # fovs with empties
        for fov_id in need_empty:
            from_fov = min(have_empty, key=lambda x: abs(x-fov_id)) # find closest FOV with an empty
            copy_result = mm3.copy_empty_stack(from_fov, fov_id, color=sub_plane)

    ### Subtract ##################################################################################
    if do_subtraction:
        mm3.information("Subtracting channels for channel {}.".format(sub_plane))
        for fov_id in fov_id_list:
            # send to function which will create empty stack for each fov.
            subtraction_result = mm3.subtract_fov_stack(fov_id, specs,
                                                        color=sub_plane, method=sub_method)
        mm3.information("Finished subtraction.")

    # Else just end, they only wanted to do empty averaging.
    else:
        mm3.information("Skipping subtraction.")
        pass
