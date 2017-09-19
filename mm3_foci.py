#!/usr/bin/python
from __future__ import print_function

# import modules
import sys
import os
import inspect
import getopt
try:
    import cPickle as pickle
except:
    import pickle
import scipy.io as sio

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

import mm3_helpers as mm3

# when using this script as a function and not as a library the following will execute
if __name__ == "__main__":
    '''
    This script adds foci location information onto an existing dictionary of cells.
    '''
    # switches which may be overwritten
    param_file_path = 'yaml_templates/params_SJ110_100X.yaml'

    # get switches and parameters
    try:
        opts, args = getopt.getopt(sys.argv[1:],"f:")
    except getopt.GetoptError:
        mm3.warning('No arguments detected (-f), using hardcoded parameters.')

    for opt, arg in opts:
        if opt == '-f':
            param_file_path = arg # parameter file path

    # Load the project parameters file & initialized the helper library
    p = mm3.init_mm3_helpers(param_file_path)

    mm3.information("Loading cell dictionary.")
    with open(os.path.join(p['cell_dir'], 'complete_cells.pkl'), 'r') as cell_file:
        Cells = pickle.load(cell_file)
    mm3.information("Finished loading cell dictionary.")

    ### foci analysis
    mm3.information("Starting foci analysis.")

    # Do it
    Complete_Cells_foci = mm3.foci_analysis(Cells)

    # Output data to both dictionary and the .mat format used by the GUI
    with open(os.path.join(p['cell_dir'], '/complete_cells_foci.pkl'), 'wb') as cell_file:
        pickle.dump(Complete_Cells_foci, cell_file, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(p['cell_dir'], '/complete_cells_foci.mat'), 'wb') as cell_file:
        sio.savemat(cell_file, Complete_Cells_foci)

    mm3.information("Finished foci analysis.")
