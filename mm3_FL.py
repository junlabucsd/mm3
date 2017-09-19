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
from pprint import pprint # for human readable file output
try:
    import cPickle as pickle
except:
    import pickle
import numpy as np
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

    param_file_path = 'yaml_templates/params_SJ110_100X.yaml'

    # Load the project parameters file
    if len(param_file_path) == 0:
        raise ValueError("a parameter file must be specified (-f <filename>).")
    information ('Loading experiment parameters.')
    with open(param_file_path, 'r') as param_file:
        p = yaml.safe_load(param_file) # load parameters into dictionary

    mm3.init_mm3_helpers(param_file_path) # initialized the helper library

    # assign shorthand directory names
    ana_dir = p['experiment_directory'] + p['analysis_directory']
    seg_dir = p['experiment_directory'] + p['analysis_directory'] + 'segmented/'
    cell_dir = p['experiment_directory'] + p['analysis_directory'] + 'cell_data/'

    information("load cell lineage data.")
    with open(cell_dir + '/complete_cells.pkl', 'r') as cell_file:
        Complete_Cells = pickle.load(cell_file)
    information("Finished loading cell lineage data.")

    ### foci analysis
    information("Foci analysis.")

    foci_dir = p['experiment_directory'] + p['analysis_directory'] + 'overlay/'
    if not os.path.exists(foci_dir):
        os.makedirs(foci_dir)

    Complete_Cells_foci = mm3.foci_analysis(Complete_Cells, p)

    with open(cell_dir + '/complete_cells_foci.pkl', 'wb') as cell_file:
        pickle.dump(Complete_Cells_foci, cell_file, protocol=pickle.HIGHEST_PROTOCOL)

    with open(cell_dir + '/all_cells_foci.mat', 'wb') as cell_file:
        sio.savemat(cell_file, Complete_Cells_foci)

    information("Finished foci analysis.")
