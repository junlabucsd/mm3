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
import mm3_plots as mm3_plots

# when using this script as a function and not as a library the following will execute
if __name__ == "__main__":
    '''
    This script adds foci location information onto an existing dictionary of cells.
    '''
    # switches which may be overwritten
    param_file_path = 'yaml_templates/params_SJ110_100X.yaml'
    cell_filename = 'complete_cells.pkl'
    cell_file_path = None

    # get switches and parameters
    try:
        unixoptions="f:c:"
        gnuoptions=["paramfile=","cellfile="]
        opts, args = getopt.getopt(sys.argv[1:],unixoptions,gnuoptions)
    except getopt.GetoptError:
        mm3.warning('No arguments detected (-f -c), using hardcoded parameters.')

    for opt, arg in opts:
        if opt in ['-f',"--paramfile"]:
            param_file_path = arg # parameter file path
        if opt in ['-c',"--cellfile="]:
            cell_file_path = arg
            cell_filename = os.path.basename(cell_file_path)

    # Load the project parameters file & initialized the helper library
    p = mm3.init_mm3_helpers(param_file_path)

    # load specs file
    with open(os.path.join(p['ana_dir'],'specs.pkl'), 'r') as specs_file:
        specs = pickle.load(specs_file)

    # make list of FOVs to process (keys of channel_mask file)
    fov_id_list = sorted([fov_id for fov_id in specs.keys()])

    mm3.information("Loading cell dictionary.")
    if cell_file_path == None:
        cell_file_path = os.path.join(p['cell_dir'], cell_filename)
    with open(cell_file_path, 'r') as cell_file:
        Cells = pickle.load(cell_file)
    mm3.information("Finished loading cell dictionary.")

    ### foci analysis
    mm3.information("Starting foci analysis.")

    # create dictionary which organizes cells by fov and peak_id
    Cells_by_peak = mm3_plots.organize_cells_by_channel(Cells, specs)

    # for each set of cells in one fov/peak, find the foci
    for fov_id in fov_id_list:
        for peak_id, Cells_of_peak in Cells_by_peak[fov_id].items():
            # test
            print ('Peak no',peak_id)
            print ('Cells_of_peak')
            print (Cells_of_peak)
            if (len(Cells_of_peak) == 0):
                continue

            mm3.foci_analysis(fov_id, peak_id, Cells_of_peak)

            # test
            sys.exit()

    # Output data to both dictionary and the .mat format used by the GUI
    with open(os.path.join(p['cell_dir'], cell_filename[:-4] + '_foci.pkl'), 'wb') as cell_file:
        pickle.dump(Cells, cell_file, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(p['cell_dir'], cell_filename[:-4] + '_foci.mat'), 'wb') as cell_file:
        sio.savemat(cell_file, Cells)

    mm3.information("Finished foci analysis.")
