#!/usr/bin/python
from __future__ import print_function

# import modules
import sys
import os
import inspect
import argparse
import yaml
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

    # set switches and parameters
    parser = argparse.ArgumentParser(prog='python mm3_Foci.py',
                                     description='Finds foci.')
    parser.add_argument('-f', '--paramfile', type=file,
                        required=True, help='Yaml file containing parameters.')
    parser.add_argument('-c', '--cellfile', type=file,
                        required=False, help='Path to Cell object dicionary to analyze. Defaults to complete_cells.pkl.')
    namespace = parser.parse_args()

    # Load the project parameters file
    mm3.information('Loading experiment parameters.')
    if namespace.paramfile.name:
        param_file_path = namespace.paramfile.name
    else:
        mm3.warning('No param file specified. Using 100X template.')
        param_file_path = 'yaml_templates/params_SJ110_100X.yaml'
    p = mm3.init_mm3_helpers(param_file_path) # initialized the helper library

    # load cell file
    mm3.information('Loading cell data.')
    if namespace.cellfile:
        cell_file_path = namespace.cellfile.name
    else:
        mm3.warning('No cell file specified. Using complete_cells.pkl.')
        cell_file_path = os.path.join(p['cell_dir'], 'complete_cells.pkl')

    with open(cell_file_path, 'r') as cell_file:
        Cells = pickle.load(cell_file)

    # load specs file
    try:
        with open(os.path.join(p['ana_dir'],'specs.yaml'), 'r') as specs_file:
            specs = yaml.safe_load(specs_file)
    except:
        mm3.warning('Could not load specs file.')
        raise ValueError

    # make list of FOVs to process (keys of channel_mask file)
    fov_id_list = sorted([fov_id for fov_id in specs.keys()])

    ### foci analysis
    mm3.information("Starting foci analysis.")

    # create dictionary which organizes cells by fov and peak_id
    Cells_by_peak = mm3_plots.organize_cells_by_channel(Cells, specs)

    # for each set of cells in one fov/peak, find the foci
    for fov_id in fov_id_list:
        if not fov_id in Cells_by_peak:
            continue

        for peak_id, Cells_of_peak in Cells_by_peak[fov_id].items():
            # test
            # print ('Peak no',peak_id)
            # print ('Cells_of_peak')
            # print (Cells_of_peak)
            if (len(Cells_of_peak) == 0):
                continue

            mm3.foci_analysis(fov_id, peak_id, Cells_of_peak)

            # test
            # sys.exit()

    # Output data to both dictionary and the .mat format used by the GUI
    cell_filename = os.path.basename(cell_file_path)
    with open(os.path.join(p['cell_dir'], cell_filename[:-4] + '_foci.pkl'), 'wb') as cell_file:
        pickle.dump(Cells, cell_file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(p['cell_dir'], cell_filename[:-4] + '_foci.mat'), 'wb') as cell_file:
        sio.savemat(cell_file, Cells)

    mm3.information("Finished foci analysis.")
