#!/usr/bin/python
from __future__ import print_function

# import modules
import sys
import os
import time
import inspect
import argparse
import yaml
from pprint import pprint # for human readable file output
try:
    import cPickle as pickle
except:
    import pickle
import numpy as np
from scipy.io import savemat

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
    This script is to act as a template for how to do post segmentation/lineage analysis.
    The example uses calculating total fluorescence per cell.
    '''

    # switches which may be overwritten
    param_file_path = ''
    user_spec_fovs = []

    # set switches and parameters
    parser = argparse.ArgumentParser(prog='python mm3_Colors.py',
                                     description='Calculates total and average fluorescence per cell.')
    parser.add_argument('-f', '--paramfile', type=file,
                        required=True, help='Yaml file containing parameters.')
    parser.add_argument('-o', '--fov', type=str,
                        required=False, help='List of fields of view to analyze. Input "1", "1,2,3", etc. ')
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

    if namespace.fov:
        user_spec_fovs = [int(val) for val in namespace.fov.split(",")]
    else:
        user_spec_fovs = []

    # load cell file
    mm3.information('Loading cell data.')
    if namespace.cellfile:
        cell_file_path = namespace.cellfile.name
    else:
        mm3.warning('No cell file specified. Using complete_cells.pkl.')
        cell_file_path = os.path.join(p['cell_dir'], 'complete_cells.pkl')

    with open(cell_file_path, 'r') as cell_file:
        Complete_Cells = pickle.load(cell_file)

    # load specs file
    try:
        with open(os.path.join(p['ana_dir'],'specs.yaml'), 'r') as specs_file:
            specs = yaml.safe_load(specs_file)
    except:
        mm3.warning('Could not load specs file.')
        raise ValueError

    # make list of FOVs to process (keys of channel_mask file)
    fov_id_list = sorted([fov_id for fov_id in specs.keys()])

    # remove fovs if the user specified so
    if user_spec_fovs:
        fov_id_list[:] = [fov for fov in fov_id_list if fov in user_spec_fovs]

    mm3.information("Processing %d FOVs." % len(fov_id_list))

    # create dictionary which organizes cells by fov and peak_id
    Cells_by_peak = mm3_plots.organize_cells_by_channel(Complete_Cells, specs)

    # for each set of cells in one fov/peak, compute the fluorescence
    for fov_id in fov_id_list:
        if fov_id in Cells_by_peak:
            mm3.information('Processing FOV {}.'.format(fov_id))
            for peak_id, Cells in Cells_by_peak[fov_id].items():
                mm3.information('Processing peak {}.'.format(peak_id))
                mm3.find_cell_intensities(fov_id, peak_id, Cells, midline=False)

    # Just the complete cells, those with mother and daugther
    with open(os.path.join(p['cell_dir'], 'complete_cells_fl.pkl'), 'wb') as cell_file:
        pickle.dump(Complete_Cells, cell_file, protocol=pickle.HIGHEST_PROTOCOL)

    mm3.information('Finished.')
