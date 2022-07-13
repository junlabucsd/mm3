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
from io import IOBase
try:
    import cPickle as pickle
except:
    import pickle
import numpy as np
from scipy.io import savemat
import multiprocessing
from multiprocessing import Pool
from functools import partial

# user modules
# realpath() will make your script run, even if you symlink it
cmd_folder = os.path.realpath(os.path.abspath(
                          os.path.split(inspect.getfile(inspect.currentframe()))[0]))
mm3_helper_folder = os.path.join(cmd_folder, '..')
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)
if mm3_helper_folder not in sys.path:
    sys.path.insert(0, mm3_helper_folder)

# This makes python look for modules in ./external_lib
cmd_subfolder = os.path.realpath(os.path.abspath(
                             os.path.join(os.path.split(inspect.getfile(
                             inspect.currentframe()))[0], "external_lib")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import mm3_helpers as mm3
import mm3_plots


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
    parser.add_argument('-f', '--paramfile', type=argparse.FileType('r'),
                        required=True, help='Yaml file containing parameters.')
    parser.add_argument('-s', '--seg_method', type=str, required = False,help='Segmentation method (Otsu or Unet) to look for. Defaults to otsu')
    parser.add_argument('-o', '--fov', type=str,
                        required=False, help='List of fields of view to analyze. Input "1", "1,2,3", etc. ')
    parser.add_argument('-c', '--cellfile', type=argparse.FileType('r'),
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

    with open(cell_file_path, 'rb') as cell_file:
        Complete_Cells = pickle.load(cell_file)

    if namespace.seg_method:
        seg_method = 'seg_'+str(namespace.seg_method)
    else:
        mm3.warning('Defaulting to otsu segmented cells')
        seg_method = 'seg_otsu'

    # load specs file
    specs = mm3.load_specs()

    # load time table. Puts in params dictionary
    mm3.load_time_table()

    # make list of FOVs to process (keys of channel_mask file)
    fov_id_list = sorted([fov_id for fov_id in specs.keys()])

    # remove fovs if the user specified so
    if user_spec_fovs:
        fov_id_list[:] = [fov for fov in fov_id_list if fov in user_spec_fovs]

    mm3.information("Processing %d FOVs." % len(fov_id_list))

    # create dictionary which organizes cells by fov and peak_id
    Cells_by_peak = mm3_plots.organize_cells_by_channel(Complete_Cells, specs)
    
    # multiprocessing 
    color_multiproc = False
    if color_multiproc:
        Cells_to_pool = []
        for fov_id in fov_id_list:
            peak_ids = Cells_by_peak[fov_id].keys()
            peak_id_Cells = Cells_by_peak[fov_id].values()
            fov_ids = [fov_id] * len(peak_ids)

            Cells_to_pool += zip(fov_ids, peak_ids, peak_id_Cells)
        # print(Cells_to_pool[0:5])
        pool = Pool(processes=p['num_analyzers'])

        mapfunc = partial(mm3.find_cell_intensities_worker, channel=namespace.channel)
        Cells_updates = pool.starmap_async(mapfunc, Cells_to_pool)
        # [pool.apply_async(mm3.find_cell_intensities(fov_id, peak_id, Cells, midline=True, channel=namespace.channel)) for fov_id in fov_id_list for peak_id, Cells in Cells_by_peak[fov_id].items()]

        pool.close() # tells the process nothing more will be added.
        pool.join()
        update_cells = Cells_updates.get() # the result is a list of Cells dictionary, each dict contains several cells
        update_cells = {cell_id: cell for cells in update_cells for cell_id, cell in cells.items()}
        for cell_id, cell in update_cells.items():
            Complete_Cells[cell_id] = cell
    
    # for each set of cells in one fov/peak, compute the fluorescence
    else:
        for fov_id in fov_id_list:
            if fov_id in Cells_by_peak:
                mm3.information('Processing FOV {}.'.format(fov_id))
                for peak_id, Cells in Cells_by_peak[fov_id].items():
                    mm3.information('Processing peak {}.'.format(peak_id))
                    mm3.find_cell_intensities(fov_id, peak_id, Cells, seg_method=seg_method,midline=False)

    # Just the complete cells, those with mother and daugther
    cell_filename = os.path.basename(cell_file_path)
    with open(os.path.join(p['cell_dir'], cell_filename[:-4] + '_fl.pkl'), 'wb') as cell_file:
        pickle.dump(Complete_Cells, cell_file, protocol=pickle.HIGHEST_PROTOCOL)

    mm3.information('Finished.')
