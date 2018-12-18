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

# when using this script as a function and not as a library the following will execute
if __name__ == "__main__":

    # set switches and parameters
    parser = argparse.ArgumentParser(prog='python mm3_Segment.py',
                                     description='Segment cells and create lineages.')
    parser.add_argument('-f', '--paramfile',  type=file,
                        required=True, help='Yaml file containing parameters.')
    parser.add_argument('-o', '--fov',  type=str,
                        required=False, help='List of fields of view to analyze. Input "1", "1,2,3", etc. ')
    parser.add_argument('-j', '--nproc',  type=int,
                        required=False, help='Number of processors to use.')
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

    # number of threads for multiprocessing
    if namespace.nproc:
        p['num_analyzers'] = namespace.nproc

    # create segmenteation and cell data folder if they don't exist
    if not os.path.exists(p['seg_dir']) and p['output'] == 'TIFF':
        os.makedirs(p['seg_dir'])
    if not os.path.exists(p['cell_dir']):
        os.makedirs(p['cell_dir'])

    # load specs file
    try:
        with open(os.path.join(p['ana_dir'], 'specs.yaml'), 'r') as specs_file:
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

    ### Do Segmentation by FOV and then peak #######################################################
    if p['segment']['do_segmentation']:
        mm3.information("Segmenting channels.")

        for fov_id in fov_id_list:
            # determine which peaks are to be analyzed (those which have been subtracted)
            ana_peak_ids = []
            for peak_id, spec in specs[fov_id].items():
                if spec == 1: # 0 means it should be used for empty, -1 is ignore, 1 is analyzed
                    ana_peak_ids.append(peak_id)
            ana_peak_ids = sorted(ana_peak_ids) # sort for repeatability

            for peak_id in ana_peak_ids:
                # send to segmentation
                mm3.segment_chnl_stack(fov_id, peak_id)

        mm3.information("Finished segmentation.")

    ### Create cell lineages from segmented images
    if p['segment']['do_lineages']:
        mm3.information("Creating cell lineages.")

        # Load time table, which goes into params
        mm3.load_time_table()

        # This dictionary holds information for all cells
        Cells = {}

        # do lineage creation per fov, so pooling can be done by peak
        for fov_id in fov_id_list:
            # update will add the output from make_lineages_function, which is a
            # dict of Cell entries, into Cells
            Cells.update(mm3.make_lineages_fov(fov_id, specs))

        mm3.information("Finished lineage creation.")

        ### Now prune and save the data.
        mm3.information("Curating and saving cell data.")

        # this returns only cells with a parent and daughters
        Complete_Cells = mm3.find_complete_cells(Cells)

        ### save the cell data. Use the script mm3_OutputData for additional outputs.
        # All cell data (includes incomplete cells)
        with open(p['cell_dir'] + '/all_cells.pkl', 'wb') as cell_file:
            pickle.dump(Cells, cell_file, protocol=pickle.HIGHEST_PROTOCOL)

        # Just the complete cells, those with mother and daugther
        # This is a dictionary of cell objects.
        with open(os.path.join(p['cell_dir'],'complete_cells.pkl'), 'wb') as cell_file:
            pickle.dump(Complete_Cells, cell_file, protocol=pickle.HIGHEST_PROTOCOL)

        mm3.information("Finished curating and saving cell data.")
