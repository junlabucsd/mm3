#!/usr/bin/env python3
from __future__ import print_function, division
import six

# import modules
import sys
import os
# import time
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

from skimage import measure
from tensorflow.keras import models

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
    parser = argparse.ArgumentParser(prog='python mm3_TrackFoci.py',
                                     description='Track fluorescent foci.')
    parser.add_argument('-f', '--paramfile', type=str,
                        required=True, help='Yaml file containing parameters.')
    parser.add_argument('-o', '--fov', type=str,
                        required=False, help='List of fields of view to analyze. Input "1", "1,2,3", or "1-10", etc.')
    parser.add_argument('-j', '--nproc', type=int,
                        required=False, help='Number of processors to use.')
    parser.add_argument('-c', '--channel', type=int,
                        required=False, default=2,
                        help='Which channel, as an integer, contains your fluorescent foci data? \
                            Accepts integers. Default is 2.')
    namespace = parser.parse_args()

    # Load the project parameters file
    mm3.information('Loading experiment parameters.')
    if namespace.paramfile:
        param_file_path = namespace.paramfile
    else:
        mm3.warning('No param file specified. Using 100X template.')
        param_file_path = 'yaml_templates/params_SJ110_100X.yaml'
    p = mm3.init_mm3_helpers(param_file_path) # initialized the helper library

    if namespace.fov:
        if '-' in namespace.fov:
            user_spec_fovs = range(int(namespace.fov.split("-")[0]),
                                   int(namespace.fov.split("-")[1])+1)
        else:
            user_spec_fovs = [int(val) for val in namespace.fov.split(",")]
    else:
        user_spec_fovs = []

    # number of threads for multiprocessing
    if namespace.nproc:
        p['num_analyzers'] = namespace.nproc
    mm3.information('Using {} threads for multiprocessing.'.format(p['num_analyzers']))

    if not os.path.exists(p['cell_dir']):
        sys.exit("You haven't tracked cells yet. Track cells using mm3_Track.py or mm3_Track-Standard.py, then track foci after that has completed.")
    if not os.path.exists(p['foci_seg_dir']):
        sys.exit('''You haven't segmented fluorescent foci yet.
        Segment foci using mm3_DetectFoci.py, then come back to focus tracking.''')
    if not os.path.exists(p['foci_track_dir']):
        os.makedirs(p['foci_track_dir'])

    # set segmentation image name for saving and loading segmented images
    p['seg_img'] = 'foci_seg_unet'

    # load specs file
    specs = mm3.load_specs()

    # pprint(specs) # for debugging

    # make list of FOVs to process (keys of channel_mask file)
    fov_id_list = sorted([fov_id for fov_id in specs.keys()])

    # remove fovs if the user specified so
    if user_spec_fovs:
        fov_id_list[:] = [fov for fov in fov_id_list if fov in user_spec_fovs]

    mm3.information("Processing %d FOVs." % len(fov_id_list))

    mm3.information("Creating focus tracks.")

    # NOTE: deep learning for focus tracking not yet implemented
    # read in models as dictionary
    # keys are 'migrate_model', 'child_model', 'appear_model', 'disappear_model', etc.
    # NOTE on 2019-07-15: For now, some of the models are ignored by the tracking algorithm, as they don't yet perform well
    # model_dict = mm3.get_tracking_model_dict()

    # Load time table
    mm3.load_time_table()

    # Read in cell information
    with open(os.path.join(p['cell_dir'],'all_cells.pkl'), 'rb') as cell_file:
        Cells = pickle.load(cell_file)

    ########################################################################################################################
    ########## TO DO: reorganize how tracking is done, so that it goes cell-by-cell, rather than frame-by-frame. ###########
    ########################################################################################################################
    foci = {}
    # foci_info_unet modifies foci dictionary in place, so nothing returned here
    # mm3.foci_info_unet(
    #     foci,
    #     Cells,
    #     specs,
    #     p['time_table'],
    #     channel_name="c{}".format(namespace.channel)
    # )
    
    mm3.foci_info_unet(foci,
                    Cells,
                    specs,
                    p['time_table'],
                    channel_name="c{}".format(namespace.channel))

    # update cell information with newly generated focus information
    #  again, the Cells dictionary are updated in place, so nothing returned
    mm3.update_cell_foci(Cells, foci)

    mm3.information("Finished focus tracking.")

    ### Now prune and save the data.
    mm3.information("Saving focus data.")

    ### save the cell data. Use the script mm3_OutputData for additional outputs.
    # All cell data (includes incomplete cells)
    if not os.path.isdir(p['foci_track_dir']):
        os.mkdir(p['foci_track_dir'])

    with open(os.path.join(p['foci_track_dir'], 'all_foci.pkl'), 'wb') as foci_file:
        pickle.dump(foci, foci_file, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(p['cell_dir'],'all_cells_with_foci.pkl'), 'wb') as cell_file:
        pickle.dump(Cells, cell_file, protocol=pickle.HIGHEST_PROTOCOL)

    mm3.information("Finished curating and saving focus data in {} and updated cell data in {}.".format(os.path.join(p['foci_track_dir'], 'all_foci.pkl'),
                                                                                                        os.path.join(p['cell_dir'], 'all_cells_with_foci.pkl')))
