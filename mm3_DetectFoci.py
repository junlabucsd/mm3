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
    parser = argparse.ArgumentParser(prog='python mm3_Segment.py',
                                     description='Segment cells and create lineages.')
    parser.add_argument('-f', '--paramfile', type=str,
                        required=True, help='Yaml file containing parameters.')
    parser.add_argument('-o', '--fov', type=str,
                        required=False, help='List of fields of view to analyze. Input "1", "1,2,3", or "1-10", etc.')
    parser.add_argument('-j', '--nproc', type=int,
                        required=False, help='Number of processors to use.')
    parser.add_argument('-m', '--modelfile', type=str,
                        required=False, help='Path to trained U-net model.')
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

    # create segmentation and prediction folder if they don't exist
    if not os.path.exists(p['foci_seg_dir']) and p['output'] == 'TIFF':
        os.makedirs(p['foci_seg_dir'])

    # set segmentation image name for saving and loading segmented images
    p['seg_img'] = 'foci_seg_unet'
    p['pred_img'] = 'foci_pred_unet'

    # load specs file
    specs = mm3.load_specs()
    # print(specs) # for debugging

    # make list of FOVs to process (keys of channel_mask file)
    fov_id_list = sorted([fov_id for fov_id in specs.keys()])

    # remove fovs if the user specified so
    if user_spec_fovs:
        fov_id_list[:] = [fov for fov in fov_id_list if fov in user_spec_fovs]

    mm3.information("Processing %d FOVs." % len(fov_id_list))

    ### Do Segmentation by FOV and then peak #######################################################
    mm3.information("Detecting foci in channel {} using U-net.".format(p['foci']['foci_plane']))

    # load model to pass to algorithm
    mm3.information("Loading model...")

    if namespace.modelfile:
        model_file_path = namespace.modelfile
    else:
        model_file_path = p['foci']['foci_model_file']
    # *** Need parameter for weights
    seg_model = models.load_model(model_file_path,
                                custom_objects={'bce_dice_loss': mm3.bce_dice_loss,
                                                'dice_loss': mm3.dice_loss,
                                                'precision_m': mm3.precision_m,
                                                'recall_m': mm3.recall_m,
                                                'f_precision_m': mm3.f_precision_m})
    mm3.information("Model loaded.")

    for fov_id in fov_id_list:
        mm3.segment_fov_foci_unet(fov_id, specs, seg_model, color=p['foci']['foci_plane']) 

    del seg_model

    mm3.information("Finished segmentation.")

    # ### Create cell lineages from segmented images
    # if p['segment']['do_lineages']:
    #     mm3.information("Creating cell lineages.")
    #     mm3.information("Reading track model in {}.".format(p['track_model']))

    #     track_model = models.load_model(p['track_model'])

    #     # Load time table, which goes into params
    #     mm3.load_time_table()

    #     # This dictionary holds information for all cells
    #     # Cells = {}

    #     # do lineage creation per fov, so pooling can be done by peak
    #     for i,fov_id in enumerate(fov_id_list):
    #         # update will add the output from make_lineages_function, which is a
    #         # dict of Cell entries, into Cells
    #         ana_peak_ids = [peak_id for peak_id in specs[fov_id].keys() if peak_id == 1]
    #         for j,peak_id in enumerate(ana_peak_ids):
    #             if i == 0 and j == 0:
    #                 tracks = mm3.deep_lineage_chnl_stack(fov_id, peak_id, track_model)
    #                 tracks = tracks['fov_id'] = fov_id
    #                 tracks = tracks['peak_id'] = peak_id
    #             else:
    #                 tmp_df = mm3.deep_lineage_chnl_stack(fov_id, peak_id, track_model)
    #                 tmp_df['fov_id'] = fov_id
    #                 tmp_df['peak_id'] = peak_id
    #                 tracks.append(tmp_df, ignore_index=True)

    #     mm3.information("Finished lineage creation.")

    #     ### Now prune and save the data.
    #     mm3.information("Curating and saving cell data.")

    #     with open(p['cell_dir'] + '/tracks.pkl', 'wb') as track_file:
    #         track.to_pickle(track_file)

    #     # # this returns only cells with a parent and daughters
    #     # Complete_Cells = mm3.find_complete_cells(Cells)

    #     ### save the cell data. Use the script mm3_OutputData for additional outputs.
    #     # All cell data (includes incomplete cells)
    #     # with open(p['cell_dir'] + '/all_cells.pkl', 'wb') as cell_file:
    #     #     pickle.dump(Cells, cell_file, protocol=pickle.HIGHEST_PROTOCOL)

    #     # # Just the complete cells, those with mother and daugther
    #     # # This is a dictionary of cell objects.
    #     # with open(os.path.join(p['cell_dir'], 'complete_cells.pkl'), 'wb') as cell_file:
    #     #     pickle.dump(Complete_Cells, cell_file, protocol=pickle.HIGHEST_PROTOCOL)

    #     mm3.information("Finished curating and saving cell data.")
