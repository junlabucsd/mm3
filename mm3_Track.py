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
    parser = argparse.ArgumentParser(prog='python mm3_Track.py',
                                     description='Track cells and create lineages.')
    parser.add_argument('-f', '--paramfile', type=str,
                        required=True, help='Yaml file containing parameters.')
    parser.add_argument('-o', '--fov', type=str,
                        required=False, help='List of fields of view to analyze. Input "1", "1,2,3", or "1-10", etc.')
    # parser.add_argument('-p', '--peak', type=str,
    #                     required=False, help='List of peak ids to analyze. Input "1", "1,2,3", or "1-10", etc.')
    parser.add_argument('-j', '--nproc', type=int,
                        required=False, help='Number of processors to use.')
    parser.add_argument('-m', '--modelfile', type=str,
                        required=False, help='Path to trained model.')
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

    # if namespace.peak:
    #     if '-' in namespace.peak:
    #         user_spec_peaks = range(int(namespace.fov.split("-")[0]),
    #                                int(namespace.fov.split("-")[1])+1)
    #     else:
    #         user_spec_peaks = [int(val) for val in namespace.fov.split(",")]
    # else:
    #     user_spec_peaks = []

    # number of threads for multiprocessing
    if namespace.nproc:
        p['num_analyzers'] = namespace.nproc
    mm3.information('Using {} threads for multiprocessing.'.format(p['num_analyzers']))

    if not os.path.exists(p['cell_dir']):
        os.makedirs(p['cell_dir'])

    # set segmentation image name for saving and loading segmented images
    p['seg_img'] = 'seg_unet'

    # load specs file
    specs = mm3.load_specs()
    # pprint(specs) # for debugging

    # make list of FOVs to process (keys of channel_mask file)
    fov_id_list = sorted([fov_id for fov_id in specs.keys()])
    
    # remove fovs if the user specified so
    if user_spec_fovs:
        fov_id_list[:] = [fov for fov in fov_id_list if fov in user_spec_fovs]

    mm3.information("Processing %d FOVs." % len(fov_id_list))

    mm3.information("Creating cell lineages.")
    mm3.information("Reading track models. This could take a few minutes.")

    # read in models as dictionary
    # keys are 'migrate_model', 'child_model', 'appear_model', 'die_model', 'disappear_model', etc.
    # NOTE on 2019-07-15: For now, some of the models are ignored by the tracking algorithm, as they don't yet perform well
    model_dict = mm3.get_tracking_model_dict()

    # Load time table, which goes into params
    mm3.load_time_table()

    # This dictionary holds information for all cells
    # Cells = {}

    # do lineage creation per fov, per trap
    tracks = {}
    for i,fov_id in enumerate(fov_id_list):
        # tracks[fov_id] = {}
        # update will add the output from make_lineages_function, which is a
        # dict of Cell entries, into Cells
        ana_peak_ids = [peak_id for peak_id in specs[fov_id].keys() if specs[fov_id][peak_id] == 1]
        # ana_peak_ids = [9,13,15,19,25,33,36,37,38,39] # was used for debugging
        for j,peak_id in enumerate(ana_peak_ids):

            seg_stack = mm3.load_stack(fov_id, peak_id, color=p['seg_img'])
            # run predictions for each tracking class
            # consider only the top six cells for a given trap when doing tracking
            cell_number = 6
            frame_number = seg_stack.shape[0]
            # get region properties
            regions_by_time = [measure.regionprops(label_image=img) for img in seg_stack]

            # have generator yield info for top six cells in all frames
            prediction_generator = mm3.PredictTrackDataGenerator(regions_by_time, batch_size=frame_number, dim=(cell_number,5,9))
            cell_info = prediction_generator.__getitem__(0)

            predictions_dict = {}
            # run data through each classification model
            for key,mod in model_dict.items():

                # Run predictions and add to dictionary
                if key in ['zero_cell_model', 'one_cell_model' , 'two_cell_model', 'geq_three_cell_model']:
                    continue

                mm3.information('Predicting probability of {} events in FOV {}, trap {}.'.format('_'.join(key.split('_')[:-1]), fov_id, peak_id))
                predictions_dict['{}_predictions'.format(key)] =  mod.predict(cell_info)

            G,graph_df = mm3.initialize_track_graph(peak_id=peak_id, 
                                                    fov_id=fov_id, 
                                                    experiment_name=p['experiment_name'], 
                                                    predictions_dict=predictions_dict,
                                                    regions_by_time = regions_by_time,
                                                    born_threshold=0.85,
                                                    appear_threshold=0.85)

            # tracks[fov_id][peak_id] = mm3.create_lineages_from_graph_2(G, graph_df, fov_id, peak_id)
            tracks.update(mm3.create_lineages_from_graph(G, graph_df, fov_id, peak_id))

    mm3.information("Finished lineage creation.")

    ### Now prune and save the data.
    mm3.information("Saving cell data.")

    ### save the cell data. Use the script mm3_OutputData for additional outputs.
    # All cell data (includes incomplete cells)
    if not os.path.isdir(p['cell_dir']):
        os.mkdir(p['cell_dir'])

    with open(p['cell_dir'] + '/all_cells.pkl', 'wb') as cell_file:
        pickle.dump(tracks, cell_file, protocol=pickle.HIGHEST_PROTOCOL)

    mm3.information("Finished curating and saving cell data.")