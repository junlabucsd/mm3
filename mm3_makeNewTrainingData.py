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

from tensorflow.python.keras import models

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
                        required=False, help='List of fields of view to analyze. Input "1", "1,2,3", etc. ')
    parser.add_argument('-j', '--nproc', type=int,
                        required=False, help='Number of processors to use.')
    parser.add_argument('-m', '--modelfile', type=str,
                        required=False, help='Path to trained U-net model.')
    parser.add_argument('-d', '--traindir', type=str,
                        required=True, help='Path to top directory with training data.')
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
        user_spec_fovs = [int(val) for val in namespace.fov.split(",")]
    else:
        user_spec_fovs = []

    # number of threads for multiprocessing
    if namespace.nproc:
        p['num_analyzers'] = namespace.nproc
    mm3.information('Using {} threads for multiprocessing.'.format(p['num_analyzers']))

    if namespace.traindir:
        p['training_dir'] = namespace.traindir
    mm3.information('Will save newly curated training data to {}'.format(p['training_dir']))

    # create segmenteation and cell data folder if they don't exist
    if not os.path.exists(p['seg_dir']) and p['output'] == 'TIFF':
        os.makedirs(p['seg_dir'])
    if not os.path.exists(p['cell_dir']):
        os.makedirs(p['cell_dir'])

    # set segmentation image name for saving and loading segmented images
    p['seg_img'] = 'seg_unet'

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
    if p['segment']['do_segmentation']:
        mm3.information("Segmenting channels using U-net.")

        # load model to pass to algorithm
        mm3.information("Loading model...")

        if namespace.modelfile:
            model_file_path = namespace.modelfile
        else:
            model_file_path = p['segment']['model_file']
        # *** Need parameter for weights
        model = models.load_model(model_file_path,
                                  custom_objects={'bce_dice_loss': mm3.bce_dice_loss,
                                                  'dice_loss': mm3.dice_loss})
        mm3.information("Model loaded.")

        for fov_id in fov_id_list:
            k = mm3.segment_fov_unet(fov_id, specs, model, make_training_data=True, training_dir=p['training_dir'])
            if k:
                if k == ord('c'):
                    break
        if k:
            if k == ord('c'):
                mm3.information("Training data curation cancelled.")
                sys.exit()

        mm3.information("Finished segmentation.")
