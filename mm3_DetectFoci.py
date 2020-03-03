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

def segment_foci_single_file(infile_name, params, namespace):

    mm3.information("Segmenting image {}.".format(infile_name))
    # load model to pass to algorithm
    mm3.information("Loading model...")

    if namespace.modelfile:
        model_file_path = namespace.modelfile
    else:
        model_file_path = params['foci']['foci_model_file']
        
    seg_model = models.load_model(model_file_path,
                              custom_objects={'bce_dice_loss': mm3.bce_dice_loss,
                                                'dice_loss': mm3.dice_loss,
                                                'precision_m': mm3.precision_m,
                                                'recall_m': mm3.recall_m,
                                                'f_precision_m': mm3.f_precision_m})
    mm3.information("Model loaded.")
    mm3.segment_stack_unet(infile_name, seg_model, mode='foci')
    sys.exit("Completed segmenting image {}.".format(infile_name))

# when using this script as a function and not as a library the following will execute
if __name__ == "__main__":

    # set switches and parameters
    parser = argparse.ArgumentParser(prog='python mm3_Segment.py',
                                     description='Segment cells and create lineages.')
    parser.add_argument('-f', '--paramfile', type=str,
                        required=True, help='Yaml file containing parameters.')
    parser.add_argument('-i', '--infile', type=str,
                        required=False, help='Use this argument to segment ONLY on image. Name the single file to be segmented.')
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

    if namespace.infile:

        fname = namespace.infile
        segment_foci_single_file(fname, p, namespace)

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
    seg_model = models.load_model(
        model_file_path,
        custom_objects = {
            'weighted_bce': mm3.weighted_bce,
            'bce_dice_loss': mm3.bce_dice_loss,
            'dice_loss': mm3.dice_loss,
            'precision_m': mm3.precision_m,
            'recall_m': mm3.recall_m,
            'f_precision_m': mm3.f_precision_m
        }
    )
    mm3.information("Model loaded.")

    for fov_id in fov_id_list:
        mm3.segment_fov_foci_unet(fov_id, specs, seg_model, color=p['foci']['foci_plane'])

    del seg_model

    mm3.information("Finished segmentation.")
