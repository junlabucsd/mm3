#!/usr/bin/python
from __future__ import print_function
import six
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
import multiprocessing
from multiprocessing import Pool
from functools import partial
from skimage.measure import profile_line # used for ring an nucleoid analysis
import matplotlib.pyplot as plt
import matplotlib as mpl
import tifffile as tiff
from skimage import exposure

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

    # set switches and parameters
    parser = argparse.ArgumentParser(prog='python mm3_kymograph.py',
                                     description='y projection by channel')
    parser.add_argument('-f', '--paramfile',  type=str,
                        required=True, help='Yaml file containing parameters.')
    parser.add_argument('-o', '--fov',  type=str,
                        required=False, help='List of fields of view to analyze. Input "1", "1,2,3", or "1-10", etc.')
    parser.add_argument('-j', '--nproc',  type=int,
                        required=False, help='Number of processors to use.')
    namespace = parser.parse_args()

    # Load the project parameters file
    mm3.information('Loading experiment parameters.')
    if namespace.paramfile:
        param_file_path = namespace.paramfile
    else:
        mm3.warning('No param file specified. Using 100X template.')
        param_file_path = 'yaml_templates/params_SJ110_100X.yaml'
    params = mm3.init_mm3_helpers(param_file_path) # initialized the helper library

    if namespace.fov:
        if '-' in namespace.fov:
            user_spec_fovs = range(int(namespace.fov.split("-")[0]),
                                   int(namespace.fov.split("-")[1])+1)
        else:
            user_spec_fovs = [int(val) for val in namespace.fov.split(",")]
    else:
        user_spec_fovs = []

    plot_dir = os.path.join(params['experiment_directory'],params['analysis_directory'],'kymograph')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    specs = mm3.load_specs()

    fov_id_list = sorted([fov_id for fov_id in specs.keys()])

    # remove fovs if the user specified so
    if user_spec_fovs:
        fov_id_list[:] = [fov for fov in fov_id_list if fov in user_spec_fovs]

    for fov_id in fov_id_list:
        # determine which peaks are to be analyzed (those which have been subtracted)
        ana_peak_ids = []
        for peak_id, spec in six.iteritems(specs[fov_id]):
            if spec == 1: # 0 means it should be used for empty, -1 is ignore, 1 is analyzed
                ana_peak_ids.append(peak_id)
        ana_peak_ids = sorted(ana_peak_ids) # sort for repeatability
        for peak_id in ana_peak_ids:

            sub_stack_fl = mm3.load_stack(fov_id, peak_id, color='sub_'+params['foci']['foci_plane'])
            #
            # fl_proj = np.transpose(np.array([profile_line(sub_stack_fl[t],(0,18),(255,18),linewidth=30,order=1, mode='constant', cval=0) for t in range(sub_stack_fl.shape[0])]))
            fl_proj = np.transpose(np.max(sub_stack_fl, axis=2))
            channel_filename = os.path.join(plot_dir, params['experiment_name'] + '_xy%03d_p%04d_%s.tif' % (fov_id, peak_id, params['foci']['foci_plane']))

            # seg_stack = mm3.load_stack(fov_id, peak_id, color='seg_otsu')
            # seg_proj = np.transpose(np.max(seg_stack, axis=2))
            # seg_channel_filename = os.path.join(plot_dir, params['experiment_name'] + '_xy%03d_p%04d_seg.tif' % (fov_id, peak_id))

            # save stack
            print('Saving FOV %2d, peak %3d' % (fov_id, peak_id))
            tiff.imsave(channel_filename, fl_proj)
            # tiff.imsave(seg_channel_filename, seg_proj)
