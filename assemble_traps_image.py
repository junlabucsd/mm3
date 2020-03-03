#!/usr/bin/env python3
from __future__ import print_function, division
import six

# import modules
import sys
import os
# import time
import inspect
import argparse
import numpy as np
import yaml
from pprint import pprint # for human readable file output
try:
    import cPickle as pickle
except:
    import pickle
import numpy as np
from scipy.io import savemat
from skimage import io
import skimage
import matplotlib.pyplot as plt

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

if __name__ == "__main__":

    # set switches and parameters
    parser = argparse.ArgumentParser(prog='python assemble_traps_image.py',
                                     description='Assemble and save images of traps side-by-side.')
    parser.add_argument('-f', '--paramfile', type=str,
                        required=True, help='Yaml file containing parameters.')
    parser.add_argument('--frame', required=True, type=int, help="Defines the frame (1-indexed) to be sliced from each stack.")
    parser.add_argument('-o', '--fov', type=str,
                        required=False, help='List of fields of view to analyze. Input "1", "1,2,3", or "1-10", etc.')
    parser.add_argument('--cell_segs', action='store_true',
                        required=False, help='Apply this argument if you would like cell segmentation results in addition to phase images.')
    parser.add_argument('--focus_segs', action='store_true',
                        required=False, help='Apply this argument if you would like focus segmentation results in addition to phase images.')
    parser.add_argument('--fluorescent_channels', type=str,
                        required=False, help='List of channels (as integers) to include in addition to the phase channel.')
    
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

    # load specs file
    specs = mm3.load_specs()
    # print(specs) # for debugging

    # make list of FOVs to process (keys of channel_mask file)
    fov_id_list = sorted([fov_id for fov_id in specs.keys()])

    # remove fovs if the user specified so
    if user_spec_fovs:
        fov_id_list[:] = [fov for fov in fov_id_list if fov in user_spec_fovs]

    peaks_list = [peak_id for peak_id,val in specs[fov_id_list[0]].items() if val == 1]
    img_height, img_width = mm3.load_stack(fov_id_list[0], peaks_list[0], color=p['phase_plane'])[0,:,:].shape

    # how many images total will we concatenate horizontally?
    img_count = 0
    for fov_id in fov_id_list:
        fov_peak_count = len([peak_id for peak_id,val in specs[fov_id].items() if val == 1])
        img_count += fov_peak_count

    # placeholder array of img_height, and proper width to hold all pixels from this fov
    phase_arr = np.zeros((230,img_width*img_count), 'uint16')

    if namespace.cell_segs:
        # p['seg_dir'] = 'segmented'
        seg_arr = np.zeros((img_230height,img_width*img_count), 'uint16')

    if namespace.focus_segs:
        # p['foci_seg_dir'] = 'segmented_foci'
        focus_arr = np.zeros((230,img_width*img_count), 'uint16')

    if namespace.fluorescent_channels:
        fluor_arrays = {}
        for fluorescent_channel in namespace.fluorescent_channels:
            fluor_arrays[fluorescent_channel] = np.zeros((230,img_width*img_count), 'uint16')

    frame = namespace.frame
    frame_idx = frame - 1

    img_counter = 0
    for fov_id in fov_id_list:

        print("concatenating images from fov_id {}.".format(fov_id))
        peaks_list = [peak_id for peak_id,val in specs[fov_id].items() if val == 1]

        for i,peak_id in enumerate(peaks_list):

            start_x = img_counter * img_width
            end_x = start_x + img_width

            if namespace.cell_segs:
                # set segmentation image name for saving and loading segmented images
                img = mm3.load_stack(fov_id, peak_id, color='seg_unet')[frame_idx,:230,:]
                seg_arr[:,start_x:end_x] = img

            if namespace.focus_segs:
                # set segmentation image name for saving and loading segmented images
                img = mm3.load_stack(fov_id, peak_id, color='foci_seg_unet')[frame_idx,:230,:]
                focus_arr[:,start_x:end_x] = img
                
            if namespace.fluorescent_channels:
                for fluorescent_channel in namespace.fluorescent_channels:
                    img = mm3.load_stack(fov_id, peak_id, color='c{}'.format(fluorescent_channel))[frame_idx,:230,:]
                    fluor_arrays[fluorescent_channel][:,start_x:end_x] = img
            
            # now for the phase images
            img = mm3.load_stack(fov_id, peak_id, color=p['phase_plane'])[frame_idx,:230,:]
            phase_arr[:,start_x:end_x] = img

            img_counter += 1

    fname = os.path.join(p['experiment_directory'], '{}_t{:0=4}_combined_phase_peaks.png'.format(p['experiment_name'], frame))
    io.imsave(fname, skimage.img_as_ubyte(phase_arr))

    if namespace.cell_segs:
        fname = os.path.join(p['experiment_directory'], '{}_t{:0=4}_combined_cell_seg_peaks.png'.format(p['experiment_name'], frame))
        io.imsave(fname, skimage.img_as_ubyte(seg_arr))

    if namespace.focus_segs:
        fname = os.path.join(p['experiment_directory'], '{}_t{:0=4}_combined_focus_seg_peaks.png'.format(p['experiment_name'], frame))
        io.imsave(fname, skimage.img_as_ubyte(focus_arr))

    if namespace.fluorescent_channels:
        for fluorescent_channel in namespace.fluorescent_channels:
            fname = os.path.join(p['experiment_directory'], '{}_t{:0=4}_combined_c{}_peaks.png'.format(p['experiment_name'], frame, fluorescent_channel))
            io.imsave(fname, fluor_arrays[fluorescent_channel])
    
