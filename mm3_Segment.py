#!/usr/bin/python
from __future__ import print_function
def warning(*objs):
    print(time.strftime("%H:%M:%S Warning:", time.localtime()), *objs, file=sys.stderr)
def information(*objs):
    print(time.strftime("%H:%M:%S", time.localtime()), *objs, file=sys.stdout)

# import modules
import sys
import os
import time
import inspect
import getopt
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
    # hardcoded parameters
    do_segmentation = True # make or load segmentation?
    do_lineages = True # should lineages be made after segmentation?

    # get switches and parameters
    try:
        opts, args = getopt.getopt(sys.argv[1:],"f:o:s:")
        # switches which may be overwritten
        specify_fovs = False
        user_spec_fovs = []
        start_with_fov = -1
        param_file = ""
    except getopt.GetoptError:
        warning('No arguments detected (-f -s -o).')

    for opt, arg in opts:
        if opt == '-o':
            try:
                specify_fovs = True
                for fov_to_proc in arg.split(","):
                    user_spec_fovs.append(int(fov_to_proc))
            except:
                warning("Couldn't convert argument to an integer:",arg)
                raise ValueError
        if opt == '-s':
            try:
                start_with_fov = int(arg)
            except:
                warning("Couldn't convert argument to an integer:",arg)
                raise ValueError
        if opt == '-f':
            param_file_path = arg # parameter file path

    # Load the project parameters file
    if len(param_file_path) == 0:
        raise ValueError("a parameter file must be specified (-f <filename>).")
    information ('Loading experiment parameters.')
    with open(param_file_path, 'r') as param_file:
        p = yaml.safe_load(param_file) # load parameters into dictionary

    mm3.init_mm3_helpers(param_file_path) # initialized the helper library

    # assign shorthand directory names
    ana_dir = p['experiment_directory'] + p['analysis_directory']
    seg_dir = p['experiment_directory'] + p['analysis_directory'] + 'segmented/'
    cell_dir = p['experiment_directory'] + p['analysis_directory'] + 'cell_data/'

    # create segmenteation and cell data folder if it doesn't exist
    if not os.path.exists(seg_dir):
        os.makedirs(seg_dir)
    if not os.path.exists(cell_dir):
        os.makedirs(cell_dir)

    # load specs file
    try:
        with open(ana_dir + '/specs.pkl', 'r') as specs_file:
            specs = pickle.load(specs_file)
    except:
        warning('Could not load specs file.')
        raise ValueError

    # make list of FOVs to process (keys of channel_mask file)
    fov_id_list = sorted([fov_id for fov_id in specs.keys()])

    # remove fovs if the user specified so
    if specify_fovs:
        fov_id_list[:] = [fov for fov in fov_id_list if fov in user_spec_fovs]
    if start_with_fov > 0:
        fov_id_list[:] = [fov for fov in fov_id_list if fov_id >= start_with_fov]

    information("Found %d FOVs to process." % len(fov_id_list))

    ### Do Segmentation by FOV and then peak #######################################################
    if do_segmentation:
        information("Segmenting channels.")

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

        information("Finished segmentation.")

    ### Create cell lineages from segmented images
    if do_lineages:
        information("Creating cell lineages.")

        # This dictionary holds information for all cells
        Cells = {}

        # do lineage creation per fov, so pooling can be done by peak
        for fov_id in fov_id_list:
            # update will add the output from make_lineages_function, which is a
            # dict of Cell entries, into Cells
            Cells.update(mm3.make_lineages_fov(fov_id, specs))

        information("Finished lineage creation.")

        ### Now prune and save the data.
        information("Curating and saving cell data.")

        # this returns only cells with a parent and daughters
        Complete_Cells = mm3.find_complete_cells(Cells)

        ### save the cell data. Edit this for how you want the data

        # All cell data (includes incomplete cells)
        # with open(cell_dir + '/all_cells.pkl', 'wb') as cell_file:
        #     pickle.dump(Cells, cell_file)

        # Just the complete cells, those with mother and daugther
        with open(cell_dir + '/complete_cells.pkl', 'wb') as cell_file:
            pickle.dump(Complete_Cells, cell_file)

        # Same thing but save it to a mat file
        # with open(cell_dir + '/complete_cells.mat', 'wb') as cell_file:
        #     savemat(cell_file, Complete_Cells)

        # convert the objects in the dictionary to dictionaries and save it to pickle and text
        # Complete_Cells_dict = {cell_id : vars(cell) for cell_id, cell in Complete_Cells.iteritems()}
        # with open(cell_dir + '/complete_cells_dict.pkl', 'wb') as cell_file:
        #     pickle.dump(Complete_Cells_dict, cell_file)
        # with open(cell_dir + '/complete_cells_dict.txt', 'w') as cell_file:
        #     pprint(Complete_Cells_dict, stream=cell_file)


        information("Finished curating and saving cell data.")
