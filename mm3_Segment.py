#!/usr/bin/python
from __future__ import print_function

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
        opts, args = getopt.getopt(sys.argv[1:],"f:o:")
        # switches which may be overwritten
        specify_fovs = False
        user_spec_fovs = []
        param_file_path = ''
    except getopt.GetoptError:
        mm3.warning('No arguments detected (-f -o).')

    for opt, arg in opts:
        if opt == '-o':
            try:
                specify_fovs = True
                for fov_to_proc in arg.split(","):
                    user_spec_fovs.append(int(fov_to_proc))
            except:
                mm3.warning("Couldn't convert argument to an integer:",arg)
                raise ValueError
        if opt == '-f':
            param_file_path = arg # parameter file path

    # Load the project parameters file & initialized the helper library
    p = mm3.init_mm3_helpers(param_file_path)

    # create segmenteation and cell data folder if they don't exist
    if not os.path.exists(p['seg_dir']) and p['output'] == 'TIFF':
        os.makedirs(p['seg_dir'])
    if not os.path.exists(p['cell_dir']):
        os.makedirs(p['cell_dir'])

    # load specs file
    try:
        with open(p['ana_dir'] + '/specs.pkl', 'r') as specs_file:
            specs = pickle.load(specs_file)
    except:
        mm3.warning('Could not load specs file.')
        raise ValueError

    # make list of FOVs to process (keys of channel_mask file)
    fov_id_list = sorted([fov_id for fov_id in specs.keys()])

    # remove fovs if the user specified so
    if specify_fovs:
        fov_id_list[:] = [fov for fov in fov_id_list if fov in user_spec_fovs]

    mm3.information("Processing %d FOVs." % len(fov_id_list))

    ### Do Segmentation by FOV and then peak #######################################################
    if do_segmentation:
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
    if do_lineages:
        mm3.information("Creating cell lineages.")

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
        # with open(p['cell_dir'] + '/all_cells.pkl', 'wb') as cell_file:
        #     pickle.dump(Cells, cell_file)

        # Just the complete cells, those with mother and daugther
        # This is a dictionary of cell objects.
        with open(p['cell_dir'] + '/complete_cells.pkl', 'wb') as cell_file:
            pickle.dump(Complete_Cells, cell_file)

        mm3.information("Finished curating and saving cell data.")
