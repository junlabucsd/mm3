#!/usr/bin/python
from __future__ import print_function
def warning(*objs):
    print(time.strftime("%H:%M:%S Error:", time.localtime()), *objs, file=sys.stderr)
def information(*objs):
    print(time.strftime("%H:%M:%S", time.localtime()), *objs, file=sys.stdout)

# import modules
import sys
import os
import time
import inspect
import getopt
import yaml
import traceback
import fnmatch
import glob
import math
import copy
import json
try:
    import cPickle as pickle
except:
    import pickle
import numpy as np
import pims_nd2

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

import tifffile as tiff

### Main script
if __name__ == "__main__":



    # switches
    try:
        opts, args = getopt.getopt(sys.argv[1:], "f:o:s:x:n:")
        # parameters to be overwritten by switches
        param_file = ""
        specify_fovs = []
        start_fov = -1
        external_directory = ""
        fov_num_offset = 0
    except getopt.GetoptError:
        warning('No arguments detected (-f -o -s -x -n).')

    for opt, arg in opts:
        if opt == '-f':
            param_file_path = arg
        if opt == '-o':
            arg.replace(" ", "")
            [specify_fovs.append(int(argsplit)) for argsplit in arg.split(",")]
        if opt == '-s':
            try:
                start_fov = int(arg)
            except:
                warning("Could not convert start parameter (%s) to an integer." % arg)
                raise ValueError
        if opt == '-x':
            external_directory = arg
        if opt == '-n':
            try:
                fov_num_offset = int(arg)
            except:
                raise ValueError("Could not convert FOV numbering offset (%s) to an integer." % arg)
            if fov_num_offset < 0:
                raise ValueError("FOV offset (%s) should probably be positive." % fov_num_offset)

    # Load the project parameters file
    if len(param_file_path) == 0:
        raise ValueError("A parameter file must be specified (-f <filename>).")
    information ('Loading experiment parameters.')
    with open(param_file_path, 'r') as param_file:
        p = yaml.safe_load(param_file) # load parameters into dictionary

    # assign shorthand directory names
    TIFF_dir = p['experiment_directory'] + p['image_directory'] # source of images
    ana_dir = p['experiment_directory'] + p['analysis_directory']

    # set up image and analysis folders if they do not already exist
    if not os.path.exists(TIFF_dir):
        os.makedirs(TIFF_dir)
    if not os.path.exists(ana_dir):
        os.makedirs(ana_dir)

    # Load ND2 files into a list for processing
    if len(external_directory) > 0:
        nd2files = glob.glob(external_directory + "*.nd2")
        information("Found %d files to analyze from external file." % len(nd2files))
    else:
        nd2files = glob.glob(p['experiment_directory'] + "*.nd2")
        information("Found %d files to analyze in experiment directory." % len(nd2files))

    for nd2_file in nd2files:
        information('Extracting %s ...' % nd2_file)

        i_mdata = {} # saves image metadata

        # load the nd2. the nd2f file object has lots of information thanks to pims
        with pims_nd2.ND2_Reader(nd2_file) as nd2f:
            starttime = nd2f.metadata['time_start_jdn'] # starttime is jd

            # # set bundle colors together if there are multiple colors
            # if u'c' in nd2f.sizes.keys():
            #     nd2f.bundle_axes = [u'c', u'y', u'x']

            # get the color names out. Kinda roundabout way.
            file_colors = [nd2f.metadata[md]['name'] for md in nd2f.metadata if md[0:6] == u'plane_' and not md == u'plane_count']

            # get timepoints for extraction. Check is for multiple FOVs or not
            if len(nd2f) == 1:
                extraction_range = [0,]
            else:
                extraction_range = range(0, len(nd2f) - 1)

            # loop through time points
            for t_id in extraction_range:
                for fov in range(0, nd2f.sizes[u'm']): # for every FOV

                    # skip FOVs as specified above
                    if len(specify_fovs) > 0 and not (fov + 1) in specify_fovs:
                        continue
                    if start_fov > -1 and fov + 1 < start_fov:
                        continue

                    # set the FOV we are working on in the nd2 file object
                    nd2f.default_coords[u'm'] = fov

                    # get time picture was taken
                    seconds = copy.deepcopy(nd2f[t_id].metadata['t_ms']) / 1000.
                    minutes = seconds / 60.
                    hours = minutes / 60.
                    days = hours / 24.
                    acq_time = starttime + days

                    # Put in the calcultated absolute acquisition time to the existing meta
                    nd2f[t_id].metadata['acq_time'] = acq_time
                    # This json of the metadata will be put in the tiff directly
                    metadata_json = json.dumps(nd2f[t_id].metadata)

                    # # save stack of colors if there are colors
                    # if u'c' in nd2f.sizes.keys():
                    #     for c_id in range(0, nd2f.sizes[u'c']):
                    #         tif_filename = nd2_file.split(".nd") + "_t%04dxy%03dc%01d.tif" %
                    #                        (t_id+1, fov+1 + fov_num_offset, c_id+1)
                    #         if len(np.unique(nd2f[t_id][c_id])) > 1:
                    #             tiff.imsave(p['experiment_directory'] + p['image_directory'] +
                    #                         tif_filename, nd2f[t_id][c_id])
                    #             information('Saving %s.' % tif_filename)
                    #
                    #             # Pull out the metadata
                    #             i_mdata[tif_filename] = nd2f[t_id][c_id].metadata
                    #             # Put in the calcultate absolute acquisition time
                    #             i_mdata[tif_filename]['acq_time'] = acq_time

                    # save a single phase image if no colors
                    # else:

                    tif_filename = nd2_file.split(".nd")[0].split("/")[-1] + "_t%04dxy%03dc1.tif" % (t_id+1, fov+1 + fov_num_offset)
                    information('Saving %s.' % tif_filename)
                    tiff.imsave(TIFF_dir + tif_filename, nd2f[t_id], description=metadata_json)
                    # Save image metadata to dictionary to be saved separately.
                    i_mdata[tif_filename] = nd2f[t_id].metadata

            # # Save the metadata.
            # with open(p['experiment_directory'] + p['analysis_directory'] +
            #     nd2_file.split(".nd")[0] + "_image_metadata.pkl", 'wb') as metadata_file:
            #     pickle.dump(i_mdata, metadata_file, protocol=2)
