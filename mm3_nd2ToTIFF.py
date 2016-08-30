#!/usr/bin/python
from __future__ import print_function
def warning(*objs):
    print(time.strftime("%H:%M:%S Error:", time.localtime()), *objs, file=sys.stderr)
def information(*objs):
    print(time.strftime("%H:%M:%S", time.localtime()), *objs, file=sys.stdout)

# import modules
import sys
import os
#import time
import inspect
import getopt
import yaml
import traceback
#import h5py
import fnmatch
#import struct
#import re
#import glob
#import gevent
import math
import copy
#import datetime
#import jdcal
#import marshal
try:
    import cPickle as pickle
except:
    import pickle
#import marshal
#from multiprocessing import Pool, Manager
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

    # parameters to be overwritten by switches
    param_file = ""
    specify_fovs = []
    start_fov = -1
    external_file = ""
    fov_num_offset = 0
    file_prefix = ""

    # switches
    try:
        opts, args = getopt.getopt(sys.argv[1:],"f:o:s:x:n:")
    except getopt.GetoptError:
        warning('No arguments detected (-f -o -s -x -n).')
    for opt, arg in opts:
        if opt == '-f':
            param_file = arg
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
            external_file = arg
        if opt == '-n':
            try:
                fov_num_offset = int(arg)
            except:
                raise ValueError("Could not convert FOV numbering offset (%s) to an integer." % arg)
            if fov_num_offset < 0:
                raise ValueError("FOV offset (%s) should probably be positive." % fov_num_offset)

    # Load the project parameters file into a dictionary named p
    if len(param_file) == 0:
        raise ValueError("A parameter file must be specified (-f <filename>).")
    information('Loading experiment parameters...')
    with open(param_file) as pfile:
        p = yaml.load(pfile)

    # Load ND2 files into a list for processing
    if len(external_file) > 0:
        nd2files = np.asarray((external_file,))
        information("Found %d files to analyze from external file." % len(nd2files))
    else:
        nd2files = fnmatch.filter(os.listdir(p['experiment_directory']), "*.nd2")
        nd2files = np.asarray(nd2files)
        information("Found %d files to analyze in experiment directory." % len(nd2files))

    # set up image and analysis folders if they do not already exist
    if not os.path.exists(os.path.abspath(p['experiment_directory'] + p['image_directory'])):
        os.makedirs(os.path.abspath(p['experiment_directory'] + p['image_directory']))
    if not os.path.exists(os.path.abspath(p['experiment_directory'] + p['analysis_directory'])):
        os.makedirs(os.path.abspath(p['experiment_directory'] + p['analysis_directory']))

    for nd2_file in nd2files:
        information('Extracting %s ...' % nd2_file)

        idata = [] # saves metadata for all files.
        try:
            # get this specific file name
            if len(external_file) > 0:
                filename = external_file
            else:
                filename = p['experiment_directory'] + nd2_file

            # load the nd2. the nd2f file object has lots of information thanks to pims
            with pims_nd2.ND2_Reader(filename) as nd2f:
                starttime = nd2f.metadata['time_start_jdn'] # starttime is jd
                # set planar bundling if necessary & get color names
                if u'c' in nd2f.sizes.keys():
                    nd2f.bundle_axes = [u'c', u'y', u'x']
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

                        if u'c' in nd2f.sizes.keys():
                            for c_id in range(0, nd2f.sizes[u'c']):
                                new_filename = file_prefix + filename.split(".nd")[0].split("/")[-1] + "_t%04dxy%03dc%01d.tif" % (t_id+1, fov+1 + fov_num_offset, c_id+1)
                                if len(np.unique(nd2f[t_id][c_id])) > 1:
                                    tiff.imsave(p['experiment_directory'] + p['image_directory'] +
                                                new_filename, nd2f[t_id][c_id])
                                    information('Saving %s.' % new_filename)
                                    idata.append((new_filename, 0, acq_time,
                                                  nd2f[t_id].metadata['x_um'],
                                                  nd2f[t_id].metadata['y_um']))
                        else:
                            new_filename = file_prefix + nd2_file.split(".nd")[0].split("/")[-1] + "_t%04dxy%03dc1.tif" % (t_id+1, fov+1 + fov_num_offset)
                            tiff.imsave(p['experiment_directory'] + p['image_directory'] +
                                        new_filename, nd2f[t_id])
                            information('Saving %s.' % new_filename)
                            idata.append((new_filename, 0, acq_time, nd2f[t_id].metadata['x_um'], nd2f[t_id].metadata['y_um']))

            with open(p['experiment_directory'] + p['analysis_directory'] + file_prefix +
                    filename.split(".nd")[0].split("/")[-1] + "_acqtimes.pkl", 'wb') as jdates_file:
                pickle.dump(idata, jdates_file, protocol=2)

        except:
            warning("Error extracting data from " + nd2_file)
            print(sys.exc_info()[0])
            print(sys.exc_info()[1])
            print(traceback.print_tb(sys.exc_info()[2]))
