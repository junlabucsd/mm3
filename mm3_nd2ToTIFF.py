#!/usr/bin/python
from __future__ import print_function

# import modules
import sys
import os
import time
import inspect
import argparse
import yaml
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
import warnings

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

# supress the warning this always gives
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import tifffile as tiff

# this is the mm3 module with all the useful functions and classes
import mm3_helpers as mm3

### Main script
if __name__ == "__main__":
    '''
    This script converts a Nikon Elements .nd2 file to individual TIFF files per time point. Multiple color planes are stacked in each time point to make a multipage TIFF.
    '''

    # set switches and parameters
    parser = argparse.ArgumentParser(prog='python mm3_nd2ToTIFF.py',
                                     description='Export .nd2 files to TIFF.')
    parser.add_argument('-f', '--paramfile',  type=file,
                        required=True, help='Yaml file containing parameters.')
    parser.add_argument('-o', '--fov',  type=str,
                        required=False, help='List of fields of view to analyze. Input "1", "1,2,3", etc. ')
    namespace = parser.parse_args()

    # Load the project parameters file
    mm3.information('Loading experiment parameters.')
    if namespace.paramfile.name:
        param_file_path = namespace.paramfile.name
    else:
        mm3.warning('No param file specified. Using 100X template.')
        param_file_path = 'yaml_templates/params_SJ110_100X.yaml'
    p = mm3.init_mm3_helpers(param_file_path) # initialized the helper library

    if namespace.fov:
        user_spec_fovs = [int(val) for val in namespace.fov.split(",")]
    else:
        user_spec_fovs = []

    # number of rows of channels. Used for cropping.
    number_of_rows = p['nd2ToTIFF']['number_of_rows']

    # cropping
    vertical_crop = []
    if number_of_rows == 1:
        try:
            # This is for percentage crop
            vertical_crop = [p['nd2ToTIFF']['crop_ymin'], p['nd2ToTIFF']['crop_ymax']]
        except KeyError:
            pass
    elif number_of_rows == 2:
        # if there is more than one row, make a list of pairs
        # [[y1_min, y1_max], [y2_min, y2_max]]
        vertical_crop = p['nd2ToTIFF']['2row_crop']

    # number between 0 and 9, 0 is no compression, 9 is most compression.
    tif_compress = p['nd2ToTIFF']['tiff_compress']

    # set up image and analysis folders if they do not already exist
    if not os.path.exists(p['TIFF_dir']):
        os.makedirs(p['TIFF_dir'])

    # Load ND2 files into a list for processing
    if p['nd2ToTIFF']['external_directory']:
        nd2files = glob.glob(os.path.join(p['nd2ToTIFF']['external_directory'], "*.nd2"))
        mm3.information("Found %d files to analyze from external directory." % len(nd2files))
    else:
        mm3.information("Experiment directory: {:s}".format(p['experiment_directory']))
        nd2files = glob.glob(os.path.join(p['experiment_directory'], "*.nd2"))
        mm3.information("Found %d files to analyze in experiment directory." % len(nd2files))

    for nd2_file in nd2files:
        file_prefix = os.path.split(os.path.splitext(nd2_file)[0])[1]
        mm3.information('Extracting %s ...' % file_prefix)

        # load the nd2. the nd2f file object has lots of information thanks to pims
        with pims_nd2.ND2_Reader(nd2_file) as nd2f:
            try:
                starttime = nd2f.metadata['time_start_jdn'] # starttime is jd
                mm3.information('Starttime got from nd2 metadata.')
            except ValueError:
                # problem with the date
                jdn = mm3.julian_day_number()
                nd2f._lim_metadata_desc.dTimeStart = jdn
                starttime = nd2f.metadata['time_start_jdn'] # starttime is jd
                mm3.information('Starttime found from lim.')

            # get the color names out. Kinda roundabout way.
            planes = [nd2f.metadata[md]['name'] for md in nd2f.metadata if md[0:6] == u'plane_' and not md == u'plane_count']

            # this insures all colors will be saved when saving tiff
            if len(planes) > 1:
                nd2f.bundle_axes = [u'c', u'y', u'x']

            # extraction range is the time points that will be taken out. Note the indexing,
            # it is zero indexed to grab from nd2, but TIFF naming starts at 1.
            # if there is more than one FOV (len(nd2f) != 1), make sure the user input
            # last time index is before the actual time index. Ignore it.
            if (p['image_start'] < 1):
                p['image_start']=1
            if len(nd2f) > 1 and len(nd2f) < p['image_end']:
                p['image_end'] = len(nd2f)
            extraction_range = range(p['image_start'], p['image_end']+1)

            # loop through time points
            for t in extraction_range:
                # timepoint output name (1 indexed rather than 0 indexed)
                t_id = t - 1
                # set counter for FOV output name
                #fov = fov_naming_start

                for fov_id in range(0, nd2f.sizes[u'm']): # for every FOV
                    # fov_id is the fov index according to elements, fov is the output fov ID
                    fov = fov_id + 1

                    # skip FOVs as specified above
                    if len(user_spec_fovs) > 0 and not (fov in user_spec_fovs):
                        continue

                    # set the FOV we are working on in the nd2 file object
                    nd2f.default_coords[u'm'] = fov_id

                    # get time picture was taken
                    seconds = copy.deepcopy(nd2f[t_id].metadata['t_ms']) / 1000.
                    minutes = seconds / 60.
                    hours = minutes / 60.
                    days = hours / 24.
                    acq_time = starttime + days

                    # get physical location FOV on stage
                    x_um = nd2f[t_id].metadata['x_um']
                    y_um = nd2f[t_id].metadata['y_um']

                    # make dictionary which will be the metdata for this TIFF
                    metadata_t = { 'fov': fov,
                                   't' : t,
                                   'jd': acq_time,
                                   'x': x_um,
                                   'y': y_um,
                                   'planes': planes}
                    metadata_json = json.dumps(metadata_t)

                    # get the pixel information
                    image_data = nd2f[t_id]

                    # crop tiff if specified. Lots of flags for if there are double rows or
                    # multiple colors
                    if vertical_crop:
                        # add extra axis to make below slicing simpler.
                        if len(image_data.shape) < 3:
                            image_data = np.expand_dims(image_data, axis=0)

                        # for just a simple crop
                        if number_of_rows == 1:
                            nc, H, W = image_data.shape
                            ylo = int(vertical_crop[0]*H)
                            yhi = int(vertical_crop[1]*H)
                            image_data = image_data[:, ylo:yhi, :]

                            # save the tiff
                            tif_filename = file_prefix + "_t%04dxy%02d.tif" % (t, fov)
                            mm3.information('Saving %s.' % tif_filename)
                            tiff.imsave(os.path.join(p['TIFF_dir'], tif_filename), image_data, description=metadata_json, compress=tif_compress, photometric='minisblack')

                        # for dealing with two rows of channel
                        elif number_of_rows == 2:
                            # cut and save top row
                            image_data_one = image_data[:,vertical_crop[0][0]:vertical_crop[0][1],:]
                            tif_filename = file_prefix + "_t%04dxy%02d_1.tif" % (t, fov)
                            mm3.information('Saving %s.' % tif_filename)
                            tiff.imsave(os.path.join(p['TIFF_dir'], tif_filename), image_data_one, description=metadata_json, compress=tif_compress, photometric='minisblack')

                            # cut and save bottom row
                            metadata_t['fov'] = fov # update metdata
                            metadata_json = json.dumps(metadata_t)
                            image_data_two = image_data[:,vertical_crop[1][0]:vertical_crop[1][1],:]
                            tif_filename = file_prefix + "_t%04dxy%02d_2.tif" % (t, fov)
                            mm3.information('Saving %s.' % tif_filename)
                            tiff.imsave(os.path.join(p['TIFF_dir'], tif_filename), image_data_two, description=metadata_json, compress=tif_compress, photometric='minisblack')

                    else: # just save the image if no cropping was done.
                        tif_filename = file_prefix + "_t%04dxy%02d.tif" % (t, fov)
                        mm3.information('Saving %s.' % tif_filename)
                        tiff.imsave(os.path.join(p['TIFF_dir'], tif_filename), image_data, description=metadata_json, compress=tif_compress, photometric='minisblack')

                    # increase FOV counter
                    fov += 1
