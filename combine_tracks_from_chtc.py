#!/usr/bin/env python3

# import modules
import sys
import os
import inspect
import glob
import argparse
import skimage
from skimage import measure, io
from pprint import pprint # for human readable file output
try:
    import cPickle as pickle
except:
    import pickle

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

#%%
# when using this script as a function and not as a library the following will execute
if __name__ == "__main__":

    # set switches and parameters
    parser = argparse.ArgumentParser(
        prog='python combine_tracks_from_chtc.py',
        description='CHTC saves a separate track file for each fov/peak. Here we combine them.'
    )
    parser.add_argument(
        '-f',
        '--paramfile',
        type=str,
        required=True,
        help='Yaml file containing parameters.'
    )

    namespace = parser.parse_args()

    # Load the project parameters file
    mm3.information('Loading experiment parameters.')
    if namespace.paramfile:
        param_file_path = namespace.paramfile
    else:
        mm3.warning('No param file specified. Using 100X template.')
        param_file_path = 'yaml_templates/params_SJ110_100X.yaml'
    p = mm3.init_mm3_helpers(param_file_path) # initialized the helper library

    # Get file names
    fnames = glob.glob(os.path.join(p['cell_dir'], "{}*_tracks.pkl".format(p['experiment_name'])))

    ### Now prune and save the data.
    mm3.information("Reading cell data from each file and combining into one.")

    tracks = {}

    for fname in fnames:
        with open(fname, 'rb') as cell_file:
            cell_data = pickle.load(cell_file)
        os.remove(fname)
        tracks.update(cell_data)

    with open(p['cell_dir'] + '/all_cells.pkl', 'wb') as cell_file:
        pickle.dump(tracks, cell_file, protocol=pickle.HIGHEST_PROTOCOL)

    if os.path.isfile(os.path.join(p['cell_dir'], 'complete_cells.pkl')):
        os.remove(os.path.join(p['cell_dir'], 'complete_cells.pkl'))

    os.symlink(
        os.path.join(p['cell_dir'], 'all_cells.pkl'),
        os.path.join(p['cell_dir'], 'complete_cells.pkl')
    )

    mm3.information("Finished curating and saving cell data.")
