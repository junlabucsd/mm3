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
import pandas as pd
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

    # get switches and parameters
    try:
        opts, args = getopt.getopt(sys.argv[1:],"f:o:")
        # switches which may be overwritten
        param_file_path = ''
    except getopt.GetoptError:
        warning('No arguments detected (-f -o).')

    for opt, arg in opts:
        if opt == '-f':
            param_file_path = arg # parameter file path

    # Load the project parameters file
    if len(param_file_path) == 0:
        raise ValueError("A parameter file must be specified (-f <filename>).")
    information ('Loading experiment parameters.')
    p = mm3.init_mm3_helpers(param_file_path) # loads and returns

    # load specs file
    with open(p['ana_dir'] + '/specs.pkl', 'r') as specs_file:
        specs = pickle.load(specs_file)

    # load cell data dict.
    with open(p['cell_dir'] + 'complete_cells.pkl', 'r') as cell_file:
        Complete_Cells = pickle.load(cell_file)

    # Just get the mother cells, as most people only care about those.
    Mother_Cells = mm3.find_mother_cells(Complete_Cells)

    ### From here, change flags to True for different data transformations that you want
    # Save complete cells into a dictionary of dictionaries.
    if False:
        # # Use all cells
        # Cells_dict = {cell_id : vars(cell) for cell_id, cell in Complete_Cells.iteritems()}

        # Or just mothers
        Cells_dict = {cell_id : vars(cell) for cell_id, cell in Mother_Cells.iteritems()}

        # save pickle version.
        with open(p['cell_dir']r + '/cells_dict.pkl', 'wb') as cell_file:
            pickle.dump(Cells_dict, cell_file)

        # # The text file version of the dictionary is good for easy glancing
        # with open(p['cell_dir'] + '/complete_cells_dict.txt', 'w') as cell_file:
        #     pprint(Cells_dict, stream=cell_file)

    # All cells and mother cells saved to a matlab file.
    if False:
        # # All cells
        # with open(p['cell_dir'] + '/complete_cells.mat', 'wb') as cell_file:
        #     savemat(cell_file, Complete_Cells)

        # Just mother cells
        with open(p['cell_dir'] + '/mother_cells.mat', 'wb') as cell_file:
            savemat(cell_file, Mother_Cells)

    # Save a big .csv of all the cell data (JT's format)
    if False:
        # # use all cells
        # Cells_dict = {cell_id : vars(cell) for cell_id, cell in Complete_Cells.iteritems()}

        # or just mothers
        Cells_dict = {cell_id : vars(cell) for cell_id, cell in Mother_Cells.iteritems()}

        # pandas dataframe wants to be converted from a dict of dicts.
        Cells_df = pd.DataFrame(Cells_dict).transpose() # columns as data types
        # organize the order of the rows
        Cells_df = Cells_df.sort(columns=['fov', 'peak', 'birth_time', 'birth_label'])
        # decide the order of the columns
        include_columns = ['id', 'fov', 'peak', 'birth_label',
                           'birth_time', 'division_time',
                           'sb', 'sd', 'delta', 'elong_rate', 'tau', 'septum_position',
                           'parent', 'daughters']
                           #'times', 'lengths', 'widths', 'areas',

        Cells_df = Cells_df[include_columns]
        # convert some columns to numeric type for better formatting
        float_columns = ['sb', 'sd', 'delta', 'elong_rate', 'tau', 'septum_position']
        Cells_df[float_columns] = Cells_df[float_columns].astype(np.float)

        Cells_df.to_csv(p['cell_dir'] + 'cells.csv', sep='\t', float_format='%.4f',
                        header=True, index=False)

    # Save csv in Sattar's format
