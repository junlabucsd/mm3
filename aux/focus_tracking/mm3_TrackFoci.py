#!/usr/bin/python
from __future__ import print_function
import six

# import modules
import sys
import os
import inspect
import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import cPickle as pickle
except:
    import pickle
import scipy.io as sio

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
import mm3_GUI_helpers as GUI

# set switches and parameters
parser = argparse.ArgumentParser(prog='python mm3_TrackFoci.py',
                                 description='Finds foci.')
parser.add_argument('-f', '--paramfile', type=str,
                    required=True, help='Yaml file containing parameters.')
parser.add_argument('-o', '--fov',  type=str,
                    required=False, help='List of fields of view to analyze. Input "1", "1,2,3", or "1-10", etc.')
parser.add_argument('-c', '--cellfile', type=str,
                    required=False, help='Path to Cell object dicionary to analyze. Defaults to complete_cells_foci.pkl.')
namespace = parser.parse_args()

# Load the project parameters file
mm3.information('Loading experiment parameters.')
if namespace.paramfile:
    param_file_path = namespace.paramfile
else:
    mm3.warning('No param file specified. Using 100X template.')
    param_file_path = 'yaml_templates/p_SJ110_100X.yaml'

p = mm3.init_mm3_helpers(param_file_path) # initialized the helper library

# load cell file
mm3.information('Loading cell data.')
if namespace.cellfile:
    cell_file_path = os.path.join(p['cell_dir'],namespace.cellfile)
else:
    cell_file_path = os.path.join(p['cell_dir'], 'complete_cells_foci.pkl')

with open(cell_file_path, 'rb') as cell_file:
    Cells = pickle.load(cell_file,encoding='latin1')

# load specs file
specs = mm3.load_specs()

# load time table. Puts in p dictionary
mm3.load_time_table()

time_table = p['time_table']
times_all = []
for fov in p['time_table']:
    times_all = np.append(times_all, [int(x) for x in time_table[fov].keys()])
times_all = np.unique(times_all)
times_all = np.sort(times_all)
times_all = np.array(times_all,np.int_)

if namespace.fov:
    if '-' in namespace.fov:
        user_spec_fovs = range(int(namespace.fov.split("-")[0]),
                               int(namespace.fov.split("-")[1])+1)
    else:
        user_spec_fovs = [int(val) for val in namespace.fov.split(",")]
else:
    user_spec_fovs = []

# make list of FOVs to process (keys of channel_mask file)
fov_id_list = sorted([fov_id for fov_id in specs.keys()])

# remove fovs if the user specified so
if user_spec_fovs:
    fov_id_list[:] = [fov for fov in fov_id_list if fov in user_spec_fovs]
### foci analysis

# create dictionary which organizes cells by fov and peak_id
Cells_by_peak = mm3_plots.organize_cells_by_channel(Cells, specs)

color = p['foci']['foci_plane']

ncc = p['foci']['n_cc']

## get all replication cycles as a nested dictionary, indexed by FOV & peak
rep_traces = mm3.extract_foci_dict(fov_id_list,Cells_by_peak)

## do some pruning of traces that are obviously erroneous ids
min_trace_length = p['foci']['min_c'] # minimum C period length in time steps
max_trace_length = p['foci']['max_c'] # max C period length in time steps

fTraces = {}
for fov_id in fov_id_list:
    fTraces[fov_id] = {}
    for peak_id, spec in specs[fov_id].items():
        fTraces[fov_id][peak_id] = {}
        try:
            for trace_id,trace in rep_traces[fov_id][peak_id].items():
                if min_trace_length < len(trace.times) < max_trace_length:
                    fTraces[fov_id][peak_id][trace_id] = trace

        except KeyError:
            pass


cell_filename = os.path.basename(cell_file_path)

## save out dictionary of trace objects
with open(os.path.join(p['cell_dir'], 'rep_traces.pkl'), 'wb') as trace_file:
    pickle.dump(fTraces, trace_file, protocol=pickle.HIGHEST_PROTOCOL)
