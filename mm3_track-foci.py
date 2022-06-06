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

# # user modules
# # realpath() will make your script run, even if you symlink it
# cmd_folder = os.path.realpath(os.path.abspath(
#                               os.path.split(inspect.getfile(inspect.currentframe()))[0]))
# if cmd_folder not in sys.path:
#     sys.path.insert(0, cmd_folder)
#
# # This makes python look for modules in directory above this one
# mm3_dir = os.path.realpath(os.path.abspath(
#                                  os.path.join(os.path.split(inspect.getfile(
#                                  inspect.currentframe()))[0], '..')))
# if mm3_dir not in sys.path:
#     sys.path.insert(0, mm3_dir)

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
parser = argparse.ArgumentParser(prog='python mm3_track-foci.py',
                                 description='Finds foci.')
parser.add_argument('-f', '--paramfile', type=str,
                    required=True, help='Yaml file containing parameters.')
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
    cell_file_path = namespace.cellfile
else:
    # mm3.warning('No cell file specified. Using complete_cells_mothers.pkl.')
    # cell_file_path = os.path.join(p['cell_dir'], 'complete_cells_mothers.pkl')
    # cell_file_path = os.path.join(p['cell_dir'], 'complete_cells_mothers_foci.pkl')
    cell_file_path = os.path.join(p['cell_dir'], 'complete_cells_test_foci.pkl')
    # cell_file_path = os.path.join(p['cell_dir'], 'complete_cells_test_foci.pkl')

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

# make list of FOVs to process (keys of channel_mask file)
fov_id_list = sorted([fov_id for fov_id in specs.keys()])
### foci analysis

# create dictionary which organizes cells by fov and peak_id
Cells_by_peak = mm3_plots.organize_cells_by_channel(Cells, specs)

color = p['foci']['foci_plane']

ncc = p['foci']['n_cc']

rep_traces = mm3.extract_foci_array(fov_id_list,Cells_by_peak)

### save out the replication traces

## do some pruning of traces athat are obviously erroneous ids

min_trace_length = p['min_c']
max_trace_length = p['max_c']

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

## initialize cell cycle attributes to None
for cell_id, cell in Cells.items():
    cell.initiation_time = None
    cell.init_l = None
    cell.terminination_time = None
    cell.initiation_time_n = None
    cell.init_l_n = None
    cell.C = None
    cell.D = None

## link them to cell divisions and calculate cell cycle parameters. color code division line and trace in gui

for fov, peaks in fTraces.items():
    for peak, Traces in peaks.items():
        for trace_id, trace in Traces.items():
            ## analysis depends on number of overlapping cell cycles
            if ncc == 1:
                # no overlap of cell cycles
                cell_ids = np.unique(trace.cell_ids)
                if len(cell_ids)>1:
                    # print('No overlap set but replication crosses generation')
                    pass

                else:
                    cell = Cells[cell_id]
                    cell.initiation_time = trace.initiation_time
                    init_i = np.where(trace.initiation_time == cell_m.times)
                    cell.init_l = cell.lengths[init_i]/2**(ncc - 1)*p['pxl2um']
                    cell.init_s = cell.volumes[init_i]/2**(ncc - 1)*p['pxl2um']**3
                    cell.termination_time = trace.termination_time
                    cell.C = (cell.termination_time - cell.initiation_time)*p['min_per_frame']
                    cell.D = (cell.division_time - cell.termination_time)*p['min_per_frame']
            elif ncc == 2:
                # initiation occurs in previous generation
                cell_ids = np.unique(trace.cell_ids)
                # print(cell_ids)
                if len(cell_ids) != 2:
                    # print('Found '+str(len(cell_ids)) + ' cells but n_cc is 2')
                    pass
                else:
                    cell_m = Cells[cell_ids[0]]
                    cell_d = Cells[cell_ids[1]]
                    cell_d.initiation_time = trace.initiation_time
                    cell_m.initiation_time_n = trace.initiation_time
                    # print(np.where(trace.initiation_time == cell_m.times))
                    init_i = np.where(trace.initiation_time == cell_m.times)
                    cell_d.init_l = np.array(cell_m.lengths)[init_i]/2**(ncc - 1) * p['pxl2um']
                    cell_m.init_l_n = np.array(cell_m.lengths)[init_i]/2**(ncc - 1)* p['pxl2um']

                    cell_d.init_s = cell_m.volumes[init_i]/2**(ncc - 1)*p['pxl2um']**3
                    cell_m.init_s_n = cell_m.volumes[init_i]/2**(ncc - 1)*p['pxl2um']**3
                    ## should be able to do these in one line
                    cell_d.termination_time = trace.termination_time
                    cell_m.termination_time_n = trace.termination_time
                    try:
                        cell_d.C = (cell_d.termination_time - cell_d.initiation_time)*p['min_per_frame']
                    except TypeError:
                        pass
                    try:
                        cell_d.D = (cell_d.division_time - cell_d.termination_time)*p['min_per_frame']
                    except TypeError:
                        pass
            elif ncc == 3:
                cell_ids = np.unique(trace.cell_ids)
                if len(cell_ids) != 3:
                    print('Found '+str(len(cell_ids)) + ' cells but n_cc is 3')
                    pass
                else:
                    cell_m = Cells[cell_ids[0]]
                    cell_d = Cells[cell_ids[1]]
                    cell_gd = Cells[cell_ids[2]]
                    cell_gd.initiation_time = trace.initiation_time
                    cell_m.initiation_time_n = trace.initiation_time
                    init_i = np.where(trace.initiation_time == cell_m.times)
                    cell_gd.init_l = cell_m.lengths[init_i]/2**(ncc - 1)* p['pxl2um']
                    cell_m.init_l_n = cell_m.lengths[init_i]/2**(ncc - 1)* p['pxl2um']

                    cell_gd.init_s = cell_m.volumes[init_i]/2**(ncc - 1)*p['pxl2um']**3
                    cell_m.init_s_n = cell_m.volumes[init_i]/2**(ncc - 1)*p['pxl2um']**3

                    ## should be able to do these in one line
                    cell_gd.termination_time = trace.termination_time
                    cell_m.termination_time_n = trace.termination_time
                    cell_gd.C = (cell_gd.termination_time - cell_gd.initiation_time) *p['min_per_frame']
                    try:
                        cell_gd.D = (cell_gd.division_time - cell_gd.termination_time)*p['min_per_frame']
                    except TypeError:
                        pass



cell_filename = os.path.basename(cell_file_path)

## save out dictionary of trace objects
with open(os.path.join(p['cell_dir'], p['experiment_name'] + '_rep_traces.pkl'), 'wb') as trace_file:
    pickle.dump(fTraces, trace_file, protocol=pickle.HIGHEST_PROTOCOL)

# save out dictionary of cell objects
with open(os.path.join(p['cell_dir'], cell_filename[:-4] + '_cc.pkl'), 'wb') as cell_file:
    pickle.dump(Cells, cell_file, protocol=pickle.HIGHEST_PROTOCOL)
