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
parser = argparse.ArgumentParser(prog='python mm3_nf.py',
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
fov_id_list = range(1,10,1)
### foci analysis

# create dictionary which organizes cells by fov and peak_id
Cells_by_peak = mm3_plots.organize_cells_by_channel(Cells, specs)

color = p['foci']['foci_plane']

rep_traces = mm3.extract_foci_array(fov_id_list,Cells_by_peak)

for fov, peaks in six.iteritems(Cells_by_peak):
    for peak, Cells in six.iteritems(peaks):
        fig=plt.figure()
        ax=plt.axes()
        fig.set_facecolor('black')
        ax.set_facecolor('black')
        for trace_id,trace in rep_traces[fov][peak].items():
            y_pos = [p[1] for p in trace.positions]
            if len(y_pos)>3:
                ax.plot(trace.times,y_pos,marker='o',ls='-',ms=2)
            else:
                ax.plot(trace.times,y_pos,marker='o',color='lightgray',ls='',ms=2)
        plt.show()


### save out the replication traces

# cell_filename = os.path.basename(cell_file_path)
# with open(os.path.join(p['cell_dir'], cell_filename[:-4] + '_foci_mod.pkl'), 'wb') as cell_file:
#     pickle.dump(Cells, cell_file, protocol=pickle.HIGHEST_PROTOCOL)
