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

# This makes python look for modules in directory above this one
mm3_dir = os.path.realpath(os.path.abspath(
                                 os.path.join(os.path.split(inspect.getfile(
                                 inspect.currentframe()))[0], '../..')))
if mm3_dir not in sys.path:
    sys.path.insert(0, mm3_dir)

import mm3_helpers as mm3
import mm3_plots as mm3_plots
import mm3_GUI_helpers as GUI

# set switches and parameters
parser = argparse.ArgumentParser(prog='python mm3_OutputCC.py',
                                 description='Calculate cell cycle parameters')
parser.add_argument('-f', '--paramfile', type=str,
                    required=True, help='Yaml file containing parameters.')
parser.add_argument('-c', '--cellfile', type=str,
                    required=False, help='Path to Cell object dicionary to analyze. Defaults to complete_cells_foci.pkl.')
parser.add_argument('-t', '--tracefile', type=str,
                    required=False, help='Path to trace object dicionary to analyze. Defaults to rep_traces.pkl.')
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
    cell_file_path = os.path.join(p['cell_dir'], namespace.cellfile)
else:
    cell_file_path = os.path.join(p['cell_dir'], 'complete_cells_foci.pkl')

with open(cell_file_path, 'rb') as cell_file:
    Cells = pickle.load(cell_file,encoding='latin1')

if namespace.tracefile:
    trace_file_path = namespace.tracefile
else:
    trace_file_path = os.path.join(p['cell_dir'], 'rep_traces_mod.pkl')

with open(trace_file_path, 'rb') as tf:
    Traces = pickle.load(tf, encoding='latin1')

# load specs file
specs = mm3.load_specs()

color = p['foci']['foci_plane']

ncc = p['foci']['n_cc']

## initialize cell cycle attributes to None
for cell_id, cell in Cells.items():
    cell.initiation_time = None
    cell.init_l = None
    cell.init_s = None
    cell.terminination_time = None
    cell.initiation_time_n = None
    cell.init_l_n = None
    cell.init_s_n = None
    cell.C = None
    cell.D = None
    cell.ncc = None

## link them to cell divisions and calculate cell cycle parameters. color code division line and trace in gui
if ncc == None:
    print('Number of overlapping cell cycles not provided \nInferring from focus tracks')

for fov, peaks in Traces.items():
    for peak, pTraces in peaks.items():
        for trace_id, trace in pTraces.items():
            if ncc == None:
                ## infer number of overlapping cell cycles
                cell_ids = np.unique(trace.cell_ids)
                if len(cell_ids)==1:
                    try:
                        cell = Cells[cell_id]
                    except KeyError:
                        continue
                    init_i = np.squeeze(np.where(cell.times == trace.initiation_time))
                    term_i = np.squeeze(np.where(cell.times == trace.termination_time))
                    if init_i.size ==0 or term_i.size==0:
                        continue
                    cell.initiation_time = trace.initiation_time
                    # length at initiation per origin in uM
                    cell.init_l = np.array(cell.lengths)[init_i]*p['pxl2um']
                    # volume at initiation per origin in uM^3
                    cell.init_s = np.array(cell.volumes)[init_i]*p['pxl2um']**3
                    cell.termination_time = trace.termination_time
                    # B period in minutes
                    cell.B = (cell.initiation_time - cell.birth_time)*p['min_per_frame']
                    # C period in minutes
                    cell.C = (cell.termination_time - cell.initiation_time)*p['min_per_frame']
                    # D period in minutes
                    cell.D = (cell.division_time - cell.termination_time)*p['min_per_frame']
                    cell.ncc = 1

                elif len(cell_ids)==2:
                    try:
                        cell_m = Cells[cell_ids[0]]
                        cell_d = Cells[cell_ids[1]]
                    except KeyError:
                        continue

                    init_i = np.squeeze(np.nonzero(cell_m.times == trace.initiation_time))
                    term_i = np.squeeze(np.where(cell_d.times == trace.termination_time))

                    if init_i.size==0 or term_i.size==0:
                        continue
                    cell_d.initiation_time = trace.initiation_time
                    cell_m.initiation_time_n = trace.initiation_time

                    cell_d.init_l = np.array(cell_m.lengths)[init_i]/2 * p['pxl2um']
                    cell_m.init_l_n = np.array(cell_m.lengths)[init_i]/2* p['pxl2um']

                    cell_d.init_s = np.array(cell_m.volumes)[init_i]/2*p['pxl2um']**3
                    cell_m.init_s_n = np.array(cell_m.volumes)[init_i]/2*p['pxl2um']**3
                    cell_d.termination_time = trace.termination_time
                    cell_m.termination_time_n = trace.termination_time
                    cell_d.C = (cell_d.termination_time - cell_d.initiation_time)*p['min_per_frame']
                    try:
                        cell_d.D = (cell_d.division_time - cell_d.termination_time)*p['min_per_frame']
                    except TypeError:
                        pass
                    cell.ncc = 2

                elif len(cell_ids)==3:
                    try:
                        cell_m = Cells[cell_ids[0]]
                        cell_d = Cells[cell_ids[1]]
                        cell_gd = Cells[cell_ids[2]]
                    except KeyError:
                        continue

                    init_i = np.squeeze(np.where(cell_m.times == trace.initiation_time))
                    term_i = np.squeeze(np.where(cell_gd.times == trace.termination_time))

                    if init_i.size==0 or term_i.size==0:
                        continue
                    cell_gd.initiation_time = trace.initiation_time
                    cell_m.initiation_time_n = trace.initiation_time
                    init_i = np.squeeze(np.where(cell_m.times == trace.initiation_time))
                    cell_gd.init_l = np.array(cell_m.lengths)[init_i]/4* p['pxl2um']
                    cell_m.init_l_n = np.array(cell_m.lengths)[init_i]/4* p['pxl2um']

                    cell_gd.init_s = np.array(cell_m.volumes)[init_i]/4*p['pxl2um']**3
                    cell_m.init_s_n = np.array(cell_m.volumes)[init_i]/4*p['pxl2um']**3

                    cell_gd.termination_time = trace.termination_time
                    cell_m.termination_time_n = trace.termination_time
                    cell_gd.C = (cell_gd.termination_time - cell_gd.initiation_time) *p['min_per_frame']
                    try:
                        cell_gd.D = (cell_gd.division_time - cell_gd.termination_time)*p['min_per_frame']
                    except TypeError:
                        pass
                    cell.ncc = 3

            ## analysis depends on number of overlapping cell cycles
            if ncc == 1:
                # no overlap of cell cycles
                cell_ids = np.unique(trace.cell_ids)
                if len(cell_ids)>1:
                    # print('No overlap set but replication crosses generation')
                    pass

                else:
                    try:
                        cell = Cells[cell_id]
                    except KeyError:
                        continue
                    cell.initiation_time = trace.initiation_time
                    init_i = np.squeeze(np.where(cell.times == trace.initiation_time))
                    term_i = np.squeeze(np.where(cell.times == trace.termination_time))
                    if init_i.size ==0 or term_i.size==0:
                        continue
                    # length at initiation per origin in uM
                    cell.init_l = np.array(cell.lengths)[init_i]/2**(ncc - 1)*p['pxl2um']
                    # volume at initiation per origin in uM^3
                    cell.init_s = np.array(cell.volumes)[init_i]/2**(ncc - 1)*p['pxl2um']**3
                    cell.termination_time = trace.termination_time
                    # B period in minutes
                    cell.B = (cell.initiation_time - cell.birth_time)*p['min_per_frame']
                    # C period in minutes
                    cell.C = (cell.termination_time - cell.initiation_time)*p['min_per_frame']
                    # D period in minutes
                    cell.D = (cell.division_time - cell.termination_time)*p['min_per_frame']
            elif ncc == 2:
                # initiation occurs in previous generation
                cell_ids = np.unique(trace.cell_ids)
                # print(cell_ids)
                if len(cell_ids) != 2:
                    # print('Found '+str(len(cell_ids)) + ' cells but n_cc is 2')
                    pass
                else:
                    try:
                        cell_m = Cells[cell_ids[0]]
                        cell_d = Cells[cell_ids[1]]
                    except KeyError:
                        continue

                    init_i = np.squeeze(np.nonzero(cell_m.times == trace.initiation_time))
                    term_i = np.squeeze(np.where(cell_d.times == trace.termination_time))

                    if init_i.size ==0 or term_i.size==0:
                        continue

                    cell_d.initiation_time = trace.initiation_time
                    cell_m.initiation_time_n = trace.initiation_time

                    cell_d.init_l = np.array(cell_m.lengths)[init_i]/2**(ncc - 1) * p['pxl2um']
                    cell_m.init_l_n = np.array(cell_m.lengths)[init_i]/2**(ncc - 1)* p['pxl2um']

                    cell_d.init_s = np.array(cell_m.volumes)[init_i]/2**(ncc - 1)*p['pxl2um']**3
                    cell_m.init_s_n = np.array(cell_m.volumes)[init_i]/2**(ncc - 1)*p['pxl2um']**3
                    cell_d.termination_time = trace.termination_time
                    cell_m.termination_time_n = trace.termination_time
                    cell_d.C = (cell_d.termination_time - cell_d.initiation_time)*p['min_per_frame']

                    try:
                        cell_d.D = (cell_d.division_time - cell_d.termination_time)*p['min_per_frame']
                    except TypeError:
                        pass
            elif ncc == 3:
                cell_ids = np.unique(trace.cell_ids)
                if len(cell_ids) != 3:
                    # print('Found '+str(len(cell_ids)) + ' cells but n_cc is 3')
                    pass
                else:
                    ## now we have mother / daughter / granddaughter
                    try:
                        cell_m = Cells[cell_ids[0]]
                        cell_d = Cells[cell_ids[1]]
                        cell_gd = Cells[cell_ids[2]]
                    except KeyError:
                        continue
                    init_i = np.squeeze(np.where(cell_m.times == trace.initiation_time))
                    term_i = np.squeeze(np.where(cell_gd.times == trace.termination_time))
                    if init_i.size ==0 or term_i.size==0:
                        continue

                    cell_gd.initiation_time = trace.initiation_time
                    cell_m.initiation_time_n = trace.initiation_time
                    cell_gd.init_l = np.array(cell_m.lengths)[init_i]/2**(ncc - 1)* p['pxl2um']
                    cell_m.init_l_n = np.array(cell_m.lengths)[init_i]/2**(ncc - 1)* p['pxl2um']

                    cell_gd.init_s = np.array(cell_m.volumes)[init_i]/2**(ncc - 1)*p['pxl2um']**3
                    cell_m.init_s_n = np.array(cell_m.volumes)[init_i]/2**(ncc - 1)*p['pxl2um']**3

                    cell_gd.termination_time = trace.termination_time
                    cell_m.termination_time_n = trace.termination_time
                    cell_gd.C = (cell_gd.termination_time - cell_gd.initiation_time) *p['min_per_frame']
                    try:
                        cell_gd.D = (cell_gd.division_time - cell_gd.termination_time)*p['min_per_frame']
                    except TypeError:
                        pass

cell_filename = os.path.basename(cell_file_path)

# save out dictionary of cell objects
with open(os.path.join(p['cell_dir'], cell_filename[:-4] + '_cc.pkl'), 'wb') as cell_file:
    pickle.dump(Cells, cell_file, protocol=pickle.HIGHEST_PROTOCOL)
print('Finished analysis, saving updated cellfile')
