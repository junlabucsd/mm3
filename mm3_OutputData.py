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
import mm3_plots

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
    mm3.information ('Loading experiment parameters and cell data.')
    p = mm3.init_mm3_helpers(param_file_path) # loads and returns

    # load specs file
    with open(p['ana_dir'] + '/specs.pkl', 'r') as specs_file:
        specs = pickle.load(specs_file)

    # load cell data dict.
    with open(p['cell_dir'] + 'complete_cells.pkl', 'r') as cell_file:
        Cells = pickle.load(cell_file)

    ### Filters you may want to apply
    if True:
        # Filter based on birth_label. mother cells : label_num=1
        Cells = mm3_plots.find_cells_of_birth_label(Cells, label_num=[1,2,3])

        # Filter based on birth time point
        # Cells = mm3_plots.find_cells_born_before(Cells, born_before=1000)

        # Filter based on statistics
        Cells = mm3_plots.filter_by_stat(Cells, center_stat='mean', std_distance=3)

        # save cell pickle of filtered cells.
        if False:
            with open(p['cell_dir'] + 'complete_cells_filtered.pkl', 'wb') as cell_file:
                pickle.dump(Cells, cell_file, protocol=pickle.HIGHEST_PROTOCOL)

    ### From here, change flags to True for different data transformations that you want
    # Save complete cells into a dictionary of dictionaries.
    if False:
        mm3.information('Saving dictionary of cells.')
        Cells_dict = mm3_plots.cells2dict(Cells)

        if False:
            # save pickle version.
            with open(p['cell_dir'] + '/cells_dict.pkl', 'wb') as cell_file:
                pickle.dump(Cells_dict, cell_file, protocol=pickle.HIGHEST_PROTOCOL)

        if False:
            # The text file version of the dictionary is good for easy glancing
            with open(p['cell_dir'] + '/cells_dict.txt', 'w') as cell_file:
                pprint(Cells_dict, stream=cell_file)

    # Saved to a matlab file.
    if False:
        mm3.information('Saving .mat file of cells.')

        with open(p['cell_dir'] + '/complete_cells.mat', 'wb') as cell_file:
            savemat(cell_file, Cells)

    # Save a big .csv of all the cell data (JT's format)
    if False:
        mm3.information('Saving .csv table of cells.')

        Cells_dict = mm3_plots.cells2dict(Cells)

        # pandas dataframe wants to be converted from a dict of dicts.
        Cells_df = pd.DataFrame(Cells_dict).transpose() # columns as data types
        # organize the order of the rows
        Cells_df = Cells_df.sort(columns=['fov', 'peak', 'birth_time', 'birth_label'])
        # in future releases this should be:
        # Cells_df = Cells_df.sort_values(by=['fov', 'peak', 'birth_time', 'birth_label'])

        # deal with the daughters, put them in two columns
        Cells_df['daughter1'], Cells_df['daughter2'] = Cells_df['daughters'].astype(str).str.split(' ', 1).str
        Cells_df['daughter1'] = Cells_df['daughter1'].str.split("'").str[1]
        Cells_df['daughter2'] = Cells_df['daughter2'].str.split("'").str[1]

        # decide the order of the columns
        include_columns = ['id', 'fov', 'peak', 'birth_label',
                           'birth_time', 'division_time',
                           'sb', 'sd', 'delta', 'elong_rate', 'tau', 'septum_position',
                           'parent', 'daughter1', 'daughter2']
                           #'times', 'lengths', 'widths', 'areas',

        Cells_df = Cells_df[include_columns]
        # convert some columns to numeric type for better formatting
        float_columns = ['sb', 'sd', 'delta', 'elong_rate', 'tau', 'septum_position']
        Cells_df[float_columns] = Cells_df[float_columns].astype(np.float)

        Cells_df.to_csv(p['cell_dir'] + 'complete_cells.csv', sep=',', float_format='%.4f',
                        header=True, index=False)

    # Save csv in Sattar's format for Igor plotting
    if False:
        mm3.information('Saving Igor style .txt files of cells.')
        # function which recursivly adds data from a lineage
        def add_lineage_data(Cells, cell_id, cells_df, lineage_df, channel_id=None, cell_age=0):
            Cell = Cells[cell_id] # this is our cell, yay!
            # add the whole cell data to the big table
            # 'FOV_ID', 'channel_ID', 'elongation_rate', 'gen_time', 's_b', 's_d', 'delta'
            # convert this data into a temporary dataframe for appending purposes.
            current_df = pd.DataFrame([[Cell.fov, channel_id, Cell.elong_rate, Cell.tau,
                                        Cell.sb, Cell.sd, Cell.delta, cell_age, Cell.birth_time]],
                                        columns=['FOV_ID', 'channel_ID', 'elongation_rate',
                                                 'gen_time', 's_b', 's_d', 'delta',
                                                 'replicative_age', 'wall_time'])
            cells_df = cells_df.append(current_df) # add it as a new row to the dataframe.

            # Now add information from every timepoint in this cell to the lineage datframe
            cur_lineage_data = []
            for i, t in enumerate(Cell.times):
                # set the birth and division flags.
                birth = 1 if i == 0 else 0
                div = 1 if i == len(Cell.times)-1 else 0
                cur_lineage_data.append([cell_age, t, Cell.lengths[i], div, birth])

            cur_lineage_df = pd.DataFrame(cur_lineage_data, columns=['replicative_age',
                                          'lab_time_mins', 'cell_length',
                                          'division_yes_no', 'birth_yes_no'])
            lineage_df = lineage_df.append(cur_lineage_df)

            # get the daughter which may have kid (always first daughter in list)
            # if this child has a kid, recurse
            if Cell.daughters[0] in Cells:
                cell_age += 1
                cells_df, lineage_df = add_lineage_data(Cells, Cell.daughters[0], cells_df,
                                                        lineage_df, channel_id, cell_age)

            # otherwise back out with the data
            return cells_df, lineage_df

        # make a directory to hold these csvs
        igor_dir = p['cell_dir'] + 'igor_csvs/'
        if not os.path.exists(igor_dir):
            os.makedirs(igor_dir)

        # Make a dataframe that will hold data for all cells
        cells_df = pd.DataFrame(columns=['FOV_ID', 'channel_ID', 'elongation_rate',
                                'gen_time', 's_b', 's_d', 'delta', 'replicative_age'])

        # organize these cells first
        Cells_by_peak = mm3.organize_cells_by_channel(Cells, specs)

        # each channel is processed individually
        for fov_id, peaks in Cells_by_peak.items():
            # channel id start. 10 == 1. Should add 10 floor to tens place each new channel
            channel_id = 10
            for peak_id, Cells in sorted(peaks.items()):

                # Go through cells and find root cells
                root_cells = []

                for cell_id, Cell in Cells.items():
                    # see if the parent is in the list of all cells
                    if Cell.parent not in Cells.keys():
                        root_cells.append(cell_id) # if so it is a root cell

                # sort the root cells so we process them in time order
                root_cells = sorted(root_cells)

                # go through each root cell to collect data
                for cell_id in root_cells:
                    # Initialize dataframe for this root cell (lineage)
                    lineage_df = pd.DataFrame(columns=['replicative_age', 'lab_time_mins',
                                                       'cell_length',
                                                       'division_yes_no', 'birth_yes_no'])

                    # give this peak a channel id, which is not the real channel id,
                    # but a marker for lineage number
                    # the tens digit is the channel number, the ones position
                    # indicates orphan lineages.
                    channel_id += 1

                    # add the data for this lineage
                    cells_df, lineage_df = add_lineage_data(Cells_by_peak[fov_id][peak_id],
                                        cell_id, cells_df, lineage_df, channel_id, cell_age=0)

                    # convert linage_df for saving
                    float_columns = ['lab_time_mins', 'cell_length']
                    int_columns = ['replicative_age', 'division_yes_no', 'birth_yes_no']
                    lineage_df[float_columns] = lineage_df[float_columns].astype(np.float)
                    lineage_df[int_columns] = lineage_df[int_columns].astype(np.int)

                    # save it
                    lineage_filepath = igor_dir + p['experiment_name'] + '_FOV%02d_Ch%03d.txt' % (fov_id, channel_id)
                    lineage_df.to_csv(lineage_filepath, sep='\t', float_format='%.4f',
                                        header=True, index=False)
                # update channel id
                channel_id = int(channel_id / 10) * 10 + 10 # go to the next 10s

        # convert cells_df for saving
        float_columns = ['elongation_rate', 'gen_time', 's_b', 's_d', 'delta']
        int_columns = ['FOV_ID', 'channel_ID', 'replicative_age']
        cells_df[float_columns] = cells_df[float_columns].astype(np.float)
        cells_df[int_columns] = cells_df[int_columns].astype(np.int)

        # save em
        cells_df_filepath = igor_dir + p['experiment_name'] + '_data.txt'
        cells_df.to_csv(cells_df_filepath, sep='\t', float_format='%.4f',
                            header=True, index=False)

    # Some plotting things
    if True:
        # This shall use the cells as filtered above.

        # make a directory to hold these csvs
        plot_dir = p['cell_dir'] + 'plots/'
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        # Data frame is convinient for plotting.
        Cells_df = mm3_plots.cells2df(Cells)

        mm3.information('Plotting violin plots by FOV.')
        fig, ax = mm3_plots.violin_fovs(Cells_df)
        fig.savefig(plot_dir + 'cell_parameters_by_fov.png', dpi=100)

        mm3.information('Plotting stats by birth label.')
        fig, ax = mm3_plots.violin_birth_label(Cells_df)
        fig.savefig(plot_dir + 'cell_parameters_by_birth_label.png', dpi=100)

        mm3.information('Plotting stats over time.')
        fig, ax = mm3_plots.hex_time_plot(Cells_df, time_mark='birth_time')
        fig.savefig(plot_dir + 'cell_parameters_over_time.png', dpi=100)

        mm3.information('Plotting traces over time.')
        fig, ax = mm3_plots.plot_traces(Cells)
        fig.savefig('./plots/traces.png', dpi=100)

        mm3.information('Plotting parameter distributions.')
        fig, ax = mm3_plots.plot_distributions(Cells_df)
        fig.savefig(plot_dir + 'distributions.png', dpi=100)

        mm3.information('Plotting rescaled parameter ditributions.')
        fig, ax = mm3_plots.plot_rescaled_distributions(Cells_df)
        fig.savefig(plot_dir + 'rescaled_distributions.png', dpi=100)

        mm3.information('Plotting rescaled parameter correlations.')
        Cells_df_r = mm3_plots.cells2df(Cells, rescale=True)
        g = mm3_plots.plot_correlations(Cells_df_r, rescale=True)
        g.fig.savefig(plot_dir + 'correlations_rescaled.png', dpi=100)
