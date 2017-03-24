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
    mm3.information ('Loading experiment parameters.')
    p = mm3.init_mm3_helpers(param_file_path) # loads and returns

    # load specs file
    with open(p['ana_dir'] + '/specs.pkl', 'r') as specs_file:
        specs = pickle.load(specs_file)

    # load cell data dict.
    with open(p['cell_dir'] + 'complete_cells.pkl', 'r') as cell_file:
        Complete_Cells = pickle.load(cell_file)

    ### Filters you may want to apply
    # Just get the mother cells, as most people only care about those.
    Cells = mm3.find_mother_cells(Complete_Cells)

    # before and after a certain time
    # Cells = {cell_id : Cell for cell_id, Cell in Cells.iteritems() if Cell.birth_time <=200}
    # Cells = {cell_id : Cell for cell_id, Cell in Cells.iteritems() if Cell.division_time <=250}

    ### From here, change flags to True for different data transformations that you want
    # Save complete cells into a dictionary of dictionaries.
    if True:
        mm3.information('Saving dictionary of cells.')
        # Or just mothers
        Cells_dict = {cell_id : vars(cell) for cell_id, cell in Cells.iteritems()}

        # save pickle version.
        with open(p['cell_dir'] + '/cells_dict.pkl', 'wb') as cell_file:
            pickle.dump(Cells_dict, cell_file)

        # The text file version of the dictionary is good for easy glancing
        with open(p['cell_dir'] + '/cells_dict.txt', 'w') as cell_file:
            pprint(Cells_dict, stream=cell_file)

    # All cells and mother cells saved to a matlab file.
    if True:
        mm3.information('Saving .mat file of cells.')

        with open(p['cell_dir'] + '/cells.mat', 'wb') as cell_file:
            savemat(cell_file, Cells)

    # Save a big .csv of all the cell data (JT's format)
    if True:
        mm3.information('Saving .csv table of cells.')

        Cells_dict = {cell_id : vars(cell) for cell_id, cell in Cells.iteritems()}

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

        Cells_df.to_csv(p['cell_dir'] + 'cells.csv', sep=',', float_format='%.4f',
                        header=True, index=False)

    # Save csv in Sattar's format for Igor plotting
    if True:
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
        # regather cells in case previous operations manipulated it
        Cells = mm3.find_mother_cells(Complete_Cells)

        # make a directory to hold these csvs
        plot_dir = p['cell_dir'] + 'plots/'
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        # import plotting modules
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        font = {'family' : 'normal',
                'weight' : 'normal',
                'size'   : 18}
        mpl.rc('font', **font)
        mpl.rcParams['figure.figsize'] = 10, 10
        mpl.rcParams['pdf.fonttype'] = 42

        import matplotlib.pyplot as plt
        # import matplotlib.patches as mpatches

        import seaborn as sns

        sns.set(style="ticks", color_codes=True, font_scale=1.25)

        ### Traces
        fig, axes = plt.subplots(ncols=1, nrows=2, figsize=(16, 16))
        ax = axes.flat # same as axes.ravel()

        for cell_id, cell in Cells.iteritems():

            ax[0].plot(cell.times_w_div, cell.lengths_w_div, 'b-', lw=.5, alpha=0.5)
            ax[1].semilogy(cell.times_w_div, cell.lengths_w_div, 'b-', lw=.5, alpha=0.5)

        ax[0].set_title('Cell Length vs Time', size=24)
        ax[1].set_xlabel('Time [min]', size=20)
        ax[0].set_ylabel('Length [um]', size=20)
        ax[1].set_ylabel('Log(Length [um])', size=20)
        ax[0].set_ylim([0,10])
        ax[1].set_ylim([0,10])

        sns.despine()
        #plt.save
        plt.savefig(plot_dir + 'traces.png')

        ### Distributions of filtered parameters
        labels = ['sb', 'sd', 'delta', 'tau', 'elong_rate', 'septum_position']
        xlabels = ['$\mu$m', '$\mu$m', '$\mu$m', 'min', '$\lambda$', 'daughter/mother']

        fig, axes = plt.subplots(nrows=len(labels), ncols=1, figsize=[15,30], squeeze=False)
        ax = np.ravel(axes)

        # Make dataframe for plotting variables
        Cells_dict = {cell_id : vars(cell) for cell_id, cell in Cells.iteritems()}
        Cells_df = pd.DataFrame(Cells_dict).transpose() # must be transposed so data is in columns
        Cells_df = Cells_df.sort(columns=['fov', 'peak', 'birth_time', 'birth_label']) # sort for convinience
        plot_columns = ['sb', 'sd', 'delta', 'elong_rate', 'tau', 'septum_position']
        plot_df = Cells_df[plot_columns].astype(np.float)

        # Filter the data in plot_df to cells which fall into 3 standard deviations of the mean for all
        # the six parameters above
        filters = [] # list of logical indicies

        for i, label in enumerate(labels):
            current_data = plot_df[label]

            # find mean and standard deviation
            data_mean = current_data.mean()
            data_std = current_data.std()
            data_cv = data_std / data_mean
            #print('%s, mean=%.3f, std=%.3f, cv=%.3f, n=%d' % (label, data_mean, data_std, data_cv, len(current_data)))

            # constrict range to the mean plus-minus 3std on either side
            data_range = [data_mean - 3*data_std, data_mean + 3*data_std]

            # filter the data from the smaller range and repeat
            filters.append(np.logical_and(current_data > data_range[0], current_data < data_range[1]))
            #print('Cells after filtering = %d' % sum(filters[i]))

        compiled_filter = np.logical_and(filters[0], filters[1])
        for i in range(2,len(labels)):
            compiled_filter = np.logical_and(compiled_filter, filters[i])
        #     print(sum(compiled_filter))
        filtered_df = plot_df[compiled_filter]
        #print('There are %d filtered cells' % len(filtered_df))
        #print('\n')

        # Now plot the filtered data
        for i, label in enumerate(labels):
            filtered_data = filtered_df[label]

            # and the new mean and std
            fil_data_mean = filtered_data.mean()
            fil_data_std = filtered_data.std()
            fil_data_cv = fil_data_std / fil_data_mean
            #print('%s, mean=%.3f, std=%.3f, cv=%.3f' % (label, fil_data_mean, fil_data_std, fil_data_cv))

            fil_data_range = [fil_data_mean - 3*fil_data_std, fil_data_mean + 3*fil_data_std]

            ax[i].hist(filtered_data, bins=22, histtype='step', range=fil_data_range, normed=True, lw=3, alpha=0.75)

            ax[i].set_title(label, size=20)
            ax[i].set_xlabel(xlabels[i], size=20)
            ax[i].get_yaxis().set_ticks([])
            ax[i].set_ylabel('pdf', size=20)

            ax[i].legend(['$\mu$=%.3f, CV=%.2f' % (fil_data_mean, fil_data_cv)], fontsize=14)

        plt.tight_layout()
        sns.despine()
        # plt.show()
        plt.savefig(plot_dir + 'distributions.png')

        ### correlations
        g = sns.pairplot(filtered_df, kind="reg")
        # g.map_offdiag(kind='hex')

        # Make title, need a little extra space
        plt.subplots_adjust(top=0.94)
        g.fig.suptitle('Correlations and Distributions', size=24)

        for ax in g.axes.flatten():
            for t in ax.get_xticklabels():
                t.set(rotation=45)

        plt.savefig(plot_dir + 'correlations.png')

        ### parameters over time
        labels = ['sb', 'sd', 'delta', 'tau', 'elong_rate', 'septum_position']
        ylabels = ['$\mu$m', '$\mu$m', '$\mu$m', 'min', '$\lambda$', 'daughter/mother']

        fig, axes = plt.subplots(nrows=len(labels), ncols=1, figsize=[15,30], squeeze=False)
        ax = np.ravel(axes)

        # Now plot the filtered data
        for i, label in enumerate(labels):

            columns = ['birth_time', label]
            time_df = Cells_df[columns]
            time_df = time_df[columns].apply(pd.to_numeric)
            time_df = time_df.reset_index()
            time_df = time_df[columns]
            time_df.sort_values(by='birth_time', inplace=True)

        #     filtered_data = filtered_df[label]

            ax[i].scatter(time_df['birth_time'], time_df[label])

            ax[i].set_title(label, size=20)
        #     ax[i].get_yaxis().set_ticks([])
            ax[i].set_ylabel(ylabels[i], size=20)
            ax[i].set_xlim([0,200])

        ax[i].set_xlabel('birth time [min]', size=20)

        # plt.tight_layout()

        # Make title, need a little extra space
        plt.subplots_adjust(top=0.95)
        fig.suptitle('Cell Parameters Over Time', size=24)

        sns.despine()
        # plt.show()
        plt.savefig(plot_dir + 'time_plots.png')
