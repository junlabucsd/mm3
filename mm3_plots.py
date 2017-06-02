#!/usr/bin/python
from __future__ import print_function

# import modules

# number modules
import numpy as np
import scipy.stats as sps
import pandas as pd
from random import sample

# plotting modules
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks", color_codes=True, font_scale=1.25)


### Data conversion functions ######################################################################
def cells2df(Cells, rescale=False):
    '''
    Take cell data (a dicionary of Cell objects) and return a dataframe.

    rescale : boolean
        If rescale is set to True, then the 6 major parameters are rescaled by their mean.
    '''

    # columns to include
    columns = ['fov', 'peak', 'birth_time', 'division_time', 'birth_label',
               'sb', 'sd', 'delta', 'tau', 'elong_rate', 'septum_position']
    rescale_columns = ['sb', 'sd', 'delta', 'tau', 'elong_rate', 'septum_position']

    # Make dataframe for plotting variables
    Cells_dict = cells2dict(Cells)
    Cells_df = pd.DataFrame(Cells_dict).transpose() # must be transposed so data is in columns
    # Cells_df = Cells_df.sort(columns=['fov', 'peak', 'birth_time', 'birth_label']) # sort for convinience
    Cells_df = Cells_df.sort_values(by=['fov', 'peak', 'birth_time', 'birth_label'])
    Cells_df = Cells_df[columns].apply(pd.to_numeric)

    if rescale:
        for column in rescale_columns:
            Cells_df[column] = Cells_df[column] / Cells_df[column].mean()

    return Cells_df

def cells2dict(Cells):
    '''
    Take a dictionary of Cells and returns a dictionary of dictionaries
    '''

    Cells_dict = {cell_id : vars(cell) for cell_id, cell in Cells.iteritems()}

    return Cells_dict

### Filtering functions ############################################################################
def find_cells_of_birth_label(Cells, label_num=1):
    '''Return only cells whose starting region label is given.
    If no birth_label is given, returns the mother cells.
    label_num can also be a list to include cells of many birth labels
    '''

    fCells = {} # f is for filtered

    if type(label_num) is int:
        label_num = [label_num]

    for cell_id in Cells:
        if Cells[cell_id].birth_label in label_num:
            fCells[cell_id] = Cells[cell_id]

    return fCells

def find_cells_born_before(Cells, born_before=None):
    '''
    Returns Cells dictionary of cells with a birth_time before the value specified
    '''

    if born_before == None:
        return Cells

    fCells = {cell_id : Cell for cell_id, Cell in Cells.iteritems() if Cell.birth_time <= born_before}

    return fCells

def find_cells_born_after(Cells, born_after=None):
    '''
    Returns Cells dictionary of cells with a birth_time after the value specified
    '''

    if born_after == None:
        return Cells

    fCells = {cell_id : Cell for cell_id, Cell in Cells.iteritems() if Cell.birth_time >= born_after}

    return fCells

def organize_cells_by_channel(Cells, specs):
    '''
    Returns a nested dictionary where the keys are first
    the fov_id and then the peak_id (similar to specs),
    and the final value is a dictiary of cell objects that go in that
    specific channel, in the same format as normal {cell_id : Cell, ...}
    '''

    # make a nested dictionary that holds lists of cells for one fov/peak
    Cells_by_peak = {}
    for fov_id in specs.keys():
        Cells_by_peak[fov_id] = {}
        for peak_id, spec in specs[fov_id].items():
            # only make a space for channels that are analyized
            if spec == 1:
                Cells_by_peak[fov_id][peak_id] = {}

    # organize the cells
    for cell_id, Cell in Cells.items():
        Cells_by_peak[Cell.fov][Cell.peak][cell_id] = Cell

    return Cells_by_peak

def filter_by_stat(Cells, center_stat='mean', std_distance=3):
    '''
    Filters a dictionary of Cells by ensuring all of the 6 major parameters are
    within some number of standard deviations away from either the mean or median
    '''

    # Calculate stats.
    Cells_df = cells2df(Cells)
    stats_columns = ['sb', 'sd', 'delta', 'elong_rate', 'tau', 'septum_position']
    cell_stats = Cells_df[stats_columns].describe()

    # set low and high bounds for each stat attribute
    bounds = {}
    for label in stats_columns:
        low_bound = cell_stats[label][center_stat] - std_distance*cell_stats[label]['std']
        high_bound = cell_stats[label][center_stat] + std_distance*cell_stats[label]['std']
        bounds[label] = {'low' : low_bound,
                         'high' : high_bound}

    # add filtered cells to dict
    fCells = {} # dict to hold filtered cells

    for cell_id, Cell in Cells.iteritems():
        benchmark = 0 # this needs to equal 6, so it passes all tests

        for label in stats_columns:
            attribute = getattr(Cell, label) # current value of this attribute for cell
            if attribute > bounds[label]['low'] and attribute < bounds[label]['high']:
                benchmark += 1

        if benchmark == 6:
            fCells[cell_id] = Cells[cell_id]

    return fCells

def find_last_daughter(cell, Cells):
    '''Finds the last daughter in a lineage starting with a earlier cell.
    Helper function for find_continuous_lineages'''

    # go into the daugther cell if the daughter exists
    if cell.daughters[0] in Cells:
        cell = Cells[cell.daughters[0]]
        cell = find_last_daughter(cell, Cells)
    else:
        # otherwise just give back this cell
        return cell

    # finally, return the deepest cell
    return cell

def find_continuous_lineages(Lineages, t1=0, t2=1000):
    '''
    Uses a recursive function to only return cells that have continuous
    lineages between two time points. Takes a "lineage" form of Cells and
    returns a dictionary of the same format. Good for plotting
    with saw_tooth_plot()

    t1 : int
        First cell in lineage must be born before this time point
    t2 : int
        Last cell in lineage must be born after this time point
    '''

    # This is a mirror of the lineages dictionary, just for the continuous cells
    Continuous_Lineages = {}

    for fov, peaks in Lineages.iteritems():
        # Create a dictionary to hold this FOV
        Continuous_Lineages[fov] = {}

        for peak, Cells in peaks.iteritems():
            # sort the cells by time in a list for this peak
            cells_sorted = [(cell_id, cell) for cell_id, cell in Cells.iteritems()]
            cells_sorted = sorted(cells_sorted, key=lambda x: x[1].birth_time)

            # Sometimes there are not any cells for the channel even if it was to be analyzed
            if not cells_sorted:
                continue

            # check if first cell has a birth time below a cutoff
            first_cell = cells_sorted[0][1]
            if first_cell.birth_time < t1:
                # find the last daugher cell
                last_daughter = find_last_daughter(first_cell, Cells)

                # check to make sure it makes the second cut off
                if last_daughter.birth_time > t2:
                    # print(fov, peak, 'Made it')
                    # and add it to the big dictionary if possible
                    Continuous_Lineages[fov][peak] = Cells
        else:
            continue

    return Continuous_Lineages

### Statistics and analysis functions ##############################################################
def stats_table(Cells_df):
    '''Returns a Pandas dataframe with statistics about the 6 major cell parameters.
    '''

    columns = ['sb', 'sd', 'delta', 'tau', 'elong_rate', 'septum_position']
    cell_stats = Cells_df[columns].describe() # This is a nifty function

    # add a CV row
    CVs = [cell_stats[column]['std'] / cell_stats[column]['mean'] for column in columns]
    cell_stats = cell_stats.append(pd.Series(CVs, index=columns, name='CV'))

    # reorder and remove rows
    index_order = ['mean', 'std', 'CV', '50%', 'min', 'max']
    cell_stats = cell_stats.reindex(index_order)

    # rename 50% to median because I hate that name
    cell_stats = cell_stats.rename(index={'50%': 'median'})

    return cell_stats


### Plotting functions #############################################################################
def violin_fovs(Cells_df):
    '''
    Create violin plots of cell stats across FOVs
    '''

    sns.set(style="whitegrid", palette="pastel", color_codes=True, font_scale=1.25)

    columns = ['sb', 'sd', 'delta', 'tau', 'elong_rate', 'septum_position']
    ylabels = ['$\mu$m', '$\mu$m', '$\mu$m', 'min', '$\lambda$', 'daughter/mother']
    titles = ['Length at Birth', 'Length at Division', 'Delta',
              'Generation Time', 'Elongation Rate', 'Septum Position']

    fig, axes = plt.subplots(nrows=len(columns), ncols=1,
                            figsize=[15,2.5*len(columns)], squeeze=False)
    ax = np.ravel(axes)

    for i, column in enumerate(columns):
        # Draw a nested violinplot and split the violins for easier comparison
        sns.violinplot(x="fov", y=column, data=Cells_df,
                      scale="count", inner="quartile", ax=ax[i], lw=1)

        ax[i].set_title(titles[i], size=18)
        ax[i].set_ylabel(ylabels[i], size=16)
        ax[i].set_xlabel('')
        ax[i].tick_params(axis='both', which='major', labelsize=10)

    ax[i].set_xlabel('FOV', size=16)

    # plt.tight_layout()

    # Make title, need a little extra space
    plt.subplots_adjust(top=.925, hspace=0.5)
    fig.suptitle('Cell Parameters Across FOVs', size=20)

    sns.despine()
    # plt.show()

    return fig, ax

def violin_birth_label(Cells_df):
    '''
    Create violin plots of cell stats versus the birth label.

    This is a good way to test if the mother cells are aging, or the cells
    farther down the channel are not being segmented well.
    '''

    sns.set(style="whitegrid", palette="pastel", color_codes=True, font_scale=1.25)

    columns = ['sb', 'sd', 'delta', 'tau', 'elong_rate', 'septum_position']
    ylabels = ['$\mu$m', '$\mu$m', '$\mu$m', 'min', '$\lambda$', 'daughter/mother']
    titles = ['Length at Birth', 'Length at Division', 'Delta',
              'Generation Time', 'Elongation Rate', 'Septum Position']

    fig, axes = plt.subplots(nrows=len(columns), ncols=1,
                            figsize=[10,2.5*len(columns)], squeeze=False)
    ax = np.ravel(axes)

    for i, column in enumerate(columns):
        # Draw a nested violinplot and split the violins for easier comparison
        sns.violinplot(x="birth_label", y=column, data=Cells_df,
                      scale="count", inner="quartile", ax=ax[i], lw=1)

        ax[i].set_title(titles[i], size=18)
        ax[i].set_ylabel(ylabels[i], size=16)
        ax[i].set_xlabel('')
        ax[i].tick_params(axis='both', which='major', labelsize=10)

    ax[i].set_xlabel('Birth Label', size=16)

    # plt.tight_layout()

    # Make title, need a little extra space
    plt.subplots_adjust(top=.925, hspace=0.5)
    fig.suptitle('Cell Parameters vs Birth Label', size=20)

    sns.despine()

    return fig, ax

def hex_time_plot(Cells_df, time_mark='birth_time', x_extents=None, bin_extents=None):
    '''
    Plots cell parameters over time using a hex scatter plot and a moving average
    '''

    sns.set(style="whitegrid", palette="pastel", color_codes=True, font_scale=1.25)

    # lists for plotting and formatting
    columns = ['sb', 'sd', 'delta', 'tau', 'elong_rate', 'septum_position']
    titles = ['Length at Birth', 'Length at Division', 'Delta',
              'Generation Time', 'Elongation Rate', 'Septum Position']
    ylabels = ['$\mu$m', '$\mu$m', '$\mu$m', 'min', '$\lambda$','daughter/mother']

    # create figure, going to apply graphs to each axis sequentially
    fig, axes = plt.subplots(nrows=len(columns)/2, ncols=2,
                            figsize=[15,5*len(columns)/2], squeeze=False)
    ax = np.ravel(axes)

    # binning parameters, should be arguments
    binmin = 5 # minimum bin size to display
    bingrid = (50, 10) # how many bins to have in the x and y directions
    moving_window = 10 # window to calculate moving stat

    # bining parameters for each data type
    # bin_extent in within which bounds should bins go. (left, right, bottom, top)
    if x_extents == None:
        x_extents = (Cells_df['birth_time'].min(), Cells_df['birth_time'].max())

    if bin_extents == None:
        bin_extents = [(x_extents[0], x_extents[1], 0, 6),
                      (x_extents[0], x_extents[1], 0, 12),
                      (x_extents[0], x_extents[1], 0, 6),
                      (x_extents[0], x_extents[1], 0, 120),
                      (x_extents[0], x_extents[1], 0, 0.04),
                      (x_extents[0], x_extents[1], 0, 1)]

    # Now plot the filtered data
    for i, column in enumerate(columns):

        # get out just the data to be plot for one subplot
        time_df = Cells_df[[time_mark, column]].apply(pd.to_numeric)
        time_df.sort_values(by=time_mark, inplace=True)

        # plot the hex scatter plot
        p = ax[i].hexbin(time_df[time_mark], time_df[column],
                         mincnt=binmin, gridsize=bingrid, extent=bin_extents[i])

        # graph moving average
        # xlims = (time_df['birth_time'].min(), time_df['birth_time'].max()) # x lims for bins
        xlims = x_extents
        bin_mean, bin_edges, bin_n = sps.binned_statistic(time_df[time_mark], time_df[column],
                        statistic='mean', bins=np.arange(xlims[0]-1, xlims[1]+1, moving_window))
        bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2
        ax[i].plot(bin_centers, bin_mean, lw=4, alpha=0.8, color=(1.0, 1.0, 0.0))

        # formatting
        ax[i].set_title(titles[i], size=18)
        ax[i].set_ylabel(ylabels[i], size=16)

        p.set_cmap(cmap=plt.cm.Blues) # set color and style

    ax[5].legend(['%s minute binned average' % moving_window], fontsize=14, loc='lower right')
    ax[4].set_xlabel('%s [min]' % time_mark, size=16)
    ax[5].set_xlabel('%s [min]' % time_mark, size=16)

    # Make title, need a little extra space
    plt.subplots_adjust(top=0.925, hspace=0.25)
    fig.suptitle('Cell Parameters Over Time', size=24)

    sns.despine()

    return fig, ax

def derivative_plot(Cells_df, time_mark='birth_time', x_extents=None, time_window=10):
    '''
    Plots the derivtive of the moving average of the cell parameters.

    Parameters
    ----------
    time_window : int
        Time window which the parameters statistic is calculated over
    '''

    sns.set(style="whitegrid", palette="pastel", color_codes=True, font_scale=1.25)

    # lists for plotting and formatting
    columns = ['sb', 'sd', 'delta', 'tau', 'elong_rate', 'septum_position']
    titles = ['Length at Birth', 'Length at Division', 'Delta',
              'Generation Time', 'Elongation Rate', 'Septum Position']
    ylabels = ['$\mu$m', '$\mu$m', '$\mu$m', 'min', '$\lambda$','daughter/mother']

    # create figure, going to apply graphs to each axis sequentially
    fig, axes = plt.subplots(nrows=len(columns)/2, ncols=2,
                            figsize=[15,5*len(columns)/2], squeeze=False)
    ax = np.ravel(axes)

    # over what times should we calculate stats?
    if x_extents == None:
        x_extents = (Cells_df['birth_time'].min(), Cells_df['birth_time'].max())

    # Now plot the filtered data
    for i, column in enumerate(columns):

        # get out just the data to be plot for one subplot
        time_df = Cells_df[[time_mark, column]].apply(pd.to_numeric)
        time_df.sort_values(by=time_mark, inplace=True)

        # graph moving average
        xlims = x_extents
        bin_mean, bin_edges, bin_n = sps.binned_statistic(time_df[time_mark], time_df[column],
                        statistic='mean', bins=np.arange(xlims[0]-1, xlims[1]+1, time_window))
        bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2
        # ax[i].plot(bin_centers, bin_mean, lw=4, alpha=0.8, color=(1.0, 1.0, 0.0))

        mean_diff = np.diff(bin_mean)
        ax[i].plot(bin_centers[:-1], mean_diff, lw=4, alpha=0.8, color='blue')

        # formatting
        ax[i].set_title(titles[i], size=18)
        ax[i].set_ylabel(ylabels[i], size=16)

    ax[5].legend(['%s minute binned average' % time_window], fontsize=14, loc='lower right')
    ax[4].set_xlabel('%s [min]' % time_mark, size=16)
    ax[5].set_xlabel('%s [min]' % time_mark, size=16)

    # Make title, need a little extra space
    plt.subplots_adjust(top=0.925, hspace=0.25)
    fig.suptitle('Cell Parameters Over Time', size=24)

    sns.despine()

    return fig, ax

def plot_traces(Cells, trace_limit=1000):
    '''
    Plot length traces of all cells over time. Makes two plots in the figure, one a
    lineag plot, the second log linear.

    Parameters
    ----------
    trace_limit : int
        Limit the number of traces to this value, chosen randomly from the dictionary Cells.
        Plotting all the traces can be time consuming and mask the trends in the graph.
    '''

    sns.set(style="ticks", palette="pastel", color_codes=True, font_scale=1.25)

    ### Traces #################################################################################
    fig, axes = plt.subplots(ncols=1, nrows=2, figsize=(16, 16))
    ax = axes.flat # same as axes.ravel()

    if trace_limit:
        cell_id_subset = sample(list(Cells), trace_limit)
        Cells = {cell_id : Cells[cell_id] for cell_id in cell_id_subset}

    for cell_id, Cell in Cells.iteritems():

        ax[0].plot(Cell.times_w_div, Cell.lengths_w_div, 'b-', lw=.5, alpha=0.5)
        ax[1].semilogy(Cell.times_w_div, Cell.lengths_w_div, 'b-', lw=.5, alpha=0.5)

    ax[0].set_title('Cell Length vs Time', size=24)
    ax[1].set_xlabel('Time [min]', size=16)
    ax[0].set_ylabel('Length [um]', size=16)
    ax[1].set_ylabel('Log(Length [um])', size=16)

    plt.subplots_adjust(top=0.925, hspace=0.1)

    sns.despine()

    return fig, ax

def saw_tooth_plot(Lineages, FOVs=None, tif_width=2000, mothers=True):
    '''
    Plot individual cell traces, where each FOV gets its own subplot.

    tif_width : int
        Width of the original .tif image in pixels. This is used to color the traces,
        where the color of the line corresponds to the peak position.
    mothers : boolean
        If mothers is True, connecting lines will be drawn between cells in the
        same channel which share a division and birth time. If False, then connecting lines
        will not be drawn.
    '''
    # fig, axes = plt.subplots(ncols=1, nrows=2,
    #                      figsize=(16, 2*3))

    sns.set(style="ticks", palette="pastel", color_codes=True, font_scale=1.25)

    if FOVs == None:
        FOVs = range(1, max(Lineages.keys())+1)

    fig, axes = plt.subplots(ncols=1, nrows=len(FOVs), figsize=(15, 2.5*len(FOVs)), squeeze=False)
    ax = axes.flat

    for i, fov in enumerate(FOVs):
        # record max div length for whole FOV to set y lim
        max_div_length = 0

        for peak, lin in Lineages[fov].iteritems():
            # this is to map mothers to daugthers with lines
            last_div_time = None
            last_length = None

            # turn it into a list so it retains time order
            lin = [(cell_id, cell) for cell_id, cell in lin.iteritems()]
            # sort cells by birth time for the hell of it.
            lin = sorted(lin, key=lambda x: x[1].birth_time)

            peak_color = plt.cm.jet(int(255*peak/tif_width))

            for cell_id, cell in lin:
                ax[i].semilogy(cell.times_w_div, cell.lengths_w_div,
                               color=peak_color, lw=1, alpha=0.75)

                if mothers:
                    # draw a connecting lines betwee mother and daughter
                    if cell.birth_time == last_div_time:
                        ax[i].semilogy([last_div_time, cell.birth_time],
                                       [last_length, cell.sb],
                                       color=peak_color, lw=1, alpha=0.75)

                    # record the last division time and length for next time
                    last_div_time = cell.division_time

                # save the max div length for axis plotting
                last_length = cell.sd
                if last_length > max_div_length:
                    max_div_length = last_length

        title_string = 'FOV %d' % fov
        ax[i].set_title(title_string, size=18)
        ax[i].set_ylabel('Log(Length [um])', size=16)
        ax[i].set_ylim([0, max_div_length + 2])

    ax[-1].set_xlabel('Time [min]', size=16)

    plt.tight_layout()
    # plt.subplots_adjust(top=0.875, bottom=0.1) #, hspace=0.25)
    # fig.suptitle('Cell Length vs Time ', size=24)

    sns.despine()
    # plt.subplots_adjust(hspace=0.5)

    return fig, ax

def plot_distributions(Cells_df):
    '''
    Plot distributions of the 6 major parameters
    '''

    sns.set(style="ticks", palette="pastel", color_codes=True, font_scale=1.25)

    columns = ['sb', 'sd', 'delta', 'tau', 'elong_rate', 'septum_position']
    xlabels = ['$\mu$m', '$\mu$m', '$\mu$m', 'min', '$\lambda$', 'daughter/mother']
    titles = ['Length at Birth', 'Length at Division', 'Delta',
              'Generation Time', 'Elongation Rate', 'Septum Position']
    hist_options = {'histtype' : 'step', 'lw' : 2, 'color' : 'b'}
    kde_options = {'lw' : 2, 'linestyle' : '--', 'color' : 'b'}

    # create figure, going to apply graphs to each axis sequentially
    fig, axes = plt.subplots(nrows=len(columns)/2, ncols=2,
                            figsize=[15,5*len(columns)/2], squeeze=True)
    ax = np.ravel(axes)

    # Plot each distribution
    for i, column in enumerate(columns):
        data = Cells_df[column]

        # get stats for legend
        data_mean = data.mean()
        data_std = data.std()
        data_cv = data_std / data_mean

        # set tau bins to be in 1 minute intervals
        if column == 'tau':
            bin_edges = np.array(range(0, int(data.max())+1, 2)) + 0.5
            sns.distplot(data, ax=ax[i], bins=bin_edges,
                         hist_kws=hist_options, kde_kws=kde_options)

        else:
            sns.distplot(data, ax=ax[i], bins=50,
                         hist_kws=hist_options, kde_kws=kde_options)

        ax[i].set_title(titles[i], size=18)
        ax[i].set_xlabel(xlabels[i], size=16)
        ax[i].get_yaxis().set_ticks([])
        ax[i].set_ylabel('pdf', size=16)
        ax[i].legend(['$\mu$=%.3f, CV=%.2f' % (data_mean, data_cv)], fontsize=14)

    # Make title, need a little extra space
    plt.subplots_adjust(top=0.925, hspace=0.35)
    fig.suptitle('Cell Parameter Distributions', size=24)

    sns.despine()

    return fig, ax

def plot_rescaled_distributions(Cells_df):
    '''
    Plot the 6 major cell distributions with all values normalized by the mean.
    '''

    sns.set(style="ticks", palette="pastel", color_codes=True, font_scale=1.25)

    columns = ['sb', 'sd', 'delta', 'tau', 'elong_rate', 'septum_position']
    # xlabels = ['Rescaled Length at Birth', 'Rescaled Length at Division', 'Rescaled Delta',
    #           'Rescaled Generation Time', 'Rescaled Elongation Rate', 'Rescaled Septum Position']
    xlabels = ['$L_b$ /<$L_b$>', '$L_d$ /<$L_d$>', '$\Delta$ /<$\Delta$>',
               '$\\tau$ /<$\\tau$>', '$\lambda$ /<$\lambda$>',
               '$L_\\frac{1}{2}$ /<$L_\\frac{1}{2}$>']
    hist_options = {'histtype' : 'step', 'lw' : 2, 'color' : 'b'}
    kde_options = {'lw' : 2, 'linestyle' : '--', 'color' : 'b'}

    # create figure, going to apply graphs to each axis sequentially
    fig, axes = plt.subplots(nrows=1, ncols=len(columns),
                             figsize=[15, 5], squeeze=False)
    ax = np.ravel(axes)

    # Plot each distribution
    for i, column in enumerate(columns):
        data = Cells_df[column]

        # get stats for legend and rescaling
        data_mean = data.mean()
        data_std = data.std()
        data_cv = data_std / data_mean

        plot_data = data / data_mean

        # set tau bins to be in 1 minute intervals
        if column == 'tau':
            bin_edges = (np.array(range(0, int(data.max())+1, 1)) + 0.5) / data_mean
            sns.distplot(plot_data, ax=ax[i], bins=bin_edges,
                         hist_kws=hist_options, kde_kws=kde_options)

        else:
            sns.distplot(plot_data, ax=ax[i], bins=50,
                         hist_kws=hist_options, kde_kws=kde_options)

        ax[i].set_xlabel(xlabels[i], size=16)
        ax[i].set_xlim([0.4, 1.6])
        ax[i].get_yaxis().set_visible(False)
        # ax[i].legend(['CV=%.2f' % data_cv], size=12)
        # plt.legend(markerscale=0) # this will remove the line next the label
        ax[i].annotate('CV=%.2f' % data_cv, xy=(0.75,0.85), xycoords='axes fraction', size=12)

        for t in ax[i].get_xticklabels():
            t.set(rotation=45)

    # Make title, need a little extra space
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, hspace=0.35, left=None, right=None)
    fig.suptitle('Rescaled Cell Parameter Distributions', size=24)

    sns.despine(left=True)

    return fig, ax

def plot_correlations(Cells_df, rescale=False):
    '''
    Plot correlations of each major cell parameter against one another

    rescale : boolean
        If rescale is set to True, then axis labeling reflects rescaled data.
    '''

    sns.set(style="ticks", palette="pastel", color_codes=True, font_scale=1.25)

    columns = ['sb', 'sd', 'delta', 'tau', 'elong_rate', 'septum_position']
    titles = ['Length at Birth', 'Length at Division', 'Delta',
                  'Generation Time', 'Elongation Rate', 'Septum Position']
    labels = ['$\mu$m', '$\mu$m', '$\mu$m', 'min', '$\lambda$', 'daughter/mother']
    rlabels = ['$L_b$ /<$L_b$>', '$L_d$ /<$L_d$>', '$\Delta$ /<$\Delta$>',
               '$\\tau$ /<$\\tau$>', '$\lambda$ /<$\lambda$>',
               '$L_\\frac{1}{2}$ /<$L_\\frac{1}{2}$>']

    # It's just one function from seaborn
    g = sns.pairplot(Cells_df[columns], kind="reg", diag_kind="kde",
                     plot_kws={'scatter':True,
                               'x_bins':10,
                               'scatter_kws':{'alpha':0.25}})

    # Make title, need a little extra space
    plt.subplots_adjust(top=0.95, left=0.075, right=0.95)
    g.fig.suptitle('Correlations and Distributions', size=24)

    for i, ax in enumerate(g.axes.flatten()):

        if not rescale:
            if i <= 5:
                ax.set_title(titles[i], size=16)
            if i % 6 == 0:
                ax.set_ylabel(titles[i / 6], size=16)
            if i >= 30:
                ax.set_xlabel(labels[i - 30], size=16)

        if rescale:
            ax.set_ylim([0.4, 1.6])
            ax.set_xlim([0.4, 1.6])

            # if i <= 5:
            #     ax.set_title(titles[i], size=16)
            if i % 6 == 0:
                ax.set_ylabel(rlabels[i / 6], size=16)
            if i >= 30:
                ax.set_xlabel(rlabels[i - 30], size=16)

        for t in ax.get_xticklabels():
            t.set(rotation=45)

    return g
