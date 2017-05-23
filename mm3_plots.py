#!/usr/bin/python
from __future__ import print_function

# import modules

# number modules
import numpy as np
import scipy.stats as sps
import pandas as pd

# plotting modules
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks", color_codes=True, font_scale=1.25)

def cells2df(Cells):
    '''
    Take cell data (a dicionary of Cell objects) and return a dataframe.
    '''

    # columns to include
    columns = ['fov', 'peak', 'birth_time', 'division_time', 'birth_label',
               'sb', 'sd', 'delta', 'tau', 'elong_rate', 'septum_position']

        # Make dataframe for plotting variables
    Cells_dict = {cell_id : vars(cell) for cell_id, cell in Cells.iteritems()}
    Cells_df = pd.DataFrame(Cells_dict).transpose() # must be transposed so data is in columns
    Cells_df = Cells_df.sort(columns=['fov', 'peak', 'birth_time', 'birth_label']) # sort for convinience
    Cells_df = Cells_df[columns].apply(pd.to_numeric)

    return Cells_df

def hex_time_plot(Cells_df, time_mark='birth_time', x_extents=None, bin_extents=None):
    '''
    Plots cell parameters over time using a hex scatter plot and a moving average
    '''

    # lists for plotting and formatting
    columns = ['sb', 'sd', 'delta', 'tau', 'elong_rate', 'w']
    titles = ['Length at Birth', 'Length at Division', 'Delta',
              'Generation Time', 'Elongation Rate', 'Width at ']
    ylabels = ['$\mu$m', '$\mu$m', '$\mu$m', 'min', '$\lambda$', '$\mu$m']

    # binning parameters, should be arguments
    binmin = 5 # minimum bin size to display
    bingrid = (50, 10) # how many bins to have in the x and y directions
    moving_window = 10 # window to calculate moving stat

    # bining parameters for each data type
    # bin_extent in within which bounds should bins go. (left, right, bottom, top)
    if x_extents == None:
        x_extents = (0, 1500)

    if bin_extents == None:
        bin_extents = [(x_extents[0], x_extents[1], 0, 10),
                      (x_extents[0], x_extents[1], 0, 15),
                      (x_extents[0], x_extents[1], 0, 10),
                      (x_extents[0], x_extents[1], 0, 75),
                      (x_extents[0], x_extents[1], 0, 0.06),
                      (x_extents[0], x_extents[1], 0, 1)]

    # create figure, going to apply graphs to each axis sequentially
    fig, axes = plt.subplots(nrows=len(columns)/2, ncols=2,
                            figsize=[15,5*len(columns)/2], squeeze=False)
    ax = np.ravel(axes)

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
        ax[i].set_title(titles[i], size=20)
        ax[i].set_ylabel(ylabels[i], size=20)

        p.set_cmap(cmap=plt.cm.Blues) # set color and style

    ax[0].legend(['%s minute binned average' % moving_window], fontsize=14)
    ax[4].set_xlabel('%s [min]' % time_mark, size=20)
    ax[5].set_xlabel('%s [min]' % time_mark, size=20)

    # Make title, need a little extra space
    plt.subplots_adjust(top=0.925, hspace=0.25)
    fig.suptitle('Cell Parameters Over Time', size=24)

    # # additional formatting
    # for i, axis in enumerate(ax):
    #     axis.set_xlim([50, 1500])
    #
    #     if i in [0,1,2]:
    #         axis.set_ylim([0, 15])
    #
    #     if i == 3:
    #         axis.set_ylim([0, 70])
    #
    #     if i == 4:
    #         axis.set_ylim([0, 0.05])
    #
    #     if i == 5:
    #         axis.set_ylim([0.4, 0.6])

    sns.despine()
    # plt.show() # it's better to show the plot in the main script incaes you want to edit it

    return fig, ax

def violin_fovs(Cells_df):
    '''
    Create violin plots of cell stats across FOVs
    '''

    sns.set(style="whitegrid", palette="pastel", color_codes=True)

    columns = ['sb', 'sd', 'delta', 'tau', 'elong_rate', 'septum_position']
    ylabels = ['$\mu$m', '$\mu$m', '$\mu$m', 'min', '$\lambda$', 'daughter/mother']
    titles = ['Length at Birth', 'Length at Division', 'Delta',
                  'Generation Time', 'Elongation Rate', 'Septum Position']

    fig, axes = plt.subplots(nrows=len(columns), ncols=1,
                            figsize=[20,5*len(columns)], squeeze=False)
    ax = np.ravel(axes)

    for i, column in enumerate(columns):
        # Draw a nested violinplot and split the violins for easier comparison
        sns.violinplot(x="fov", y=column, data=Cells_df,
                      scale="count", inner="quartile", ax=ax[i], lw=3)

        ax[i].set_title(titles[i], size=24)
        ax[i].set_ylabel(ylabels[i], size=24)
        ax[i].set_xlabel('')
        ax[i].tick_params(axis='both', which='major', labelsize=15)

    ax[i].set_xlabel('FOV', size=24)

    # plt.tight_layout()

    # Make title, need a little extra space
    plt.subplots_adjust(top=0.95)
    fig.suptitle('Cell Parameters Across FOVs', size=30)

    sns.despine()
    # plt.show()

    return fig, ax

def saw_tooth_plot(Lineages):
    fig, axes = plt.subplots(ncols=1, nrows=3,
                         figsize=(16, 3*3))

    # fig, axes = plt.subplots(ncols=1, nrows=len(Lineages.keys()),
    #                           figsize=(16, 3*len(Lineages.keys())))
    ax = axes.flat

    # for i, fov in enumerate(Lineages.keys()):
    for i, fov in enumerate([1,2,3]):
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
    #         pprint(lin)

            for cell_id, cell in lin:
                ax[i].semilogy(cell.times_w_div, cell.lengths_w_div,
                               color=plt.cm.jet(int(255*peak/2600)), lw=1, alpha=0.75)

                # draw a connecting lines betwee mother and daughter

    #             print(cell.birth_time, last_div_time)
                if cell.birth_time == last_div_time:
                    ax[i].semilogy([last_div_time, cell.birth_time],
                                   [last_length, cell.sb],
                                   color=plt.cm.jet(int(255*peak/2600)), lw=1, alpha=0.75)

                # record the last division time and length for next time
                last_div_time = cell.division_time
                last_length = cell.sd

                # same the max div length for axis plotting
                if last_length > max_div_length:
                    max_div_length = last_length

        title_string = 'Cell Length vs Time FOV %d' % fov
        ax[i].set_title(title_string, size=18)
        # ax[1].set_xlabel('Time [min]', size=20)
        ax[i].set_ylabel('Log(Length [um])', size=14)
        ax[i].set_ylim([0, max_div_length + 2])

    sns.despine()
    plt.subplots_adjust(hspace=0.5)

    return fig, ax
