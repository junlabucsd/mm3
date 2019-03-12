#!/usr/bin/python
from __future__ import print_function
import six

# import modules
import os # interacting with file systems

# number modules
import numpy as np
import scipy.stats as sps
from scipy.optimize import least_squares, curve_fit
import pandas as pd
from random import sample

# image analysis modules
from skimage.measure import regionprops # used for creating lineages

# plotting modules
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines

import seaborn as sns
sns.set(style='ticks', color_codes=True)
sns.set_palette('deep')

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Myriad Pro'

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# set axes and tick width
plt.rc('axes', linewidth=0.5)
mpl.rcParams['xtick.major.size'] = 2
mpl.rcParams['xtick.major.width'] = 0.5
mpl.rcParams['xtick.minor.size'] = 2
mpl.rcParams['xtick.minor.width'] = 0.5
mpl.rcParams['ytick.major.size'] = 2
mpl.rcParams['ytick.major.width'] = 0.5
mpl.rcParams['ytick.minor.size'] = 2
mpl.rcParams['ytick.minor.width'] = 0.5
# additional parameters
mpl.rcParams['lines.linewidth'] = 0.5

import mm3_helpers as mm3

# This dataframe contains the titles and units corresponding to
# parameters in the data.
pnames = dict(sb=dict(label='birth length', symbol='L$_b$',
                    unit='$\mu$m'),
              sd=dict(label='division length', symbol='L$_d$',
                    unit='$\mu$m'),
              delta=dict(label='added length', symbol='$\Delta_d$',
                    unit='$\mu$m'),
              elong_rate=dict(label='elongation rate', symbol='$\\lambda$',
                    unit='1/hours'),
              tau=dict(label='generation time', symbol='$\\tau$',
                    unit='minutes'),
              septum_position=dict(label='septum position',
                    symbol='L$_\\frac{1}{2}$',
                    unit='L$_b$/L$_d$'),
              width=dict(label='width', symbol='W', unit='$\mu$m'),

              # These are the columns in the cell cycle df
              # [u'Unnamed: 0', u'cell_id', u'initiation_time', u'initiation_length',
              # u'termination_time', u'B', u'C', u'D', u'tau_cyc', u'n_oc', u'width',
              # u'tau', u'birth_time', u'division_time', u'elong_rate', u'birth_length',
              # u'division_length', u'delta', u'septum_position', u'added_length',
              # u'mother_width', u'true_initiation_length', u'true_initiation_volume',
              # u'initiation_vol', u'unit_cell'],

              # cell cycle related ones
              unit_cell=dict(label='unit cell', symbol='S$_0$',
                    unit='$\mu$m$^3$'),
              # So=dict(label='unit cell (mean width)', symbol='S$_0$',
              #         unit='$\mu$m$^3$'),
              # Vi=dict(label='true initiation volume', symbol='V$_i$',
              #       unit='$\mu$m$^3$'),
              # initiation_vol=dict(label='initiation volume', symbol='v$_i$',
              #       unit='$\mu$m$^3$'), # avoid using this one
              initiation_volume=dict(label='initiation volume', symbol='v$_i$',
                    unit='$\mu$m$^3$'),
              true_initiation_volume=dict(label='true initiation volume', symbol='V$_i$',
                    unit='$\mu$m$^3$'),
              initiation_length=dict(label='initiation length', symbol='l$_i$',
                    unit='$\mu$m'),
              true_initiation_length=dict(label='true initiation length', symbol='L$_i$',
                    unit='$\mu$m'),
              B=dict(label='B period', symbol='B',
                      unit='minutes'),
              C=dict(label='C period', symbol='C',
                    unit='minutes'),
              D=dict(label='D period', symbol='D',
                    unit='minutes'),
              tau_cyc=dict(label='C+D', symbol='$\\tau_{cyc}$',
                           unit='minutes'),
              initiation_delta=dict(label='added initiation length', symbol='$\Delta_i$',
                    unit='$\mu$m$^3$'),
              n_oc=dict(label='overlapping cell cycles', symbol='n$_{oc}$',
                        unit='n'))

              # not cell cycle related.
             # birth_length=dict(label='birth length', symbol='L$_b$',
             #                 unit='$\mu$m'),
             # division_length=dict(label='division length', symbol='L$_d$',
             #     unit='$\mu$m'))

### Data conversion functions ######################################################################
def cells2df(Cells, rescale=False):
    '''
    Take cell data (a dicionary of Cell objects) and return a dataframe.

    rescale : boolean
        If rescale is set to True, then the 6 major parameters are rescaled by their mean.
    '''

    # columns to include
    columns = ['fov', 'peak', 'birth_time', 'division_time', 'birth_label',
               'sb', 'sd', 'width', 'delta', 'tau', 'elong_rate', 'septum_position']
    rescale_columns = ['sb', 'sd', 'delta', 'tau', 'elong_rate', 'septum_position']

    # should not need this as of unet
    # for cell_tmp in Cells:
    #     Cells[cell_tmp].width = np.mean(Cells[cell_tmp].widths_w_div)

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

def cells2_ccdf(Cells):
    '''
    Take cell data (a dicionary of Cell objects) and return a dataframe. Looks for cell cycle info as well.
    '''

    # columns to include
    columns = ['fov', 'peak', 'birth_label',
               'birth_time', 'division_time',
               'sb', 'sd', 'width', 'delta', 'tau', 'elong_rate', 'septum_position',
               'initiation_time', 'termination_time', 'n_oc',
               'true_initiation_length', 'initiation_length',
               'true_initiation_volume', 'initiation_volume',
               'unit_cell', 'initiation_delta',
               'B', 'C', 'D', 'tau_cyc']

    # should not need this as of unet
    # for cell_tmp in Cells:
    #     Cells[cell_tmp].width = np.mean(Cells[cell_tmp].widths_w_div)

    # Make dataframe for plotting variables
    Cells_dict = cells2dict(Cells)
    Cells_df = pd.DataFrame(Cells_dict).transpose() # must be transposed so data is in columns
    Cells_df = Cells_df.sort_values(by=['fov', 'peak', 'birth_time', 'birth_label'])
    Cells_df = Cells_df[columns].apply(pd.to_numeric)

    return Cells_df

def cells2dict(Cells):
    '''
    Take a dictionary of Cells and returns a dictionary of dictionaries
    '''

    Cells_dict = {cell_id : vars(cell) for cell_id, cell in six.iteritems(Cells)}

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

def find_cells_of_fov(Cells, FOVs=[]):
    '''Return only cells from certain FOVs.

    Parameters
    ----------
    FOVs : int or list of ints
    '''

    fCells = {} # f is for filtered

    if type(FOVs) is int:
        FOVs = [FOVs]

    fCells = {cell_id : cell_tmp for cell_id, cell_tmp in six.iteritems(Cells) if cell_tmp.fov in FOVs}

    return fCells

def find_cells_of_fov_and_peak(Cells, fov_id, peak_id):
    '''Return only cells from a specific fov/peak
    Parameters
    ----------
    fov_id : int corresponding to FOV
    peak_id : int correstonging to peak
    '''

    fCells = {} # f is for filtered

    for cell_id in Cells:
        if Cells[cell_id].fov == fov_id and Cells[cell_id].peak == peak_id:
            fCells[cell_id] = Cells[cell_id]

    return fCells

def find_cells_born_before(Cells, born_before=None):
    '''
    Returns Cells dictionary of cells with a birth_time before the value specified
    '''

    if born_before == None:
        return Cells

    fCells = {cell_id : Cell for cell_id, Cell in six.iteritems(Cells) if Cell.birth_time <= born_before}

    return fCells

def find_cells_born_after(Cells, born_after=None):
    '''
    Returns Cells dictionary of cells with a birth_time after the value specified
    '''

    if born_after == None:
        return Cells

    fCells = {cell_id : Cell for cell_id, Cell in six.iteritems(Cells) if Cell.birth_time >= born_after}

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

    # remove peaks and that do not contain cells
    remove_fovs = []
    for fov_id, peaks in six.iteritems(Cells_by_peak):
        remove_peaks = []
        for peak_id in peaks.keys():
            if not peaks[peak_id]:
                remove_peaks.append(peak_id)

        for peak_id in remove_peaks:
            peaks.pop(peak_id)

        if not Cells_by_peak[fov_id]:
            remove_fovs.append(fov_id)

    for fov_id in remove_fovs:
        Cells_by_peak.pop(fov_id)

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

    for cell_id, Cell in six.iteritems(Cells):
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

    for fov, peaks in six.iteritems(Lineages):
       # print("fov = {:d}".format(fov))
        # Create a dictionary to hold this FOV
        Continuous_Lineages[fov] = {}

        for peak, Cells in six.iteritems(peaks):
           # print("{:<4s}peak = {:d}".format("",peak))
            # sort the cells by time in a list for this peak
            cells_sorted = [(cell_id, cell) for cell_id, cell in six.iteritems(Cells)]
            cells_sorted = sorted(cells_sorted, key=lambda x: x[1].birth_time)

            # Sometimes there are not any cells for the channel even if it was to be analyzed
            if not cells_sorted:
                continue

            # look through list to find the cell born immediately before t1
            # and divides after t1, but not after t2
            for i, cell_data in enumerate(cells_sorted):
                cell_id, cell = cell_data
                if cell.birth_time < t1 and t1 <= cell.division_time < t2:
                    first_cell_index = i
                    break

            # filter cell_sorted or skip if you got to the end of the list
            if i == len(cells_sorted) - 1:
                continue
            else:
                cells_sorted = cells_sorted[i:]

            # get the first cell and it's last contiguous daughter
            first_cell = cells_sorted[0][1]
            last_daughter = find_last_daughter(first_cell, Cells)

            # check to the daughter makes the second cut off
            if last_daughter.birth_time > t2:
                # print(fov, peak, 'Made it')

                # now retrieve only those cells within the two times
                # use the function to easily return in dictionary format
                Cells_cont = find_cells_born_after(Cells, born_after=t1)
                Cells_cont = find_cells_born_before(Cells_cont, born_before=t2)

                # append the first cell which was filtered out in the above step
                Cells_cont[first_cell.id] = first_cell

                # and add it to the big dictionary
                Continuous_Lineages[fov][peak] = Cells_cont

        # remove keys that do not have any lineages
        if not Continuous_Lineages[fov]:
            Continuous_Lineages.pop(fov)

    return Continuous_Lineages

def find_generation_gap(cell, Cells, gen):
    '''Finds how many continuous ancestors this cell has.'''

    if cell.parent in Cells:
        gen += 1
        gen = find_generation_gap(Cells[cell.parent], Cells, gen)

    return gen

def return_ancestors(cell, Cells, ancestors):
    '''Returns all ancestors of a cell. Returns them in reverse age.'''

    if cell.parent in Cells:
        ancestors.append(cell.parent)
        ancestors = return_ancestors(Cells[cell.parent], Cells, ancestors)

    return ancestors

def find_lineages_of_length(Cells, n_gens=5, remove_ends=False):
    '''Returns cell lineages of at least a certain length, indicated by n_gens.

    Parameters
    ----------
    Cells - Dictionary of cell objects
    n_gens - int. Minimum number generations in lineage to be included.
    remove_ends : bool. Remove the first and last cell from the list. So number of minimum cells in a lineage is n_gens - 2.
    '''

    filtered_cells = []

    for cell_id, cell_tmp in six.iteritems(Cells):
        # find the last continuous daughter
        last_daughter = find_last_daughter(cell_tmp, Cells)

        # check if last daughter is n generations away from this cell
        gen = 0
        gen = find_generation_gap(last_daughter, Cells, gen)

        if gen >= n_gens:
            ancestors = return_ancestors(last_daughter, Cells, [last_daughter.id])

            # remove first cell and last cell, they may be weird
            if remove_ends:
                ancestors = ancestors[1:-1]

            filtered_cells += ancestors

    # remove all the doubles
    filtered_cells = sorted(list(set(filtered_cells)))

    # add all the cells that made it back to a new dictionary.
    Filtered_Cells = {}
    for cell_id in filtered_cells:
        Filtered_Cells[cell_id] = Cells[cell_id]

    return Filtered_Cells

def lineages_to_dict(Lineages):
    '''Converts the lineage structure of cells organized by peak back
    to a dictionary of cells. Useful for filtering but then using the
    dictionary based plotting functions'''

    Cells = {}

    for fov, peaks in six.iteritems(Lineages):
        for peak, cells in six.iteritems(peaks):
            Cells.update(cells)

    return Cells

### Statistics and analysis functions ##############################################################
def stats_table(Cells_df):
    '''Returns a Pandas dataframe with statistics about the 6 major cell parameters.
    '''

    columns = ['sb', 'sd', 'width', 'delta', 'tau', 'elong_rate', 'septum_position']
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

def channel_locations(channel_file, filetype='specs'):
    '''Plot the location of the channels across FOVs

    Parameters
    ----------
    channel_dict : dict
        Either channels_masks or specs dictionary.
    filetype : str, either 'specs' or 'channel_masks'
        What type of file is provided, which effects the plot output.

    '''

    fig = plt.figure(figsize=(4,4))

    point_size = 10

    # Using the channel masks
    if filetype == 'channel_masks':
        for key, values in six.iteritems(channel_file):
        # print('FOV {} has {} channels'.format(key, len(values)))
            y = (np.ones(len(values))) + key - 1
            x = values.keys()
            plt.scatter(x, y, s=point_size)

    # Using the specs file
    if filetype == 'specs':
        for key, values in six.iteritems(channel_file):
            y = list((np.ones(len(values))) + key - 1)
            x = values.keys()

            # green for analyze (==1)
            greenx = [x[i] for i, v in enumerate(values.values()) if v == 1]
            greeny = [y[i] for i, v in enumerate(values.values()) if v == 1]
            plt.scatter(greenx, greeny, color='g', s=point_size)

            # blue for empty (==0)
            bluex = [x[i] for i, v in enumerate(values.values()) if v == 0]
            bluey = [y[i] for i, v in enumerate(values.values()) if v == 0]
            plt.scatter(bluex, bluey, color='b', s=point_size)

            # red for ignore (==-1)
            redx = [x[i] for i, v in enumerate(values.values()) if v == -1]
            redy = [y[i] for i, v in enumerate(values.values()) if v == -1]
            plt.scatter(redx, redy, color='r', s=point_size)

    plt.title('Channel locations across FOVs')
    plt.xlabel('peak position [x pixel location of channel in TIFF]')
    plt.ylabel('FOV')

    return fig

def cell_counts(Cells, title='counts'):
    '''Returns dataframe of counts of cells based on poll age and region number

    Parameters
    ----------
    Cells : dict
        Dictionary of cell objects.
    title : str
        Optional column title.

    '''
    index_names = ['all cells', 'with pole age', 'without pole age',
                   'mothers', '01 cells', '10 cells', '02 cells', 'other pole age',
                   'r1 cels', 'r2 cells', 'r3 cells', 'r4 cells', 'r>4 cells']
    count_df = pd.DataFrame([], index=index_names)


    with_poleage = 0
    without_poleage = 0

    n1000_0 = 0
    n01 = 0
    n10 = 0
    n02 = 0
    n20 = 0
    unknown = 0

    nr1 = 0
    nr2 = 0
    nr3 = 0
    nr4 = 0
    nrmore = 0

    for cell_id, cell_tmp in six.iteritems(Cells):
        if cell_tmp.poleage:
            with_poleage += 1
            if cell_tmp.poleage == (1000, 0):
                n1000_0 += 1
            elif cell_tmp.poleage == (0, 1) and cell_tmp.birth_label <= 2:
                n01 += 1
            elif cell_tmp.poleage == (1, 0) and cell_tmp.birth_label <= 3:
                n10 += 1
            elif cell_tmp.poleage == (0, 2):
                n02 += 1
            else:
                unknown += 1
        elif cell_tmp.poleage == None:
            without_poleage += 1

        if cell_tmp.birth_label == 1:
            nr1 += 1
        elif cell_tmp.birth_label == 2:
            nr2 += 1
        elif cell_tmp.birth_label == 3:
            nr3 += 1
        elif cell_tmp.birth_label == 4:
            nr4 += 1
        else:
            nrmore += 1

    # make a tuple of this data, which will become a row of the dataframe
    count_df[title] = pd.Series([len(Cells), with_poleage, without_poleage, n1000_0, n01, n10, n02, unknown, nr1, nr2, nr3, nr4, nrmore], index=index_names)

    return count_df

def add_cc_info(Cells, matlab_df, time_int):
    '''Adds cell cycle information from the Matlab Cycle Picker .csv to the Cell objects.
    Only cell_id, initiation_time, and termination_time are used from the .csv.
    The times in the information from Matlab should be experimental index.

    Parameters
    ----------
    Cells : dict
        Dictionary of cell objects
    matlab_df : DataFrame or Path
        Dataframe of .csv or Path to the .csv output from Matlab.
    time_int : int or float
        Picture taking interval for the experiment.
    '''

    if type(matlab_df) == str:
        matlab_df = pd.read_csv(matlab_df)

    # counters for which cells do and do not have cell cycle info
    n_in_cc_df = 0
    n_not_in_cc_df = 0

    population_width = np.mean(cells2df(Cells)['width'])

    for cell_id, cell_tmp in six.iteritems(Cells):

        # intialize dictionary of attributes to add to cells
        attributes = dict(initiation_time=None,
                          termination_time=None,
                          n_oc=2,
                          true_initiation_length=None,
                          initiation_length=None,
                          true_initiation_volume=None,
                          initiation_volume=None,
                          unit_cell=None,
                          B=None,
                          C=None,
                          D=None,
                          tau_cyc=None,
                          initiation_delta=None)


        if matlab_df['cell_id'].str.contains(cell_id).any():
            n_in_cc_df += 1

            ### pull data straight from dataframe
            cell_cc_row = matlab_df[matlab_df['cell_id'] == cell_id]
            attributes['initiation_time'] = cell_cc_row.iloc[0]['initiation_time'] # time is image interval
            attributes['termination_time'] = cell_cc_row.iloc[0]['termination_time']

            ### calculated values
            # get mother and upper generation ids just in case.
            mother_id = gmother_id = ggmother_id = None
            mother_id = cell_tmp.parent
            if mother_id in Cells:
                gmother_id = Cells[mother_id].parent
            if gmother_id in Cells:
                ggmother_id = Cells[gmother_id].parent

            # 1 overlapping cell cycle, initiation time is in this cell's times
            if attributes['initiation_time'] in cell_tmp.times:
                attributes['n_oc'] = 1
                init_cell_id = cell_id
            elif attributes['initiation_time'] in Cells[mother_id].times:
                attributes['n_oc'] = 2
                init_cell_id = mother_id
            elif attributes['initiation_time'] in Cells[gmother_id].times:
                attributes['n_oc'] = 3
                init_cell_id = gmother_id
            elif attributes['initiation_time'] in Cells[ggmother_id].times:
                attributes['n_oc'] = 4
                init_cell_id = ggmother_id
            else:
                print('Initiation cell not found for {}'.format(cell_id))

            # find index of intiation in that cell. Note if the time was recorded with real time or not
            try:
                # index in the initaition cell
                init_index = Cells[init_cell_id].times.index(attributes['initiation_time'])
            except:
                print('{} with n_oc {} has initiation index {}'.format(cell_id, attributes['n_oc'], attributes['initiation_time']))
                for key, value in attributes.items():
                    setattr(cell_tmp, key, value)

                continue # just skip this cell for the rest of the info

            attributes['true_initiation_length'] = Cells[init_cell_id].lengths_w_div[init_index]
            attributes['initiation_length'] = (Cells[init_cell_id].lengths_w_div[init_index] /
                                               2**(attributes['n_oc'] - 1))
    #         print(attributes['initiation_length'], cell_cc_row.iloc[0]['initiation_length'],
    #               attributes['n_oc'], attributes['true_initiation_length'], cell_tmp.id)
            attributes['true_initiation_volume'] = Cells[init_cell_id].volumes_w_div[init_index]
            attributes['initiation_volume'] = (Cells[init_cell_id].volumes_w_div[init_index] /
                                                    2**(attributes['n_oc'] - 1))

            # use population width for unit cell
            pop_rads = population_width / 2
            # volume is cylinder + sphere using with as radius
            cyl_lengths = attributes['initiation_length'] - pop_rads
            pop_init_vol = ((4/3) * np.pi * np.power(pop_rads, 3)) + (np.pi * np.power(pop_rads, 2) * cyl_lengths)
            attributes['unit_cell'] = pop_init_vol * np.log(2)

            # use the time_int to give the true elapsed time in minutes.
            attributes['B'] = (Cells[cell_id].birth_time - attributes['initiation_time']) * time_int
            attributes['C'] = (attributes['termination_time'] - attributes['initiation_time']) * time_int
            attributes['D'] = (Cells[cell_id].division_time - attributes['termination_time']) * time_int
            attributes['tau_cyc'] = attributes['C'] + attributes['D']

        else:
            n_not_in_cc_df += 1

        for key, value in attributes.items():
            setattr(cell_tmp, key, value)

    print('There are {} cells in the cell cycle .csv and {} not.'.format(n_in_cc_df, n_not_in_cc_df))

    # Loop through cells again to determine initiation adder size
    # Fangwei's definition is the added unit cell size between a cell and it's daughter
    n_init_delta = 0
    for cell_id, cell_tmp in six.iteritems(Cells):
        # this cell must have a unit cell size, a daughter, and that daughter must have an So
        # We always use daughter 1 for cell cycle picking.
        if cell_tmp.unit_cell != None and cell_tmp.daughters[0] in Cells:
            if Cells[cell_tmp.daughters[0]].unit_cell != None:
                cell_tmp.initiation_delta = (2*Cells[cell_tmp.daughters[0]].unit_cell -
                                             cell_tmp.unit_cell)
                n_init_delta += 1
    print('There are {} cells with an initiation delta'.format(n_init_delta))

    return Cells

# def add_cellcycle_df(df):
#     '''Adds additional columns to a cell cycle dataframe'''
#
#     # use average width of population to calculate population initiation volume (So)
#     avg_width = np.mean(df['width'])
#     df['So'] = (((4/3) * np.pi * np.power(avg_width/2, 3)) + \
#                 (np.pi * np.power(avg_width/2, 2) *
#                 (df['true_initiation_length'] - avg_width))) / (2**(df['n_oc']-1)) * np.log(2)
#
#     return df

### Plotting functions #############################################################################
### Distrbutions -----------------------------------------------------------------------------------
def plot_distributions(Cells_df, color='b', title='Distributions'):
    '''
    Plot distributions of the 6 major parameters
    '''

    columns = ['sb', 'elong_rate', 'sd', 'tau', 'delta', 'septum_position']
    xlabels = ['$\mu$m', '$\lambda$', '$\mu$m', 'min', '$\mu$m', 'daughter/mother']
    titles = ['length at birth', 'elongation rate', 'length at division',
              'generation time', 'added length', 'septum position']
    hist_options = {'histtype' : 'step', 'lw' : 2, 'color' : color}
    kde_options = {'lw' : 2, 'linestyle' : '--', 'color' : color}

    # create figure, going to apply graphs to each axis sequentially
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=[8,10])
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
            bin_edges = np.array(range(0, int(data.max())+1, 4)) + 2
            sns.distplot(data, ax=ax[i], bins=bin_edges,
                         hist_kws=hist_options, kde_kws=kde_options)

        else:
            sns.distplot(data, ax=ax[i], bins=20,
                         hist_kws=hist_options, kde_kws=kde_options)

        ax[i].set_title(titles[i])
        ax[i].set_xlabel(xlabels[i])

        ax[i].set_xlim(0, 2*np.mean(data_mean))

        ax[i].get_yaxis().set_ticks([])
        # ax[i].set_ylabel('pdf')
        ax[i].legend(['$\mu$=%.3f, CV=%.2f' % (data_mean, data_cv)], fontsize=SMALL_SIZE, loc=1, frameon=False)

    sns.despine(left=True)
    plt.tight_layout()
    plt.subplots_adjust(top=0.925, hspace=0.35)
    fig.suptitle(title)

    return fig, ax

def plot_rescaled_distributions(Cells_df):
    '''
    Plot the 6 major cell distributions with all values normalized by the mean.
    '''

    # sns.set(style="ticks", palette="pastel", color_codes=True, font_scale=1.25)

    columns = ['sb', 'sd', 'delta', 'tau', 'elong_rate', 'septum_position']
    # xlabels = ['Rescaled Length at Birth', 'Rescaled Length at Division', 'Rescaled Delta',
    #           'Rescaled Generation Time', 'Rescaled Elongation Rate', 'Rescaled Septum Position']
    xlabels = ['$L_b$ /<$L_b$>', '$L_d$ /<$L_d$>', '$\Delta_d$ /<$\Delta_d$>',
               '$\\tau$ /<$\\tau$>', '$\lambda$ /<$\lambda$>',
               '$L_\\frac{1}{2}$ /<$L_\\frac{1}{2}$>']
    hist_options = {'histtype' : 'step', 'lw' : 2, 'color' : 'b'}
    kde_options = {'lw' : 2, 'linestyle' : '--', 'color' : 'b'}

    # create figure, going to apply graphs to each axis sequentially
    fig, axes = plt.subplots(nrows=1, ncols=len(columns), figsize=[8, 3])
    ax = axes.flat

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
            bin_edges = (np.array(range(0, int(data.max())+1, 4)) + 1) / data_mean
            sns.distplot(plot_data, ax=ax[i], bins=bin_edges,
                         hist_kws=hist_options, kde_kws=kde_options)

        else:
            bin_edges = (np.arange(0.4, 1.6, 0.05))
            sns.distplot(plot_data, ax=ax[i], bins=bin_edges,
                         hist_kws=hist_options, kde_kws=kde_options)

        ax[i].set_xlabel(xlabels[i])
        ax[i].set_xlim([0.4, 1.6])
        ax[i].get_yaxis().set_visible(False)
        # ax[i].legend(['CV=%.2f' % data_cv], size=12)
        # plt.legend(markerscale=0) # this will remove the line next the label
        ax[i].annotate('CV=%.2f' % data_cv, xy=(0.75,0.85), xycoords='axes fraction')

        for t in ax[i].get_xticklabels():
            t.set(rotation=45)

    # Make title, need a little extra space
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, hspace=0.35, left=None, right=None)
    fig.suptitle('Rescaled cell parameter distributions')

    sns.despine(left=True)

    return fig, ax

def plotmulti_dist(data, exps, columns=None, df_key='df', disttype='line', nbins=20, rescale_data=False, figlabelcols=None, figlabelfontsize=SMALL_SIZE, individual_legends=True, legend_stat='mean', legendfontsize=SMALL_SIZE/2):
    '''
    Plot distributions of specified parameters.

    Parameters
    ----------
    data : dictionary
        Contains all dataframes, names, colors, etc.
    exps : list
        List of strings of experimental ids to plot
    df_key : str
        The key of dataframe within the data dicionary. Defaults to 'df', but somtimes 'cc_df' is used.
    columns : list
        List of column names to plot
    disttype : 'line' or 'hist'
        'line' plots a continuous line which moves from the center of the bins of the histogram.
        'hist' plots a stepwise histogram.
    nbins : int
        Number of bins to use for histograms. If 'tau' method is being plotted, bins are calculated based on the time interval
    rescale_data : bool
        If True, normalize all data by the mean
    figlabelcols : int
        Number of columns to put in the figure label which says the experiment names.
    figlabelfontsize : int
        Font size to use for bottom figure legend
    individual_legends : bool
        Plot median/mean and CV for each individual plot
    legend_stat : 'mean' or 'median'
        Whether to plot the mean or median in the stat. CV is always plotted.
    legendfontsize : int
        Font size for plot legends
    '''

    if columns == None:
        columns = ['sb', 'elong_rate', 'sd', 'tau', 'delta', 'septum_position']

    no_p = len(columns)

    # holds number of rows, columns, and fig height. All figs are 8 in width
    fig_dims = ((0,0,0), (1,1,8), (1,2,4), (1,3,3), (1,4,3), (2,3,6), (2,3,6),
                (3,3,8), (3,3,8), (3,3,8),
                (4,3,10), (4,3,10), (4,3,10), # 10, 11, 12
                (4,4,8), (4,4,8), (4,4,8), (4,4,8))
    bottom_pad = (0, 0.125, 0.25, 0.35, 0.125, 0.175, 0.175, 0.125, 0.125, 0.125,
                  0.075, 0.075, 0.075,
                  0.1, 0.1, 0.1, 0.1)
    h_pad = (0, 0.375, 0.375, 0.375, 0.375, 0.375, 0.375, 0.375, 0.375, 0.375,
             0.5, 0.5, 0.5,
             0.6, 0.6, 0.6, 0.6)

    fig, axes = plt.subplots(nrows=fig_dims[no_p][0], ncols=fig_dims[no_p][1],
                             figsize=(8,fig_dims[no_p][2]), squeeze=False)
    ax = axes.flat

    xlimmaxs = [0 for col in columns]

    for key in exps:
        df_temp = data[key][df_key]
        color = data[key]['color']
        if 'line_style' in data[key].keys():
            line_style = data[key]['line_style']
        else:
            line_style = '-'

        # Plot each distribution
        for i, column in enumerate(columns):
            data_temp = df_temp[column]

            # remove rows where value is none or NaN
            data_temp = data_temp.dropna()

            # get stats for legend and limits
            data_mean = data_temp.mean()
            data_std = data_temp.std()
            data_cv = data_std / data_mean
            data_max = data_temp.max() # used for setting tau bins
            data_med = data_temp.median()

            if legend_stat == 'mean':
                leg_stat = '$\\mu$={:0.2f}, CV={:0.2f}'.format(data_mean, data_cv)
            elif legend_stat == 'median':
                leg_stat = 'Md={:0.2f}, CV={:0.2f}'.format(data_med, data_cv)

            if rescale_data:
                # rescale data to be centered at mean.
                data_temp = data_temp / np.float(data_mean)

            # set x lim by the highest mean
            if data_mean > xlimmaxs[i]:
                xlimmaxs[i] = data_mean

            if disttype == 'line':
                # use this for line histogram
                if column == 'tau': # make good bin sizes for the not float data
                    time_int = data[key]['t_int']
                    bin_edges = np.arange(0, data_max, step=time_int*2) + time_int/2.0
                    if rescale_data:
                        bin_edges /= data_mean
                    bin_vals, bin_edges = np.histogram(data_temp, bins=bin_edges, density=True)
                else:
                    bin_vals, bin_edges = np.histogram(data_temp, bins=nbins, density=True)

                bin_distance = bin_edges[1] - bin_edges[0]
                bin_centers = bin_edges[:-1] + (bin_distance)/2
                # add zeros to the next points outside this
                bin_centers = np.insert(bin_centers, 0, bin_centers[0] - bin_distance)
                bin_centers = np.append(bin_centers, bin_centers[-1] + bin_distance)
                bin_vals = np.insert(bin_vals, 0, 0)
                bin_vals = np.append(bin_vals, 0)

                ax[i].plot(bin_centers, bin_vals,
                           color=color, ls=line_style, lw=1, alpha=0.75,
                           label=leg_stat)

            elif disttype == 'hist':
            # produce stepwise histogram
                if column == 'tau':
                    time_int = data[key]['t_int']
                    bin_edges = np.arange(0, data_max, step=time_int*2) + time_int/2.0

                    if rescale_data:
                        bin_edges /= data_mean
                    ax[i].hist(data_temp, bins=bin_edges, histtype='step', density=True,
                               color=color, lw=1, alpha=0.5,
                               label=['$\mu$=%.3f, CV=%.2f' % (data_mean, data_cv)])

                else:
                    ax[i].hist(data_temp, bins=nbins, histtype='step', density=True,
                               color=color, lw=1, alpha=0.5,
                               label=leg_stat)

    ### plot formatting
    # settings based on size of figure
    if no_p <= 12:
        title_and_xlabel_fontsize = MEDIUM_SIZE
    else:
        title_and_xlabel_fontsize = SMALL_SIZE

    # title and axis
    for i, col in enumerate(columns):
        ax[i].set_title(pnames[col]['label'] + ', ' + pnames[col]['symbol'],
                        fontsize=title_and_xlabel_fontsize)
        if not rescale_data: # no units if rescaled plotting is on
            ax[i].set_xlabel(pnames[col]['unit'], fontsize=title_and_xlabel_fontsize)
        ax[i].get_yaxis().set_ticks([])
        if rescale_data:
            ax[i].set_xlim(0, 2)
        else:
            ax[i].set_xlim(0, 2*xlimmaxs[i])
        ax[i].set_ylim(0, None)
        if individual_legends:
            ax[i].legend(loc=1, fontsize=legendfontsize)

    # remove axis for plots that are not there
    for ax_no in range(fig_dims[no_p][0] * fig_dims[no_p][1]):
        if ax_no >= no_p:
            sns.despine(ax=ax[ax_no], left=True, bottom=True)
            ax[ax_no].set_xticklabels([])
            ax[ax_no].set_xticks([])
            ax[ax_no].set_yticklabels([])
            ax[ax_no].set_yticks([])
        else:
            sns.despine(ax=ax[ax_no], left=True)

    # legend for whole figure
    handles, _ = ax[0].get_legend_handles_labels()
    labels = [data[key]['name'] for key in exps]
    if figlabelcols == None:
        figlabelcols = int(len(exps)/2)
    fig.legend(handles, labels,
               ncol=figlabelcols, loc=8, fontsize=figlabelfontsize, frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=bottom_pad[no_p], hspace=h_pad[no_p])
    # fig.suptitle('Cell cycle parameter distributions')

    return fig, ax

def plotmulti_phase_dist(data, exps, figlabelcols=None):
    '''
    Plot distributions of the 6 major parameters.
    This is an easy to use function with less customization. Use plotmulti_dist for more options.

    Need to fix to use the pnames dictionary

    Usage
    -----
    dataset_ids = ['exp_key_1', 'exp_key_2']
    fig, ax = mm3_plots.plotmulti_phase_dist(data, dataset_ids)
    fig.show()
    '''

    columns = ['sb', 'elong_rate', 'sd', 'tau', 'delta', 'septum_position']
    xlabels = ['$\mu$m', '$\lambda$', '$\mu$m', 'min', '$\mu$m', 'daughter/mother']
    titles = ['birth length', 'elongation rate', 'length at division',
              'generation time', 'added length', 'septum position']

    # create figure, going to apply graphs to each axis sequentially
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=[8,10])
    ax = np.ravel(axes)

    xlimmaxs = [0 for col in columns]

    for key in exps:

        title = key
        Cells = data[key]['Cells']
        Cells_df = cells2df(Cells)
        time_int = data[key]['t_int']
        color = data[key]['color']

        # Plot each distribution
        for i, column in enumerate(columns):
            data_temp = Cells_df[column]

            # get stats for legend
            data_mean = data_temp.mean()
            data_std = data_temp.std()
            data_cv = data_std / data_mean

            # set x lim to the highest mean
            if data_mean > xlimmaxs[i]:
                xlimmaxs[i] = data_mean

            # set tau bins to be in appropriate interval
            if column == 'tau':
                bin_edges = np.arange(0, data_temp.max(), step=time_int) + time_int/2
                ax[i].hist(data_temp, bins=bin_edges, histtype='step', density=True,
                           color=color, lw=2, alpha=0.5,
                           label=['$\mu$=%.3f, CV=%.2f' % (data_mean, data_cv)])

            else:
                ax[i].hist(data_temp, bins=20, histtype='step', density=True,
                           color=color, lw=2, alpha=0.5,
                           label=['$\mu$=%.3f, CV=%.2f' % (data_mean, data_cv)])

    # plot formatting
    for i, column in enumerate(columns):
        ax[i].set_title(titles[i])
        ax[i].set_xlabel(xlabels[i])
        ax[i].get_yaxis().set_ticks([])
        ax[i].legend(loc=1, frameon=False)
        ax[i].set_xlim(0, 2*xlimmaxs[i])

    # legend for whole figure
    handles, _ = ax[-1].get_legend_handles_labels()
    labels = [data[key]['name'] for key in exps]
    if len(exps) <= 6:
        labelsize=MEDIUM_SIZE
    else:
        labelsize=SMALL_SIZE
    if figlabelcols == None:
        figlabelcols = int(len(exps)/2)

    fig.legend(handles, labels,
               ncol=figlabelcols, loc=8, fontsize=SMALL_SIZE, frameon=False)

    sns.despine(left=True)
    plt.tight_layout()
    plt.subplots_adjust(top=0.925, bottom=0.09, hspace=0.35)
    fig.suptitle('Distributions')

    return fig, ax

def plot_violin_fovs(Cells_df):
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

def plot_violin_birth_label(Cells_df):
    '''
    Create violin plots of cell stats versus the birth label.

    This is a good way to test if the mother cells are aging, or the cells
    farther down the channel are not being segmented well.
    '''

    # sns.set(style="whitegrid", palette="pastel", color_codes=True, font_scale=1.25)

    columns = ['sb', 'sd', 'delta', 'tau', 'elong_rate', 'septum_position']
    ylabels = ['$\mu$m', '$\mu$m', '$\mu$m', 'min', '$\lambda$', 'daughter/mother']
    titles = ['Length at Birth', 'Length at Division', 'Delta',
              'Generation Time', 'Elongation Rate', 'Septum Position']

    fig, axes = plt.subplots(nrows=len(columns), ncols=1,
                            figsize=[8,2.5*len(columns)], squeeze=False)
    ax = np.ravel(axes)

    for i, column in enumerate(columns):
        # Draw a nested violinplot and split the violins for easier comparison
        sns.violinplot(x="birth_label", y=column, data=Cells_df,
                      scale="count", inner="quartile", ax=ax[i], lw=1)

        ax[i].set_title(titles[i])
        ax[i].set_ylabel(ylabels[i])
        ax[i].set_xlabel('')
        ax[i].tick_params(axis='both', which='major', labelsize=10)

    ax[i].set_xlabel('Birth Label')

    # plt.tight_layout()

    # Make title, need a little extra space
    plt.subplots_adjust(top=.925, hspace=0.5)
    fig.suptitle('Cell Parameters vs Birth Label')

    sns.despine()

    return fig, ax

### Time series ------------------------------------------------------------------------------------
def plot_paramtime(Cells_df, scatter_color='b', line_color='blue', time_int=1, window=30, title='Parameters over time'):
    '''
    Plots cell parameters over time using a scatter plot and a moving average

    window : in or float
        Window for averaging in minutes
    '''

    # lists for plotting and formatting
    columns = ['sb', 'elong_rate', 'sd', 'tau', 'delta', 'septum_position']
    titles = ['length at birth', 'elongation rate', 'length at division',
              'generation time', 'added length', 'septum position']
    ylabels = ['$\mu$m', '$\lambda$', '$\mu$m', 'min', '$\mu$m','daughter/mother']

    # create figure, going to apply graphs to each axis sequentially
    fig, axes = plt.subplots(nrows=len(columns)/2, ncols=2,
                             figsize=[8,8], squeeze=False)
    ax = np.ravel(axes)

    # adjust time so it is in hours from first cell
    first_time = Cells_df['birth_time'].min()
    first_time = first_time * time_int / 60.0

    # bining parameters for each data type
    # bin_extent in within which bounds should bins go. (left, right, bottom, top)
    xlims = (0, Cells_df['birth_time'].max() * time_int / 60.0 - first_time)
    bins = np.arange(xlims[0], xlims[1], window / time_int / 60.0)

    # Now plot the filtered data
    for i, column in enumerate(columns):

        # get out just the data to be plot for one subplot
        time_df = Cells_df[['birth_time', column]].apply(pd.to_numeric)
        time_df.sort_values(by='birth_time', inplace=True)

        # plot the hex scatter plot
        times = time_df['birth_time'] * time_int / 60.0 - first_time
        p = ax[i].scatter(times, time_df[column],
                          s=5, alpha=0.25, color=scatter_color,
                          rasterized=True)

        # graph moving average
        bin_mean, bin_edges, bin_n = sps.binned_statistic(times, time_df[column],
                                            statistic='mean', bins=bins)
        bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2
        ax[i].plot(bin_centers, bin_mean, lw=2, alpha=1, color=line_color)

        # formatting
        ax[i].set_title(titles[i])
        ax[i].set_ylabel(ylabels[i])

        ax[i].set_xlim(xlims)
        ax[i].set_ylim(0, 2*np.mean(bin_mean))

    ax[4].set_xlabel('birth time [hours]')
    ax[5].set_xlabel('birth time [hours]')

    sns.despine()
    plt.tight_layout()
    plt.subplots_adjust(top=0.925, hspace=0.35)
    fig.suptitle(title)

    return fig, ax

def plotmulti_phase_paramtime(data, exps, window=30):
    '''
    Plots cell parameters over time using a scatter plot and a moving average.
    Plots multiple datasets onto one
    '''

    # lists for plotting and formatting
    columns = ['sb', 'elong_rate', 'sd', 'tau', 'delta', 'septum_position']
    titles = ['length at birth', 'elongation rate', 'length at division',
              'generation time', 'added length', 'septum position']
    ylabels = ['$\mu$m', '$\lambda$', '$\mu$m', 'min', '$\mu$m','daughter/mother']

    # create figure, going to apply graphs to each axis sequentially
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=[8,10])
    ax = np.ravel(axes)

    ylimmaxs = [0 for col in columns]
    xlimmax = 0

    for key in exps:

        title = key
        Cells = data[key]['Cells']
        Cells_df = cells2df(Cells)
        time_int = data[key]['t_int']
        scatter_color = data[key]['color_light']
        line_color = data[key]['color']

        # adjust time so it is in hours from first cell
        # This may actually be bad, because different datasets have different first_time
        first_time = Cells_df['birth_time'].min()
        first_time = first_time * time_int / 60.0

        # time average window
        xlims = (0, Cells_df['birth_time'].max() * time_int / 60.0 - first_time)
        bins = np.arange(xlims[0], xlims[1], window / time_int / 60.0)

        # set overall xlim max:
        if xlims[1] > xlimmax:
            xlimmax = xlims[1]

        # Now plot the filtered data
        for i, column in enumerate(columns):

            # get out just the data to be plot for one subplot
            time_df = Cells_df[['birth_time', column]].apply(pd.to_numeric)
            time_df.sort_values(by='birth_time', inplace=True)

            # plot the hex scatter plot
            times = time_df['birth_time'] * time_int / 60.0 - first_time
            ax[i].scatter(times, time_df[column],
                          s=5, alpha=0.25, color=scatter_color,
                          rasterized=True)

            # graph moving average
            bin_mean, bin_edges, bin_n = sps.binned_statistic(times, time_df[column],
                                                statistic='mean', bins=bins)
            bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2
            ax[i].plot(bin_centers, bin_mean, lw=2, alpha=0.75, color=line_color)

            # set y lim to the highest mean
            if np.mean(bin_mean) > ylimmaxs[i]:
                ylimmaxs[i] = np.mean(bin_mean)

    # formatting
    for i, column in enumerate(columns):
        ax[i].set_title(titles[i])
        ax[i].set_ylabel(ylabels[i])
        ax[i].set_xlim(0, xlimmax)
        ax[i].set_ylim(0, 2*ylimmaxs[i])

    ax[4].set_xlabel('birth time [hours]')
    ax[5].set_xlabel('birth time [hours]')

    # figure legend
    handles = []
    labels = []
    for key in exps:
        handles.append(mlines.Line2D([], [], color=data[key]['color'],
                                     lw=2, alpha=0.9))
        labels.append(data[key]['name'])
    fig.legend(handles, labels,
               ncol=4, loc=8, fontsize=MEDIUM_SIZE, frameon=False)

    sns.despine()
    plt.tight_layout()
    plt.subplots_adjust(top=0.925, bottom=0.1, hspace=0.35)
    fig.suptitle('Parameters over time')

    return fig, ax

def plot_hex_time(Cells_df, time_mark='birth_time', x_extents=None, bin_extents=None):
    '''
    Plots cell parameters over time using a hex scatter plot and a moving average
    '''

    # lists for plotting and formatting
    columns = ['sb', 'elong_rate', 'sd', 'tau', 'delta', 'septum_position']
    titles = ['Length at Birth', 'Elongation Rate', 'Length at Division',
              'Generation Time', 'Delta', 'Septum Position']
    ylabels = ['$\mu$m', '$\lambda$', '$\mu$m', 'min', '$\mu$m','daughter/mother']

    # create figure, going to apply graphs to each axis sequentially
    fig, axes = plt.subplots(nrows=len(columns)/2, ncols=2,
                             figsize=[8,8], squeeze=False)
    ax = np.ravel(axes)

    # binning parameters, should be arguments
    binmin = 3 # minimum bin size to display
    bingrid = (20, 10) # how many bins to have in the x and y directions
    moving_window = 10 # window to calculate moving stat

    # bining parameters for each data type
    # bin_extent in within which bounds should bins go. (left, right, bottom, top)
    if x_extents == None:
        x_extents = (Cells_df['birth_time'].min(), Cells_df['birth_time'].max())

    if bin_extents == None:
        bin_extents = [(x_extents[0], x_extents[1], 0, 4),
                      (x_extents[0], x_extents[1], 0, 1.5),
                      (x_extents[0], x_extents[1], 0, 8),
                      (x_extents[0], x_extents[1], 0, 140),
                      (x_extents[0], x_extents[1], 0, 4),
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
        ax[i].set_title(titles[i])
        ax[i].set_ylabel(ylabels[i])

        p.set_cmap(cmap=plt.cm.Blues) # set color and style

    ax[5].legend(['%s frame binned average' % moving_window], loc='lower right',
                 frameon=False)
    ax[4].set_xlabel('%s [frame]' % time_mark)
    ax[5].set_xlabel('%s [frame]' % time_mark)

    plt.tight_layout()

    return fig, ax

def plot_derivative(Cells_df, time_mark='birth_time', x_extents=None, time_window=10):
    '''
    Plots the derivtive of the moving average of the cell parameters.

    Parameters
    ----------
    time_window : int
        Time window which the parameters statistic is calculated over
    '''

    sns.set(style="whitegrid", palette="pastel", color_codes=True, font_scale=1.25)

    # lists for plotting and formatting
    columns = ['sb', 'elong_rate', 'sd', 'tau', 'delta', 'septum_position']
    titles = ['Length at Birth', 'Elongation Rate', 'Length at Division',
              'Generation Time', 'Delta', 'Septum Position']
    ylabels = ['$\mu$m', '$\lambda$', '$\mu$m', 'min', '$\mu$m','daughter/mother']

    # create figure, going to apply graphs to each axis sequentially
    fig, axes = plt.subplots(nrows=len(columns)/2, ncols=2,
                            figsize=[10,5*len(columns)/2], squeeze=False)
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

    ax[5].legend(['%s minute binned average' % time_window], fontsize=14, loc='lower right',
                 frameon=False)
    ax[4].set_xlabel('Frame [min/5]', size=16)
    ax[5].set_xlabel('Frame [min/5]', size=16)

    # Make title, need a little extra space
    plt.subplots_adjust(top=0.9, hspace=0.25)
    fig.suptitle('Cell Parameters Over Time', size=24)

    # sns.despine()

    return fig, ax

def plot_average_derivative(Cells, n_diff=1, t_int=1, shift=False, t_shift=0):
    '''
    Plot the average numerical derivative (instantaneous elongation rate in 1/hours) against
    time in minutes. If shift is set to True, then the x axis is renumbered to be relative to
    the shift time. Differentiation is currently just the difference. They units for the y axis
    are scaled to be lambda (1/hours) = ln(2) / tau

    Parameters
    ----------
    Cells : dict
        Dictionary of cell objects
    n_diff : int
        The number of time steps to differentiate over.
    t_int : int
        Time interval of the picturing taking.
    shift : boolean
        Flag for if the time scale should be shifted
    t_shift : int
        Time frame in which shift occured.

    Returns
    -------
    fig, ax : Matplotlib figure and axis objects.
    '''

    ### Calculate the stats
    # This dictionary carries all the lengths by time point, and rate of change by timepoint
    stats_by_time = {'diffs_by_time' : {},
                     'all_diff_times' : [],
                     'diff_means' : [],
                     'diff_stds' : [],
                     'diff_SE' : [],
                     'diff_n' : []}

    # we loop through each cell to find the rate of length change
    for cell_id, Cell in Cells.items():

            # convert lengths to um from pixels and take log
            log_lengths = np.log(np.array(Cell.lengths))

            # take numerical n-step derivative
            lengths_diff = np.diff(log_lengths[::n_diff])

            # convert units to lambda [hours^-1] = ln(2) / tau [hours]
            lengths_diff *= 60 / n_diff / t_int

            # get corresponding times (will be length-1)
            diff_times = Cell.times[n_diff::n_diff]

            # convert from time frame to minutes
            diff_times = np.array(diff_times) * t_int

            # and change to relative shift if flagged
            if shift:
                diff_times -= t_shift * t_int

            # add data to time point centric dictionary
            for i, t in enumerate(diff_times):
                if t in stats_by_time['diffs_by_time']:
                    stats_by_time['diffs_by_time'][t].append(lengths_diff[i])
                else:
                    stats_by_time['diffs_by_time'][t] = [lengths_diff[i]]

    # calculate timepoint by timepoint stats
    # note, you want to go over the dictionary in time order
    for t in sorted(stats_by_time['diffs_by_time']):
        values = stats_by_time['diffs_by_time'][t]

        stats_by_time['all_diff_times'].append(t)
        stats_by_time['diff_means'].append(np.mean(values))
        stats_by_time['diff_stds'].append(np.std(values))
        stats_by_time['diff_SE'].append(np.std(values) / np.sqrt(len(values)))
        stats_by_time['diff_n'].append(len(values))

    ### Plot the graph

    sns.set(style="ticks", palette="pastel", color_codes=True, font_scale=1.25)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))

    # plot average and standard deviation of the rate of change.
    ax.plot(stats_by_time['all_diff_times'], stats_by_time['diff_means'], c='r', lw=2, alpha=1)
    ax.fill_between(stats_by_time['all_diff_times'],
                    np.array(stats_by_time['diff_means']) - np.array(stats_by_time['diff_SE']),
                    np.array(stats_by_time['diff_means']) + np.array(stats_by_time['diff_SE']),
                    facecolor='r', alpha=0.5)
    # ax.errorbar(stats_by_time['all_diff_times'], stats_by_time['diff_means'], stats_by_time['diff_SE'],
    #                c='r', lw=2, alpha=1, elinewidth=1, capsize=1, barsabove=True, ecolor='r', capthick=1,
    #                label='Average inst. rate of change with SE')

    # vertical lines for shift up time
    if shift:
        ax.axvline(x=t_shift*t_int - t_shift*t_int, linewidth=2, color='g', ls='--', alpha=0.5, label='Shift-up time')

    # format plot
    ax.set_title('Average instantaneous growth rate with SE, Time Step = {}'.format(n_diff*t_int), size=22)
    ax.set_ylabel('Growth rate [hours$^{-1}$]', size=20)
    ax.set_xlabel('Time [min]', size=20)
    ax.legend(loc='lower right', fontsize=16, frameon=False)

    return fig, ax

### Correlations -----------------------------------------------------------------------------------
def plotmulti_crosscorrs(data, exps, plot_params=None, pearson=False, legend_loc=1):
    '''
    Plot cross correlation plot with pairwise comparisons. Plots distributions along diagonal.
    Currently geared towards the cell cycle dataframe but it's not specific.

    data : dictionary
        Data dictionary which contains cell data in dataframe, name, color.
    exps : list
        List of strings which are the keys to the data dictionary.
    plot_params : list
        List of parametes to include. Parameter name must match column name
        in df.
    '''

    if plot_params == None:
        plot_params = ['sb', 'delta', 'growth rate']

    no_p = len(plot_params)
    fig, axes = plt.subplots(nrows=no_p, ncols=no_p, figsize=(8,8))
    ax = axes.flat

    # xlimmaxs = np.zeros(len(ax))
    # ylimmaxs = np.zeros(len(ax))

    for exp in exps:
        df = data[exp]['df']
        c = data[exp]['color']

        try:
            scat_c = data[exp]['color_light']
        except:
            scat_c = c

        for i, prow in enumerate(plot_params):
            for j, pcol in enumerate(plot_params):
                ax_no = i * no_p + j

                if i == j: # plot distribution on diagonal
                    # print(prow, len(df[prow]))
                    data_temp = df[prow]

                    # remove rows where value is none or NaN
                    data_temp = data_temp.dropna()

                    # get stats for legend
                    data_mean = data_temp.mean()
                    data_std = data_temp.std()
                    data_cv = data_std / data_mean

                    # line histogram
                    bin_vals, bin_edges = np.histogram(data_temp, bins=20, density=True)
                    bin_distance = bin_edges[1] - bin_edges[0]
                    bin_centers = bin_edges[:-1] + (bin_distance)/2
                    # add zeros to the next points outside this
                    bin_centers = np.insert(bin_centers, 0, bin_centers[0] - bin_distance)
                    bin_centers = np.append(bin_centers, bin_centers[-1] + bin_distance)
                    bin_vals = np.insert(bin_vals, 0, 0)
                    bin_vals = np.append(bin_vals, 0)

                    ax[ax_no].plot(bin_centers, bin_vals,
                               color=c, lw=1, alpha=0.75,
                               label='$\mu$={:.3f}, CV={:.2f}'.format(data_mean, data_cv))

                else: # else plot the scatter plot
                    # grab data where both attributes are not NaN. Great function
                    df_temp = df[[pcol, prow]].dropna(how='any')

                    ax[ax_no].scatter(df_temp[pcol], df_temp[prow],
                                      s=5, alpha=0.25, color=scat_c, label=None,
                                      rasterized=True)

                    ax[ax_no].plot(df_temp[pcol].mean(), df_temp[prow].mean(),
                                   marker='o', ms=5, alpha=1, color=c,
                                   mec='k', mew=0.5, label=None)

                    # pearson correlation
                    if pearson:
                        r = np.corrcoef(df_temp[pcol], df_temp[prow])[0][1]
                        bin_c, bin_m = binned_stat(df_temp[pcol], df_temp[prow], binmin=25)
                        ax[ax_no].plot(bin_c, bin_m,
                                       lw=1, alpha=0.75, color=c, label='Pearson coeff. = {:.2f}'.format(r))
                    else:
                        bin_c, bin_m = binned_stat(df_temp[pcol], df_temp[prow], binmin=25)
                        ax[ax_no].plot(bin_c, bin_m,
                                       lw=1, alpha=0.75, color=c, label=None)

                if j == 0:
                    yl = pnames[prow]['label']+ ' ['+pnames[prow]['unit']+ ']'
                    if len(plot_params) > 3:
                        ax[ax_no].set_ylabel(yl, fontsize=SMALL_SIZE*.75)
                    else:
                        ax[ax_no].set_ylabel(yl, fontsize=MEDIUM_SIZE)
                if i == len(plot_params) - 1:
                    xl = pnames[pcol]['label']+ ' ['+pnames[pcol]['unit']+ ']'
                    if len(plot_params) > 3:
                        ax[ax_no].set_xlabel(xl, fontsize=SMALL_SIZE*.75)
                    else:
                        ax[ax_no].set_xlabel(xl, fontsize=MEDIUM_SIZE)

    for a in ax:
        a.set_xlim(0, None)
        a.set_ylim(0, None)
        if pearson:
            a.legend(loc=legend_loc, fontsize=SMALL_SIZE*0.75, frameon=False)

    for i, prow in enumerate(plot_params):
        for j, pcol in enumerate(plot_params):
            ax_no = i * no_p + j

            if i == j:
                ax[ax_no].get_yaxis().set_ticks([])

    # figure legend
    handles = []
    labels = []
    for exp in exps:
        handles.append(mlines.Line2D([], [], color=data[exp]['color'],
                                     lw=2, alpha=0.9))
        labels.append(data[exp]['name'])
    fig.legend(handles, labels,
               ncol=len(exps), loc=8, fontsize=MEDIUM_SIZE, frameon=False)

    # sns.despine()
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    # fig.suptitle('Parameters over time')

    return fig, ax

def plotmulti_corrs_vs_one(data, exps, y_params=None, x_param='elong_rate'):
    '''
    Plot correlations against one parameter (default is growth rate)
    Currently geared towards the cell cycle dataframe but it's not specific.

    data : dictionary
        Data dictionary which contains cell data in dataframe, name, color.
    exps : list
        List of strings which are the keys to the data dictionary.
    params : list
        List of parametes to include. Parameter name must match column name
        in df.
    x_param : str
        Name of parameter to plot on X axis. Must match column name in dataframe.
    '''

    if y_params == None:
        y_params = ['sb', 'delta', 'elong_rate']

    no_p = len(y_params)

    # holds number of rows, columns, and fig height. All figs are 8 in width
    fig_dims = ((0,0,0), (1,1,8), (1,2,4), (1,3,3), (2,2,8), (2,3,6), (2,3,6),
                (3,3,8), (3,3,8), (3,3,8))

    bottom_pad = (0, 0.125, 0.25, 0.35, 0.125, 0.175, 0.175, 0.125, 0.125, 0.125)

    fig, axes = plt.subplots(nrows=fig_dims[no_p][0], ncols=fig_dims[no_p][1],
                             figsize=(8,fig_dims[no_p][2]), squeeze=False)
    ax = axes.flat

    # xlimmaxs = np.zeros(len(ax))
    # ylimmaxs = np.zeros(len(ax))

    for exp in exps:
        df = data[exp]['df']
        c = data[exp]['color']

        try:
            scat_c = data[exp]['color_light']
        except:
            scat_c = c

        for ax_no, param in enumerate(y_params):
            df_temp = df[[param, x_param]].dropna(how='any')

            ax[ax_no].scatter(df_temp[x_param], df_temp[param],
                              s=5, alpha=0.25, color=scat_c, label=None,
                              rasterized=True)

            ax[ax_no].plot(df_temp[x_param].mean(), df_temp[param].mean(),
                           marker='o', ms=5, alpha=1, color=c,
                           mec='k', mew=0.5, label=None)

            bin_c, bin_m = binned_stat(df_temp[x_param], df_temp[param], binmin=5)
            ax[ax_no].plot(bin_c, bin_m,
                             alpha=0.75, color=c, label=None)

            yl = pnames[param]['label'] + ' ['+pnames[param]['unit'] + ']'
            ax[ax_no].set_ylabel(yl)

            if ax_no >= no_p - fig_dims[no_p][1]:
                xl = pnames[x_param]['label'] + ' ['+pnames[x_param]['unit'] + ']'
                ax[ax_no].set_xlabel(xl)

    for a in ax:
        a.set_xlim(0, None)
        a.set_ylim(0, None)

    # remove axis for plots that are not there
    for ax_no in range(fig_dims[no_p][0] * fig_dims[no_p][1]):
        if ax_no >= no_p:
            sns.despine(ax=ax[ax_no], left=True, bottom=True)
            ax[ax_no].set_xticklabels([])
            ax[ax_no].set_xticks([])
            ax[ax_no].set_yticklabels([])
            ax[ax_no].set_yticks([])

    # figure legend
    handles = []
    labels = []
    for exp in exps:
        handles.append(mlines.Line2D([], [], color=data[exp]['color'],
                                     lw=2, alpha=0.9))
        labels.append(data[exp]['name'])
    fig.legend(handles, labels,
               ncol=4, loc=8, fontsize=MEDIUM_SIZE, frameon=False)

    # sns.despine()
    fig.tight_layout()
    fig.subplots_adjust(bottom=bottom_pad[no_p])
    # fig.suptitle('Parameters over time')

    return fig, ax

def plotmulti_corr(data, exps, param_pairs=None, rescale_data=False, binmin=20, plot_scatter=True, plot_mean=True, pearson=False, figlabelcols=None, figlabelfontsize=SMALL_SIZE, legendfontsize=SMALL_SIZE/2):
    '''
    Plot correlations against one parameter (default is growth rate)
    Currently geared towards the cell cycle dataframe but it's not specific.

    data : dictionary
        Data dictionary which contains cell data in dataframe, name, color.
    exps : list
        List of strings which are the keys to the data dictionary.
    param_pairs : list of tuple pairs for x and y axis
    rescale_data : bool
    binmin : int
        Minimum number of items per bin for binned avergae
    pearson : bool
        Plot Pearson coefficient.
    figlabelcols : int
        Number of columns to put in the figure label which says the experiment names.
    figlabelfontsize : int
        Font size to use for bottom figure legend
    legendfontsize : int
        Font size for plot legends
    '''

    no_p = len(param_pairs)
    # holds number of rows, columns, and fig height. All figs are 8 in width
    fig_dims = ((0,0,0), (1,1,8), (2,1,8), (1,3,3), (2,2,8), (2,3,6), (2,3,6),
                (3,3,8), (3,3,8), (3,3,8))
    bottom_pad = (0, 0.125, 0.25, 0.25, 0.125, 0.175, 0.175, 0.125, 0.125, 0.125)

    fig, axes = plt.subplots(nrows=fig_dims[no_p][0], ncols=fig_dims[no_p][1],
                             figsize=(4,fig_dims[no_p][2]), squeeze=False)
    ax = axes.flat

    for exp in exps:
        df = data[exp]['df']
        c = data[exp]['color']
        ls = data[exp]['line_style']

        try:
            scat_c = data[exp]['color_light']
        except:
            scat_c = c

        for ax_no, param_pair in enumerate(param_pairs):
            pcol, prow = param_pair

            df_temp = df[[pcol, prow]].dropna(how='any')

            col = df_temp[pcol]
            row = df_temp[prow]
            if rescale_data:
                col /= col.mean()
                row /= row.mean()

            # scatter plot on bottom
            if plot_scatter:
                ax[ax_no].scatter(col, row,
                                  s=5, alpha=0.25, color=scat_c, label=None,
                                  rasterized=True)

            # pearson correlation
            if pearson:
                r = np.corrcoef(col, row)[0][1]
                bin_c, bin_m = binned_stat(col, row, binmin=binmin)
                ax[ax_no].plot(bin_c, bin_m,
                               lw=1, alpha=0.75, color=c, ls=ls,
                               label='Pearson coeff. = {:.2f}'.format(r))
            else:
                bin_c, bin_m = binned_stat(col, row, binmin=binmin)
                ax[ax_no].plot(bin_c, bin_m,
                               lw=1, alpha=0.75, color=c, label=' ')

            # plot mean symbol on top
            if plot_mean:
                ax[ax_no].plot(col.mean(), row.mean(),
                               marker='o', ms=5, alpha=1, color=c, ls=ls,
                               mec='k', mew=0.5, label=None)

            # plot title and labels
            if not rescale_data:
                xl = pnames[pcol]['label'] + ' ['+pnames[pcol]['unit'] + ']'
                ax[ax_no].set_xlabel(xl)
                yl = pnames[prow]['label'] + ' ['+pnames[prow]['unit'] + ']'
                ax[ax_no].set_ylabel(yl)
            else:
                xl = 'rescaled ' + pnames[pcol]['label']
                ax[ax_no].set_xlabel(xl)
                yl = 'rescaled ' + pnames[prow]['label']
                ax[ax_no].set_ylabel(yl)

    for a in ax:


        if rescale_data:
            a.set_xlim(0.6, 1.4)
            a.set_ylim(0.6, 1.4)
        else:
            a.set_xlim(0, None)
            a.set_ylim(0, None)

        if pearson:
            a.legend(loc=1, fontsize=legendfontsize, frameon=False)

    # remove axis for plots that are not there
    for ax_no in range(fig_dims[no_p][0] * fig_dims[no_p][1]):
        if ax_no >= no_p:
            sns.despine(ax=ax[ax_no], left=True, bottom=True)
            ax[ax_no].set_xticklabels([])
            ax[ax_no].set_xticks([])
            ax[ax_no].set_yticklabels([])
            ax[ax_no].set_yticks([])

    # legend for whole figure
    handles, _ = ax[0].get_legend_handles_labels()
    labels = [data[key]['name'] for key in exps]
    if figlabelcols == None:
        figlabelcols = int(len(exps)/2)
    fig.legend(handles, labels,
               ncol=figlabelcols, loc=8, fontsize=figlabelfontsize, frameon=False)

    # sns.despine()
    fig.tight_layout()
    fig.subplots_adjust(bottom=bottom_pad[no_p])
    # fig.suptitle('Parameters over time')

    return fig, ax

def plot_correlations_sns(Cells_df, rescale=False):
    '''
    Plot correlations of each major cell parameter against one another

    rescale : boolean
        If rescale is set to True, then axis labeling reflects rescaled data.
    '''

    columns = ['sb', 'sd', 'delta', 'tau', 'elong_rate', 'septum_position']
    labels = ['$L_b$ [$\mu$m]', '$L_d$ [$\mu$m]', '$\Delta$ [$\mu$m]',
               '$\\tau$ [min]', '$\lambda$ [1/hours]',
               '$L_\\frac{1}{2}$']
    rlabels = ['$L_b$ /<$L_b$>', '$L_d$ /<$L_d$>', '$\Delta$ /<$\Delta$>',
               '$\\tau$ /<$\\tau$>', '$\lambda$ /<$\lambda$>',
               '$L_\\frac{1}{2}$ /<$L_\\frac{1}{2}$>']

    # It's just one function from seaborn
    g = sns.pairplot(Cells_df[columns], kind="reg", diag_kind="kde",
                     plot_kws={'scatter':True,
                               'x_bins':10,
                               'scatter_kws':{'alpha':0.25}})
    g.fig.set_size_inches([8,8])

    # Make title, need a little extra space
    # plt.subplots_adjust(top=0.95, left=0.075, right=0.95)
    # g.fig.suptitle('Correlations and Distributions', size=24)

    for i, ax in enumerate(g.axes.flatten()):

        if not rescale:
            if i % 6 == 0:
                ax.set_ylabel(labels[int(i / 6)])
            if i >= 30:
                ax.set_xlabel(labels[i - 30])

        if rescale:
            ax.set_ylim([0.4, 1.6])
            ax.set_xlim([0.4, 1.6])

            if i % 6 == 0:
                ax.set_ylabel(rlabels[int(i / 6)])
            if i >= 30:
                ax.set_xlabel(rlabels[i - 30])


        for t in ax.get_xticklabels():
            t.set(rotation=45)

    plt.subplots_adjust(top=0.95)

    return g

### Traces -----------------------------------------------------------------------------------------
def plot_feather_traces(Cells, trace_limit=1000, color='b', time_int=1, title='Cell traces'):
    '''
    Plot length traces of all cells over time.

    Parameters
    ----------
    trace_limit : int
        Limit the number of traces to this value, chosen randomly from the dictionary Cells.
        Plotting all the traces can be time consuming and mask the trends in the graph.
    color : matplotlib color
        color to plot traces
    time_int : int or float
        Number of minutes per frame. Used to adjust timing.
    '''

    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(8, 4))
    ax = [axes] # same as axes.ravel()

    if trace_limit and trace_limit < len(Cells):
        cell_id_subset = sample(list(Cells), trace_limit)
        Cells = {cell_id : Cells[cell_id] for cell_id in cell_id_subset}

    # adjust time so it is in hours from first cell
    Cells_df = cells2df(Cells)
    first_time = Cells_df['birth_time'].min()
    first_time = first_time * time_int / 60.0

    for cell_id, Cell in six.iteritems(Cells):
        times = np.array(Cell.times_w_div) * time_int / 60.0 - first_time

        ax[0].plot(times, Cell.lengths_w_div, 'b-', lw=.5, alpha=0.25,
                   color=color)

    ax[0].set_xlabel('time [hours]')
    # ax[0].set_xlim(0, None)
    ax[0].set_ylabel('length [$\mu$m]')
    ax[0].set_ylim(0, None)

    sns.despine()
    plt.tight_layout()
    plt.subplots_adjust(top=0.925)
    fig.suptitle(title)

    return fig, ax

def plot_saw_tooth(Lineages, FOVs=None, peaks=None, tif_width=2000, mothers=True):
    '''
    Plot individual cell traces, where each channel gets its own subplot.

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

    # if specific FOVs not noted
    if FOVs == None:
        FOVs = Lineages.keys()

    #fig, axes = plt.subplots(ncols=1, nrows=len(FOVs), figsize=(15, 2.5*len(FOVs)), squeeze=False)
    #ax = axes.flat

    figs=[]

    for i, fov in enumerate(FOVs):
        if peaks == None:
            peaks = Lineages[fov].keys()
        npeaks = len(peaks)

        if (npeaks == 0):
            continue

        fig = plt.figure(num=i, facecolor='w', figsize=(15, 2.5*npeaks))
        gs = gridspec.GridSpec(nrows=npeaks, ncols=1)

        print(fov, npeaks)

        # record max div length for whole FOV to set y lim
        max_div_length = 0

        for r, (peak, lin) in enumerate(Lineages[fov].items()):
            # append axes
            ax = fig.add_subplot(gs[r,0])

            # continue if peaks is not selected
            if not (peak in peaks):
                print ("passing peak {:d}".format(peak))
                continue
            print ("Processing peak {:d}".format(peak))

            # this is to map mothers to daugthers with lines
            last_div_time = None
            last_length = None

            # turn it into a list so it retains time order
            lin = [(cell_id, cell) for cell_id, cell in lin.items()]
            # sort cells by birth time for the hell of it.
            lin = sorted(lin, key=lambda x: x[1].birth_time)

            peak_color = plt.cm.jet(int(255*peak/tif_width))

            for k,(cell_id, cell) in enumerate(lin):
                ax.semilogy(np.array(cell.times_w_div), cell.lengths_w_div,
                        color=peak_color, lw=1, alpha=0.75)

                if mothers:
                    # draw a connecting lines betwee mother and daughter
                    if cell.birth_time == last_div_time:
                        ax.semilogy([last_div_time, cell.birth_time],
                                       [last_length, cell.sb],
                                       color=peak_color, lw=1, alpha=0.75)

                    # record the last division time and length for next time
                    last_div_time = cell.division_time

                # save the max div length for axis plotting
                last_length = cell.sd
                if last_length > max_div_length:
                    max_div_length = last_length

            ax.set_ylabel('Length [um]', size=16)
            ax.set_title("peak {:d}".format(peak), fontsize=14)

        #ax[i].legend(loc='upper center',frameon=True, bbox_to_anchor=(0.5,-0.6),ncol= 6, fontsize=14)
        ax.set_xlabel('Time [min]', size=16)
        # ax[i].set_ylim([0, max_div_length + 2])
        title_string = 'FOV %d' % fov
        fig.suptitle(title_string, size=18)


    rect=[0.,0.,1.,1.0]
    gs.tight_layout(fig,rect=rect)
    # plt.subplots_adjust(top=0.875, bottom=0.1) #, hspace=0.25)
    figs.append(fig)

    sns.despine()
    # plt.subplots_adjust(hspace=0.5)

    return figs

def plot_saw_tooth_fov(Lineages, FOVs=None, tif_width=2000, mothers=True):
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

    # if specific FOVs not noted
    if FOVs == None:
        FOVs = Lineages.keys()

    fig, axes = plt.subplots(ncols=1, nrows=len(FOVs), figsize=(15, 2.5*len(FOVs)), squeeze=False)
    ax = axes.flat

    for i, fov in enumerate(FOVs):
        # record max div length for whole FOV to set y lim
        max_div_length = 0

        for peak, lin in Lineages[fov].items():
            # this is to map mothers to daugthers with lines
            last_div_time = None
            last_length = None

            # turn it into a list so it retains time order
            lin = [(cell_id, cell) for cell_id, cell in lin.items()]
            # sort cells by birth time for the hell of it.
            lin = sorted(lin, key=lambda x: x[1].birth_time)

            peak_color = plt.cm.jet(int(255*peak/tif_width))

            for cell_id, cell in lin:
                ax[i].plot(np.array(cell.times_w_div), cell.lengths_w_div,
                               color=peak_color, lw=1, alpha=0.75)

                if mothers:
                    # draw a connecting lines betwee mother and daughter
                    if cell.birth_time == last_div_time:
                        ax[i].plot([last_div_time, cell.birth_time],
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
        ax[i].set_ylabel('Length [um]', size=16)
        ax[i].set_yscale('symlog')
        ax[i].yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%d"))
        ax[i].set_yticks([1, 4, 8])
        ax[i].set_ylim([1, 8])

    ax[-1].set_xlabel('Time point [5 min]', size=16)

    plt.tight_layout()
    # plt.subplots_adjust(top=0.875, bottom=0.1) #, hspace=0.25)
    # fig.suptitle('Cell Length vs Time ', size=24)

    sns.despine()
    # plt.subplots_adjust(hspace=0.5)

    return fig, ax

def plot_saw_tooth_fl(Cells, time_int=1, fl_plane='c2', fl_int=1, plot_flconc=False, scat_s=10, y_adj_px=3, alt_time='birth', pxl2um=1, plot_foci=False, foci_size=100):
    '''Plot a cell lineage with profile information.

    Parameters
    ----------
    Cells : dict of Cell objects
        All the cells should come from a single peak.
    alt_time : None, 'birth', or float
        Time by which to shift X axis. If 'birth', time will be shited to start with 0.
    foci_size : int
        Factor by which to reduce foci size
    '''

    peak_color = 'blue'

    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(8, 3))
    ax = [axes]

    if plot_flconc:
        ax_fl = ax[0].twinx()

    # this is to map mothers to daugthers with lines
    last_div_time = None
    last_length = None

    # turn it into a list so it retains time order
    lin = [(cell_id, cell) for cell_id, cell in six.iteritems(Cells)]
    # sort cells by birth time for the hell of it.
    lin = sorted(lin, key=lambda x: x[1].birth_time)
    # align time to first birth or shift time
    if alt_time == None:
        alt_time = 0
    elif alt_time == 'birth':
        alt_time = lin[0][1].birth_time * time_int / 60.0

    # Choose colormap. Need to add alpha to color map and normalization
    # green/c2
    max_c2_int = 0
    min_c2_int = float('inf')
    for cell_id, cell in lin:
        for profile_t in getattr(cell, 'fl_profiles_' + fl_plane):
            if max(profile_t) > max_c2_int:
                max_c2_int = max(profile_t)
            if min(profile_t) < min_c2_int:
                min_c2_int = min(profile_t)
    cmap_c2 = plt.cm.Greens
    color_norm_c2 = mpl.colors.Normalize(vmin=min_c2_int, vmax=max_c2_int)

    for cell_id, cell in lin:
        # plot cell length and division lines
        ax[0].plot(np.array(cell.times_w_div) * time_int / 60.0 - alt_time, cell.lengths_w_div,
                    color=peak_color, lw=0.5, alpha=1)
        # draw a connecting lines betwee mother and daughter
        if cell.birth_time == last_div_time:
            ax[0].plot([last_div_time * time_int / 60.0 - alt_time,
                         cell.birth_time * time_int / 60.0 - alt_time],
                        [last_length, cell.sb],
                        color=peak_color, lw=0.5)
        # record the last division time and length for next time
        last_div_time = cell.division_time
        # save the last length to check for division
        last_length = cell.sd

        # plot fluorescence on every time point for which it exists.
        for i, t in enumerate(cell.times):
            if t % fl_int == 1:
                nuc_x = np.ones(len(getattr(cell, 'fl_profiles_' + fl_plane)[i])) * t * time_int / 60.0 - alt_time
                nuc_y = (np.arange(0, len(getattr(cell, 'fl_profiles_' + fl_plane)[i])) - y_adj_px) * pxl2um
                nuc_z = getattr(cell, 'fl_profiles_' + fl_plane)[i]
                ax[0].scatter(nuc_x, nuc_y, c=nuc_z, cmap=cmap_c2,
                               marker='s', s=scat_s, norm=color_norm_c2,
                               rasterized=False)

        # plot fluorescence concentration
        if plot_flconc:
            # pull out time point and data when there is fl data
            fl_data = [(t, cell.fl_vol_avgs[i]) for i, t in enumerate(cell.times) if (t - 1) % fl_int == 0]
            fl_times, fl_vols = zip(*fl_data)
            ax_fl.plot(np.array(fl_times) * time_int / 60.0 - alt_time, fl_vols,
                       color='green', lw=1, ls='--', alpha=0.75)

        # plot foci
        if plot_foci:
            for i, t in enumerate(cell.times):
                for j, foci_y in enumerate(cell.disp_l[i]):
                    foci_y_pos = (foci_y + cell.lengths[i]/2) * pxl2um
                    ax[0].scatter(t * time_int / 60.0 - alt_time, foci_y_pos,
                                   s=cell.foci_h[i][j]/foci_size, linewidth=0.5,
                                   edgecolors='k', facecolors='none', alpha=0.5,
                                   rasterized=False)

    # axis and figure formatting options
    ax[0].set_xlabel('time [hour]')
    ax[0].set_xlim(0, None)
    ax[0].set_ylabel('length [um]')
    ax[0].set_ylim(0, None)

    if plot_flconc:
        ax_fl.set_ylabel('fluorescence concentration [AU]')
        ax_fl.set_ylim([0, None])
        ax[0].spines['top'].set_visible(False)
        ax_fl.spines['top'].set_visible(False)
    else:
        sns.despine()

    plt.tight_layout()

    return fig, ax

def plot_saw_tooth_foci(Cells, fl_plane='c2', alt_time='birth', time_int=1, fl_int=1, pxl2um=1.0, y_adj_px=3, scat_s=None, foci_size=100, xlims=None):
    '''Plot a cell lineage with profile information. Assumes you want to plot the foci.

    Parameters
    ----------
    Cells : dict of Cell objects
        All the cells should come from a single peak.
    fl_plane : str
        Plane from which to get florescent data
    alt_time : None, 'birth', or float
        Time by which to shift X axis. If 'birth', time will be shited to start with 0.
    time_int : int or float
        Used to adjust the X axis to plot in hours
    fl_int : int
        Used to plot the florescence at the correct interval. Interval is relative to time interval, i.e., every time is 1, every other time is 2, etc.
    plx2um : float
        Conversion factor between pixels and microns.
    y_adj_px : int
        Y displacement for fluorescent profile information.
    scat_s : int or None
        Size to plot y fluorescent profile information. If None will calculate optimal size based on xlims.
    foci_size : int
        Factor by which to reduce foci size
    xlims : [int, int] or None
        Manually set xlims. If None then set automatically.
    '''

    peak_color = 'blue'

    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(8, 3))
    ax = [axes]
    ax_foci = ax[0].twinx() # for plotting foci counts

    # this is to map mothers to daugthers with lines
    last_div_time = None
    last_length = None

    # turn it into a list so it retains time order
    lin = [(cell_id, cell) for cell_id, cell in six.iteritems(Cells)]
    # sort cells by birth time for the hell of it.
    lin = sorted(lin, key=lambda x: x[1].birth_time)
    # align time to first birth or shift time
    if alt_time == None:
        alt_time = 0
    elif alt_time == 'birth':
        alt_time = lin[0][1].birth_time * time_int / 60.0

    # determine last time for xlims
    if xlims == None:
        last_time = (lin[-1][1].times[-1] + 10) * time_int / 60.0 - alt_time
        xlims = (0, last_time)
    elif xlims[1] == None: # just replace the last time
        last_time = (lin[-1][1].times[-1] + 10) * time_int / 60.0 - alt_time
        xlims[1] = last_time

    if scat_s == None:
        # adjust scatter marker size so colors touch but do not overlap
        # uses size of figure in inches, with the dpi (ppi) to convert to points.
        # scatter marker size is points squared.
        bbox = ax[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        width, height = np.float(bbox.width), np.float(bbox.height)
        scat_s = (((fig.dpi * width) / ((xlims[1] - xlims[0]) * 60.0 / time_int) * fl_int))**2

    # Choose colormap. Need to add alpha to color map and normalization
    # green/c2
    # max_c2_int = 0
    # min_c2_int = float('inf')
    # for cell_id, cell in lin:
    #     for profile_t in getattr(cell, 'fl_profiles_' + fl_plane):
    #         if max(profile_t) > max_c2_int:
    #             max_c2_int = max(profile_t)
    #         if min(profile_t) < min_c2_int:
    #             min_c2_int = min(profile_t)
    # cmap_c2 = plt.cm.Greens
    # color_norm_c2 = mpl.colors.Normalize(vmin=min_c2_int, vmax=max_c2_int)

    for cell_id, cell in lin:
        # plot cell length and division lines
        ax[0].plot(np.array(cell.times_w_div) * time_int / 60.0 - alt_time, cell.lengths_w_div,
                    color=peak_color, lw=0.5, alpha=1)
        # draw a connecting lines betwee mother and daughter
        if cell.birth_time == last_div_time:
            ax[0].plot([last_div_time * time_int / 60.0 - alt_time,
                         cell.birth_time * time_int / 60.0 - alt_time],
                        [last_length, cell.sb],
                        color=peak_color, lw=0.5)
        # record the last division time and length for next time
        last_div_time = cell.division_time
        # save the last length to check for division
        last_length = cell.sd

        # plot fluorescence on every time point for which it exists.
        # determine coloring on a cell by cell basis
        max_c2_int = 0
        min_c2_int = float('inf')
        for profile_t in getattr(cell, 'fl_profiles_' + fl_plane):
            if max(profile_t) > max_c2_int:
                max_c2_int = max(profile_t)
            if min(profile_t) < min_c2_int:
                min_c2_int = min(profile_t)
        cmap_c2 = plt.cm.Greens
        color_norm_c2 = mpl.colors.Normalize(vmin=min_c2_int, vmax=max_c2_int)

        for i, t in enumerate(cell.times):
            if t % fl_int == 1 or fl_int == 1:
                nuc_x = np.ones(len(getattr(cell, 'fl_profiles_' + fl_plane)[i])) * t * time_int / 60.0 - alt_time
                nuc_y = (np.arange(0, len(getattr(cell, 'fl_profiles_' + fl_plane)[i])) - y_adj_px) * pxl2um
                nuc_z = getattr(cell, 'fl_profiles_' + fl_plane)[i]
                ax[0].scatter(nuc_x, nuc_y, c=nuc_z, cmap=cmap_c2,
                               marker='s', s=scat_s, norm=color_norm_c2,
                               rasterized=False)

        # plot foci
        foci_counts = np.zeros_like(cell.times)
        foci_total_h = np.zeros_like(cell.times).astype(np.float)
        for i, t in enumerate(cell.times):
            for j, foci_y in enumerate(cell.disp_l[i]):
                foci_y_pos = (foci_y + cell.lengths[i]/2) * pxl2um
                ax[0].scatter(t * time_int / 60.0 - alt_time, foci_y_pos,
                               s=cell.foci_h[i][j]/foci_size, linewidth=0.5,
                               edgecolors='k', facecolors='none', alpha=0.5,
                               rasterized=False)
            foci_counts[i] = len(cell.disp_l[i])
            foci_total_h[i] = sum(cell.foci_h[i])

        # plot foci counts
        # pull out time point and data when there is foci data
        # foci_data = [(t, foci_total_h[i]) for i, t in enumerate(cell.times) if (t - 1) % fl_int == 0]
        foci_data = [(t, foci_counts[i]) for i, t in enumerate(cell.times) if (t - 1) % fl_int == 0]
        foci_times, foci_info = zip(*foci_data)
        ax_foci.plot(np.array(foci_times) * time_int / 60.0 - alt_time, foci_info,
                   color='green', lw=1, ls='--', alpha=0.75)

    # axis and figure formatting options
    ax[0].set_xlabel('time [hour]')
    ax[0].set_xlim(0, None)
    ax[0].set_ylabel('length [um]')
    ax[0].set_ylim(0, None)

    # foci counts multiples of 2 line
    ax_foci.axhline(2, color='green', lw=0.5, ls='-', alpha=0.5)
    ax_foci.axhline(4, color='green', lw=0.5, ls='-', alpha=0.5)
    ax_foci.axhline(8, color='green', lw=0.5, ls='-', alpha=0.5)
    ax_foci.axhline(16, color='green', lw=0.5, ls='-', alpha=0.5)

    # ax_foci.set_ylabel('foci total height')
    ax_foci.set_yscale('log', basey=2)
    ax_foci.set_ylabel('foci counts')
    yticks = [1, 2, 4, 8]#[2, 4, 8, 16]
    ax_foci.set_ylim([1, yticks[-1]+1])
    ax_foci.set_yticks(yticks)
    ax_foci.set_yticklabels([str(l) for l in yticks])
    ax[0].spines['top'].set_visible(False)
    ax_foci.spines['top'].set_visible(False)

    plt.tight_layout()

    return fig, ax

def plot_channel_traces(Cells, time_int=1.0, fl_plane='c2', fl_int=1.0, plot_fl=False, plot_foci=False, plot_pole=False, pxl2um=1.0, xlims=None, foci_size=100):
    '''Plot a cell lineage with profile information. Plots cells at their Y location in the growth channel.

    Parameters
    ----------
    Cells : dict of Cell objects
        All the cells should come from a single peak.
    time_int : int or float
        Used to adjust the X axis to plot in hours
    fl_plane : str
        Plane from which to get florescent data
    plot_fl : boolean
        Flag to plot florescent line profile.
    plot_foci : boolean
        Flag to plot foci or not.
    plot_pole : boolean
        If true, plot different colors for cells with different pole ages.
    plx2um : float
        Conversion factor between pixels and microns.
    xlims : [float, float]
        Manually set xlims. If None then set automatically.
    '''

    time_int = float(time_int)
    fl_int = float(fl_int)

    y_adj_px = 3 # number of pixels to adjust down y profile
    color = 'b' # overwritten if plot_pole == True

    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(8, 3))
    ax = [axes]

    # turn it into a list to fidn first time
    lin = [(cell_id, cell) for cell_id, cell in six.iteritems(Cells)]
    lin = sorted(lin, key=lambda x: x[1].birth_time)

    # align time to first birth or shift time
    # alt_time = 0
    alt_time = lin[0][1].birth_time * time_int / 60.0
#     alt_time = shift_t * time_int

    # determine last time for xlims
    if xlims == None or xlims[1] == None:
        last_time = (lin[-1][1].times[-1] + 10) * time_int / 60.0 - alt_time
        xlims = (0, last_time)

    # adjust scatter marker size so colors touch but do not overlap
    # uses size of figure in inches, with the dpi (ppi) to convert to points.
    # scatter marker size is points squared.
    bbox = ax[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = np.float(bbox.width), np.float(bbox.height)
    # print(fig.dpi, width, xlims[1], xlims[0],  time_int)
    # print(((fig.dpi * width) / ((xlims[1] - xlims[0]) * 60.0 / time_int)))
    # print((fig.dpi * width) / ((xlims[1] - xlims[0]) * 60.0))
    # print((fig.dpi * width) / ((xlims[1] - xlims[0]) * 60.0 / time_int)**2)
    scat_s = (((fig.dpi * width) / ((xlims[1] - xlims[0]) * 60.0 / time_int) * fl_int))**2
    # print(time_int)
    # print(scat_s)

    # Choose colormap. Need to add alpha to color map and normalization
    # green/c2
    if plot_fl:
        max_c2_int = 0
        min_c2_int = float('inf')
        for cell_id, cell in lin:
            for profile_t in getattr(cell, 'fl_profiles_' + fl_plane):
                if max(profile_t) > max_c2_int:
                    max_c2_int = max(profile_t)
                if min(profile_t) < min_c2_int:
                    min_c2_int = min(profile_t)
        cmap_c2 = plt.cm.Greens
        color_norm_c2 = mpl.colors.Normalize(vmin=min_c2_int, vmax=max_c2_int)

    for cell_id, cell in six.iteritems(Cells):

        # if this is a complete cell plot till division with a line at the end
        cell_times = np.array(cell.times) * time_int / 60.0 - alt_time
        cell_yposs = np.array([y for y, x in cell.centroids]) * pxl2um
        cell_halflengths = np.array(cell.lengths) / 2.0 * pxl2um
        ytop = cell_yposs + cell_halflengths
        ybot = cell_yposs - cell_halflengths

        if plot_pole:
            if cell.poleage:
                color_choices = sns.hls_palette(4)
                if cell.poleage == (1000, 0):
                    color = color_choices[0]
                elif cell.poleage == (0, 1) and cell.birth_label <= 2:
                    color = color_choices[1]
                elif cell.poleage == (1, 0) and cell.birth_label <= 3:
                    color = color_choices[2]
                elif cell.poleage == (0, 2):
                    color = color_choices[3]
                # elif cell.poleage == (2, 0):
                #     color = color_choices[4]
                else:
                    color = 'k'
            elif cell.poleage == None:
                    color = 'k'

        # plot two lines for top and bottom of cell
        ax[0].plot(cell_times, ybot, cell_times, ytop,
                   color=color, alpha=0.75, lw=1)
#         ax[0].fill_between(cell_times, ybot, ytop,
#                            lw=0.5, alpha=1)

        # plot lines for birth and division
        ax[0].plot([cell_times[0], cell_times[0]], [ybot[0], ytop[0]],
                      color=color, alpha=0.75, lw=1)
        ax[0].plot([cell_times[-1], cell_times[-1]], [ybot[-1], ytop[-1]],
                      color=color, alpha=0.75, lw=1)

        # plot fluorescence line profile
        if plot_fl:
            for i, t in enumerate(cell_times):
                if cell.times[i] % fl_int == 1:
                    fl_x = np.ones(len(getattr(cell, 'fl_profiles_' + fl_plane)[i])) * t # times
                    fl_ymin = cell_yposs[i] - (len(getattr(cell, 'fl_profiles_' + fl_plane)[i])/2 * pxl2um)
                    fl_ymax = fl_ymin + (len(getattr(cell, 'fl_profiles_' + fl_plane)[i]) * pxl2um)
                    fl_y = np.linspace(fl_ymin, fl_ymax, len(getattr(cell, 'fl_profiles_' + fl_plane)[i]))
                    fl_z = getattr(cell, 'fl_profiles_' + fl_plane)[i]
                    ax[0].scatter(fl_x, fl_y, c=fl_z, cmap=cmap_c2,
                                  marker='s', s=scat_s, norm=color_norm_c2,
                                  rasterized=True)

        # plot foci
        if plot_foci:
            for i, t in enumerate(cell_times):
                if cell.times[i] % fl_int == 1:
                    for j, foci_y in enumerate(cell.disp_l[i]):
                        foci_y_pos = cell_yposs[i] + (foci_y * pxl2um)
                        ax[0].scatter(t, foci_y_pos,
                                       s=cell.foci_h[i][j]/foci_size, linewidth=0.5,
                                       edgecolors='k', facecolors='none', alpha=0.5,
                                       rasterized=False)

    ax[0].set_xlabel('time [hours]')
    ax[0].set_xlim(xlims)
    ax[0].set_ylabel('position [$\mu$m]')
    ax[0].set_ylim([0, None])
#     ax[0].set_yticklabels([0,2,4,6,8,10])
    sns.despine()
    plt.tight_layout()

    return fig, ax

def plot_lineage_images(Cells, fov_id, peak_id, Cells2=None, bgcolor='c1', fgcolor='seg', plot_tracks=True, trim_time=False, time_set=(0,100), t_adj=1):
    '''
    Plot linages over images across time points for one FOV/peak.
    Parameters
    ----------
    bgcolor : Designation of background to use. Subtracted images look best if you have them.
    fgcolor : Designation of foreground to use. This should be a segmented image.
    Cells2 : second set of linages to overlay. Useful for comparing lineage output.
    plot_tracks : bool
        If to plot cell traces or not.
    t_adj : int
        adjust time indexing for differences between t index of image and image number
    '''

    # filter cells
    Cells = find_cells_of_fov_and_peak(Cells, fov_id, peak_id)

    # load subtracted and segmented data
    image_data_bg = mm3.load_stack(fov_id, peak_id, color=bgcolor)

    if fgcolor:
        image_data_seg = mm3.load_stack(fov_id, peak_id, color=fgcolor)

    if trim_time:
        image_data_bg = image_data_bg[time_set[0]:time_set[1]]
        if fgcolor:
            image_data_seg = image_data_seg[time_set[0]:time_set[1]]

    n_imgs = image_data_bg.shape[0]
    image_indicies = range(n_imgs)

    if fgcolor:
        # calculate the regions across the segmented images
        regions_by_time = [regionprops(timepoint) for timepoint in image_data_seg]

        # Color map for good label colors
        vmin = 0.1 # values under this color go to black
        vmax = 100 # max y value
        cmap = mpl.colors.ListedColormap(sns.husl_palette(vmax, h=0.5, l=.8, s=1))
        cmap.set_under(color='black')

    # Trying to get the image size down
    figxsize = image_data_bg.shape[2] * n_imgs / 100.0
    figysize = image_data_bg.shape[1] / 100.0

    # plot the images in a series
    fig, axes = plt.subplots(ncols=n_imgs, nrows=1,
                             figsize=(figxsize, figysize),
                             facecolor='black', edgecolor='black')
    fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)
    transFigure = fig.transFigure.inverted()

    # change settings for each axis
    ax = axes.flat # same as axes.ravel()
    for a in ax:
        a.set_axis_off()
        a.set_aspect('equal')
        ttl = a.title
        ttl.set_position([0.5, 0.05])

    for i in image_indicies:
        ax[i].imshow(image_data_bg[i], cmap=plt.cm.gray, aspect='equal')

        if fgcolor:
            # make a new version of the segmented image where the
            # regions are relabeled by their y centroid position.
            # scale it so it falls within 100.
            seg_relabeled = image_data_seg[i].copy()
            for region in regions_by_time[i]:
                rescaled_color_index = region.centroid[0]/image_data_seg.shape[1] * vmax
                seg_relabeled[seg_relabeled == region.label] = rescaled_color_index
            ax[i].imshow(seg_relabeled, cmap=cmap, alpha=0.55, vmin=vmin, vmax=vmax)

        ax[i].set_title(str(i + t_adj), color='white')

    # save just the segmented images
    # lin_dir = params['experiment_directory'] + params['analysis_directory'] + 'lineages/'
    # if not os.path.exists(lin_dir):
    #     os.makedirs(lin_dir)
    # lin_filename = params['experiment_name'] + '_xy%03d_p%04d_nolin.png' % (fov_id, peak_id)
    # lin_filepath = lin_dir + lin_filename
    # fig.savefig(lin_filepath, dpi=75)
    # plt.close()

    # Annotate each cell with information
    if plot_tracks:
        print('plotting traces')
        for cell_id in Cells:
            for n, t in enumerate(Cells[cell_id].times):
                t -= t_adj # adjust for special indexing

                # don't look at time points out of the interval
                if trim_time:
                    if t < time_set[0] or t >= time_set[1]-1:
                        break

                x = Cells[cell_id].centroids[n][1]
                y = Cells[cell_id].centroids[n][0]

                # add a circle at the centroid for every point in this cell's life
                circle = mpatches.Circle(xy=(x, y), radius=2, color='white', lw=0, alpha=0.5)
                ax[t].add_patch(circle)

                # draw connecting lines between the centroids of cells in same lineage
                try:
                    if n < len(Cells[cell_id].times) - 1:
                        # coordinates of the next centroid
                        x_next = Cells[cell_id].centroids[n+1][1]
                        y_next = Cells[cell_id].centroids[n+1][0]
                        t_next = Cells[cell_id].times[n+1] - t_adj # adjust for special indexing

                        # get coordinates for the whole figure
                        coord1 = transFigure.transform(ax[t].transData.transform([x, y]))
                        coord2 = transFigure.transform(ax[t_next].transData.transform([x_next, y_next]))

                        # create line
                        line = mpl.lines.Line2D((coord1[0], coord2[0]), (coord1[1], coord2[1]),
                                                transform=fig.transFigure,
                                                color='white', lw=1, alpha=0.5)

                        # add it to plot
                        fig.lines.append(line)
                except:
                    pass


                # draw connecting between mother and daughters
                try:
                    if n == len(Cells[cell_id].times)-1 and Cells[cell_id].daughters:
                        # daughter ids
                        d1_id = Cells[cell_id].daughters[0]
                        d2_id = Cells[cell_id].daughters[1]

                        # both daughters should have been born at the same time.
                        t_next = Cells[d1_id].times[0] - t_adj

                        # coordinates of the two daughters
                        x_d1 = Cells[d1_id].centroids[0][1]
                        y_d1 = Cells[d1_id].centroids[0][0]
                        x_d2 = Cells[d2_id].centroids[0][1]
                        y_d2 = Cells[d2_id].centroids[0][0]

                        # get coordinates for the whole figure
                        coord1 = transFigure.transform(ax[t].transData.transform([x, y]))
                        coordd1 = transFigure.transform(ax[t_next].transData.transform([x_d1, y_d1]))
                        coordd2 = transFigure.transform(ax[t_next].transData.transform([x_d2, y_d2]))

                        # create line and add it to plot for both
                        for coord in [coordd1, coordd2]:
                            line = mpl.lines.Line2D((coord1[0],coord[0]),(coord1[1],coord[1]),
                                                    transform=fig.transFigure,
                                                    color='white', lw=1, alpha=0.5, ls='dashed')
                            # add it to plot
                            fig.lines.append(line)
                except:
                    pass

        # this is for plotting the traces from a second set of cells
        if Cells2 and plot_tracks:
            Cells2 = find_cells_of_fov_and_peak(Cells2, fov_id, peak_id)
            for cell_id in Cells2:
                for n, t in enumerate(Cells2[cell_id].times):
                    t -= t_adj

                    # don't look at time points out of the interval
                    if trim_time:
                        if t < time_set[0] or t >= time_set[1]-1:
                            break

                    x = Cells2[cell_id].centroids[n][1]
                    y = Cells2[cell_id].centroids[n][0]

                    # add a circle at the centroid for every point in this cell's life
                    circle = mpatches.Circle(xy=(x, y), radius=2, color='yellow', lw=0, alpha=0.25)
                    ax[t].add_patch(circle)

                    # draw connecting lines between the centroids of cells in same lineage
                    try:
                        if n < len(Cells2[cell_id].times) - 1:
                            # coordinates of the next centroid
                            x_next = Cells2[cell_id].centroids[n+1][1]
                            y_next = Cells2[cell_id].centroids[n+1][0]
                            t_next = Cells2[cell_id].times[n+1] - t_adj

                            # get coordinates for the whole figure
                            coord1 = transFigure.transform(ax[t].transData.transform([x, y]))
                            coord2 = transFigure.transform(ax[t_next].transData.transform([x_next, y_next]))

                            # create line
                            line = mpl.lines.Line2D((coord1[0],coord2[0]),(coord1[1],coord2[1]),
                                                    transform=fig.transFigure,
                                                    color='yellow', lw=1, alpha=0.25)

                            # add it to plot
                            fig.lines.append(line)
                    except:
                        pass

                    # draw connecting between mother and daughters
                    try:
                        if n == len(Cells2[cell_id].times)-1 and Cells2[cell_id].daughters:
                            # daughter ids
                            d1_id = Cells2[cell_id].daughters[0]
                            d2_id = Cells2[cell_id].daughters[1]

                            # both daughters should have been born at the same time.
                            t_next = Cells2[d1_id].times[0] - t_adj

                            # coordinates of the two daughters
                            x_d1 = Cells2[d1_id].centroids[0][1]
                            y_d1 = Cells2[d1_id].centroids[0][0]
                            x_d2 = Cells2[d2_id].centroids[0][1]
                            y_d2 = Cells2[d2_id].centroids[0][0]

                            # get coordinates for the whole figure
                            coord1 = transFigure.transform(ax[t].transData.transform([x, y]))
                            coordd1 = transFigure.transform(ax[t_next].transData.transform([x_d1, y_d1]))
                            coordd2 = transFigure.transform(ax[t_next].transData.transform([x_d2, y_d2]))

                            # create line and add it to plot for both
                            for coord in [coordd1, coordd2]:
                                line = mpl.lines.Line2D((coord1[0],coord[0]),(coord1[1],coord[1]),
                                                        transform=fig.transFigure,
                                                        color='yellow', lw=1, alpha=0.25, ls='dashed')
                                # add it to plot
                                fig.lines.append(line)
                    except:
                        pass

            # this is for putting cell id on first time cell appears and when it divides
            # this is broken, need to translate coordinates to correct location on figure.
            # if n == 0 or n == len(Cells[cell_id].times)-1:
            #     print(x/100.0, y/100.0, cell_id[9:])
            #     ax[t].text(x/100.0, y/100.0, cell_id[9:], color='red', size=10, ha='center', va='center')

    return fig, ax

### Fitting functions ##############################################################################
def produce_cell_fit(Cell):
    '''
    Given a cell object, produce a fit for its elongation.
    Use log of length and linear regression. However,
    return what the actual lengths would be.
    '''

    x = Cell.times_w_div - Cell.times_w_div[0] # time points
    y = np.log(Cell.lengths_w_div) # log(lengths)

    slope, intercept, r_value, p_value, std_err = sps.linregress(x, y)
    r_squared = r_value**2

    y_fit = x * slope + intercept
    y_fit = np.exp(y_fit)

    # print(Cell.elong_rate, slope)
    # print(y, y_fit, r_squared, intercept)

    return y_fit, r_squared

def produce_cell_bilin_fit(Cell):
    '''
    Use Guillaume's code to produce a bilinear fit
    '''

    # Get X and Y. X is time, Y is length
    X = np.array(Cell.times_w_div, dtype=np.float_)
    Y = np.log(Cell.lengths_w_div)

    ## change origin of times
    X_t0 = X[0]
    X = X-X_t0

    # make bilinear fit
    p_init = bilinear_init(X, Y)
    par = fit_xy(X, Y, p_init=p_init, funcfit_f=bilinear_f, funcfit_df=bilinear_df)
    Z = np.array([bilinear_f(par, xi) for xi in X])
    chi_bilin = np.mean((Y - Z)**2)
    r2 = coefficient_determination_r2(Y, Z)
    r_bilin = np.sqrt(r2)

    t_shift = par[3] + X_t0

    # convert back for plotting
    y_fit = np.exp(Z)

    # determine the length at the shift up time for plotting
    len_at_shift = np.exp(bilinear_f(par, par[3]))

    return y_fit, r2, t_shift, len_at_shift

class FitRes:
    """
    Object used to fit a data set to a particular function.
    Input:
        o x: x coordinates of the input data set.
        o y: y coordinates of the input data set.
        o s: standard deviations for the y values.
        o funcfit: fitting function.
    """
    def __init__(self, x, y, funcfit_f, funcfit_df, yerr=None):
        self.x = np.array(x)
        self.y = np.array(y)
        self.funcfit_f = funcfit_f
        self.funcfit_df = funcfit_df
        if (yerr == None):
            self.yerr = np.ones(self.x.shape)
        else:
            self.yerr = yerr

    def residual_f(self, par):
        """
        Return the vector of residuals.
        """
        fx = np.array([self.funcfit_f(par,xi) for xi in self.x])
        return (fx-self.y)/self.yerr

    def residual_df(self, par):
        dfx = np.array([np.array(self.funcfit_df(par,xi))/si for (xi,si) in zip(self.x,self.yerr)])
        return dfx

def coefficient_determination_r2(y,yfit):
    """
    Determine the coefficient of determination (r^2) obtained for the
    linear fit yfit to the input data y.
    """
    ymean = np.mean(y)
    s_res = np.sum((y-yfit)**2)
    s_tot = np.sum((y-ymean)**2)
    r2 = 1.0 - s_res / s_tot
    return r2

def fit_xy(x, y, p_init, funcfit_f, funcfit_df=None, least_squares_args={'loss':'cauchy'},):
    """
    1) Extract x- (y-) coordinates from attribute key_x (key_y).
    2) Fit the resulting data set according to a model function funcfit_f
    """
    # define FitRes object -- define the residuals
    fitres = FitRes(x, y, funcfit_f=funcfit_f, funcfit_df=funcfit_df)

    # perform the least_square minimization
    try:
        if (funcfit_df == None):
            res = least_squares(x0=p_init, fun=fitres.residual_f, **least_squares_args)
        else:
            res = least_squares(x0=p_init, fun=fitres.residual_f,
                                jac=fitres.residual_df, **least_squares_args)
        par=res.x

    except ValueError as e:
        print(e)
   #      sys.exit(1)
   # print res
    return par

def bilinear_f(par,xi):
    """
    f(x) =  a + b (x-x0), if x <= x0
            a + c (x-x0), otherwise
    """
    a = par[0]
    b = par[1]
    c = par[2]
    x0 = par[3]

    if not (xi > x0):
        return a + b*(xi-x0)
    else:
        return a + c*(xi-x0)

def bilinear_df(par,xi):
    """
    f(x) =  a + b (x-x0), if x <= x0
            a + c (x-x0), otherwise
    """
    a = par[0]
    b = par[1]
    c = par[2]
    x0 = par[3]
    if not (xi > x0):
        return np.array([ 1.0, xi-x0, 0, -b])
    else:
        return np.array([ 1.0, 0, xi-x0, -c])

def bilinear_init(x, y):
    x = np.array(x)
    y = np.array(y)
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]

    # main secant
    xN = x[-1]
    x0 = x[0]
    yN = y[-1]
    y0 = y[0]
    vdir = np.array([1.0, (yN-y0)/(xN-x0)])
    # distances to main secant
    hs = -vdir[1]*(x-x0) + vdir[0]*(y-y0)

    # maximum distance
    imid = np.argmin(hs)
    xmid = x[imid]
    ymid = y[imid]

    # return bilinear init
    if ( xmid == x0):
        return np.array([ymid,0.0,(yN-ymid)/(xN-xmid),xmid])
    elif ( xmid == xN):
        return np.array([ymid,(ymid-y0)/(xmid-x0),0.0,xmid])
    else:
        return np.array([ymid,(ymid-y0)/(xmid-x0),(yN-ymid)/(xN-xmid),xmid])

### Random tools ###################################################################################
def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

def binned_stat(x, y, bin_edges=10, binmin=None):
    '''Calculate binned mean on X. Returns plotting variables

    bin_edges : int or list/array
        If int, this is the number of bins. If it is a list it defines the bin edges.

    '''

    bin_result = sps.binned_statistic(x, y,
                                      statistic='mean', bins=bin_edges)
    bin_means, bin_edges, bin_n = bin_result
    bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2

    # remove bins with not enought datapoints
    if binmin:
        bin_count_results = sps.binned_statistic(x, y,
                                                 statistic='count', bins=bin_edges)
        bin_counts, null1, null2 = bin_count_results
        delete_me = []
        for i, points in enumerate(bin_counts):
            if points < binmin:
                delete_me.append(i)
        delete_me = tuple(delete_me)
        bin_centers = np.delete(bin_centers, delete_me)
        bin_means = np.delete(bin_means, delete_me)

    # only keep locations where there is data
    bin_centers = bin_centers[~np.isnan(bin_means)]
    bin_means = bin_means[~np.isnan(bin_means)]

    return bin_centers, bin_means
