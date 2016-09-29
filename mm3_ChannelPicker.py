#!/usr/bin/python
from __future__ import print_function
def warning(*objs):
    print(time.strftime("%H:%M:%S Warning:", time.localtime()), *objs, file=sys.stderr)
def information(*objs):
    print(time.strftime("%H:%M:%S", time.localtime()), *objs, file=sys.stdout)

# import modules
import sys
import os
import time
import inspect
import getopt
import yaml
import traceback
# import h5py
import fnmatch
import gc
import random
from pprint import pprint # for human readable file output
try:
    import cPickle as pickle
except:
    import pickle
import multiprocessing
from multiprocessing import Pool #, Manager
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from skimage.feature import match_template # match template used for correlation
from skimage.exposure import rescale_intensity
from sklearn.cluster import KMeans

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

import tifffile as tiff
import mm3_helpers as mm3

### functions
# funtion which makes the UI plot
def fov_choose_channels_UI(fov_file, fov_xcorrs):
    '''Creates a plot with the channels with guesses for empties and full channels,
    and requires the user to choose which channels to use for analysis and which to
    average for empties and subtraction.

    Parameters
    fov_file : str
        file name of the hdf5 file name in originals
    fov_xcorrs : dictionary
        dictionary for cross correlation values for all fovs.

    Returns
    bgdr_peaks : list
        list of peak id's (int) of channels to be used for subtraction
    spec_file_...pkl : pickle file
        saves the lists cell_peaks, bgrd_peaks, and drop_peaks to a pkl file

    '''
    fov = int(fov_file.split("original_")[1].split(".hdf5")[0])

    information("Starting channel picking for FOV %03d." % fov)
    crosscorr_peaks = fov_xcorrs[fov] # get xcorr information for the fov

    # set up arrays for which peaks are kept and which are ditched or used for subtraction
    cell_peaks = [] # cell_peaks will be a list of the channels with cells in them
    bgrd_peaks = [] # bgrd will be a list of peaks to be used for background subtraction
    drop_peaks = [] # not used but could be helpful, not gonna mess with format

    # open the h5f file "original_%02d.hdf5" for the current fov
    # h5f = h5py.File(experiment_directory + analysis_directory + 'originals/' + fov_file, 'r', libver='latest', swmr=True)
    h5f = h5py.File(experiment_directory + analysis_directory + 'originals/' + fov_file, 'r', libver='earliest')

    # calculate histogram of cross correlations down length of channel
    hmaxvals = np.array([get_hist_peak(p[1]) for p in crosscorr_peaks])

    # use k means clustering for full and empty channel
    k_means = KMeans(init = 'k-means++', n_clusters = 2, n_init = 10)
    k_means.fit(np.expand_dims(hmaxvals, 1))
    empty_label = np.argmax(k_means.cluster_centers_) # empties are the higher corr group
    peak_labels = k_means.labels_ # get them labels for all peaks

    # define functions here so they have access to variables
    # for UI. remove and add "cell holding" channels
    def onclick_cells(event):
        channel_id = int(event.inaxes.get_title())
        if channel_id in cell_peaks:
            cell_peaks.remove(channel_id)
            information("Removed " + str(channel_id))
        else:
            cell_peaks.append(channel_id)
            information("Added " + str(channel_id) + " as cell-containing channel")
        return cell_peaks

    # for UI. define function for adding empty channels "peaks" to be used for BG subtraction
    def onclick_background(event):
        # add the channel to the list of channels to use for bg subtraction
        # note that the title of the subplot must be the same name as that of the channel "peak"
        bgrd_peaks.append(int(event.inaxes.get_title()))
        # tell the user what they did
        information(str(bgrd_peaks[-1]))

    # set up figure for user assited choosing
    fig = plt.figure(figsize=(20,13))
    ax = [] # for axis handles

    # plot the peaks column by column
    for i_peak, peak in enumerate(crosscorr_peaks):
        # append an axis handle to ax list while adding a subplot to the figure which has a
        # column for each peak and 3 rows

        # plot the first image in each channel
        ax.append(fig.add_subplot(3, len(crosscorr_peaks), i_peak+1))
        # plot first image (phase) of channels
        ax[-1].imshow(rescale_intensity(h5f[u'channel_%04d' % peak[0]][0,:,:,0]),
                      cmap=plt.cm.gray, interpolation = 'nearest') # none not suppored on osx
        ax = format_channel_plot(ax, peak[0])
        if i_peak == 0:
            ax[-1].set_ylabel("time index = 0")

        # plot a middle image of each channel with highlighting for empty/full
        ax.append(fig.add_subplot(3, len(crosscorr_peaks), i_peak+1+len(crosscorr_peaks)))
        # use the most recent image for second row
        show_index = min(500, len(h5f[u'channel_%04d' % peak[0]]) - 1)
        if peak_labels[i_peak] == empty_label:
            is_empty = True
        else:
            is_empty = False
            cell_peaks.append(peak[0])

        # plot images in second row
        ax[-1].imshow(rescale_intensity(h5f[u'channel_%04d' % peak[0]][show_index,:,:,0]),
                      cmap=plt.cm.gray, interpolation='nearest')
        # color over
        ones_array = np.ones_like(h5f[u'channel_%04d' % peak[0]][show_index,:,:,0])
        if is_empty:
            ax[-1].imshow(np.dstack((ones_array, ones_array * 0.1, ones_array * 0.1)),
                          alpha = 0.5)
        else:
            ax[-1].imshow(np.dstack((ones_array * 0.1, ones_array, ones_array * 0.1)),
                          alpha = 0.5)
        ax = format_channel_plot(ax, peak[0])
        if i_peak == 0:
            ax[-1].set_ylabel("time index = %d" % show_index)

        # finall plot the cross correlations
        ax.append(fig.add_subplot(3, len(crosscorr_peaks), i_peak+1+2*len(crosscorr_peaks)))

        # creat a histogram from the list of correaltion values
        cc_hist = np.histogram(peak[1], bins = 200, range = ((0,1)), normed=True)

        # plot it up
        ax[-1].plot(cc_hist[0], cc_hist[1][1:])
        ax[-1].set_ylim((0.8,1))
        ax[-1].get_xaxis().set_ticks([])
        if not i_peak == 0:
            ax[-1].get_yaxis().set_ticks([])
        else:
            ax[-1].set_ylabel("cross correlation")
        ax[-1].set_title(str(peak[0]), fontsize = 8)

    h5f.close()

    # show the plot finally
    fig.suptitle("FOV %d" % fov)
    plt.show(block=False)

    # enter user input
    # ask the user to correct cell/nocell calls
    cells_handler = fig.canvas.mpl_connect('button_press_event', onclick_cells)
    raw_input("Click cell-containing (i.e. green) channels to NOT analyze and red channel TO analyze, then press enter.\n") # raw input waits for enter key
    fig.canvas.mpl_disconnect(cells_handler)

    # ask the user to add the channels that look like good background channels
    # use the function onclick_background defined above
    cells_handler = fig.canvas.mpl_connect('button_press_event', onclick_background)
    raw_input("Click background (i.e. red) channels to use for subtraction, then press enter.\n")
    fig.canvas.mpl_disconnect(cells_handler)
    plt.close('all') # close the figure

    # inform the user that the program is writing the data to file
    information("Writing peak spec file for FOV %00d." % fov)
    # create specs/ if it doesn't exist
    if not os.path.exists(os.path.abspath(experiment_directory + analysis_directory +
                                          "specs/")):
        os.makedirs(os.path.abspath(experiment_directory + analysis_directory + "specs/"))
    with open(experiment_directory + analysis_directory + "specs/specs_%03d.pkl" % fov, 'wb') as out_params:
        pickle.dump((drop_peaks, cell_peaks, bgrd_peaks), out_params)

    return bgrd_peaks

# function for better formatting of channel plot
def format_channel_plot(ax, peak):
    '''Removes axis and puts peak as title from plot for channels'''
    ax[-1].get_xaxis().set_ticks([])
    ax[-1].get_yaxis().set_ticks([])
    ax[-1].set_title(str(peak), fontsize = 8)
    return ax

# use k-means clustering to try guessing the empty & full channels
def get_hist_peak(ccdata):
    cc_hist = np.histogram(ccdata, bins = 200, range = ((0,1)))
    return cc_hist[1][np.argmax(cc_hist[0])]

### For when this script is run from the terminal ##################################
if __name__ == "__main__":
    # hardcoded parameters
    load_crosscorrs = False

    # get switches and parameters
    try:
        opts, args = getopt.getopt(sys.argv[1:],"f:o:s:")
        # switches which may be overwritten
        specify_fovs = False
        user_spec_fovs = []
        start_with_fov = -1
        param_file = ""
    except getopt.GetoptError:
        warning('No arguments detected (-f -s -o).')

    for opt, arg in opts:
        if opt == '-o':
            try:
                specify_fovs = True
                for fov_to_proc in arg.split(","):
                    user_spec_fovs.append(int(fov_to_proc))
            except:
                warning("Couldn't convert argument to an integer:",arg)
                raise ValueError
        if opt == '-s':
            try:
                start_with_fov = int(arg)
            except:
                warning("Couldn't convert argument to an integer:",arg)
                raise ValueError
        if opt == '-f':
            param_file_path = arg # parameter file path

    # Load the project parameters file
    if len(param_file_path) == 0:
        raise ValueError("A parameter file must be specified (-f <filename>).")
    information('Loading experiment parameters.')
    with open(param_file_path, 'r') as param_file:
        p = yaml.safe_load(param_file) # load parameters into dictionary

    mm3.init_mm3_helpers(param_file_path) # initialized the helper library

    # set up how to manage cores for multiprocessing
    cpu_count = multiprocessing.cpu_count()
    if cpu_count == 32:
        num_analyzers = 20
    elif cpu_count == 8:
        num_analyzers = 14
    else:
        num_analyzers = cpu_count*2 - 2

    # assign shorthand directory names
    ana_dir = p['experiment_directory'] + p['analysis_directory']
    chnl_dir = p['experiment_directory'] + p['analysis_directory'] + 'channels/'

    # load channel masks
    try:
        with open(ana_dir + '/channel_masks.pkl', 'r') as cmask_file:
            channel_masks = pickle.load(cmask_file)
    except:
        warning('Could not load channel mask file.')
        raise ValueError

    # make list of FOVs to process (keys of channel_mask file)
    fov_id_list = sorted([fov_id for fov_id in channel_masks.keys()])

    information("Found %d FOVs to process." % len(fov_id_list))

    ### Cross correlations ########################################################################
    if load_crosscorrs: # load precalculate ones if indicated
        information('Loading precalculated cross-correlations.')

        with open(ana_dir + '/crosscorrs.pkl') as xcorrs_file:
            crosscorrs = pickle.load(xcorrs_file)

    else:
        # a nested dict to hold cross corrs per channel per fov.
        crosscorrs = {}

        # for each fov find cross correlations (sending to pull)
        for fov_id in fov_id_list:
            # skip fovs if they are not user specified
            if specify_fovs and not (fov_id in user_spec_fovs):
                continue
            if start_with_fov > 0 and fov_id < start_with_fov:
                continue

            information("Calculating cross-correlations for FOV %d." % fov_id)

            # nested dict keys are peak_ids and values are cross correlations
            crosscorrs[fov_id] = {}

            # initialize pool for analyzing image metadata
            pool = Pool(num_analyzers)

            # find all peak ids in the current FOV
            for peak_id in sorted(channel_masks[fov_id].keys()):
                # determine the channel file name and path
                channel_filename = p['experiment_name'] + '_xy%03d_p%04d.tif' % (fov_id, peak_id)
                channel_filepath = chnl_dir + channel_filename

                information("Calculating cross-correlations for peak %d." % peak_id)

                # linear loop
                crosscorrs[fov_id][peak_id] = mm3.channel_xcorr(channel_filepath)

                # multiprocessing verion
                crosscorrs[fov_id][peak_id] = pool.apply_async(mm3.channel_xcorr,
                                                               args=(channel_filepath))

            information('Waiting for cross correlation pool to finish for FOV %d.' % fov_id)

            pool.close() # tells the process nothing more will be added.
            pool.join() # blocks script until everything has been processed and workers exit



            information('Cross correlation pool finished for FOV %d.' % fov_id)

        # get results from the pool and put them in a dictionary
        for fn, result in analyzed_imgs.iteritems():
            if result.successful():
                analyzed_imgs[fn] = result.get() # put the metadata in the dict if it's good
            else:
                analyzed_imgs[fn] = False # put a false there if it's bad

        information('Got results from analyzed images.')

        # calculate guess on full or empty using k means clustering

            # # start up mutliprocessing
            # pool = Pool(num_analyers)
            # pool_result = pool.map_async(get_crosscorrs, peaks) # worker function
            # pool.close()
            #
            #
            # pool.close() # tells the process nothing more will be added.
            # pool.join() # blocks script until everything has been processed and workers exit


            # # store the data to crosscorr_peaks
            # crosscorr_peaks = pool_result.get()

            # # add key value pair to cross correlations
            # fov_xcorrs[fov] = crosscorr_peaks

            information("Finished cross-correlations for FOV %d." % fov_id)

        # write cross-correlations to pickle and text
        information("Writing cross correlations file.")
        with open(ana_dir+ "/crosscorrs.pkl", 'wb') as xcorrs_file:
            pickle.dump(crosscorrs, xcorrs_file)
        with open(ana_dir + "/crosscorrs.txt", 'w') as xcorrs_file:
            pprint(crosscorrs, stream=xcorrs_file)
    #
    #     information("Done calculating cross correlations. Starting channel picking.")
    #
    # raw_input('hit enter to continue...')
    #
    # ### User selection (channel picking) and averaging of empties #######################
    # # set up Multiprocessing
    # m = Manager()
    # q = m.Queue()
    # pool = Pool(num_procs)
    #
    # # dictionary for bgrd lists
    # bgrd_peaks_all = []
    #
    # # go through the fovs again, same as above
    # for fov_file in fov_file_list:
    #     # don't do channel picking or aveaging if user specifies not to
    #     fov = int(fov_file.split("original_")[1].split(".hdf5")[0])
    #     if specify_fovs and not (fov in user_spec_fovs):
    #         continue
    #     if start_with_fov > 0 and fov < start_with_fov:
    #         continue
    #
    #     # see if spec file exists, load if so, other wise calculate it
    #     try:
    #         with open(experiment_directory + analysis_directory + 'specs/specs_%03d.pkl' % fov, 'rb') as pkl_file:
    #             user_picks = pickle.load(pkl_file)
    #             bgrd_peaks = user_picks[2] # get the backgrounds out
    #             information('Loaded precalculated specs file for FOV %03d.' % fov)
    #     except:
    #         # launch UI and calcuate that stuff otherwise
    #         bgrd_peaks = fov_choose_channels_UI(fov_file, fov_xcorrs)
    #
    #     ### Average empty channels
    #     # either do it right after you do the UI selection
    #     pool_result = pool.apply_async(average_picked_empties, [(fov_file, bgrd_peaks),])
    #
    #     # or put information and do it all after
    #     #bgrd_peaks_all.append((fov_file, bgrd_peaks))
    #
    # #pool_result = pool.map_async(average_picked_empties, bgrd_peaks_all)
    # pool.close() # no more data after this loop has been gone through all the times
    #
    # # loop through and give updates on progress
    # while (True):
    #     # update on how much is finished
    #     size = q.qsize()
    #     information("Completed", size, "of", len(fov_file_list), "time averages...")
    #     if (pool_result.ready()): break # break out if everything is done
    #     time.sleep(10)
    #
    # if not pool_result.successful():
    #     warning("Processing pool not successful!")
    #     raise AttributeError
    # else:
    #     information("Finished averaging empties.")
