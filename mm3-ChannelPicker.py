#!/usr/bin/python -u
from __future__ import print_function
def warning(*objs):
    print(time.strftime("%H:%M:%S ChannelPicker Error:", time.localtime()), *objs, file=sys.stderr)
def information(*objs):
    print(time.strftime("%H:%M:%S ChannelPicker:", time.localtime()), *objs, file=sys.stdout)

# import modules
import sys
import os
import time
import inspect
import getopt
import yaml
import traceback
import h5py
import fnmatch
import gc
import random
try:
    import cPickle as pickle
except:
    import pickle
import multiprocessing
from multiprocessing import Pool, Manager
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

from subtraction_helpers import *
import tifffile as tiff

### functions
# calculate the cross correlation between images in a stack and use that to estimate full vs empty
def get_crosscorrs(peak):
    try:
        # load a sample of the initial images in the experiment for calculating xcorrs
        # h5f = h5py.File(experiment_directory + analysis_directory + 'originals/' + fov_file, 'r', libver='latest', swmr=True)
        h5f = h5py.File(experiment_directory + analysis_directory + 'originals/' + fov_file, 'r', libver='earliest')
        stack_shape = h5f[u'channel_%04d' % peak].shape
        # if the dataset is short, start at 1/3 of the way into the experiment through
        # the end of the experiment, otherwise start at index 100 and go through 300
        if stack_shape[0] < 300:
            index_start, index_end = (stack_shape[0] / 3, stack_shape[0] - 1)
        else:
            index_start, index_end = (100, 300)

        # use every other phase contrast image
        p_stack = h5f[u'channel_%04d' % peak][index_start:index_end:2,:,:,0]
        h5f.close()

        # copy the first image from the stack to which the cross corrleation will be calculated
        last_image = p_stack[0]
        crosscorr_values = [] # initialize correlation values list
        # loop over value strating 25% of the way into p_stack to the end of p_stack
        for i_index in range(len(p_stack)/4, len(p_stack)):
            # append the maximum value of correlation between the first image and the curent image
            # using the match_template function. Use this function because i know it will ultimately # invoke the FFTW version of the convolution, which is substantially faster than the
            # LINPACK version. ultimately this could be replaced by scipy.signal.correlate2d if
            # that is for some reason more appropriate
            crosscorr_values.append(np.amax(match_template(last_image, p_stack[i_index],
                                                           pad_input = True)))
        q.put(0)
        # return the peaks name (x pos at t_index=1) and the list of cross corr values
        return((peak, crosscorr_values))
    # error handling
    except:
        print (sys.exc_info()[1])
        print (traceback.print_tb(sys.exc_info()[2]))
        #raise BaseException()
        return((peak, [-1,]))

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

# average empty channels together from hdf5
def average_picked_empties(args):
    '''Takes the fov file name and the peak names of the designated empties,
    averages them and saves the image

    Parameters
    ags is a tuple like this: (fov_file, bgrd_peaks)
    fov_file : str
        file name of hdf5
    bgrd_peaks : list
        list of peaks to use for empties for this fov

    '''
    try:
        fov_file, bgrd_peaks = args

        fov = int(fov_file.split("_")[1].split(".")[0])
        information("Creating average empty channel for FOV %03d." % fov)

        # if there is not background data to work with for this FOV exit in error
        if len(bgrd_peaks) == 0:
            q.put(1)
            return 1

        # declare variables
        empty_composite = [] # list of empty channel images alligned to overlap
        paddings = [] # paddings is a list of the paddings used to allign the images
        reference_image = None # stores the index 0 image with edge padding
                               # instead of nan padding; edge padding improves the
                               # alignment by eliminating the cliff of "normal values"
                               # into np.nan values; this image is still stored as a nan-padded
                               # image in the empty_composite list

        # go over the list of background peaks
        for peak in bgrd_peaks:
            # load the images from the given fov_file
            # with h5py.File(experiment_directory + analysis_directory + 'originals/' + fov_file, 'r', libver='latest', swmr=True) as h5f:
            with h5py.File(experiment_directory + analysis_directory + 'originals/' + fov_file, 'r', libver='earliest') as h5f:
                start_i = len(h5f[u'channel_%04d' % peak]) / 4 # same as start in cross corr function
                final_i = len(h5f[u'channel_%04d' % peak]) - 1 # last image
                ch_ds = h5f[u'channel_%04d' % peak]
                images = np.zeros([final_i-start_i, ch_ds.shape[1], ch_ds.shape[2]])
                ch_ds.read_direct(images, np.s_[start_i:final_i, :, :, 0])

            # get a random subset of 40% of images in the peak
            images_random_subset = []
            for image in images[:300]:
                if random.random() < 0.4:
                    images_random_subset.append(image)
            del(images) # clear up memory

            # images should all be the same size, so this can probably go
            #max_x = np.max([image.shape[0] for image in images_random_subset])
            #max_y = np.max([image.shape[1] for image in images_random_subset])

            # Build a stack of overlapping images of empty channels by aligning and padding the images
            for image in images_random_subset:
                if reference_image is None:
                    h_im, w_im = image.shape # get the shape of the current image

                    # increase the frame size
                    h_full = h_im + 50
                    w_full = w_im + 50

                    # calculate the padding size needed to fit the orignial into the full frame
                    h_diff = h_full - h_im
                    w_diff = w_full - w_im

                    # pad the image with NaNs for stacking
                    im_padded = np.pad(image,
                                       ((h_diff/2, h_diff-(h_diff/2)),
                                        (w_diff/2, w_diff-(w_diff/2))),
                                       mode='constant',
                                       constant_values=np.nan)

                    # pad the image with edge values for alignment for the reference image
                    reference_image = np.pad(image,
                                             ((h_diff/2, h_diff-(h_diff/2)),
                                              (w_diff/2, w_diff-(w_diff/2))),
                                             mode='edge')

                    empty_composite.append(im_padded) # add this image to the list empty_composite
                else:
                    # use the match template function to find the overlap position of maximum corr
                    # Note the numbers that just the end is used to match
                    match_result = match_template(np.nan_to_num(reference_image[:200]),
                                                  np.nan_to_num(image[:175]))
                    # find the best correlation
                    y, x = np.unravel_index(np.argmax(match_result), match_result.shape)

                    # pad the image with np.nan
                    im_padded = np.pad(image,
                                       ((y, empty_composite[0].shape[0] - (y + image.shape[0])),
                                        (x, empty_composite[0].shape[1] - (x + image.shape[1]))),
                                       mode='constant',
                                       constant_values=np.nan)

                    empty_composite.append(im_padded)

        # stack the aligned data along axis 2
        empty_composite = np.dstack(empty_composite)

        # get a boolean mask of non-NaN positions
        nanmap = ~np.isnan(empty_composite)

        # sum up the total non-NaN values along the depth
        counts = nanmap.sum(2).astype(np.float16)

        # divide counts by it highest value
        counts /= np.amax(counts)

        # get a rectangle of the region where at least half the images have real data
        binary_core = counts > 0.5

        # get all rows/columns in the common region that are True
        poscols = np.any(binary_core, axis = 1) # column positions where true (any)
        posrows = np.any(binary_core, axis = 0) # row positions where true (any)

        # get the mix/max row/column for the binary_core
        min_row = np.amin(np.where(posrows)[0])
        max_row = np.amax(np.where(posrows)[0])
        min_col = np.amin(np.where(poscols)[0])
        max_col = np.amax(np.where(poscols)[0])

        # crop the composite to the common core
        empty_composite = empty_composite[min_col:max_col, min_row:max_row]

        # get a mean image along axis 2
        empty_mean = np.nanmean(empty_composite, axis = 2)

        # old version
        '''
        # strip out the rows and or columns of all false
        overlap_shape = trim_false_2d(overlap)
        # extract the shape of the stripped array
        overlap_shape = overlap_shape.shape
        # sum allong the list getting the mean value ignoring the NANs  ie  for [1,2,NAN] it does (1+2+NAN)/2=1.5 not (1+2+NAN)/3=1
        empty_mean = np.nanmean(np.asarray(empty_composite), axis = 0)
        # reshape the numpy array to include elements with data in more than half of the list memebers
        empty_mean = np.reshape(empty_mean[overlap], overlap_shape)
        # empty_mean is the mean of the empty channels
        '''

        # if the empties subdirectory doesn't exist, create it
        if not os.path.exists(os.path.abspath(experiment_directory + analysis_directory + "empties/")):
            os.makedirs(os.path.abspath(experiment_directory + analysis_directory + "empties/"))
        # save a tif image of the mena of the empty channel images for this FOV
        tiff.imsave(experiment_directory + analysis_directory + 'empties/fov_%03d_emptymean.tif' % fov, empty_mean.astype('uint16'))

        # put the return status in the queue
        q.put(0)
        # return a zero to signify a succesful exit status
        return 0
    # failure detail
    except:
        warning(sys.exc_info()[0])
        warning(sys.exc_info()[1])
        warning(traceback.print_tb(sys.exc_info()[2]))
        raise

### For when this script is run from the terminal ##################################
if __name__ == "__main__":
    # get user specified options
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
            # Load the project parameters file
            param_file = arg
            if len(param_file) == 0:
                raise ValueError("A parameter file must be specified (-f <filename>).")
            information('Loading parameters.')
            parameters_file = open(param_file, 'r')
            globals().update(yaml.load(parameters_file))
            information('Parameters loaded from %s' % param_file)

    cpu_count = multiprocessing.cpu_count()
    if cpu_count == 32:
        num_procs = 16
    elif cpu_count == 8:
        num_procs = 7
    else:
        raise ValueError("host CPU count (%d) not in pre-determined utilization numbers (8, 32).")

    # find all channel files for processing
    fov_file_list = fnmatch.filter(os.listdir(experiment_directory + analysis_directory +
                                              'originals/'), 'original_???.hdf5')
    fov_file_list.sort()
    information("Found %d FOVs to process." % len(fov_file_list))

    ### Cross correlations ####################################
    # see if the cross correlations already exist and load them if so
    if os.path.exists(experiment_directory + analysis_directory + 'crosscorrs.pkl'):
        # disable garbage collection. Why get rid of gc ?? -jt
        gc.disable()
        with open(experiment_directory + analysis_directory + "crosscorrs.pkl", 'r') as xcorr_file:
            fov_xcorrs = pickle.load(xcorr_file)
        gc.enable()
        information('Loaded precalculated cross-correlations.')

    # otherwise make it
    else:
        # a dict to contain one tuple for each FOV. key: fov_id value: [xcorr0, xcorr1, ...]
        fov_xcorrs = {}

        # for each fov find cross correlations (sending to pull)
        for fov_file in fov_file_list:
            # use split to pull out the FOV number
            fov = int(fov_file.split("original_")[1].split(".hdf5")[0])

            # adjust processing based on user-provided switches
            if specify_fovs and not (fov in user_spec_fovs):
                continue
            if start_with_fov > 0 and fov < start_with_fov:
                continue

            information("Calculating cross-correlations for FOV %d." % fov)

            # find all channel IDs in the current FOV
            peaks = []
            # h5f = h5py.File(experiment_directory + analysis_directory + "originals/" + fov_file, 'r', libver='latest', swmr=True)
            h5f = h5py.File(experiment_directory + analysis_directory + "originals/" + fov_file, 'r', libver='earliest')
            for item in h5f.keys():
                if item.split("el_")[0] == "chann": # just a check to make sure we got the channels
                    peaks.append(int(item.split("channel_")[1])) # get peak (should maybe come from att)
            h5f.close()

            # start up mutliprocessing
            m = Manager()
            q = m.Queue()
            pool = Pool(num_procs)
            pool_result = pool.map_async(get_crosscorrs, peaks) # worker function
            pool.close()

            # print the output
            try:
                while (True):
                    if (pool_result.ready()): break
                    size = q.qsize()
                    information("Finished %d of %d peaks for FOV %d..." % (size, len(peaks), fov))
                    time.sleep(10)
            except KeyboardInterrupt:
                    warning("Caught KeyboardInterrupt, terminating workers")
                    pool.terminate()
                    pool.join()
                    raise
            except:
                warning(sys.exc_info()[1])
                warning(traceback.print_tb(sys.exc_info()[2]))
                raise BaseException("Error in crosscorrelation calculation.")

            # warn user if it fails
            if not pool_result.successful():
                raise BaseException("Processing pool not successful!")

            # store the data to crosscorr_peaks
            crosscorr_peaks = pool_result.get()

            # add key value pair to cross correlations
            fov_xcorrs[fov] = crosscorr_peaks

            information("Finished cross-correlations for FOV %d." % fov)

        # write cross-correlations for channel picking
        information("Writing cross correlations file.")
        with open(experiment_directory + analysis_directory + "crosscorrs.pkl", 'wb') as crosscorrs:
            pickle.dump(fov_xcorrs, crosscorrs)

        information("Done calculating cross correlations. Starting channel picking.")

    raw_input('hit enter to continue...')

    ### User selection (channel picking) and averaging of empties #######################
    # set up Multiprocessing
    m = Manager()
    q = m.Queue()
    pool = Pool(num_procs)

    # dictionary for bgrd lists
    bgrd_peaks_all = []

    # go through the fovs again, same as above
    for fov_file in fov_file_list:
        # don't do channel picking or aveaging if user specifies not to
        fov = int(fov_file.split("original_")[1].split(".hdf5")[0])
        if specify_fovs and not (fov in user_spec_fovs):
            continue
        if start_with_fov > 0 and fov < start_with_fov:
            continue

        # see if spec file exists, load if so, other wise calculate it
        try:
            with open(experiment_directory + analysis_directory + 'specs/specs_%03d.pkl' % fov, 'rb') as pkl_file:
                user_picks = pickle.load(pkl_file)
                bgrd_peaks = user_picks[2] # get the backgrounds out
                information('Loaded precalculated specs file for FOV %03d.' % fov)
        except:
            # launch UI and calcuate that stuff otherwise
            bgrd_peaks = fov_choose_channels_UI(fov_file, fov_xcorrs)

        ### Average empty channels
        # either do it right after you do the UI selection
        pool_result = pool.apply_async(average_picked_empties, [(fov_file, bgrd_peaks),])

        # or put information and do it all after
        #bgrd_peaks_all.append((fov_file, bgrd_peaks))

    #pool_result = pool.map_async(average_picked_empties, bgrd_peaks_all)
    pool.close() # no more data after this loop has been gone through all the times

    # loop through and give updates on progress
    while (True):
        # update on how much is finished
        size = q.qsize()
        information("Completed", size, "of", len(fov_file_list), "time averages...")
        if (pool_result.ready()): break # break out if everything is done
        time.sleep(10)

    if not pool_result.successful():
        warning("Processing pool not successful!")
        raise AttributeError
    else:
        information("Finished averaging empties.")
