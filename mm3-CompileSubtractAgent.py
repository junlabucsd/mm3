#!/usr/bin/python
from __future__ import print_function
def warning(*objs):
    print(time.strftime("%H:%M:%S CompileAgent WARNING:", time.localtime()), *objs, file=sys.stderr)
def information(*objs):
    print(time.strftime("%H:%M:%S CompileAgent:", time.localtime()), *objs, file=sys.stdout)

# import modules
import sys
import os
import time
import inspect
import getopt
import yaml
import traceback
import h5py
import struct
import re
import glob
import gevent
import marshal
import fnmatch
try:
    import cPickle as pickle
except:
    import pickle
from sys import platform as platform
import h5py
import multiprocessing
from multiprocessing import Pool, Manager, Lock, Event
import numpy as np
import numpy.ma as ma
from scipy import ndimage
import scipy.signal as spsig
import scipy.stats as spstats
from skimage.feature import match_template
from sklearn.cluster import KMeans
from skimage.exposure import rescale_intensity, equalize_hist
from skimage.segmentation import clear_border

# for watching file events
use_inotify = False
use_watchdog = False
if platform == "linux" or platform == "linux2":
    # linux
    import gevent_inotifyx as inotify
    from gevent.queue import Queue as gQueue
    use_inotify = True
elif platform == "darwin":
    # OS X
    from watchdog.observers import Observer
    from watchdog.events import PatternMatchingEventHandler
    use_watchdog = True

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
from subtraction_helpers import *
import mm3_helpers as mm3

# debug modules
#import pprint as pp
#import matplotlib.pyplot as plt

# non-daemonic process pool subclass to allow sub-processes
class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

class SubtractPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess

# setup function for analyzer pool
def setup(event):
    global unpaused
    unpaused = event

# make initial clusters using K-means
def create_clusters(image_metadata):
    '''
    Takes your image metadata which has x and y locations, and groups the fovs by their location

    Parameters
    image_metadata
    num_fovs : int
        Number of fovs, this is a global.

    Returns
    image_metadata
        But the fov has been edited.
    k_means
        k_mean fitting object
    '''
    information("Starting initial cluster formation...")
    # extract XY positions
    fns = [k for k in image_metadata.keys()]
    x = [image_metadata[fn]['metadata']['x'] for fn in fns]
    y = [image_metadata[fn]['metadata']['y'] for fn in fns]

    # cluster FOVs by k-means
    positions = np.array(list(zip(x, y)))
    k_means = KMeans(init = 'k-means++', n_clusters = num_fovs, n_init = 50) # creates the cluster object
    k_means.fit(positions) # does the clutering on the positions
    labels = k_means.labels_ # pulls out the labels

    information("Number of clusters: %d." % len(k_means.cluster_centers_))
    #information(np.unique(labels, return_counts = True))

    # write the k_means object to disk in case of script failure
    with open(experiment_directory + analysis_directory + 'kmeans_fitter.pkl', 'w') as km_fh:
        pickle.dump(k_means, km_fh)

    # set the FOV IDs in the image metadata
    for lindex in range(len(fns)):
        image_metadata[fns[lindex]]['fov'] = labels[lindex]

    return image_metadata, k_means

# make masks from initial set of images (same images as clusters)
def make_masks(image_metadata):
    '''
    Make masks goes through the images given in image_metadata and builds a consensus
    Mask for each image per fov, which it returns as dictionary named channel_masks.
    The keys in this dictionary are fov id, and the values is a another dictionary. This dict's keys are channel locations (peaks) and the values is a [2][2] array:
    [[minrow, maxrow],[mincol, maxcol]] of pixel locations designating the corner of each mask
    for each channel on the whole image

    One important consequence of these function is that the channel ids and the size of the
    channel slices are decided now. Updates to mask must coordinate with these values

    Parameters
    image_metadata : dict
        image information created by get_params
    crop_half_width : int, global


    Returns
    channel_masks : dict
        dictionary of consensus channel masks. Appended to image_metadata in __main__

    Called By
    __main__

    Calls
    '''
    information("Determining initial channel masks...")

    #intiaize dictionary
    channel_masks = {}

    # get the size of the images (hope they are the same)
    for img_k, img_v in image_metadata.iteritems():
        image_rows = img_v['image_size'][0]
        image_cols = img_v['image_size'][1]
        break # just need one. using iteritems mean the whole dict doesn't load

    # get the fov ids
    fovs = []
    for img_k, img_v in image_metadata.iteritems():
        if img_v['fov'] not in fovs:
            fovs.append(img_v['fov'])

    # max width and length across all and all fovs. channels will get expanded by these values
    # this important for later updates to the masks, which should be the same
    max_chnl_mask_len = 0
    max_chnl_mask_wid = 0

    # for each fov make a channel_mask dictionary from consensus mask for each fov
    for fov in fovs:
        # initialize a the dict and consensus mask
        channel_masks_1fov = [] # list which holds channel masks [[peak1,[[y1, y2],[x1,x2]],...]
        consensus_mask = np.zeros([image_rows, image_cols]) # mask for labeling

        # bring up information for each image
        for img_k, img_v in image_metadata.iteritems():
            # skip this one if it is not of the current fov
            if img_v['fov'] != fov:
                continue

            # for each channel in each image make a single mask
            img_chnl_mask = np.zeros([image_rows, image_cols])

            # and add the channel mask to it
            for chnl_peak, peak_ends in img_v['channels'].iteritems():
                # pull out the peak location and top and bottom location
                # and expand by padding (more padding done later for width)
                x1 = max(chnl_peak-int(crop_half_width), 0)
                x2 = min(chnl_peak+int(crop_half_width), image_cols)
                y1 = peak_ends['closed_end_px']
                y2 = peak_ends['open_end_px']

                # add it to the mask for this image
                img_chnl_mask[y1:y2, x1:x2] = 1

            # add it to the consensus mask
            consensus_mask += img_chnl_mask

        # average the consensus mask
        consensus_mask = consensus_mask.astype('float32') / float(np.amax(consensus_mask))

        # threshhold and homogenize each channel mask within the mask, label them
        # label when value is above 0.1 (so 90% occupancy), transpose.
        # the [0] is for the array ([1] is the number of regions)
        # It transposes and then transposes again so regions are labeled left to right
        consensus_mask = ndimage.label(clear_border(consensus_mask.T > 0.1))[0].T

        # go through each label
        for label in np.unique(consensus_mask):
            if label == 0: # label zero is the background
                continue
            binary_core = consensus_mask == label

            # clean up the rough edges
            poscols = np.any(binary_core, axis = 0) # column positions where true (any)
            posrows = np.any(binary_core, axis = 1) # row positions where true (any)

            # channel_id givin by horizontal position
            # this is important. later updates to the positions will have to check
            # if their channels contain this median value to match up
            channel_id = int(np.median(np.where(poscols)[0]))

            # store the edge locations of the channel mask in the dictionary
            min_row = np.min(np.where(posrows)[0]) - channel_length_pad # pad length
            max_row = np.max(np.where(posrows)[0]) + channel_length_pad
            min_col = max(np.min(np.where(poscols)[0]) - 5, 0) # pad width
            max_col = min(np.max(np.where(poscols)[0]) + 5, image_cols)

            # if the min/max cols are within the image bounds,
            # add the mask, as 4 points, to the dictionary
            if min_col > 0 and max_col < image_cols:
                channel_masks_1fov.append([channel_id, [[min_row, max_row], [min_col, max_col]]])

                # find the largest channel width and height while you go round
                max_chnl_mask_len = max(max_chnl_mask_len, max_row - min_row)
                max_chnl_mask_wid = max(max_chnl_mask_wid, max_col - min_col)

        # add channel_mask dictionary to the fov dictionary, use copy to play it safe
        channel_masks[fov] = channel_masks_1fov[:]


    for fov in fovs:
        #saving_mask = np.zeros([image_rows, image_cols]) # binary mask for debug

        # go back through each label and update the dictionary
        for n, pk_and_mask in enumerate(channel_masks[fov]):
            chnl_mask = pk_and_mask[1]

            # just add length to the open end (top of image, low column)
            if chnl_mask[0][1] - chnl_mask[0][0] !=  max_chnl_mask_len:
                channel_masks[fov][n][1][0][1] = chnl_mask[0][0] + max_chnl_mask_len
            # enlarge widths around the middle, but make sure you don't get floats
            if chnl_mask[1][1] - chnl_mask[1][0] != max_chnl_mask_wid:
                wid_diff = max_chnl_mask_wid - (chnl_mask[1][1] - chnl_mask[1][0])
                if wid_diff % 2 == 0:
                    channel_masks[fov][n][1][1][0] = max(chnl_mask[1][0] - wid_diff/2, 0)
                    channel_masks[fov][n][1][1][1] = min(chnl_mask[1][1] + wid_diff/2, image_cols - 1)
                else:
                    channel_masks[fov][n][1][1][0] = max(chnl_mask[1][0] - (wid_diff-1)/2, 0)
                    channel_masks[fov][n][1][1][1] = min(chnl_mask[1][1] + (wid_diff+1)/2, image_cols - 1)

            # update the saving mask with the final locatino for debug
            # saving_mask[channel_masks[fov][n][1][0][0]:
            # channel_masks[fov][n][1][0][1],
            # channel_masks[fov][n][1][1][0]:channel_masks[fov][n][1][1][1]] = True

    #save the channel mask dictionary
    if not os.path.exists(os.path.abspath(experiment_directory + analysis_directory + 'channel_masks/')):
        os.makedirs(os.path.abspath(experiment_directory + analysis_directory + 'channel_masks/'))
    with open(experiment_directory + analysis_directory + 'channel_masks/channel_masks.pkl' % fov, 'w') as cmask_file:
        pickle.dump(channel_masks, cmask_file)

    information("Channel masks saved.")

    return channel_masks

# worker function for extracting image metadata
def extract_metadata(tif):
    """
    This gets the specific meta information from a Elements created tiff file.

    Called by get_params

    arguments:
        fname (tifffile.TiffFile): TIFF file object from which data will be extracted
    returns:
        dictionary of values:
            'jdn' (float)
            'x' (float)
            'y' (float)
            'plane_names' (list of strings)

    """
    idata = { 'jdn': 0.0,
              'x': 0.0,
              'y': 0.0,
              'plane_names': []}

    for page in tif:
        #print("Page found.")
        for tag in page.tags.values():
            #print("Checking tag",tag.name,tag.value)
            t = tag.name, tag.value
            t_string = u""
            time_string = u""
            # Interesting tag names: 65330, 65331 (binary data; good stuff), 65332
            # we wnat to work with the tag of the name 65331
            # if the tag name is not in the set of tegs we find interesting then skip this cycle of the loop
            if tag.name not in ('65331', '65332', 'strip_byte_counts', 'image_width', 'orientation', 'compression', 'new_subfile_type', 'fill_order', 'max_sample_value', 'bits_per_sample', '65328', '65333'):
                #print("*** " + tag.name)
                #print(tag.value)
                pass
            #if tag.name == '65330':
            #    return tag.value
            if tag.name in ('65331'):
                # make info list a list of the tag values 0 to 65535 by zipoing up a paired list of two bytes, at two byte intervals i.e. ::2
                # note that 0X100 is hex for 256
                infolist = [a+b*0x100 for a,b in zip(tag.value[0::2], tag.value[1::2])]
                # get char values for each element in infolist
                for c_entry in range(0, len(infolist)):
                    # the element corresponds to an ascii char for a letter or bracket (and a few other things)
                    if infolist[c_entry] < 127 and infolist[c_entry] > 64:
                        # add the letter to the unicode string t_string
                        t_string += chr(infolist[c_entry])
                    #elif infolist[c_entry] == 0:
                    #    continue
                    else:
                        t_string += " "

                # this block will find the dTimeAbsolute and print the subsequent integers
                # index 170 is counting seconds, and rollover of index 170 leads to increment of index 171
                # rollover of index 171 leads to increment of index 172
                # get the position of the array by finding the index of the t_string at which dTimeAbsolute is listed not that 2*len(dTimeAbsolute)=26
                #print(t_string)

                arraypos = t_string.index("dXPos") * 2 + 16
                xarr = tag.value[arraypos:arraypos+4]
                b = ''.join(chr(i) for i in xarr)
                idata['x'] = float(struct.unpack('<f', b)[0])

                arraypos = t_string.index("dYPos") * 2 + 16
                yarr = tag.value[arraypos:arraypos+4]
                b = ''.join(chr(i) for i in yarr)
                idata['y'] = float(struct.unpack('<f', b)[0])

                arraypos = t_string.index("dTimeAbsolute") * 2 + 26
                shortarray = tag.value[arraypos+2:arraypos+10]
                b = ''.join(chr(i) for i in shortarray)
                idata['jdn'] = float(struct.unpack('<d', b)[0])

                # extract plane names
                il = [a+b*0x100 for a,b in zip(tag.value[0::2], tag.value[1::2])]
                li = [a+b*0x100 for a,b in zip(tag.value[1::2], tag.value[2::2])]

                strings = list(zip(il, li))

                allchars = ""
                for c_entry in range(0, len(strings)):
                    if 31 < strings[c_entry][0] < 127:
                        allchars += chr(strings[c_entry][0])
                    elif 31 < strings[c_entry][1] < 127:
                        allchars += chr(strings[c_entry][1])
                    else:
                        allchars += " "

                allchars = re.sub(' +',' ', allchars)

                words = allchars.split(" ")

                #print(words)

                planes = []
                for idx in [i for i, x in enumerate(words) if x == "sOpticalConfigName"]:
                    try: # this try is in case this is just an imported function
                        if Goddard:
                            if words[idx-1] == "uiCompCount":
                                planes.append(words[idx+1])
                        else:
                            planes.append(words[idx+1])
                    except:
                        planes.append(words[idx+1])
                idata['plane_names'] = planes
    return idata

# Worker function for getting image parameters and information
def get_params(image_filename, find_channels=True):
    '''This is a damn important function for getting the information
    out of an image

    it returns a dictionary like this for each image:

    { 'filename': image_filename,
             'metadata': image_metadata,
             'image_size' : image_size, # [image.shape[0], image.shape[1]]
             'channels': cp_dict,
             'analyze_success': True,
             'fov': -1, # fov is found later with kmeans clustering
             'sent_to_write': False,
             'write_success': False}

    Called by __main__

    Calls
    extract_metadata
    '''
    unpaused.wait()
    try:
        # open up file and get metadata
        try:
            with tiff.TiffFile(image_filename) as tif:
                image_data = tif.asarray()
                image_metadata = extract_metadata(tif)
        except:
            time.sleep(2)
            with tiff.TiffFile(image_filename) as tif:
                image_data = tif.asarray()
                image_metadata = extract_metadata(tif)

        # make the image 3d and crop the top & bottom per the image_vertical_crop parameter
        if len(image_data.shape) == 2:
            image_data = np.expand_dims(image_data, 0)
        if image_vertical_crop >= 0:
            image_data = image_data[:,image_vertical_crop:image_data.shape[1]-image_vertical_crop,:]
        else:
            padsize = abs(image_vertical_crop)
            image_data = np.pad(image_data, ((0,0), (padsize, padsize), (0,0)), mode='edge')

        # fix the image orientation and get the number of planes
        image_data = fix_orientation_perfov(image_data, image_filename)
        image_planes = image_data.shape[0]

        # if the image data has more than 1 plane restrict image_data to just the first
        if len(image_data.shape) > 2:
            ph_index = np.argmax([np.mean(image_data[ci]) for ci in range(image_data.shape[0])])
            image_data = image_data[ph_index]

        # save the shape data
        image_size = [image_data.shape[0], image_data.shape[1]]

        # find channels if desired
        if find_channels:
            # structure for channel dimensions
            channel_params = []
            # Structure of channel_params
            # 0 = peak position ('peak_px')
            # 1 = index of max (the closed end)
            # 2 = index of min (the open end)
            # 3 = length of the channel (min - max)

            # Detect peaks in the x projection (i.e. find the channels).
            projection_x = image_data.sum(axis = 0)
            # find_peaks_cwt is a finction which attempts to find the peaks in a 1-D array by
            # convolving it with a wave. here the wave is the default wave used by the algorythm
            # but the minimum signal to noise ratio is specified
            peaks = spsig.find_peaks_cwt(projection_x, np.arange(channel_width-5,channel_width+5),
                                         min_snr = channel_detection_snr)

            # If the left-most peak position is within half of a channel separation,
            # discard the channel from the list.
            if peaks[0] < (channel_midline_separation / 2):
                peaks = peaks[1:]
            # If the difference between the right-most peak position and the right edge
            # of the image is less than half of a channel separation, discard the channel.
            if image_data.shape[0] - peaks[len(peaks)-1] < (channel_midline_separation / 2):
                peaks = peaks[:-1]

            # Find the average channel ends for the y-projected image
            projection_y = image_data.sum(axis = 1)
            # diff returns the array of the differences between each element and its neighbor
            # in a given array. View takes a snapshot of data in memory and allows it to be
            # reinterpreted as annother data type. this appears to be the index of the
            # maximum of the derivative of the uper 2/3's of the y projection rebinned
            image_deriv_max = np.diff(projection_y[:int(projection_y.shape[0]*(1./3.))].view(np.int32)[0::2]).argmax()
            # the same only the indexl of the min of the derivative of the  lower 2/3's
            # of the projection plus 2/3 of the "height" of the y projection
            image_deriv_min = np.diff(projection_y[int(projection_y.shape[0]*(2./3.)):].view(np.int32)[0::2]).argmin() + int(projection_y.shape[0]*(2./3.))

            # Slice up the image into an array of channel strips
            channel_strips = []
            for peak in peaks:
                # Structure of channel_strips
                # 0 = peak position, 1 = image of the strip (AKA the channel) itself
                channel_strips.append([peak, image_data[0:image_data.shape[0], peak-crop_half_width:peak+crop_half_width]])

            # Find channel starts and ends based on the maximum derivative of the channel profile;
            # min of the first derivative is usually the open end and max is usually the closed end.

            # THE FOLLOWING IS SIMILAR TO WHAT THE CODE ALREADY DID TO THE WHOLE IMAGE ONLY NOW IT IS
            # DOING ANALYSIS OF THE IMAGES OF INDIVIDUAL CHANNELS! i.e. it is a first order correction
            # on the previous calculations done at the whole image level
            # loop through the list of channel strip structures that we created

            # create these slice bounds to ensure we are within the image
            px_window = 20 # Search for a channel bounds in 20px windows around the image global bounds
            low_for_max = max(0, image_deriv_max-(px_window))
            high_for_max = min(image_deriv_max+px_window, image_data.shape[0])
            low_for_min = max(0, image_deriv_min-px_window)
            high_for_min = min(image_deriv_min+(px_window), image_data.shape[0])

            for strip in channel_strips:
                # get the projection of the image of the channel strip onto the y axis
                slice_projection_y = strip[1].sum(axis = 1)
                # get the derivative of the projection
                first_derivative = np.diff(slice_projection_y.view(np.int32)[0::2])

                # find the postion of the maximum value of the derivative of the projection
                # of the slice onto the y axis within the distance of px_window from the edge of slice
                maximum_index = first_derivative[low_for_max:high_for_max].argmax()
                # same for the min
                minimum_index = first_derivative[low_for_min:high_for_min].argmin()
                # attach the calculated data to the list channel_params, corrected
                channel_params.append([strip[0], # peak position (x)
                    int(maximum_index + low_for_max), # close end position (y)
                    int(minimum_index + low_for_min), # open end position (y)
                    int(abs((minimum_index + low_for_min) - (maximum_index + low_for_max))), # length y
                    False]) # not sure what false is for

            # Guide a re-detection of the min/max indices to smaller windows of the modes for this image
            # here mode is meant in the statistical sence ie mode([1,2,2,3,3,3,3,4]) give 3
            # channel_modes is a list of the modes (and a list of ther frequencies) for each of the
            # coordinates sorted in each element of the channel_params list elements
            channel_modes = spstats.mode(channel_params, axis = 0)
            # channel_medians is a list of the medians in the same fashion
            channel_medians = spstats.nanmedian(channel_params, axis = 0)

            # Sanity-check boundaries:
            #  Reset modes
            channel_modes = spstats.mode(channel_params, axis = 0)
            channel_medians = spstats.nanmedian(channel_params, axis = 0)
            max_baseline = 0
            min_baseline = 0
            len_baseline = 0
            # set min_baseline
            try:
                if channel_modes[0][0][2] > 0:
                    min_baseline = int(channel_modes[0][0][2])
                # use median information if modes are no use
                else:
                    min_baseline = int(channel_medians[2])
                # if everything is unreasonable COMPLAIN!
                if min_baseline <= 0:
                    warning("No reasonable baseline minumum found!")
                    warning("Image:",image_filename)
                    warning("Medians:",channel_medians)
                    warning("Modes:",channel_modes)
                    raise
            except:
                warning('%s: error in mode/median analysis; maybe the device is delaminated?' % image_filename.split("/")[-1])
                print("-")
                return [image_filename, -1]

            # set max_baseline
            if channel_modes[0][0][1] > 0:
                max_baseline = int(channel_modes[0][0][1])
            # use median information if modes are no use
            else:
                max_baseline = int(channel_medians[1])
            # if everything is unreasonable COMPLAIN!
            if max_baseline <= 0:
                warning("%s: no reasonable baseline maximum found." % image_filename.split("/")[-1])
                print("-")
                return [image_filename, -1]

            # set len_baseline
            if channel_modes[0][0][3] > 0:
                len_baseline = channel_modes[0][0][3]
            # use median information if modes are no use
            else:
                len_baseline = channel_medians[3]

            # check each channel for a length that is > 50% different from the mode length
            # 20150525: using 10% as a threshold for reassignment is problematic for SJW103
            # dual-trench devices because the channels on either side of the double-tall FOVs are not the same.
            # doing a C style for loop an alternative is to use "for n,channel in enumerate(channel_params)"
            # assigments to the list using the index n will stick!
            for n in range(0,len(channel_params)):
                if float(abs(channel_params[n][3] - len_baseline)) / float(len_baseline) > 0.5:
                    information("Correcting  diff(len) > 0.3...")
                    information("...")
                    if abs(channel_params[n][1] - max_baseline) < abs(channel_params[n][2] - min_baseline):
                        channel_params[n][2] = int(channel_params[n][1]) + int(len_baseline)
                    else:
                        channel_params[n][1] = int(channel_params[n][2]) - int(len_baseline)
                    channel_params[n][3] = int(abs(channel_params[n][1] - channel_params[n][2]))

            # check each channel for a closed end that is inside the image boundaries
            for n in range(0,len(channel_params)):
                if channel_params[n][1] < 0:
                    information("Correcting [n][1] < 0 in",image_filename,"at peak",channel_params[n][0])
                    information("...", max_baseline, min_baseline, int(max_baseline), int(min_baseline))
                    channel_params[n][1] = max_baseline
                    channel_params[n][2] = min_baseline
                    channel_params[n][3] = abs(channel_params[n][1] - channel_params[n][2])

            # check each channel for an open end that is inside the image boundaries
            for n in range(0,len(channel_params)):
                if channel_params[n][2] > image_data.shape[0]:
                    information("Correcting [n][2] > image_data.shape[0] in",image_filename,"at peak",channel_params[n][0])
                    information("...", max_baseline, min_baseline, int(max_baseline), int(min_baseline))
                    channel_params[n][1] = max_baseline
                    channel_params[n][2] = min_baseline
                    channel_params[n][3] = abs(channel_params[n][1] - channel_params[n][2])

            # create a dictionary of channel starts and ends
            cp_dict = {cp[0]:
                      {'closed_end_px': cp[1], 'open_end_px': cp[2]} for cp in channel_params}

        else:
            cp_dict = -1

        # return the file name, the data for the channels in that image, and the metadata
        return { 'filename': image_filename,
                 'metadata': image_metadata,
                 'image_size' : image_size,
                 'channels': cp_dict,
                 'analyze_success': True,
                 'fov': -1, # fov is found later with kmeans clustering
                 'sent_to_write': False,
                 'write_success': False,
                 'write_plane_order' : False} # this is found after get_params
    except:
        warning('failed get_params for ' + image_filename.split("/")[-1])
        print(sys.exc_info()[0])
        print(sys.exc_info()[1])
        print(traceback.print_tb(sys.exc_info()[2]))
        return {'filename': image_filename, 'analyze_success': False}

# define function for flipping the images on an FOV by FOV basis
def fix_orientation_perfov(image_d, filename_d):
    '''
    called by data_writer
    '''

    # double is for FOVs with channels pointed up and down
    if image_orientation == "double":
        # make a projection so the values along the x axis (each row) are summed into one value at each y position
        # axis = 1 means sum the rows to get one column of data
        projection_y = image_d.sum(axis = 1)
        # re zero this array
        projection_y = np.max(projection_y) - projection_y
        # get an int? value  two fifths of the hight of the image
        cut_px = image_d.shape[0] / 2.5
        # get a whole number value one eith of the height of the image
        search_cut = image_d.shape[0] / 8
        # use the function match_template on the vertial stacks of ( (two copies of the projection with a margin cut off at both ends) and (two copies of the projection with a larger margin cut off at both ends) )

        # the match function finds the smaller image in the larger image (OR AT LEAST THINGS LIKT IT) by means of calculating correltation between a smaller image and a larger image at each position in thet matrix
        # it returns the correlation matrix (which is presumably a reveled one dimensional list of correlations? or maybe 2 dimensioanl?)
        match_result = match_template(np.vstack((projection_y[search_cut:projection_y.shape[0]-search_cut], projection_y[search_cut:projection_y.shape[0]-search_cut])), np.fliplr(np.vstack((projection_y[cut_px:image_d.shape[0] - cut_px], projection_y[cut_px:image_d.shape[0] - cut_px]))))
        # we want the position of the correaltin matrix
        # np.unravel_index is a function which returns the rown and column position for 1 dimensional ravled indexes (indexes where you count as you go along an axis and then loop back when you hit the end and keep counting)
        # you can feed it an array of these index position valuse and it will return a tuple with an array for the row positions and an array for the column positions for each index given
        # note that numpy's argmax function returns the "Indices of the maximum values along an axis"
        ij = np.unravel_index(np.argmax(match_result), match_result.shape)
        # get the second dimension array into "c_peak" and the first dimension array into "y"
        # the ::-1 returns the elements starting at the back, this should be the same as "y, c_peak  = ij[::1]"
        c_peak, y = ij[::-1]
        # shift the position values represented in c_peak by the margin calculated earlier
        c_peak += search_cut
        # make "offset" equal to half of the distance between the peak
        offset = (c_peak - cut_px) / 2
        # set cutposiotn to be the offset plus half od the image size
        cut_position = (image_d.shape[0]/2) + offset
        # take the image from a position 28 pixels up from the cut_cut position
        # take that porion and flip it
        # it is now the top image called "im_top"
        im_top = np.flipud(image_d[:cut_position-28,:])
        # keep the portion of the image for im_bottom in a symettrical fashion
        im_bottom = image_d[cut_position+28:,:]

        #code block for testing
        """
        fig = plt.figure(figsize=(20,7))
        rect = fig.patch
        rect.set_facecolor('white')
        ax = fig.add_subplot(1,1,1)
        p_template = np.flipud(projection_y[cut_px:image_d.shape[0] - cut_px])
        ax.plot(range(c_peak,c_peak+len(p_template)), p_template)
        ax.plot(projection_y)
        #ax2 = ax.twinx()
        #ax2.plot(py_conv, color = 'r')
        #ax.plot(py_conv)
        plt.show(block=False)
        """
        # make the tow images have the same size by "padding" them
        if im_top.shape[0] > im_bottom.shape[0]:
            im_bottom = np.pad(im_bottom, ((0, abs(im_top.shape[0] - im_bottom.shape[0])),(0,0)), mode = "edge")
        else:
            im_top = np.pad(im_top, ((0, abs(im_bottom.shape[0] - im_top.shape[0])),(0,0)), mode = "edge")

        #code block for testing
        """
        fig = plt.figure(figsize=(20,7))
        rect = fig.patch
        rect.set_facecolor('white')
        ax = fig.add_subplot(1,1,1)
        #ax.imshow(image_data, alpha = 0.5)
        #ax.imshow(np.pad(np.flipud(image_data), ((c_peak - (len(projection_y)/2), 0), (0, 0)), mode = "edge"), alpha = 0.5)
        #xv = np.linspace(-100,1000,1100)
        #yv = c_peak
        #ax.plot(xv, [yv for x in xv])
        ax.imshow(np.hstack((im_bottom, im_top)))
        plt.show(block=False)
        """
        # return the two images "glued together horizauntally" by usning numpy's horizauntal stack function
        return np.hstack((im_bottom, im_top))
        #mind you we are now done dealing with the double condition
    # next we deal with the FOV's to be flipped
    # setting image_orientation to 'auto' will use autodetection, otherwise
    # the usual user-defined flipping will ensue
    if image_orientation == "auto":
        if len(image_d.shape) == 2:
            image_d = np.expand_dims(image_d, 0)
        # for multichannel images, pick the plane to analyze with the highest mean px value
        ph_channel = np.argmax([np.mean(image_d[ci]) for ci in range(image_d.shape[0])])
        if np.argmax(image_d[ph_channel].mean(axis = 1)) > image_d[ph_channel].shape[0] / 2:
            return image_d
        else:
            return image_d[:,::-1,:]
    else:
        if len(image_d.shape) == 2:
            image_d = np.expand_dims(image_d, 0)
        # flip the images if "up" is the specified image orrientation
        if image_orientation == "up":
            return image_d[:,::-1,:]
        # do not flip the images if "down is the specified image orientation"
        elif image_orientation == "down":
            return image_d
        # this image has not fallen into any of the specified categories then -> "HUSTON WE HAVE A PROBLEM"
        # you will want to edit your YAML parameters file!
        else:
            raise AttributeError

# extract image and do early processing on tifs
def process_tif(image_data):
    '''
    Processes tif images, after opening them. fixes the orientation, reorders the planes,
    and rotates the way the data is stored.

    Parameters
    image_data : dictionary of image data per image
        Made by get_params and edited by __main__

    Returns
    image_edited : numpy array with planes
        this is the tiff image data

    Called By
    data_writer
    find_channels_init

    Calls
    tiff.imread
    fix_orientation_perfov
    '''

    # this gets the original picture again from the folder.
    image_pixeldata = tiff.imread(image_data['filename'])
    image_planes = image_data['metadata']['plane_names']

    plane_order = image_data['write_plane_order']

    if len(image_planes) > len(plane_order):
        warning('image_planes (%d, %s) longer than plane_order (%d)!' % (len(image_planes), str(image_planes), len(plane_order)))
        return False

    image_pixeldata = fix_orientation_perfov(image_pixeldata, image_data['filename'])
    assert(len(image_pixeldata.shape) > 2)
    assert(np.argmin(image_pixeldata.shape) == 0)

    # re-stack planes of the image data by the plane_names order
    aligned_planes = np.zeros([len(plane_order), image_pixeldata.shape[1], image_pixeldata.shape[2]])
    for pn_i, pn in enumerate(plane_order):
        if pn in image_planes:
            aligned_planes[pn_i] = image_pixeldata[image_planes.index(pn)]

    # rotate image_data such that data is stored per-pixel instead of per-plane;
    # there is no reason this is required other than it being a common standard
    # in image data e.g. if you want to get just a section of the image you can
    # omit the extra :, at the beginning of indexing notation
    image_pixeldata = np.rollaxis(aligned_planes, 0, 3)

    # pad/crop the image as appropriate
    if image_vertical_crop >= 0:
        image_pixeldata = image_pixeldata[image_vertical_crop:image_pixeldata.shape[1]-image_vertical_crop,:,:]
    else:
        padsize = abs(image_vertical_crop)
        image_pixeldata = np.pad(image_pixeldata, ((padsize, padsize), (0,0), (0,0)), mode='edge')

    return image_pixeldata

# for doing subtraction when just starting and there is a backlog
def subtract_backlog(fov_id):
    return_value = -1
    try:
        information('Subtracting backlog of images fov_id %03d.' % fov_id)

        # load cell peaks and empty mean
        cell_peaks = mm3.load_cell_peaks(fov_id) # load list of cell peaks from spec file
        empty_mean = mm3.load_empty_tif(fov_id) # load empty mean image
        empty_mean = mm3.trim_zeros_2d(empty_mean) # trim any zero data from the image
        information('There are %d cell peaks for fov_id %03d.' % (len(cell_peaks), fov_id))

        # aquire lock, giving other threads which may be blocking a chance to clear
        lock_acquired = hdf5_locks[fov_id].acquire(block = True)

        # peaks_images will be a list of [fov_id, peak, images, empty_mean]
        with h5py.File(experiment_directory + analysis_directory +
                       'originals/' + 'original_%03d.hdf5' % fov_id, 'r', libver='earliest') as h5f:

            for peak in sorted(cell_peaks):
                images = h5f[u'channel_%04d' % peak][:] # get all the images (and whole stack)
                plane_names = h5f[u'channel_%04d' % peak].attrs['plane_names'] # get plane names

                # move the images into a long list of list with the image next to empty mean.
                images_with_empties = []
                for image in images:
                    images_with_empties.append([image, empty_mean])
                del images

                # set up multiprocessing
                spool = Pool(num_blsubtract_subthreads) # for 'subpool of alignment/subtraction '.
                # send everything to be processed
                pool_result = spool.map_async(subtract_phase, images_with_empties,
                                              chunksize = 10)
                spool.close()
                information('Subtraction started for FOV %d, peak %04d.' % (fov_id, peak))

                # just loop around waiting for the peak to be done.
                try:
                    while (True):
                        time.sleep(1)
                        if pool_result.ready():
                            break
                    # inform user once this is done
                    information("Completed peak %d in FOV %03d (%d timepoints)" %
                                (peak, fov_id, len(images_with_empties)))
                except KeyboardInterrupt:
                    raise

                if not pool_result.successful():
                    warning('Processing pool not successful for peak %d.' % peak)
                    raise AttributeError

                # get the results and clean up memory
                subtracted_data = pool_result.get() # this is a list of (sub_image, offset)
                subtracted_images = zip(*subtracted_data)[0] # list of subtracted images
                offsets = zip(*subtracted_data)[1] # list of offsets for x and y
                # free some memory
                del pool_result
                del subtracted_data
                del images_with_empties

                # write the subtracted data to disk
                with h5py.File(experiment_directory + analysis_directory + 'subtracted/subtracted_%03d.hdf5' % fov_id, 'a', libver='earliest') as h5s:
                    # create data set, use first image to set chunk and max size
                    h5si = h5s.create_dataset("subtracted_%04d" % peak,
                                              data=np.asarray(subtracted_images, dtype=np.uint16),
                                              chunks=(1, subtracted_images[0].shape[0],
                                                      subtracted_images[0].shape[1], 1),
                                              maxshape=(None, subtracted_images[0].shape[0],
                                                        subtracted_images[0].shape[1], None),
                                              fletcher32=True, compression='lzf')

                    # rearrange plane names
                    plane_names = plane_names.tolist()
                    plane_names.append(plane_names.pop(0))
                    plane_names.insert(0, 'subtracted_phase')
                    h5si.attrs['plane_names'] = plane_names

                    # create dataset for offset information
                    # create and write first metadata
                    h5os = h5s.create_dataset(u'offsets_%04d' % peak,
                                              data=np.array(offsets),
                                              maxshape = (None, 2))

            # move over metadata once peaks have all peaks have been written
            with h5py.File(experiment_directory + analysis_directory + 'subtracted/subtracted_%03d.hdf5' % fov_id, 'a', libver='earliest') as h5s:
                sub_mds = h5s.create_dataset("metadata", data=h5f[u'metadata'],
                                             maxshape=(None, 3))

        # return 0 to the parent loop if everything was OK
        return_value = 0
    except:
        warning("Failed for fov_id: %03d" % fov_id)
        warning(sys.exc_info()[1])
        warning(traceback.print_tb(sys.exc_info()[2]))
        return_value = 1

    # release lock
    hdf5_locks[fov_id].release()

    return return_value

# worker function for doing subtraction
def subtract_phase(dataset):
    '''subtract_phase_only is the main worker function for doign alignment and subtraction.
    Modified from subtract_phase_only by jt on 20160511
    The subtracted image returned is the same size as the image given. It may however include
    data points around the edge that are meaningless but not marked.

    parameters
    ---------
    dataset : list of length two with; [image, empty_mean]

    returns
    ---------
    (subtracted_image, offset) : tuple with the subtracted_image as well as the ammount it
        was shifted to be aligned with the empty. offset = (x, y), negative or positive
        px values.
    '''
    try:
        if matching_length is not None:
            pass
    except:
        matching_length = 180

    try:
        # get out data and pad
        cropped_channel, empty_channel = dataset # [channel slice, empty slice]
        # rescale empty to levels of channel image
        #empty_channel = rescale_intensity(empty_channel,
        #                                  out_range=(np.amin(cropped_channel[:,:,0]),
        #                                             np.amax(cropped_channel[:,:,0])))

        ### Pad empty channel.
        # Rough padding amount for empty to become template in match_template
        start_padding = (25, 25, 25, 25) # (top, bottom, left, right)

        # adjust padding for empty so padded_empty is same size as channel later.
        # it is important that the adjustment is made on the bottom and right sides,
        # as later the alignment is measured from the top and left.
        y_diff = cropped_channel.shape[0] - empty_channel.shape[0]
        x_diff = cropped_channel.shape[1] - empty_channel.shape[1]

        # numpy.pad-compatible padding tuple: ((top, bottom), (left, right))
        empty_paddings = ((start_padding[0], start_padding[1] + y_diff), # add y_diff to sp[1]
                          (start_padding[2], start_padding[3] + x_diff)) # add x_diff to sp[3]

        # edge-pad the empty channel using these paddings
        padded_empty = np.pad(empty_channel, empty_paddings, 'edge')

        ### Align channel to empty using match template.
        # get a vertical chunk of the image of the empty channel
        empty_subpart = padded_empty[:matching_length+start_padding[0]+start_padding[1]]
        # get a vertical chunk of the channel to be subtracted from
        chan_subpart = cropped_channel[:matching_length,:,0] # phase data = 0

        # equalize histograms for alignment
        empty_subpart = equalize_hist(empty_subpart)
        chan_subpart = equalize_hist(chan_subpart)

        # use match template to get a correlation array and find the position of maximum overlap
        match_result = match_template(empty_subpart, chan_subpart)
        # get row and colum of max correlation value in correlation array
        y, x = np.unravel_index(np.argmax(match_result), match_result.shape)

        # this is how much it was shifted in x and y.
        # important to store for getting exact px position later
        offset = (x - start_padding[2], y - start_padding[0])

        ### pad the original cropped channel image.
        # align the data between the empty image and the cropped images
        # using the offsets to create a padding that adjusts the location of
        # the channel
        channel_paddings = ((y, start_padding[0] + start_padding[1] - y), # include sp[0] with sp[1]
                            (x, start_padding[2] + start_padding[3] - x), # include sp[2] with sp[3]
                            (0,0))

        # the difference of padding on different sides relocates the channel to same
        # relative location as the empty channel (in the padded version.)
        shifted_padded_channel = np.pad(cropped_channel.astype('int32'),
                                        channel_paddings,
                                        mode="edge")

        # trim down the pad-shifted channel to the same region where the empty image has data
        channel_for_sub = shifted_padded_channel[start_padding[0]:-1*start_padding[1],
                                                 start_padding[2]:-1*start_padding[3]]
        empty_for_sub = padded_empty[start_padding[0]:-1*start_padding[1],
                                     start_padding[2]:-1*start_padding[3]]

        ### rescale the empty image intensity based on the pixel intensity ratios
        # calculate the ratio of pixel intensities
        pxratios = channel_for_sub[:,:,0].astype('float')/empty_for_sub.astype('float')
        # calculate the rough peak of intensity values
        pxrdist = np.histogram(pxratios, range=(0.5,1.5), bins=100)
        # get the peak value for rescaling the empty image
        distcenters = pxrdist[1][:-1]+np.diff(pxrdist[1])/2
        pxrpeak = distcenters[np.argmax(pxrdist[0])]
        # rescale the empty image
        empty_for_sub = (empty_for_sub.astype('float')/pxrpeak).astype('uint16')

        # add dimension to empty channel to give it Z=1 size
        if len(empty_for_sub.shape) < 3:
            # this function as called adds a third axis to empty channel which is flat now
            empty_for_sub = np.expand_dims(empty_for_sub, axis = 2)
        padded_empty_3d0 = np.zeros_like(empty_for_sub)
        for color in xrange(1, cropped_channel.shape[2]):
            # depth-stack the non-phase planes of the cropped image with zero arrays of same size
            empty_for_sub = np.dstack((empty_for_sub, padded_empty_3d0))

        ### Compute the difference between the empty and channel phase contrast images
        # subtract the empty image from the cropped channel image
        channel_subtracted = channel_for_sub.astype('int32') - empty_for_sub.astype('int32')

        channel_subtracted[:,:,0] *= -1 # make cells high-intensity
        # Reset the zero level in the image by subtracting the min value
        channel_subtracted[:,:,0] -= np.min(channel_subtracted[:,:,0])
        # add one to everything so there are no zeros in the image
        channel_subtracted[:,:,0] += 1
        # Stack the phase-contrast image used for subtraction to the bottom of the stack
        channel_subtracted = np.dstack((channel_subtracted, channel_for_sub[:,:,0]))

        return((channel_subtracted, offset))
    except:
        warning("Error in subtracting_phase:")
        warning(sys.exc_info()[1])
        warning(traceback.print_tb(sys.exc_info()[2]))
        raise

# writer function for appending to originals_nnn.hdf5
def data_writer(image_data, channel_masks, subtract=False, save_originals=False, compress_hdf5=True):
    '''
    data_writer saves an hdf5 file for each fov which contains the original images for that fov,
    the meta_data about the images (location, time), and the sliced channels based on the
    channel_masks.

    Called by:
    __main__

    Calls
    mm3.cut_slice
    process_tif
    mm3.load_cell_peaks
    mm3.load_empyt_tif
    mm3.trim_zeros_2d
    '''
    returnvalue = False
    h5s = None
    try:
        # imporant information for saving
        fov_id = image_data['fov']
        plane_order = image_data['write_plane_order']

        # load the image and process it
        image_pixeldata = process_tif(image_data)

        # acquire the write lock for this FOV; if there is a block longer than 3 seconds,
        # acquisition should timeout since backlog subtraction may be running
        t_s = time.time()
        lock_acquired = hdf5_locks[fov_id].acquire(block = True, timeout = 3.0)
        t_e = time.time() - t_s
        if t_e > 1 and lock_acquired:
            information("data_writer: lock acquisition delay %0.2f s" % t_e)
        if not lock_acquired:
            # no need to release the lock since we don't have it.
            information("data_writer: unable to obtain lock for FOV %d." % fov_id)
            return returnvalue

        # if doing subtraction, load cell peaks from spec file for fov_id
        if subtract:
            try:
                cell_peaks = mm3.load_cell_peaks(fov_id) # load list of cell peaks from spec file
                empty_mean = mm3.load_empty_tif(fov_id) # load empty mean image
                empty_mean = mm3.trim_zeros_2d(empty_mean) # trim any zero data from the image
                h5s = h5py.File(experiment_directory + analysis_directory +
                                'subtracted/subtracted_%03d.hdf5' % fov_id, 'r+', libver='earliest')
                #h5s.swmr_mode = False
            except:
                subtract = False

        with h5py.File(experiment_directory + analysis_directory + 'originals/original_%03d.hdf5' % fov_id, 'a', libver='earliest') as h5f:
            if not 'metadata' in h5f.keys(): # make datasets if this is first time
                # create and write first metadata
                h5mdds = h5f.create_dataset(u'metadata',
                                    data = np.array([[image_data['metadata']['x'],
                                    image_data['metadata']['y'], image_data['metadata']['jdn']],]),
                                    maxshape = (None, 3))

                # create and write first original
                if save_originals:
                    h5ds = h5f.create_dataset(u'originals',
                                              data = np.expand_dims(image_pixeldata, 0),
                                              chunks = (1, image_pixeldata.shape[0], 30, 1),
                                              maxshape = (None, image_pixeldata.shape[0], image_pixeldata.shape[1], None),
                                              shuffle = True, compression = "gzip")
                    h5ds.attrs['plane_names'] = [p.encode('utf8') for p in plane_order]

                # find the channel locations for this fov. slice out channels and save
                for channel_loc in channel_masks[fov_id]:
                    # get id and slice
                    channel_id, channel_slice = mm3.cut_slice(image_pixeldata, channel_loc)
                    if compress_hdf5:
                        h5nds = h5f.create_dataset(u'channel_%04d' % channel_id,
                                                   data=np.expand_dims(channel_slice, 0),
                                                   chunks=(1, channel_slice.shape[0],
                                                           channel_slice.shape[1], 1),
                                                   maxshape=(None, channel_slice.shape[0],
                                                             channel_slice.shape[1], None),
                                                   shuffle=True, compression="gzip")
                    else:
                        h5nds = h5f.create_dataset(u'channel_%04d' % channel_id,
                                                   data=np.expand_dims(channel_slice, 0),
                                                   chunks=(1, channel_slice.shape[0],
                                                           channel_slice.shape[1], 1),
                                                   maxshape=(None, channel_slice.shape[0],
                                                             channel_slice.shape[1], None),
                                                   shuffle=True)
                    h5nds.attrs['channel_id'] = channel_loc[0]
                    h5nds.attrs['channel_loc'] = channel_loc[1]
                    h5nds.attrs['plane_names'] = [p.encode('utf8') for p in plane_order]

                returnvalue = True
            else: # append to current datasets
                #h5f.swmr_mode = False
                # write additional metadata
                h5mdds = h5f[u'metadata']
                h5mdds.resize(h5mdds.shape[0] + 1, axis = 0)
                h5mdds.flush()
                h5mdds[-1] = np.array([image_data['metadata']['x'],
                                    image_data['metadata']['y'], image_data['metadata']['jdn']])
                h5mdds.flush()

                # write additional originals
                if save_originals:
                    h5ds = h5f[u'originals']
                    # adjust plane names if need be
                    if len(h5ds.attrs['plane_names']) < len(plane_order):
                        h5ds.resize(len(plane_order, axis = 3))
                        h5ds.flush()
                        h5ds.attrs['plane_names'] = [p.encode('utf8') for p in plane_order]
                    h5ds.resize(h5ds.shape[0] + 1, axis = 0)
                    h5ds.flush()
                    h5ds[-1] = image_pixeldata
                    h5ds.flush()

                # write channels
                for channel_loc in channel_masks[fov_id]:
                    channel_id, channel_slice = mm3.cut_slice(image_pixeldata, channel_loc)

                    h5nds = h5f[u'channel_%04d' % channel_id]
                    # adjust plane names if need be
                    if len(h5nds.attrs['plane_names']) < len(plane_order):
                        h5nds.resize(len(plane_order), axis = 3)
                        h5nds.flush()
                        h5nds.attrs['plane_names'] = [p.encode('utf8') for p in plane_order]
                    h5nds.resize(h5nds.shape[0] + 1, axis = 0)
                    h5nds.flush()
                    h5nds[-1] = channel_slice
                    h5nds.flush()

                    # add to subtraction if it has been initiated and flag is on
                    if subtract and channel_id in cell_peaks:
                            # do subtraction
                            subtracted_data = subtract_phase([channel_slice, empty_mean])
                            subtracted_image = subtracted_data[0] # subtracted image
                            offset = subtracted_data[1]

                            # get the image data set for this channel and append sub image
                            h5si = h5s[u'subtracted_%04d' % channel_id]
                            h5si.resize(h5si.shape[0] + 1, axis = 0) # add a space fow new s image
                            h5si.flush()
                            h5si[-1] = subtracted_image
                            h5si.flush()

                            # adjust plane names if need be
                            if len(h5si.attrs['plane_names']) < (len(plane_order)+1):
                                h5si.resize(len(plane_order, axis = 3))
                                h5si.flush()
                                # rearrange plane names
                                plane_names = [p.encode('utf8') for p in plane_order]
                                plane_names.append(plane_names.pop(0)) # move phas to back
                                plane_names.insert(0, 'subtracted_phase') # put sub first
                                h5si.attrs['plane_names'] = plane_names # set new attribute

                            # add offset information
                            h5os = h5s[u'offsets_%04d' % channel_id]
                            h5os.resize(h5os.shape[0] + 1, axis = 0) # add a space fow new s image
                            h5os.flush()
                            h5os[-1] = offset
                            h5os.flush()

                # append metdata for subtraction
                if subtract:
                    sub_mds = h5s[u'metadata']
                    sub_mds.resize(sub_mds.shape[0] + 1, axis = 0)
                    sub_mds.flush()
                    sub_mds[-1] = np.array([image_data['metadata']['x'],
                                    image_data['metadata']['y'], image_data['metadata']['jdn']])
                    sub_mds.flush()

                    h5s.close() # close file (can't use `with` context manager here)

                returnvalue = True
    except:
        try:
            if h5s is not None:
                h5s.close()
        except:
            pass
        warning(sys.exc_info()[0])
        warning(sys.exc_info()[1])
        warning(traceback.print_tb(sys.exc_info()[2]))

    # release the write lock
    hdf5_locks[fov_id].release()

    return returnvalue

# class which responds to file addition events and saves their file names
if platform == "darwin":
    class tif_sentinal(PatternMatchingEventHandler):
        '''tif_sentinal will react to file additions that have .tif at the end.
        It only looks at file creations, not modifications or deletions.
        It inherits from the pattern match event handler in watchdog.
        There are two class attributes, events_filenames and events_buffer.
        events_filenames is the name of all added .tif files. Use the
        self.get_filenames method to return all events.
        events_buffer are just the filenames since last time the buffer was returned
        use self.get_buffer to return the buffer (which also clears the buffer).
        '''
        patterns = ["*.tif"]

        def __init__(self):
            # Makes sure to do the init function from the parent (super) class first
            super(tif_sentinal, self).__init__()
            self.events_filenames = [] # keeps all event filenames
            self.events_buffer = [] # keeps just event filenames since last call to buf

        def process(self, event):
            """
            called when an event (file change) occurs

            event.event_type
                'modified' | 'created' | 'moved' | 'deleted'
            event.is_directory
                True | False
            event.src_pquitath
                path/to/observed/file
            """
            filename = event.src_path.split('/')[-1] # get just the file name
            # append the file name to the class attributes for later retrieving
            self.events_filenames.append(filename)
            self.events_buffer.append(filename)

        def get_filenames(self):
            '''returns all event file names'''
            return self.events_filenames

        def get_buffer(self):
            '''returns all events since last call to print buffer'''
            # make a temporary buffer for printing
            buf = [x for x in self.events_buffer]
            self.events_buffer = [] # reset buffer
            return buf

        # could use this to do something on file modifications
        # def on_modified(self, event):
        #     self.process(event)

        def on_created(self, event):
            self.process(event)

# when using this script as a function and not as a library the following will execute
if __name__ == "__main__":
    # default values and flags, some to be overwritten
    param_file = ""
    mmm_image_end = -1
    external_master_file = ""
    multiple_sources = False
    source_directory = '.'
    min_timepoints_for_clusters = 30
    Goddard = False # flag for if goddard is scope. Extra check in extract_metadata
    save_originals = False # flag for if orignals should be saved to hdf5
    compress_hdf5 = True # flag for if images should be gzip compressed in hdf5
    clusters_created = False # flag for if clusters and channels masks should be made
    do_subtraction = True # flag for if subtraction should be performed if possible

    # the number of subprocesses is a balance between CPU and disk throughputs.
    # on an 8 core system, the IO overhead leaves the CPUs idle too much, so multiple
    # processes can be running on a single core. too many processes, though, will result
    # in degraded IO performance. on a higher-core system, the disk is limiting
    # is limiting - overhead on a single-file basis is too expensive to exceed 80-100
    # MB/s.
    cpu_count = multiprocessing.cpu_count()
    global num_blsubtract_subthreads
    if cpu_count == 32:
        num_writers = 10
        num_analyzers = 20
        num_blsubtractors = 4
        num_blsubtract_subthreads = 12
    elif cpu_count == 8:
        num_analyzers = 10
        num_writers = 7
        num_blsubtractors = 2
        num_blsubtract_subthreads = 8
    else:
        raise ValueError("host CPU count (%d) not in pre-determined utilization numbers (8, 32).")

    # switches
    # option f allows user to specfy the YAML based param_file
    try:
        opts, args = getopt.getopt(sys.argv[1:],"f:m:sxu")
    except getopt.GetoptError:
        print('no arguments detected (-f).')
    for opt, arg in opts:
        if opt == '-f':
            # parameter file
            param_file = arg
        if opt == '-m':
            external_master_file = arg
        if opt == '-s':
            use_parallel_write = False
        if opt == '-x':
            multiple_sources = True
        if opt == '-u':
            # if subtraction should not done if possible
            do_subtraction = False

    # Load the project parameters file
    # if the paramfile string has no length ie it has not been specified, ERROR
    if len(param_file) == 0:
        raise ValueError("a parameter file must be specified (-f <filename>).")
    information ('Loading experiment parameters...')
    globals().update(yaml.load(open(param_file, 'r'))) # load parameters into global
    mm3.init_mm3_helpers(param_file) # initialized the helper library
    source_directory = experiment_directory + image_directory # source of images

    # create the analysis folder if it doesn't exist
    if not os.path.exists(experiment_directory):
        os.makedirs(experiment_directory)
    if not os.path.exists(experiment_directory + analysis_directory):
        os.makedirs(experiment_directory + analysis_directory)
    if not os.path.exists(experiment_directory + analysis_directory + 'originals/'):
        os.makedirs(experiment_directory + analysis_directory + 'originals/')

    # if a kmeans fitter and channel_mask was previously saved to disk, load them
    if os.path.exists(experiment_directory + analysis_directory + 'kmeans_fitter.pkl') and \
        os.path.exists(experiment_directory + analysis_directory + 'channel_masks/channel_masks.pkl'):
        with open(experiment_directory + analysis_directory + 'kmeans_fitter.pkl', 'r') as kmp_fh:
            k_means = pickle.load(kmp_fh)
        with open(experiment_directory + analysis_directory +
                  'channel_masks/channel_masks.pkl', 'r') as kmp_fh:
            channel_masks = pickle.load(kmp_fh)
        clusters_created = True
        information('Loaded prior FOV clustering and channel masks.')

    # set up a dictionary of locks to prevent HDF5 disk collisions
    global hdf5_locks
    hdf5_locks = {x: Lock() for x in range(num_fovs)} # num_fovs is global parameter

    # multiprocessing system for analyzing & stacking newly uploaded image files
    m = Manager()
    q = m.Queue()
    event = Event() # controller to put processing on hold if the metadata backlog gets too big
    pool = Pool(num_analyzers, setup, (event,)) # pull for anlyzing metadata from images
    wpool = Pool(num_writers) # pool for writing out to hdf5 files
    subtract_backlog_pool = SubtractPool(num_blsubtractors)
    event.set() # set the pool running

    # declare information variables
    plane_order = [] # set up the plane ordering schema
    known_files = [] # filenames which have started the analysis pipeline
    failed_files = [] # filenames for which metadata extraction failed
    cp_result_dict = {} # for storing get_params pool results. Removed once moved to image_metadata
    image_metadata = {} # for storing image metadata from get_params. written images are removed
    w_result_dict = {} # for storing write objects set to write. Are removed once written

    # check for existing subtracted data and specs, etc. if true, start doing subtraction immediately.
    # this is not to account for starting subtraction as specs & empties get written - just to kick things off.
    '''
    if clusters_created and do_subtraction and \
        np.all([os.path.exists(experiment_directory + analysis_directory + 'empties/fov_%03d_emptymean.tif' % f)  for f in range(num_fovs)]) and \
        np.all([os.path.exists(experiment_directory + analysis_directory + 'subtracted/subtracted_%03d.hdf5' % f)  for f in range(num_fovs)]) and \
        np.all([os.path.exists(experiment_directory + analysis_directory + 'specs/specs_%03d.pkl' % f)  for f in range(num_fovs)]):
        subtract_on_datawrite = {f: True for f in range(num_fovs)} # T/F for each FOV which is passed to datawrite after backlog completes
        subtract_backlog_results = {f: True for f in range(num_fovs)} # stores the result objects of backlog subtraction processes;
        # subtract_backlog_results per-key value options:
        #   None (not started)
        #   True (completed OK)
        #   False (failed to complete)
        #   multiprocessing.pool.AsyncResult object (backlog still running or not yet cleaned up)
    else:
        subtract_backlog_results = {f: None for f in range(num_fovs)}
        subtract_on_datawrite = {f: False for f in range(num_fovs)}
    '''
    # setup subtraction tracking dicts
    # subtract_backlog_results per-key value options:
    #   None (not started)
    #   True (completed OK)
    #   False (failed to complete)
    #   multiprocessing.pool.AsyncResult object (backlog still running or not yet cleaned up)
    subtract_backlog_results = {f: None for f in range(num_fovs)}
    # subtract_on_datawrite per-key value options: True, False
    subtract_on_datawrite = {f: False for f in range(num_fovs)}

    known_files_last_save_size = 100 # counter for deciding when to save image metadata
    known_files_last_save_time = time.time() # timer for deciding when to save image metadata
    # counters
    successful_analysis_count = 0 # counter for number of successful image reads
    loop_analysis_count = 0 # for the last loop
    successful_write_count = 0 # counter for number of successful image writes
    loop_write_count = 0 # just for the loop

    # set up file change watcher
    try:
        if use_inotify:
            # gevent_inotify system for moving events between the monitor and __main__
            fd = inotify.init()
            # when using WinSCP to keep source_directory updated, the final system event
            # that gevent_inotifyx catches is an attribute correction; this is not an elegant
            # way to discover new files. the other events that pop up are a CREATE for the .tif
            # and a DELETE for the .filepart...
            wd = inotify.add_watch(fd, source_directory, inotify.IN_ATTRIB)
            # look for subdirs and add those
            image_subdirs = next(os.walk(source_directory))[1]
            fdsd = {sd: inotify.init() for sd in image_subdirs}
            for k in fdsd.keys():
                inotify.add_watch(fdsd[k], source_directory + k + "/", inotify.IN_ATTRIB)
                information('added subdirectory %s to monitor rotation.' % k)
        elif use_watchdog:
            # tiff_sentinal looks for tifs in path and
            event_handler = tif_sentinal()
            # obsever is the class from watchdog which gets system events.
            observer = Observer()
            # schedule the sentinal. recursive means look in subfolders in path too
            observer.schedule(event_handler, source_directory, recursive=True)
            observer.start() # start it up
    except:
        warning("Watching method failed.")
        warning(sys.exc_info()[0])
        warning(sys.exc_info()[1])
        warning(traceback.print_tb(sys.exc_info()[2]))
        pool.close()
        wpool.close()

    try:
        # get all the TIFFs that may already be in the folder
        known_files = sorted(glob.glob(source_directory + '*.tif'))
        source_subdirs = sorted(next(os.walk(source_directory))[1])
        for sd in source_subdirs:
            known_files.extend(sorted(glob.glob(source_directory + sd + '/*.tif')))

        # get information for all these starting tiffs
        if len(known_files) > 0:
            information("Priming with %d pre-existing files." % len(known_files))
            for fn in known_files:
                # get_params gets the image metadata and puts it in the cp_result dictionary
                # for each file name. True means look for channels
                cp_result_dict[fn] = pool.apply_async(get_params, [fn, True])

        # loop for processing images as they arrive
        information("Starting monitor-process loop.")
        try:
            count = 0
            while True:
                # go over the result objects to see if any are ready for writing
                # if they are indeed ready, send the result data to the writer
                collected = 0
                count += 1
                if count-1 % 1000 == 0:
                    t_s_loop = time.time() # timer for loop time

                # get that information from all the new images and move them from cp_result_dict to image_metadata dictionary
                for reskey in cp_result_dict.keys():
                    if cp_result_dict[reskey].ready():
                        collected += 1
                        cp_result = cp_result_dict[reskey].get() # get image data dict
                        del cp_result_dict[reskey] # and remove it from the result dict
                        # if result is a success, copy data to image_metadata and clean up
                        if cp_result['analyze_success']:
                            #information("analysis ok for " + cp_result['filename'].split("/")[-1])
                            successful_analysis_count += 1 # just add count instead of printing
                            loop_analysis_count += 1
                            image_metadata[cp_result['filename']] = cp_result # now put it in the official image_metadata dictionary

                            # make sure all planes in the image data are in the plane_order
                            # set whether or not to reorder for phase after compiling plane names
                            if len(plane_order) == 0:
                                reorder_for_phase = True
                                information("No plane data in global, setting to reorder.")
                            else:
                                reorder_for_phase = False
                            # add any new planes to the plane order
                            for pn in cp_result['metadata']['plane_names']:
                                if not pn in plane_order:
                                    plane_order.append(pn)
                            # if this was the first image to be analyzed, move the phase-contrast data to the front of plane_order
                            if reorder_for_phase:
                                try:
                                    information("Reordering planes...")
                                    for pn in plane_order:
                                        if pn.find("Ph") > -1 or pn.find("PH") > -1:
                                            phase_plane = pn
                                            break
                                    if phase_plane == "":
                                        raise ValueError("No plane name containing Ph or PH found: " + str(plane_order))
                                    # move the phase plane to index 0 in the list
                                    plane_order.insert(0, plane_order.pop(plane_order.index(phase_plane)))
                                    information("Planes ordered:", plane_order)
                                except:
                                    raise ValueError('Did not find phase plane information.')

                            # put in the plane name for everybody
                            image_metadata[cp_result['filename']]['write_plane_order'] = plane_order

                        else:
                            warning('Failed image analysis for ' + cp_result['filename'])
                            failed_files.append(cp_result['filename'])
                        # no need to do more than 100 here at the start
                        if collected > 100: break

                # run k-means clustering to start assigning FOV IDs
                if clusters_created:
                    # extract x, y, fn, and t for images that have not been assigned to an FOV
                    fns = [k for k, v in image_metadata.items() if v['fov'] < 0]
                    if len(fns) > 0:
                        x = [image_metadata[fn]['metadata']['x'] for fn in fns]
                        y = [image_metadata[fn]['metadata']['y'] for fn in fns]

                        positions = np.array(list(zip(x, y)))
                        #raw_input('wat')
                        labels = k_means.predict(positions) # finds k means positions after the original clusters have been determined

                        # put the fov into image_metadata for each image
                        for lindex in range(len(fns)):
                            image_metadata[fns[lindex]]['fov'] = labels[lindex]

                # make the k means cluster data if not made, need 30 per fov.
                # also make channel masks here
                elif not clusters_created and len(image_metadata) > min_timepoints_for_clusters * num_fovs:
                    # first try to load an existing fitter object

                    # find initial clusters
                    image_metadata, k_means = create_clusters(image_metadata)
                    clusters_created = True

                    # now find the channels and make the mask for slicing
                    channel_masks = make_masks(image_metadata)

                # if clusters have been created, subtraction is indicated, and an FOV has the
                # prerequisite files but has not started backlogged subtraction, start it.
                # if it was previously launch and has now completed, set it to True
                if clusters_created and do_subtraction:
                    for fov in subtract_backlog_results.keys():

                        # if the result is True or False, everything is done or there is nothing to do; skip it.
                        if type(subtract_backlog_results[fov]) == type(True):
                            continue

                        # if there is no result object, try to launch one.
                        elif subtract_backlog_results[fov] is None:
                            # only launch if prerequisites exist
                            if os.path.exists(experiment_directory + analysis_directory + 'empties/fov_%03d_emptymean.tif' % fov) and \
                                os.path.exists(experiment_directory + analysis_directory + 'specs/specs_%03d.pkl' % fov) and \
                                os.path.exists(experiment_directory + analysis_directory + 'originals/original_%03d.hdf5' % fov):

                                # make subtracted folder if it does not exists
                                if not os.path.exists(experiment_directory + analysis_directory + 'subtracted/'):
                                    os.makedirs(os.path.abspath(experiment_directory + analysis_directory + "subtracted/"))

                                # subtract those backgrounds!
                                information('Started background subtraction for FOV %03d.' % fov)
                                subtract_backlog_results[fov] = subtract_backlog_pool.apply_async(subtract_backlog, [fov,])
                                subtract_on_datawrite[fov] = True

                        # if the result was not True or None, it should be a result object or False.
                        # if the result is ready, process it.
                        elif subtract_backlog_results[fov].ready():
                            sb_result = subtract_backlog_results[fov].get()
                            # if the result is 0, processing was successful; keep going.
                            if sb_result == 0:
                                subtract_backlog_results[fov] = True
                                information('Completed backlogged subtractions for FOV %d.' % fov)
                            # if the result is not 0, something went wrong; clean up.
                            else:
                                # deal with dangling data_writers by waiting for locks to clear and canceling subtract-on-write.
                                lock_acquired = hdf5_locks[fov].acquire(block = True)
                                hdf5_locks[fov_id].release()
                                subtract_on_datawrite[fov] = False
                                # if the result is 1, warn the user and block future backlog attempts.
                                if sb_result == 1:
                                    subtract_backlog_results[fov] = False
                                    warning('Failed backlog subtraction; no future subtraction for FOV %d.' % fov)
                                # if the result is not 0 or 1, stop the show.
                                else:
                                    subtract_backlog_results[fov] = False
                                    raise ValueError('No idea what happened in backlog subtraction for FOV %d. Good luck!' % fov)
                        elif not subtract_backlog_results[fov].ready():
                            information('subtraction backlog for FOV %d not yet ready.' % fov)
                        else:
                            raise ValueError('subtract_backlog_results[%d] in unknown state (%s).' % (fov, str(subtract_backlog_results[fov])))

                # check write results and set the success flag as appropriate in the metadata.
                # this makes sure things don't get written twice. writing happens next.
                # print('dummy')
                for wfn in w_result_dict.keys():
                    if w_result_dict[wfn].ready():
                        w_result = w_result_dict[wfn].get()
                        del w_result_dict[wfn]
                        if w_result:
                            image_metadata[wfn]['write_success'] = True
                            successful_write_count += 1 # just add to count
                            loop_write_count += 1
                            #information("wrote to originals hdf5 %s" % wfn.split("/")[-1])
                            del image_metadata[wfn]
                        else:
                            image_metadata[wfn]['sent_to_write'] = False
                            information("Failed to write %s" % wfn.split("/")[-1])

                t_s_inner = time.time()

                # send writes to the pool based on the lowest jdn not yet written after queued writes clear
                # for big existing file lists, this is slow like molasses
                if clusters_created: # only write if cluster FOVs have been assigned
                    # get a list of lists of all the fovs not yet written, jdn is the julian date time (exact time)
                    def get_not_written_for_fov(fov):
                        return [[k, v['metadata']['jdn']] for k, v in image_metadata.items() if v['fov'] == fov and not v['write_success'] and v['write_plane_order']]
                    fov_ready_for_write = map(get_not_written_for_fov, range(num_fovs))

                    # writing is done here
                    for fov_fns_jdns in fov_ready_for_write:
                        if len(fov_fns_jdns) > 0:
                            # check if any of these filenames have been sent to write and haven't yet been processed for exit status
                            waiting_on_write = np.sum([fn in w_result_dict.keys() for fn, fjdn in fov_fns_jdns]) > 0
                            # if no files are waiting on write, send the next image for write
                            if not waiting_on_write:
                                # sort the filenames-jdns by jdn
                                fov_fns_jdns = sorted(fov_fns_jdns, key=lambda imd: imd[1])
                                # data_writer is the major hdf5 writing function.
                                # not switched for saving originals and doing subtraction
                                w_result_dict[fov_fns_jdns[0][0]] = wpool.apply_async(data_writer,
                                                            [image_metadata[fov_fns_jdns[0][0]],
                                                            channel_masks, subtract_on_datawrite[image_metadata[fov_fns_jdns[0][0]]['fov']],
                                                            save_originals])
                                image_metadata[fov_fns_jdns[0][0]]['sent_to_write'] = True

                t_e_inner = time.time() - t_s_inner

                # if the loop takes more than a second to run, give it a break
                if event.is_set() and t_e_inner > 0.5:
                    information('Metadata list (%d) analysis exceeded 0.5 second, pausing metadata extraction.' % len(image_metadata))
                    event.clear()
                if not event.is_set() and t_e_inner < 0.1:
                    information('Metadata list (%d) analysis dropped below 0.1 seconds, resuming metadata extraction.' % len(image_metadata))
                    event.set()

                # get new file creation events
                if use_inotify: # for inotify in linux
                    inotify_events = inotify.get_events(fd, timeout = 0)
                    file_events = [i_event.name for i_event in inotify_events]
                    # if subdirectories are being watched, check them for events and add those to the list
                    if len(fdsd.keys()) > 0:
                        for k in fdsd.keys():
                            inotify_sd_events = inotify.get_events(fdsd[k], timeout = 0)
                            file_events.extend([k + "/" + i_event.name for i_event in inotify_sd_events])
                    # check for new subdirectories to watch and add them if they aren't being watched
                    source_subdirs = next(os.walk(source_directory))[1]
                    for sd in source_subdirs:
                        if not sd in fdsd.keys():
                            fdsd[sd] = inotify.init()
                            inotify.add_watch(fdsd[sd], source_directory + sd + "/", inotify.IN_ATTRIB)
                            information('added subdirectory %s to monitor rotation.' % sd)
                elif use_watchdog: # use watchdog for mac
                    file_events = event_handler.get_buffer()

                # start metadata analysis for new files
                for newfname in file_events:
                    if newfname.split(".")[-1] == "tif" and not source_directory + newfname in known_files:
                        information("discovered " + newfname)
                        if clusters_created:
                            # don't look for channels
                            cp_result_dict[newfname] = pool.apply_async(get_params, [source_directory + newfname, False])
                        else:
                            # do look for channels
                            cp_result_dict[newfname] = pool.apply_async(get_params, [source_directory + newfname, True])
                        known_files.append(source_directory + newfname)

                # write the list of known files to disk
                # marshal is 10x faster and 50% more space efficient than cPickle here, but dtypes
                # are all python primitives
                if len(known_files) >= known_files_last_save_size * 1.1 or time.time() - known_files_last_save_time > 900:
                    with open(experiment_directory + analysis_directory + 'known_files.mrshl', 'w') as outfile:
                        marshal.dump(known_files, outfile)
                    known_files_last_save_time = time.time()
                    known_files_last_save_size = len(known_files) * 1.1
                    information("Saved intermediate known_files (%d)." % len(known_files))

                # update user current progress
                if count % 1000 == 0:
                    information("1000 loop time %0.2fs, running metadata total: %d" % (time.time() - t_s_loop, len(image_metadata)))
                    information("Analysis ok for %d images (%d total)." %
                                (loop_analysis_count, successful_analysis_count))
                    loop_analysis_count = 0 # reset counter
                    information("Wrote %d images to hdf5 (%d total)." %
                                (loop_write_count, successful_write_count))
                    loop_write_count = 0 # reset loop counter
                    count = 0

                    # if there's nothing going on, don't hog the CPU
                    if (len(cp_result_dict) == 0 and len(w_result_dict) == 0 and
                        len(image_metadata) == 0):
                        information("Queues are empty; waiting 5 seconds to loop.")
                        information("%d images analyzed, %d written to hdf5." %
                                    (successful_analysis_count, successful_write_count))
                        time.sleep(5)

        except KeyboardInterrupt:
                warning("Caught KeyboardInterrupt, terminating workers...")
                wpool.close()
                pool.close()
                subtract_backlog_pool.close()
                wpool.terminate()
                pool.terminate()
                subtract_backlog_pool.terminate()

    except:
        warning("Try block failed.")
        print(sys.exc_info()[0])
        print(sys.exc_info()[1])
        print(traceback.print_tb(sys.exc_info()[2]))
        pool.close()
        wpool.close()
        subtract_backlog_pool.close()

    finally:
        if use_inotify:
            os.close(fd)
        if use_watchdog:
            observer.stop()
