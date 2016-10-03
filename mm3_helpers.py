#!/usr/bin/python
from __future__ import print_function
def warning(*objs):
    print(time.strftime("%H:%M:%S", time.localtime()), *objs, file=sys.stderr)
def information(*objs):
    print(time.strftime("%H:%M:%S", time.localtime()), *objs, file=sys.stdout)

# import modules
import sys
import os
import time
import inspect
import yaml
try:
    import cPickle as pickle # pickle
except:
    import pickle
import numpy as np
import scipy.signal as spsig
import scipy.stats as spstats
import struct # for interpretting strings as binary data
import re # regular expressions
import traceback
import copy
from scipy import ndimage # used in make_masks
from skimage.segmentation import clear_border # used in make_masks
from skimage.feature import match_template # used to align images

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

### functions ###########################################################
# load the parameters file into a global dictionary for this module
def init_mm3_helpers(param_file_path):
    # load all the parameters into a global dictionary
    global params
    with open(param_file_path, 'r') as param_file:
        params = yaml.safe_load(param_file)
    return

### functions about loading files
# loading the channel id of cell containing peaks
def load_cell_peaks(fov_id):
    '''Returns and array of the cell peaks from the spec file for a given fov number'''
    exp_dir = params['experiment_directory']
    ana_dir = params['analysis_directory']
    if not os.path.exists(exp_dir + ana_dir + 'specs/specs_%03d.pkl' % fov_id):
        warning("Spec file missing for " + fov_id)
        return -1
    else:
        with open(exp_dir + ana_dir + 'specs/specs_%03d.pkl' % fov_id, 'rb') as pkl_file:
            user_picks = pickle.load(pkl_file) # tuple = (drop_peaks, cell_peaks, bgrd_peaks)
    # get the cell-containing peaks
    cell_peaks = user_picks[1]
    cell_peaks = np.array(cell_peaks) # it is currently just a list of ints
    return cell_peaks

# load empty tif for an fov_id (empty for background subtraction)
def load_empty_tif(fov_id):
    exp_dir = params['experiment_directory']
    ana_dir = params['analysis_directory']
    if not os.path.exists(exp_dir + ana_dir + "empties/fov_%03d_emptymean.tif" % fov_id):
        warning("Empty mean .tif file missing for " + fov_id)
        return -1
    else:
        empty_mean = tiff.imread(exp_dir + ana_dir + "empties/fov_%03d_emptymean.tif" % fov_id)
    return empty_mean

# finds metdata in a tiff image.
def get_tif_metadata_elements(tif):
    '''This function pulls out the metadata from a tif file and returns it as a dictionary.
    This if tiff files as exported by Nikon Elements as a stacked tiff, each for one tpoint.
    tif is an opened tif file (using the package tifffile)


    arguments:
        fname (tifffile.TiffFile): TIFF file object from which data will be extracted
    returns:
        dictionary of values:
            'jdn' (float)
            'x' (float)
            'y' (float)
            'plane_names' (list of strings)

    Called by
    mm3.get_tif_params

    '''

    # image Metadata
    idata = { 'fov': -1,
              't' : -1,
              'jd': -1 * 0.0,
              'x': -1 * 0.0,
              'y': -1 * 0.0,
              'planes': []}

    # get the fov and t simply from the file name
    idata['fov'] = int(tif.fname.split('xy')[1].split('.tif')[0])
    idata['t'] = int(tif.fname.split('xy')[0].split('t')[-1])

    # a page is plane, or stack, in the tiff. The other metdata is hidden down in there.
    for page in tif:
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
                idata['jd'] = float(struct.unpack('<d', b)[0])

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

                planes = []
                for idx in [i for i, x in enumerate(words) if x == "sOpticalConfigName"]:
                    planes.append(words[idx+1])

                idata['planes'] = planes

    return idata

# finds the location of channels in a tif
def find_channel_locs(image_data):
    '''Finds the location of channels from a phase contrast image. The channels are returned in
    a dictionary where the key is the x position of the channel in pixel and the value is a
    dicionary with the open and closed end in pixels in y.


    Called by
    get_tif_params

    '''

    try:
        # declare temp variables from yaml parameter dict.
        chan_w = params['channel_width']
        chan_midline_sep = params['channel_midline_separation']
        crop_hw = params['crop_half_width']
        chan_snr = params['channel_detection_snr']

        # structure for channel dimensions
        channel_params = []
        # Structure of channel_params
        # 0 = peak position ('peak_px')
        # 1 = index of max (the closed end)
        # 2 = index of min (the open end)
        # 3 = length of the channel (min - max)

        # Detect peaks in the x projection (i.e. find the channels)
        projection_x = image_data.sum(axis = 0)
        # find_peaks_cwt is a function which attempts to find the peaks in a 1-D array by
        # convolving it with a wave. here the wave is the default wave used by the algorythm
        # but the minimum signal to noise ratio is specified
        peaks = spsig.find_peaks_cwt(projection_x, np.arange(chan_w-5,chan_w+5),
                                     min_snr=chan_snr)

        # If the left-most peak position is within half of a channel separation,
        # discard the channel from the list.
        if peaks[0] < (chan_midline_sep / 2):
            peaks = peaks[1:]
        # If the difference between the right-most peak position and the right edge
        # of the image is less than half of a channel separation, discard the channel.
        if image_data.shape[0] - peaks[len(peaks)-1] < (chan_midline_sep / 2):
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
            channel_strips.append([peak, image_data[0:image_data.shape[0],
                                   peak - crop_hw:peak + crop_hw]])

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

        return cp_dict

    except:
        warning('Channel locating failed.')
        print(sys.exc_info()[0])
        print(sys.exc_info()[1])
        print(traceback.print_tb(sys.exc_info()[2]))
        return -1

# make masks from initial set of images (same images as clusters)
def make_masks(analyzed_imgs):
    '''
    Make masks goes through the channel locations in the image metadata and builds a consensus
    Mask for each image per fov, which it returns as dictionary named channel_masks.
    The keys in this dictionary are fov id, and the values is a another dictionary. This dict's keys are channel locations (peaks) and the values is a [2][2] array:
    [[minrow, maxrow],[mincol, maxcol]] of pixel locations designating the corner of each mask
    for each channel on the whole image

    One important consequence of these function is that the channel ids and the size of the
    channel slices are decided now. Updates to mask must coordinate with these values.

    Parameters
    analyzed_imgs : dict
        image information created by get_params

    Returns
    channel_masks : dict
        dictionary of consensus channel masks.

    Called By
    mm3_Compile.py

    Calls
    '''
    information("Determining initial channel masks...")

    # declare temp variables from yaml parameter dict.
    crop_hw = params['crop_half_width']
    chan_lp = params['channel_length_pad']

    #intiaize dictionary
    channel_masks = {}

    # get the size of the images (hope they are the same)
    for img_k, img_v in analyzed_imgs.iteritems():
        image_rows = img_v['shape'][0] # x pixels
        image_cols = img_v['shape'][1] # y pixels
        break # just need one. using iteritems mean the whole dict doesn't load

    # get the fov ids
    fovs = []
    for img_k, img_v in analyzed_imgs.iteritems():
        if img_v['fov'] not in fovs:
            fovs.append(img_v['fov'])

        # if you have found as many fovs as params indicates break out
        if len(fovs) == params['num_fovs']:
            break

    # max width and length across all fovs. channels will get expanded by these values
    # this important for later updates to the masks, which should be the same
    max_chnl_mask_len = 0
    max_chnl_mask_wid = 0

    # for each fov make a channel_mask dictionary from consensus mask for each fov
    for fov in fovs:
        # initialize a the dict and consensus mask
        channel_masks_1fov = {} # dict which holds channel masks {peak : [[y1, y2],[x1,x2]],...}
        consensus_mask = np.zeros([image_rows, image_cols]) # mask for labeling

        # bring up information for each image
        for img_k, img_v in analyzed_imgs.iteritems():
            # skip this one if it is not of the current fov
            if img_v['fov'] != fov:
                continue

            # for each channel in each image make a single mask
            img_chnl_mask = np.zeros([image_rows, image_cols])

            # and add the channel mask to it
            for chnl_peak, peak_ends in img_v['channels'].iteritems():
                # pull out the peak location and top and bottom location
                # and expand by padding (more padding done later for width)
                x1 = max(chnl_peak - int(crop_hw), 0)
                x2 = min(chnl_peak + int(crop_hw), image_cols)
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
        # clear border it to make sure the channels are off the edge
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
            min_row = np.min(np.where(posrows)[0]) - chan_lp # pad length
            max_row = np.max(np.where(posrows)[0]) + chan_lp
            min_col = max(np.min(np.where(poscols)[0]) - 5, 0) # pad width
            max_col = min(np.max(np.where(poscols)[0]) + 5, image_cols)

            # if the min/max cols are within the image bounds,
            # add the mask, as 4 points, to the dictionary
            if min_col > 0 and max_col < image_cols:
                channel_masks_1fov[channel_id] = [[min_row, max_row], [min_col, max_col]]

                # find the largest channel width and height while you go round
                max_chnl_mask_len = int(max(max_chnl_mask_len, max_row - min_row))
                max_chnl_mask_wid = int(max(max_chnl_mask_wid, max_col - min_col))

        # add channel_mask dictionary to the fov dictionary, use copy to play it safe
        channel_masks[fov] = channel_masks_1fov.copy()

    # update all channel masks to be the max size
    cm_copy = channel_masks.copy()

    for fov, peaks in channel_masks.iteritems():
        # f_id = int(fov)
        for peak, chnl_mask in peaks.iteritems():
            # p_id = int(peak)
            # just add length to the open end (top of image, low column)
            if chnl_mask[0][1] - chnl_mask[0][0] !=  max_chnl_mask_len:
                cm_copy[fov][peak][0][1] = chnl_mask[0][0] + max_chnl_mask_len
            # enlarge widths around the middle, but make sure you don't get floats
            if chnl_mask[1][1] - chnl_mask[1][0] != max_chnl_mask_wid:
                wid_diff = max_chnl_mask_wid - (chnl_mask[1][1] - chnl_mask[1][0])
                if wid_diff % 2 == 0:
                    cm_copy[fov][peak][1][0] = max(chnl_mask[1][0] - wid_diff/2, 0)
                    cm_copy[fov][peak][1][1] = min(chnl_mask[1][1] + wid_diff/2, image_cols - 1)
                else:
                    cm_copy[fov][peak][1][0] = max(chnl_mask[1][0] - (wid_diff-1)/2, 0)
                    cm_copy[fov][peak][1][1] = min(chnl_mask[1][1] + (wid_diff+1)/2, image_cols - 1)

    return cm_copy

### functions about trimming, padding, and manipulating images
# cuts out a channel from an tiff image (that has been processed)
# define function for flipping the images on an FOV by FOV basis
def fix_orientation(image_data):
    '''
    Fix the orientation. The standard direction for channels to open to is down.

    called by
    process_tif
    get_params
    '''

    # user parameter indicates how things should be flipped
    image_orientation = params['image_orientation']

    # if this is just a phase image give in an extra layer so rest of code is fine
    flat = False # flag for if the image is flat or multiple levels
    if len(image_data.shape) == 2:
        image_data = np.expand_dims(image_data, 0)
        flat = True

    # setting image_orientation to 'auto' will use autodetection
    if image_orientation == "auto":
        # crop image to just the area of the channels
        if params['image_vertical_crop'] >= 0:
            image_data = image_data[:,params['image_vertical_crop']:image_data.shape[1] -
                                      params['image_vertical_crop'],:]

        # Pick the plane to analyze with the highest mean px value (should be phase)
        ph_channel = np.argmax([np.mean(image_data[ci]) for ci in range(image_data.shape[0])])

        # flip based on the index of the higest average row value
        # this should be closer to the opening
        if np.argmax(image_data[ph_channel].mean(axis = 1)) < image_data[ph_channel].shape[0] / 2:
            image_data = image_data[:,::-1,:]
        else:
            pass # no need to do anything

    # flip if up is chosen
    elif image_orientation == "up":
        return image_data[:,::-1,:]

    # do not flip the images if "down is the specified image orientation"
    elif image_orientation == "down":
        pass

    if flat:
        image_data = image_data[0] # just return that first layer

    return image_data

# cuts out channels from the image
def cut_slice(image_data, channel_loc):
    '''Takes an image and cuts out the channel based on the slice location
    slice location is the list with the peak information, in the form
    [][y1, y2],[x1, x2]]. Returns the channel slice as a numpy array.
    The numpy array will be a stack if there are multiple planes.

    if you want to slice all the channels from a picture with the channel_masks
    dictionary use a loop like this:

    for channel_loc in channel_masks[fov_id]: # fov_id is the fov of the image
        channel_slice = cut_slice[image_pixel_data, channel_loc]
        # ... do something with the slice

    NOTE: this function will try to determine what the shape of your
    image is and slice accordingly. It expects the images are in the order
    [t, x, y, c]. It assumes images with three dimensions are [x, y, c] not
    [t, x, y].
    '''

    # case where image is in form [x, y]
    if len(image_data.shape) == 2:
        # make slice object
        channel_slicer = np.s_[channel_loc[0][0]:channel_loc[0][1],
                               channel_loc[1][0]:channel_loc[1][1]]

    # case where image is in form [x, y, c]
    elif len(image_data.shape) == 3:
        channel_slicer = np.s_[channel_loc[0][0]:channel_loc[0][1],
                               channel_loc[1][0]:channel_loc[1][1],:]

    # case where image in form [t, x , y, c]
    elif len(image_data.shape) == 4:
        channel_slicer = np.s_[:,channel_loc[0][0]:channel_loc[0][1],
                                 channel_loc[1][0]:channel_loc[1][1],:]

    # slice based on appropriate slicer object.
    channel_slice = image_data[channel_slicer]

    return channel_slice

# remove margins of zeros from 2d numpy array
def trim_zeros_2d(array):
    # make the array equal to the sub array which has columns of all zeros removed
    # "all" looks along an axis and says if all of the valuse are such and such for each row or column
    # ~ is the inverse operator
    # using logical indexing
    array = array[~np.all(array == 0, axis = 1)]
    # transpose the array
    array = array.T
    # make the array equal to the sub array which has columns of all zeros removed
    array = array[~np.all(array == 0, axis = 1)]
    # transpose the array again
    array = array.T
    # return the array
    return array

# calculat cross correlation between pixels in channel stack
def channel_xcorr(channel_filepath):
    '''
    Function calculates the cross correlation of images in a
    stack to the first image in the stack. The output is an
    array that is the length of the stack with the best cross
    correlation between that image and the first image.

    The very first value should be 1.
    '''

    # load up the stack. should be 4D [t, x, y, c]
    with tiff.TiffFile(channel_filepath) as tif:
        image_data = tif.asarray()

    # just use the first plane, which should be the phase images
    image_data = image_data[:,:,:,0]

    # if there are more than 100 images, use 100 images evenly
    # spaced across the range
    if image_data.shape[0] > 100:
        spacing = np.floor(image_data.shape[0] / 100)
        image_data = image_data[::spacing,:,:]
        if image_data.shape[0] > 100:
            image_data = image_data[:100,:,:]

    # we will compare all images to this one
    first_img = image_data[0,:,:]

    xcorr_array = [] # array holds cross correlation vaues
    for img in image_data[1:,:,:]:
        # use match_template to find all cross correlations for the
        # current image against the first image.
        xcorr_array.append(np.max(match_template(img, first_img)))

    return xcorr_array

### functions about subtraction
# worker function for doing subtraction
# average empty channels from stacks, making another TIFF stack
def average_empties_stack(fov_id, specs):
    '''Takes the fov file name and the peak names of the designated empties,
    averages them and saves the image

    Parameters
    fov_id : int
        FOV number
    specs : dict
        specifies whether a channel should be analyzed (1), used for making
        an average empty (0), or ignored (-1).

    Returns
        True if succesful.
        Saves empty stack to analysis folder

    '''

    information("Creating average empty channel for FOV %d." % fov_id)

    # directories for saving
    chnl_dir = params['experiment_directory'] + params['analysis_directory'] + 'channels/'
    empty_dir = params['experiment_directory'] + params['analysis_directory'] + 'empties/'

    # get peak ids of empty channels for this fov
    empty_peak_ids = []
    for peak_id, spec in specs[fov_id].items():
        if spec == 0: # 0 means it should be used for empty
            empty_peak_ids.append(peak_id)
    empty_peak_ids = sorted(empty_peak_ids) # sort for repeatability

    # depending on how many empties there are choose what to do
    # if there is no empty the user is going to have to copy another empty stack
    if len(empty_peak_ids) == 0:
        information("No empty channel designated for FOV %d." % fov_id)
        return False

    # if there is just one then you can just copy that channel
    elif len(empty_peak_ids) == 1:
        peak_id = empty_peak_ids[0]
        information("One empty channel (%d) designated for FOV %d." % (peak_id, fov_id))

        # copy that tiff stack with a new name as empty
        channel_filename = params['experiment_name'] + '_xy%03d_p%04d.tif' % (fov_id, peak_id)
        channel_filepath = chnl_dir + channel_filename # chnl_dir read from scope above

        with tiff.TiffFile(channel_filepath) as tif:
            avg_empty_stack = tif.asarray()

        return True

    # but if there is more than one empty you need to align and average them per timepoint
    elif len(empty_peak_ids) > 1:
        # load the image stacks into memory
        empty_stacks = [] # list which holds phase image stacks of designated empties
        for peak_id in empty_peak_ids:
            # load stack
            channel_filename = params['experiment_name'] + '_xy%03d_p%04d.tif' % (fov_id, peak_id)
            channel_filepath = chnl_dir + channel_filename # chnl_dir read from scope above
            with tiff.TiffFile(channel_filepath) as tif:
                image_data = tif.asarray()

            # just get phase data and put it in list
            image_data = image_data[:,:,:,0]
            empty_stacks.append(image_data)

        information("%d empty channels designated for FOV %d." % (len(empty_stacks), fov_id))

        # go through time points and create list of averaged empties
        avg_empty_stack = [] # list will be later concatentated into numpy array
        time_points = range(image_data.shape[0]) # index is time
        for t in time_points:
            # get images from one timepoint at a time and send to alignment and averaging
            imgs = [stack[t] for stack in empty_stacks]
            avg_empty = average_empties(imgs) # function is in mm3
            avg_empty = np.expand_dims(avg_empty, 0) # add dimension for time
            avg_empty_stack.append(avg_empty)

        # concatenate list and then save out to tiff stack
        avg_empty_stack = np.concatenate(avg_empty_stack, axis=0)

    # save out data
    # make new name, empty_dir should be found in above scope
    empty_filename = params['experiment_name'] + '_xy%03d_empty.tif' % fov_id
    empty_filepath = empty_dir + empty_filename

    tiff.imsave(empty_filepath, avg_empty_stack) # save it

    information("Saved empty channel %s." % empty_filename)

    return True


# averages a list of empty channels
def average_empties(imgs):
    '''
    This function averages a set of images (empty channels) and returns a single image
    of the same size. It first aligns the images to the first image before averaging.

    Alignment is done by enlarging the first image using edge padding.
    Subsequent images are then aligned to this image and the offset recorded.
    These images are padded such that they are the same size as the first (padde) image but
    with the image in the correct (aligned) place. Edge padding is again used.
    The images are then placed in a stack and aveaged. This image is trimmed so it is the size
    of the original images

    Called by
    average_empties_stack

    '''

    aligned_imgs = [] # list contains the alingned, padded images
    pad_size = 10 # pixel size to use for padding (ammount that alignment could be off)

    for n, img in enumerate(imgs):
        # if this is the first image, pad it and add it to the stack
        if n == 0:
            ref_img = np.pad(img, pad_size, mode='edge') # padded reference image
            aligned_imgs.append(ref_img)

        # otherwise align this image to the first padded image
        else:
            # find correlation between a convolution of img against the padded reference
            match_result = match_template(ref_img, img)

            # find index of highest correlation (relative to top left corner of img)
            y, x = np.unravel_index(np.argmax(match_result), match_result.shape)

            # pad img so it aligns and is the same size as reference image
            pad_img = np.pad(img, ((y, ref_img.shape[0] - (y + img.shape[0])),
                                   (x, ref_img.shape[1] - (x + img.shape[1]))), mode='edge')
            aligned_imgs.append(pad_img)

    # stack the aligned data along 3rd axis
    aligned_imgs = np.dstack(aligned_imgs)
    # get a mean image along 3rd axis
    avg_empty = np.nanmean(aligned_imgs, axis=2)
    # trim off the padded edges
    avg_empty = avg_empty[pad_size:-1*pad_size, pad_size:-1*pad_size]
    # change type back to unsigned 16 bit not floats
    avg_empty = avg_empty.astype(dtype='uint16')

    return avg_empty


#
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

# for doing subtraction when just starting and there is a backlog
def subtract_backlog(fov_id):
    return_value = -1
    try:
        information('subtract_backlog: Subtracting backlog of images FOV %03d.' % fov_id)

        # load cell peaks and empty mean
        cell_peaks = mm3.load_cell_peaks(fov_id) # load list of cell peaks from spec file
        empty_mean = mm3.load_empty_tif(fov_id) # load empty mean image
        empty_mean = mm3.trim_zeros_2d(empty_mean) # trim any zero data from the image
        information('subtract_backlog: There are %d cell peaks for FOV %03d.' % (len(cell_peaks), fov_id))

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
                information('subtract_backlog: Subtraction started for FOV %d, peak %04d.' % (fov_id, peak))

                # just loop around waiting for the peak to be done.
                try:
                    while (True):
                        time.sleep(1)
                        if pool_result.ready():
                            break
                    # inform user once this is done
                    information("subtract_backlog: Completed peak %d in FOV %03d (%d timepoints)" %
                                (peak, fov_id, len(images_with_empties)))
                except KeyboardInterrupt:
                    raise

                if not pool_result.successful():
                    warning('subtract_backlog: Processing pool not successful for peak %d.' % peak)
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
                                              compression="gzip", shuffle=True)

                    # rearrange plane names
                    plane_names = plane_names.tolist()
                    plane_names.append(plane_names.pop(0))
                    plane_names.insert(0, 'subtracted_phase')
                    h5si.attrs['plane_names'] = plane_names

                    # create dataset for offset information
                    # create and write first metadata
                    h5os = h5s.create_dataset(u'offsets_%04d' % peak,
                                              data=np.array(offsets),
                                              maxshape=(None, 2))

            # move over metadata once peaks have all peaks have been written
            with h5py.File(experiment_directory + analysis_directory + 'subtracted/subtracted_%03d.hdf5' % fov_id, 'a', libver='earliest') as h5s:
                sub_mds = h5s.create_dataset("metadata", data=h5f[u'metadata'],
                                             maxshape=(None, 3))

        # return 0 to the parent loop if everything was OK
        return_value = 0
    except:
        warning("subtract_backlog: Failed for FOV: %03d" % fov_id)
        warning(sys.exc_info()[1])
        warning(traceback.print_tb(sys.exc_info()[2]))
        return_value = 1

    # release lock
    hdf5_locks[fov_id].release()

    return return_value

### functions about converting dates and times
### Functions
def days_to_hmsm(days):
    hours = days * 24.
    hours, hour = math.modf(hours)
    mins = hours * 60.
    mins, min = math.modf(mins)
    secs = mins * 60.
    secs, sec = math.modf(secs)
    micro = round(secs * 1.e6)
    return int(hour), int(min), int(sec), int(micro)

def hmsm_to_days(hour=0, min=0, sec=0, micro=0):
    days = sec + (micro / 1.e6)
    days = min + (days / 60.)
    days = hour + (days / 60.)
    return days / 24.

def date_to_jd(year,month,day):
    if month == 1 or month == 2:
        yearp = year - 1
        monthp = month + 12
    else:
        yearp = year
        monthp = month
    # this checks where we are in relation to October 15, 1582, the beginning
    # of the Gregorian calendar.
    if ((year < 1582) or
        (year == 1582 and month < 10) or
        (year == 1582 and month == 10 and day < 15)):
        # before start of Gregorian calendar
        B = 0
    else:
        # after start of Gregorian calendar
        A = math.trunc(yearp / 100.)
        B = 2 - A + math.trunc(A / 4.)
    if yearp < 0:
        C = math.trunc((365.25 * yearp) - 0.75)
    else:
        C = math.trunc(365.25 * yearp)
    D = math.trunc(30.6001 * (monthp + 1))
    jd = B + C + D + day + 1720994.5
    return jd

def datetime_to_jd(date):
    days = date.day + hmsm_to_days(date.hour,date.minute,date.second,date.microsecond)
    return date_to_jd(date.year, date.month, days)
