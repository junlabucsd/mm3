#!/usr/bin/python
from __future__ import print_function
def warning(*objs):
    print(time.strftime("%H:%M:%S WARNING:", time.localtime()), *objs, file=sys.stderr)
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
# import struct
import re
import glob
import gevent
import marshal
import json # used to write data out in human readable format
try:
    import cPickle as pickle
except:
    import pickle
from sys import platform as platform
import multiprocessing
from multiprocessing import Pool #, Manager, Lock
import numpy as np
import numpy.ma as ma
from scipy import ndimage
#import scipy.signal as spsig
# import scipy.stats as spstats
# from skimage.feature import match_template
# from sklearn.cluster import KMeans
# from skimage.exposure import rescale_intensity, equalize_hist
from skimage.segmentation import clear_border

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
# from subtraction_helpers import *
import mm3_helpers as mm3

# debug modules
#import pprint as pp
#import matplotlib.pyplot as plt

# make masks from initial set of images (same images as clusters)
def make_masks(analyzed_imgs):
    '''
    Make masks goes through the channel locations in the image metadata and builds a consensus
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
    for img_k, img_v in analyzed_imgs.iteritems():
        image_rows = img_v['image_size'][0]
        image_cols = img_v['image_size'][1]
        break # just need one. using iteritems mean the whole dict doesn't load

    # get the fov ids
    fovs = []
    for img_k, img_v in analyzed_imgs.iteritems():
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

    return channel_masks

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

# # writer function for appending to originals_nnn.hdf5
# def data_writer(image_data, channel_masks, subtract=False, save_originals=False, compress_hdf5=True):
#     '''
#     data_writer saves an hdf5 file for each fov which contains the original images for that fov,
#     the meta_data about the images (location, time), and the sliced channels based on the
#     channel_masks.
#
#     Called by:
#     __main__
#
#     Calls
#     mm3.cut_slice
#     process_tif
#     mm3.load_cell_peaks
#     mm3.load_empyt_tif
#     mm3.trim_zeros_2d
#     '''
#     returnvalue = False
#     h5s = None
#     try:
#         # imporant information for saving
#         fov_id = image_data['fov']
#         plane_order = image_data['write_plane_order']
#
#         # load the image and process it
#         image_pixeldata = process_tif(image_data)
#
#         # acquire the write lock for this FOV; if there is a block longer than 3 seconds,
#         # acquisition should timeout since backlog subtraction may be running
#         t_s = time.time()
#         lock_acquired = hdf5_locks[fov_id].acquire(block = True, timeout = 3.0)
#         t_e = time.time() - t_s
#         if t_e > 1 and lock_acquired:
#             information("data_writer: lock acquisition delay %0.2f s" % t_e)
#         if not lock_acquired:
#             # no need to release the lock since we don't have it.
#             information("data_writer: unable to obtain lock for FOV %d." % fov_id)
#             return returnvalue
#
#         # if doing subtraction, load cell peaks from spec file for fov_id
#         if subtract:
#             try:
#                 cell_peaks = mm3.load_cell_peaks(fov_id) # load list of cell peaks from spec file
#                 empty_mean = mm3.load_empty_tif(fov_id) # load empty mean image
#                 empty_mean = mm3.trim_zeros_2d(empty_mean) # trim any zero data from the image
#                 h5s = h5py.File(experiment_directory + analysis_directory +
#                                 'subtracted/subtracted_%03d.hdf5' % fov_id, 'r+', libver='earliest')
#                 #h5s.swmr_mode = False
#             except:
#                 subtract = False
#
#         with h5py.File(experiment_directory + analysis_directory + 'originals/original_%03d.hdf5' % fov_id, 'a', libver='earliest') as h5f:
#             if not 'metadata' in h5f.keys(): # make datasets if this is first time
#                 # create and write first metadata
#                 h5mdds = h5f.create_dataset(u'metadata',
#                                     data = np.array([[image_data['metadata']['x'],
#                                     image_data['metadata']['y'], image_data['metadata']['jdn']],]),
#                                     maxshape = (None, 3))
#
#                 # create and write first original
#                 if save_originals:
#                     h5ds = h5f.create_dataset(u'originals',
#                                               data=np.expand_dims(image_pixeldata, 0),
#                                               chunks=(1, image_pixeldata.shape[0], 30, 1),
#                                               maxshape=(None, image_pixeldata.shape[0], image_pixeldata.shape[1], None),
#                                               compression="gzip", shuffle=True)
#                     h5ds.attrs['plane_names'] = [p.encode('utf8') for p in plane_order]
#
#                 # find the channel locations for this fov. slice out channels and save
#                 for channel_loc in channel_masks[fov_id]:
#                     # get id and slice
#                     channel_id, channel_slice = mm3.cut_slice(image_pixeldata, channel_loc)
#                     if compress_hdf5:
#                         h5nds = h5f.create_dataset(u'channel_%04d' % channel_id,
#                                                    data=np.expand_dims(channel_slice, 0),
#                                                    chunks=(1, channel_slice.shape[0],
#                                                            channel_slice.shape[1], 1),
#                                                    maxshape=(None, channel_slice.shape[0],
#                                                              channel_slice.shape[1], None),
#                                                    compression="gzip", shuffle=True)
#                     else:
#                         h5nds = h5f.create_dataset(u'channel_%04d' % channel_id,
#                                                    data=np.expand_dims(channel_slice, 0),
#                                                    chunks=(1, channel_slice.shape[0],
#                                                            channel_slice.shape[1], 1),
#                                                    maxshape=(None, channel_slice.shape[0],
#                                                              channel_slice.shape[1], None))
#                     h5nds.attrs['channel_id'] = channel_loc[0]
#                     h5nds.attrs['channel_loc'] = channel_loc[1]
#                     h5nds.attrs['plane_names'] = [p.encode('utf8') for p in plane_order]
#
#                 returnvalue = True
#             else: # append to current datasets
#                 #h5f.swmr_mode = False
#                 # write additional metadata
#                 h5mdds = h5f[u'metadata']
#                 h5mdds.resize(h5mdds.shape[0] + 1, axis = 0)
#                 h5mdds.flush()
#                 h5mdds[-1] = np.array([image_data['metadata']['x'],
#                                     image_data['metadata']['y'], image_data['metadata']['jdn']])
#                 h5mdds.flush()
#
#                 # write additional originals
#                 if save_originals:
#                     h5ds = h5f[u'originals']
#                     # adjust plane names if need be
#                     if len(h5ds.attrs['plane_names']) < len(plane_order):
#                         h5ds.resize(len(plane_order, axis = 3))
#                         h5ds.flush()
#                         h5ds.attrs['plane_names'] = [p.encode('utf8') for p in plane_order]
#                     h5ds.resize(h5ds.shape[0] + 1, axis = 0)
#                     h5ds.flush()
#                     h5ds[-1] = image_pixeldata
#                     h5ds.flush()
#
#                 # write channels
#                 for channel_loc in channel_masks[fov_id]:
#                     channel_id, channel_slice = mm3.cut_slice(image_pixeldata, channel_loc)
#
#                     h5nds = h5f[u'channel_%04d' % channel_id]
#                     # adjust plane names if need be
#                     if len(h5nds.attrs['plane_names']) < len(plane_order):
#                         h5nds.resize(len(plane_order), axis = 3)
#                         h5nds.flush()
#                         h5nds.attrs['plane_names'] = [p.encode('utf8') for p in plane_order]
#                     h5nds.resize(h5nds.shape[0] + 1, axis = 0)
#                     h5nds.flush()
#                     h5nds[-1] = channel_slice
#                     h5nds.flush()
#
#                     # add to subtraction if it has been initiated and flag is on
#                     if subtract and channel_id in cell_peaks:
#                             # do subtraction
#                             subtracted_data = subtract_phase([channel_slice, empty_mean])
#                             subtracted_image = subtracted_data[0] # subtracted image
#                             offset = subtracted_data[1]
#
#                             # get the image data set for this channel and append sub image
#                             h5si = h5s[u'subtracted_%04d' % channel_id]
#                             h5si.resize(h5si.shape[0] + 1, axis = 0) # add a space fow new s image
#                             h5si.flush()
#                             h5si[-1] = subtracted_image
#                             h5si.flush()
#
#                             # adjust plane names if need be
#                             if len(h5si.attrs['plane_names']) < (len(plane_order)+1):
#                                 h5si.resize(len(plane_order, axis = 3))
#                                 h5si.flush()
#                                 # rearrange plane names
#                                 plane_names = [p.encode('utf8') for p in plane_order]
#                                 plane_names.append(plane_names.pop(0)) # move phas to back
#                                 plane_names.insert(0, 'subtracted_phase') # put sub first
#                                 h5si.attrs['plane_names'] = plane_names # set new attribute
#
#                             # add offset information
#                             h5os = h5s[u'offsets_%04d' % channel_id]
#                             h5os.resize(h5os.shape[0] + 1, axis = 0) # add a space fow new s image
#                             h5os.flush()
#                             h5os[-1] = offset
#                             h5os.flush()
#
#                 # append metdata for subtraction
#                 if subtract:
#                     sub_mds = h5s[u'metadata']
#                     sub_mds.resize(sub_mds.shape[0] + 1, axis = 0)
#                     sub_mds.flush()
#                     sub_mds[-1] = np.array([image_data['metadata']['x'],
#                                     image_data['metadata']['y'], image_data['metadata']['jdn']])
#                     sub_mds.flush()
#
#                     h5s.close() # close file (can't use `with` context manager here)
#
#                 returnvalue = True
#     except:
#         try:
#             if h5s is not None:
#                 h5s.close()
#         except:
#             pass
#         warning(sys.exc_info()[0])
#         warning(sys.exc_info()[1])
#         warning(traceback.print_tb(sys.exc_info()[2]))
#
#     # release the write lock
#     hdf5_locks[fov_id].release()
#
#     return returnvalue

# when using this script as a function and not as a library the following will execute
if __name__ == "__main__":

    # get switches and parameters
    try:
        opts, args = getopt.getopt(sys.argv[1:],"f:")
    except getopt.GetoptError:
        print('No arguments detected (-f).')

    # set parameters
    for opt, arg in opts:
        if opt == '-f':
            param_file_path = arg # parameter file path


    # Load the project parameters file
    # if the paramfile string has no length ie it has not been specified, ERROR
    if len(param_file_path) == 0:
        raise ValueError("a parameter file must be specified (-f <filename>).")
    information ('Loading experiment parameters...')
    with open(param_file_path, 'r') as param_file:
        p = yaml.safe_load(param_file) # load parameters into dictionary

    mm3.init_mm3_helpers(param_file_path) # initialized the helper library


    # hardcoded variables and flags
    # p['compress_hdf5'] = True # flag for if images should be gzip compressed in hdf5
    # p['tif_creator'] = 'elements' # what is the source of the TIFFs?

    ### multiprocessing variables and set up.
    # set up a dictionary of locks to prevent HDF5 disk collisions
    # global hdf5_locks
    # hdf5_locks = {x: Lock() for x in range(p['num_fovs'])} # num_fovs is global parameter

    # set up how to manage cores for multiprocessing
    cpu_count = multiprocessing.cpu_count()
    if cpu_count == 32:
        num_analyzers = 20
    elif cpu_count == 8:
        num_analyzers = 14
    else:
        num_analyzers = cpu_count*2 - 2

    # create the analysis folder if it doesn't exist
    if not os.path.exists(p['experiment_directory'] + p['analysis_directory']):
        os.makedirs(p['experiment_directory'] + p['analysis_directory'])
    # create folder for sliced data.
    if not os.path.exists(p['experiment_directory'] + p['analysis_directory'] + 'originals/'):
        os.makedirs(p['experiment_directory'] + p['analysis_directory'] + 'originals/')

    # assign shorthand directory names
    TIFF_dir = p['experiment_directory'] + p['image_directory'] # source of images
    ana_dir = p['experiment_directory'] + p['analysis_directory']

    # declare information variables
    analyzed_imgs = {} # for storing get_params pool results.
    written_imgs = {} # for storing write objects set to write. Are removed once written

    # process TIFFs for metadata
    try:
        # get all the TIFFs in the folder
        found_files = sorted(glob.glob(TIFF_dir + '*.tif'))

        # get information for all these starting tiffs
        if len(found_files) > 0:
            information("Priming with %d pre-existing files." % len(found_files))
        else:
            warning('No TIFF files found')

        # initialize pool for analyzing image metadata
        pool = Pool(num_analyzers)

        # loop over images and get information
        for fn in found_files:
            # get_params gets the image metadata and puts it in analyzed_imgs dictionary
            # for each file name. True means look for channels

            # This is the non-parallelized version (useful for debug)
            #analyzed_imgs[fn] = mm3.get_tif_params(fn, True)

            # Parallelized
            analyzed_imgs[fn] = pool.apply_async(mm3.get_tif_params, args=(fn, True))

        information('Waiting for image analysis pool to be finished.')

        pool.close() # tells the process nothing more will be added.
        pool.join() # blocks script until everything has been processed and workers exit

        information('Image analyses pool finished, getting results.')

        # get results from the pool and put them in a dictionary
        for fn, result in analyzed_imgs.iteritems():
            if result.successful():
                analyzed_imgs[fn] = result.get() # put the metadata in the dict if it's good
            else:
                analyzed_imgs[fn] = False # put a false there if it's bad

        information('Got results from analyzed images.')

        # save metadata to a .pkl and a human readable json file
        with open(ana_dir + '/TIFF_metadata.pkl', 'w') as tiff_metadata:
            pickle.dump(analyzed_imgs, tiff_metadata)
        with open(ana_dir + '/TIFF_metadata.pkl', 'w') as tiff_metadata:
            json.dump(analyzed_imgs, tiff_metadata)

        information('Saved metadata from analyzed images')

    except:
        warning("Image parameter analysis try block failed.")
        print(sys.exc_info()[0])
        print(sys.exc_info()[1])
        print(traceback.print_tb(sys.exc_info()[2]))

    # Make consensus channel masks and get other shared metadata
    try:
        # Uses channel information from the already processed image data
        channel_masks = make_masks(analyzed_imgs)

        #save the channel mask dictionary
        with open(ana_dir + 'channel_masks.pkl', 'w') as cmask_file:
            pickle.dump(channel_masks, cmask_file)

        information("Channel masks saved.")

    except:
        warning("Mask creation try block failed.")
        print(sys.exc_info()[0])
        print(sys.exc_info()[1])
        print(traceback.print_tb(sys.exc_info()[2]))

    # Slice and write TIFF files into channel
    try:
        pass
        # # check write results and set the success flag as appropriate in the metadata.
        # # this makes sure things don't get written twice. writing happens next.
        # # print('dummy')
        # for wfn in w_result_dict.keys():
        #     if w_result_dict[wfn].ready():
        #         w_result = w_result_dict[wfn].get()
        #         del w_result_dict[wfn]
        #         if w_result:
        #             image_metadata[wfn]['write_success'] = True
        #             successful_write_count += 1 # just add to count
        #             loop_write_count += 1
        #             #information("wrote to originals hdf5 %s" % wfn.split("/")[-1])
        #             del image_metadata[wfn]
        #         else:
        #             image_metadata[wfn]['sent_to_write'] = False
        #             information("Failed to write %s" % wfn.split("/")[-1])
        #
        #
        # # send writes to the pool based on lowest jdn not written after queued writes clear
        # # for big existing file lists, this is slow like molasses
        # if mask_created: # only slice and write if channel mask is made
        #     # get a list of lists of all the fovs not yet written, jdn is the julian date time (exact time)
        #     def get_not_written_for_fov(fov):
        #         return [[k, v['metadata']['jdn']] for k, v in image_metadata.items() if v['fov'] == fov and not v['write_success'] and v['write_plane_order']]
        #     fov_ready_for_write = map(get_not_written_for_fov, range(num_fovs))
        #
        #     # writing is done here
        #     for fov_fns_jdns in fov_ready_for_write:
        #         if len(fov_fns_jdns) > 0:
        #             # check if any of these filenames have been sent to write and haven't yet been processed for exit status
        #             waiting_on_write = np.sum([fn in w_result_dict.keys() for fn, fjdn in fov_fns_jdns]) > 0
        #             # if no files are waiting on write, send the next image for write
        #             if not waiting_on_write:
        #                 # sort the filenames-jdns by jdn
        #                 fov_fns_jdns = sorted(fov_fns_jdns, key=lambda imd: imd[1])
        #                 # data_writer is the major hdf5 writing function.
        #                 # not switched for saving originals and doing subtraction
        #                 w_result_dict[fov_fns_jdns[0][0]] = wpool.apply_async(data_writer,
        #                                             [image_metadata[fov_fns_jdns[0][0]],
        #                                             channel_masks, subtract_on_datawrite[image_metadata[fov_fns_jdns[0][0]]['fov']],
        #                                             save_originals])
        #                 image_metadata[fov_fns_jdns[0][0]]['sent_to_write'] = True
        #
        #
        # # write the list of known files to disk
        # # marshal is 10x faster and 50% more space efficient than cPickle here, but dtypes
        # # are all python primitives
        # if len(found_files) >= known_files_last_save_size * 1.1 or time.time() - known_files_last_save_time > 900:
        #     with open(experiment_directory + analysis_directory + 'found_files.mrshl', 'w') as outfile:
        #         marshal.dump(found_files, outfile)
        #     known_files_last_save_time = time.time()
        #     known_files_last_save_size = len(found_files) * 1.1
        #     information("Saved intermediate found_files (%d)." % len(found_files))
        #
        # # update user current progress
        # if count % 1000 == 0:
        #     information("1000 loop time %0.2fs, running metadata total: %d" % (time.time() - t_s_loop, len(image_metadata)))
        #     information("Analysis ok for %d images (%d total)." %
        #                 (loop_analysis_count, successful_analysis_count))
        #     loop_analysis_count = 0 # reset counter
        #     information("Wrote %d images to hdf5 (%d total)." %
        #                 (loop_write_count, successful_write_count))
        #     loop_write_count = 0 # reset loop counter
        #     count = 0
        #
        #     # if there's nothing going on, don't hog the CPU
        #     if (len(cp_result_dict) == 0 and len(w_result_dict) == 0 and
        #         len(image_metadata) == 0):
        #         information("Queues are empty; waiting 5 seconds to loop.")
        #         information("%d images analyzed, %d written to hdf5." %
        #                     (successful_analysis_count, successful_write_count))
        #         time.sleep(5)

        # except KeyboardInterrupt:
        #         warning("Caught KeyboardInterrupt, terminating workers...")
    except:
        warning("Channel slicing try block failed.")
        print(sys.exc_info()[0])
        print(sys.exc_info()[1])
        print(traceback.print_tb(sys.exc_info()[2]))
