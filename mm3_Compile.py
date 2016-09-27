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
# import struct
import re
import glob
import gevent
import marshal
# import json # used to write data out in human readable format
from pprint import pprint # for human readable file output
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

    ### process TIFFs for metadata #################################################################
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
            # analyzed_imgs[fn] = mm3.get_tif_params(fn, True)

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

        # save metadata to a .pkl and a human readable txt file
        with open(ana_dir + '/TIFF_metadata.pkl', 'w') as tiff_metadata:
            pickle.dump(analyzed_imgs, tiff_metadata)
        with open(ana_dir + '/TIFF_metadata.txt', 'w') as tiff_metadata:
            pprint(analyzed_imgs, stream=tiff_metadata)

        information('Saved metadata from analyzed images')

    except:
        warning("Image parameter analysis try block failed.")
        print(sys.exc_info()[0])
        print(sys.exc_info()[1])
        print(traceback.print_tb(sys.exc_info()[2]))

    ### Make consensus channel masks and get other shared metadata #################################
    try:
        # Uses channel information from the already processed image data
        channel_masks = mm3.make_masks(analyzed_imgs)

        #save the channel mask dictionary to a pickle and a text file
        with open(ana_dir + '/channel_masks.pkl', 'w') as cmask_file:
            pickle.dump(channel_masks, cmask_file)
        with open(ana_dir + '/channel_masks.txt', 'w') as cmask_file:
            pprint(channel_masks, stream=cmask_file)

        information("Channel masks saved.")

    except:
        warning("Mask creation try block failed.")
        print(sys.exc_info()[0])
        print(sys.exc_info()[1])
        print(traceback.print_tb(sys.exc_info()[2]))

    ### Slice and write TIFF files into channels ###################################################
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
