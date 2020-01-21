#!/usr/bin/env python3
from __future__ import print_function, division
import six

# import modules
import sys
import os
import time
import inspect
import argparse
import yaml
import glob
import re
from skimage import io, measure, morphology
from skimage.external import tifffile as tiff
from scipy import stats
from pprint import pprint # for human readable file output
try:
    import cPickle as pickle
except:
    import pickle
import multiprocessing
from multiprocessing import Pool
import numpy as np
import warnings
import h5py

from matplotlib import pyplot as plt

from tensorflow.python.keras import models

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

# this is the mm3 module with all the useful functions and classes
import mm3_helpers as mm3

# when using this script as a function and not as a library the following will execute
if __name__ == "__main__":
    '''mm3_Compile.py locates and slices out mother machine channels into image stacks.
    '''

    # set switches and parameters
    parser = argparse.ArgumentParser(prog='python mm3_Compile.py',
                                     description='Identifies and slices out channels into individual TIFF stacks through time.')
    parser.add_argument('-f', '--paramfile',  type=str,
                        required=True, help='Yaml file containing parameters.')
    parser.add_argument('-p', '--path',  type=str,
                        required=False, help='Path to data directory. Overrides what is in param file')
    parser.add_argument('-o', '--fov',  type=str,
                        required=False, help='List of fields of view to analyze. Input "1", "1,2,3", or "1-10", etc.')
    parser.add_argument('-j', '--nproc',  type=int,
                        required=False, help='Number of processors to use.')
    parser.add_argument('-m', '--modelfile', type=str,
                        required=False, help='Path to trained U-net model.')
    namespace = parser.parse_args()

    # Load the project parameters file
    mm3.information('Loading experiment parameters.')
    if namespace.paramfile:
        param_file_path = namespace.paramfile
    else:
        mm3.warning('No param file specified. Using 100X template.')
        param_file_path = 'yaml_templates/params_SJ110_100X.yaml'
    p = mm3.init_mm3_helpers(param_file_path) # initialized the helper library

    if namespace.path:
        p = mm3.init_mm3_helpers(param_file_path, datapath=namespace.path) # initialized the helper library
    else:
        p = mm3.init_mm3_helpers(param_file_path, datapath=None)

    if namespace.fov:
        if '-' in namespace.fov:
            user_spec_fovs = range(int(namespace.fov.split("-")[0]),
                                   int(namespace.fov.split("-")[1])+1)
        else:
            user_spec_fovs = [int(val) for val in namespace.fov.split(",")]
    else:
        user_spec_fovs = []

    # number of threads for multiprocessing
    if namespace.nproc:
        p['num_analyzers'] = namespace.nproc
    mm3.information('Using {} threads for multiprocessing.'.format(p['num_analyzers']))

    # only analyze images up until this t point. Put in None otherwise
    if 't_end' in p['compile']:
        t_end = p['compile']['t_end']
        if t_end == 'None':
            t_end = None
    else:
        t_end = None
    # only analyze images at and after this t point. Put in None otherwise
    if 't_start' in p['compile']:
        t_start = p['compile']['t_start']
        if t_start == 'None':
            t_start = None
    else:
        t_start = None

    # create the subfolders if they don't
    if not os.path.exists(p['ana_dir']):
        os.makedirs(p['ana_dir'])
    if p['output'] == 'TIFF':
        if not os.path.exists(p['chnl_dir']):
            os.makedirs(p['chnl_dir'])
    elif p['output'] == 'HDF5':
        if not os.path.exists(p['hdf5_dir']):
            os.makedirs(p['hdf5_dir'])

    # declare information variables
    analyzed_imgs = {} # for storing get_params pool results.

    ### process TIFFs for metadata #################################################################
    if not p['compile']['do_metadata']:
        mm3.information("Loading image parameters dictionary.")

        with open(os.path.join(p['ana_dir'], 'TIFF_metadata.pkl'), 'rb') as tiff_metadata:
            analyzed_imgs = pickle.load(tiff_metadata)

    else:
        mm3.information("Finding image parameters.")

        # get all the TIFFs in the folder
        found_files = glob.glob(os.path.join(p['TIFF_dir'],'*.tif')) # get all tiffs
        found_files = [filepath.split('/')[-1] for filepath in found_files] # remove pre-path
        found_files = sorted(found_files) # should sort by timepoint

        # keep images starting at this timepoint
        if t_start is not None:
            mm3.information('Removing images before time {}'.format(t_start))
            # go through list and find first place where timepoint is equivalent to t_start
            for n, ifile in enumerate(found_files):
                string = re.compile('t{:0=3}xy|t{:0=4}xy'.format(t_start,t_start)) # account for 3 and 4 digit
                # if re.search == True then a match was found
                if re.search(string, ifile):
                    # cut off every file name prior to this one and quit the loop
                    found_files = found_files[n:]
                    break

        # remove images after this timepoint
        if t_end is not None:
            mm3.information('Removing images after time {}'.format(t_end))
            # go through list and find first place where timepoint is equivalent to t_end
            for n, ifile in enumerate(found_files):
                string = re.compile('t%03dxy|t%04dxy' % (t_end, t_end)) # account for 3 and 4 digit
                if re.search(string, ifile):
                    found_files = found_files[:n]
                    break


        # if user has specified only certain FOVs, filter for those
        if (len(user_spec_fovs) > 0):
            mm3.information('Filtering TIFFs by FOV.')
            fitered_files = []
            for fov_id in user_spec_fovs:
                fov_string = 'xy%02d' % fov_id # xy01
                fitered_files += [ifile for ifile in found_files if fov_string in ifile]

            found_files = fitered_files[:]

        # get information for all these starting tiffs
        if len(found_files) > 0:
            mm3.information("Found %d image files." % len(found_files))
        else:
            mm3.warning('No TIFF files found')

        if p['compile']['find_channels_method'] == 'peaks':

            # initialize pool for analyzing image metadata
            pool = Pool(p['num_analyzers'])

            # loop over images and get information
            for fn in found_files:
                # get_params gets the image metadata and puts it in analyzed_imgs dictionary
                # for each file name. True means look for channels

                # This is the non-parallelized version (useful for debug)
                # analyzed_imgs[fn] = mm3.get_tif_params(fn, True)

                # Parallelized
                analyzed_imgs[fn] = pool.apply_async(mm3.get_tif_params, args=(fn, True))

            mm3.information('Waiting for image analysis pool to be finished.')

            pool.close() # tells the process nothing more will be added.
            pool.join() # blocks script until everything has been processed and workers exit

            mm3.information('Image analysis pool finished, getting results.')

            # get results from the pool and put them in a dictionary
            for fn in analyzed_imgs.keys():
                result = analyzed_imgs[fn]
                if result.successful():
                    analyzed_imgs[fn] = result.get() # put the metadata in the dict if it's good
                else:
                    analyzed_imgs[fn] = False # put a false there if it's bad

        elif p['compile']['find_channels_method'] == 'Unet':
            # Use Unet trained on trap and central channel locations to locate, crop, and align traps
            mm3.information("Identifying channel locations and aligning images using U-net.")

            # load model to pass to algorithm
            mm3.information("Loading model...")

            if namespace.modelfile:
                model_file_path = namespace.modelfile
            else:
                model_file_path = p['compile']['model_file_traps']
            # *** Need parameter for weights
            model = models.load_model(model_file_path,
                                      custom_objects={'tversky_loss': mm3.tversky_loss,
                                                      'cce_tversky_loss': mm3.cce_tversky_loss})
            mm3.information("Model loaded.")

            # initialize pool for getting image metadata
            pool = Pool(p['num_analyzers'])

            # loop over images and get information
            for fn in found_files:
                # get_params gets the image metadata and puts it in analyzed_imgs dictionary
                # for each file name. Won't look for channels, just gets the metadata for later use by Unet

                # This is the non-parallelized version (useful for debug)
                # analyzed_imgs[fn] = mm3.get_initial_tif_params(fn)

                # Parallelized
                analyzed_imgs[fn] = pool.apply_async(mm3.get_initial_tif_params, args=(fn,))

            mm3.information('Waiting for image metadata pool to be finished.')
            pool.close() # tells the process nothing more will be added.
            pool.join() # blocks script until everything has been processed and workers exit

            mm3.information('Image metadata pool finished, getting results.')

            # get results from the pool and put them in a dictionary
            for fn in analyzed_imgs.keys():
               result = analyzed_imgs[fn]
               if result.successful():
                   analyzed_imgs[fn] = result.get() # put the metadata in the dict if it's good
               else:
                   analyzed_imgs[fn] = False # put a false there if it's bad

            # print(analyzed_imgs)

            # set up some variables for Unet and image aligment/cropping
            file_names = [key for key in analyzed_imgs.keys()]
            file_names.sort() # sort the file names by time
            file_names = np.asarray(file_names)
            fov_ids = [analyzed_imgs[key]['fov'] for key in analyzed_imgs.keys()]

            unique_fov_ids = np.unique(fov_ids)

            if p['compile']['do_channel_masks']:
                channel_masks = {}

            for fov_id in unique_fov_ids:

                mm3.information('Performing trap segmentation for fov_id: {}'.format(fov_id))

                #print(analyzed_imgs)
                fov_indices = np.where(fov_ids == fov_id)[0]
                # print(fov_indices)
                fov_file_names = [file_names[idx] for idx in fov_indices]
                trap_align_metadata = {'first_frame_name': fov_file_names[0],
                                    'frame_count': len(fov_file_names),
                                    'plane_number': len(analyzed_imgs[fn]['planes']),
                                    'trap_height': p['compile']['trap_crop_height'],
                                    'trap_width': p['compile']['trap_crop_width'],
                                    'phase_plane': p['phase_plane'],
                                    'phase_plane_index': p['moviemaker']['phase_plane_index'],
                                    'shift_distance': 256,
                                    'full_frame_size': 2048}

                dilator = np.ones((1,300))

                # create weights for taking weighted mean of several runs of Unet over various crops of the first image in the series. This helps remove "blind spots" from the neural network at the edges of each crop of the original image.
                stack_weights = mm3.get_weights_array(np.zeros((trap_align_metadata['full_frame_size'],trap_align_metadata['full_frame_size'])), trap_align_metadata['shift_distance'], subImageNumber=16, padSubImageNumber=25)[0,...]
                # print(stack_weights.shape) #uncomment for debugging

                # get prediction of where traps are located in first image
                imgPath = os.path.join(p['experiment_directory'], p['image_directory'],
                                       trap_align_metadata['first_frame_name'])
                img = io.imread(imgPath)
                # detect if there are multiple imaging channels, and rearrange image if necessary, keeping only the phase image
                img = mm3.permute_image(img, trap_align_metadata)
                if p['debug']:
                    io.imshow(img/np.max(img));
                    plt.title("Initial phase image");
                    plt.show();

                # produces predition stack with 3 "pages", index 0 is for traps, index 1 is for central tough, index 2 is for background
                mm3.information("Predicting trap locations for first frame.")
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    first_frame_trap_prediction = mm3.get_frame_predictions(img, 
                                                                            model, 
                                                                            stack_weights, 
                                                                            trap_align_metadata['shift_distance'], 
                                                                            subImageNumber=16, 
                                                                            padSubImageNumber=25, 
                                                                            debug=p['debug'])

                if p['debug']:
                    fig,ax = plt.subplots(nrows=1, ncols=4, figsize=(12,12))
                    ax[0].imshow(img);
                    for i in range(first_frame_trap_prediction.shape[-1]):
                        ax[i+1].imshow(first_frame_trap_prediction[:,:,i]);
                    plt.show();

                # flatten prediction stack such that each pixel of the resulting 2D image is the index of the prediction image above with the highest predicted probability
                class_predictions = np.argmax(first_frame_trap_prediction, axis=2)

                traps = class_predictions == 0 # returns boolean array where our intial guesses at trap locations are True

                if p['debug']:
                    io.imshow(traps);
                    plt.title('Initial trap masks')
                    plt.show();

                trap_labels = measure.label(traps)
                trap_props = measure.regionprops(trap_labels)

                trap_area_threshold = p['compile']['trap_area_threshold']
                trap_bboxes = mm3.get_frame_trap_bounding_boxes(trap_labels,
                                                                   trap_props,
                                                                   trapAreaThreshold=trap_area_threshold,
                                                                   trapWidth=trap_align_metadata['trap_width'],
                                                                   trapHeight=trap_align_metadata['trap_height'])

                # create boolean array to contain filtered, correctly-shaped trap bounding boxes
                first_frame_trap_mask = np.zeros(traps.shape)
                for i,bbox in enumerate(trap_bboxes):
                    first_frame_trap_mask[bbox[0]:bbox[2],bbox[1]:bbox[3]] = True

                good_trap_labels = measure.label(first_frame_trap_mask)
                good_trap_props = measure.regionprops(good_trap_labels)

                # widen the traps to merge them into "trap regions" above and below the central trough
                dilated_traps = morphology.dilation(first_frame_trap_mask, dilator)

                if p['debug']:
                    io.imshow(dilated_traps);
                    plt.title('Dilated trap masks');
                    plt.show();

                dilated_trap_labels = measure.label(dilated_traps)
                dilated_trap_props = measure.regionprops(dilated_trap_labels)
                # filter merged trap regions by area
                areas = [reg.area for reg in dilated_trap_props]
                labels = [reg.label for reg in dilated_trap_props]

                for idx,area in enumerate(areas):
                    if area < p['compile']['merged_trap_region_area_threshold']:

                        label = labels[idx]
                        dilated_traps[dilated_trap_labels == label] = 0

                dilated_trap_labels = measure.label(dilated_traps)
                dilated_trap_props = measure.regionprops(dilated_trap_labels)

                if p['debug']:
                    io.imshow(dilated_traps);
                    plt.title("Area-filtered dilated traps");
                    plt.show();

                # get centroids for each "trap region" identified in first frame
                centroids = np.round(np.asarray([reg.centroid for reg in dilated_trap_props]))
                if p['debug']:
                    print(centroids)

                # test whether we could crop a (512,512) square from each "trap region", with the centroids as the centers of the crops, withoug going out-of-bounds
                top_test = centroids[:,0]-256 > 0
                bottom_test = centroids[:,0]+256 < dilated_trap_labels.shape[0]
                test_array = np.stack((top_test,bottom_test))

                # get the index of the first identified "trap region" that we can get our (512,512) crop from, use that centroid for nucleus of cropping a stack of phase images with shape (frame_number,512,512,1) from all images in series
                if p['debug']:
                    print(test_array)
                    print(np.all(test_array,axis=0))

                good_trap_region_index = np.where(np.all(test_array, axis=0))[0][0]
                centroid = centroids[good_trap_region_index,:].astype('uint16')
                if p['debug']:
                    print(centroid)

                # get the (frame_number,512,512,1)-sized stack for image aligment
                align_region_stack = np.zeros((trap_align_metadata['frame_count'],512,512,1), dtype='uint16')

                for frame,fn in enumerate(fov_file_names):
                    imgPath = os.path.join(p['experiment_directory'],p['image_directory'],fn)
                    frame_img = io.imread(imgPath)
                    # detect if there are multiple imaging channels, and rearrange image if necessary, keeping only the phase image
                    frame_img = mm3.permute_image(frame_img, trap_align_metadata)
                    align_region_stack[frame,:,:,0] = frame_img[centroid[0]-256:centroid[0]+256,
                                                             centroid[1]-256:centroid[1]+256]

                # if p['debug']:
                #     colNum = 10
                #     fig,ax = plt.subplots(ncols=colNum, figsize=(20,20))

                #     for pltIdx in range(colNum):
                #         ax[pltIdx].imshow(align_region_stack[pltIdx*10,:,:,0])

                #     plt.title('Alignment stack images');
                #     plt.show();

                # run model on all frames
                batch_size=p['compile']['channel_prediction_batch_size']
                mm3.information("Predicting trap regions for (512,512) slice through all frames.")

                data_gen_args = {'batch_size':batch_size,
                         'n_channels':1,
                         'normalize_to_one':True,
                         'shuffle':False}
                predict_gen_args = {'verbose':1,
                        'use_multiprocessing':True,
                        'workers':p['num_analyzers']}

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    img_generator = mm3.TrapSegmentationDataGenerator(align_region_stack, **data_gen_args)

                    align_region_predictions = model.predict_generator(img_generator, **predict_gen_args)
                #align_region_stack = mm3.apply_median_filter_and_normalize(align_region_stack)
                #align_region_predictions = model.predict(align_region_stack, batch_size=batch_size)
                # reduce dimensionality such that the class predictions are now (frame_number,512,512), and each voxel is labelled as the predicted region, i.e., 0=trap, 1=central trough, 2=background.
                align_region_class_predictions = np.argmax(align_region_predictions, axis=3)

                # if p['debug']:
                #     colNum = 10
                #     fig,ax = plt.subplots(ncols=colNum, figsize=(20,20))

                #     for pltIdx in range(colNum):
                #         ax[pltIdx].imshow(align_region_class_predictions[pltIdx*10,:,:])

                #     plt.title('Alignment stack predictions');
                #     plt.show();

                # get boolean array where trap predictions are True
                align_traps = align_region_class_predictions == 0

                # if p['debug']:
                #     colNum = 10
                #     fig,ax = plt.subplots(ncols=colNum, figsize=(20,20))

                #     for pltIdx in range(colNum):
                #         ax[pltIdx].imshow(align_traps[pltIdx*10,:,:])

                #     plt.title('Alignment trap masks');
                #     plt.show();

                # allocate array to store filtered traps over time
                align_trap_mask_stack = np.zeros(align_traps.shape)
                for frame in range(trap_align_metadata['frame_count']):

                    frame_trap_labels = measure.label(align_traps[frame,:,:])
                    frame_trap_props = measure.regionprops(frame_trap_labels)

                    trap_bboxes = mm3.get_frame_trap_bounding_boxes(frame_trap_labels,
                                                                    frame_trap_props,
                                                                    trapAreaThreshold=trap_area_threshold,
                                                                    trapWidth=trap_align_metadata['trap_width'],
                                                                    trapHeight=trap_align_metadata['trap_height'])

                    for i,bbox in enumerate(trap_bboxes):
                        align_trap_mask_stack[frame,bbox[0]:bbox[2],bbox[1]:bbox[3]] = True

                # if p['debug']:
                #     colNum = 10
                #     fig,ax = plt.subplots(ncols=colNum, figsize=(20,20))

                #     for pltIdx in range(colNum):
                #         ax[pltIdx].imshow(align_trap_mask_stack[pltIdx*10,:,:])

                #     plt.title('Filtered alignment trap masks');
                #     plt.show();

                labelled_align_trap_mask_stack = measure.label(align_trap_mask_stack)

                trapTriggered = False
                for frame in range(trap_align_metadata['frame_count']):
                    anyTraps = np.any(labelled_align_trap_mask_stack[frame,:,:] > 0)
                    # if anyTraps is False, that means no traps were detected for this frame. This usuall occurs due to a bug in our imaging system,
                    #    which can cause it to miss the occasional frame. Should be fine to snag labels from prior frame.
                    if not anyTraps:
                        trapTriggered = True
                        mm3.information("Frame at index {} has no detected traps. Borrowing labels from an adjacent frame.".format(frame))
                        if frame > 0:
                            labelled_align_trap_mask_stack[frame,:,:] = labelled_align_trap_mask_stack[frame-1,:,:]
                        else:
                            labelled_align_trap_mask_stack[frame,:,:] = labelled_align_trap_mask_stack[frame+1,:,:]

                if trapTriggered:
                    repaired_align_trap_mask_stack = labelled_align_trap_mask_stack > 0
                    labelled_align_trap_mask_stack = measure.label(repaired_align_trap_mask_stack)

                align_trap_props = measure.regionprops(labelled_align_trap_mask_stack)

                areas = np.array([trap.area for trap in align_trap_props])
                labels = [trap.label for trap in align_trap_props]
                good_align_trap_props = []
                bad_align_trap_props = []
                #mode_area = stats.mode(areas)[0]
                expected_area = trap_align_metadata['trap_width'] * trap_align_metadata['trap_height'] * trap_align_metadata['frame_count']

                if p['debug']:
                    pprint(areas)
                    print(expected_area)

                    if not expected_area in areas:
                        print("No trap has expected total area. Saving labelled masks for debugging as labelled_align_trap_mask_stack.tif")
                        io.imsave("labelled_align_trap_mask_stack.tif", labelled_align_trap_mask_stack.astype('uint8'))
                        io.imsave("masks.tif", align_traps.astype('uint8'))
                        # occasionally our microscope misses an image, resulting in no traps for a single frame. This obviously messes up image alignment here....

                for trap in align_trap_props:
                    if trap.area != expected_area:
                        bad_align_trap_props.append(trap.label)
                    else:
                        good_align_trap_props.append(trap)

                for label in bad_align_trap_props:
                    labelled_align_trap_mask_stack[labelled_align_trap_mask_stack == label] = 0

                align_centroids = []
                for frame in range(trap_align_metadata['frame_count']):
                    align_centroids.append([reg.centroid for reg in measure.regionprops(labelled_align_trap_mask_stack[frame,:,:])])

                align_centroids = np.asarray(align_centroids)
                shifts = np.mean(align_centroids - align_centroids[0,:,:], axis=1)
                integer_shifts = np.round(shifts).astype('int16')

                good_trap_bboxes_dict = {}
                for trap in good_trap_props:
                    good_trap_bboxes_dict[trap.label] = trap.bbox

                # pprint(good_trap_bboxes_dict) # uncomment for debugging
                bbox_shift_dict = mm3.shift_bounding_boxes(good_trap_bboxes_dict, integer_shifts, img.shape[0])
                # pprint(bbox_shift_dict) # uncomment for debugging

                trap_images_fov_dict, trap_closed_end_px_dict = mm3.crop_traps(fov_file_names, good_trap_props, good_trap_labels, bbox_shift_dict, trap_align_metadata)

                for fn in fov_file_names:
                    analyzed_imgs[fn]['channels'] = trap_closed_end_px_dict[fn]

                if p['compile']['do_channel_masks']:
                    fov_channel_masks = mm3.make_channel_masks_CNN(bbox_shift_dict)
                    channel_masks[fov_id] = fov_channel_masks
                    # pprint(channel_masks) # uncomment for debugging

                if p['compile']['do_slicing']:

                    if p['output'] == "TIFF":

                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            mm3.save_tiffs(trap_images_fov_dict, analyzed_imgs, fov_id)

                    elif p['output'] == "HDF5":
                        # Or write it to hdf5
                        mm3.save_hdf5(trap_images_fov_dict, fov_file_names, analyzed_imgs, fov_id, channel_masks)

        # save metadata to a .pkl and a human readable txt file
        mm3.information('Saving metadata from analyzed images...')
        with open(os.path.join(p['ana_dir'], 'TIFF_metadata.pkl'), 'wb') as tiff_metadata:
            pickle.dump(analyzed_imgs, tiff_metadata, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(p['ana_dir'], 'TIFF_metadata.txt'), 'w') as tiff_metadata:
            pprint(analyzed_imgs, stream=tiff_metadata)

        mm3.information('Saved metadata from analyzed images.')

    ### Make table for jd time to FOV and time point
    if not p['compile']['do_time_table']:
        mm3.information('Skipping time table creation.')
    else:
        time_table = mm3.make_time_table(analyzed_imgs)

    ### Make consensus channel masks and get other shared metadata #################################
    if not p['compile']['do_channel_masks'] and p['compile']['do_slicing']:
        channel_masks = mm3.load_channel_masks()

    elif p['compile']['do_channel_masks']:

        if p['compile']['find_channels_method'] == 'peaks':
            # only calculate channels masks from images before t_end in case it is specified
            if t_start:
                analyzed_imgs = {fn : i_metadata for fn, i_metadata in six.iteritems(analyzed_imgs) if i_metadata['t'] >= t_start}
            if t_end:
                analyzed_imgs = {fn : i_metadata for fn, i_metadata in six.iteritems(analyzed_imgs) if i_metadata['t'] <= t_end}

            # Uses channel mm3.information from the already processed image data
            channel_masks = mm3.make_masks(analyzed_imgs)

        elif p['compile']['find_channels_method'] == 'Unet':

            # save the channel mask dictionary to a pickle and a text file
            with open(os.path.join(p['ana_dir'], 'channel_masks.pkl'), 'wb') as cmask_file:
                pickle.dump(channel_masks, cmask_file, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(p['ana_dir'], 'channel_masks.txt'), 'w') as cmask_file:
                pprint(channel_masks, stream=cmask_file)

    ### Slice and write TIFF files into channels ###################################################
    if p['compile']['do_slicing']:

        mm3.information("Saving channel slices.")

        if p['compile']['find_channels_method'] == 'peaks':

            # do it by FOV. Not set up for multiprocessing
            for fov, peaks in six.iteritems(channel_masks):

                # skip fov if not in the group
                if user_spec_fovs and fov not in user_spec_fovs:
                    continue

                mm3.information("Loading images for FOV %03d." % fov)

                # get filenames just for this fov along with the julian date of acquistion
                send_to_write = [[k, v['t']] for k, v in six.iteritems(analyzed_imgs) if v['fov'] == fov]

                # sort the filenames by jdn
                send_to_write = sorted(send_to_write, key=lambda time: time[1])

                if p['output'] == 'TIFF':
                    #This is for loading the whole raw tiff stack and then slicing through it
                    mm3.tiff_stack_slice_and_write(send_to_write, channel_masks, analyzed_imgs)

                elif p['output'] == 'HDF5':
                    # Or write it to hdf5
                    mm3.hdf5_stack_slice_and_write(send_to_write, channel_masks, analyzed_imgs)

            mm3.information("Channel slices saved.")
