#!/usr/bin/python -u
from __future__ import print_function
def warning(*objs):
    print(time.strftime("%H:%M:%S", time.localtime()) + " Error:", *objs, file=sys.stderr)
    sys.stderr.flush()
def information(*objs):
    print(time.strftime("%H:%M:%S", time.localtime()), *objs, file=sys.stdout)
    sys.stdout.flush()
    # add writing to text file
    #print(time.strftime("%H:%M:%S", time.localtime()), *objs, file=sys.stdout)

import sys
sys.path.insert(1, '/usr/local/lib/python2.7/site-packages')
import getopt, yaml, traceback, random, os, math, gc, time, fnmatch, h5py, copy, hashlib, marshal

try:
    import cPickle as pickle
except:
    import pickle

import multiprocessing
from multiprocessing import Pool, Manager

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import matplotlib.lines as mplines
from matplotlib.widgets import RectangleSelector

from scipy import ndimage, misc
import scipy.stats as spstats
import scipy.signal as spsignal
from scipy.fftpack import fft2
from scipy.optimize import curve_fit, leastsq

from skimage import data, io, exposure
from skimage.morphology import *
from skimage.feature import peak_local_max, blob_dog
from skimage.exposure import adjust_gamma, rescale_intensity
import skimage.filters as skif
from skimage.segmentation import mark_boundaries, relabel_sequential, clear_border
from skimage.measure import regionprops, profile_line
import skimage.restoration as skir

try:
    import pydot
except:
    pydot = None

from subtraction_helpers import *
import morphsnakes

reuse_segmentations = False
verbose_mapcheck = False
prevent_reload = False
specify_fovs = False
user_spec_fovs = []
start_with_fov = -1
end_with_fov = -1
analysis_length = 330
show_plots = False
one_peak = -1
debugging = False
use_algorithm = 'segment_cells_ac6'
remove_blurry = False
remove_blurry_percent = 15.0

cpu_count = multiprocessing.cpu_count()
if cpu_count == 32:
    num_procs = 24
elif cpu_count == 8:
    num_procs = 7
else:
    raise ValueError("host CPU count (%d) not in pre-determined utilization numbers (8, 32).")


### start 2d gaussian helper functions
### adapted from http://wiki.scipy.org/Cookbook/FittingData#head-11870c917b101bb0d4b34900a0da1b7deb613bf7
def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*np.exp(-(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = math.sqrt(abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = math.sqrt(abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y

def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) - data)
    p, success = leastsq(errorfunction, params)
    return list(p)
### end 2d gaussian helper functions

def trim_zeros_br(array):
    array = array[~np.all(array[:,:,0] == 0, axis = 1)]
    array = np.swapaxes(array, 0, 1)
    array = array[~np.all(array[:,:,0] == 0, axis = 1)]
    array = np.swapaxes(array, 0, 1)
    return array

def merge_labeled_segments(labeled_array, labels_to_merge, new_label):
    new_array = np.array(labeled_array, copy=True)
    for label in labels_to_merge:
        temp_array = new_array == label
        new_array -= (temp_array * label)
        new_array += (temp_array * new_label)
        #new_array *= 0
    return new_array


# worker function for segmenting an image
def segment_cells_ac6(idata, forced_splits = []):
    # Structure of image_data[n]:
    # 0 / source_data: original image
    # 1 / phase_segments: segmented (labeled) image
    # 2 / regionprops: RegionProps for segments in [1]
    # 3 / lowest_cell_label: index of "lowest" cell segment in [2]
    # 4 / image_time: acquisition time of the source data
    # 5 / blob_segments: is a dictionary of blobbed fluoro segmentation data
    # 6 / feature_data: foci/blob data lists for the image
    # 7 / otsu_threshold: Otsu threshold used for the image
    # 8 / daughter_map: labels of daughter cell(s) in the subsequent image

    try:
        if len(idata['source_data'].shape) > 2:
            image = idata['source_data'][:,:,0][:]
        else:
            image = idata['source_data']
            warning("Running on non-stacked data! Expected a subtracted and normal phase contrast image.")
        # If the subtracted image is blank, set a similarly sized blank segmentation & return
        if len(np.unique(image)) == 1:
            idata['phase_segments'] = np.zeros_like(image)
            idata['regionprops'] = [regionprops(label_image = idata[1]), ]
            idata['lowest_cell_label'] = -1
            warning("Subtracted image is blank!")
            return idata

        phase = idata['source_data'][:,:,-1]

        # switch for leveling the length-wise brightness of the phase image
        if False:
            bg = np.expand_dims(image[:,(0,1,-2,-1)].min(axis = 1), axis = 1)
            bg = np.pad(bg, ((0, 0), (0, phase.shape[0] - bg.shape[0])), mode = 'edge')
            phase = phase.astype('float64') - bg.astype('float64')
            bg = np.expand_dims(phase.max(axis = 1), axis = 1)
            bg = np.pad(bg, ((0, 0), (0, phase.shape[0] - bg.shape[0])), mode = 'edge')
            phase = phase.astype('float64') / bg.astype('float64')

            #bg = np.expand_dims(phase[:,(0,-1)].mean(axis = 1), axis = 1)
            #bg = np.pad(bg, ((0, 0), (0, phase.shape[0] - bg.shape[0])), mode = 'edge')
            #phase = phase.astype('int32') - bg.astype('int32')
            #idata[5] = phase[:]

        # normalize subtracted image to the image minimum
        analysis_length = 300
        s_image = image[:analysis_length].astype('int32')
        #s_image = image.astype('int32')
        s_image -= np.min(image)

        idata['otsu_threshold'] = skif.threshold_otsu(s_image)

        gI = morphsnakes.gborders(s_image, alpha=0.1, sigma=1.)
        macwe = morphsnakes.MorphGAC(gI, smoothing=2, threshold=0.5, balloon=-1)
        #macwe.levelset = binary_dilation(remove_small_objects(s_image > (idata['otsu_threshold'] + (0.2 * float(np.percentile(s_image, 99) - idata['otsu_threshold'])))))
        macwe.levelset = binary_dilation(remove_small_objects(s_image > (idata['otsu_threshold'])))
        macwe.run(30)
        end_set = macwe.levelset[:].astype('bool')

        r_image = exposure.rescale_intensity(s_image.astype('float64'))

        # set to True to enable more aggressive image denoising
        v = 'v2'
        if v == 'v2':
            r_image = skir.denoise_bilateral(r_image)
        elif v == 'v1':
            r_image = skir.denoise_bilateral(r_image, win_size = 7, sigma_range = 0.9, sigma_spatial = 3.00)
        else:
            r_image = skir.denoise_bilateral(r_image, win_size = 7)

        v2d = np.diff(np.diff(r_image, axis = 0), axis = 0)
        v2d = remove_small_objects(v2d > np.percentile(v2d, 90))
        v2d = np.pad(v2d, ((1, 1), (0, 0)), mode = 'constant', constant_values = False)

        rgl = ndimage.filters.gaussian_laplace(r_image, sigma = 1.5, mode = 'nearest')
        rglp = (rgl < 0).astype('int8')
        rglp[~end_set] = 0
        seed = np.copy(rglp)
        seed[1:-1, 1:-1] = rglp.max()
        mask = rglp
        rglp = reconstruction(seed, mask, method='erosion')

        # new method, using the union of two diagonal erosions
        d_size = 7
        left_diag = np.zeros((d_size,d_size))
        for i in range(d_size):
            left_diag[i,i] = 1
        rglp_l = binary_erosion(rglp, selem = left_diag)
        rglp_r = binary_erosion(rglp, selem = np.fliplr(left_diag))
        rglp = rglp_l * rglp_r
        #rglp = rglp_l

        rglp = ndimage.filters.median_filter(rglp, footprint = rectangle(3,3))
        rglp[v2d] = 0

        markers = remove_small_objects(rglp)
        markers = ndimage.measurements.label(markers)[0]
        #markers = clear_border(markers)

        for s_label, s_centroid in forced_splits:
            # hack version
            markers[s_centroid[0]-5:s_centroid[0]+5] = 0

        segmentation = watershed((s_image.astype('int32') * -1), markers, mask = end_set)

        idata['phase_segments'] = np.pad(segmentation, ((0,idata['source_data'].shape[0]-analysis_length),(0,0)), mode='constant', constant_values=0)

        # eliminate any regions whose bounding box is wider than 2px less than the image width or which touch the border
        seglabels_for_del = []
        # find unique labels on the border: top, bottom, left, right edge
        border_values = np.unique(np.concatenate((idata['phase_segments'][0],
                                                  idata['phase_segments'][-1],
                                                  idata['phase_segments'][:,0],
                                                  idata['phase_segments'][:,-1])))
        for rp in regionprops(idata['phase_segments']):
            if rp.bbox[3] - rp.bbox[1] >= s_image.shape[1] - 2:
                seglabels_for_del.append(rp.label)
                continue # no need to check if it touches
            if rp.label in border_values:
                pass

        idata['phase_segments'] = merge_labeled_segments(idata['phase_segments'], seglabels_for_del, 0)
        
        # write fluorescence intensity maps
        rp_fl = []
        rp_fl.append(regionprops(label_image = idata['phase_segments']))
        for i_layer in range(1, idata['source_data'].shape[2]):
            rp_fl.append(regionprops(label_image = idata['phase_segments'], intensity_image = idata['source_data'][:,:,i_layer]))

        idata['regionprops'] = rp_fl
        
        return idata
    except:
        warning("Exception in cell segmentation:")
        warning("exc_info:\n", sys.exc_info()[1])
        warning("Traceback:\n", traceback.print_tb(sys.exc_info()[2]))
        warning(idata[1])
        raise


# worker function for foci detection
def find_foci(datalist):
    warning('foci not adapted to mm3 format!')
    raise Error
    idata, plane_name = datalist
    try:
        flindex = plane_names.tolist().index(plane_name)
    except:
        warning("Exception in find_foci:")
        warning(sys.exc_info()[0])
        warning(sys.exc_info()[1])
        warning(traceback.print_tb(sys.exc_info()[2]))
        raise ValueError("foci: plane name %s not found in fluorescence_channels %s." % (plane_name, plane_names))
    try:
        flimage = idata['source_data'][:,:,flindex]
        if np.sum(flimage) > 1:
            blobs_dog = blob_dog(np.ascontiguousarray(trim_zeros_2d(flimage)), min_sigma = 0.5, max_sigma = 6, threshold = 0.00005)
            g_fits = []
            for blob in blobs_dog:
                y, x, r = blob
                # 2d gaussian adapted from http://wiki.scipy.org/Cookbook/FittingData#head-11870c917b101bb0d4b34900a0da1b7deb613bf7
                if x >= 6 and y >= 6:
                    g_fit = fitgaussian(flimage[y-6:y+6,x-6:x+6])
                    g_fit[1] += (y - 6)
                    g_fit[2] += (x - 6)
                    segval = 0
                    try:
                        segval = idata['phase_segments'][int(g_fit[1]),int(g_fit[2])]
                    except:
                        pass
                    g_fit.append(segval)
                    g_fit.append(np.median(flimage))
                    g_fits.append(g_fit)
            idata['feature_data']['foci'][plane_name] = g_fits
            return idata
        else:
            return idata
    except:
        warning("Exception in find_foci:")
        warning(idata[0].shape)
        warning(sys.exc_info()[1])
        warning(traceback.print_tb(sys.exc_info()[2]))

# worker function for blob detection
def find_blobs(datalist):
    idata, plane_name = datalist
    try:
        flindex = plane_names.tolist().index(plane_name)
    except:
        raise ValueError("blobs: plane name %s not found in fluorescence_channels %s." % (plane_name, plane_names))
    try:
        flimage = idata['source_data'][:,:,flindex]
        if len(np.unique(flimage)) > 2:
            if plane_name in idata['blob_segments'].keys():
                pass
            else:
                macwe = morphsnakes.MorphACWE(flimage, smoothing = 1, lambda1 = 2, lambda2 = 1)
                startingbox = binary_dilation(remove_small_objects(flimage > skif.threshold_otsu(flimage)))
                # clear the borders
                startingbox[:,0:5] = False
                startingbox[:,-5:] = False
                macwe.levelset = startingbox
                macwe.run(30)
                found_contours = macwe.levelset[:].astype('int')
                segmentation = ndimage.measurements.label(found_contours)[0]
                segmentation[idata['phase_segments'] == 0] = 0
                idata['blob_segments'][plane_name] = segmentation

            blobs = regionprops(label_image = idata['blob_segments'][plane_name], intensity_image = flimage)

            blob_assignments = []
            for blob in blobs:
                segval = spstats.mstats.mode(idata['phase_segments'][segmentation == blob.label])[0][0]
                blob_assignments.append((segval, blob))
            idata['feature_data']['blobs'][plane_name] = blob_assignments
            return idata
        else:
            idata['blob_segments'][plane_name] = np.zeros_like(flimage)
            return idata
    except:
        warning("Exception in find_blobs:")
        warning(str(idata['phase_segments'].shape) + " " + str(segmentation.shape))
        warning(sys.exc_info()[1])
        warning(traceback.print_tb(sys.exc_info()[2]))

# worker function for getting long-axis fluorescence profiles
def find_profiles(datalist):
    warning('profiles not adapted to mm3 format!')
    raise Error

    idata, plane_name = datalist
    try:
        flindex = plane_names.tolist().index(plane_name)
    except:
        raise ValueError('profiles: plane name %s not found in p %s.' % (plane_name, plane_names))
    try:
        flimage = idata[0][:,:,flindex]
        if len(np.unique(flimage)) > 2:
            # get the profile lines for all cells
            all_rp_profiles = {}
            for rps in idata[2][0]: # for each rp in the no-fluorescence rp set

                y0, x0 = rps.centroid
                cosorient = math.cos(rps.orientation)
                sinorient = math.sin(rps.orientation)

                x1 = x0 + cosorient * 0.5 * rps.major_axis_length
                y1 = y0 - sinorient * 0.5 * rps.major_axis_length
                x2 = x0 - cosorient * 0.5 * rps.major_axis_length
                y2 = y0 + sinorient * 0.5 * rps.major_axis_length

                # make sure that the start of the profile is at the start of the image
                if y2 < y1:
                    x1, y1, x2, y2 = x2, y2, x1, y1

                # first compute the Feret length of the cell
                b_image = (idata[1] == rps.label).astype('int8')
                b_profile = np.array(profile_line(b_image, (y1, x1), (y2, x2), linewidth = 2, order = 0)).astype('bool')

                # next compute the fluorescence profile line and use the b_profile to mask it to relevant values
                l_profile = np.array(profile_line(flimage, (y1, x1), (y2, x2), linewidth = 2, order = 0)).astype('float32')
                all_rp_profiles[rps.label] = l_profile[b_profile]

            idata[6]['profiles'][plane_name] = all_rp_profiles
            return idata
        else:
            idata[6]['profiles'][plane_name] = {}
            return idata
    except:
        warning("Exception in find_profile:")
        warning(sys.exc_info()[0])
        warning(sys.exc_info()[1])
        warning(traceback.print_tb(sys.exc_info()[2]))


# worker function for getting a length-profile of fluorescence intensity
def intensity_profile(idata, flindex = 1):
    try:
        flimage = idata[:,:,flindex]
        if len(np.unique(flimage)) > 2:
            pass
        else:
            # need a new place to put this data. maybe idata[6] could be a dictionary of named functions?
            idata[6] = []
            return idata
    except:
        warning("Exception in find_blobs:")
        print(sys.exc_info()[1])
        print(traceback.print_tb(sys.exc_info()[2]))
        raise


# if this is the main show, run it!
if __name__ == "__main__":
    param_file = ""

    # switches
    try:
        opts, args = getopt.getopt(sys.argv[1:],"ro:s:f:e:p:lk:a:")
    except getopt.GetoptError:
        print('No arguments detected (-s <fov> -o <fov(s)> -f <params> -e <fov>).')
    for opt, arg in opts:
        if opt == '-o':
            try:
                specify_fovs = True
                for fov_to_proc in arg.split(","):
                    user_spec_fovs.append(int(fov_to_proc))
            except:
                print("Couldn't convert argument to an integer:",arg)
                raise ValueError
        if opt == '-s':
            try:
                start_with_fov = int(arg)
            except:
                print("Couldn't convert start FOV to an integer:",arg)
                raise ValueError
        if opt == '-e':
            try:
                end_with_fov = int(arg)
            except:
                print("Couldn't convert end FOV to an integer:",arg)
                raise ValueError
        if opt == '-k':
            try:
                one_peak = int(arg)
            except:
                print("Couldn't convert single peak index to an integer:", arg)
                raise ValueError
        if opt == '-p':
            try:
                num_procs = int(arg)
            except:
                print("Couldn't convert processor count to an integer:",arg)
                raise ValueError
        if opt == '-f':
            param_file = arg
        if opt == '-r':
            reuse_segmentations = True
        if opt == '-l':
            show_plots = True
        if opt == '-a':
            algos = ["segment_cells_threshsobel", "segment_cells_ac6"]
            if arg in algos:
                use_algorithm = arg
            else:
                print("Invalid algorithm selection, please use one of these: " + str(algos))
                raise ValueError

    # Load the project parameters file
    if len(param_file) == 0:
        raise ValueError("A parameter file must be specified (-f <filename>).")
    information("Loading parameters...")
    parameters_file = open(param_file, 'r')
    globals().update(yaml.load(parameters_file))

    # Make the channel_otsu value a global
    channel_otsu = 0

    # get a list of all FOVs to process
    # Load the images into a list for processing
    fov_file_list = fnmatch.filter(os.listdir(experiment_directory + analysis_directory + 'subtracted/'), 'subtracted_*.hdf5')
    fov_file_list.sort()
    information("-------------------------------------------------")
    information("Found %d FOVs to process." % len(fov_file_list))

    for fov_file in fov_file_list:

        fov = int(fov_file.split("_")[1].split(".")[0])

        if specify_fovs and not (fov in user_spec_fovs):
            continue
        if start_with_fov > 0 and fov < start_with_fov:
            continue
        if end_with_fov > 0 and fov > end_with_fov:
            continue

        information("*** Starting FOV %d" % fov)

        with h5py.File(experiment_directory + analysis_directory + 'subtracted/' + fov_file, 'r', libver='latest', swmr=True) as h5f:
            # load channel peaks into a list for processing & clear out pre-existing segment arrays
            peaks = []
            for item in h5f.keys():
                if item.split("ted_")[0] == "subtrac":
                    peaks.append(int(item.split("ted_")[1]))

        fov_mother_map = []
        fov_otsu_data = []

        # Collect a list of all channels with cells from the first t_index from the FOV in question
        peaks_with_cells = []

        original_n_peak = -1

        ## ok the big loop:
        for n_peak in peaks:
            information("* starting analysis of peak %d, FOV %d." % (n_peak, fov))
            original_n_peak = n_peak
            start_analysis_index = 0
            prior_flat_cells = None

            try:
                if one_peak > -1 and original_n_peak != one_peak:
                    continue

                ##### reload the previous analysis to know where to restart
                # first try to load data from a previously saved cell chain
                # prior_flat_iterator loops down the cell chain;
                # restart_iter stays one cell behind prior_flat_iter & it's
                #     cell_id is later used to join the new data to the old
                #     growing chain
                if os.path.exists(experiment_directory + analysis_directory + 'segmentations/segments_%03d.hdf5' % fov):
                    with h5py.File(experiment_directory + analysis_directory + 'segmentations/segments_%03d.hdf5' % fov, 'r', libver='latest', swmr=True) as h5f:
                        segkey = 'segments_%04d' % original_n_peak
                        if segkey in h5f.keys():
                            start_segmentation = h5f[segkey].shape[0]
                else:
                    start_segmentation = 0

                # get raw images & metadata from the hdf5 file
                with h5py.File(experiment_directory + analysis_directory + 'subtracted/' + fov_file, 'r', libver='latest', swmr=True) as h5f:
                    if h5f[u'subtracted_%04d' % n_peak].shape[0] <= start_segmentation:
                        information('no new data for FOV %d, peak %d; skipping.' % (fov, original_n_peak))
                        continue
                    images = h5f[u'subtracted_%04d' % n_peak][start_segmentation:]
                    information("subtracted image array shape:", images.shape)

                    global plane_names
                    if 'plane_names' in h5f[u'subtracted_%04d' % n_peak].attrs:
                        plane_names = h5f[u'subtracted_%04d' % n_peak].attrs['plane_names']
                    else:
                        plane_names = global_plane_names

                    i_times = h5f[u'metadata'][:,2]

                '''
                # reload segmentation data if possible, otherwise set
                # some variables to None
                if start_analysis_index > 0:
                    if os.path.exists(experiment_directory + analysis_directory + 'segmentations/segments_%03d.hdf5' % fov):
                        with h5py.File(experiment_directory + analysis_directory + 'segmentations/segments_%03d.hdf5' % fov, 'r', libver='latest') as h5segs:
                            if 'segments_%04d' % original_n_peak in h5segs.keys():
                                segment_data = h5segs[u'segments_%04d' % n_peak][start_analysis_index:]
                            else:
                                warning('segmentation data not found for peak %d' % fov)
                                reuse_segmentations = False
                            blobsegs = {}
                            for bplane in blob_planes:
                                if "blobs_%s_%04d" % (bplane, original_n_peak) in h5segs.keys():
                                    blobsegs[bplane] = h5segs["blobs_%s_%04d" % (bplane, original_n_peak)][start_analysis_index:]
                                else:
                                    warning('blob segmentation data not found for peak %d, plane %s.' % (original_n_peak, bplane))
                                    blobsegs[bplane] = None
                    else:
                        warning('segmentation data file not found.')
                        reuse_segmentations = False
                '''

                # set up image_data
                information("populating main data structure...")
                image_data = []
                for n in xrange(0,len(images)):
                    blobsegsn = {}
                    segments = None
                    if reuse_segmentations:
                        segments = segment_data[n]
                        for bplane in blob_planes:
                            if blobsegs[bplane] is not None:
                                blobsegsn[bplane] = blobsegs[bplane][n]

                    image_data.append({'source_data': images[n],
                                       'phase_segments': segments,
                                       'regionprops': None,
                                       'lowest_cell_label': None,
                                       'image_time': i_times[n],
                                       'acquisition_index': n + start_analysis_index,
                                       'blob_segments': blobsegsn,
                                       'feature_data': {'blobs': {}, 'foci': {}, 'profiles': {}},
                                       'otsu_threshold': -1,
                                       'daughter_map': None,
                                       'fov': fov,
                                       'peak': original_n_peak,
                                       'warnings': [],
                                       'flag_next': []})

                # First-pass segmentation
                information('First-pass image segmentation...')
                pool = Pool(num_procs)
                image_data = pool.map(eval(use_algorithm), image_data)
                pool.close()
                pool.join()

                ## fluorescent foci/blob/profile detection
                for foci_plane in foci_planes:
                    information("running foci detection...")
                    pool = Pool(num_procs)
                    image_data = pool.map(find_foci, zip(image_data, [foci_plane for n in range(len(image_data))]))
                    pool.close()
                    pool.join()
                for blob_plane in blob_planes:
                    information("running blob detection...")
                    pool = Pool(num_procs)
                    image_data = pool.map(find_blobs, zip(image_data, [blob_plane for n in range(len(image_data))]))
                    pool.close()
                    pool.join()
                for profile_plane in profile_planes:
                    information("running profile analysis...")
                    pool = Pool(num_procs)
                    image_data = pool.map(find_profiles, zip(image_data, [profile_plane for n in range(len(image_data))]))
                    pool.close()
                    pool.join()
                ## end fluorescence analysis

                ### write out labeled segments
                information('saving phase contrast segmentations...')
                ## build an array containing segmented data
                seg_images = [0 for i in image_data]
                # get the max image size for saving
                max_x = np.amax([idata['phase_segments'].shape[1] for idata in image_data])
                max_y = np.amax([idata['phase_segments'].shape[0] for idata in image_data])
                assert(max_x > 0)
                assert(max_y > 0)
                for i in range(0,len(image_data)):
                    if image_data[i]['phase_segments'].shape[0] < max_y or image_data[i]['phase_segments'].shape[1] < max_x:
                        seg_images[i] = np.pad(image_data[i]['phase_segments'],
                                               ((0, max_y - image_data[i]['phase_segments'].shape[0]), (0, max_y - image_data[i]['phase_segments'].shape[0])),
                                               mode = 'constant', constant_values = 0)
                    else:
                        seg_images[i] = image_data[i]['phase_segments']
                # write the array to disk, if not reusing old segmentations
                try:
                    if not os.path.exists(experiment_directory + analysis_directory + 'segmentations/'):
                        os.makedirs(experiment_directory + analysis_directory + 'segmentations/')
                    with h5py.File(experiment_directory + analysis_directory + 'segmentations/segments_%03d.hdf5' % fov) as h5segs:
                        if "segments_%04d" % original_n_peak in h5segs.keys():
                            h5phsegs = h5segs["segments_%04d" % original_n_peak]
                            prior_ax0_size = h5phsegs.shape[0]
                            new_ax0_size = prior_ax0_size + len(seg_images)
                            h5phsegs.resize(new_ax0_size, axis = 0)
                            h5phsegs[prior_ax0_size:] = np.array(seg_images)
                        else:
                            h5phsegs = h5segs.create_dataset("segments_%04d" % original_n_peak,
                                                             data=np.array(seg_images),
                                                             maxshape = (None, max_y, max_x),
                                                             compression="gzip", shuffle = True, fletcher32 = True)
                except:
                    warning("failed to save segmentations!")
                    print(sys.exc_info()[0])
                    print(sys.exc_info()[1])
                    print(traceback.print_tb(sys.exc_info()[2]))

                ### write out the segmented blobs to disk
                # build an array of blob segments for each plane which was blobbed
                for bplane in blob_planes:
                    information('saving blob segmentations...')
                    max_x = np.amax([idata['blob_segments'][bplane].shape[1] for idata in image_data if bplane in idata['blob_segments'].keys()])
                    max_y = np.amax([idata['blob_segments'][bplane].shape[0] for idata in image_data if bplane in idata['blob_segments'].keys()])
                    assert(max_x > 0)
                    assert(max_y > 0)

                    # get the maps and resize them to be uniform
                    blobsegs = [idata['blob_segments'][bplane] if bplane in idata['blob_segments'].keys() else np.zeros((max_y, max_x)) for idata in image_data]
                    for i in range(len(blobsegs)):
                        if blobsegs[i].shape[0] < max_y or blobsegs[i].shape[1] < max_x:
                            blobsegs[i] = np.pad(blobsegs[i],
                                                 ((0, max_y - blobsegs[i].shape[0]), (0, max_y - blobsegs[i].shape[0])),
                                                 mode = 'constant', constant_values = 0)

                    try:
                        if not os.path.exists(experiment_directory + analysis_directory + 'segmentations/'):
                            os.makedirs(experiment_directory + analysis_directory + 'segmentations/')
                        with h5py.File(experiment_directory + analysis_directory + 'segmentations/segments_%03d.hdf5' % fov) as h5segs:
                            if "blobs_%s_%04d" % (bplane, original_n_peak) in h5segs.keys():
                                h5blobsegs = h5segs["blobs_%s_%04d" % (bplane, original_n_peak)]
                                prior_ax0_size = h5blobsegs.shape[0]
                                new_ax0_size = prior_ax0_size + len(blobsegs)
                                h5blobsegs.resize(new_ax0_size, axis = 0)
                                h5blobsegs[prior_ax0_size:] = np.array(blobsegs[prior_ax0_size:])
                            else:
                                segset = h5segs.create_dataset("blobs_%s_%04d" % (bplane, original_n_peak),
                                                            data=np.asarray(blobsegs),
                                                            maxshape = (None, max_y, max_x),
                                                            compression="gzip", shuffle = True, fletcher32 = True)
                    except:
                        warning("failed to save blob segmentations (%s)!" % bplane)
                        print(sys.exc_info()[0])
                        print(sys.exc_info()[1])
                        print(traceback.print_tb(sys.exc_info()[2]))

            except KeyboardInterrupt:
                warning("******** Keyboard Interrupt ********")
                raise KeyboardInterrupt
            except:
                warning("******** FAILED CHANNEL %d" % original_n_peak)
                print(sys.exc_info()[0])
                print(sys.exc_info()[1])
                print(traceback.print_tb(sys.exc_info()[2]))
                raise

        # Write out the accumulated fov_mother_map for the FOV
        information("Writing mother map file for FOV %02d..." % fov)
        if not os.path.exists(experiment_directory + analysis_directory + "mothermaps/"):
            os.makedirs(experiment_directory + analysis_directory + "mothermaps/")
        with open(experiment_directory + analysis_directory + "mothermaps/mother_map_%03d.pkl" % fov, 'wb') as out_params:
            pickle.dump(fov_mother_map, out_params, protocol = 2)
