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
    num_procs = 15
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
        s_image = image.astype('int32')
        s_image -= np.min(image)
                
        idata['otsu_threshold'] = skif.threshold_otsu(s_image)

        gI = morphsnakes.gborders(s_image, alpha=0.1, sigma=1.) 
        macwe = morphsnakes.MorphGAC(gI, smoothing=2, threshold=0.5, balloon=-1)
        macwe.levelset = binary_dilation(remove_small_objects(s_image > (idata['otsu_threshold'] + (0.2 * float(np.amax(s_image) - idata['otsu_threshold'])))), selem = disk(5))
        macwe.run(30)
        end_set = macwe.levelset[:].astype('bool')

        r_image = exposure.rescale_intensity(s_image.astype('float64'))
                
        # set to True to enable more aggressive image denoising
        if True:
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
        # old method
        #rglp = binary_erosion(rglp, selem = disk(3))
        left_diag = np.zeros((6,6))
        # new method, using the union of two diagonal erosions
        for i in range(6):
            left_diag[i,i] = 1
        rglp_l = binary_erosion(rglp, selem = left_diag)
        rglp_r = binary_erosion(rglp, selem = np.fliplr(left_diag))
        rglp = rglp_l * rglp_r

        rglp = ndimage.filters.median_filter(rglp, footprint = rectangle(3,3))
        rglp[v2d] = 0
        
        markers = remove_small_objects(rglp)
        markers = ndimage.measurements.label(markers)[0]
        #markers = clear_border(markers)
                
        for s_label, s_centroid in forced_splits:
            # hack version
            markers[s_centroid[0]-5:s_centroid[0]+5] = 0
        
        segmentation = watershed((s_image.astype('int32') * -1), markers, mask = end_set)
        
        idata['phase_segments'] = segmentation
        
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
        idata['lowest_cell_label'] = get_lowest_cell_label(idata['regionprops'][0])
        return idata
    except:
        warning("Exception in cell segmentation:")
        warning("exc_info:\n", sys.exc_info()[1])
        warning("Traceback:\n", traceback.print_tb(sys.exc_info()[2]))
        warning(idata[1])
        raise

# run time-dependent resegmentation...in theory
def resegment(bad_index):
    '''
    resegment an index based on the bracketing indexes
    this is definitely not written yet...
    '''
    
# get ground-truth length/width data for a segment
def get_lw(rps, b_image):
    y0, x0 = rps.centroid
    cosorient = math.cos(rps.orientation)
    sinorient = math.sin(rps.orientation)

    x1 = x0 + cosorient * 0.5 * rps.major_axis_length
    y1 = y0 - sinorient * 0.5 * rps.major_axis_length
    x2 = x0 - cosorient * 0.5 * rps.major_axis_length
    y2 = y0 + sinorient * 0.5 * rps.major_axis_length

    l_profile = np.array(profile_line(b_image, (y1, x1), (y2, x2), linewidth = 2, order = 0)).astype('float64')
    l_imp = (l_profile.sum() * rps.major_axis_length)/float(len(l_profile))

    x1 = x0 - sinorient * 0.5 * rps.minor_axis_length
    y1 = y0 - cosorient * 0.5 * rps.minor_axis_length
    x2 = x0 + sinorient * 0.5 * rps.minor_axis_length
    y2 = y0 + cosorient * 0.5 * rps.minor_axis_length

    w_profile = np.array(profile_line(b_image, (y1, x1), (y2, x2), linewidth = 2, order = 0)).astype('float64')
    w_imp = (w_profile.sum() * rps.minor_axis_length)/float(len(w_profile))
    
    return (l_imp, w_imp)
# map cell segments at a given index to the next index
def map_segments(mindex):
    # if the index is the last in the dataset, return without mapping
    if mindex == len(image_data) - 1:
        return image_data[mindex]
    
    map_from = image_data[mindex]
    map_to = image_data[mindex+1]
        
    map_results = {}
    
    # first, collapse the gap regions
    nz_from = (map_from['phase_segments'] != 0).sum(1) # sum along rows
    nz_to = (map_to['phase_segments'] != 0).sum(1)
    collapsed_from = map_from['phase_segments'][nz_from > 0,:]
    collapsed_to = map_to['phase_segments'][nz_to > 0,:]
    
    # resize the collapsed arrays
    max_length = max(collapsed_to.shape[0], collapsed_from.shape[0])
    if collapsed_from.shape[0] < max_length:
        collapsed_from = np.pad(collapsed_from, ((0, max_length - collapsed_from.shape[0]), (0, 0)), mode = 'constant', constant_values = 0)
    if collapsed_to.shape[0] < max_length:
        collapsed_to = np.pad(collapsed_to, ((0, max_length - collapsed_to.shape[0]), (0, 0)), mode = 'constant', constant_values = 0)
    
    # for each label in the source, find the max overlap in the destination
    for slabel in np.unique(collapsed_from[collapsed_from > 0]):
        binary_map_source = collapsed_from == slabel
        
        dest_labels = collapsed_to[binary_map_source]
        
        # if there are no non-zero labels in the destination but the destination is indeed there
        if np.count_nonzero(dest_labels) == 0:
            map_results[slabel] = []
            continue
            
        overlap_labels, overlap_label_counts = np.unique(dest_labels[dest_labels > 0], return_counts = True)
        
        # if there is only one unique overlap label, then we'll say that's it.
        if len(overlap_labels) == 1:
            map_results[slabel] = [overlap_labels[0], ]
            continue
                                            
        # sort the overlaps to see what's where
        sortorder = np.argsort(overlap_label_counts)[::-1]
        overlap_labels = overlap_labels[sortorder]
        overlap_label_counts = overlap_label_counts[sortorder]
        
        # if the next mapped cell has an area > 2x the current cell's area,
        # there is probably a segmentation problem.
        if slabel != np.amax(collapsed_from):
            area_next = float(np.sum(map_to['phase_segments'] == overlap_labels[0]))
            area_current = float(np.sum(binary_map_source))
            area_ratio = area_next/area_current
            if area_ratio > 2.:
                map_from['warnings'].append('map_segments: sudden increase in cell size at next frame (%d, %d, %d, %d, %0.4f, %s, %s)' % (slabel, overlap_labels[0], area_current, area_next, area_ratio, str(overlap_labels), str(overlap_label_counts)))
                map_from['flag_next'].append(overlap_labels[0])
                
        # check to see if the top two destination segments overlap better with anything else
        label_a, label_b = overlap_labels[0], overlap_labels[1]
        area_a, area_b = float((collapsed_to == label_a).sum()), float((collapsed_to == label_b).sum())
        ct_a = collapsed_from[collapsed_to == label_a]
        ct_b = collapsed_from[collapsed_to == label_b]
        ct_a_labels, ct_a_labels_counts = np.unique(ct_a[ct_a > 0], return_counts = True)
        ct_b_labels, ct_b_labels_counts = np.unique(ct_b[ct_b > 0], return_counts = True)
        
        # if both of the top two overlaps have only one backwards overlap, then call it a division
        if len(ct_a_labels) == 1 and len(ct_b_labels) == 1:
            map_results[slabel] = [label_a, label_b]
            continue
        
        # if one dest. label has only 1 overlap and the other dest. label is mostly with slabel, call the division
        if ((len(ct_a_labels) == 1) != (len(ct_b_labels) == 1)) and (float(np.sum(ct_a == slabel)/area_a > 0.8) != float(np.sum(ct_b == slabel)/area_b > 0.8)):
            map_results[slabel] = [label_a, label_b]
            continue
            
        # if both destination labels have more than 80% overlap with slabel, call the division
        if (float(np.sum(ct_a == slabel)/area_a > 0.8) and float(np.sum(ct_b == slabel)/area_b > 0.8)):
            map_results[slabel] = [label_a, label_b]
            continue

        # if the above switches haven't caught, let 'em all hang around
        map_results[slabel] = list(overlap_labels)

    # collapse multiple forward-mapping events into a single mapping event.
    # this is kind of a hack...
    for dlabel in np.unique(collapsed_to):
        if dlabel == 0:
            continue
        # get all source cells that map to this one
        slabel_matches = [slabel for slabel in map_results.keys() if dlabel in map_results[slabel]]
        # if there is more than one source label, clean it up.
        if len(slabel_matches) > 1:
            # compute how much each of slabels are covered by dlabel
            overlaps = [float(((collapsed_from == slabel)[collapsed_to == dlabel]).sum())/float((collapsed_to == dlabel).sum()) for slabel in slabel_matches]
            # if one is less than 10%, it's probably just normal growth
            if np.amin(overlaps) < 0.15:
                winner = slabel_matches.pop(np.argmax(overlaps))
                for losing_label in slabel_matches:
                    map_results[losing_label].pop(map_results[losing_label].index(dlabel))
            # or if one of the slabels is the max slabel, it's probably not worth thinking about; cut the tracking short
            elif np.all([s >= 3 for s in slabel_matches]):
                for sl in slabel_matches:
                    map_results[sl] = []
            else:
                warning('collapsing division (index %d, labels %s -> label %d)' % (map_from['acquisition_index'], str(slabel_matches), dlabel))
                # map to dlabel from the lowest slabel
                winner = slabel_matches.pop(np.argmin(overlaps))
                for losing_label in slabel_matches:
                    map_results[losing_label].pop(map_results[losing_label].index(dlabel))
    
    # a rough sanity check: if there are segments in map_from and map_to, make sure something, at least, maps between them
    if len(np.unique(map_from['phase_segments'])) > 1 and len(np.unique(map_to['phase_segments'])) > 1:
        something_maps = False
        for slabel in map_results.keys():
            has_match = np.any(np.array(map_results[slabel]) > 0)
            if has_match:
                something_maps = True
                break
        if not something_maps:
            raise ValueError('segment disjoint at at index %d' % mindex)
    
    map_from['daughter_map'] = map_results
    return map_from
    
# write segment graph
def saveGraph(name):
    if pydot is None:
        warning('pydot is not defined')
        return -1
    else:
        pdgraph = pydot.Dot(graph_type='graph')
        for node in cellgraph.keys():
            for destnode in cellgraph[node]:
                edge = pydot.Edge(node, destnode)
                pdgraph.add_edge(edge)
        return pdgraph

# get a hash for a given fov+peak+birth_source_index+segment_label
def hashCell(fov, peak, birth_acquisition_index, birth_label):
    cdata = (fov, peak, birth_acquisition_index, birth_label)
    return hashlib.sha224(str(['_'.join([str(d) for d in cdata])] * 10)).hexdigest()

# construct a dictionary graph based on the source_index of a cell and it's label
def getDaughterGraph():
    graph = {}
    for i in range(len(image_data) - 1):
        for cl in image_data[i]['daughter_map'].keys():
            if image_data[i]['daughter_map'][cl] is not None:
                graph[hashcell(i, cl)] = [hashcell(i + 1, dcl) if dcl is not None else hashcell(i+1,99) for dcl in image_data[i]['daughter_map'][cl]]
    return graph
# recursively map a chain of cell segments                
def getSegmentChain(image_index, segment_label):
    '''
    Recursively genererate a chain of cell segments from one timepoint to the next
    '''
    #if not type(image_index) == type(0):
    #    raise ValueError('image_index type not int (%s).' % str(type(image_index)))
    #if not type(segment_label) == type(0):
    #    raise ValueError('segment_label type not int (%s).' % str(type(segment_label)))
    
    if image_index == len(image_data) - 1:
        return None
    
    if image_index is None:
        raise ValueError('recieved NoneType for image_index.')
    if segment_label is None:
        raise ValueError('recieved NoneType for segment_label.')

    #information('Index %d: Tracking cell %d...' % (image_index, segment_label))
    try:
        parent_label = -1
        parent_area = -1
        parent_rp = -1

        # confirm the existence of the cell to track
        for (segindex, crp) in enumerate(image_data[image_index]['regionprops'][0]):
            if crp.label == segment_label:
                parent_label = crp.label
                parent_rp = crp
                parent_area = crp.area
                break
            
        if parent_label < 0:
            warning("Frame %04d: cell label %d not found." % (image_index, segment_label))
            return None

        # recursion
        if image_data[image_index]['daughter_map'][parent_label] is None or image_index == len(image_data) - 1:
            daughters = None
        else:
            #daughters = [getSegmentChain(image_index+1, label) for label in image_data[image_index]['daughter_map'][parent_label]]
            daughters = []
            # run through the sorted list such that the mothers get visited first
            for dlabel in sorted(image_data[image_index]['daughter_map'][parent_label]):
                if dlabel is not None:
                    daughters.append(getSegmentChain(image_index+1, dlabel))
    
        feature_data = {}
        for dt in image_data[image_index]['feature_data'].keys():
            feature_data[dt] = {}
            for fplane in image_data[image_index]['feature_data'][dt]:
                feature_data[dt][fplane] = [f[1] for f in image_data[image_index]['feature_data'][dt][fplane] if f[0] == parent_label]

        return {'rp': crp, 
                'segment_hash': hashCell(fov = image_data[image_index]['fov'], 
                                         peak = image_data[image_index]['peak'], 
                                         birth_acquisition_index = image_data[image_index]['acquisition_index'], 
                                         birth_label = crp.label),
                'daughters': daughters, 
                'feature_data': feature_data,
                'time': image_data[image_index]['image_time'], 
                'source_index': image_index,
                'acquisition_index': image_data[image_index]['acquisition_index'],
                'fov': image_data[image_index]['fov'],
                'peak': image_data[image_index]['peak'],
                }
    except:
        warning("Exception in getSegmentChain (index %d, segment %d):" % (image_index, segment_label))
        warning(sys.exc_info()[0])
        warning(sys.exc_info()[1])
        warning(traceback.print_tb(sys.exc_info()[2]))
        raise

# recursively flatten a cell chain
def getFlattenedChain(chain):
    '''
    flatten chains of single-segment daughters into aggregated data and
    recursively flatten daughter cells
    '''

    if chain['daughters'] is None or chain is None:
        return None
    
    # first, enumerate the cell chain up to the point where the number of daughters != 1
    cell_life = []
    while (chain is not None) and (chain['daughters'] is not None) and (len(chain['daughters']) == 1):
        cell_life.append({k: v for k, v in chain.items() if not k == 'daughters'})
        chain = chain['daughters'][0]
    else:
        if chain is None:
            return None
        else:
            cell_life.append({k: v for k, v in chain.items() if not k == 'daughters'})
    # this leaves chain on the last measurement before division but includes that last measurement in cell_life
        
    ##### collate cell life data over time
    # get all real lengths/widths
    cell_lws = [get_lw(c['rp'], image_data[c['source_index']]['phase_segments'] == c['rp'].label) for c in cell_life]
    
    # start with basic data which doesn't require computation or sorting
    cell = {'cell_id': cell_life[0]['segment_hash'],
            'fov': cell_life[0]['fov'],
            'peak': cell_life[0]['peak'],
            'all_timepoints': [c['time'] for c in cell_life],
            'all_acquisition_indexes': [c['acquisition_index'] for c in cell_life],
            'all_source_indexes': [c['source_index'] for c in cell_life],
            'all_lengths': [clw[0] for clw in cell_lws],
            'all_widths': [clw[1] for clw in cell_lws],
            'birth_time': cell_life[0]['time'],
            'birth_index': cell_life[0]['acquisition_index'],
            'birth_length': cell_lws[0][0],
            'birth_width': cell_lws[0][1],
            'birth_area': cell_life[0]['rp'].area,
            'division_length': cell_lws[-1][0],
            'division_width': cell_lws[-1][1],
            'division_area': cell_life[-1]['rp'].area,
            'division_time': cell_life[-1]['time'],
            'division_index': cell_life[-1]['acquisition_index'],
            'thresholds': [image_data[c['source_index']]['otsu_threshold'] for c in cell_life],
            'bounding_boxes': [c['rp'].bbox for c in cell_life],
            'solidities': [c['rp'].solidity for c in cell_life],
            'centroids': [c['rp'].centroid for c in cell_life],
            'equivalent_diameters': [c['rp'].equivalent_diameter for c in cell_life],
            'circularities': [(4.*math.pi*float(c['rp'].area))/(c['rp'].perimeter ** 2.) for c in cell_life], # 1 = perfectly circular, 0 = not at all circular
            }
            
    # elongation rate
    if len(cell_life) > 1:
        res = spstats.linregress([t * 24. for t in cell['all_timepoints']], np.log(cell['all_lengths']))
        cell['elongation_rate'], cell['elongation_bintercept'], cell['elongation_r'], m_er_p, m_er_stderr = res
        cell['elongation_rsquared'] = cell['elongation_r']**2
    else:
        cell['elongation_rate'], cell['elongation_bintercept'], cell['elongation_r'], cell['elongation_rsquared'] = [np.nan] * 4
    
    # division length inferred from daughters
    if len(chain['daughters']) == 2:
        d0 = chain['daughters'][0]
        d1 = chain['daughters'][1]
        d0l = get_lw(d0['rp'], image_data[d0['source_index']]['phase_segments'] == d0['rp'].label)[0]
        d1l = get_lw(d0['rp'], image_data[d1['source_index']]['phase_segments'] == d1['rp'].label)[0]
        cell['division_length_inf'] = d0l + d1l
    else:
        cell['division_length_inf'] = np.nan
    
    # mean fluorescence intensities
    #...
    
    # daughter data
    cell['daughters'] = [getFlattenedChain(d) for d in chain['daughters']]
    
    return cell
    

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
# return the label of the segment closest to the closed end of the channel
def get_lowest_cell_label(rps, area_cutoff = 150):
    '''
    takes a list of regionprops objects and returns the label of the lowest cell
    '''
    lowestcell = None
    for crp in rps:
        if (lowestcell is None) or (crp.centroid[0] < lowestcell.centroid[0] and crp.area > area_cutoff):
            lowestcell = crp
    if lowestcell is None:
        return None
    else:
        return lowestcell.label
    
# worker function to load segment properties
def load_rp_data(idata):
    try:
        # write fluorescence intensity maps
        rp_fl = []
        rp_fl.append(regionprops(label_image = idata['phase_segments']))
        for i_layer in range(1, idata['source_data'].shape[2]):
            rp_fl.append(regionprops(label_image = idata['phase_segments'], intensity_image = idata['source_data'][:,:,i_layer]))
        idata['regionprops'] = rp_fl
        
        idata['lowest_cell_label'] = get_lowest_cell_label(idata['regionprops'][0])
            
        return idata
    except:
        print ("Exception in load_rp_data:")
        print (sys.exc_info()[1])
        print (traceback.print_tb(sys.exc_info()[2]))
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
    TEMP_LAST_IMAGE = 500
    
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
                if os.path.exists(experiment_directory + analysis_directory + 'cell_trees/fov_%03d_peak_%04d.mshl' % (fov, original_n_peak)):
                    information('Loading prior flat chain...')
                    with open(experiment_directory + analysis_directory + 'cell_trees/fov_%03d_peak_%04d.mshl' % (fov, original_n_peak), 'r') as fh:
                        prior_flat_cells = marshal.load(fh)
                    information('Analyzing prior flat chain...')
                    prior_flat_iterator = prior_flat_cells
                    restart_iter = prior_flat_cells
                    chain_count = 0
                    while prior_flat_iterator['daughters'] is not None:
                        if len(prior_flat_iterator['daughters']) == 0:
                            if len(sib_iterator['daughters']) > 0:
                                prior_flat_iterator, sib_iterator = sib_iterator, prior_flat_iterator
                            else:
                                raise ValueError('no daughter trap.')
                        elif len(prior_flat_iterator['daughters']) == 1:
                            restart_iter = prior_flat_iterator
                            prior_flat_iterator = prior_flat_iterator['daughters'][0]
                        elif len(prior_flat_iterator['daughters']) == 2:
                            restart_iter = prior_flat_iterator
                            d0 = prior_flat_iterator['daughters'][0]
                            d1 = prior_flat_iterator['daughters'][1]
                            if d0['centroids'][0][0] > d1['centroids'][0][0]:
                                prior_flat_iterator = d1
                                sib_iterator = d0
                            else:
                                prior_flat_iterator = d0
                                sib_iterator = d1
                    # get the last acquisition index for the last cell observed,
                    # then set the start point to 20 points before that to get the 
                    start_analysis_index = min(restart_iter['all_acquisition_indexes'][0] - 20, 0)
                    # if the new analysis start point is 0, don't bother with reloading anything
                    if start_analysis_index > 0:
                        reuse_segmentations = True

                # get raw images & metadata from the hdf5 file
                with h5py.File(experiment_directory + analysis_directory + 'subtracted/' + fov_file, 'r', libver='latest', swmr=True) as h5f:
                    images = h5f[u'subtracted_%04d' % n_peak][start_analysis_index:TEMP_LAST_IMAGE]
                    information("subtracted image array shape:", images.shape)
                
                    global plane_names
                    if 'plane_names' in h5f[u'subtracted_%04d' % n_peak].attrs:
                        plane_names = h5f[u'subtracted_%04d' % n_peak].attrs['plane_names']
                    else:
                        plane_names = global_plane_names
                
                    i_times = h5f[u'metadata'][:,2]
                
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
                
                print ("-")
            
                # trim images in parallel
                m = Manager()
                q = m.Queue()
                qglobal = m.Queue()
                pool = Pool(num_procs)
                pool_result = pool.map_async(trim_zeros_br, images, chunksize = 20)
                pool.close()
                '''
                try:
                    while (True):
                        if (pool_result.ready()): break
                        size = q.qsize()
                        CURSOR_UP_ONE = '\x1b[1A'
                        ERASE_LINE = '\x1b[2K'
                        print(CURSOR_UP_ONE + ERASE_LINE + CURSOR_UP_ONE)
                        information("trimmed", size, "of", len(images), "images...")
                        time.sleep(1)
                except KeyboardInterrupt:
                        warning("caught KeyboardInterrupt, terminating workers.")
                        pool.terminate()
                        pool.join()
                        raise KeyboardInterrupt
                except:
                    print ("Error in parallel image trimming:")
                    print (sys.exc_info()[1])
                    print (traceback.print_tb(sys.exc_info()[2]))
                    raise Exception("processing pool not successful.")

                if not pool_result.successful():
                    print (sys.exc_info()[1])
                    print (traceback.print_tb(sys.exc_info()[2]))
                    raise Exception("processing pool not successful.")

                images = pool_result.get()
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
            
                if not reuse_segmentations:        
                    # First-pass segmentation
                    information('First-pass image segmentation...')
                    pool = Pool(num_procs)
                    image_data = pool.map(eval(use_algorithm), image_data)
                    pool.close()
                    pool.join()
                else: # if reusing segmentation data, reload the regionprops for each image
                    information('Reloading segment properties...')
                    pool = Pool(num_procs) # 4 workers is good on my MacBook Pro with a 2.52 GHz Core 2 Duo & 8 GB RAM; 8 is better on lab iMacs
                    image_data = pool.map(load_rp_data, image_data)
                    pool.close()
                    pool.join()
                    
                ## start fluorescent foci/blob/profile detection
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
                            new_ax0_size = prior_ax0_size - start_analysis_index + len(seg_images)
                            h5phsegs.resize(new_ax0_size, axis = 0)
                            h5phsegs[prior_length_ax0:] = np.array(seg_images[new_ax0_size - prior_ax0_size:])
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
                                new_ax0_size = prior_ax0_size - start_analysis_index + len(blobsegs)
                                h5blobsegs.resize(new_ax0_size, axis = 0)
                                h5blobsegs[prior_length_ax0:] = np.array(blobsegs[new_ax0_size - prior_ax0_size:])
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
                
                # drop images that are in the blurriest arbitrary n% of images
                # in a channel; could be improved to get an "absolute" meaning
                # of "blurry" - maybe as a percent of pixels in the image?
            
                # first, calculate distribution of "sharp" PC image FFT amplitudes
                blur_px_counts = []
                cutoff = 0
                if remove_blurry:
                    for i in range(len(image_data)):    
                        imfft = fft2(trim_zeros_2d(image_data[i]['source_data'][:,:,-1]))
                        logamp = np.log(np.sqrt((np.real(imfft) ** 2) + (np.imag(imfft) ** 2)))
                        shla = np.roll(logamp, logamp.shape[0]/2, axis = 0)
                        shla = np.roll(shla, shla.shape[1]/2, axis = 1)
                        bin_mask = logamp > 10.1
                        blur_px_counts.append(bin_mask.sum())
                    cutoff = np.percentile(blur_px_counts, [remove_blurry_percent,])[0]
                    information("blurry count cutoff, min, max: %d, %d, %d" % (cutoff, np.min(blur_px_counts), np.max(blur_px_counts)))
                # second, find images which:
                #     lack data or 
                #     has a blur statistic below cutoff or
                #     has no cell segments identified
                blurry_count = 0
                nosegments_count = 0
                drop_images = []
                for data_index in range(len(image_data)):
                    if image_data[data_index]['source_data'].shape == (0, 0) or isinstance(image_data[data_index]['source_data'], int):
                        warning("zero-size image at index %d." % data_index)
                        drop_images.append(data_index)
                    elif remove_blurry and blur_px_counts[data_index] < cutoff:
                        blurry_count += 1
                        warning('blurry image at index %d.' % data_index)
                        drop_images.append(data_index)
                    elif len(np.unique(image_data[data_index]['phase_segments'])) == 1:
                        nosegments_count += 1
                        warning('no cell segments at index %d.' % data_index)
                        drop_images.append(data_index)
                information("dropping blurry/no-data/no-segments images (%d, %d, %d): %s" % (blurry_count, len(drop_images) - blurry_count - nosegments_count, nosegments_count, str(drop_images)))
                # third, reverse the list to avoid screwing up the data order while deleting and then delete
                drop_images.reverse()
                for d_index in drop_images:
                    image_data.pop(d_index)
                
                # map each cell segment to it's daughter segment(s)
                information("mapping cell lineage by multiproc...")
                pool = Pool(num_procs)
                image_data = pool.map(map_segments, range(len(image_data)))
                pool.close()
                pool.join()
                
                '''
                # re-segment any images with whose prior suggested one or more errors
                resegment_indexes = [ni+1 for ni, i in enumerate(image_data) if i['flag_next']]
                # if there are any 2 resegmentation suggestions within 2 
                # indexes of each other, time-dependent segmentation won't work
                if np.any(np.diff(resegment_indexes) < 2):
                    raise ValueError('runs of multiple resegmentation flags found.')
                '''
                
                # get a mother cell chain if there is no prior; otherwise, 
                # map new cells into the prior
                if prior_flat_cells is None:
                    # Trace the mother cell & it's daughters
                    information('Tracking mother cell lineage...')
                    cell_lineage_raw = getSegmentChain(0, 1)
                    # flatten cell lives into aggregate data, still recursively linked
                    information('Flattening cell chain...')
                    flattened_cells = getFlattenedChain(cell_lineage_raw)
                    del cell_lineage_raw
                else:
                    information('Tracking new mother cell lineage...')
                    new_mother_chain = getSegmentChain(0, 1)
                    information('Flattening new mother cell chain...')
                    new_flattened_cells = getFlattenedChain(new_mother_chain)
                    # next, iterate through the prior_flat_cells data until
                    #    a daughter cell hash matches the hash of the new
                    #    mother chain
                    prior_iter = prior_flat_cells
                    while restart_iter['cell_id'] not in [d['cell_id'] for d in prior_iter['daughters']]:
                        if len(prior_iter['daughters']) == 1:
                            prior_iter = prior_iter['daughters'][0]
                        elif len(prior_iter['daughters']) == 2:
                            d0 = prior_iter['daughters'][0]
                            d1 = prior_iter['daughters'][1]
                            if d0['centroids'][0][0] > d1['centroids'][0][0]:
                                prior_iter = d1
                            else:
                                prior_iter = d0
                    # when the while loop exits, find the matching daughter and
                    #     join the new chain to it.
                    else:
                        for j in range(len(prior_iter['daughters'])):
                            if prior_iter['daughters'][j]['cell_id'] == restart_iter['cell_id']:
                                prior_iter['daughters'][j] = new_flattened_cells
                    flattened_cells = prior_flat_cells
                    del new_mother_chain
                
                if not os.path.exists(experiment_directory + analysis_directory + 'cell_trees/'):
                    os.makedirs(experiment_directory + analysis_directory + 'cell_trees/')
                with open(experiment_directory + analysis_directory + 'cell_trees/fov_%03d_peak_%04d.mshl' % (fov, original_n_peak), 'wb') as fh:
                    marshal.dump(flattened_cells, fh)
                
                # flatten mother cell data into a big structured array
                # this presumes that there is at least one cell lifetime to start with
                information('Making mother cell list...')
                mothercells = []
                motherpointer = flattened_cells
                while motherpointer['daughters'] is not None:
                    mothercells.append({k: v for k, v in motherpointer.items() if k != 'daughters'})
                    if len(motherpointer['daughters']) == 1:
                        motherpointer = motherpointer['daughters'][0]
                    elif len(motherpointer['daughters']) == 2:
                        d0 = motherpointer['daughters'][0]
                        d1 = motherpointer['daughters'][1]
                        if d0['centroids'][0][0] > d1['centroids'][0][0]:
                            motherpointer = d1
                            mothercells[-1]['sibling'] = {k: v for k, v in d0.items() if not k == 'daughters'}
                        else:
                            motherpointer = d0
                            mothercells[-1]['sibling'] = {k: v for k, v in d1.items() if not k == 'daughters'}
                    else:
                        warning('unknown daughter cell count (%d), ending mothercell list.' % len(motherpointer['daughters']))
                        break
                '''
                # convert the sibling data into individual entries in the mother cells
                information('Converting sibling data to flat records...')
                for mi in range(len(mothercells)):
                    if 'sibling' in mothercells[mi].keys():
                        mothercells[mi].update({str('sibling_' + k): v for k, v in mothercells[mi]['sibling'].items() if k != 'daughters'})
                    else:
                        mothercells[mi].update({str('sibling_' + k): None for k, v in mothercells[mi].items() if k != 'daughters'})
                        
                # convert mothercells to a NumPy structured array
                information('Converting mother cells to NumPy array...')
                mkeys = [k for k in mothercells[0].keys()]
                kformats = {}
                r = np.core.records.fromrecords([(mc[k] for k in mkeys) for mc in mothercells], 
                                                names = mkeys, formats = [kformats[k] for k in mkeys])
                # write out the mother cell array
                if not os.path.exists(experiment_directory + analysis_directory + 'mother_cells/'):
                    os.makedirs(experiment_directory + analysis_directory + 'mother_cells/')
                np.save(experiment_directory + analysis_directory + 'mother_cells/mcells_%03d_%04d.npy' % (fov, original_n_peak), r)
                '''
                raise ValueError('you have no errors yet dude.')
                    
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