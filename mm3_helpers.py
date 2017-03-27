#!/usr/bin/python
from __future__ import print_function

# import modules
import sys # input, output, errors, and files
import os # interacting with file systems
import time # getting time
import inspect # getting passed parameters
import yaml # parameter importing
import json # for importing tiff metadata
try:
    import cPickle as pickle # loading and saving python objects
except:
    import pickle
import numpy as np # numbers package
import scipy.signal as spsig # used in channel finding
from scipy.optimize import curve_fit # fitting elongation rate
import struct # for interpretting strings as binary data
import re # regular expressions
import traceback # for error messaging
import warnings # error messaging
import copy # not sure this is needed
import h5py # working with HDF5 files

# Image analysis modules
from scipy import ndimage as ndi # labeling and distance transform
from skimage import segmentation # used in make_masks and segmentation
from skimage.feature import match_template # used to align images
from skimage.filters import threshold_otsu # segmentation
from skimage import morphology # many functions is segmentation used from this
from skimage.measure import regionprops # used for creating lineages

# Parralelization modules
import multiprocessing
from multiprocessing import Pool

# Plotting for debug
import matplotlib as mpl
import matplotlib.pyplot as plt
font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 12}
mpl.rc('font', **font)
mpl.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

# supress the warning tifffile always gives
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import tifffile as tiff

### functions ###########################################################
# alert the use what is up
def warning(*objs):
    print(time.strftime("%H:%M:%S Warning:", time.localtime()), *objs, file=sys.stderr)

def information(*objs):
    print(time.strftime("%H:%M:%S", time.localtime()), *objs, file=sys.stdout)

# load the parameters file into a global dictionary for this module
def init_mm3_helpers(param_file_path):
    # load all the parameters into a global dictionary
    global params
    with open(param_file_path, 'r') as param_file:
        params = yaml.safe_load(param_file)

    # set up how to manage cores for multiprocessing
    cpu_count = multiprocessing.cpu_count()
    params['num_analyzers'] = cpu_count*2 - 2

    # useful folder shorthands for opening files
    params['TIFF_dir'] = params['experiment_directory'] + params['image_directory']
    params['ana_dir'] = params['experiment_directory'] + params['analysis_directory']
    params['hdf5_dir'] = params['ana_dir'] + 'hdf5/'
    params['chnl_dir'] = params['ana_dir'] + 'channels/'
    params['empty_dir'] = params['ana_dir'] + 'empties/'
    params['sub_dir'] = params['ana_dir'] + 'subtracted/'
    params['seg_dir'] = params['ana_dir'] + 'segmented/'
    params['cell_dir'] = params['ana_dir'] + 'cell_data/'

    return params

# loads and image stack from TIFF or HDF5 using mm3 conventions
def load_stack(fov_id, peak_id, color='c1'):
    '''
    Loads a image stack.

    Supports reading TIFF stacks or HDF5 files.

    Parameters
    ----------
    fov_id : int
        The FOV id
    peak_id : int
        The peak (channel) id. Dummy None value incase color='empty'
    color : str
        The image stack type to return. Can be:
        c1 : phase stack
        cN : where n is an integer for arbitrary color channel
        sub : subtracted images
        seg : segmented images
        empty : get the empty channel for this fov, slightly different

    Returns
    -------
    image_stack : np.ndarray
        The image stack through time. Shape is (t, y, x)
    '''

    # things are slightly different for empty channels
    if color == 'empty':
        if params['output'] == 'TIFF':
            img_filename = params['experiment_name'] + '_xy%03d_%s.tif' % (fov_id, color)

            with tiff.TiffFile(params['empty_dir'] + img_filename) as tif:
                img_stack = tif.asarray()

        if params['output'] == 'HDF5':
            with h5py.File(params['hdf5_dir'] + 'xy%03d.hdf5' % fov_id, 'r') as h5f:
                img_stack = h5f['empty_channel'][:]

        return img_stack

    # load normal images for either TIFF or HDF5
    if params['output'] == 'TIFF':
        if color[0] == 'c':
            img_dir = params['chnl_dir']
        elif color == 'sub':
            img_dir = params['sub_dir']
        elif color == 'seg':
            img_dir = params['seg_dir']

        img_filename = params['experiment_name'] + '_xy%03d_p%04d_%s.tif' % (fov_id, peak_id, color)

        with tiff.TiffFile(img_dir + img_filename) as tif:
            img_stack = tif.asarray()

    if params['output'] == 'HDF5':
        with h5py.File(params['hdf5_dir'] + 'xy%03d.hdf5' % fov_id, 'r') as h5f:
            # normal naming
            # need to use [:] to get a copy, else it references the closed hdf5 dataset
            img_stack = h5f['channel_%04d/p%04d_%s' % (peak_id, peak_id, color)][:]

    return img_stack

### Functions for dealing with raw TIFF images
# get params is the major function which processes raw TIFF images
def get_tif_params(image_filename, find_channels=True):
    '''This is a damn important function for getting the information
    out of an image. It loads a tiff file, pulls out the image data, and the metadata,
    including the location of the channels if flagged.

    it returns a dictionary like this for each image:

    'filename': image_filename,
    'fov' : image_metadata['fov'], # fov id
    't' : image_metadata['t'], # time point
    'jdn' : image_metadata['jdn'], # absolute julian time
    'x' : image_metadata['x'], # x position on stage [um]
    'y' : image_metadata['y'], # y position on stage [um]
    'plane_names' : image_metadata['plane_names'] # list of plane names
    'channels': cp_dict, # dictionary of channel locations

    Called by
    mm3_Compile.py __main__

    Calls
    mm3.extract_metadata
    mm3.find_channels
    '''

    try:
        # open up file and get metadata
        with tiff.TiffFile(params['TIFF_dir'] + image_filename) as tif:
            image_data = tif.asarray()

            if params['TIFF_source'] == 'elements':
                image_metadata = get_tif_metadata_elements(tif)
            elif params['TIFF_source'] == 'nd2ToTIFF':
                image_metadata = get_tif_metadata_nd2ToTIFF(tif)

        # look for channels if flagged
        if find_channels:
            # fix the image orientation and get the number of planes
            image_data = fix_orientation(image_data)

            # if the image data has more than 1 plane restrict image_data to phase,
            # which should have highest mean pixel data
            if len(image_data.shape) > 2:
                ph_index = np.argmax([np.mean(image_data[ci]) for ci in range(image_data.shape[0])])
                image_data = image_data[ph_index]

            # get shape of single plane
            img_shape = [image_data.shape[0], image_data.shape[1]]

            # find channels on the processed image
            chnl_loc_dict = find_channel_locs(image_data)

        information('Analyzed %s' % image_filename)

        # return the file name, the data for the channels in that image, and the metadata
        return {'filepath': params['TIFF_dir'] + image_filename,
                'fov' : image_metadata['fov'], # fov id
                't' : image_metadata['t'], # time point
                'jd' : image_metadata['jd'], # absolute julian time
                'x' : image_metadata['x'], # x position on stage [um]
                'y' : image_metadata['y'], # y position on stage [um]
                'planes' : image_metadata['planes'], # list of plane names
                'shape' : img_shape, # image shape x y in pixels
                'channels' : chnl_loc_dict} # dictionary of channel locations

    except:
        warning('Failed get_params for ' + image_filename.split("/")[-1])
        print(sys.exc_info()[0])
        print(sys.exc_info()[1])
        print(traceback.print_tb(sys.exc_info()[2]))
        return {'filepath': TIFF_dir + image_filename, 'analyze_success': False}

# finds metdata in a tiff image which has been expoted with Nikon Elements.
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
    mm3.Compile

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

# finds metdata in a tiff image which has been expoted with nd2ToTIFF.py.
def get_tif_metadata_nd2ToTIFF(tif):
    '''This function pulls out the metadata from a tif file and returns it as a dictionary.
    This if tiff files as exported by the mm3 function mm3_nd2ToTIFF.py. All the metdata
    is found in that script and saved in json format to the tiff, so it is simply extracted here

    Paramters:
        tif: TIFF file object from which data will be extracted
    Returns:
        dictionary of values:
            'fov': int,
            't' : int,
            'jdn' (float)
            'x' (float)
            'y' (float)
            'planes' (list of strings)

    Called by
    mm3_Compile.get_tif_params

    '''
    # get the first page of the tiff and pull out image description
    # this dictionary should be in the above form
    idata = tif[0].image_description
    idata = json.loads(idata.decode('utf-8'))

    return idata

# slice_and_write cuts up the image files one at a time and writes them out to tiff stacks
def tiff_stack_slice_and_write(images_to_write, channel_masks, analyzed_imgs):
    '''Writes out 4D stacks of TIFF images per channel.
    Loads all tiffs from and FOV into memory and then slices all time points at once.

    Called by
    __main__
    '''

    # make an array of images and then concatenate them into one big stack
    image_fov_stack = []

    # go through list of images and get the file path
    for n, image in enumerate(images_to_write):
        # analyzed_imgs dictionary will be found in main scope. [0] is the key, [1] is jd
        image_params = analyzed_imgs[image[0]]

        information("Loading %s." % image_params['filepath'].split('/')[-1])

        if n == 1:
            # declare identification variables for saving using first image
            fov_id = image_params['fov']

        # load the tif and store it in array
        with tiff.TiffFile(image_params['filepath']) as tif:
            image_data = tif.asarray()

        # channel finding was also done on images after orientation was fixed
        image_data = fix_orientation(image_data)

        # add additional axis if the image is flat
        if len(image_data.shape) == 2:
            image_data = np.expand_dims(image_data, 0)

        #change axis so it goes X, Y, Plane
        image_data = np.rollaxis(image_data, 0, 3)

        # add it to list. The images should be in time order
        image_fov_stack.append(image_data)

    # concatenate the list into one big ass stack
    image_fov_stack = np.stack(image_fov_stack, axis=0)

    # cut out the channels as per channel masks for this fov
    for peak, channel_loc in channel_masks[fov_id].iteritems():
        #information('Slicing and saving channel peak %s.' % channel_filename.split('/')[-1])
        information('Slicing and saving channel peak %d.' % peak)

        # slice out channel.
        # The function should recognize the shape length as 4 and cut all time points
        channel_stack = cut_slice(image_fov_stack, channel_loc)

        # save a different time stack for all colors
        for color_index in range(channel_stack.shape[3]):
            # this is the filename for the channel
            # # chnl_dir and p will be looked for in the scope above (__main__)
            channel_filename = params['chnl_dir'] + params['experiment_name'] + '_xy%03d_p%04d_c%1d.tif' % (fov_id, peak, color_index+1)
            # save stack
            tiff.imsave(channel_filename, channel_stack[:,:,:,color_index], compress=4)

    return

# same thing but do it for hdf5
def hdf5_stack_slice_and_write(images_to_write, channel_masks, analyzed_imgs):
    '''Writes out 4D stacks of TIFF images to an HDF5 file.

    Called by
    __main__
    '''

    # make an array of images and then concatenate them into one big stack
    image_fov_stack = []

    # make arrays for filenames and times
    image_filenames = []
    image_times = [] # times is still an integer but may be indexed arbitrarily
    image_jds = [] # jds = julian dates (times)

    # go through list of images, load and fix them, and create arrays of metadata
    for n, image in enumerate(images_to_write):
        image_name = image[0] # [0] is the key, [1] is jd

        # analyzed_imgs dictionary will be found in main scope.
        image_params = analyzed_imgs[image_name]
        information("Loading %s." % image_params['filepath'].split('/')[-1])

        # add information to metadata arrays
        image_filenames.append(image_name)
        image_times.append(image_params['t'])
        image_jds.append(image_params['jd'])

        # declare identification variables for saving using first image
        if n == 1:
            # same across fov
            fov_id = image_params['fov']
            x_loc = image_params['x']
            y_loc = image_params['y']
            image_shape = image_params['shape']
            image_planes = image_params['planes']

        # load the tif and store it in array
        with tiff.TiffFile(image_params['filepath']) as tif:
            image_data = tif.asarray()

        # channel finding was also done on images after orientation was fixed
        image_data = fix_orientation(image_data)

        # add additional axis if the image is flat
        if len(image_data.shape) == 2:
            image_data = np.expand_dims(image_data, 0)

        #change axis so it goes X, Y, Plane
        image_data = np.rollaxis(image_data, 0, 3)

        # add it to list. The images should be in time order
        image_fov_stack.append(image_data)

    # concatenate the list into one big ass stack
    image_fov_stack = np.stack(image_fov_stack, axis=0)

    # create the HDF5 file for the FOV, first time this is being done.
    with h5py.File(params['hdf5_dir'] + 'xy%03d.hdf5' % fov_id, 'w', libver='earliest') as h5f:

        # add in metadata for this FOV
        # these attributes should be common for all channel
        h5f.attrs.create('fov_id', fov_id)
        h5f.attrs.create('stage_x_loc', x_loc)
        h5f.attrs.create('stage_y_loc', y_loc)
        h5f.attrs.create('image_shape', image_shape)
        # encoding is because HDF5 has problems with numpy unicode
        h5f.attrs.create('planes', [plane.encode('utf8') for plane in image_planes])
        h5f.attrs.create('peaks', sorted(channel_masks[fov_id].keys()))

        # this is for things that change across time, for these create a dataset
        h5ds = h5f.create_dataset(u'filenames', data=np.expand_dims(image_filenames, 1),
                                  chunks=True, maxshape=(None, 1), dtype='S100',
                                  compression="gzip", shuffle=True, fletcher32=True)
        h5ds = h5f.create_dataset(u'times', data=np.expand_dims(image_times, 1),
                                  chunks=True, maxshape=(None, 1),
                                  compression="gzip", shuffle=True, fletcher32=True)
        h5ds = h5f.create_dataset(u'times_jd', data=np.expand_dims(image_jds, 1),
                                  chunks=True, maxshape=(None, 1),
                                  compression="gzip", shuffle=True, fletcher32=True)

        # cut out the channels as per channel masks for this fov
        for peak, channel_loc in channel_masks[fov_id].iteritems():
            #information('Slicing and saving channel peak %s.' % channel_filename.split('/')[-1])
            information('Slicing and saving channel peak %d.' % peak)

            # create group for this channel
            h5g = h5f.create_group('channel_%04d' % peak)

            # add attribute for peak_id, channel location
            h5g.attrs.create('peak_id', peak)
            h5g.attrs.create('channel_loc', channel_loc)

            # slice out channel.
            # The function should recognize the shape length as 4 and cut all time points
            channel_stack = cut_slice(image_fov_stack, channel_loc)

            # save a different dataset  for all colors
            for color_index in range(channel_stack.shape[3]):

                # create the dataset for the image. Review docs for these options.
                h5ds = h5g.create_dataset(u'p%04d_c%1d' % (peak, color_index+1),
                                data=channel_stack[:,:,:,color_index],
                                chunks=(1, channel_stack.shape[1], channel_stack.shape[2]),
                                maxshape=(None, channel_stack.shape[1], channel_stack.shape[2]),
                                compression="gzip", shuffle=True, fletcher32=True)

                h5ds.attrs.create('plane', image_planes[color_index].encode('utf8'))

                # write the data even though we have more to write (free up memory)
                h5f.flush()

    return

# finds the location of channels in a tif
def find_channel_locs(image_data):
    '''Finds the location of channels from a phase contrast image. The channels are returned in
    a dictionary where the key is the x position of the channel in pixel and the value is a
    dicionary with the open and closed end in pixels in y.


    Called by
    mm3_Compile.get_tif_params

    '''

    # declare temp variables from yaml parameter dict.
    chan_w = params['channel_width']
    chan_sep = params['channel_separation']
    crop_wp = int(params['channel_width_pad'] + params['channel_width']/2)
    chan_snr = params['channel_detection_snr']

    # Detect peaks in the x projection (i.e. find the channels)
    projection_x = image_data.sum(axis=0)
    # find_peaks_cwt is a function which attempts to find the peaks in a 1-D array by
    # convolving it with a wave. here the wave is the default wave used by the algorithm
    # but the minimum signal to noise ratio is specified
    peaks = spsig.find_peaks_cwt(projection_x, np.arange(chan_w-5,chan_w+5),
                                 min_snr=chan_snr)

    # If the left-most peak position is within half of a channel separation,
    # discard the channel from the list.
    if peaks[0] < (chan_sep / 2):
        peaks = peaks[1:]
    # If the diference between the right-most peak position and the right edge
    # of the image is less than half of a channel separation, discard the channel.
    if image_data.shape[1] - peaks[-1] < (chan_sep / 2):
        peaks = peaks[:-1]

    # Find the average channel ends for the y-projected image
    projection_y = image_data.sum(axis=1)
    # find derivative, must use int32 because it was unsigned 16b before.
    proj_y_d = np.diff(projection_y.astype(np.int32))
    # use the top third to look for closed end, is pixel location of highest deriv
    onethirdpoint_y = int(projection_y.shape[0]/3.0)
    default_closed_end_px = proj_y_d[:onethirdpoint_y].argmax()
    # use bottom third to look for open end, pixel location of lowest deriv
    twothirdpoint_y = int(projection_y.shape[0]*2.0/3.0)
    default_open_end_px = twothirdpoint_y + proj_y_d[twothirdpoint_y:].argmin()
    default_length = default_open_end_px - default_closed_end_px # used for checks

    # go through peaks and assign information
    # dict for channel dimensions
    chnl_loc_dict = {}
    # key is peak location, value is dict with {'closed_end_px': px, 'open_end_px': px}

    for peak in peaks:
        # set defaults
        chnl_loc_dict[peak] = {'closed_end_px': default_closed_end_px,
                                 'open_end_px': default_open_end_px}
        # redo the previous y projection finding with just this channel
        channel_slice = image_data[:, peak-crop_wp:peak+crop_wp]
        slice_projection_y = channel_slice.sum(axis = 1)
        slice_proj_y_d = np.diff(slice_projection_y.astype(np.int32))
        slice_closed_end_px = slice_proj_y_d[:onethirdpoint_y].argmax()
        slice_open_end_px = twothirdpoint_y + slice_proj_y_d[twothirdpoint_y:].argmin()
        slice_length = slice_open_end_px - slice_closed_end_px

        # check if these values make sense. If so, use them. If not, use default
        # make sure lenght is not 30 pixels bigger or smaller than default
        if slice_length + 15 < default_length or slice_length - 15 > default_length:
            continue
        # make sure ends are greater than 15 pixels from image edge
        if slice_closed_end_px < 15 or slice_open_end_px > image_data.shape[0] - 15:
            continue

        # if you made it to this point then update the entry
        chnl_loc_dict[peak] = {'closed_end_px': slice_closed_end_px,
                                 'open_end_px': slice_open_end_px}

    return chnl_loc_dict

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
    crop_wp = int(params['channel_width_pad'] + params['channel_width']/2)
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
                x1 = max(chnl_peak - crop_wp, 0)
                x2 = min(chnl_peak + crop_wp, image_cols)
                y1 = max(peak_ends['closed_end_px'] - chan_lp, 0)
                y2 = min(peak_ends['open_end_px'] + chan_lp, image_rows)

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
        consensus_mask = segmentation.clear_border(consensus_mask.T > 0.1)
        consensus_mask = ndi.label(consensus_mask)[0].T

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
            min_row = np.min(np.where(posrows)[0])
            max_row = np.max(np.where(posrows)[0])
            min_col = np.min(np.where(poscols)[0])
            max_col = np.max(np.where(poscols)[0])

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

    # pad y of channel if slice happened to be outside of image
    y_difference  = (channel_loc[0][1] - channel_loc[0][0]) - channel_slice.shape[1]
    if y_difference > 0:
        paddings = [[0, 0], # t
                    [0, y_difference], # y
                    [0, 0], # x
                    [0, 0]] # c
        channel_slice = np.pad(channel_slice, paddings, mode='edge')

    return channel_slice

# calculate cross correlation between pixels in channel stack
def channel_xcorr(fov_id, peak_id):
    '''
    Function calculates the cross correlation of images in a
    stack to the first image in the stack. The output is an
    array that is the length of the stack with the best cross
    correlation between that image and the first image.

    The very first value should be 1.
    '''

    # load the phase contrast images
    image_data = load_stack(fov_id, peak_id, color='c1')

    # just use the first plane just in case there are many colors (shouldn't be)
    if len(image_data.shape) > 3: # if there happen to be multiple planes
        image_data = image_data[:,:,:,0]

    # if there are more than 100 images, use 100 images evenly
    # spaced across the range
    if image_data.shape[0] > 100:
        spacing = int(image_data.shape[0] / 100)
        image_data = image_data[::spacing,:,:]
        if image_data.shape[0] > 100:
            image_data = image_data[:100,:,:]

    # we will compare all images to this one, needs to be padded to account for image drift
    first_img = np.pad(image_data[0,:,:], 10, mode='reflect')

    xcorr_array = [] # array holds cross correlation vaues
    for img in image_data:
        # use match_template to find all cross correlations for the
        # current image against the first image.
        xcorr_array.append(np.max(match_template(first_img, img)))

    return xcorr_array

### functions about subtraction
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

        # load the one phase contrast as the empties
        avg_empty_stack = load_stack(fov_id, peak_id, color='c1')

        # get just the phase data if it is multidimensional (it shouldn't be)
        if len(avg_empty_stack.shape) > 3:
            avg_empty_stack = avg_empty_stack[:,:,:,0]

    # but if there is more than one empty you need to align and average them per timepoint
    elif len(empty_peak_ids) > 1:
        # load the image stacks into memory
        empty_stacks = [] # list which holds phase image stacks of designated empties
        for peak_id in empty_peak_ids:
            # load data and append to list
            image_data = load_stack(fov_id, peak_id, color='c1')

            # just get phase data and put it in list
            if len(image_data.shape) > 3:
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
            avg_empty_stack.append(avg_empty)

        # concatenate list and then save out to tiff stack
        avg_empty_stack = np.stack(avg_empty_stack, axis=0)

    # save out data
    if params['output'] == 'TIFF':
        # make new name and save it
        empty_filename = params['experiment_name'] + '_xy%03d_empty.tif' % fov_id
        tiff.imsave(params['empty_dir'] + empty_filename, avg_empty_stack, compress=4)

    if params['output'] == 'HDF5':
        h5f = h5py.File(params['hdf5_dir'] + 'xy%03d.hdf5' % fov_id, 'r+')

        # delete the dataset if it exists (important for debug)
        if 'empty_channel' in h5f:
            del h5f[u'empty_channel']

        # the empty channel should be it's own dataset
        h5ds = h5f.create_dataset(u'empty_channel',
                        data=avg_empty_stack,
                        chunks=(1, avg_empty_stack.shape[1], avg_empty_stack.shape[2]),
                        maxshape=(None, avg_empty_stack.shape[1], avg_empty_stack.shape[2]),
                        compression="gzip", shuffle=True, fletcher32=True)

        # give attribute which says which channels contribute
        h5ds.attrs.create('empty_channels', empty_peak_ids)
        h5f.close()

    information("Saved empty channel for FOV %d." % fov_id)

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
            ref_img = np.pad(img, pad_size, mode='reflect') # padded reference image
            aligned_imgs.append(ref_img)

        # otherwise align this image to the first padded image
        else:
            # find correlation between a convolution of img against the padded reference
            match_result = match_template(ref_img, img)

            # find index of highest correlation (relative to top left corner of img)
            y, x = np.unravel_index(np.argmax(match_result), match_result.shape)

            # pad img so it aligns and is the same size as reference image
            pad_img = np.pad(img, ((y, ref_img.shape[0] - (y + img.shape[0])),
                                   (x, ref_img.shape[1] - (x + img.shape[1]))), mode='reflect')
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

# Do subtraction for an fov over many timepoints
def subtract_fov_stack(fov_id, specs):
    '''
    For a given FOV, loads the precomputed empty stack and does subtraction on
    all peaks in the FOV designated to be analyzed


    Called by
    mm3_Subtract.py

    Calls
    mm3.subtract_phase

    '''

    information('Subtracting peaks for FOV %d.' % fov_id)

    # load empty stack feed dummy peak number to get empty
    avg_empty_stack = load_stack(fov_id, 0, color='empty')

    # determine which peaks are to be analyzed
    ana_peak_ids = []
    for peak_id, spec in specs[fov_id].items():
        if spec == 1: # 0 means it should be used for empty, -1 is ignore
            ana_peak_ids.append(peak_id)
    ana_peak_ids = sorted(ana_peak_ids) # sort for repeatability
    information("Subtracting %d channels for FOV %d." % (len(ana_peak_ids), fov_id))

    # load images for the peak and get phase images
    for peak_id in ana_peak_ids:
        information('Subtracting peak %d.' % peak_id)

        image_data = load_stack(fov_id, peak_id, color='c1')

        # it should just be the phase data, but check just to make sure.
        if len(image_data.shape) > 3:
            image_data = image_data[:,:,:,0] # just get phase data and put it in list

        # make a list for all time points to send to a multiprocessing pool
        # list will length of image_data with tuples (image, empty)
        subtract_pairs = zip(image_data, avg_empty_stack)

        # set up multiprocessing pool to do subtraction. Should wait until finished
        pool = Pool(processes=params['num_analyzers'])

        subtracted_imgs = pool.map(subtract_phase, subtract_pairs, chunksize=10)

        pool.close() # tells the process nothing more will be added.
        pool.join() # blocks script until everything has been processed and workers exit

        # stack them up along a time axis
        subtracted_stack = np.stack(subtracted_imgs, axis=0)

        # save out the subtracted stack
        if params['output'] == 'TIFF':
            sub_filename = params['experiment_name'] + '_xy%03d_p%04d_sub.tif' % (fov_id, peak_id)
            tiff.imsave(params['sub_dir'] + sub_filename, subtracted_stack, compress=4) # save it

        if params['output'] == 'HDF5':
            h5f = h5py.File(params['hdf5_dir'] + 'xy%03d.hdf5' % fov_id, 'r+')

            # put subtracted channel in correct group
            h5g = h5f['channel_%04d' % peak_id]

            # delete the dataset if it exists (important for debug)
            if 'p%04d_sub' % peak_id in h5g:
                del h5g['p%04d_sub' % peak_id]

            h5ds = h5g.create_dataset(u'p%04d_sub' % peak_id,
                            data=subtracted_stack,
                            chunks=(1, subtracted_stack.shape[1], subtracted_stack.shape[2]),
                            maxshape=(None, subtracted_stack.shape[1], subtracted_stack.shape[2]),
                            compression="gzip", shuffle=True, fletcher32=True)

        information("Saved subtracted channel %d." % peak_id)

    if params['output'] == 'HDF5':
        h5f.close()

    return True

# subtracts one image from another.
def subtract_phase(image_pair):
    '''subtract_phase aligns and subtracts a .
    Modified from subtract_phase_only by jt on 20160511
    The subtracted image returned is the same size as the image given. It may however include
    data points around the edge that are meaningless but not marked.

    We align the empty channel to the phase channel, then subtract.

    Parameters
    image_pair : tuple of length two with; (image, empty_mean)

    Returns
    (subtracted_image, offset) : tuple with the subtracted_image as well as the ammount it
        was shifted to be aligned with the empty. offset = (x, y), negative or positive
        px values.

    Called by
    subtract_fov_stack
    '''
    # get out data and pad
    cropped_channel, empty_channel = image_pair # [channel slice, empty slice]

    ### Pad cropped channel.
    pad_size = 10 # pixel size to use for padding (ammount that alignment could be off)
    padded_chnl = np.pad(cropped_channel, pad_size, mode='reflect')

    # ### Align channel to empty using match template.
    # use match template to get a correlation array and find the position of maximum overlap
    match_result = match_template(padded_chnl, empty_channel)
    # get row and colum of max correlation value in correlation array
    y, x = np.unravel_index(np.argmax(match_result), match_result.shape)

    # pad the empty channel according to alignment to be overlayed on padded channel.
    empty_paddings = [[y, padded_chnl.shape[0] - (y + empty_channel.shape[0])],
                      [x, padded_chnl.shape[1] - (x + empty_channel.shape[1])]]
    aligned_empty = np.pad(empty_channel, empty_paddings, mode='reflect')
    # now trim it off so it is the same size as the original channel
    aligned_empty = aligned_empty[pad_size:-1*pad_size, pad_size:-1*pad_size]

    ### Compute the difference between the empty and channel phase contrast images
    # subtract cropped cell image from empty channel.
    channel_subtracted = aligned_empty.astype('int32') - cropped_channel.astype('int32')

    # just zero out anything less than 0. This is what Sattar does
    channel_subtracted[channel_subtracted < 0] = 0
    channel_subtracted = channel_subtracted.astype('uint16') # change back to 16bit

    return channel_subtracted

### functions that deal with segmentation and lineages
# Do segmentation for an channel time stack
def segment_chnl_stack(fov_id, peak_id):
    '''
    For a given fov and peak (channel), do segmentation for all images in the
    subtracted .tif stack.

    Called by
    mm3_Segment.py

    Calls
    mm3.segment_image
    '''

    information('Segmenting FOV %d, channel %d.' % (fov_id, peak_id))

    # load subtracted images
    sub_stack = load_stack(fov_id, peak_id, color='sub')

    # set up multiprocessing pool to do segmentation. Will do everything before going on.
    pool = Pool(processes=params['num_analyzers'])

    # send the 3d array to multiprocessing
    segmented_imgs = pool.map(segment_image, sub_stack, chunksize=8)

    pool.close() # tells the process nothing more will be added.
    pool.join() # blocks script until everything has been processed and workers exit

    # # image by image for debug
    # segmented_imgs = []
    # for sub_image in sub_stack:
    #     segmented_imgs.append(segment_image(sub_image))

    # stack them up along a time axis
    segmented_imgs = np.stack(segmented_imgs, axis=0)
    segmented_imgs = segmented_imgs.astype('uint16')

    # save out the subtracted stack
    if params['output'] == 'TIFF':
        seg_filename = params['experiment_name'] + '_xy%03d_p%04d_seg.tif' % (fov_id, peak_id)
        tiff.imsave(params['seg_dir'] + seg_filename,
                    segmented_imgs.astype('uint16'), compress=4)

    if params['output'] == 'HDF5':
        h5f = h5py.File(params['hdf5_dir'] + 'xy%03d.hdf5' % fov_id, 'r+')

        # put segmented channel in correct group
        h5g = h5f['channel_%04d' % peak_id]

        # delete the dataset if it exists (important for debug)
        if 'p%04d_seg' % peak_id in h5g:
            del h5g['p%04d_seg' % peak_id]

        h5ds = h5g.create_dataset(u'p%04d_seg' % peak_id,
                        data=segmented_imgs,
                        chunks=(1, segmented_imgs.shape[1], segmented_imgs.shape[2]),
                        maxshape=(None, segmented_imgs.shape[1], segmented_imgs.shape[2]),
                        compression="gzip", shuffle=True, fletcher32=True)
        h5f.close()

    information("Saved segmented channel %d." % peak_id)

    return True

# segmentation algorithm
def segment_image(image):
    '''Segments a subtracted image and returns a labeled image

    Parameters
    image : a ndarray which is an image. This should be the subtracted image

    Returns
    labeled_image : a ndarray which is also an image. Labeled values, which
        should correspond to cells, all have the same integer value starting with 1.
        Non labeled area should have value zero.
    '''

    # load in segmentation parameters
    first_opening_size = params['first_opening_size']
    distance_threshold = params['distance_threshold']
    second_opening_size = params['second_opening_size']
    min_object_size = params['min_object_size']

    # threshold image
    thresh = threshold_otsu(image) # finds optimal OTSU thershhold value
    threshholded = image > thresh # will create binary image

    # if there are no cells, good to clear the border
    # because otherwise the OTSU is just for random bullshit, most
    # likely on the side of the image
    threshholded = segmentation.clear_border(threshholded)

    # Opening = erosion then dialation.
    # opening smooths images, breaks isthmuses, and eliminates protrusions.
    # "opens" dark gaps between bright features.
    morph = morphology.binary_opening(threshholded, morphology.disk(first_opening_size))

    # if this image is empty at this point (likely if there were no cells), just return
    # the morphed image which is a zero array
    if np.amax(morph) == 0:
        return morph

    # zero out rows that have very few pixels
    # widens or creates gaps between cells
    # sum of rows (how many pixels are occupied in each row)
    line_profile = np.sum(morph, axis=1)
    # find highest value, aka width of fattest cell
    max_width = max(line_profile)
    # find indexes of rows where sum is less than 1/5th of this value.
    zero_these_indicies = np.all([line_profile < (max_width/4), line_profile > 0], axis=0)
    zero_these_indicies = np.where(zero_these_indicies)
    # zero out those rows
    morph[zero_these_indicies] = 0

    ### Calculate distnace matrix, use as markers for random walker (diffusion watershed)
    # Generate the markers based on distance to the background
    distance = ndi.distance_transform_edt(morph)

    # threshold distance image
    distance_thresh = np.zeros_like(distance)
    distance_thresh[distance < distance_threshold] = 0
    distance_thresh[distance >= distance_threshold] = 1

    # do an extra opening on the distance
    distance_opened = morphology.binary_opening(distance_thresh,
                                                morphology.disk(second_opening_size))

    # remove artifacts connected to image border
    cleared = segmentation.clear_border(distance_opened)
    # remove small objects. Remove small objects wants a
    # labeled image and will fail if there is only one label. Return zero image in that case
    # could have used try/except but remove_small_objects loves to issue warnings.
    cleared, label_num = morphology.label(cleared, connectivity=1, return_num=True)
    if label_num > 1:
        cleared = morphology.remove_small_objects(cleared, min_size=min_object_size)
    else:
        # if there are no labels, then just return the cleared image as it is zero
        return cleared # should be zero image if there was only 1 label

    # relabel now that small objects and labels on edges have been cleared
    markers = morphology.label(cleared)

    # label using the random walker (diffusion watershed) algorithm
    try:
        # set anything outside of OTSU threshold to -1 so it will not be labeled
        markers[threshholded == 0] = -1
        # here is the main algorithm
        labeled_image = segmentation.random_walker(-1*image, markers)
        # put negative values back to zero for proper image
        labeled_image[labeled_image == -1] = 0
    except:
        return cleared # this should just be a zero array

    return labeled_image

# finds lineages for all peaks in a fov
def make_lineages_fov(fov_id, specs):
    '''
    For a given fov, create the lineages from the segmented images.

    Called by
    mm3_Segment.py

    Calls
    mm3.make_lineage_chnl_stack
    '''
    ana_peak_ids = [] # channels to be analyzed
    for peak_id, spec in specs[fov_id].items():
        if spec == 1: # 1 means analyze
            ana_peak_ids.append(peak_id)
    ana_peak_ids = sorted(ana_peak_ids) # sort for repeatability

    information('Creating lineage for FOV %d with %d channels.' % (fov_id, len(ana_peak_ids)))

    # This is a list of tuples (fov_id, peak_id) to send to the Pool command
    fov_and_peak_ids_list = [(fov_id, peak_id) for peak_id in ana_peak_ids]

    # set up multiprocessing pool. will complete pool before going on
    pool = Pool(processes=params['num_analyzers'])

    # create the lineages for each peak individually
    # the output is a list of dictionaries
    lineages = pool.map(make_lineage_chnl_stack, fov_and_peak_ids_list, chunksize=8)

    pool.close() # tells the process nothing more will be added.
    pool.join() # blocks script until everything has been processed and workers exit

    # # looped version for debugging
    # lineages = []
    # for fov_and_peak_id in fov_and_peak_ids_list:
    #     lineages.append(make_lineage_chnl_stack(fov_and_peak_id))

    # combine all dictionaries into one dictionary
    Cells = {} # create dictionary to hold all information
    for cell_dict in lineages: # for all the other dictionaries in the list
        Cells.update(cell_dict) # updates Cells with the entries in cell_dict

    return Cells

# Creates lineage for a single channel
def make_lineage_chnl_stack(fov_and_peak_id):
    '''
    Create the lineage for a set of segmented images for one channel. Start by making the regions in the first time points potenial cells. Go forward in time and map regions in the timepoint to the potential cells in previous time points, building the life of a cell. Used basic checks such as the regions should overlap, and grow by a little and not shrink too much. If regions do not link back in time, discard them. If two regions map to one previous region, check if it is a sensible division event.

    Parameters
    ----------
    fov_and_peak_ids : tuple.
        (fov_id, peak_id)

    Returns
    -------
    Cells : dict
        A dictionary of all the cells from this lineage, divided and undivided

    '''

    # load in parameters
    # if leaf regions see no action for longer than this, drop them
    lost_cell_time = params['lost_cell_time']
    # only cells with y positions below this value will recieve the honor of becoming new
    # cells, unless they are daughters of current cells
    new_cell_y_cutoff = params['new_cell_y_cutoff']

    # get the specific ids from the tuple
    fov_id, peak_id = fov_and_peak_id

    information('Creating lineage for FOV %d, channel %d.' % (fov_id, peak_id))

    # load segmented data
    image_data_seg = load_stack(fov_id, peak_id, color='seg')

    # Calculate all data for all time points.
    # this list will be length of the number of time points
    regions_by_time = [regionprops(timepoint) for timepoint in image_data_seg]

    # Set up data structures.
    Cells = {} # Dict that holds all the cell objects, divided and undivided
    cell_leaves = [] # cell ids of the current leaves of the growing lineage tree

    # go through regions by timepoint and build lineages
    for t, regions in enumerate(regions_by_time):
        # if there are cell leaves where too much time has passed
        # since they have grown or divided, remove them from the running
        for leaf_id in cell_leaves:
            if t - Cells[leaf_id].times[-1] > lost_cell_time:
                cell_leaves.remove(leaf_id)

        # make all the regions leaves if there are no current leaves
        if not cell_leaves:
            for region in regions:
                if region.centroid[0] < new_cell_y_cutoff:
                    # Create cell and put in cell dictionary
                    cell_id = create_cell_id(region, t, peak_id, fov_id)
                    Cells[cell_id] = Cell(cell_id, region, t, parent_id=None)

                    # add thes id to list of current leaves
                    cell_leaves.append(cell_id)

        # Determine if the regions are children of current leaves
        else:
            ### create mapping between regions and leaves
            leaf_region_map = {}
            leaf_region_map = {leaf_id : [] for leaf_id in cell_leaves}

            # get the last y position of current leaves and create tuple with the id
            current_leaf_positions = [(leaf_id, Cells[leaf_id].y_positions[-1]) for leaf_id in cell_leaves]

            # go through regions, they will come off in Y position order
            for r, region in enumerate(regions):
                # create tuple which is cell_id of closest leaf, distance
                current_closest = (None, 1000) # 1000 is just a large number

                # check this region against all positions of all current leaf regions,
                # find the closest one in y.
                for leaf in current_leaf_positions:
                    # calculate distance between region and leaf
                    y_dist_region_to_leaf = abs(region.centroid[0] - leaf[1])

                    # if the distance is closer than before, update
                    if y_dist_region_to_leaf < current_closest[1]:
                        current_closest = (leaf[0], y_dist_region_to_leaf)

                # update map with the closest region
                leaf_region_map[current_closest[0]].append((r, y_dist_region_to_leaf))

            # go through the current leaf regions.
            # limit by the closest two current regions if there are three regions to the leaf
            for leaf_id, region_links in leaf_region_map.iteritems():
                if len(region_links) > 2:
                    closest_two_regions = sorted(region_links, key=lambda x: x[1])[:2]
                    # but sort by region order so top region is first
                    closest_two_regions = sorted(closest_two_regions, key=lambda x: x[0])
                    # replace value in dictionary
                    leaf_region_map[leaf_id] = closest_two_regions

                    # for the discarded regions, put them as new leaves
                    # if they are near the closed end of the channel
                    discarded_regions = sorted(region_links, key=lambda x: x[1])[2:]
                    for discarded_region in discarded_regions:
                        region = regions[discarded_region[0]]
                        if region.centroid[0] < new_cell_y_cutoff:
                            cell_id = create_cell_id(region, t, peak_id, fov_id)
                            Cells[cell_id] = Cell(cell_id, region, t, parent_id=None)
                            cell_leaves.append(cell_id) # add to leaves

            ### iterate over the leaves, looking to see what regions connect to them.
            for leaf_id, region_links in leaf_region_map.iteritems():

                # if there is just one suggested descendant,
                # see if it checks out and append the data
                if len(region_links) == 1:
                    region = regions[region_links[0][0]] # grab the region from the list

                    # check if the pairing makes sense based on size and position
                    # this function returns true if things are okay
                    if check_growth_by_region(Cells[leaf_id], region):
                        # grow the cell by the region in this case
                        Cells[leaf_id].grow(region, t)

                # there may be two daughters, or maybe there is just one child and a new cell
                elif len(region_links) == 2:
                    # grab these two daughters
                    region1 = regions[region_links[0][0]]
                    region2 = regions[region_links[1][0]]

                    # check_division returns 3 if cell divided,
                    # 1 if first region is just the cell growing and the second is trash
                    # 2 if the second region is the cell, and the first is trash
                    # or 0 if it cannot be determined.
                    check_division_result = check_division(Cells[leaf_id], region1, region2)
                    if check_division_result == 3:
                        # create two new cells and divide the mother
                        daughter1_id = create_cell_id(region1, t, peak_id, fov_id)
                        daughter2_id = create_cell_id(region2, t, peak_id, fov_id)
                        Cells[daughter1_id] = Cell(daughter1_id, region1, t,
                                                   parent_id=leaf_id)
                        Cells[daughter2_id] = Cell(daughter2_id, region2, t,
                                                   parent_id=leaf_id)
                        Cells[leaf_id].divide(Cells[daughter1_id], Cells[daughter2_id], t)

                        # add the daughter ids to list of current leaves, remove mother
                        cell_leaves.append(daughter1_id)
                        cell_leaves.append(daughter2_id)
                        cell_leaves.remove(leaf_id)

                    # 1 means that daughter 1 is just a continuation of the mother
                    elif check_division_result == 1:
                        Cells[leaf_id].grow(region1, t)

                    # ditto for 2
                    elif check_division_result == 2:
                        Cells[leaf_id].grow(region2, t)

    # Also save an image of the lineages superimposed on the segmented images
    if params['print_lineages']:
        information('Creating lineage image.')

        # load subtracted data
        image_data_sub = load_stack(fov_id, peak_id, color='sub')

        n_imgs = len(regions_by_time)
        image_indicies = range(n_imgs)

        # Color map for good label colors
        cmap = plt.cm.jet
        cmap.set_under(color='black')
        vmin = 0.1 # values under this color go to black
        vmax = image_data_seg.shape[1] # max y value
        # Trying to get the image size down
        figxsize = image_data_seg.shape[2] * n_imgs / 100.0
        figysize = image_data_seg.shape[1] / 100.0

        # plot the images in a series
        fig, axes = plt.subplots(ncols=n_imgs, nrows=1,
                                 figsize=(figxsize, figysize))
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
            ax[i].imshow(image_data_sub[i], cmap=plt.cm.gray, aspect='equal')

            # make a new version of the segmented image where the
            # regions are relabeled by their y centroid position.
            seg_relabeled = image_data_seg[i].copy()
            for region in regions_by_time[i]:
                seg_relabeled[seg_relabeled == region.label] = region.centroid[0]

            ax[i].imshow(seg_relabeled, cmap=cmap, alpha=0.5, vmin=vmin, vmax=vmax)
            ax[i].set_title(str(i), color='white')

        # save just the segmented images
        # lin_dir = params['experiment_directory'] + params['analysis_directory'] + 'lineages/'
        # if not os.path.exists(lin_dir):
        #     os.makedirs(lin_dir)
        # lin_filename = params['experiment_name'] + '_xy%03d_p%04d_nolin.png' % (fov_id, peak_id)
        # lin_filepath = lin_dir + lin_filename
        # fig.savefig(lin_filepath, dpi=75)
        # plt.close()

        # Annotate each cell with information
        for cell_id in Cells:
            for n, t in enumerate(Cells[cell_id].times):
                x = Cells[cell_id].x_positions[n]
                y = Cells[cell_id].y_positions[n]

                # add a circle at the centroid for every point in this cell's life
                circle = mpatches.Circle(xy=(x, y), radius=3, color='white', lw=0, alpha=0.5)
                ax[t].add_patch(circle)

                # draw connecting lines between the centroids of cells in same lineage
                if n < len(Cells[cell_id].times)-1:
                    # coordinates of the next centroid
                    x_next = Cells[cell_id].x_positions[n+1]
                    y_next = Cells[cell_id].y_positions[n+1]
                    t_next = Cells[cell_id].times[n+1]

                    # get coordinates for the whole figure
                    coord1 = transFigure.transform(ax[t].transData.transform([x, y]))
                    coord2 = transFigure.transform(ax[t_next].transData.transform([x_next, y_next]))

                    # create line
                    line = mpl.lines.Line2D((coord1[0],coord2[0]),(coord1[1],coord2[1]),
                                            transform=fig.transFigure,
                                            color='white', lw=2, alpha=0.3)

                    # add it to plot
                    fig.lines.append(line)

                # draw connecting between mother and daughters
                if n == len(Cells[cell_id].times)-1 and Cells[cell_id].daughters:
                    # daughter ids
                    d1_id = Cells[cell_id].daughters[0]
                    d2_id = Cells[cell_id].daughters[1]

                    # both daughters should have been born at the same time.
                    t_next = Cells[d1_id].times[0]

                    # coordinates of the two daughters
                    x_d1 = Cells[d1_id].x_positions[0]
                    y_d1 = Cells[d1_id].y_positions[0]
                    x_d2 = Cells[d2_id].x_positions[0]
                    y_d2 = Cells[d2_id].y_positions[0]

                    # get coordinates for the whole figure
                    coord1 = transFigure.transform(ax[t].transData.transform([x, y]))
                    coordd1 = transFigure.transform(ax[t_next].transData.transform([x_d1, y_d1]))
                    coordd2 = transFigure.transform(ax[t_next].transData.transform([x_d2, y_d2]))

                    # create line and add it to plot for both
                    for coord in [coordd1, coordd2]:
                        line = mpl.lines.Line2D((coord1[0],coord[0]),(coord1[1],coord[1]),
                                                transform=fig.transFigure,
                                                color='white', lw=2, alpha=0.3, ls='dashed')
                        # add it to plot
                        fig.lines.append(line)

        #         # this is for putting cell id on first time cell appears and when it divides
        #         if n == 0 or n == len(Cells[cell_id].times)-1:
        #             ax[t].text(x, y, cell_id, color='red', size=10, ha='center', va='center')

            # save image to segmentation subfolder
            lin_dir = params['experiment_directory'] + params['analysis_directory'] + 'lineages/'
            if not os.path.exists(lin_dir):
                os.makedirs(lin_dir)
            lin_filename = params['experiment_name'] + '_xy%03d_p%04d_lin.png' % (fov_id, peak_id)
            lin_filepath = lin_dir + lin_filename
            fig.savefig(lin_filepath, dpi=75)
            plt.close()

    # return the dictionary with all the cells
    return Cells

# Cell class and related functions
class Cell():
    '''
    The Cell class is one cell that has been born. It is not neccesarily a cell that
    has divided.
    '''

    # initialize (birth) the cell
    def __init__(self, cell_id, region, t, parent_id=None):
        '''The cell must be given a unique cell_id and passed the region
        information from the segmentation

        Parameters
        __________

        cell_id : str
            cell_id is a string in the form fXpXtXrX
            f is 3 digit FOV number
            p is 4 digit peak number
            t is 4 digit time point at time of birth
            r is region label for that segmentation
            Use the function create_cell_id to do return a proper string.

        region : region properties object
            Information about the labeled region from
            skimage.measure.regionprops()

        parent_id : str
            id of the parent if there is one.
            '''

        # create all the attributes
        # id
        self.id = cell_id

        # identification convenience
        self.fov = int(cell_id.split('f')[1].split('p')[0])
        self.peak = int(cell_id.split('p')[1].split('t')[0])
        self.birth_label = int(cell_id.split('r')[1])

        # parent id may be none
        self.parent = parent_id

        # daughters is updated when cell divides
        # if this is none then the cell did not divide
        self.daughters = None

        # birth and division time
        self.birth_time = t
        self.division_time = None # filled out if cell divides

        # the following information is on a per timepoint basis
        self.times = [t]
        self.labels = [region.label]
        self.bboxes = [region.bbox]
        self.areas = [region.area]
        self.x_positions = [region.centroid[1]]
        self.y_positions = [region.centroid[0]]
        self.lengths = [region.major_axis_length]
        self.widths = [region.minor_axis_length]

        # these two are special, as they include information from the daugthers for division
        # computed upon division
        self.times_w_div = None
        self.lengths_w_div = None

        # this information is the "production" information that
        # we want to extract at the end. Some of this is for convenience.
        # This is only filled out if a cell divides.
        self.sb = None # in um
        self.sd = None # this should be combined lengths of daughters, in um
        self.delta = None
        self.tau = None
        self.sum_cov = None
        self.septum_position = None

    def grow(self, region, t):
        '''Append data from a region to this cell.
        use cell.times[-1] to get most current value'''

        self.times.append(t)
        self.labels.append(region.label)
        self.bboxes.append(region.bbox)
        self.areas.append(region.area)
        self.x_positions.append(region.centroid[1])
        self.y_positions.append(region.centroid[0])
        self.lengths.append(region.major_axis_length)
        self.widths.append(region.minor_axis_length)

    def divide(self, daughter1, daughter2, t):
        '''Divide the cell and update stats.
        daugther1 and daugther2 are instances of the Cell class.
        daughter1 is the daugther closer to the closed end.'''

        # put the daugther ids into the cell
        self.daughters = [daughter1.id, daughter2.id]

        # give this guy a division time
        self.division_time = daughter1.birth_time

        # flesh out the stats for this cell
        # size at birth
        self.sb = self.lengths[0] * params['pxl2um']

        # force the division length to be the combined lengths of the daughters
        self.sd = (daughter1.lengths[0] + daughter2.lengths[0]) * params['pxl2um']

        # delta is here for convinience
        self.delta = self.sd - self.sb

        # generation time
        self.tau = self.division_time - self.birth_time

        # growth rate (inst. elong rate alpha) sd = sb * 2 ^ (gr * tau)
        # include the data points from the daughters
        self.times_w_div = np.append(self.times, self.division_time)
        self.lengths_w_div = np.append(self.lengths, daughter1.lengths[0] + daughter2.lengths[0]) * params['pxl2um']
        try:
            with warnings.catch_warnings(): # ignore the warnings if it can't converge
                warnings.simplefilter("ignore")
                popt, pcov = curve_fit(cell_growth_func, self.times_w_div - self.birth_time,
                                       np.log(self.lengths_w_div),
                                       p0=(np.log(self.sb), np.log(2)/self.tau))
                elong_rate = popt[1] # 0 is the guessed sb, 1 is the guessed elong_rate
        except:
            elong_rate = float('NaN')
            pcov = float('Nan')

        self.elong_rate = elong_rate
        self.sum_cov = np.sum(pcov)

        # calculate the septum position as a number between 0 and 1
        # which indicates the size of daughter closer to the closed end
        # compared to the total size
        self.septum_position = daughter1.lengths[0] / (daughter1.lengths[0] + daughter2.lengths[0])

    def print_info(self):
        '''prints information about the cell'''
        print('id = %s' % self.id)
        print('times = {}'.format(', '.join('{}'.format(t) for t in self.times)))
        print('lengths = {}'.format(', '.join('{:.2f}'.format(l) for l in self.lengths)))

# take info and make string for cell id
def create_cell_id(region, t, peak, fov):
    '''Make a unique cell id string for a new cell'''
    cell_id = ['f', str(fov), 'p', str(peak), 't', str(t), 'r', str(region.label)]
    cell_id = ''.join(cell_id)
    return cell_id

# function for a growing cell, used to calculate growth rate
def cell_growth_func(t, sb, elong_rate):
    '''
    Assumes you have taken log of the data.
    It also allows the size at birth to be a free parameter, rather than fixed
    at the actual size at birth (but still uses that as a guess)
    Assumes natural log, not base 2 (though I think that makes less sense)

    old form: sb*2**(alpha*t)
    '''
    return sb+elong_rate*t

# functions for checking if a cell has divided or not
# this function should also take the variable t to
# weight the allowed changes by the difference in time as well
def check_growth_by_region(cell, region):
    '''Checks to see if it makes sense
    to grow a cell by a particular region'''
    # load parameters for checking
    max_growth_length = params['max_growth_length']
    min_growth_length = params['min_growth_length']
    max_growth_area = params['max_growth_area']
    min_growth_area = params['min_growth_area']

    # check if length is not too much longer
    if cell.lengths[-1]*max_growth_length < region.major_axis_length:
        return False

    # check if it is not too short (cell should not shrink really)
    if cell.lengths[-1]*min_growth_length > region.major_axis_length:
        return False

    # check if area is not too great
    if cell.areas[-1]*max_growth_area < region.area:
        return False

    # check if area is not too small
    if cell.lengths[-1]*min_growth_area > region.area:
        return False

    # check if y position of region is within
    # the quarter positions of the bounding box
    lower_quarter = cell.bboxes[-1][0] + (region.major_axis_length / 4)
    upper_quarter = cell.bboxes[-1][2] - (region.major_axis_length / 4)
    if lower_quarter > region.centroid[0] or upper_quarter < region.centroid[0]:
        return False

    # return true if you get this far
    return True

# see if a cell has reasonably divided
def check_division(cell, region1, region2):
    '''Checks to see if it makes sense to divide a
    cell into two new cells based on two regions.

    Return 0 if nothing should happend and regions ignored
    Return 1 if cell should grow by region 1
    Return 2 if cell should grow by region 2
    Return 3 if cell should divide into the regions.'''

    # load in parameters
    max_growth_length = params['max_growth_length']
    min_growth_length = params['min_growth_length']

    # see if either region just could be continued growth,
    # if that is the case then just return
    # these shouldn't return true if the cells are divided
    # as they would be too small
    if check_growth_by_region(cell, region1):
        return 1

    if check_growth_by_region(cell, region2):
        return 2

    # make sure combined size of daughters is not too big
    combined_size = region1.major_axis_length + region2.major_axis_length
    # check if length is not too much longer
    if cell.lengths[-1]*max_growth_length < combined_size:
        return 0
    # and not too small
    if cell.lengths[-1]*min_growth_length > combined_size:
        return 0

    # centroids of regions should be in the upper and lower half of the
    # of the mother's bounding box, respectively
    # top region within top half of mother bounding box
    if cell.bboxes[-1][0] > region1.centroid[0] or cell.y_positions[-1] < region1.centroid[0]:
        return 0
    # bottom region with bottom half of mother bounding box
    if cell.y_positions[-1] > region2.centroid[0] or cell.bboxes[-1][2] < region2.centroid[0]:
        return 0

    # if you got this far then divide the mother
    return 3

### functions for pruning a dictionary of cells
# find cells with both a mother and two daughters
def find_complete_cells(Cells):
    '''Go through a dictionary of cells and return another dictionary
    that contains just those with a parent and daughters'''

    Complete_Cells = {}

    for cell_id in Cells:
        if Cells[cell_id].daughters and Cells[cell_id].parent:
            Complete_Cells[cell_id] = Cells[cell_id]

    return Complete_Cells

# finds cells whose birht label is 1
def find_mother_cells(Cells):
    '''Return only cells whose starting region label is 1.'''

    Mother_Cells = {}

    for cell_id in Cells:
        if Cells[cell_id].birth_label == 1:
            Mother_Cells[cell_id] = Cells[cell_id]

    return Mother_Cells

# return a dictionary of cells organized by fov and peak_id
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

    return Cells_by_peak

### functions for additional cell centric analysis
# finds total and average intenstiy timepoint in cells
def find_cell_intensities(fov_id, peak_id, Cells):
    '''
    Finds fluorescenct information for cells. All the cell in Cells
    should be from one fov/peak. See the function
    organize_cells_by_channel()

    '''
    # Load fluorescent images and segmented images for this channel
    fl_stack = load_stack(fov_id, peak_id, color='c1')
    seg_stack = load_stack(fov_id, peak_id, color='seg')

    # Loop through cells
    for Cell in Cells.values():
        # give this cell two lists to hold new information
        Cell.fl_tots = [] # total fluorescence per time point
        Cell.fl_avgs = [] # avg fluorescence per time point

        # and the time points that make up this cell's life
        for n, t in enumerate(Cell.times):
            # create fluorescent image only for this cell and timepoint.
            fl_image_masked = np.copy(fl_stack[t])
            fl_image_masked[seg_stack[t] != Cell.labels[n]] = 0

            # append total flourescent image
            Cell.fl_tots.append(np.sum(fl_image_masked))

            # and the average fluorescence
            Cell.fl_avgs.append(np.sum(fl_image_masked) / Cell.areas[n])

    # The cell objects in the original dictionary will be updated,
    # no need to return anything specifically.
    return

### functions about converting dates and times
### Functions
# def days_to_hmsm(days):
#     hours = days * 24.
#     hours, hour = math.modf(hours)
#     mins = hours * 60.
#     mins, min = math.modf(mins)
#     secs = mins * 60.
#     secs, sec = math.modf(secs)
#     micro = round(secs * 1.e6)
#     return int(hour), int(min), int(sec), int(micro)
#
# def hmsm_to_days(hour=0, min=0, sec=0, micro=0):
#     days = sec + (micro / 1.e6)
#     days = min + (days / 60.)
#     days = hour + (days / 60.)
#     return days / 24.
#
# def date_to_jd(year,month,day):
#     if month == 1 or month == 2:
#         yearp = year - 1
#         monthp = month + 12
#     else:
#         yearp = year
#         monthp = month
#     # this checks where we are in relation to October 15, 1582, the beginning
#     # of the Gregorian calendar.
#     if ((year < 1582) or
#         (year == 1582 and month < 10) or
#         (year == 1582 and month == 10 and day < 15)):
#         # before start of Gregorian calendar
#         B = 0
#     else:
#         # after start of Gregorian calendar
#         A = math.trunc(yearp / 100.)
#         B = 2 - A + math.trunc(A / 4.)
#     if yearp < 0:
#         C = math.trunc((365.25 * yearp) - 0.75)
#     else:
#         C = math.trunc(365.25 * yearp)
#     D = math.trunc(30.6001 * (monthp + 1))
#     jd = B + C + D + day + 1720994.5
#     return jd
#
# def datetime_to_jd(date):
#     days = date.day + hmsm_to_days(date.hour,date.minute,date.second,date.microsecond)
#     return date_to_jd(date.year, date.month, days)
