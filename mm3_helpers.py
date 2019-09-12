#! /usr/bin/env python3

from __future__ import print_function, division
import six

# import modules
import sys # input, output, errors, and files
import os # interacting with file systems
import time # getting time
import datetime
import inspect # get passed parameters
import yaml # parameter importing
import json # for importing tiff metadata
try:
    import cPickle as pickle # loading and saving python objects
except:
    import pickle
import numpy as np # numbers package
import struct # for interpretting strings as binary data
import re # regular expressions
from pprint import pprint # for human readable file output
import traceback # for error messaging
import warnings # error messaging
import copy # not sure this is needed
import h5py # working with HDF5 files
import pandas as pd
import networkx as nx
import collections

# scipy and image analysis
from scipy.signal import find_peaks_cwt # used in channel finding
from scipy.optimize import curve_fit # fitting ring profile
from scipy.optimize import leastsq # fitting 2d gaussian
from scipy import ndimage as ndi # labeling and distance transform
from skimage import io
from skimage import segmentation # used in make_masks and segmentation
from skimage.transform import rotate
from skimage.feature import match_template # used to align images
from skimage.feature import blob_log # used for foci finding
from skimage.filters import threshold_otsu, median # segmentation
from skimage import filters
from skimage import morphology # many functions is segmentation used from this
from skimage.measure import regionprops # used for creating lineages
from skimage.measure import profile_line # used for ring an nucleoid analysis
from skimage import util, measure, transform, feature
from skimage.external import tifffile as tiff
from sklearn import metrics

# deep learning
import tensorflow as tf # ignore message about how tf was compiled
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models
from tensorflow.keras import losses
from tensorflow.keras import utils
from tensorflow.keras import backend as K
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # supress warnings

# Parralelization modules
import multiprocessing
from multiprocessing import Pool

# Plotting for debug
import matplotlib as mpl
from matplotlib import pyplot as plt
font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 12}
mpl.rc('font', **font)
mpl.rcParams['pdf.fonttype'] = 42
from matplotlib.patches import Ellipse

# user modules
# realpath() will make your script run, even if you symlink it
cmd_folder = os.path.realpath(os.path.abspath(
                              os.path.split(inspect.getfile(inspect.currentframe()))[0]))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

### functions ###########################################################
# alert the use what is up

# print a warning
def warning(*objs):
    print(time.strftime("%H:%M:%S WARNING:", time.localtime()), *objs, file=sys.stderr)

# print information
def information(*objs):
    print(time.strftime("%H:%M:%S", time.localtime()), *objs, file=sys.stdout)

# load the parameters file into a global dictionary for this module
def init_mm3_helpers(param_file_path):
    # load all the parameters into a global dictionary
    global params
    with open(param_file_path, 'r') as param_file:
        params = yaml.safe_load(param_file)

    # set up how to manage cores for multiprocessing
    params['num_analyzers'] = multiprocessing.cpu_count()

    # useful folder shorthands for opening files
    params['TIFF_dir'] = os.path.join(params['experiment_directory'], params['image_directory'])
    params['ana_dir'] = os.path.join(params['experiment_directory'], params['analysis_directory'])
    params['hdf5_dir'] = os.path.join(params['ana_dir'], 'hdf5')
    params['chnl_dir'] = os.path.join(params['ana_dir'], 'channels')
    params['empty_dir'] = os.path.join(params['ana_dir'], 'empties')
    params['sub_dir'] = os.path.join(params['ana_dir'], 'subtracted')
    params['seg_dir'] = os.path.join(params['ana_dir'], 'segmented')
    params['pred_dir'] = os.path.join(params['ana_dir'], 'predictions')
    params['foci_seg_dir'] = os.path.join(params['ana_dir'], 'segmented_foci')
    params['foci_pred_dir'] = os.path.join(params['ana_dir'], 'predictions_foci')
    params['cell_dir'] = os.path.join(params['ana_dir'], 'cell_data')
    params['track_dir'] = os.path.join(params['ana_dir'], 'tracking')

    # use jd time in image metadata to make time table. Set to false if no jd time
    if params['TIFF_source'] == 'elements' or params['TIFF_source'] == 'nd2ToTIFF':
        params['use_jd'] = True
    else:
        params['use_jd'] = False

    if not 'save_predictions' in params['segment']['unet'].keys():
        params['segment']['unet']['save_predictions'] = False

    return params

def julian_day_number():
    """
    Need this to solve a bug in pims_nd2.nd2reader.ND2_Reader instance initialization.
    The bug is in /usr/local/lib/python2.7/site-packages/pims_nd2/ND2SDK.py in function `jdn_to_datetime_local`, when the year number in the metadata (self._lim_metadata_desc) is not in the correct range. This causes a problem when calling self.metadata.
    https://en.wikipedia.org/wiki/Julian_day
    """
    dt=datetime.datetime.now()
    tt=dt.timetuple()
    jdn=(1461.*(tt.tm_year + 4800. + (tt.tm_mon - 14.)/12))/4. + (367.*(tt.tm_mon - 2. - 12.*((tt.tm_mon -14.)/12)))/12. - (3.*((tt.tm_year + 4900. + (tt.tm_mon - 14.)/12.)/100.))/4. + tt.tm_mday - 32075

    return jdn

def get_plane(filepath):
    pattern = '(c\d+).tif'
    res = re.search(pattern,filepath)
    if (res != None):
        return res.group(1)
    else:
        return None

def get_fov(filepath):
    pattern = 'xy(\d+)\w*.tif'
    res = re.search(pattern,filepath)
    if (res != None):
        return int(res.group(1))
    else:
        return None

def get_time(filepath):
    pattern = 't(\d+)xy\w+.tif'
    res = re.search(pattern,filepath)
    if (res != None):
        return np.int_(res.group(1))
    else:
        return None

# loads and image stack from TIFF or HDF5 using mm3 conventions
def load_stack(fov_id, peak_id, color='c1', image_return_number=None):
    '''
    Loads an image stack.

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
    if 'empty' in color:
        if params['output'] == 'TIFF':
            img_filename = params['experiment_name'] + '_xy%03d_%s.tif' % (fov_id, color)

            with tiff.TiffFile(os.path.join(params['empty_dir'],img_filename)) as tif:
                img_stack = tif.asarray()

        if params['output'] == 'HDF5':
            with h5py.File(os.path.join(params['hdf5_dir'],'xy%03d.hdf5' % fov_id), 'r') as h5f:
                img_stack = h5f[color][:]

        return img_stack

    # load normal images for either TIFF or HDF5
    if params['output'] == 'TIFF':
        if color[0] == 'c':
            img_dir = params['chnl_dir']
        elif 'sub' in color:
            img_dir = params['sub_dir']
        elif 'foci' in color:
            img_dir = params['foci_seg_dir']
        elif 'seg' in color:
            img_dir = params['seg_dir']

        img_filename = params['experiment_name'] + '_xy%03d_p%04d_%s.tif' % (fov_id, peak_id, color)

        with tiff.TiffFile(os.path.join(img_dir, img_filename)) as tif:
            img_stack = tif.asarray()

    if params['output'] == 'HDF5':
        with h5py.File(os.path.join(params['hdf5_dir'], 'xy%03d.hdf5' % fov_id), 'r') as h5f:
            # normal naming
            # need to use [:] to get a copy, else it references the closed hdf5 dataset
            img_stack = h5f['channel_%04d/p%04d_%s' % (peak_id, peak_id, color)][:]

    return img_stack

# load the time table and add it to the global params
def load_time_table():
    '''Add the time table dictionary to the params global dictionary.
    This is so it can be used during Cell creation.
    '''

    # try first for yaml, then for pkl
    try:
        with open(os.path.join(params['ana_dir'], 'time_table.yaml'), 'rb') as time_table_file:
            params['time_table'] = yaml.safe_load(time_table_file)
    except:
        with open(os.path.join(params['ana_dir'], 'time_table.pkl'), 'rb') as time_table_file:
            params['time_table'] = pickle.load(time_table_file)

    return

# function for loading the channel masks
def load_channel_masks():
    '''Load channel masks dictionary. Should be .yaml but try pickle too.
    '''
    information("Loading channel masks dictionary.")

    # try loading from .yaml before .pkl
    try:
        information('Path:', os.path.join(params['ana_dir'], 'channel_masks.yaml'))
        with open(os.path.join(params['ana_dir'], 'channel_masks.yaml'), 'r') as cmask_file:
            channel_masks = yaml.safe_load(cmask_file)
    except:
        warning('Could not load channel masks dictionary from .yaml.')

        try:
            information('Path:', os.path.join(params['ana_dir'], 'channel_masks.pkl'))
            with open(os.path.join(params['ana_dir'], 'channel_masks.pkl'), 'rb') as cmask_file:
                channel_masks = pickle.load(cmask_file)
        except ValueError:
            warning('Could not load channel masks dictionary from .pkl.')

    return channel_masks

# function for loading the specs file
def load_specs():
    '''Load specs file which indicates which channels should be analyzed, used as empties, or ignored.'''

    try:
        with open(os.path.join(params['ana_dir'], 'specs.yaml'), 'r') as specs_file:
            specs = yaml.safe_load(specs_file)
    except:
        try:
            with open(os.path.join(params['ana_dir'], 'specs.pkl'), 'rb') as specs_file:
                specs = pickle.load(specs_file)
        except ValueError:
            warning('Could not load specs file.')

    return specs

### functions for dealing with raw TIFF images

# get params is the major function which processes raw TIFF images
def get_initial_tif_params(image_filename):
    '''This is a function for getting the information
    out of an image for later trap identification, cropping, and aligning with Unet. It loads a tiff file and pulls out the image metadata.

    it returns a dictionary like this for each image:

    'filename': image_filename,
    'fov' : image_metadata['fov'], # fov id
    't' : image_metadata['t'], # time point
    'jdn' : image_metadata['jdn'], # absolute julian time
    'x' : image_metadata['x'], # x position on stage [um]
    'y' : image_metadata['y'], # y position on stage [um]
    'plane_names' : image_metadata['plane_names'] # list of plane names

    Called by
    mm3_Compile.py __main__

    Calls
    mm3.extract_metadata
    mm3.find_channels
    '''

    try:
        # open up file and get metadata
        with tiff.TiffFile(os.path.join(params['TIFF_dir'],image_filename)) as tif:
            image_data = tif.asarray()
            #print(image_data.shape) # uncomment for debug
            img_shape = [image_data.shape[1],image_data.shape[2]]
            plane_list = [str(i+1) for i in range(image_data.shape[0])]
            #print(plane_list) # uncomment for debug

            if params['TIFF_source'] == 'elements':
                image_metadata = get_tif_metadata_elements(tif)
            elif params['TIFF_source'] == 'nd2ToTIFF':
                image_metadata = get_tif_metadata_nd2ToTIFF(tif)
            else:
                image_metadata = get_tif_metadata_filename(tif)

        information('Analyzed %s' % image_filename)

        # return the file name, the data for the channels in that image, and the metadata
        return {'filepath': os.path.join(params['TIFF_dir'], image_filename),
                'fov' : image_metadata['fov'], # fov id
                't' : image_metadata['t'], # time point
                'jd' : image_metadata['jd'], # absolute julian time
                'x' : image_metadata['x'], # x position on stage [um]
                'y' : image_metadata['y'], # y position on stage [um]
                'planes' : plane_list, # list of plane names
                'shape' : img_shape} # image shape x y in pixels

    except:
        warning('Failed get_params for ' + image_filename.split("/")[-1])
        print(sys.exc_info()[0])
        print(sys.exc_info()[1])
        print(traceback.print_tb(sys.exc_info()[2]))
        return {'filepath': os.path.join(params['TIFF_dir'],image_filename), 'analyze_success': False}

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
    'channels': cp_dict, # dictionary of channel locations, in the case of Unet-based channel segmentation, it's a dictionary of channel labels

    Called by
    mm3_Compile.py __main__

    Calls
    mm3.extract_metadata
    mm3.find_channels
    '''

    try:
        # open up file and get metadata
        with tiff.TiffFile(os.path.join(params['TIFF_dir'],image_filename)) as tif:
            image_data = tif.asarray()

            if params['TIFF_source'] == 'elements':
                image_metadata = get_tif_metadata_elements(tif)
            elif params['TIFF_source'] == 'nd2ToTIFF':
                image_metadata = get_tif_metadata_nd2ToTIFF(tif)
            else:
                image_metadata = get_tif_metadata_filename(tif)

        # look for channels if flagged
        if find_channels:
            # fix the image orientation and get the number of planes
            image_data = fix_orientation(image_data)

            # if the image data has more than 1 plane restrict image_data to phase,
            # which should have highest mean pixel data
            if len(image_data.shape) > 2:
                #ph_index = np.argmax([np.mean(image_data[ci]) for ci in range(image_data.shape[0])])
                ph_index = int(params['phase_plane'][1:]) - 1
                image_data = image_data[ph_index]

            # get shape of single plane
            img_shape = [image_data.shape[0], image_data.shape[1]]

            # find channels on the processed image
            chnl_loc_dict = find_channel_locs(image_data)

        information('Analyzed %s' % image_filename)

        # return the file name, the data for the channels in that image, and the metadata
        return {'filepath': os.path.join(params['TIFF_dir'], image_filename),
                'fov' : image_metadata['fov'], # fov id
                't' : image_metadata['t'], # time point
                'jd' : image_metadata['jd'], # absolute julian time
                'x' : image_metadata['x'], # x position on stage [um]
                'y' : image_metadata['y'], # y position on stage [um]
                'planes' : image_metadata['planes'], # list of plane names
                'shape' : img_shape, # image shape x y in pixels
                # 'channels' : {1 : {'A' : 1, 'B' : 2}, 2 : {'C' : 3, 'D' : 4}}}
                'channels' : chnl_loc_dict} # dictionary of channel locations

    except:
        warning('Failed get_params for ' + image_filename.split("/")[-1])
        print(sys.exc_info()[0])
        print(sys.exc_info()[1])
        print(traceback.print_tb(sys.exc_info()[2]))
        return {'filepath': os.path.join(params['TIFF_dir'],image_filename), 'analyze_success': False}

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

# Finds metadata from the filename
def get_tif_metadata_filename(tif):
    '''This function pulls out the metadata from a tif file and returns it as a dictionary.
    This just gets the tiff metadata from the filename and is a backup option when the known format of the metadata is not known.

    Paramters:
        tif: TIFF file object from which data will be extracted
    Returns:
        dictionary of values:
            'fov': int,
            't' : int,
            'jdn' (float)
            'x' (float)
            'y' (float)

    Called by
    mm3_Compile.get_tif_params

    '''
    idata = {'fov' : get_fov(tif.filename), # fov id
             't' : get_time(tif.filename), # time point
             'jd' : -1 * 0.0, # absolute julian time
             'x' : -1 * 0.0, # x position on stage [um]
             'y' : -1 * 0.0} # y position on stage [um]

    return idata

# make a lookup time table for converting nominal time to elapsed time in seconds
def make_time_table(analyzed_imgs):
    '''
    Loops through the analyzed images and uses the jd time in the metadata to find the elapsed
    time in seconds that each picture was taken. This is later used for more accurate elongation
    rate calculation.

    Parametrs
    ---------
    analyzed_imgs : dict
        The output of get_tif_params.
    params['use_jd'] : boolean
        If set to True, 'jd' time will be used from the image metadata to use to create time table. Otherwise the 't' index will be used, and the parameter 'seconds_per_time_index' will be used from the parameters.yaml file to convert to seconds.

    Returns
    -------
    time_table : dict
        Look up dictionary with keys for the FOV and then the time point.
    '''
    information('Making time table...')

    # initialize
    time_table = {}

    first_time = float('inf')

    # need to go through the data once to find the first time
    for iname, idata in six.iteritems(analyzed_imgs):
        if params['use_jd']:
            if idata['jd'] < first_time:
                first_time = idata['jd']
        else:
            if idata['t'] < first_time:
                first_time = idata['t']

        # init dictionary for specific times per FOV
        if idata['fov'] not in time_table:
            time_table[idata['fov']] = {}

    for iname, idata in six.iteritems(analyzed_imgs):
        if params['use_jd']:
            # convert jd time to elapsed time in seconds
            t_in_seconds = np.around((idata['jd'] - first_time) * 24*60*60, decimals=0).astype('uint32')
        else:
            t_in_seconds = np.around((idata['t'] - first_time) * params['moviemaker']['seconds_per_time_index'], decimals=0).astype('uint32')

        time_table[int(idata['fov'])][int(idata['t'])] = int(t_in_seconds)

    # save to .pkl. This pkl will be loaded into the params
    # with open(os.path.join(params['ana_dir'], 'time_table.pkl'), 'wb') as time_table_file:
    #     pickle.dump(time_table, time_table_file, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(os.path.join(params['ana_dir'], 'time_table.txt'), 'w') as time_table_file:
    #     pprint(time_table, stream=time_table_file)
    with open(os.path.join(params['ana_dir'], 'time_table.yaml'), 'w') as time_table_file:
        yaml.dump(data=time_table, stream=time_table_file, default_flow_style=False, tags=None)
    information('Time table saved.')

    return time_table

# saves traps sliced via Unet
def save_tiffs(imgDict, analyzed_imgs, fov_id):

    savePath = os.path.join(params['experiment_directory'],
                            params['analysis_directory'],
                            params['chnl_dir'])
    img_names = [key for key in analyzed_imgs.keys()]
    image_params = analyzed_imgs[img_names[0]]

    for peak,img in six.iteritems(imgDict):

        img = img.astype('uint16')
        if not os.path.isdir(savePath):
            os.mkdir(savePath)

        for planeNumber in image_params['planes']:

            channel_filename = os.path.join(savePath, params['experiment_name'] + '_xy{0:0=3}_p{1:0=4}_c{2}.tif'.format(fov_id, peak, planeNumber))
            io.imsave(channel_filename, img[:,:,:,int(planeNumber)-1])

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

        # change axis so it goes Y, X, Plane
        image_data = np.rollaxis(image_data, 0, 3)

        # add it to list. The images should be in time order
        image_fov_stack.append(image_data)

    # concatenate the list into one big ass stack
    image_fov_stack = np.stack(image_fov_stack, axis=0)

    # cut out the channels as per channel masks for this fov
    for peak, channel_loc in six.iteritems(channel_masks[fov_id]):
        #information('Slicing and saving channel peak %s.' % channel_filename.split('/')[-1])
        information('Slicing and saving channel peak %d.' % peak)

        # channel masks should only contain ints, but you can use this for hard fix
        # for i in range(len(channel_loc)):
        #     for j in range(len(channel_loc[i])):
        #         channel_loc[i][j] = int(channel_loc[i][j])

        # slice out channel.
        # The function should recognize the shape length as 4 and cut all time points
        channel_stack = cut_slice(image_fov_stack, channel_loc)

        # save a different time stack for all colors
        for color_index in range(channel_stack.shape[3]):
            # this is the filename for the channel
            # # chnl_dir and p will be looked for in the scope above (__main__)
            channel_filename = os.path.join(params['chnl_dir'], params['experiment_name'] + '_xy%03d_p%04d_c%1d.tif' % (fov_id, peak, color_index+1))
            # save stack
            tiff.imsave(channel_filename, channel_stack[:,:,:,color_index], compress=4)

    return

# saves traps sliced via Unet to an hdf5 file
def save_hdf5(imgDict, img_names, analyzed_imgs, fov_id, channel_masks):
    '''Writes out 4D stacks of images to an HDF5 file.

    Called by
    mm3_Compile.py
    '''

    savePath = params['hdf5_dir']

    if not os.path.isdir(savePath):
        os.mkdir(savePath)

    img_times = [analyzed_imgs[key]['t'] for key in img_names]
    img_jds = [analyzed_imgs[key]['jd'] for key in img_names]
    fov_ids = [analyzed_imgs[key]['fov'] for key in img_names]

    # get image_params from first image from current fov
    image_params = analyzed_imgs[img_names[0]]

    # establish some variables for hdf5 attributes
    fov_id = image_params['fov']
    x_loc = image_params['x']
    y_loc = image_params['y']
    image_shape = image_params['shape']
    image_planes = image_params['planes']

    fov_channel_masks = channel_masks[fov_id]

    with h5py.File(os.path.join(savePath,'{}_xy{:0=2}.hdf5'.format(params['experiment_name'],fov_id)), 'w', libver='earliest') as h5f:

        # add in metadata for this FOV
        # these attributes should be common for all channel
        h5f.attrs.create('fov_id', fov_id)
        h5f.attrs.create('stage_x_loc', x_loc)
        h5f.attrs.create('stage_y_loc', y_loc)
        h5f.attrs.create('image_shape', image_shape)
        # encoding is because HDF5 has problems with numpy unicode
        h5f.attrs.create('planes', [plane.encode('utf8') for plane in image_planes])
        h5f.attrs.create('peaks', sorted([key for key in imgDict.keys()]))

        # this is for things that change across time, for these create a dataset
        img_names = np.asarray(img_names)
        img_names = np.expand_dims(img_names, 1)
        img_names = img_names.astype('S100')
        h5ds = h5f.create_dataset(u'filenames', data=img_names,
                                  chunks=True, maxshape=(None, 1), dtype='S100',
                                  compression="gzip", shuffle=True, fletcher32=True)
        h5ds = h5f.create_dataset(u'times', data=np.expand_dims(img_times, 1),
                                  chunks=True, maxshape=(None, 1),
                                  compression="gzip", shuffle=True, fletcher32=True)
        h5ds = h5f.create_dataset(u'times_jd', data=np.expand_dims(img_jds, 1),
                                  chunks=True, maxshape=(None, 1),
                                  compression="gzip", shuffle=True, fletcher32=True)

        # cut out the channels as per channel masks for this fov
        for peak,channel_stack in six.iteritems(imgDict):

            channel_stack = channel_stack.astype('uint16')
            # create group for this trap
            h5g = h5f.create_group('channel_%04d' % peak)

            # add attribute for peak_id, channel location
            # add attribute for peak_id, channel location
            h5g.attrs.create('peak_id', peak)
            channel_loc = fov_channel_masks[peak]
            h5g.attrs.create('channel_loc', channel_loc)

            # save a different dataset for all colors
            for color_index in range(channel_stack.shape[3]):

                # create the dataset for the image. Review docs for these options.
                h5ds = h5g.create_dataset(u'p%04d_c%1d' % (peak, color_index+1),
                                data=channel_stack[:,:,:,color_index],
                                chunks=(1, channel_stack.shape[1], channel_stack.shape[2]),
                                maxshape=(None, channel_stack.shape[1], channel_stack.shape[2]),
                                compression="gzip", shuffle=True, fletcher32=True)

                # h5ds.attrs.create('plane', image_planes[color_index].encode('utf8'))

                # write the data even though we have more to write (free up memory)
                h5f.flush()

    return

# same thing as tiff_stack_slice_and_write but do it for hdf5
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
    with h5py.File(os.path.join(params['hdf5_dir'],'xy%03d.hdf5' % fov_id), 'w', libver='earliest') as h5f:

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
        for peak, channel_loc in six.iteritems(channel_masks[fov_id]):
            #information('Slicing and saving channel peak %s.' % channel_filename.split('/')[-1])
            information('Slicing and saving channel peak %d.' % peak)

            # create group for this channel
            h5g = h5f.create_group('channel_%04d' % peak)

            # add attribute for peak_id, channel location
            h5g.attrs.create('peak_id', peak)
            h5g.attrs.create('channel_loc', channel_loc)

            # channel masks should only contain ints, but you can use this for a hard fix
            # for i in range(len(channel_loc)):
            #     for j in range(len(channel_loc[i])):
            #         channel_loc[i][j] = int(channel_loc[i][j])

            # slice out channel.
            # The function should recognize the shape length as 4 and cut all time points
            channel_stack = cut_slice(image_fov_stack, channel_loc)

            # save a different dataset for all colors
            for color_index in range(channel_stack.shape[3]):

                # create the dataset for the image. Review docs for these options.
                h5ds = h5g.create_dataset(u'p%04d_c%1d' % (peak, color_index+1),
                                data=channel_stack[:,:,:,color_index],
                                chunks=(1, channel_stack.shape[1], channel_stack.shape[2]),
                                maxshape=(None, channel_stack.shape[1], channel_stack.shape[2]),
                                compression="gzip", shuffle=True, fletcher32=True)

                # h5ds.attrs.create('plane', image_planes[color_index].encode('utf8'))

                # write the data even though we have more to write (free up memory)
                h5f.flush()

    return

def tileImage(img, subImageNumber):
    divisor = int(np.sqrt(subImageNumber))
    M = img.shape[0]//divisor
    N = img.shape[0]//divisor
    #print(img.shape, M, N, divisor, subImageNumber)
    tiles = np.asarray([img[x:x+M,y:y+N] for x in range(0,img.shape[0],M) for y in range(0,img.shape[1],N)])
    return(tiles)

def get_weights(img, subImageNumber):
    divisor = int(np.sqrt(subImageNumber))
    M = img.shape[0]//divisor
    N = img.shape[0]//divisor
    weights = np.ones((img.shape[0],img.shape[1]),dtype='uint8')
    for i in range(divisor-1):
        weights[(M*(i+1))-25:(M*(i+1)+25),:] = 0
        weights[:,(N*(i+1))-25:(N*(i+1)+25)] = 0
    return(weights)

def permute_image(img, trap_align_metadata):
    # are there three dimensions?
    if len(img.shape) == 3:
        if img.shape[0] < 3: # for tifs with fewer than three imageing channels, the first dimension separates channels
            # img = np.transpose(img, (1,2,0))
            img = img[trap_align_metadata['phase_plane_index'],:,:] # grab just the phase channel
        else:
            img = img[:,:,trap_align_metadata['phase_plane_index']] # grab just the phase channel

    return(img)

def imageConcatenatorFeatures(imgStack, subImageNumber = 64):

    rowNumPerImage = int(np.sqrt(subImageNumber)) # here I'm assuming our large images are square, with equal number of crops in each dimension
    #print(rowNumPerImage)
    imageNum = int(imgStack.shape[0]/subImageNumber) # total number of sub-images divided by the number of sub-images in each original large image
    iterNum = int(imageNum*rowNumPerImage)
    imageDims = int(np.sqrt(imgStack.shape[1]*imgStack.shape[2]*subImageNumber))
    featureNum = int(imgStack.shape[3])
    bigImg = np.zeros(shape=(imageNum, imageDims, imageDims, featureNum), dtype='float32') # create array to store reconstructed images

    featureRowDicts = []

    for j in range(featureNum):

        rowDict = {}

        for i in range(iterNum):
            baseNum = int(i*iterNum/imageNum)
            # concatenate columns of 256x256 images to build each 256x2048 row
            rowDict[i] = np.column_stack((imgStack[baseNum,:,:,j],imgStack[baseNum+1,:,:,j],
                                          imgStack[baseNum+2,:,:,j], imgStack[baseNum+3,:,:,j]))#,
                                          #imgStack[baseNum+4,:,:,j],imgStack[baseNum+5,:,:,j],
                                          #imgStack[baseNum+6,:,:,j],imgStack[baseNum+7,:,:,j]))
        featureRowDicts.append(rowDict)

    for j in range(featureNum):

        for i in range(imageNum):
            baseNum = int(i*rowNumPerImage)
            # concatenate appropriate 256x2048 rows to build a 2048x2048 image and place it into bigImg
            bigImg[i,:,:,j] = np.row_stack((featureRowDicts[j][baseNum],featureRowDicts[j][baseNum+1],
                                            featureRowDicts[j][baseNum+2],featureRowDicts[j][baseNum+3]))#,
                                            #featureRowDicts[j][baseNum+4],featureRowDicts[j][baseNum+5],
                                            #featureRowDicts[j][baseNum+6],featureRowDicts[j][baseNum+7]))

    return(bigImg)

def imageConcatenatorFeatures2(imgStack, subImageNumber = 81):

    rowNumPerImage = int(np.sqrt(subImageNumber)) # here I'm assuming our large images are square, with equal number of crops in each dimension
    imageNum = int(imgStack.shape[0]/subImageNumber) # total number of sub-images divided by the number of sub-images in each original large image
    iterNum = int(imageNum*rowNumPerImage)
    imageDims = int(np.sqrt(imgStack.shape[1]*imgStack.shape[2]*subImageNumber))
    featureNum = int(imgStack.shape[3])
    bigImg = np.zeros(shape=(imageNum, imageDims, imageDims, featureNum), dtype='float32') # create array to store reconstructed images

    featureRowDicts = []

    for j in range(featureNum):

        rowDict = {}

        for i in range(iterNum):
            baseNum = int(i*iterNum/imageNum)
            # concatenate columns of 256x256 images to build each 256x2048 row
            rowDict[i] = np.column_stack((imgStack[baseNum,:,:,j],imgStack[baseNum+1,:,:,j],
                                          imgStack[baseNum+2,:,:,j], imgStack[baseNum+3,:,:,j],
                                          imgStack[baseNum+4,:,:,j]))#,imgStack[baseNum+5,:,:,j],
                                          #imgStack[baseNum+6,:,:,j],imgStack[baseNum+7,:,:,j],
                                         #imgStack[baseNum+8,:,:,j]))
        featureRowDicts.append(rowDict)

    for j in range(featureNum):

        for i in range(imageNum):
            baseNum = int(i*rowNumPerImage)
            # concatenate appropriate 256x2048 rows to build a 2048x2048 image and place it into bigImg
            bigImg[i,:,:,j] = np.row_stack((featureRowDicts[j][baseNum],featureRowDicts[j][baseNum+1],
                                            featureRowDicts[j][baseNum+2],featureRowDicts[j][baseNum+3],
                                            featureRowDicts[j][baseNum+4]))#,featureRowDicts[j][baseNum+5],
                                            #featureRowDicts[j][baseNum+6],featureRowDicts[j][baseNum+7],
                                            #featureRowDicts[j][baseNum+8]))

    return(bigImg)

def get_weights_array(arr=np.zeros((2048,2048)), shiftDistance=128, subImageNumber=64, padSubImageNumber=81):

    originalImageWeights = get_weights(arr, subImageNumber=subImageNumber)
    shiftLeftWeights = np.pad(originalImageWeights, pad_width=((0,0),(0,shiftDistance)),
                      mode='constant', constant_values=((0,0),(0,0)))[:,shiftDistance:]
    shiftRightWeights = np.pad(originalImageWeights, pad_width=((0,0),(shiftDistance,0)),
                      mode='constant', constant_values=((0,0),(0,0)))[:,:(-1*shiftDistance)]
    shiftUpWeights = np.pad(originalImageWeights, pad_width=((0,shiftDistance),(0,0)),
                      mode='constant', constant_values=((0,0),(0,0)))[shiftDistance:,:]
    shiftDownWeights = np.pad(originalImageWeights, pad_width=((shiftDistance,0),(0,0)),
                      mode='constant', constant_values=((0,0),(0,0)))[:(-1*shiftDistance),:]
    expandedImageWeights = get_weights(np.zeros((arr.shape[0]+2*shiftDistance,arr.shape[1]+2*shiftDistance)), subImageNumber=padSubImageNumber)[shiftDistance:-shiftDistance,shiftDistance:-shiftDistance]

    allWeights = np.stack((originalImageWeights, expandedImageWeights, shiftUpWeights, shiftDownWeights, shiftLeftWeights,shiftRightWeights), axis=-1)
    stackWeights = np.stack((allWeights,allWeights),axis=0)
    stackWeights = np.stack((stackWeights,stackWeights,stackWeights),axis=3)
    return(stackWeights)

# predicts locations of channels in an image using deep learning model
def get_frame_predictions(img,model,stackWeights, shiftDistance=256, subImageNumber=16, padSubImageNumber=25, debug=False):

    pred = predict_first_image_channels(img, model, shiftDistance=shiftDistance,
                                     subImageNumber=subImageNumber, padSubImageNumber=padSubImageNumber, debug=debug)[0,...]
    # print(pred.shape)
    if debug:
        print(pred.shape)


    compositePrediction = np.average(pred, axis=3, weights=stackWeights)
    # print(compositePrediction.shape)

    padSize = (compositePrediction.shape[0]-img.shape[0])//2
    compositePrediction = util.crop(compositePrediction,((padSize,padSize),
                                                        (padSize,padSize),
                                                        (0,0)))
    # print(compositePrediction.shape)

    return(compositePrediction)

def apply_median_filter_normalize(imgs):

    selem = morphology.disk(3)

    for i in range(imgs.shape[0]):
        # Store sample
        tmpImg = imgs[i,:,:,0]
        medImg = median(tmpImg, selem)
        tmpImg = medImg/np.max(medImg)
        tmpImg = np.expand_dims(tmpImg, axis=-1)
        imgs[i,:,:,:] = tmpImg

    return(imgs)


def predict_first_image_channels(img, model,
                              subImageNumber=16, padSubImageNumber=25,
                              shiftDistance=128, batchSize=1,
                              debug=False):
    imgSize = img.shape[0]
    padSize = (2048-imgSize)//2 # how much to pad on each side to get up to 2048x2048?
    imgStack = np.pad(img, pad_width=((padSize,padSize),(padSize,padSize)),
                      mode='constant', constant_values=((0,0),(0,0))) # pad the images to make them 2048x2048
    # pad the stack by 128 pixels on each side to get complemetary crops that I can run the network on. This
    #    should help me fill in low-confidence regions where the crop boundaries were for the original image
    imgStackExpand = np.pad(imgStack, pad_width=((shiftDistance,shiftDistance),(shiftDistance,shiftDistance)),
                            mode='constant', constant_values=((0,0),(0,0)))
    imgStackShiftRight = np.pad(imgStack, pad_width=((0,0),(0,shiftDistance)),
                                mode='constant', constant_values=((0,0),(0,0)))[:,shiftDistance:]
    imgStackShiftLeft = np.pad(imgStack, pad_width=((0,0),(shiftDistance,0)),
                                mode='constant', constant_values=((0,0),(0,0)))[:,:-shiftDistance]
    imgStackShiftDown = np.pad(imgStack, pad_width=((0,shiftDistance),(0,0)),
                               mode='constant', constant_values=((0,0),(0,0)))[shiftDistance:,:]
    imgStackShiftUp = np.pad(imgStack, pad_width=((shiftDistance,0),(0,0)),
                               mode='constant', constant_values=((0,0),(0,0)))[:-shiftDistance,:]
    #print(imgStackShiftUp.shape)

    crops = tileImage(imgStack, subImageNumber=subImageNumber)
    crops = np.expand_dims(crops, -1)

    data_gen_args = {'batch_size':params['compile']['channel_prediction_batch_size'],
                         'n_channels':1,
                         'normalize_to_one':True,
                         'shuffle':False}
    predict_gen_args = {'verbose':1,
                        'use_multiprocessing':True,
                        'workers':params['num_analyzers']}

    img_generator = TrapSegmentationDataGenerator(crops, **data_gen_args)
    predictions = model.predict_generator(img_generator, **predict_gen_args)
    prediction = imageConcatenatorFeatures(predictions, subImageNumber=subImageNumber)
    #print(prediction.shape)

    cropsExpand = tileImage(imgStackExpand, subImageNumber=padSubImageNumber)
    cropsExpand = np.expand_dims(cropsExpand, -1)
    img_generator = TrapSegmentationDataGenerator(cropsExpand, **data_gen_args)
    predictions = model.predict_generator(img_generator, **predict_gen_args)
    predictionExpand = imageConcatenatorFeatures2(predictions, subImageNumber=padSubImageNumber)
    predictionExpand = util.crop(predictionExpand, ((0,0),(shiftDistance,shiftDistance),(shiftDistance,shiftDistance),(0,0)))
    #print(predictionExpand.shape)

    cropsShiftLeft = tileImage(imgStackShiftLeft, subImageNumber=subImageNumber)
    cropsShiftLeft = np.expand_dims(cropsShiftLeft, -1)
    img_generator = TrapSegmentationDataGenerator(cropsShiftLeft, **data_gen_args)
    predictions = model.predict_generator(img_generator, **predict_gen_args)
    predictionLeft = imageConcatenatorFeatures(predictions, subImageNumber=subImageNumber)
    predictionLeft = np.pad(predictionLeft, pad_width=((0,0),(0,0),(0,shiftDistance),(0,0)),
                      mode='constant', constant_values=((0,0),(0,0),(0,0),(0,0)))[:,:,shiftDistance:,:]
    #print(predictionLeft.shape)

    cropsShiftRight = tileImage(imgStackShiftRight, subImageNumber=subImageNumber)
    cropsShiftRight = np.expand_dims(cropsShiftRight, -1)
    img_generator = TrapSegmentationDataGenerator(cropsShiftRight, **data_gen_args)
    predictions = model.predict_generator(img_generator, **predict_gen_args)
    predictionRight = imageConcatenatorFeatures(predictions, subImageNumber=subImageNumber)
    predictionRight = np.pad(predictionRight, pad_width=((0,0),(0,0),(shiftDistance,0),(0,0)),
                      mode='constant', constant_values=((0,0),(0,0),(0,0),(0,0)))[:,:,:(-1*shiftDistance),:]
    #print(predictionRight.shape)

    cropsShiftUp = tileImage(imgStackShiftUp, subImageNumber=subImageNumber)
    #print(cropsShiftUp.shape)
    cropsShiftUp = np.expand_dims(cropsShiftUp, -1)
    img_generator = TrapSegmentationDataGenerator(cropsShiftUp, **data_gen_args)
    predictions = model.predict_generator(img_generator, **predict_gen_args)
    predictionUp = imageConcatenatorFeatures(predictions, subImageNumber=subImageNumber)
    predictionUp = np.pad(predictionUp, pad_width=((0,0),(0,shiftDistance),(0,0),(0,0)),
                      mode='constant', constant_values=((0,0),(0,0),(0,0),(0,0)))[:,shiftDistance:,:,:]
    #print(predictionUp.shape)

    cropsShiftDown = tileImage(imgStackShiftDown, subImageNumber=subImageNumber)
    cropsShiftDown = np.expand_dims(cropsShiftDown, -1)
    img_generator = TrapSegmentationDataGenerator(cropsShiftDown, **data_gen_args)
    predictions = model.predict_generator(img_generator, **predict_gen_args)
    predictionDown = imageConcatenatorFeatures(predictions, subImageNumber=subImageNumber)
    predictionDown = np.pad(predictionDown, pad_width=((0,0),(shiftDistance,0),(0,0),(0,0)),
                      mode='constant', constant_values=((0,0),(0,0),(0,0),(0,0)))[:,:(-1*shiftDistance),:,:]
    #print(predictionDown.shape)

    allPredictions = np.stack((prediction, predictionExpand,
                               predictionUp, predictionDown,
                               predictionLeft, predictionRight), axis=-1)

    return(allPredictions)

# takes initial U-net centroids for trap locations, and creats bounding boxes for each trap at the defined height and width
def get_frame_trap_bounding_boxes(trapLabels, trapProps, trapAreaThreshold=2000, trapWidth=27, trapHeight=256):

    badTrapLabels = [reg.label for reg in trapProps if reg.area < trapAreaThreshold] # filter out small "trap" regions
    goodTraps = trapLabels.copy()

    for label in badTrapLabels:
        goodTraps[goodTraps == label] = 0 # re-label bad traps as background (0)

    goodTrapProps = measure.regionprops(goodTraps)
    trapCentroids = [(int(np.round(reg.centroid[0])),int(np.round(reg.centroid[1]))) for reg in goodTrapProps] # get centroids as integers
    trapBboxes = []

    for centroid in trapCentroids:
        rowIndex = centroid[0]
        colIndex = centroid[1]

        minRow = rowIndex-trapHeight//2
        maxRow = rowIndex+trapHeight//2
        minCol = colIndex-trapWidth//2
        maxCol = colIndex+trapWidth//2
        if trapWidth % 2 != 0:
            maxCol += 1

        coordArray = np.array([minRow,maxRow,minCol,maxCol])

        # remove any traps at edges of image
        if np.any(coordArray > goodTraps.shape[0]):
            continue
        if np.any(coordArray < 0):
            continue

        trapBboxes.append((minRow,minCol,maxRow,maxCol))

    return(trapBboxes)

# this function performs image alignment as defined by the shifts passed as an argument
def crop_traps(fileNames, trapProps, labelledTraps, bboxesDict, trap_align_metadata):

    frameNum = trap_align_metadata['frame_count']
    channelNum = trap_align_metadata['plane_number']
    trapImagesDict = {key:np.zeros((frameNum,
                                       trap_align_metadata['trap_height'],
                                       trap_align_metadata['trap_width'],
                                       channelNum)) for key in bboxesDict}
    trapClosedEndPxDict = {}
    flipImageDict = {}
    trapMask = labelledTraps

    for frame in range(frameNum):

        if (frame+1) % 20 == 0:
            print("Cropping trap regions for frame number {} of {}.".format(frame+1, frameNum))

        imgPath = os.path.join(params['experiment_directory'],params['image_directory'],fileNames[frame])
        fullFrameImg = io.imread(imgPath)
        if len(fullFrameImg.shape) == 3:
            if fullFrameImg.shape[0] < 3: # for tifs with less than three imaging channels, the first dimension separates channels
                fullFrameImg = np.transpose(fullFrameImg, (1,2,0))
        trapClosedEndPxDict[fileNames[frame]] = {key:{} for key in bboxesDict.keys()}

        for key in trapImagesDict.keys():

            bbox = bboxesDict[key][frame]
            trapImagesDict[key][frame,:,:,:] = fullFrameImg[bbox[0]:bbox[2],bbox[1]:bbox[3],:]

            #tmpImg = np.reshape(fullFrameImg[trapMask==key], (trapHeight,trapWidth,channelNum))

            if frame == 0:
                medianProfile = np.median(trapImagesDict[key][frame,:,:,0],axis=1) # get intensity of middle column of trap
                maxIntensityRow = np.argmax(medianProfile)

                if maxIntensityRow > trap_align_metadata['trap_height']//2:
                    flipImageDict[key] = 0
                else:
                    flipImageDict[key] = 1

            if flipImageDict[key] == 1:
                trapImagesDict[key][frame,:,:,:] = trapImagesDict[key][frame,::-1,:,:]
                trapClosedEndPxDict[fileNames[frame]][key]['closed_end_px'] = bbox[0]
                trapClosedEndPxDict[fileNames[frame]][key]['open_end_px'] = bbox[2]
            else:
                trapClosedEndPxDict[fileNames[frame]][key]['closed_end_px'] = bbox[2]
                trapClosedEndPxDict[fileNames[frame]][key]['open_end_px'] = bbox[0]
                continue

    return(trapImagesDict, trapClosedEndPxDict)

# gets shifted bounding boxes to crop traps through time
def shift_bounding_boxes(bboxesDict, shifts, imgSize):
    bboxesShiftDict = {}

    for key in bboxesDict.keys():
        bboxesShiftDict[key] = []
        bboxes = bboxesDict[key]

        for i in range(shifts.shape[0]):

            if i == 0:
                bboxesShiftDict[key].append(bboxes)
            else:
                minRow = bboxes[0]+shifts[i,0]
                minCol = bboxes[1]+shifts[i,1]
                maxRow = bboxes[2]+shifts[i,0]
                maxCol = bboxes[3]+shifts[i,1]
                bboxesShiftDict[key].append((minRow,
                                            minCol,
                                            maxRow,
                                            maxCol))
                if np.any(np.asarray([minRow,minCol,maxRow,maxCol]) < 0):
                    print("channel {} removed: out of frame".format(key))
                    del bboxesShiftDict[key]
                    break
                if np.any(np.asarray([minRow,minCol,maxRow,maxCol]) > imgSize):
                    print("channel {} removed: out of frame".format(key))
                    del bboxesShiftDict[key]
                    break

    return(bboxesShiftDict)

# finds the location of channels in a tif
def find_channel_locs(image_data):
    '''Finds the location of channels from a phase contrast image. The channels are returned in
    a dictionary where the key is the x position of the channel in pixel and the value is a
    dicionary with the open and closed end in pixels in y.


    Called by
    mm3_Compile.get_tif_params

    '''

    # declare temp variables from yaml parameter dict.
    chan_w = params['compile']['channel_width']
    chan_sep = params['compile']['channel_separation']
    crop_wp = int(params['compile']['channel_width_pad'] + chan_w/2)
    chan_snr = params['compile']['channel_detection_snr']

    # Detect peaks in the x projection (i.e. find the channels)
    projection_x = image_data.sum(axis=0).astype(np.int32)
    # find_peaks_cwt is a function which attempts to find the peaks in a 1-D array by
    # convolving it with a wave. here the wave is the default Mexican hat wave
    # but the minimum signal to noise ratio is specified
    # *** The range here should be a parameter or changed to a fraction.
    peaks = find_peaks_cwt(projection_x, np.arange(chan_w-5,chan_w+5), min_snr=chan_snr)

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
        # *** This 15 should probably be a parameter or at least changed to a fraction.
        if slice_length + 15 < default_length or slice_length - 15 > default_length:
            continue
        # make sure ends are greater than 15 pixels from image edge
        if slice_closed_end_px < 15 or slice_open_end_px > image_data.shape[0] - 15:
            continue

        # if you made it to this point then update the entry
        chnl_loc_dict[peak] = {'closed_end_px' : slice_closed_end_px,
                                 'open_end_px' : slice_open_end_px}

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
    crop_wp = int(params['compile']['channel_width_pad'] + params['compile']['channel_width']/2)
    chan_lp = int(params['compile']['channel_length_pad'])

    #intiaize dictionary
    channel_masks = {}

    # get the size of the images (hope they are the same)
    for img_k in analyzed_imgs.keys():
        img_v = analyzed_imgs[img_k]
        image_rows = img_v['shape'][0] # x pixels
        image_cols = img_v['shape'][1] # y pixels
        break # just need one. using iteritems mean the whole dict doesn't load

    # get the fov ids
    fovs = []
    for img_k in analyzed_imgs.keys():
        img_v = analyzed_imgs[img_k]
        if img_v['fov'] not in fovs:
            fovs.append(img_v['fov'])

    # max width and length across all fovs. channels will get expanded by these values
    # this important for later updates to the masks, which should be the same
    max_chnl_mask_len = 0
    max_chnl_mask_wid = 0

    # for each fov make a channel_mask dictionary from consensus mask
    for fov in fovs:
        # initialize a the dict and consensus mask
        channel_masks_1fov = {} # dict which holds channel masks {peak : [[y1, y2],[x1,x2]],...}
        consensus_mask = np.zeros([image_rows, image_cols]) # mask for labeling

        # bring up information for each image
        for img_k in analyzed_imgs.keys():
            img_v = analyzed_imgs[img_k]
            # skip this one if it is not of the current fov
            if img_v['fov'] != fov:
                continue

            # for each channel in each image make a single mask
            img_chnl_mask = np.zeros([image_rows, image_cols])

            # and add the channel mask to it
            for chnl_peak, peak_ends in six.iteritems(img_v['channels']):
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

        # Normalize concensus mask between 0 and 1.
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

            # store the edge locations of the channel mask in the dictionary. Will be ints
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

    for fov, peaks in six.iteritems(channel_masks):
        # f_id = int(fov)
        for peak, chnl_mask in six.iteritems(peaks):
            # p_id = int(peak)
            # just add length to the open end (bottom of image, low column)
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

            # convert all values to ints
            chnl_mask[0][0] = int(chnl_mask[0][0])
            chnl_mask[0][1] = int(chnl_mask[0][1])
            chnl_mask[1][0] = int(chnl_mask[1][0])
            chnl_mask[1][1] = int(chnl_mask[1][1])

            # cm_copy[fov][peak] = {'y_top': chnl_mask[0][0],
            #                       'y_bot': chnl_mask[0][1],
            #                       'x_left': chnl_mask[1][0],
            #                       'x_right': chnl_mask[1][1]}
            # print(type(cm_copy[fov][peak][1][0]), cm_copy[fov][peak][1][0])

    #save the channel mask dictionary to a pickle and a text file
    # with open(os.path.join(params['ana_dir'], 'channel_masks.pkl'), 'wb') as cmask_file:
    #     pickle.dump(cm_copy, cmask_file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(params['ana_dir'], 'channel_masks.txt'), 'w') as cmask_file:
        pprint(cm_copy, stream=cmask_file)
    with open(os.path.join(params['ana_dir'], 'channel_masks.yaml'), 'w') as cmask_file:
        yaml.dump(data=cm_copy, stream=cmask_file, default_flow_style=False, tags=None)

    information("Channel masks saved.")

    return cm_copy

# get each fov_id, peak_id, frame's mask bounding box from bounding boxes arrived at by convolutional neural network
def make_channel_masks_CNN(bboxes_dict):
    '''
    The keys in this dictionary are peak_ids and the values of each is an array of shape (frameNumber,2,2):
    Each frameNumber's 2x2 slice of the array represents the given peak_id's [[minrow, maxrow],[mincol, maxcol]].

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

    # initialize the new channel_masks dict
    channel_masks = {}

    # reorder elements of tuples in bboxes_dict to match [[minrow, maxrow], [mincol, maxcol]] convention above
    peak_ids = [peak_id for peak_id in bboxes_dict.keys()]
    peak_ids.sort()

    bbox_array = np.zeros((len(bboxes_dict[peak_ids[0]]),2,2), dtype='uint16')
    for peak_id in peak_ids:
        # get each frame's bounding boxes for the given peak_id
        frame_bboxes = bboxes_dict[peak_id]

        for frame_index in range(len(frame_bboxes)):
            # replace the values in bbox_array with the proper ones from frame_bboxes
            minrow = frame_bboxes[frame_index][0]
            maxrow = frame_bboxes[frame_index][2]
            mincol = frame_bboxes[frame_index][1]
            maxcol = frame_bboxes[frame_index][3]
            bbox_array[frame_index,0,0] = minrow
            bbox_array[frame_index,0,1] = maxrow
            bbox_array[frame_index,1,0] = mincol
            bbox_array[frame_index,1,1] = maxcol

        channel_masks[peak_id] = bbox_array

    return(channel_masks)

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
    image_orientation = params['compile']['image_orientation']

    # if this is just a phase image give in an extra layer so rest of code is fine
    flat = False # flag for if the image is flat or multiple levels
    if len(image_data.shape) == 2:
        image_data = np.expand_dims(image_data, 0)
        flat = True

    # setting image_orientation to 'auto' will use autodetection
    if image_orientation == "auto":
         # use 'phase_plane' to find the phase plane in image_data, assuming c1, c2, c3... naming scheme here.
        try:
            ph_channel = int(re.search('[0-9]', params['phase_plane']).group(0)) - 1
        except:
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

    pad_size = params['subtract']['alignment_pad']

    # Use this number of images to calculate cross correlations
    number_of_images = 20

    # load the phase contrast images
    image_data = load_stack(fov_id, peak_id, color=params['phase_plane'])

    # if there are more images than number_of_images, use number_of_images images evenly
    # spaced across the range
    if image_data.shape[0] > number_of_images:
        spacing = int(image_data.shape[0] / number_of_images)
        image_data = image_data[::spacing,:,:]
        if image_data.shape[0] > number_of_images:
            image_data = image_data[:number_of_images,:,:]

    # we will compare all images to this one, needs to be padded to account for image drift
    first_img = np.pad(image_data[0,:,:], pad_size, mode='reflect')

    xcorr_array = [] # array holds cross correlation vaues
    for img in image_data:
        # use match_template to find all cross correlations for the
        # current image against the first image.
        xcorr_array.append(np.max(match_template(first_img, img)))

    return xcorr_array

### functions about subtraction

# average empty channels from stacks, making another TIFF stack
def average_empties_stack(fov_id, specs, color='c1', align=True):
    '''Takes the fov file name and the peak names of the designated empties,
    averages them and saves the image

    Parameters
    fov_id : int
        FOV number
    specs : dict
        specifies whether a channel should be analyzed (1), used for making
        an average empty (0), or ignored (-1).
    color : string
        Which plane to use.
    align : boolean
        Flag that is passed to the worker function average_empties, indicates
        whether images should be aligned be for averaging (use False for fluorescent images)

    Returns
        True if succesful.
        Saves empty stack to analysis folder

    '''

    information("Creating average empty channel for FOV %d." % fov_id)

    # get peak ids of empty channels for this fov
    empty_peak_ids = []
    for peak_id, spec in six.iteritems(specs[fov_id]):
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
        avg_empty_stack = load_stack(fov_id, peak_id, color=color)

    # but if there is more than one empty you need to align and average them per timepoint
    elif len(empty_peak_ids) > 1:
        # load the image stacks into memory
        empty_stacks = [] # list which holds phase image stacks of designated empties
        for peak_id in empty_peak_ids:
            # load data and append to list
            image_data = load_stack(fov_id, peak_id, color=color)

            empty_stacks.append(image_data)

        information("%d empty channels designated for FOV %d." % (len(empty_stacks), fov_id))

        # go through time points and create list of averaged empties
        avg_empty_stack = [] # list will be later concatentated into numpy array
        time_points = range(image_data.shape[0]) # index is time
        for t in time_points:
            # get images from one timepoint at a time and send to alignment and averaging
            imgs = [stack[t] for stack in empty_stacks]
            avg_empty = average_empties(imgs, align=align) # function is in mm3
            avg_empty_stack.append(avg_empty)

        # concatenate list and then save out to tiff stack
        avg_empty_stack = np.stack(avg_empty_stack, axis=0)

    # save out data
    if params['output'] == 'TIFF':
        # make new name and save it
        empty_filename = params['experiment_name'] + '_xy%03d_empty_%s.tif' % (fov_id, color)
        tiff.imsave(os.path.join(params['empty_dir'],empty_filename), avg_empty_stack, compress=4)

    if params['output'] == 'HDF5':
        h5f = h5py.File(os.path.join(params['hdf5_dir'],'xy%03d.hdf5' % fov_id), 'r+')

        # delete the dataset if it exists (important for debug)
        if 'empty_%s' % color in h5f:
            del h5f[u'empty_%s' % color]

        # the empty channel should be it's own dataset
        h5ds = h5f.create_dataset(u'empty_%s' % color,
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
def average_empties(imgs, align=True):
    '''
    This function averages a set of images (empty channels) and returns a single image
    of the same size. It first aligns the images to the first image before averaging.

    Alignment is done by enlarging the first image using edge padding.
    Subsequent images are then aligned to this image and the offset recorded.
    These images are padded such that they are the same size as the first (padded) image but
    with the image in the correct (aligned) place. Edge padding is again used.
    The images are then placed in a stack and aveaged. This image is trimmed so it is the size
    of the original images

    Called by
    average_empties_stack
    '''

    aligned_imgs = [] # list contains the aligned, padded images

    if align:
        # pixel size to use for padding (ammount that alignment could be off)
        pad_size = params['subtract']['alignment_pad']

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
    else:
        # don't align, just link the names to go forward easily
        aligned_imgs = imgs

    # stack the aligned data along 3rd axis
    aligned_imgs = np.dstack(aligned_imgs)
    # get a mean image along 3rd axis
    avg_empty = np.nanmean(aligned_imgs, axis=2)
    # trim off the padded edges (only if images were alinged, otherwise there was no padding)
    if align:
        avg_empty = avg_empty[pad_size:-1*pad_size, pad_size:-1*pad_size]
    # change type back to unsigned 16 bit not floats
    avg_empty = avg_empty.astype(dtype='uint16')

    return avg_empty

# this function is used when one FOV doesn't have an empty
def copy_empty_stack(from_fov, to_fov, color='c1'):
    '''Copy an empty stack from one FOV to another'''

    # load empty stack from one FOV
    information('Loading empty stack from FOV {} to save for FOV {}.'.format(from_fov, to_fov))
    avg_empty_stack = load_stack(from_fov, 0, color='empty_{}'.format(color))

    # save out data
    if params['output'] == 'TIFF':
        # make new name and save it
        empty_filename = params['experiment_name'] + '_xy%03d_empty_%s.tif' % (to_fov, color)
        tiff.imsave(os.path.join(params['empty_dir'],empty_filename), avg_empty_stack, compress=4)

    if params['output'] == 'HDF5':
        h5f = h5py.File(os.path.join(params['hdf5_dir'],'xy%03d.hdf5' % to_fov), 'r+')

        # delete the dataset if it exists (important for debug)
        if 'empty_%s' % color in h5f:
            del h5f[u'empty_%s' % color]

        # the empty channel should be it's own dataset
        h5ds = h5f.create_dataset(u'empty_%s' % color,
                        data=avg_empty_stack,
                        chunks=(1, avg_empty_stack.shape[1], avg_empty_stack.shape[2]),
                        maxshape=(None, avg_empty_stack.shape[1], avg_empty_stack.shape[2]),
                        compression="gzip", shuffle=True, fletcher32=True)

        # give attribute which says which channels contribute. Just put 0
        h5ds.attrs.create('empty_channels', [0])
        h5f.close()

    information("Saved empty channel for FOV %d." % to_fov)

# Do subtraction for an fov over many timepoints
def subtract_fov_stack(fov_id, specs, color='c1', method='phase'):
    '''
    For a given FOV, loads the precomputed empty stack and does subtraction on
    all peaks in the FOV designated to be analyzed

    Parameters
    ----------
    color : string, 'c1', 'c2', etc.
        This is the channel to subtraction. will be appended to the word empty.

    Called by
    mm3_Subtract.py

    Calls
    mm3.subtract_phase

    '''

    information('Subtracting peaks for FOV %d.' % fov_id)

    # load empty stack feed dummy peak number to get empty
    avg_empty_stack = load_stack(fov_id, 0, color='empty_{}'.format(color))

    # determine which peaks are to be analyzed
    ana_peak_ids = []
    for peak_id, spec in six.iteritems(specs[fov_id]):
        if spec == 1: # 0 means it should be used for empty, -1 is ignore
            ana_peak_ids.append(peak_id)
    ana_peak_ids = sorted(ana_peak_ids) # sort for repeatability
    information("Subtracting %d channels for FOV %d." % (len(ana_peak_ids), fov_id))

    # just break if there are to peaks to analize
    if not ana_peak_ids:
        return False

    # load images for the peak and get phase images
    for peak_id in ana_peak_ids:
        information('Subtracting peak %d.' % peak_id)

        image_data = load_stack(fov_id, peak_id, color=color)

        # make a list for all time points to send to a multiprocessing pool
        # list will length of image_data with tuples (image, empty)
        subtract_pairs = zip(image_data, avg_empty_stack)

        # set up multiprocessing pool to do subtraction. Should wait until finished
        pool = Pool(processes=params['num_analyzers'])

        if method == 'phase':
            subtracted_imgs = pool.map(subtract_phase, subtract_pairs, chunksize=10)
        elif method == 'fluor':
            subtracted_imgs = pool.map(subtract_fluor, subtract_pairs, chunksize=10)

        pool.close() # tells the process nothing more will be added.
        pool.join() # blocks script until everything has been processed and workers exit

        # linear loop for debug
        # subtracted_imgs = [subtract_phase(subtract_pair) for subtract_pair in subtract_pairs]

        # stack them up along a time axis
        subtracted_stack = np.stack(subtracted_imgs, axis=0)

        # save out the subtracted stack
        if params['output'] == 'TIFF':
            sub_filename = params['experiment_name'] + '_xy%03d_p%04d_sub_%s.tif' % (fov_id, peak_id, color)
            tiff.imsave(os.path.join(params['sub_dir'],sub_filename), subtracted_stack, compress=4) # save it

        if params['output'] == 'HDF5':
            h5f = h5py.File(os.path.join(params['hdf5_dir'],'xy%03d.hdf5' % fov_id), 'r+')

            # put subtracted channel in correct group
            h5g = h5f['channel_%04d' % peak_id]

            # delete the dataset if it exists (important for debug)
            if 'p%04d_sub_%s' % (peak_id, color) in h5g:
                del h5g['p%04d_sub_%s' % (peak_id, color)]

            h5ds = h5g.create_dataset(u'p%04d_sub_%s' % (peak_id, color),
                            data=subtracted_stack,
                            chunks=(1, subtracted_stack.shape[1], subtracted_stack.shape[2]),
                            maxshape=(None, subtracted_stack.shape[1], subtracted_stack.shape[2]),
                            compression="gzip", shuffle=True, fletcher32=True)

        information("Saved subtracted channel %d." % peak_id)

    if params['output'] == 'HDF5':
        h5f.close()

    return True

# subtracts one phase contrast image from another.
def subtract_phase(image_pair):
    '''subtract_phase aligns and subtracts a .
    Modified from subtract_phase_only by jt on 20160511
    The subtracted image returned is the same size as the image given. It may however include
    data points around the edge that are meaningless but not marked.

    We align the empty channel to the phase channel, then subtract.

    Parameters
    image_pair : tuple of length two with; (image, empty_mean)

    Returns
    channel_subtracted : np.array
        The subtracted image

    Called by
    subtract_fov_stack
    '''
    # get out data and pad
    cropped_channel, empty_channel = image_pair # [channel slice, empty slice]

    # this is for aligning the empty channel to the cell channel.
    ### Pad cropped channel.
    pad_size = params['subtract']['alignment_pad'] # pixel size to use for padding (ammount that alignment could be off)
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
    # channel_subtracted = cropped_channel.astype('int32') - aligned_empty.astype('int32')

    # just zero out anything less than 0. This is what Sattar does
    channel_subtracted[channel_subtracted < 0] = 0
    channel_subtracted = channel_subtracted.astype('uint16') # change back to 16bit

    return channel_subtracted

# subtract one fluorescence image from another.
def subtract_fluor(image_pair):
    ''' subtract_fluor does a simple subtraction of one image to another. Unlike subtract_phase,
    there is no alignment. Also, the empty channel is subtracted from the full channel.

    Parameters
    image_pair : tuple of length two with; (image, empty_mean)

    Returns
    channel_subtracted : np.array
        The subtracted image.

    Called by
    subtract_fov_stack
    '''
    # get out data and pad
    cropped_channel, empty_channel = image_pair # [channel slice, empty slice]

    # check frame size of cropped channel and background, always keep crop channel size the same
    crop_size = np.shape(cropped_channel)[:2]
    empty_size = np.shape(empty_channel)[:2]
    if crop_size != empty_size:
        if crop_size[0] > empty_size[0] or crop_size[1] > empty_size[1]:
            pad_row_length = max(crop_size[0]  - empty_size[0], 0) # prevent negatives
            pad_column_length = max(crop_size[1]  - empty_size[1], 0)
            empty_channel = np.pad(empty_channel,
                [[np.int(.5*pad_row_length), pad_row_length-np.int(.5*pad_row_length)],
                [np.int(.5*pad_column_length),  pad_column_length-np.int(.5*pad_column_length)],
                [0,0]], 'edge')
            # mm3.information('size adjusted 1')
        empty_size = np.shape(empty_channel)[:2]
        if crop_size[0] < empty_size[0] or crop_size[1] < empty_size[1]:
            empty_channel = empty_channel[:crop_size[0], :crop_size[1],]

    ### Compute the difference between the empty and channel phase contrast images
    # subtract cropped cell image from empty channel.
    channel_subtracted = cropped_channel.astype('int32') - empty_channel.astype('int32')
    # channel_subtracted = cropped_channel.astype('int32') - aligned_empty.astype('int32')

    # just zero out anything less than 0.
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
    sub_stack = load_stack(fov_id, peak_id, color='sub_{}'.format(params['phase_plane']))

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
    segmented_imgs = segmented_imgs.astype('uint8')

    # save out the segmented stack
    if params['output'] == 'TIFF':
        seg_filename = params['experiment_name'] + '_xy%03d_p%04d_%s.tif' % (fov_id, peak_id, params['seg_img'])
        tiff.imsave(os.path.join(params['seg_dir'],seg_filename),
                    segmented_imgs, compress=5)

    if params['output'] == 'HDF5':
        h5f = h5py.File(os.path.join(params['hdf5_dir'],'xy%03d.hdf5' % fov_id), 'r+')

        # put segmented channel in correct group
        h5g = h5f['channel_%04d' % peak_id]

        # delete the dataset if it exists (important for debug)
        if 'p%04d_%s' % (peak_id, params['seg_img']) in h5g:
            del h5g['p%04d_%s' % (peak_id, params['seg_img'])]

        h5ds = h5g.create_dataset(u'p%04d_%s' % (peak_id, params['seg_img']),
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
    OTSU_threshold = params['segment']['otsu']['OTSU_threshold']
    first_opening_size = params['segment']['otsu']['first_opening_size']
    distance_threshold = params['segment']['otsu']['distance_threshold']
    second_opening_size = params['segment']['otsu']['second_opening_size']
    min_object_size = params['segment']['min_object_size']

    # threshold image
    try:
        thresh = threshold_otsu(image) # finds optimal OTSU threshhold value
    except:
        return np.zeros_like(image)

    threshholded = image > OTSU_threshold*thresh # will create binary image

    # if there are no cells, good to clear the border
    # because otherwise the OTSU is just for random bullshit, most
    # likely on the side of the image
    threshholded = segmentation.clear_border(threshholded)

    # Opening = erosion then dialation.
    # opening smooths images, breaks isthmuses, and eliminates protrusions.
    # "opens" dark gaps between bright features.
    morph = morphology.binary_opening(threshholded, morphology.disk(first_opening_size))

    # if this image is empty at this point (likely if there were no cells), just return
    # zero array
    if np.amax(morph) == 0:
        return np.zeros_like(image)

    ### Calculate distance matrix, use as markers for random walker (diffusion watershed)
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
        return np.zeros_like(image)

    # relabel now that small objects and labels on edges have been cleared
    markers = morphology.label(cleared, connectivity=1)

    # just break if there is no label
    if np.amax(markers) == 0:
        return np.zeros_like(image)

    # the binary image for the watershed, which uses the unmodified OTSU threshold
    threshholded_watershed = threshholded
    threshholded_watershed = segmentation.clear_border(threshholded_watershed)

    # label using the random walker (diffusion watershed) algorithm
    try:
        # set anything outside of OTSU threshold to -1 so it will not be labeled
        markers[threshholded_watershed == 0] = -1
        # here is the main algorithm
        labeled_image = segmentation.random_walker(-1*image, markers)
        # put negative values back to zero for proper image
        labeled_image[labeled_image == -1] = 0
    except:
        return np.zeros_like(image)

    return labeled_image

# loss functions for model
def dice_coeff(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    loss = losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss

def tversky_loss(y_true, y_pred):
    alpha = 0.5
    beta  = 0.5

    ones = K.ones((512,512,3)) #K.ones(K.shape(y_true))
    p0 = y_pred      # proba that voxels are class i
    p1 = ones-y_pred # proba that voxels are not class i
    g0 = y_true
    g1 = ones-y_true

    num = K.sum(p0*g0, (0,1,2))
    den = num + alpha*K.sum(p0*g1,(0,1,2)) + beta*K.sum(p1*g0,(0,1,2))

    T = K.sum(num/den) # when summing over classes, T has dynamic range [0 Ncl]

    Ncl = K.cast(K.shape(y_true)[-1], 'float32')
    return Ncl-T

def cce_tversky_loss(y_true, y_pred):
    loss = losses.categorical_crossentropy(y_true, y_pred) + tversky_loss(y_true, y_pred)
    return loss

def get_pad_distances(unet_shape, img_height, img_width):
    '''Finds padding and trimming sizes to make the input image the same as the size expected by the U-net model.

    Padding is done evenly to the top and bottom of the image. Trimming is only done from the right or bottom.
    '''

    half_width_pad = (unet_shape[1]-img_width)/2
    if half_width_pad > 0:
        left_pad = int(np.floor(half_width_pad))
        right_pad = int(np.ceil(half_width_pad))
        right_trim = 0
    else:
        left_pad = 0
        right_pad = 0
        right_trim = img_width - unet_shape[1]

    half_height_pad = (unet_shape[0]-img_height)/2
    if half_height_pad > 0:
        top_pad = int(np.floor(half_height_pad))
        bottom_pad = int(np.ceil(half_height_pad))
        bottom_trim = 0
    else:
        top_pad = 0
        bottom_pad = 0
        bottom_trim = img_height - unet_shape[0]

    pad_dict = {'top_pad' : top_pad,
                'bottom_pad' : bottom_pad,
                'right_pad' : right_pad,
                'left_pad' : left_pad,
                'bottom_trim' : bottom_trim,
                'right_trim' : right_trim}

    return pad_dict

def segment_cells_unet(ana_peak_ids, fov_id, pad_dict, unet_shape, model):

    batch_size = params['segment']['unet']['batch_size']
    cellClassThreshold = params['segment']['unet']['cell_class_threshold']
    if cellClassThreshold == 'None': # yaml imports None as a string
        cellClassThreshold = False
    min_object_size = params['segment']['min_object_size']

    # arguments to data generator
    # data_gen_args = {'batch_size':batch_size,
    #                  'n_channels':1,
    #                  'normalize_to_one':False,
    #                  'shuffle':False}
    # arguments to predict_generator
    predict_args = dict(use_multiprocessing=True,
                        workers=params['num_analyzers'],
                        verbose=1)

    for peak_id in ana_peak_ids:
        information('Segmenting peak {}.'.format(peak_id))

        img_stack = load_stack(fov_id, peak_id, color=params['phase_plane'])

        # med_stack = np.zeros(img_stack.shape)
        # selem = morphology.disk(1)
        #
        # for frame_idx in range(img_stack.shape[0]):
        #     tmpImg = img_stack[frame_idx,...]
        #     med_stack[frame_idx,...] = median(tmpImg, selem)
        #
        # # robust normalization of peak's image stack to 1
        # max_val = np.max(med_stack)
        # img_stack = img_stack/max_val
        # img_stack[img_stack > 1] = 1

        # trim and pad image to correct size
        img_stack = img_stack[:, :unet_shape[0], :unet_shape[1]]
        img_stack = np.pad(img_stack,
                           ((0,0),
                           (pad_dict['top_pad'],pad_dict['bottom_pad']),
                           (pad_dict['left_pad'],pad_dict['right_pad'])),
                           mode='constant')
        img_stack = np.expand_dims(img_stack, -1) # TF expects images to be 4D
        # set up image generator
        # image_generator = CellSegmentationDataGenerator(img_stack, **data_gen_args)
        image_datagen = ImageDataGenerator()
        image_generator = image_datagen.flow(x=img_stack,
                                             batch_size=batch_size,
                                             shuffle=False) # keep same order

        # predict cell locations. This has multiprocessing built in but I need to mess with the parameters to see how to best utilize it. ***
        predictions = model.predict_generator(image_generator, **predict_args)

        # post processing
        # remove padding including the added last dimension
        predictions = predictions[:, pad_dict['top_pad']:unet_shape[0]-pad_dict['bottom_pad'],
                                     pad_dict['left_pad']:unet_shape[1]-pad_dict['right_pad'], 0]

        # pad back incase the image had been trimmed
        predictions = np.pad(predictions,
                             ((0,0),
                             (0,pad_dict['bottom_trim']),
                             (0,pad_dict['right_trim'])),
                             mode='constant')

        if params['segment']['unet']['save_predictions']:
            pred_filename = params['experiment_name'] + '_xy%03d_p%04d_%s.tif' % (fov_id, peak_id, params['pred_img'])
            if not os.path.isdir(params['pred_dir']):
                os.makedirs(params['pred_dir'])
            int_preds = (predictions * 255).astype('uint8')
            tiff.imsave(os.path.join(params['pred_dir'], pred_filename),
                            int_preds, compress=4)

        # binarized and label (if there is a threshold value, otherwise, save a grayscale for debug)
        if cellClassThreshold:
            predictions[predictions >= cellClassThreshold] = 1
            predictions[predictions < cellClassThreshold] = 0
            predictions = predictions.astype('uint8')

            segmented_imgs = np.zeros(predictions.shape, dtype='uint8')
            # process and label each frame of the channel
            for frame in range(segmented_imgs.shape[0]):
                # get rid of small holes
                predictions[frame,:,:] = morphology.remove_small_holes(predictions[frame,:,:], min_object_size)
                # get rid of small objects.
                predictions[frame,:,:] = morphology.remove_small_objects(morphology.label(predictions[frame,:,:], connectivity=1), min_size=min_object_size)
                # remove labels which touch the boarder
                predictions[frame,:,:] = segmentation.clear_border(predictions[frame,:,:])
                # relabel now
                segmented_imgs[frame,:,:] = morphology.label(predictions[frame,:,:], connectivity=1)

        else: # in this case you just want to scale the 0 to 1 float image to 0 to 255
            information('Converting predictions to grayscale.')
            segmented_imgs = np.around(predictions * 100)

        # both binary and grayscale should be 8bit. This may be ensured above and is unneccesary
        segmented_imgs = segmented_imgs.astype('uint8')

        # save out the segmented stacks
        if params['output'] == 'TIFF':
            seg_filename = params['experiment_name'] + '_xy%03d_p%04d_%s.tif' % (fov_id, peak_id, params['seg_img'])
            tiff.imsave(os.path.join(params['seg_dir'], seg_filename),
                            segmented_imgs, compress=4)

        if params['output'] == 'HDF5':
            h5f = h5py.File(os.path.join(params['hdf5_dir'],'xy%03d.hdf5' % fov_id), 'r+')
            # put segmented channel in correct group
            h5g = h5f['channel_%04d' % peak_id]
            # delete the dataset if it exists (important for debug)
            if 'p%04d_%s' % (peak_id, params['seg_img']) in h5g:
                del h5g['p%04d_%s' % (peak_id, params['seg_img'])]

            h5ds = h5g.create_dataset(u'p%04d_%s' % (peak_id, params['seg_img']),
                                data=segmented_imgs,
                                chunks=(1, segmented_imgs.shape[1], segmented_imgs.shape[2]),
                                maxshape=(None, segmented_imgs.shape[1], segmented_imgs.shape[2]),
                                compression="gzip", shuffle=True, fletcher32=True)
            h5f.close()

def segment_fov_unet(fov_id, specs, model, color=None):
    '''
    Segments the channels from one fov using the U-net CNN model.

    Parameters
    ----------
    fov_id : int
    specs : dict
    model : TensorFlow model
    '''

    information('Segmenting FOV {} with U-net.'.format(fov_id))

    if color is None:
        color = params['phase_plane']

    # load segmentation parameters
    unet_shape = (params['segment']['unet']['trained_model_image_height'],
                  params['segment']['unet']['trained_model_image_width'])

    ### determine stitching of images.
    # need channel shape, specifically the width. load first for example
    # this assumes that all channels are the same size for this FOV, which they should
    for peak_id, spec in six.iteritems(specs[fov_id]):
        if spec == 1:
            break # just break out with the current peak_id

    img_stack = load_stack(fov_id, peak_id, color=color)
    img_height = img_stack.shape[1]
    img_width = img_stack.shape[2]

    pad_dict = get_pad_distances(unet_shape, img_height, img_width)

    # dermine how many channels we have to analyze for this FOV
    ana_peak_ids = []
    for peak_id, spec in six.iteritems(specs[fov_id]):
        if spec == 1:
            ana_peak_ids.append(peak_id)
    ana_peak_ids.sort() # sort for repeatability

    segment_cells_unet(ana_peak_ids, fov_id, pad_dict, unet_shape, model)

    information("Finished segmentation for FOV {}.".format(fov_id))

    return

def segment_foci_unet(ana_peak_ids, fov_id, pad_dict, unet_shape, model):

    # batch_size = params['foci']['batch_size']
    focusClassThreshold = params['foci']['focus_threshold']
    if focusClassThreshold == 'None': # yaml imports None as a string
        focusClassThreshold = False

    # arguments to data generator
    data_gen_args = {'batch_size':params['foci']['batch_size'],
                     'n_channels':1,
                     'normalize_to_one':False,
                     'shuffle':False}
    # arguments to predict_generator
    predict_args = dict(use_multiprocessing=False,
                        # workers=params['num_analyzers'],
                        verbose=1)

    for peak_id in ana_peak_ids:
        information('Segmenting foci in peak {}.'.format(peak_id))
        # print(peak_id) # debugging a shape error at some traps

        img_stack = load_stack(fov_id, peak_id, color=params['foci']['foci_plane'])
        # med_stack = np.zeros(img_stack.shape)
        # selem = morphology.disk(1)

        # with warnings.catch_warnings():
        #     warnings.simplefilter('ignore')
        #     for frame_idx in range(img_stack.shape[0]):
        #         tmpImg = img_stack[frame_idx,...]
        #         med_stack[frame_idx,...] = median(tmpImg, selem)

        # robust normalization of peak's image stack to 1
        # max_val = np.max(med_stack)
        # img_stack = img_stack/max_val
        # img_stack[img_stack > 1] = 1

        # pad image to correct size
        img_stack = np.pad(img_stack,
                           ((0,0),
                           (pad_dict['top'],pad_dict['bottom']),
                           (pad_dict['left'],pad_dict['right'])),
                           mode='constant')
        img_stack = np.expand_dims(img_stack, -1)
        # set up image generator
        image_generator = FocusSegmentationDataGenerator(img_stack, **data_gen_args)
        #image_datagen = ImageDataGenerator()
        #image_generator = image_datagen.flow(x=img_stack,
        #                                     batch_size=batch_size,
        #                                     shuffle=False) # keep same order

        # predict cell locations. This has multiprocessing built in but I need to mess with the parameters to see how to best utilize it. ***
        predictions = model.predict_generator(image_generator, **predict_args)

        # post processing
        # remove padding including the added last dimension
        predictions = predictions[:, pad_dict['top']:unet_shape[0]-pad_dict['bottom'],
                                     pad_dict['left']:unet_shape[1]-pad_dict['right'], 0]

        if params['foci']['save_predictions']:
            pred_filename = params['experiment_name'] + '_xy%03d_p%04d_%s.tif' % (fov_id, peak_id, params['pred_img'])
            if not os.path.isdir(params['foci_pred_dir']):
                os.makedirs(params['foci_pred_dir'])
            int_preds = (predictions * 255).astype('uint8')
            tiff.imsave(os.path.join(params['foci_pred_dir'], pred_filename),
                            int_preds, compress=4)

        # binarized and label (if there is a threshold value, otherwise, save a grayscale for debug)
        if focusClassThreshold:
            predictions[predictions >= focusClassThreshold] = 1
            predictions[predictions < focusClassThreshold] = 0
            predictions = predictions.astype('uint8')

            segmented_imgs = np.zeros(predictions.shape, dtype='uint8')
            # process and label each frame of the channel
            for frame in range(segmented_imgs.shape[0]):
                # get rid of small holes
                # predictions[frame,:,:] = morphology.remove_small_holes(predictions[frame,:,:], min_object_size)
                # get rid of small objects.
                # predictions[frame,:,:] = morphology.remove_small_objects(morphology.label(predictions[frame,:,:], connectivity=1), min_size=min_object_size)
                # remove labels which touch the boarder
                predictions[frame,:,:] = segmentation.clear_border(predictions[frame,:,:])
                # relabel now
                segmented_imgs[frame,:,:] = morphology.label(predictions[frame,:,:], connectivity=2)

        else: # in this case you just want to scale the 0 to 1 float image to 0 to 255
            information('Converting predictions to grayscale.')
            segmented_imgs = np.around(predictions * 100)

        # both binary and grayscale should be 8bit. This may be ensured above and is unneccesary
        segmented_imgs = segmented_imgs.astype('uint8')

        # save out the segmented stacks
        if params['output'] == 'TIFF':
            seg_filename = params['experiment_name'] + '_xy%03d_p%04d_%s.tif' % (fov_id, peak_id, params['seg_img'])
            tiff.imsave(os.path.join(params['foci_seg_dir'], seg_filename),
                            segmented_imgs, compress=4)

        if params['output'] == 'HDF5':
            h5f = h5py.File(os.path.join(params['hdf5_dir'],'xy%03d.hdf5' % fov_id), 'r+')
            # put segmented channel in correct group
            h5g = h5f['channel_%04d' % peak_id]
            # delete the dataset if it exists (important for debug)
            if 'p%04d_%s' % (peak_id, params['seg_img']) in h5g:
                del h5g['p%04d_%s' % (peak_id, params['seg_img'])]

            h5ds = h5g.create_dataset(u'p%04d_%s' % (peak_id, params['seg_img']),
                                data=segmented_imgs,
                                chunks=(1, segmented_imgs.shape[1], segmented_imgs.shape[2]),
                                maxshape=(None, segmented_imgs.shape[1], segmented_imgs.shape[2]),
                                compression="gzip", shuffle=True, fletcher32=True)
            h5f.close()

def segment_fov_foci_unet(fov_id, specs, model, color=None):
    '''
    Segments the channels from one fov using the U-net CNN model.

    Parameters
    ----------
    fov_id : int
    specs : dict
    model : TensorFlow model
    '''

    information('Segmenting FOV {} with U-net.'.format(fov_id))

    if color is None:
        color = params['phase_plane']

    # load segmentation parameters
    unet_shape = (params['segment']['unet']['trained_model_image_height'],
                  params['segment']['unet']['trained_model_image_width'])

    ### determine stitching of images.
    # need channel shape, specifically the width. load first for example
    # this assumes that all channels are the same size for this FOV, which they should
    for peak_id, spec in six.iteritems(specs[fov_id]):
        if spec == 1:
            break # just break out with the current peak_id

    img_stack = load_stack(fov_id, peak_id, color=color)
    img_height = img_stack.shape[1]
    img_width = img_stack.shape[2]

    # find padding and trimming distances
    pad_dict = get_pad_distances(unet_shape, img_height, img_width)
    # timepoints = img_stack.shape[0]

    # dermine how many channels we have to analyze for this FOV
    ana_peak_ids = []
    for peak_id, spec in six.iteritems(specs[fov_id]):
        if spec == 1:
            ana_peak_ids.append(peak_id)
    ana_peak_ids.sort() # sort for repeatability

    k = segment_foci_unet(ana_peak_ids, fov_id, pad_dict, unet_shape, model)

    information("Finished segmentation for FOV {}.".format(fov_id))
    return(k)

# class for image generation for predicting cell locations in phase-contrast images
class CellSegmentationDataGenerator(utils.Sequence):
    'Generates data for Keras'
    def __init__(self,
                 img_array,
                 batch_size=32,
                 n_channels=1,
                 shuffle=False,
                 normalize_to_one=False):
        'Initialization'
        self.dim = (img_array.shape[1], img_array.shape[2])
        self.batch_size = batch_size
        self.img_array = img_array
        self.img_number = img_array.shape[0]
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()
        self.normalize_to_one = normalize_to_one
        if normalize_to_one:
            self.selem = morphology.disk(1)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return(int(np.ceil(self.img_number / self.batch_size)))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        array_list_temp = [self.img_array[k,:,:,0] for k in indexes]

        # Generate data
        X = self.__data_generation(array_list_temp)

        return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.img_number)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, array_list_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros((self.batch_size, self.dim[0], self.dim[1], self.n_channels))

        # Generate data
        for i in range(self.batch_size):
            # Store sample
            try:
                tmpImg = array_list_temp[i]
            except IndexError:
                X = X[:i,...]
                break

            # ensure image is uint8
            if tmpImg.dtype=="uint16":
                tmpImg = tmpImg / 2**16 * 2**8
                tmpImg = tmpImg.astype('uint8')

            if self.normalize_to_one:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    medImg = median(tmpImg, self.selem)
                    tmpImg = tmpImg/np.max(medImg)
                    tmpImg[tmpImg > 1] = 1

            X[i,:,:,0] = tmpImg

        return (X)

# class for image generation for predicting cell locations in phase-contrast images
class FocusSegmentationDataGenerator(utils.Sequence):
    'Generates data for Keras'
    def __init__(self,
                 img_array,
                 batch_size=32,
                 n_channels=1,
                 shuffle=False,
                 normalize_to_one=False):
        'Initialization'
        self.dim = (img_array.shape[1], img_array.shape[2])
        self.batch_size = batch_size
        self.img_array = img_array
        self.img_number = img_array.shape[0]
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()
        self.normalize_to_one = normalize_to_one
        if normalize_to_one:
            self.selem = morphology.disk(1)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return(int(np.ceil(self.img_number / self.batch_size)))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        array_list_temp = [self.img_array[k,:,:,0] for k in indexes]

        # Generate data
        X = self.__data_generation(array_list_temp)

        return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.img_number)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, array_list_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros((self.batch_size, self.dim[0], self.dim[1], self.n_channels), 'uint16')

        if self.normalize_to_one:
            max_pixels = []

        # Generate data
        for i in range(self.batch_size):
            # Store sample
            try:
                tmpImg = array_list_temp[i]
                if self.normalize_to_one:
                    # tmpMedian = filters.median(tmpImg, self.selem)
                    tmpMax = np.max(tmpImg)
                    max_pixels.append(tmpMax)
            except IndexError:
                X = X[:i,...]
                break

            # ensure image is uint8
            # if tmpImg.dtype=="uint16":
                # tmpImg = tmpImg / 2**16 * 2**8
                # tmpImg = tmpImg.astype('uint8')

            # if self.normalize_to_one:
            #     with warnings.catch_warnings():
            #         warnings.simplefilter('ignore')
            #         medImg = median(tmpImg, self.selem)
            #         tmpImg = tmpImg/np.max(medImg)
            #         tmpImg[tmpImg > 1] = 1

            X[i,:,:,0] = tmpImg


        if self.normalize_to_one:
            channel_max = np.max(max_pixels) / (2**8 - 1)
            # print("Channel max: {}".format(channel_max))
            # print("Array max: {}".format(np.max(X)))
            X = X/channel_max
            # print("Normalized array max: {}".format(np.max(X)))
            X[X > 1] = 1

        return (X)

# class for image generation for predicting trap locations in phase-contrast images
class TrapSegmentationDataGenerator(utils.Sequence):
    'Generates data for Keras'
    def __init__(self, img_array, batch_size=32,
                 n_channels=1, normalize_to_one=False, shuffle=False):
        'Initialization'
        self.dim = (img_array.shape[1], img_array.shape[2])
        self.img_number = img_array.shape[0]
        self.img_array = img_array
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()
        self.normalize_to_one = normalize_to_one
        if normalize_to_one:
            self.selem = morphology.disk(3)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(self.img_number / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        array_list_temp = [self.img_array[k,:,:,0] for k in indexes]

        # Generate data
        X = self.__data_generation(array_list_temp)

        return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.img_number)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, array_list_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros((self.batch_size, self.dim[0], self.dim[1], self.n_channels))

        # Generate data
        for i in range(self.batch_size):
            # Store sample
            try:
                tmpImg = array_list_temp[i]
            except IndexError:
                X = X[:i,...]
                break
            if self.normalize_to_one:
                medImg = median(tmpImg, self.selem)
                tmpImg = medImg/np.max(medImg)

            X[i,:,:,0] = tmpImg

        return (X)


# class for image generation for classifying traps as good, empty, out-of-focus, or defective
class TrapKymographPredictionDataGenerator(utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_fileNames, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_fileNames = list_fileNames
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.list_fileNames) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_fileNames_temp = [self.list_fileNames[k] for k in indexes]

        # Generate data
        X = self.__data_generation(list_fileNames_temp)

        return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_fileNames))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_fileNames_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros((self.batch_size, self.dim[0], self.dim[1], self.n_channels))

        # Generate data
        for i, fName in enumerate(list_fileNames_temp):
            # Store sample
            tmpImg = io.imread(fName)
            tmpImgShape = tmpImg.shape
            if tmpImgShape[0] < self.dim[0]:
                t_end = tmpImgShape[0]
            else:
                t_end = self.dim[0]
            X[i,:t_end,:,:] = np.expand_dims(tmpImg[:t_end,:,tmpImg.shape[-1]//2], axis=-1)

        return X

def absolute_diff(y_true, y_pred):
    y_true_sum = K.sum(y_true)
    y_pred_sum = K.sum(y_pred)
    diff = K.abs(y_pred_sum - y_true_sum)/tf.to_float(tf.size(y_true))
    return diff

def all_loss(y_true, y_pred):
    loss = losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred) + absolute_diff(y_true, y_pred)
    return loss

def absolute_dice_loss(y_true, y_pred):
    loss = dice_loss(y_true, y_pred) + absolute_diff(y_true, y_pred)
    return loss

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def f2_m(y_true, y_pred, beta=2):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    numer = (1+beta**2)*recall*precision
    denom =  recall + (beta**2)*precision + K.epsilon()
    return numer/denom

def f_precision_m(y_true, y_pred, beta=0.5):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    numer = (1+beta**2)*recall*precision
    denom =  recall + (beta**2)*precision + K.epsilon()
    return numer/denom

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
    for peak_id, spec in six.iteritems(specs[fov_id]):
        if spec == 1: # 1 means analyze
            ana_peak_ids.append(peak_id)
    ana_peak_ids = sorted(ana_peak_ids) # sort for repeatability

    information('Creating lineage for FOV %d with %d channels.' % (fov_id, len(ana_peak_ids)))

    # just break if there are no peaks to analize
    if not ana_peak_ids:
        # returning empty dictionary will add nothing to current cells dictionary
        return {}

    # This is a list of tuples (fov_id, peak_id) to send to the Pool command
    fov_and_peak_ids_list = [(fov_id, peak_id) for peak_id in ana_peak_ids]

    # set up multiprocessing pool. will complete pool before going on
    pool = Pool(processes=params['num_analyzers'])

    # create the lineages for each peak individually
    # the output is a list of dictionaries
    lineages = pool.map(make_lineage_chnl_stack, fov_and_peak_ids_list, chunksize=8)

    pool.close() # tells the process nothing more will be added.
    pool.join() # blocks script until everything has been processed and workers exit

    # This is the non-parallelized version (useful for debug)
    # lineages = []
    # for fov_and_peak_ids in fov_and_peak_ids_list:
    #     lineages.append(make_lineage_chnl_stack(fov_and_peak_ids))

    # combine all dictionaries into one dictionary
    Cells = {} # create dictionary to hold all information
    for cell_dict in lineages: # for all the other dictionaries in the list
        Cells.update(cell_dict) # updates Cells with the entries in cell_dict

    return Cells

# get number of cells in each frame and total number of pairwise interactions
def get_cell_counts(regionprops_list):

    cell_count_list = [len(time_regions) for time_regions in regionprops_list]
    interaction_count_list = []

    for i,cell_count in enumerate(cell_count_list):
        if i+1 == len(cell_count_list):
            break
        interaction_count_list.append(cell_count*cell_count_list[i+1])

    total_cells = np.sum(cell_count_list)
    total_interactions = np.sum(interaction_count_list)

    return(total_cells, total_interactions, cell_count_list, interaction_count_list)

# get cells' information for track prediction
def gather_interactions_and_events(regionprops_list):

    total_cells, total_interactions, cell_count_list, interaction_count_list = get_cell_counts(regionprops_list)

    # instantiate an array with a 2x4 array for each pair of cells'
    #   min_y, max_y, centroid_y, and area
    # in reality it would be much, much more efficient to
    #   look this information up in the data generator at run time
    #   for now, this will work
    pairwise_cell_data = np.zeros((total_interactions,2,5,1))

    # make a dictionary, the keys of which will be row indices so that we
    #   can quickly look up which timepoints/cells correspond to which
    #   rows of our model's ouput
    pairwise_cell_lookup = {}

    # populate arrays
    interaction_count = 0
    cell_count = 0

    for frame, frame_regions in enumerate(regionprops_list):

        for region in frame_regions:

            cell_label = region.label
            y,x = region.centroid
            bbox = region.bbox
            orientation = region.orientation
            min_y = bbox[0]
            max_y = bbox[2]
            area = region.area
            cell_label = region.label
            cell_info = (min_y, max_y, y, area, orientation)
            cell_count += 1

            try:
                frame_plus_one_regions = regionprops_list[frame+1]
            except IndexError as e:
                # print(e)
                break

            for region_plus_one in frame_plus_one_regions:

                paired_cell_label = region_plus_one.label
                y,x = region_plus_one.centroid
                bbox = region_plus_one.bbox
                min_y = bbox[0]
                max_y = bbox[2]
                area = region_plus_one.area
                paired_cell_label = region_plus_one.label

                pairwise_cell_data[interaction_count,0,:,0] = cell_info
                pairwise_cell_data[interaction_count,1,:,0] = (min_y, max_y, y, area, orientation)

                pairwise_cell_lookup[interaction_count] = {'frame':frame, 'cell_label':cell_label, 'paired_cell_label':paired_cell_label}

                interaction_count += 1

    return(pairwise_cell_data, pairwise_cell_lookup)

# look up which cells are interacting according to the track model
def cell_interaction_lookup(predictions, lookup_table):
    '''
    Accepts prediction matrix and
    '''
    frame = []
    cell_label = []
    paired_cell_label = []
    interaction_type = []

    # loop over rows of predictions
    for row_index in range(predictions.shape[0]):

        row_predictions = predictions[row_index]
        row_relationship = np.where(row_predictions > 0.95)[0]
        if row_relationship.size == 0:
            continue
        elif row_relationship[0] == 3:
            continue
        elif row_relationship[0] == 0:
            interaction_type.append('migration')
        elif row_relationship[0] == 1:
            interaction_type.append('child')
        elif row_relationship[0] == 2:
            interaction_type.append('false_join')

        frame.append(lookup_table[row_index]['frame'])
        cell_label.append(lookup_table[row_index]['cell_label'])
        paired_cell_label.append(lookup_table[row_index]['paired_cell_label'])

    track_df = pd.DataFrame(data={'frame':frame,
                              'cell_label':cell_label,
                              'paired_cell_label':paired_cell_label,
                              'interaction_type':interaction_type})
    return(track_df)

def get_tracking_model_dict():

    model_dict = {}

    if not 'migrate_model' in model_dict:
        model_dict['migrate_model'] = models.load_model(params['tracking']['migrate_model'],
                                                                    custom_objects={'all_loss':all_loss,
                                                                        'f2_m':f2_m})
    if not 'child_model' in model_dict:
        model_dict['child_model'] = models.load_model(params['tracking']['child_model'],
                                        custom_objects={'bce_dice_loss':bce_dice_loss,
                                                                    'f2_m':f2_m})
    if not 'appear_model' in model_dict:
        model_dict['appear_model'] = models.load_model(params['tracking']['appear_model'],
                                        custom_objects={'all_loss':all_loss,
                                                                    'f2_m':f2_m})
    if not 'die_model' in model_dict:
        model_dict['die_model'] = models.load_model(params['tracking']['die_model'],
                                        custom_objects={'all_loss':all_loss,
                                                                    'f2_m':f2_m})
    if not 'disappear_model' in model_dict:
        model_dict['disappear_model'] = models.load_model(params['tracking']['disappear_model'],
                                        custom_objects={'all_loss':all_loss,
                                                                    'f2_m':f2_m})
    if not 'born_model' in model_dict:
        model_dict['born_model'] = models.load_model(params['tracking']['born_model'],
                                        custom_objects={'all_loss':all_loss,
                                                                    'f2_m':f2_m})
    if not 'zero_cell_model' in model_dict:
        model_dict['zero_cell_model'] = models.load_model(params['tracking']['zero_cell_model'],
                                        custom_objects={'absolute_dice_loss':absolute_dice_loss,
                                                                    'f2_m':f2_m})
    if not 'one_cell_model' in model_dict:
        model_dict['one_cell_model'] = models.load_model(params['tracking']['one_cell_model'],
                                        custom_objects={'bce_dice_loss':bce_dice_loss,
                                                                    'f2_m':f2_m})
    if not 'two_cell_model' in model_dict:
        model_dict['two_cell_model'] = models.load_model(params['tracking']['two_cell_model'],
                                        custom_objects={'all_loss':all_loss,
                                                                    'f2_m':f2_m})
    if not 'geq_three_cell_model' in model_dict:
        model_dict['geq_three_cell_model'] = models.load_model(params['tracking']['geq_three_cell_model'],
                                        custom_objects={'bce_dice_loss':bce_dice_loss,
                                                                    'f2_m':f2_m})

    return(model_dict)

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
    lost_cell_time = params['track']['lost_cell_time']
    # only cells with y positions below this value will recieve the honor of becoming new
    # cells, unless they are daughters of current cells
    new_cell_y_cutoff = params['track']['new_cell_y_cutoff']
    # only regions with labels less than or equal to this value will be considered to start cells
    new_cell_region_cutoff = params['track']['new_cell_region_cutoff']

    # get the specific ids from the tuple
    fov_id, peak_id = fov_and_peak_id

    # start time is the first time point for this series of TIFFs.
    start_time_index = min(params['time_table'][fov_id].keys())

    information('Creating lineage for FOV %d, channel %d.' % (fov_id, peak_id))

    # load segmented data
    image_data_seg = load_stack(fov_id, peak_id, color=params['track']['seg_img'])
    # image_data_seg = load_stack(fov_id, peak_id, color='seg')

    # Calculate all data for all time points.
    # this list will be length of the number of time points
    regions_by_time = [regionprops(label_image=timepoint) for timepoint in image_data_seg] # removed coordinates='xy'

    # Set up data structures.
    Cells = {} # Dict that holds all the cell objects, divided and undivided
    cell_leaves = [] # cell ids of the current leaves of the growing lineage tree

    # go through regions by timepoint and build lineages
    # timepoints start with the index of the first image
    for t, regions in enumerate(regions_by_time, start=start_time_index):
        # if there are cell leaves who are still waiting to be linked, but
        # too much time has passed, remove them.
        for leaf_id in cell_leaves:
            if t - Cells[leaf_id].times[-1] > lost_cell_time:
                cell_leaves.remove(leaf_id)

        # make all the regions leaves if there are no current leaves
        if not cell_leaves:
            for region in regions:
                if region.centroid[0] < new_cell_y_cutoff and region.label <= new_cell_region_cutoff:
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
            current_leaf_positions = [(leaf_id, Cells[leaf_id].centroids[-1][0]) for leaf_id in cell_leaves]

            # go through regions, they will come off in Y position order
            for r, region in enumerate(regions):
                # create tuple which is cell_id of closest leaf, distance
                current_closest = (None, float('inf'))

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
            for leaf_id, region_links in six.iteritems(leaf_region_map):
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
                        if region.centroid[0] < new_cell_y_cutoff and region.label <= new_cell_region_cutoff:
                            cell_id = create_cell_id(region, t, peak_id, fov_id)
                            Cells[cell_id] = Cell(cell_id, region, t, parent_id=None)
                            cell_leaves.append(cell_id) # add to leaves
                        else:
                            # since the regions are ordered, none of the remaining will pass
                            break

            ### iterate over the leaves, looking to see what regions connect to them.
            for leaf_id, region_links in six.iteritems(leaf_region_map):

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

                        # remove mother from current leaves
                        cell_leaves.remove(leaf_id)

                        # add the daughter ids to list of current leaves if they pass cutoffs
                        if region1.centroid[0] < new_cell_y_cutoff and region1.label <= new_cell_region_cutoff:
                            cell_leaves.append(daughter1_id)

                        if region2.centroid[0] < new_cell_y_cutoff and region2.label <= new_cell_region_cutoff:
                            cell_leaves.append(daughter2_id)

                    # 1 means that daughter 1 is just a continuation of the mother
                    # The other region should be a leaf it passes the requirements
                    elif check_division_result == 1:
                        Cells[leaf_id].grow(region1, t)

                        if region2.centroid[0] < new_cell_y_cutoff and region2.label <= new_cell_region_cutoff:
                            cell_id = create_cell_id(region2, t, peak_id, fov_id)
                            Cells[cell_id] = Cell(cell_id, region2, t, parent_id=None)
                            cell_leaves.append(cell_id) # add to leaves

                    # ditto for 2
                    elif check_division_result == 2:
                        Cells[leaf_id].grow(region2, t)

                        if region1.centroid[0] < new_cell_y_cutoff and region1.label <=     new_cell_region_cutoff:
                            cell_id = create_cell_id(region1, t, peak_id, fov_id)
                            Cells[cell_id] = Cell(cell_id, region1, t, parent_id=None)
                            cell_leaves.append(cell_id) # add to leaves

    # return the dictionary with all the cells
    return Cells

### Cell class and related functions

# this is the object that holds all information for a detection
class Detection():
    '''
    The Detection is a single detection in a single frame.
    '''

    # initialize (birth) the cell
    def __init__(self, detection_id, region, t):
        '''The detection must be given a unique detection_id and passed the region
        information from the segmentation

        Parameters
        __________

        detection_id : str
            detection_id is a string in the form fXpXtXrX
            f is 3 digit FOV number
            p is 4 digit peak number
            t is 4 digit time point
            r is region label for that segmentation
            Use the function create_detection_id to return a proper string.

        region : region properties object
            Information about the labeled region from
            skimage.measure.regionprops()

            '''

        # create all the attributes
        # id
        self.id = detection_id

        # identification convenience
        self.fov = int(detection_id.split('f')[1].split('p')[0])
        self.peak = int(detection_id.split('p')[1].split('t')[0])
        self.t = t

        self.cell_count = 1

        # self.abs_times = [params['time_table'][self.fov][t]] # elapsed time in seconds
        if region is not None:
            self.label = region.label
            self.bbox = region.bbox
            self.area = region.area

            # calculating cell length and width by using Feret Diamter. These values are in pixels
            length_tmp, width_tmp = feretdiameter(region)
            if length_tmp == None:
                mm3.warning('feretdiameter() failed for ' + self.id + ' at t=' + str(t) + '.')
            self.length = length_tmp
            self.width = width_tmp

            # calculate cell volume as cylinder plus hemispherical ends (sphere). Unit is px^3
            self.volume = (length_tmp - width_tmp) * np.pi * (width_tmp/2)**2 + (4/3) * np.pi * (width_tmp/2)**3

            # angle of the fit elipsoid and centroid location
            self.orientation = region.orientation
            self.centroid = region.centroid

        else:
            self.label = None
            self.bbox = None
            self.area = None

            # calculating cell length and width by using Feret Diamter. These values are in pixels
            length_tmp, width_tmp = (None, None)
            self.length = None
            self.width = None

            # calculate cell volume as cylinder plus hemispherical ends (sphere). Unit is px^3
            self.volume = None

            # angle of the fit elipsoid and centroid location
            self.orientation = None
            self.centroid = None

# this is the object that holds all information for a cell
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
        self.abs_times = [params['time_table'][self.fov][t]] # elapsed time in seconds
        self.labels = [region.label]
        self.bboxes = [region.bbox]
        self.areas = [region.area]

        # calculating cell length and width by using Feret Diamter. These values are in pixels
        length_tmp, width_tmp = feretdiameter(region)
        if length_tmp == None:
            mm3.warning('feretdiameter() failed for ' + self.id + ' at t=' + str(t) + '.')
        self.lengths = [length_tmp]
        self.widths = [width_tmp]

        # calculate cell volume as cylinder plus hemispherical ends (sphere). Unit is px^3
        self.volumes = [(length_tmp - width_tmp) * np.pi * (width_tmp/2)**2 +
                       (4/3) * np.pi * (width_tmp/2)**3]

        # angle of the fit elipsoid and centroid location
        self.orientations = [region.orientation]
        self.centroids = [region.centroid]

        # these are special datatype, as they include information from the daugthers for division
        # computed upon division
        self.times_w_div = None
        self.lengths_w_div = None
        self.widths_w_div = None

        # this information is the "production" information that
        # we want to extract at the end. Some of this is for convenience.
        # This is only filled out if a cell divides.
        self.sb = None # in um
        self.sd = None # this should be combined lengths of daughters, in um
        self.delta = None
        self.tau = None
        self.elong_rate = None
        self.septum_position = None
        self.width = None
        self.death = None

    def grow(self, region, t):
        '''Append data from a region to this cell.
        use cell.times[-1] to get most current value'''

        self.times.append(t)
        self.abs_times.append(params['time_table'][self.fov][t])
        self.labels.append(region.label)
        self.bboxes.append(region.bbox)
        self.areas.append(region.area)

        #calculating cell length and width by using Feret Diamter
        length_tmp, width_tmp = feretdiameter(region)
        if length_tmp == None:
            mm3.warning('feretdiameter() failed for ' + self.id + ' at t=' + str(t) + '.')
        self.lengths.append(length_tmp)
        self.widths.append(width_tmp)
        self.volumes.append((length_tmp - width_tmp) * np.pi * (width_tmp/2)**2 +
                            (4/3) * np.pi * (width_tmp/2)**3)

        self.orientations.append(region.orientation)
        self.centroids.append(region.centroid)

    def die(self, region, t):
        '''
        Annotate cell as dying from current t to next t.
        '''
        self.death = t

    def divide(self, daughter1, daughter2, t):
        '''Divide the cell and update stats.
        daugther1 and daugther2 are instances of the Cell class.
        daughter1 is the daugther closer to the closed end.'''

        # put the daugther ids into the cell
        self.daughters = [daughter1.id, daughter2.id]

        # give this guy a division time
        self.division_time = daughter1.birth_time

        # update times
        self.times_w_div = self.times + [self.division_time]
        self.abs_times.append(params['time_table'][self.fov][self.division_time])

        # flesh out the stats for this cell
        # size at birth
        self.sb = self.lengths[0] * params['pxl2um']

        # force the division length to be the combined lengths of the daughters
        self.sd = (daughter1.lengths[0] + daughter2.lengths[0]) * params['pxl2um']

        # delta is here for convenience
        self.delta = self.sd - self.sb

        # generation time. Use more accurate times and convert to minutes
        self.tau = np.float64((self.abs_times[-1] - self.abs_times[0]) / 60.0)

        # include the data points from the daughters
        self.lengths_w_div = [l * params['pxl2um'] for l in self.lengths] + [self.sd]
        self.widths_w_div = [w * params['pxl2um'] for w in self.widths] + [((daughter1.widths[0] + daughter2.widths[0])/2) * params['pxl2um']]

        # volumes for all timepoints, in um^3
        self.volumes_w_div = []
        for i in range(len(self.lengths_w_div)):
            self.volumes_w_div.append((self.lengths_w_div[i] - self.widths_w_div[i]) *
                                       np.pi * (self.widths_w_div[i]/2)**2 +
                                       (4/3) * np.pi * (self.widths_w_div[i]/2)**3)

        # calculate elongation rate.
        try:
            times = np.float64((np.array(self.abs_times) - self.abs_times[0]) / 60.0)
            log_lengths = np.float64(np.log(self.lengths_w_div))
            p = np.polyfit(times, log_lengths, 1) # this wants float64
            self.elong_rate = p[0] * 60.0 # convert to hours
        except:
            self.elong_rate = np.float64('NaN')
            warning('Elongation rate calculate failed for {}.'.format(self.id))

        # calculate the septum position as a number between 0 and 1
        # which indicates the size of daughter closer to the closed end
        # compared to the total size
        self.septum_position = daughter1.lengths[0] / (daughter1.lengths[0] + daughter2.lengths[0])

        # calculate single width over cell's life
        self.width = np.mean(self.widths_w_div)

        # convert data to smaller floats. No need for float64
        # see https://docs.scipy.org/doc/numpy-1.13.0/user/basics.types.html
        convert_to = 'float16' # numpy datatype to convert to

        self.sb = self.sb.astype(convert_to)
        self.sd = self.sd.astype(convert_to)
        self.delta = self.delta.astype(convert_to)
        self.elong_rate = self.elong_rate.astype(convert_to)
        self.tau = self.tau.astype(convert_to)
        self.septum_position = self.septum_position.astype(convert_to)
        self.width = self.width.astype(convert_to)

        self.lengths = [length.astype(convert_to) for length in self.lengths]
        self.lengths_w_div = [length.astype(convert_to) for length in self.lengths_w_div]
        self.widths = [width.astype(convert_to) for width in self.widths]
        self.widths_w_div = [width.astype(convert_to) for width in self.widths_w_div]
        self.volumes = [vol.astype(convert_to) for vol in self.volumes]
        self.volumes_w_div = [vol.astype(convert_to) for vol in self.volumes_w_div]
        # note the float16 is hardcoded here
        self.orientations = [np.float16(orientation) for orientation in self.orientations]
        self.centroids = [(y.astype(convert_to), x.astype(convert_to)) for y, x in self.centroids]

    def print_info(self):
        '''prints information about the cell'''
        print('id = %s' % self.id)
        print('times = {}'.format(', '.join('{}'.format(t) for t in self.times)))
        print('lengths = {}'.format(', '.join('{:.2f}'.format(l) for l in self.lengths)))

class CellTree():

    def __init__(self):
        self.cells = {}
        self.scores = [] # probably needs to be different
        self.score = 0
        self.cell_id_list = []

    def add_cell(self, cell):
        self.cells[cell.id] = cell
        self.cell_id_list.append(cell.id)
        self.cell_id_list.sort()

    def update_score(self):
        pass

    def get_cell(self, cell_id):
        return(self.cells[cell_id])

    def get_top_from_cell(self, cell_id):
        pass

# this is the object that holds all information for a cell
class CellFromGraph():
    '''
    The CellFromGraph class is one cell that has been born.
    It is not neccesarily a cell that has divided.
    '''

    # initialize (birth) the cell
    def __init__(self, cell_id, region, t, parent=None):
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
        self.birth_label = int(region.label)
        self.regions = [region]

        # parent is a CellFromGraph object, can be None
        self.parent = parent

        # daughters is updated when cell divides
        # if this is none then the cell did not divide
        self.daughters = None

        # birth and division time
        self.birth_time = t
        self.division_time = None # filled out if cell divides

        # the following information is on a per timepoint basis
        self.times = [t]
        self.abs_times = [params['time_table'][self.fov][t]] # elapsed time in seconds
        self.labels = [region.label]
        self.bboxes = [region.bbox]
        self.areas = [region.area]

        # calculating cell length and width by using Feret Diamter. These values are in pixels
        length_tmp, width_tmp = feretdiameter(region)
        if length_tmp == None:
            mm3.warning('feretdiameter() failed for ' + self.id + ' at t=' + str(t) + '.')
        self.lengths = [length_tmp]
        self.widths = [width_tmp]

        # calculate cell volume as cylinder plus hemispherical ends (sphere). Unit is px^3
        self.volumes = [(length_tmp - width_tmp) * np.pi * (width_tmp/2)**2 +
                       (4/3) * np.pi * (width_tmp/2)**3]

        # angle of the fit elipsoid and centroid location
        self.orientations = [region.orientation]
        self.centroids = [region.centroid]

        # these are special datatype, as they include information from the daugthers for division
        # computed upon division
        self.times_w_div = None
        self.lengths_w_div = None
        self.widths_w_div = None

        # this information is the "production" information that
        # we want to extract at the end. Some of this is for convenience.
        # This is only filled out if a cell divides.
        self.sb = None # in um
        self.sd = None # this should be combined lengths of daughters, in um
        self.delta = None
        self.tau = None
        self.elong_rate = None
        self.septum_position = None
        self.width = None
        self.death = None
        self.disappear = None
        self.area_mean_fluorescence = {}
        self.volume_mean_fluorescence = {}
        self.total_fluorescence = {}
        self.foci = {}

    def __len__(self):
        return(len(self.times))

    def add_parent(self, parent):
        self.parent = parent

    def grow(self, region, t):
        '''Append data from a region to this cell.
        use cell.times[-1] to get most current value'''

        self.times.append(t)
        self.abs_times.append(params['time_table'][self.fov][t])
        self.labels.append(region.label)
        self.bboxes.append(region.bbox)
        self.areas.append(region.area)
        self.regions.append(region)

        #calculating cell length and width by using Feret Diamter
        length_tmp, width_tmp = feretdiameter(region)
        if length_tmp == None:
            mm3.warning('feretdiameter() failed for ' + self.id + ' at t=' + str(t) + '.')
        self.lengths.append(length_tmp)
        self.widths.append(width_tmp)
        self.volumes.append((length_tmp - width_tmp) * np.pi * (width_tmp/2)**2 +
                            (4/3) * np.pi * (width_tmp/2)**3)

        self.orientations.append(region.orientation)
        self.centroids.append(region.centroid)

    def die(self, region, t):
        '''
        Annotate cell as dying from current t to next t.
        '''
        self.death = t

    def disappears(self, region, t):
        '''
        Annotate cell as disappearing from current t to next t.
        '''
        self.disappear = t

    def add_daughter(self, daughter, t):

        if self.daughters is None:
            self.daughters = [daughter]
        else:
            self.daughters.append(daughter)
            assert len(self.daughters) < 3, "Too many daughter cells in cell {}".format(self.id)
            # sort daughters by y position, with smaller y-value first.
            # this will cause the daughter closer to the closed end of the trap to be listed first.
            self.daughters.sort(key=lambda cell: cell.centroids[0][0])
            self.divide(t)

    def divide(self, t):
        '''Divide the cell and update stats.
        daughter1 is the daugther closer to the closed end.'''

        # put the daugther ids into the cell
        # self.daughters = [daughter1.id, daughter2.id]

        # give this guy a division time
        self.division_time = self.daughters[0].birth_time

        # update times
        self.times_w_div = self.times + [self.division_time]
        self.abs_times.append(params['time_table'][self.fov][self.division_time])

        # flesh out the stats for this cell
        # size at birth
        self.sb = self.lengths[0] * params['pxl2um']

        # force the division length to be the combined lengths of the daughters
        self.sd = (self.daughters[0].lengths[0] + self.daughters[1].lengths[0]) * params['pxl2um']

        # delta is here for convenience
        self.delta = self.sd - self.sb

        # generation time. Use more accurate times and convert to minutes
        self.tau = np.float64((self.abs_times[-1] - self.abs_times[0]) / 60.0)

        # include the data points from the daughters
        self.lengths_w_div = [l * params['pxl2um'] for l in self.lengths] + [self.sd]
        self.widths_w_div = [w * params['pxl2um'] for w in self.widths] + [((self.daughters[0].widths[0] + self.daughters[1].widths[0])/2) * params['pxl2um']]

        # volumes for all timepoints, in um^3
        self.volumes_w_div = []
        for i in range(len(self.lengths_w_div)):
            self.volumes_w_div.append((self.lengths_w_div[i] - self.widths_w_div[i]) *
                                       np.pi * (self.widths_w_div[i]/2)**2 +
                                       (4/3) * np.pi * (self.widths_w_div[i]/2)**3)

        # calculate elongation rate.
        try:
            times = np.float64((np.array(self.abs_times) - self.abs_times[0]) / 60.0) # convert times to minutes
            log_lengths = np.float64(np.log(self.lengths_w_div))
            p = np.polyfit(times, log_lengths, 1) # this wants float64
            self.elong_rate = p[0] * 60.0 # convert to hours
        except:
            self.elong_rate = np.float64('NaN')
            warning('Elongation rate calculate failed for {}.'.format(self.id))

        # calculate the septum position as a number between 0 and 1
        # which indicates the size of daughter closer to the closed end
        # compared to the total size
        self.septum_position = self.daughters[0].lengths[0] / (self.daughters[0].lengths[0] + self.daughters[1].lengths[0])

        # calculate single width over cell's life
        self.width = np.mean(self.widths_w_div)

        # convert data to smaller floats. No need for float64
        # see https://docs.scipy.org/doc/numpy-1.13.0/user/basics.types.html
        convert_to = 'float16' # numpy datatype to convert to

        self.sb = self.sb.astype(convert_to)
        self.sd = self.sd.astype(convert_to)
        self.delta = self.delta.astype(convert_to)
        self.elong_rate = self.elong_rate.astype(convert_to)
        self.tau = self.tau.astype(convert_to)
        self.septum_position = self.septum_position.astype(convert_to)
        self.width = self.width.astype(convert_to)

        self.lengths = [length.astype(convert_to) for length in self.lengths]
        self.lengths_w_div = [length.astype(convert_to) for length in self.lengths_w_div]
        self.widths = [width.astype(convert_to) for width in self.widths]
        self.widths_w_div = [width.astype(convert_to) for width in self.widths_w_div]
        self.volumes = [vol.astype(convert_to) for vol in self.volumes]
        self.volumes_w_div = [vol.astype(convert_to) for vol in self.volumes_w_div]
        # note the float16 is hardcoded here
        self.orientations = [np.float16(orientation) for orientation in self.orientations]
        self.centroids = [(y.astype(convert_to), x.astype(convert_to)) for y, x in self.centroids]

    def add_focus(self, focus, t):
        '''Adds a focus to the cell. See function foci_info_unet'''
        self.foci[focus.id] = focus

    def print_info(self):
        '''prints information about the cell'''
        print('id = %s' % self.id)
        print('times = {}'.format(', '.join('{}'.format(t) for t in self.times)))
        print('lengths = {}'.format(', '.join('{:.2f}'.format(l) for l in self.lengths)))
        if self.daughters is not None:
            print('daughters = {}'.format(', '.join('{}'.format(daughter.id) for daughter in self.daughters)))
        if self.parent is not None:
            print('parent = {}'.format(self.parent.id))

    def make_wide_df(self):

        data = {}
        data['id'] = self.id
        data['fov'] = self.fov
        data['trap'] = self.peak
        data['parent'] = self.parent
        data['child1'] = None
        data['child2'] = None
        data['division_time'] = self.division_time
        data['birth_label'] = self.birth_label
        data['birth_time'] = self.birth_time
        data['sb'] = self.sb
        data['sd'] = self.sd
        data['delta'] = self.delta
        data['tau'] = self.tau
        data['elong_rate'] = self.elong_rate
        data['septum_position'] = self.septum_position
        data['death'] = self.death
        data['disappear'] = self.disappear

        if self.daughters is not None:
            data['child1'] = self.daughters[0]

            if len(self.daughters) == 2:
                data['child2'] = self.daughters[1]

        df = pd.DataFrame(data, index=[self.id])
        return(df)

    def make_long_df(self):

        data = {}
        data['id'] = [self.id]*len(self.times)
        data['times'] = self.times
        data['length'] = self.lengths
        data['volume'] = self.volumes
        data['area'] = self.areas

        # if a cell divides then there is one extra value in abs_times
        if self.division_time is None:
            data['seconds'] = self.abs_times
        else:
            data['seconds'] = self.abs_times[:-1]

        # if there is fluorescence data, place it into the dataframe
        if len(self.area_mean_fluorescence.keys()) != 0:

            for fluorescence_channel in self.area_mean_fluorescence.keys():

                data['{}_area_mean_fluorescence'.format(fluorescence_channel)] = self.area_mean_fluorescence[fluorescence_channel]
                data['{}_volume_mean_fluorescence'.format(fluorescence_channel)] = self.volume_mean_fluorescence[fluorescence_channel]
                data['{}_total_fluorescence'.format(fluorescence_channel)] = self.total_fluorescence[fluorescence_channel]

        df = pd.DataFrame(data, index=data['id'])

        return(df)

# this is the object that holds all information for a fluorescent focus
# this class can eventually be used in focus tracking, much like the Cell class
# is used for cell tracking
class Focus():
    '''
    The Focus class holds information on fluorescent foci.
    A single focus can be present in multiple different cells.
    '''

    # initialize the focus
    def __init__(self,
                 cell,
                 region,
                 seg_img,
                 intensity_image,
                 t):
        '''The cell must be given a unique cell_id and passed the region
        information from the segmentation

        Parameters
        __________

        cell : a Cell object

        region : region properties object
            Information about the labeled region from
            skimage.measure.regionprops()

        seg_img : 2D numpy array
            Labelled image of cell segmentations

        intensity_image : 2D numpy array
            Fluorescence image with foci
        '''

        # create all the attributes
        # id
        focus_id = create_focus_id(region,
                                   t,
                                   cell.peak,
                                   cell.fov)
        self.id = focus_id

        # identification convenience
        self.appear_label = int(region.label)
        self.regions = [region]
        self.fov = cell.fov
        self.peak = cell.peak

        # cell is a CellFromGraph object
        # cells are added later using the .add_cell method
        self.cells = [cell]

        # daughters is updated when focus splits
        # if this is none then the focus did not split
        self.parent = None
        self.daughters = None

        # appearance and split time
        self.appear_time = t
        self.split_time = None # filled out if focus splits

        # the following information is on a per timepoint basis
        self.times = [t]
        self.abs_times = [params['time_table'][cell.fov][t]] # elapsed time in seconds
        self.labels = [region.label]
        self.bboxes = [region.bbox]
        self.areas = [region.area]

        # calculating focus length and width by using Feret Diamter.
        #   These values are in pixels
        # NOTE: in the future, update to straighten a focus an get straightened length/width
        # print(region)
        length_tmp = region.major_axis_length
        width_tmp = region.minor_axis_length
        # length_tmp, width_tmp = feretdiameter(region)
        # if length_tmp == None:
            # mm3.warning('feretdiameter() failed for ' + self.id + ' at t=' + str(t) + '.')
        self.lengths = [length_tmp]
        self.widths = [width_tmp]

        # calculate focus volume as cylinder plus hemispherical ends (sphere). Unit is px^3
        self.volumes = [(length_tmp - width_tmp) * np.pi * (width_tmp/2)**2 +
                       (4/3) * np.pi * (width_tmp/2)**3]

        # angle of the fit elipsoid and centroid location
        self.orientations = [region.orientation]
        self.centroids = [region.centroid]

        # special information for focci
        self.elong_rate = None
        self.disappear = None
        self.area_mean_fluorescence = []
        self.volume_mean_fluorescence = []
        self.total_fluorescence = []
        self.median_fluorescence = []
        self.sd_fluorescence = []
        self.disp_l = []
        self.disp_w = []

        self.calculate_fluorescence(seg_img, intensity_image, region)

    def __len__(self):
        return(len(self.times))

    def __str__(self):
        return(self.print_info(self))

    def add_cell(self, cell):
        self.cells.append(cell)

    def add_parent_focus(self, parent):
        self.parent = parent

    def grow(self,
             region,
             t,
             seg_img,
             intensity_image,
             current_cell):
        '''Append data from a region to this focus.
        use self.times[-1] to get most current value.'''

        if current_cell is not self.cells[-1]:
            self.add_cell(current_cell)

        self.times.append(t)
        self.abs_times.append(params['time_table'][self.cells[-1].fov][t])
        self.labels.append(region.label)
        self.bboxes.append(region.bbox)
        self.areas.append(region.area)
        self.regions.append(region)

        #calculating focus length and width by using Feret Diamter
        length_tmp = region.major_axis_length
        width_tmp = region.minor_axis_length
        # length_tmp, width_tmp = feretdiameter(region)
        # if length_tmp == None:
            # mm3.warning('feretdiameter() failed for ' + self.id + ' at t=' + str(t) + '.')
        self.lengths.append(length_tmp)
        self.widths.append(width_tmp)
        self.volumes.append((length_tmp - width_tmp) * np.pi * (width_tmp/2)**2 +
                            (4/3) * np.pi * (width_tmp/2)**3)

        self.orientations.append(region.orientation)
        self.centroids.append(region.centroid)

        self.calculate_fluorescence(seg_img, intensity_image, region)

    def calculate_fluorescence(self,
                               seg_img,
                               intensity_image,
                               region):

        total_fluor = np.sum(intensity_image[seg_img == region.label])
        self.total_fluorescence.append(total_fluor)
        self.area_mean_fluorescence.append(total_fluor/self.areas[-1])
        self.volume_mean_fluorescence.append(total_fluor/self.volumes[-1])
        self.median_fluorescence.append(np.median(intensity_image[seg_img == region.label]))
        self.sd_fluorescence.append(np.std(intensity_image[seg_img == region.label]))

        # get the focus' displacement from center of cell
        # find x and y position relative to the whole image (convert from small box)

        # calculate distance of foci from middle of cell (scikit image)
        orientation = region.orientation
        if orientation < 0:
            orientation = np.pi+orientation

        cell_idx = self.cells[-1].times.index(self.times[-1]) # final time in self.times is current time
        cell_centroid = self.cells[-1].centroids[cell_idx]
        focus_centroid = region.centroid
        disp_y = (focus_centroid[0]-cell_centroid[0])*np.sin(orientation) - (focus_centroid[1]-cell_centroid[1])*np.cos(orientation)
        disp_x = (focus_centroid[0]-cell_centroid[0])*np.cos(orientation) + (focus_centroid[1]-cell_centroid[1])*np.sin(orientation)

        # append foci information to the list
        self.disp_l = np.append(self.disp_l, disp_y)
        self.disp_w = np.append(self.disp_w, disp_x)

    def disappears(self, region, t):
        '''
        Annotate focus as disappearing from current t to next t.
        '''
        self.disappear = t

    def add_daughter(self, daughter, t):

        if self.daughters is None:
            self.daughters = [daughter]
        else:
            self.daughters.append(daughter)
            # sort daughters by y position, with smaller y-value first.
            # this will cause the daughter closer to the closed end of the trap to be listed first.
            self.daughters.sort(key=lambda focus: focus.centroids[0][0])
            self.divide(t)

    def divide(self, t):
        '''Divide the cell and update stats.
        daughter1 is the daugther closer to the closed end.'''

        # put the daugther ids into the cell
        # self.daughters = [daughter1.id, daughter2.id]

        # give this guy a division time
        self.split_time = self.daughters[0].appear_time

        # convert data to smaller floats. No need for float64
        # see https://docs.scipy.org/doc/numpy-1.13.0/user/basics.types.html
        convert_to = 'float16' # numpy datatype to convert to

        self.lengths = [length.astype(convert_to) for length in self.lengths]
        self.widths = [width.astype(convert_to) for width in self.widths]
        self.volumes = [vol.astype(convert_to) for vol in self.volumes]
        # note the float16 is hardcoded here
        self.orientations = [np.float16(orientation) for orientation in self.orientations]
        self.centroids = [(y.astype(convert_to), x.astype(convert_to)) for y, x in self.centroids]

    def print_info(self):
        '''prints information about the focus'''
        print('id = %s' % self.id)
        print('times = {}'.format(', '.join('{}'.format(t) for t in self.times)))
        print('lengths = {}'.format(', '.join('{:.2f}'.format(l) for l in self.lengths)))
        if self.daughters is not None:
            print('daughters = {}'.format(', '.join('{}'.format(daughter.id) for daughter in self.daughters)))
        if self.cells is not None:
            print('cells = {}'.format([cell.id for cell in self.cells]))

    def make_wide_df(self):

        data = {}
        data['id'] = self.id
        data['cells'] = self.cells
        data['parent'] = self.parent
        data['child1'] = None
        data['child2'] = None
        # data['division_time'] = self.division_time
        data['appear_label'] = self.appear_label
        data['appear_time'] = self.appear_time
        data['disappear'] = self.disappear

        if self.daughters is not None:
            data['child1'] = self.daughters[0]

            if len(self.daughters) == 2:
                data['child2'] = self.daughters[1]

        df = pd.DataFrame(data, index=[self.id])
        return(df)

    def make_long_df(self):

        data = {}
        data['id'] = [self.id]*len(self.times)
        data['time'] = self.times
        # data['cell'] = self.cells
        data['length'] = self.lengths
        data['volume'] = self.volumes
        data['area'] = self.areas
        data['seconds'] = self.abs_times
        data['area_mean_fluorescence'] = self.area_mean_fluorescence
        data['volume_mean_fluorescence'] = self.volume_mean_fluorescence
        data['total_fluorescence'] = self.total_fluorescence
        data['median_fluorescence'] = self.median_fluorescence
        data['sd_fluorescence'] = self.sd_fluorescence
        data['disp_l'] = self.disp_l
        data['disp_w'] = self.disp_w

        # print(data['id'])

        df = pd.DataFrame(data, index=data['id'])

        return(df)

class PredictTrackDataGenerator(utils.Sequence):
    '''Generates data for running tracking class preditions
    Input is a stack of labeled images'''
    def __init__(self,
                 data,
                 batch_size=32,
                 dim=(4,5,9)):

        'Initialization'
        self.batch_size = batch_size
        self.data = data
        self.dim = dim
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate keys of the batch
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X = self.__data_generation(batch_indices)

        return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indices = np.arange(len(self.data))

    def __data_generation(self, batch_indices):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        # shape is (batch_size, max_cell_num, frame_num, cell_feature_num, 1)
        X = np.zeros((self.batch_size, self.dim[0], self.dim[1], self.dim[2], 1))

        # Generate data
        for idx in batch_indices:
            start_idx = idx-2
            end_idx = idx+3

#             print(start_idx, end_idx)
            if start_idx < 0:
                batch_frame_list = []
                for empty_idx in range(abs(start_idx)):
                    batch_frame_list.append([])
                batch_frame_list.extend(self.data[0:end_idx])

            elif end_idx > len(self.data):
                batch_frame_list = self.data[start_idx:len(self.data)+1]
                for empty_idx in range(abs(end_idx - len(self.data))):
                    batch_frame_list.extend([])

            else:
                batch_frame_list = self.data[start_idx:end_idx]

            for i,frame_region_list in enumerate(batch_frame_list):

                # shape is (max_cell_num, frame_num, cell_feature_num)
#                 tmp_x = np.zeros((self.dim[0], self.dim[1], self.dim[2]))

                if not frame_region_list:
                    continue

                for region_idx, region, in enumerate(frame_region_list):
                    y,x = region.centroid
                    bbox = region.bbox
                    orientation = region.orientation
                    min_y = bbox[0]
                    max_y = bbox[2]
                    min_x = bbox[1]
                    max_x = bbox[3]
                    area = region.area
                    length = region.major_axis_length
                    cell_label = region.label
                    cell_index = cell_label - 1
                    cell_info = (min_x, max_x, x, min_y, max_y, y, orientation, area, length)

                    if region_idx + 1 > self.dim[0]:
                        continue

                    # supplement tmp_x at (region_idx, )
#                     tmp_x[region_idx, i, :] = cell_info

                    X[idx, cell_index, i, :,0] = cell_info # tmp_x

        return X

def get_greatest_score_info(first_node, second_node, graph):
    '''A function that is useful for track linking
    '''
    score_names = [k for k in graph.get_edge_data(first_node, second_node).keys()]
    pred_scores = [val['score'] for k,val in graph.get_edge_data(first_node, second_node).items()]
    max_score_index = np.argmax(pred_scores)
    max_name = score_names[max_score_index]
    max_score = pred_scores[max_score_index]
    return(max_name, max_score)

def get_score_by_type(first_node, second_node, graph, score_type='child'):
    '''A function useful in track linking
    '''
    pred_score = graph.get_edge_data(first_node, second_node)[score_type]['score']
    return(pred_score)

def count_unvisited(G, experiment_name):
    count = 0
    for node_id in G.nodes:
        if node_id.startswith(experiment_name):
            if not G.nodes[node_id]['visited']:
                count += 1
    return(count)

def create_lineages_from_graph(graph,
                               graph_df,
                               fov_id,
                               peak_id,
                               ):
    '''
    This function iterates through nodes in a graph of detections
    to link the nodes as "CellFromGraph" objects, eventually
    leading to the ultimate goal of returning
    a CellTree object with each cell's information for the experiment.

    For now it ignores the number of cells in a detection and simply
    assumes a 1:1 relationship between detections and cell number.
    '''

    # iterate through all nodes in graph
    # graph_score = 0
    # track_dict = {}
    # tracks = CellTree()
    tracks = {}

    for node_id in graph.nodes:
        graph.nodes[node_id]['visited'] = False
    graph_df['visited'] = False
    num_unvisited = count_unvisited(graph, params['experiment_name'])

    while num_unvisited > 0:

        # which detection nodes are not yet visited
        unvisited_detection_nodes = graph_df[(~(graph_df.visited) & graph_df.node_id.str.startswith(params['experiment_name']))]
        # grab the first unvisited node_id from the dataframe
        prior_node_id = unvisited_detection_nodes.iloc[0,1]
        prior_node_time = graph.nodes[prior_node_id]['time']
        prior_node_region = graph.nodes[prior_node_id]['region']

        cell_id = create_cell_id(prior_node_region,
                                    prior_node_time,
                                    peak_id,
                                    fov_id,
                                    experiment_name=params['experiment_name'])

        current_cell = CellFromGraph(cell_id,
                                        prior_node_region,
                                        prior_node_time,
                                        parent=None)

        if not cell_id in tracks.keys():
            tracks[cell_id] = current_cell
        else:
            current_cell = tracks[cell_id]

        # for use later in establishing predecessors
        current_node_id = prior_node_id

        # set this detection's "visited" status to True in the graph and in the dataframe
        graph.nodes[prior_node_id]['visited'] = True
        graph_df.iloc[np.where(graph_df.node_id==prior_node_id)[0][0],3] = True

        # build current_track list to this detection's node
        current_track = collections.deque()
        current_track.append(current_node_id)
        predecessors_list = [k for k in graph.predecessors(prior_node_id)]
        unvisited_predecessors_list = [k for k in predecessors_list if not graph.nodes[k]['visited']]

        while len(unvisited_predecessors_list) != 0:

            # initialize a scores array to select highest score from the available options
            predecessor_scores = np.zeros(len(unvisited_predecessors_list))

            # populate array with scores
            for i in range(len(unvisited_predecessors_list)):
                predecessor_node_id = unvisited_predecessors_list[i]
                edge_type, edge_score = get_greatest_score_info(predecessor_node_id, current_node_id, graph)
                predecessor_scores[i] = edge_score

            # find highest score
            max_index = np.argmax(predecessor_scores)
            # grab the node_id corresponding to traversing the highest-scoring edge from the prior node
            current_node_id = unvisited_predecessors_list[max_index]
            current_track.appendleft(current_node_id)

            predecessors_list = [k for k in graph.predecessors(current_node_id)]
            unvisited_predecessors_list = [k for k in predecessors_list if not graph.nodes[k]['visited']]

        while prior_node_id is not 'B':

            # which nodes succeed our current node?
            successor_node_ids = [node_id for node_id in graph.successors(prior_node_id)]

            # keep only the potential successor detections that have not yet been visited
            unvisited_node_ids = []
            for i,successor_node_id in enumerate(successor_node_ids):

                # if it starts with params['experiment_name'], it is a detection node, and not born, appear, etc.
                if successor_node_id.startswith(params['experiment_name']):

                    # if it has been used in the cell track graph, i.e., if 'visited' is True,
                    #   move on. Otherwise, append to our list
                    if graph.nodes[successor_node_id]['visited']:
                        continue
                    else:
                        unvisited_node_ids.append(successor_node_id)

                # if it doesn't start with params['experiment_name'], it is a born, appear, etc., and should always be appended
                else:
                    unvisited_node_ids.append(successor_node_id)

            # initialize a scores array to select highest score from the available options
            successor_scores = np.zeros(len(unvisited_node_ids))
            successor_edge_types = []

            # populate array with scores
            for i in range(len(unvisited_node_ids)):
                successor_node_id = unvisited_node_ids[i]
                edge_type, edge_score = get_greatest_score_info(prior_node_id, successor_node_id, graph)
                successor_scores[i] = edge_score
                successor_edge_types.append(edge_type)

            # find highest score
            max_score = np.max(successor_scores)
            max_index = np.argmax(successor_scores)
            # grab the node_id corresponding to traversing the highest-scoring edge from the prior node
            next_node_id = unvisited_node_ids[max_index]
            max_edge_type = successor_edge_types[max_index]

            # if the max_score in successor_scores isn't greater than log(0.1), just make the cell disappear for now.
            if max_score < np.log(0.1):
                max_edge_type = 'disappear'
                next_node_id = [n_id for n_id in unvisited_node_ids if n_id.startswith('disappear')][0]

            # if this is a division event, add child node as a new cell,
            #   add the new cell as a daughter to current_cell,
            #   add current_cell as a parent to new cell.
            # Then, search for the second child cell, add it to current_cell, etc.
            if max_edge_type == 'child':

                new_cell_time = graph.nodes[next_node_id]['time']
                new_cell_region = graph.nodes[next_node_id]['region']
                new_cell_id = create_cell_id(new_cell_region,
                                             new_cell_time,
                                             peak_id,
                                             fov_id,
                                             experiment_name=params['experiment_name'])

                new_cell = CellFromGraph(new_cell_id,
                                         new_cell_region,
                                         new_cell_time,
                                         parent=current_cell)

                tracks[new_cell_id] = new_cell

                current_cell.add_daughter(new_cell, new_cell_time)

                # initialize a scores array to select highest score from the available options
                unvisited_detection_nodes = [unvisited_node_id for unvisited_node_id in unvisited_node_ids if unvisited_node_id.startswith(params['experiment_name'])]
                child_scores = np.zeros(len(unvisited_detection_nodes))

                # populate array with scores
                for i in range(len(unvisited_detection_nodes)):
                    successor_node_id = unvisited_detection_nodes[i]
                    if successor_node_id == next_node_id:
                        child_scores[i] = -np.inf
                        continue
                    child_score = get_score_by_type(prior_node_id, successor_node_id, graph, score_type='child')
                    child_scores[i] = child_score

                try:
                    second_daughter_score = np.max(child_scores)
                    # sometimes a second daughter doesn't exist: perhaps parent is at mouth of a trap and one
                    #  daughter is lost to the central channel at division time. In this case, do the following:
                    if second_daughter_score < np.log(0.5):
                        current_cell = new_cell

                    else:
                        second_daughter_index = np.argmax(child_scores)
                        # grab the node_id corresponding to traversing the highest-scoring edge from the prior node
                        other_daughter_node_id = unvisited_detection_nodes[second_daughter_index]

                        other_daughter_cell_time = graph.nodes[other_daughter_node_id]['time']
                        other_daughter_cell_region = graph.nodes[other_daughter_node_id]['region']
                        other_daughter_cell_id = create_cell_id(other_daughter_cell_region,
                                                                    other_daughter_cell_time,
                                                                    peak_id,
                                                                    fov_id,
                                                                    experiment_name=params['experiment_name'])

                        other_daughter_cell = CellFromGraph(other_daughter_cell_id,
                                                                other_daughter_cell_region,
                                                                other_daughter_cell_time,
                                                                parent=current_cell)

                        tracks[other_daughter_cell_id] = other_daughter_cell
                        current_cell.add_daughter(other_daughter_cell, new_cell_time)

                        # now we remove current_cell, since it's done, and move on to one of the daughters
                        current_cell = new_cell

                # sometimes a second daughter doesn't exist: perhaps parent is at mouth of a trap and one
                #  daughter is lost to the central channel at division time. In this case, do the following:
                except IndexError:
                    current_cell = new_cell

            # if this is a migration, grow the current_cell.
            elif max_edge_type == 'migrate':

                cell_time = graph.nodes[next_node_id]['time']
                cell_region = graph.nodes[next_node_id]['region']
                current_cell.grow(cell_region, cell_time)

            # if the event represents death, kill the cell
            elif max_edge_type == 'die':

                if prior_node_id.startswith(params['experiment_name']):
                    death_time = graph.nodes[prior_node_id]['time']
                    death_region = graph.nodes[prior_node_id]['region']
                    current_cell.die(death_region, death_time)

            # if the event represents disappearance, end the cell
            elif max_edge_type == 'disappear':

                if prior_node_id.startswith(params['experiment_name']):
                    disappear_time = graph.nodes[prior_node_id]['time']
                    disappear_region = graph.nodes[prior_node_id]['region']
                    current_cell.disappears(disappear_region, disappear_time)

            # set the next node to 'visited'
            graph.nodes[next_node_id]['visited'] = True
            if next_node_id != 'B':
                graph_df.iloc[np.where(graph_df.node_id==next_node_id)[0][0],3] = True

            # reset prior_node_id to iterate to next frame and append node_id to current track
            prior_node_id = next_node_id

        if num_unvisited != count_unvisited(graph, params['experiment_name']):
            same_iter_num = 0
        else:
            same_iter_num += 1

        num_unvisited = count_unvisited(graph, params['experiment_name'])
        print("{} detections remain unvisited.".format(num_unvisited))

        if same_iter_num > 10:
            print("WARNING: Ten iterations surpassed without decreasing the number of visited nodes.\n \
                   Breaking tracking loop now. You should probably not trust these results.")
            break

    return tracks

def viterbi_create_lineages_from_graph(graph,
                                        graph_df,
                                        fov_id,
                                        peak_id,
                                        ):

    '''
    This function iterates through nodes in a graph of detections
    to link the nodes as "CellFromGraph" objects, eventually
    leading to the ultimate goal of returning
    a maximally-scoring CellTree object with each cell's information for the experiment.

    For now it ignores the number of cells in a detection and simply
    assumes a 1:1 relationship between detections and cell number.
    '''

    # iterate through all nodes in G
    graph_score = 0
    # track_dict = {}
    tracks = CellTree()

    max_time = np.max([node.timepoint for node in graph.nodes])
    print(max_time)

    for node_id in graph.nodes:
        graph.nodes[node_id]['visited'] = False
    graph_df['visited'] = False
    num_unvisited = count_unvisited(graph, params['experiment_name'])

    for t in range(1,max_time+1):

        if t > 1:
            prior_time_nodes = time_nodes

        if t == 1:
            time_nodes = [node for node in G.nodes if node.time == t]
        else:
            time_nodes = next_time_nodes

        if t != max_time:
            next_time_nodes = [node for node in G.nodes if node.time == t+1]

        for node in time_nodes:
            pass



    while num_unvisited > 0:

        # which detection nodes are not yet visited
        unvisited_detection_nodes = graph_df[(~(graph_df.visited) & graph_df.node_id.str.startswith(params['experiment_name']))]
        # grab the first unvisited node_id from the dataframe
        prior_node_id = unvisited_detection_nodes.iloc[0,1]
        prior_node_time = graph.nodes[prior_node_id]['time']
        prior_node_region = graph.nodes[prior_node_id]['region']

        cell_id = create_cell_id(prior_node_region,
                                    prior_node_time,
                                    peak_id,
                                    fov_id,
                                    experiment_name=params['experiment_name'])

        current_cell = CellFromGraph(cell_id,
                                        prior_node_region,
                                        prior_node_time,
                                        parent=None)

        if not cell_id in tracks.cell_id_list:
            tracks.add_cell(current_cell)
        else:
            current_cell = tracks.get_cell(cell_id)

    #     track_dict_key = prior_node_id
        # for use later in establishing predecessors
        current_node_id = prior_node_id

        # set this detection's "visited" status to True in the graph and in the dataframe
        graph.nodes[prior_node_id]['visited'] = True
        graph_df.iloc[np.where(graph_df.node_id==prior_node_id)[0][0],3] = True

        # build current_track list to this detection's node
        current_track = collections.deque()
        current_track.append(current_node_id)
        predecessors_list = [k for k in graph.predecessors(prior_node_id)]
        unvisited_predecessors_list = [k for k in predecessors_list if not graph.nodes[k]['visited']]

        while len(unvisited_predecessors_list) != 0:

            # initialize a scores array to select highest score from the available options
            predecessor_scores = np.zeros(len(unvisited_predecessors_list))

            # populate array with scores
            for i in range(len(unvisited_predecessors_list)):
                predecessor_node_id = unvisited_predecessors_list[i]
                edge_type, edge_score = get_greatest_score_info(predecessor_node_id, current_node_id, graph)
                predecessor_scores[i] = edge_score

            # find highest score
            max_index = np.argmax(predecessor_scores)
            # grab the node_id corresponding to traversing the highest-scoring edge from the prior node
            current_node_id = unvisited_predecessors_list[max_index]
            current_track.appendleft(current_node_id)

            predecessors_list = [k for k in graph.predecessors(current_node_id)]
            unvisited_predecessors_list = [k for k in predecessors_list if not graph.nodes[k]['visited']]

        while prior_node_id is not 'B':

            # which nodes succeed our current node?
            successor_node_ids = [node_id for node_id in graph.successors(prior_node_id)]

            # keep only the potential successor detections that have not yet been visited
            unvisited_node_ids = []
            for i,successor_node_id in enumerate(successor_node_ids):

                # if it starts with params['experiment_name'], it is a detection node, and not born, appear, etc.
                if successor_node_id.startswith(params['experiment_name']):

                    # if it has been used in the cell track graph, i.e., if 'visited' is True,
                    #   move on. Otherwise, append to our list
                    if graph.nodes[successor_node_id]['visited']:
                        continue
                    else:
                        unvisited_node_ids.append(successor_node_id)

                # if it doesn't start with params['experiment_name'], it is a born, appear, etc., and should always be appended
                else:
                    unvisited_node_ids.append(successor_node_id)

            # initialize a scores array to select highest score from the available options
            successor_scores = np.zeros(len(unvisited_node_ids))
            successor_edge_types = []

            # populate array with scores
            for i in range(len(unvisited_node_ids)):
                successor_node_id = unvisited_node_ids[i]
                edge_type, edge_score = get_greatest_score_info(prior_node_id, successor_node_id, graph)
                successor_scores[i] = edge_score
                successor_edge_types.append(edge_type)

            # find highest score
            max_index = np.argmax(successor_scores)
            # grab the node_id corresponding to traversing the highest-scoring edge from the prior node
            next_node_id = unvisited_node_ids[max_index]
            max_edge_type = successor_edge_types[max_index]

            # if this is a division event, add child node as a new cell,
            #   add the new cell as a daughter to current_cell,
            #   add current_cell as a parent to new cell.
            # Then, search for the second child cell, add it to current_cell, etc.
            if max_edge_type == 'child':

                new_cell_time = graph.nodes[next_node_id]['time']
                new_cell_region = graph.nodes[next_node_id]['region']
                new_cell_id = create_cell_id(new_cell_region,
                                                new_cell_time,
                                                peak_id,
                                                fov_id,
                                                experiment_name=params['experiment_name'])

                new_cell = CellFromGraph(new_cell_id,
                                            new_cell_region,
                                            new_cell_time,
                                            parent=current_cell)

                tracks.add_cell(new_cell)

                current_cell.add_daughter(new_cell, new_cell_time)
    #             print("First daughter", current_cell.id, new_cell.id)

                # initialize a scores array to select highest score from the available options
                unvisited_detection_nodes = [unvisited_node_id for unvisited_node_id in unvisited_node_ids if unvisited_node_id.startswith(params['experiment_name'])]
                child_scores = np.zeros(len(unvisited_detection_nodes))

                # populate array with scores
                for i in range(len(unvisited_detection_nodes)):
                    successor_node_id = unvisited_detection_nodes[i]
                    if successor_node_id == next_node_id:
                        child_scores[i] = -np.inf
                        continue
                    child_score = get_score_by_type(prior_node_id, successor_node_id, graph, score_type='child')
                    child_scores[i] = child_score
    #             print(child_scores)

                try:
                    second_daughter_index = np.argmax(child_scores)
                    # grab the node_id corresponding to traversing the highest-scoring edge from the prior node
                    other_daughter_node_id = unvisited_detection_nodes[second_daughter_index]

                    other_daughter_cell_time = graph.nodes[other_daughter_node_id]['time']
                    other_daughter_cell_region = graph.nodes[other_daughter_node_id]['region']
                    other_daughter_cell_id = create_cell_id(other_daughter_cell_region,
                                                                other_daughter_cell_time,
                                                                peak_id,
                                                                fov_id,
                                                                experiment_name=params['experiment_name'])

                    other_daughter_cell = CellFromGraph(other_daughter_cell_id,
                                                            other_daughter_cell_region,
                                                            other_daughter_cell_time,
                                                            parent=current_cell)

                    tracks.add_cell(other_daughter_cell)

                    current_cell.add_daughter(other_daughter_cell, new_cell_time)

                    # now we remove current_cell, since it's done, and move on to one of the daughters
                    current_cell = new_cell
    #                 print("Second daughter", current_cell.parent.id, other_daughter_cell.id)

                # sometimes a second daughter doesn't exist: perhaps parent is at mouth of a trap and one
                #  daughter is lost to the central channel at division time. In this case, do the following:
                except IndexError:
                    current_cell = new_cell

            # if this is a migration, grow the current_cell.
            elif max_edge_type == 'migrate':

                cell_time = graph.nodes[next_node_id]['time']
                cell_region = graph.nodes[next_node_id]['region']
                current_cell.grow(cell_region, cell_time)

            # if the event represents death, kill the cell
            elif max_edge_type == 'die':

                if prior_node_id.startswith(params['experiment_name']):
                    death_time = graph.nodes[prior_node_id]['time']
                    death_region = graph.nodes[prior_node_id]['region']
                    current_cell.die(death_region, death_time)

            # if the event represents disappearance, end the cell
            elif max_edge_type == 'disappear':

                if prior_node_id.startswith(params['experiment_name']):
                    disappear_time = graph.nodes[prior_node_id]['time']
                    disappear_region = graph.nodes[prior_node_id]['region']
                    current_cell.disappears(disappear_region, disappear_time)

            # set the next node to 'visited'
            graph.nodes[next_node_id]['visited'] = True
            if next_node_id != 'B':
                graph_df.iloc[np.where(graph_df.node_id==next_node_id)[0][0],3] = True

            # reset prior_node_id to iterate to next frame and append node_id to current track
    #         current_track.append(next_node_id)
            prior_node_id = next_node_id
    #         print(current_cell.id, current_cell.parent.id)

    #     track_dict[track_dict_key][:] = current_track

        if num_unvisited != count_unvisited(graph, params['experiment_name']):
            same_iter_num = 0
        else:
            same_iter_num += 1

        num_unvisited = count_unvisited(graph, params['experiment_name'])
        print("{} detections remain unvisited.".format(num_unvisited))

        if same_iter_num > 10:
            break

    return(tracks)

def create_lineages_from_graph_2(graph,
                               graph_df,
                               fov_id,
                               peak_id,
                               ):

    '''
    This function iterates through nodes in a graph of detections
    to link the nodes as "CellFromGraph" objects, eventually
    leading to the ultimate goal of returning
    a CellTree object with each cell's information for the experiment.

    For now it ignores the number of cells in a detection and simply
    assumes a 1:1 relationship between detections and cell number.
    '''

    # iterate through all nodes in G
    # graph_score = 0
    # track_dict = {}
    tracks = CellTree()

    for node_id in graph.nodes:
        graph.nodes[node_id]['visited'] = False
    graph_df['visited'] = False
    num_unvisited = count_unvisited(graph, params['experiment_name'])

    while num_unvisited > 0:

        # which detection nodes are not yet visited
        unvisited_detection_nodes = graph_df[(~(graph_df.visited) & graph_df.node_id.str.startswith(params['experiment_name']))]
        # grab the first unvisited node_id from the dataframe
        prior_node_id = unvisited_detection_nodes.iloc[0,1]
        prior_node_time = graph.nodes[prior_node_id]['time']
        prior_node_region = graph.nodes[prior_node_id]['region']

        cell_id = create_cell_id(prior_node_region,
                                    prior_node_time,
                                    peak_id,
                                    fov_id,
                                    experiment_name=params['experiment_name'])

        current_cell = CellFromGraph(cell_id,
                                        prior_node_region,
                                        prior_node_time,
                                        parent=None)

        if not cell_id in tracks.cell_id_list:
            tracks.add_cell(current_cell)
        else:
            current_cell = tracks.get_cell(cell_id)

    #     track_dict_key = prior_node_id
        # for use later in establishing predecessors
        current_node_id = prior_node_id

        # set this detection's "visited" status to True in the graph and in the dataframe
        graph.nodes[prior_node_id]['visited'] = True
        graph_df.iloc[np.where(graph_df.node_id==prior_node_id)[0][0],3] = True

        # build current_track list to this detection's node
        current_track = collections.deque()
        current_track.append(current_node_id)
        predecessors_list = [k for k in graph.predecessors(prior_node_id)]
        unvisited_predecessors_list = [k for k in predecessors_list if not graph.nodes[k]['visited']]

        while len(unvisited_predecessors_list) != 0:

            # initialize a scores array to select highest score from the available options
            predecessor_scores = np.zeros(len(unvisited_predecessors_list))

            # populate array with scores
            for i in range(len(unvisited_predecessors_list)):
                predecessor_node_id = unvisited_predecessors_list[i]
                edge_type, edge_score = get_greatest_score_info(predecessor_node_id, current_node_id, graph)
                predecessor_scores[i] = edge_score

            # find highest score
            max_index = np.argmax(predecessor_scores)
            # grab the node_id corresponding to traversing the highest-scoring edge from the prior node
            current_node_id = unvisited_predecessors_list[max_index]
            current_track.appendleft(current_node_id)

            predecessors_list = [k for k in graph.predecessors(current_node_id)]
            unvisited_predecessors_list = [k for k in predecessors_list if not graph.nodes[k]['visited']]

        while prior_node_id is not 'B':

            # which nodes succeed our current node?
            successor_node_ids = [node_id for node_id in graph.successors(prior_node_id)]

            # keep only the potential successor detections that have not yet been visited
            unvisited_node_ids = []
            for i,successor_node_id in enumerate(successor_node_ids):

                # if it starts with params['experiment_name'], it is a detection node, and not born, appear, etc.
                if successor_node_id.startswith(params['experiment_name']):

                    # if it has been used in the cell track graph, i.e., if 'visited' is True,
                    #   move on. Otherwise, append to our list
                    if graph.nodes[successor_node_id]['visited']:
                        continue
                    else:
                        unvisited_node_ids.append(successor_node_id)

                # if it doesn't start with params['experiment_name'], it is a born, appear, etc., and should always be appended
                else:
                    unvisited_node_ids.append(successor_node_id)

            # initialize a scores array to select highest score from the available options
            successor_scores = np.zeros(len(unvisited_node_ids))
            successor_edge_types = []

            # populate array with scores
            for i in range(len(unvisited_node_ids)):
                successor_node_id = unvisited_node_ids[i]
                edge_type, edge_score = get_greatest_score_info(prior_node_id, successor_node_id, graph)
                successor_scores[i] = edge_score
                successor_edge_types.append(edge_type)

            # find highest score
            max_index = np.argmax(successor_scores)
            # grab the node_id corresponding to traversing the highest-scoring edge from the prior node
            next_node_id = unvisited_node_ids[max_index]
            max_edge_type = successor_edge_types[max_index]

            # if this is a division event, add child node as a new cell,
            #   add the new cell as a daughter to current_cell,
            #   add current_cell as a parent to new cell.
            # Then, search for the second child cell, add it to current_cell, etc.
            if max_edge_type == 'child':

                new_cell_time = graph.nodes[next_node_id]['time']
                new_cell_region = graph.nodes[next_node_id]['region']
                new_cell_id = create_cell_id(new_cell_region,
                                                new_cell_time,
                                                peak_id,
                                                fov_id,
                                                experiment_name=params['experiment_name'])

                new_cell = CellFromGraph(new_cell_id,
                                            new_cell_region,
                                            new_cell_time,
                                            parent=current_cell)

                tracks.add_cell(new_cell)

                current_cell.add_daughter(new_cell, new_cell_time)
    #             print("First daughter", current_cell.id, new_cell.id)

                # initialize a scores array to select highest score from the available options
                unvisited_detection_nodes = [unvisited_node_id for unvisited_node_id in unvisited_node_ids if unvisited_node_id.startswith(params['experiment_name'])]
                child_scores = np.zeros(len(unvisited_detection_nodes))

                # populate array with scores
                for i in range(len(unvisited_detection_nodes)):
                    successor_node_id = unvisited_detection_nodes[i]
                    if successor_node_id == next_node_id:
                        child_scores[i] = -np.inf
                        continue
                    child_score = get_score_by_type(prior_node_id, successor_node_id, graph, score_type='child')
                    child_scores[i] = child_score
    #             print(child_scores)

                try:
                    second_daughter_index = np.argmax(child_scores)
                    # grab the node_id corresponding to traversing the highest-scoring edge from the prior node
                    other_daughter_node_id = unvisited_detection_nodes[second_daughter_index]

                    other_daughter_cell_time = graph.nodes[other_daughter_node_id]['time']
                    other_daughter_cell_region = graph.nodes[other_daughter_node_id]['region']
                    other_daughter_cell_id = create_cell_id(other_daughter_cell_region,
                                                                other_daughter_cell_time,
                                                                peak_id,
                                                                fov_id,
                                                                experiment_name=params['experiment_name'])

                    other_daughter_cell = CellFromGraph(other_daughter_cell_id,
                                                            other_daughter_cell_region,
                                                            other_daughter_cell_time,
                                                            parent=current_cell)

                    tracks.add_cell(other_daughter_cell)

                    current_cell.add_daughter(other_daughter_cell, new_cell_time)

                    # now we remove current_cell, since it's done, and move on to one of the daughters
                    current_cell = new_cell
    #                 print("Second daughter", current_cell.parent.id, other_daughter_cell.id)

                # sometimes a second daughter doesn't exist: perhaps parent is at mouth of a trap and one
                #  daughter is lost to the central channel at division time. In this case, do the following:
                except IndexError:
                    current_cell = new_cell

            # if this is a migration, grow the current_cell.
            elif max_edge_type == 'migrate':

                cell_time = graph.nodes[next_node_id]['time']
                cell_region = graph.nodes[next_node_id]['region']
                current_cell.grow(cell_region, cell_time)

            # if the event represents death, kill the cell
            elif max_edge_type == 'die':

                if prior_node_id.startswith(params['experiment_name']):
                    death_time = graph.nodes[prior_node_id]['time']
                    death_region = graph.nodes[prior_node_id]['region']
                    current_cell.die(death_region, death_time)

            # if the event represents disappearance, end the cell
            elif max_edge_type == 'disappear':

                if prior_node_id.startswith(params['experiment_name']):
                    disappear_time = graph.nodes[prior_node_id]['time']
                    disappear_region = graph.nodes[prior_node_id]['region']
                    current_cell.disappears(disappear_region, disappear_time)

            # set the next node to 'visited'
            graph.nodes[next_node_id]['visited'] = True
            if next_node_id != 'B':
                graph_df.iloc[np.where(graph_df.node_id==next_node_id)[0][0],3] = True

            # reset prior_node_id to iterate to next frame and append node_id to current track
    #         current_track.append(next_node_id)
            prior_node_id = next_node_id
    #         print(current_cell.id, current_cell.parent.id)

    #     track_dict[track_dict_key][:] = current_track

        if num_unvisited != count_unvisited(graph, params['experiment_name']):
            same_iter_num = 0
        else:
            same_iter_num += 1

        num_unvisited = count_unvisited(graph, params['experiment_name'])
        print("{} detections remain unvisited.".format(num_unvisited))

        if same_iter_num > 10:
            break

    return(tracks)

# obtains cell length and width of the cell using the feret diameter
def feretdiameter(region):
    '''
    feretdiameter calculates the length and width of the binary region shape. The cell orientation
    from the ellipsoid is used to find the major and minor axis of the cell.
    See https://en.wikipedia.org/wiki/Feret_diameter.
    '''

    # y: along vertical axis of the image; x: along horizontal axis of the image;
    # calculate the relative centroid in the bounding box (non-rotated)
    # print(region.centroid)
    y0, x0 = region.centroid
    y0 = y0 - np.int16(region.bbox[0]) + 1
    x0 = x0 - np.int16(region.bbox[1]) + 1
    cosorient = np.cos(region.orientation)
    sinorient = np.sin(region.orientation)
    # print(cosorient, sinorient)
    amp_param = 1.2 #amplifying number to make sure the axis is longer than actual cell length

    # coordinates relative to bounding box
    # r_coords = region.coords - [np.int16(region.bbox[0]), np.int16(region.bbox[1])]

    # limit to perimeter coords. pixels are relative to bounding box
    region_binimg = np.pad(region.image, 1, 'constant') # pad region binary image by 1 to avoid boundary non-zero pixels
    distance_image = ndi.distance_transform_edt(region_binimg)
    r_coords = np.where(distance_image == 1)
    r_coords = list(zip(r_coords[0], r_coords[1]))

    # coordinates are already sorted by y. partion into top and bottom to search faster later
    # if orientation > 0, L1 is closer to top of image (lower Y coord)
    if region.orientation > 0:
        L1_coords = r_coords[:int(np.round(len(r_coords)/4))]
        L2_coords = r_coords[int(np.round(len(r_coords)/4)):]
    else:
        L1_coords = r_coords[int(np.round(len(r_coords)/4)):]
        L2_coords = r_coords[:int(np.round(len(r_coords)/4))]

    #####################
    # calculte cell length
    L1_pt = np.zeros((2,1))
    L2_pt = np.zeros((2,1))

    # define the two end points of the the long axis line
    # one pole.
    L1_pt[1] = x0 + cosorient * 0.5 * region.major_axis_length*amp_param
    L1_pt[0] = y0 - sinorient * 0.5 * region.major_axis_length*amp_param

    # the other pole.
    L2_pt[1] = x0 - cosorient * 0.5 * region.major_axis_length*amp_param
    L2_pt[0] = y0 + sinorient * 0.5 * region.major_axis_length*amp_param

    # calculate the minimal distance between the points at both ends of 3 lines
    # aka calcule the closest coordiante in the region to each of the above points.
    # pt_L1 = r_coords[np.argmin([np.sqrt(np.power(Pt[0]-L1_pt[0],2) + np.power(Pt[1]-L1_pt[1],2)) for Pt in r_coords])]
    # pt_L2 = r_coords[np.argmin([np.sqrt(np.power(Pt[0]-L2_pt[0],2) + np.power(Pt[1]-L2_pt[1],2)) for Pt in r_coords])]

    try:
        pt_L1 = L1_coords[np.argmin([np.sqrt(np.power(Pt[0]-L1_pt[0],2) + np.power(Pt[1]-L1_pt[1],2)) for Pt in L1_coords])]
        pt_L2 = L2_coords[np.argmin([np.sqrt(np.power(Pt[0]-L2_pt[0],2) + np.power(Pt[1]-L2_pt[1],2)) for Pt in L2_coords])]
        length = np.sqrt(np.power(pt_L1[0]-pt_L2[0],2) + np.power(pt_L1[1]-pt_L2[1],2))
    except:
        length = None

    #####################
    # calculate cell width
    # draw 2 parallel lines along the short axis line spaced by 0.8*quarter of length = 0.4, to avoid  in midcell

    # limit to points in each half
    W_coords = []
    if region.orientation > 0:
        W_coords.append(r_coords[:int(np.round(len(r_coords)/2))]) # note the /2 here instead of /4
        W_coords.append(r_coords[int(np.round(len(r_coords)/2)):])
    else:
        W_coords.append(r_coords[int(np.round(len(r_coords)/2)):])
        W_coords.append(r_coords[:int(np.round(len(r_coords)/2))])

    # starting points
    x1 = x0 + cosorient * 0.5 * length*0.4
    y1 = y0 - sinorient * 0.5 * length*0.4
    x2 = x0 - cosorient * 0.5 * length*0.4
    y2 = y0 + sinorient * 0.5 * length*0.4
    W1_pts = np.zeros((2,2))
    W2_pts = np.zeros((2,2))

    # now find the ends of the lines
    # one side
    W1_pts[0,1] = x1 - sinorient * 0.5 * region.minor_axis_length*amp_param
    W1_pts[0,0] = y1 - cosorient * 0.5 * region.minor_axis_length*amp_param
    W1_pts[1,1] = x2 - sinorient * 0.5 * region.minor_axis_length*amp_param
    W1_pts[1,0] = y2 - cosorient * 0.5 * region.minor_axis_length*amp_param

    # the other side
    W2_pts[0,1] = x1 + sinorient * 0.5 * region.minor_axis_length*amp_param
    W2_pts[0,0] = y1 + cosorient * 0.5 * region.minor_axis_length*amp_param
    W2_pts[1,1] = x2 + sinorient * 0.5 * region.minor_axis_length*amp_param
    W2_pts[1,0] = y2 + cosorient * 0.5 * region.minor_axis_length*amp_param

    # calculate the minimal distance between the points at both ends of 3 lines
    pt_W1 = np.zeros((2,2))
    pt_W2 = np.zeros((2,2))
    d_W = np.zeros((2,1))
    i = 0
    for W1_pt, W2_pt in zip(W1_pts, W2_pts):

        # # find the points closest to the guide points
        # pt_W1[i,0], pt_W1[i,1] = r_coords[np.argmin([np.sqrt(np.power(Pt[0]-W1_pt[0],2) + np.power(Pt[1]-W1_pt[1],2)) for Pt in r_coords])]
        # pt_W2[i,0], pt_W2[i,1] = r_coords[np.argmin([np.sqrt(np.power(Pt[0]-W2_pt[0],2) + np.power(Pt[1]-W2_pt[1],2)) for Pt in r_coords])]

        # find the points closest to the guide points
        pt_W1[i,0], pt_W1[i,1] = W_coords[i][np.argmin([np.sqrt(np.power(Pt[0]-W1_pt[0],2) + np.power(Pt[1]-W1_pt[1],2)) for Pt in W_coords[i]])]
        pt_W2[i,0], pt_W2[i,1] = W_coords[i][np.argmin([np.sqrt(np.power(Pt[0]-W2_pt[0],2) + np.power(Pt[1]-W2_pt[1],2)) for Pt in W_coords[i]])]

        # calculate the actual width
        d_W[i] = np.sqrt(np.power(pt_W1[i,0]-pt_W2[i,0],2) + np.power(pt_W1[i,1]-pt_W2[i,1],2))
        i += 1

    # take the average of the two at quarter positions
    width = np.mean([d_W[0],d_W[1]])

    return length, width

# take info and make string for cell id
def create_focus_id(region, t, peak, fov, experiment_name=None):
    '''Make a unique focus id string for a new focus'''
    if experiment_name is None:
        focus_id = 'f{:0=2}p{:0=4}t{:0=4}r{:0=2}'.format(fov, peak, t, region.label)
    else:
        focus_id = '{}f{:0=2}p{:0=4}t{:0=4}r{:0=2}'.format(experiment_name, fov, peak, t, region.label)
    return focus_id

# take info and make string for cell id
def create_cell_id(region, t, peak, fov, experiment_name=None):
    '''Make a unique cell id string for a new cell'''
    # cell_id = ['f', str(fov), 'p', str(peak), 't', str(t), 'r', str(region.label)]
    if experiment_name is None:
        cell_id = ['f', '%02d' % fov, 'p', '%04d' % peak, 't', '%04d' % t, 'r', '%02d' % region.label]
        cell_id = ''.join(cell_id)
    else:
        cell_id = '{}f{:0=2}p{:0=4}t{:0=4}r{:0=2}'.format(experiment_name, fov, peak, t, region.label)
    return cell_id

def create_detection_id(t, peak, fov, region_label, experiment_name=None, max_cell_number=6):
    '''Make a unique cell id string for a new cell'''
    # cell_id = ['f', str(fov), 'p', str(peak), 't', str(t), 'r', str(region.label)]
    if experiment_name is None:
        det_id = ['f', '%02d' % fov, 'p', '%04d' % peak, 't', '%04d' % t, 'r', '%02d' % region_label]
        det_id = ''.join(det_id)
    else:
        det_id = '{}f{:0=2}p{:0=4}t{:0=4}r{:0=2}'.format(experiment_name, fov, peak, t, region_label)
    return det_id

def initialize_track_graph(peak_id,
                        fov_id,
                        experiment_name,
                        predictions_dict,
                        regions_by_time,
                        max_cell_number=6,
                        born_threshold=0.75,
                        appear_threshold=0.75):

    detection_dict = {}
    frame_num = predictions_dict['migrate_model_predictions'].shape[0]

    ebunch = []

    G = nx.MultiDiGraph()
    # create common start point
    G.add_node('A')
    # create common end point
    G.add_node('B')

    last_frame = False

    node_id_list = []
    timepoint_list = []
    region_label_list = []

    for frame_idx in range(frame_num):

        timepoint = frame_idx + 1
        paired_detection_time = timepoint+1

        # get detections for this frame
        frame_regions_list = regions_by_time[frame_idx]

        # if we're at the end of the imaging, make all cells migrate to node 'B'
        if timepoint == frame_num:
            last_frame = True
        else:
            paired_frame_regions_list = regions_by_time[frame_idx+1]

        # get state change probabilities (class predictions) for this frame
        frame_prediction_dict = {key:val[frame_idx,...] for key,val in predictions_dict.items() if key != 'general_model_predictions'}
        # for i in range(len(predictions_dict['general_model_predictions'])):
            # frame_general_prediction = predictions_dict['general_model_predictions'][]

        # create the "will be born" and "will appear" nodes for this frame
        prior_born_state = 'born_{:0=4}'.format(timepoint-1)
        born_state = 'born_{:0=4}'.format(timepoint)
        G.add_node(born_state, visited=False, time=timepoint)

        prior_appear_state = 'appear_{:0=4}'.format(timepoint-1)
        appear_state = 'appear_{:0=4}'.format(timepoint)
        G.add_node(appear_state, visited=False, time=timepoint)

        if frame_idx == 0:
            ebunch.append(('A', appear_state, 'start', {'weight':appear_threshold, 'score':1*np.log(appear_threshold)}))
            ebunch.append(('A', born_state, 'start', {'weight':born_threshold, 'score':1*np.log(born_threshold)}))

        # create the "Dies" and "Disappeared" nodes to link from prior frame
        prior_dies_state = 'dies_{:0=4}'.format(timepoint-1)
        dies_state = 'dies_{:0=4}'.format(timepoint)
        next_dies_state = 'dies_{:0=4}'.format(timepoint+1)
        G.add_node(dies_state, visited=False, time=timepoint)

        prior_disappear_state = 'disappear_{:0=4}'.format(timepoint-1)
        disappear_state = 'disappear_{:0=4}'.format(timepoint)
        next_disappear_state = 'disappear_{:0=4}'.format(timepoint+1)
        G.add_node(disappear_state, visited=False, time=timepoint)

        node_id_list.extend([born_state, dies_state, appear_state, disappear_state])
        timepoint_list.extend([timepoint, timepoint, timepoint, timepoint])
        region_label_list.extend([0,0,0,0])

        if frame_idx > 0:

            ebunch.append((prior_dies_state, dies_state, 'die', {'weight':1.1, 'score':1*np.log(1.1)})) # impossible to move out of dies track
            ebunch.append((prior_disappear_state, disappear_state, 'disappear', {'weight':1.1, 'score':1*np.log(1.1)})) # impossible to move out of disappear track
            ebunch.append((prior_born_state, born_state, 'born', {'weight':born_threshold, 'score':1*np.log(born_threshold)}))
            ebunch.append((prior_appear_state, appear_state, 'appear', {'weight':appear_threshold, 'score':1*np.log(appear_threshold)}))

        if last_frame:
            ebunch.append((appear_state, 'B', 'end', {'weight':1, 'score':1*np.log(1)}))
            ebunch.append((disappear_state, 'B', 'end', {'weight':1, 'score':1*np.log(1)}))
            ebunch.append((born_state, 'B', 'end', {'weight':1, 'score':1*np.log(1)}))
            ebunch.append((dies_state, 'B', 'end', {'weight':1, 'score':1*np.log(1)}))

        for region_idx in range(max_cell_number):

            # the tracking models assume there are 6 detections in each frame, regardless of how many
            #   are actually there. Therefore, this try/except logic will catch cases where there
            #   were fewer than 6 detections in a frame.
            try:
                region = frame_regions_list[region_idx]
                region_label = region.label
            except IndexError:
                region = None
                region_label = region_idx + 1

            # create the name for this detection
            detection_id = create_detection_id(timepoint,
                                                peak_id,
                                                fov_id,
                                                region_label,
                                                experiment_name=experiment_name)

            det = Detection(detection_id, region, timepoint)
            detection_dict[det.id] = det

            if det.area is not None:
                # if the detection represents a segmentation from our imaging, add its ID,
                #   which is also its key in detection_dict, as a node in G
                G.add_node(det.id, visited=False, cell_count=1, region=region, time=timepoint)
                timepoint_list.append(timepoint)
                node_id_list.append(detection_id)
                region_label_list.append(region.label)
                # also set up all edges for this detection's node in our ebunch
                #   loop through prediction types and add each to the ebunch

                for key,val in frame_prediction_dict.items():

                    if frame_idx == 0:

                        ebunch.append(('A', detection_id, 'start', {'weight':1, 'score':1*np.log(1)}))

                    if last_frame:

                        ebunch.append((detection_id, 'B', 'end', {'weight':1, 'score':1*np.log(1)}))

                        if val.shape[0] == max_cell_number ** 2:
                            continue

                        else:
                            frame_predictions = val
                            detection_prediction = frame_predictions[region_idx]

                            if key == 'appear_model_predictions':
                                if frame_idx == 0:
                                    continue
                                elem = (prior_appear_state, detection_id, 'appear', {'weight':detection_prediction, 'score':1*np.log(detection_prediction)})

                            elif 'born' in key:
                                if frame_idx == 0:
                                    continue
                                elem = (prior_born_state, detection_id, 'born', {'weight':detection_prediction, 'score':1*np.log(detection_prediction)})

                            elif 'zero_cell' in key:
                                G.nodes[det.id]['zero_cell_weight'] = detection_prediction
                                G.nodes[det.id]['zero_cell_score'] = 1*np.log(detection_prediction)

                            elif 'one_cell' in key:
                                G.nodes[det.id]['one_cell_weight'] = detection_prediction
                                G.nodes[det.id]['zero_cell_score'] = 1*np.log(detection_prediction)

                            elif 'two_cell' in key:
                                G.nodes[det.id]['two_cell_weight'] = detection_prediction
                                G.nodes[det.id]['zero_cell_score'] = 1*np.log(detection_prediction)

                            ebunch.append(elem)

                    else:
                        # if the array is cell_number^2, reshape it to cell_number x cell_number
                        #  Then slice our detection's row and iterate over paired_cells
                        if val.shape[0] == max_cell_number**2:

                            frame_predictions = val.reshape((max_cell_number,max_cell_number))
                            detection_predictions = frame_predictions[region_idx,:]

                            # loop through paired detection predictions, test whether paired detection exists
                            #  then append the edge to our ebunch
                            for paired_cell_idx in range(detection_predictions.size):

                                # attempt to grab the paired detection. If we get an IndexError, it doesn't exist.
                                try:
                                    paired_detection = paired_frame_regions_list[paired_cell_idx]
                                except IndexError:
                                    continue

                                # create the paired detection's id for use in our ebunch
                                paired_detection_id = create_detection_id(paired_detection_time,
                                                                            peak_id,
                                                                            fov_id,
                                                                            paired_detection.label,
                                                                            experiment_name=experiment_name)

                                paired_prediction = detection_predictions[paired_cell_idx]
                                if 'child_' in key:
                                    child_weight = paired_prediction
                                    elem = (detection_id, paired_detection_id, 'child', {'child_weight':child_weight, 'score':1*np.log(child_weight)})
                                    ebunch.append(elem)

                                if 'migrate_' in key:
                                    migrate_weight = paired_prediction
                                    elem = (detection_id, paired_detection_id, 'migrate', {'migrate_weight':migrate_weight, 'score':1*np.log(migrate_weight)})
                                    ebunch.append(elem)

                                # if 'interaction_' in key:
                                #     interaction_weight = paired_prediction
                                #     elem = (detection_id, paired_detection_id, 'interaction', {'weight':interaction_weight, 'score':1*np.log(interaction_weight)})
                                #     ebunch.append(elem)

                        # if the array is cell_number long, do similar stuff as above.
                        elif val.shape[0] == max_cell_number:

                            frame_predictions = val
                            detection_prediction = frame_predictions[region_idx]

                            if key == 'appear_model_predictions':
                                if frame_idx == 0:
                                    continue
    #                             print("Linking {} to {}.".format(prior_appear_state, detection_id))
                                elem = (prior_appear_state, detection_id, 'appear', {'weight':detection_prediction, 'score':1*np.log(detection_prediction)})

                            elif 'disappear_' in key:
                                if last_frame:
                                    continue
    #                             print("Linking {} to {}.".format(detection_id, next_disappear_state))
                                elem = (detection_id, next_disappear_state, 'disappear', {'weight':detection_prediction, 'score':1*np.log(detection_prediction)})

                            elif 'born_' in key:
                                if frame_idx == 0:
                                    continue
    #                             print("Linking {} to {}.".format(prior_born_state, detection_id))
                                elem = (prior_born_state, detection_id, 'born', {'weight':detection_prediction, 'score':1*np.log(detection_prediction)})

                            elif 'die_model' in key:
                                if last_frame:
                                    continue
    #                             print("Linking {} to {}.".format(detection_id, next_dies_state))
                                elem = (detection_id, next_dies_state, 'die', {'weight':detection_prediction, 'score':1*np.log(detection_prediction)})

                            elif 'zero_cell' in key:
                                G.nodes[det.id]['zero_cell_weight'] = detection_prediction
                                G.nodes[det.id]['zero_cell_score'] = 1*np.log(detection_prediction)

                            elif 'one_cell' in key:
                                G.nodes[det.id]['one_cell_weight'] = detection_prediction
                                G.nodes[det.id]['one_cell_score'] = 1*np.log(detection_prediction)

                            elif 'two_cell' in key:
                                G.nodes[det.id]['two_cell_weight'] = detection_prediction
                                G.nodes[det.id]['two_cell_score'] = 1*np.log(detection_prediction)

                            ebunch.append(elem)

    G.add_edges_from(ebunch)
    graph_df = pd.DataFrame(data={'timepoint':timepoint_list,
                                  'node_id':node_id_list,
                                  'region_label':region_label_list})
    return(G, graph_df)

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
    max_growth_length = params['track']['max_growth_length']
    min_growth_length = params['track']['min_growth_length']
    max_growth_area = params['track']['max_growth_area']
    min_growth_area = params['track']['min_growth_area']

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

    # # check if y position of region is within
    # # the quarter positions of the bounding box
    # lower_quarter = cell.bboxes[-1][0] + (region.major_axis_length / 4)
    # upper_quarter = cell.bboxes[-1][2] - (region.major_axis_length / 4)
    # if lower_quarter > region.centroid[0] or upper_quarter < region.centroid[0]:
    #     return False

    # check if y position of region is within the bounding box of previous region
    lower_bound = cell.bboxes[-1][0]
    upper_bound = cell.bboxes[-1][2]
    if lower_bound > region.centroid[0] or upper_bound < region.centroid[0]:
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
    max_growth_length = params['track']['max_growth_length']
    min_growth_length = params['track']['min_growth_length']

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
    if cell.bboxes[-1][0] > region1.centroid[0] or cell.centroids[-1][0] < region1.centroid[0]:
        return 0
    # bottom region with bottom half of mother bounding box
    if cell.centroids[-1][0] > region2.centroid[0] or cell.bboxes[-1][2] < region2.centroid[0]:
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

# finds cells whose birth label is 1
def find_mother_cells(Cells):
    '''Return only cells whose starting region label is 1.'''

    Mother_Cells = {}

    for cell_id in Cells:
        if Cells[cell_id].birth_label == 1:
            Mother_Cells[cell_id] = Cells[cell_id]

    return Mother_Cells

def filter_foci(Foci, label, t, debug=False):

    Filtered_Foci = {}

    for focus_id, focus in Foci.items():

        # copy the times list so as not to update it in-place
        times = focus.times
        if debug:
            print(times)

        match_inds = [i for i,time in enumerate(times) if time == t]
        labels = [focus.labels[idx] for idx in match_inds]

        if label in labels:
            Filtered_Foci[focus_id] = focus

    return Filtered_Foci

def filter_cells(Cells, attr, val, idx=None, debug=False):
    '''Return only cells whose designated attribute equals "val".'''

    Filtered_Cells = {}

    for cell_id, cell in Cells.items():

        at_val = getattr(cell, attr)
        if debug:
            print(at_val)
            print("Times: ", cell.times)
        if idx is not None:
            at_val = at_val[idx]
        if at_val == val:
            Filtered_Cells[cell_id] = cell

    return Filtered_Cells

def filter_cells_containing_val_in_attr(Cells, attr, val):
    '''Return only cells that have val in list attribute, attr.'''

    Filtered_Cells = {}

    for cell_id, cell in Cells.items():

        at_list = getattr(cell, attr)
        if val in at_list:
            Filtered_Cells[cell_id] = cell

    return Filtered_Cells
### functions for additional cell centric analysis

def compile_cell_info_df(Cells):

    counter = 0
    cell_count = len(Cells.keys())
    for cell_id,cell in Cells.items():

        if counter % 100 == 0:
            print("Generating information for cell {} out of {}.".format(counter, cell_count))

        if counter == 0:
            wide_df = cell.make_wide_df()
            long_df = cell.make_long_df()
            counter += 1
        else:
            wide_df = wide_df.append(cell.make_wide_df())
            long_df = long_df.append(cell.make_long_df())
            counter += 1

    long_df.reset_index(drop=True, inplace=True)
    wide_df.reset_index(drop=True, inplace=True)

    return(wide_df, long_df)

def compile_foci_info_df(Foci):
    '''
    Parameters
    ----------------

    Foci : dictionary, keys of which are focus_ids,
           values of which are objects of class Focus

    Returns
    ----------------------

    A tuple, the first element of which is a wide
    pandas DataFrame with brief information on each focus,
    the second element of which is a long DataFrame with
    detailed information about each timepoint for each focus.
    '''

    focus_counter = 0
    focus_count = len(Foci)
    for focus_id,focus in Foci.items():

        if focus_counter % 100 == 0:
            print("Generating focus information for focus {} out of {}.".format(focus_counter+1, focus_count))

        if focus_counter == 0:
            wide_df = focus.make_wide_df()
            long_df = focus.make_long_df()
            focus_counter += 1
        else:
            wide_df = wide_df.append(focus.make_wide_df())
            long_df = long_df.append(focus.make_long_df())
            focus_counter += 1

    long_df.reset_index(drop=True, inplace=True)
    wide_df.reset_index(drop=True, inplace=True)

    return(wide_df, long_df)
    # return(long_df)

def find_all_cell_intensities(Cells,
                              specs, time_table, channel_name='sub_c2',
                              apply_background_correction=True):
    '''
    Finds fluorescenct information for cells. All the cells in Cells
    should be from one fov/peak.
    '''

    # iterate over each fov in specs
    for fov_id,fov_peaks in specs.items():

        # iterate over each peak in fov
        for peak_id,peak_value in fov_peaks.items():

            # if peak_id's value is not 1, go to next peak
            if peak_value != 1:
                continue

            print("Quantifying channel {} fluorescence in cells in fov {}, peak {}.".format(channel_name, fov_id, peak_id))
            # Load fluorescent images and segmented images for this channel
            fl_stack = load_stack(fov_id, peak_id, color=channel_name)
            corrected_stack = np.zeros(fl_stack.shape)

            for frame in range(fl_stack.shape[0]):
                # median filter will be applied to every image
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    median_filtered = median(fl_stack[frame,...], selem=morphology.disk(1))

                # subtract the gaussian-filtered image from true image to correct
                #   uneven background fluorescence
                if apply_background_correction:
                    blurred = filters.gaussian(median_filtered, sigma=10, preserve_range=True)
                    corrected_stack[frame,:,:] = median_filtered-blurred
                else:
                    corrected_stack[frame,:,:] = median_filtered

            seg_stack = load_stack(fov_id, peak_id, color='seg_unet')

            # evaluate whether each cell is in this fov/peak combination
            for cell_id,cell in Cells.items():

                cell_fov = cell.fov
                if cell_fov != fov_id:
                    continue

                cell_peak = cell.peak
                if cell_peak != peak_id:
                    continue

                cell_times = cell.times
                cell_labels = cell.labels
                cell.area_mean_fluorescence[channel_name] = []
                cell.volume_mean_fluorescence[channel_name] = []
                cell.total_fluorescence[channel_name] = []

                # loop through cell's times
                for i,t in enumerate(cell_times):
                    frame = t-1
                    cell_label = cell_labels[i]

                    total_fluor = np.sum(corrected_stack[frame, seg_stack[frame, :,:] == cell_label])

                    cell.area_mean_fluorescence[channel_name].append(total_fluor/cell.areas[i])
                    cell.volume_mean_fluorescence[channel_name].append(total_fluor/cell.volumes[i])
                    cell.total_fluorescence[channel_name].append(total_fluor)

    # The cell objects in the original dictionary will be updated,
    # no need to return anything specifically.
    return

def find_cell_intensities(fov_id, peak_id, Cells, midline=False, channel_name='sub_c2'):
    '''
    Finds fluorescenct information for cells. All the cells in Cells
    should be from one fov/peak. See the function
    organize_cells_by_channel()
    '''

    # Load fluorescent images and segmented images for this channel
    fl_stack = load_stack(fov_id, peak_id, color=channel_name)
    seg_stack = load_stack(fov_id, peak_id, color='seg_unet')

    # determine absolute time index
    times_all = []
    for fov in params['time_table']:
        times_all = np.append(times_all, time_table[fov].keys())
    times_all = np.unique(times_all)
    times_all = np.sort(times_all)
    times_all = np.array(times_all,np.int_)
    t0 = times_all[0] # first time index

    # Loop through cells
    for Cell in Cells.values():
        # give this cell two lists to hold new information
        Cell.fl_tots = [] # total fluorescence per time point
        Cell.fl_area_avgs = [] # avg fluorescence per unit area by timepoint
        Cell.fl_vol_avgs = [] # avg fluorescence per unit volume by timepoint

        if midline:
            Cell.mid_fl = [] # avg fluorescence of midline

        # and the time points that make up this cell's life
        for n, t in enumerate(Cell.times):
            # create fluorescent image only for this cell and timepoint.
            fl_image_masked = np.copy(fl_stack[t-t0])
            fl_image_masked[seg_stack[t-t0] != Cell.labels[n]] = 0

            # append total flourescent image
            Cell.fl_tots.append(np.sum(fl_image_masked))
            # and the average fluorescence
            Cell.fl_area_avgs.append(np.sum(fl_image_masked) / Cell.areas[n])
            Cell.fl_vol_avgs.append(np.sum(fl_image_masked) / Cell.volumes[n])

            if midline:
                # add the midline average by first applying morphology transform
                bin_mask = np.copy(seg_stack[t-t0])
                bin_mask[bin_mask != Cell.labels[n]] = 0
                med_mask, _ = morphology.medial_axis(bin_mask, return_distance=True)
                # med_mask[med_dist < np.floor(cap_radius/2)] = 0
                # print(img_fluo[med_mask])
                if (np.shape(fl_image_masked[med_mask])[0] > 0):
                    Cell.mid_fl.append(np.nanmean(fl_image_masked[med_mask]))
                else:
                    Cell.mid_fl.append(0)

    # The cell objects in the original dictionary will be updated,
    # no need to return anything specifically.
    return

# find foci using a difference of gaussians method
def foci_analysis(fov_id, peak_id, Cells):
    '''Find foci in cells using a fluorescent image channel.
    This function works on a single peak and all the cells therein.'''

    # make directory for foci debug
    # foci_dir = os.path.join(params['ana_dir'], 'overlay/')
    # if not os.path.exists(foci_dir):
    #     os.makedirs(foci_dir)

    # Import segmented and fluorescenct images
    image_data_seg = load_stack(fov_id, peak_id, color='seg_unet')
    image_data_FL = load_stack(fov_id, peak_id,
                               color='sub_{}'.format(params['foci']['foci_plane']))

    # determine absolute time index
    times_all = []
    for fov, times in params['time_table'].items():
        times_all = np.append(times_all, list(times.keys()))
    times_all = np.unique(times_all)
    times_all = np.sort(times_all)
    times_all = np.array(times_all, np.int_)
    t0 = times_all[0] # first time index

    for cell_id, cell in six.iteritems(Cells):

        information('Extracting foci information for %s.' % (cell_id))

        # declare lists holding information about foci.
        disp_l = []
        disp_w = []
        foci_h = []
        # foci_stack = np.zeros((np.size(cell.times),
        #                        image_data_seg[0,:,:].shape[0], image_data_seg[0,:,:].shape[1]))

        # Go through each time point of this cell
        for t in cell.times:
            # retrieve this timepoint and images.
            image_data_temp = image_data_FL[t-t0,:,:]
            image_data_temp_seg = image_data_seg[t-t0,:,:]

            # find foci as long as there is information in the fluorescent image
            if np.sum(image_data_temp) != 0:
                disp_l_tmp, disp_w_tmp, foci_h_tmp = foci_lap(image_data_temp_seg,
                                                              image_data_temp, cell, t)

                disp_l.append(disp_l_tmp)
                disp_w.append(disp_w_tmp)
                foci_h.append(foci_h_tmp)

            # if there is no information, append an empty list.
            # Should this be NaN?
            else:
                disp_l.append([])
                disp_w.append([])
                foci_h.append([])
                # foci_stack[i] = image_data_temp_seg

        # add information to the cell (will replace old data)
        cell.disp_l = disp_l
        cell.disp_w = disp_w
        cell.foci_h = foci_h

        # Create a stack of the segmented images with marked foci
        # This should poentially be changed to the fluorescent images with marked foci
        # foci_stack = np.uint16(foci_stack)
        # foci_stack = np.stack(foci_stack, axis=0)
        # # Export overlaid images
        # foci_filename = params['experiment_name'] + 't%04d_xy%03d_p%04d_r%02d_overlay.tif' % (Cells[cell_id].birth_time, Cells[cell_id].fov, Cells[cell_id].peak, Cells[cell_id].birth_label)
        # foci_filepath = foci_dir + foci_filename
        #
        # tiff.imsave(foci_filepath, foci_stack, compress=3) # save it

        # test
        # sys.exit()

    return

# foci pool (for parallel analysis)
def foci_analysis_pool(fov_id, peak_id, Cells):
    '''Find foci in cells using a fluorescent image channel.
    This function works on a single peak and all the cells therein.'''

    # make directory for foci debug
    # foci_dir = os.path.join(params['ana_dir'], 'overlay/')
    # if not os.path.exists(foci_dir):
    #     os.makedirs(foci_dir)

    # Import segmented and fluorescenct images
    image_data_seg = load_stack(fov_id, peak_id, color='seg_unet')
    image_data_FL = load_stack(fov_id, peak_id,
                               color='sub_{}'.format(params['foci']['foci_plane']))

    # Load time table to determine first image index.
    times_all = np.array(np.sort(params['time_table'][fov_id].keys()), np.int_)
    t0 = times_all[0] # first time index
    tN = times_all[-1] # last time index

    # call foci_cell for each cell object
    pool = Pool(processes=params['num_analyzers'])
    [pool.apply_async(foci_cell(cell_id, cell, t0, image_data_seg, image_data_FL)) for cell_id, cell in six.iteritems(Cells)]
    pool.close()
    pool.join()

# parralel function for each cell
def foci_cell(cell_id, cell, t0, image_data_seg, image_data_FL):
    '''find foci in a cell, single instance to be called by the foci_analysis_pool for parallel processing.
    '''
    disp_l = []
    disp_w = []
    foci_h = []
    # foci_stack = np.zeros((np.size(cell.times),
    #                        image_data_seg[0,:,:].shape[0], image_data_seg[0,:,:].shape[1]))

    # Go through each time point of this cell
    for t in cell.times:
        # retrieve this timepoint and images.
        image_data_temp = image_data_FL[t-t0,:,:]
        image_data_temp_seg = image_data_seg[t-t0,:,:]

        # find foci as long as there is information in the fluorescent image
        if np.sum(image_data_temp) != 0:
            disp_l_tmp, disp_w_tmp, foci_h_tmp = foci_lap(image_data_temp_seg,
                                                          image_data_temp, cell, t)

            disp_l.append(disp_l_tmp)
            disp_w.append(disp_w_tmp)
            foci_h.append(foci_h_tmp)

        # if there is no information, append an empty list.
        # Should this be NaN?
        else:
            disp_l.append([])
            disp_w.append([])
            foci_h.append([])
            # foci_stack[i] = image_data_temp_seg

    # add information to the cell (will replace old data)
    cell.disp_l = disp_l
    cell.disp_w = disp_w
    cell.foci_h = foci_h

# actual worker function for foci detection
def foci_lap(img, img_foci, cell, t):
    '''foci_lap finds foci using a laplacian convolution then fits a 2D
    Gaussian.

    The returned information are the parameters of this Gaussian.
    All the information is returned in the form of np.arrays which are the
    length of the number of found foci across all cells in the image.

    Parameters
    ----------
    img : 2D np.array
        phase contrast or bright field image. Only used for debug
    img_foci : 2D np.array
        fluorescent image with foci.
    cell : cell object
    t : int
        time point to which the images correspond

    Returns
    -------
    disp_l : 1D np.array
        displacement on long axis, in px, of a foci from the center of the cell
    disp_w : 1D np.array
        displacement on short axis, in px, of a foci from the center of the cell
    foci_h : 1D np.array
        Foci "height." Sum of the intensity of the gaussian fitting area.
    '''

    # pull out useful information for just this time point
    i = cell.times.index(t) # find position of the time point in lists (time points may be missing)
    bbox = cell.bboxes[i]
    orientation = cell.orientations[i]
    centroid = cell.centroids[i]
    region = cell.labels[i]

    # declare arrays which will hold foci data
    disp_l = [] # displacement in length of foci from cell center
    disp_w = [] # displacement in width of foci from cell center
    foci_h = [] # foci total amount (from raw image)

    # define parameters for foci finding
    minsig = params['foci']['foci_log_minsig']
    maxsig = params['foci']['foci_log_maxsig']
    thresh = params['foci']['foci_log_thresh']
    peak_med_ratio = params['foci']['foci_log_peak_med_ratio']
    debug_foci = params['foci']['debug_foci']

    # test
    #print ("minsig={:d}  maxsig={:d}  thres={:.4g}  peak_med_ratio={:.2g}".format(minsig,maxsig,thresh,peak_med_ratio))
    # test

    # calculate median cell intensity. Used to filter foci
    img_foci_masked = np.copy(img_foci).astype(np.float)
    img_foci_masked[img != region] = np.nan
    cell_fl_median = np.nanmedian(img_foci_masked)
    cell_fl_mean = np.nanmean(img_foci_masked)

    img_foci_masked[img != region] = 0

    # subtract this value from the cell
    if False:
        img_foci = img_foci.astype('int32') - cell_fl_median.astype('int32')
        img_foci[img_foci < 0] = 0
        img_foci = img_foci.astype('uint16')

    # int_mask = np.zeros(img_foci.shape, np.uint8)
    # avg_int = cv2.mean(img_foci, mask=int_mask)
    # avg_int = avg_int[0]

    # print('median', cell_fl_median)

    # find blobs using difference of gaussian
    over_lap = .95 # if two blobs overlap by more than this fraction, smaller blob is cut
    numsig = (maxsig - minsig + 1) # number of division to consider between min ang max sig
    blobs = blob_log(img_foci_masked, min_sigma=minsig, max_sigma=maxsig,
                     overlap=over_lap, num_sigma=numsig, threshold=thresh)

    # these will hold information about foci position temporarily
    x_blob, y_blob, r_blob = [], [], []
    x_gaus, y_gaus, w_gaus = [], [], []

    # loop through each potential foci
    for blob in blobs:
        yloc, xloc, sig = blob # x location, y location, and sigma of gaus
        xloc = int(np.around(xloc)) # switch to int for slicing images
        yloc = int(np.around(yloc))
        radius = int(np.ceil(np.sqrt(2)*sig)) # will be used to slice out area around foci

        # ensure blob is inside the bounding box
        # this might be better to check if (xloc, yloc) is in regions.coords
        if yloc > np.int16(bbox[0]) and yloc < np.int16(bbox[2]) and xloc > np.int16(bbox[1]) and xloc < np.int16(bbox[3]):

            x_blob.append(xloc) # for plotting
            y_blob.append(yloc) # for plotting
            r_blob.append(radius)

            # cut out a small image from original image to fit gaussian
            gfit_area = img_foci[yloc-radius:yloc+radius, xloc-radius:xloc+radius]
            # gfit_area_0 = img_foci[max(0, yloc-1*radius):min(img_foci.shape[0], yloc+1*radius),
            #                        max(0, xloc-1*radius):min(img_foci.shape[1], xloc+1*radius)]
            gfit_area_fixed = img_foci[yloc-maxsig:yloc+maxsig, xloc-maxsig:xloc+maxsig]

            # fit gaussian to proposed foci in small box
            p = fitgaussian(gfit_area)
            (peak_fit, x_fit, y_fit, w_fit) = p

            # print('peak', peak_fit)
            if x_fit <= 0 or x_fit >= radius*2 or y_fit <= 0 or y_fit >= radius*2:
                if debug_foci: print('Throw out foci (gaus fit not in gfit_area)')
                continue
            elif peak_fit/cell_fl_median < peak_med_ratio:
                if debug_foci: print('Peak does not pass height test.')
                continue
            else:
                # find x and y position relative to the whole image (convert from small box)
                x_rel = int(xloc - radius + x_fit)
                y_rel = int(yloc - radius + y_fit)
                x_gaus = np.append(x_gaus, x_rel) # for plotting
                y_gaus = np.append(y_gaus, y_rel) # for plotting
                w_gaus = np.append(w_gaus, w_fit) # for plotting

                if debug_foci: print('x', xloc, x_rel, x_fit, 'y', yloc, y_rel, y_fit, 'w', sig, radius, w_fit, 'h', np.sum(gfit_area), np.sum(gfit_area_fixed), peak_fit)

                # calculate distance of foci from middle of cell (scikit image)
                if orientation < 0:
                    orientation = np.pi+orientation
                disp_y = (y_rel-centroid[0])*np.sin(orientation) - (x_rel-centroid[1])*np.cos(orientation)
                disp_x = (y_rel-centroid[0])*np.cos(orientation) + (x_rel-centroid[1])*np.sin(orientation)

                # append foci information to the list
                disp_l = np.append(disp_l, disp_y)
                disp_w = np.append(disp_w, disp_x)
                foci_h = np.append(foci_h, np.sum(gfit_area_fixed))
                # foci_h = np.append(foci_h, peak_fit)
        else:
            if debug_foci:
                print ('Blob not in bounding box.')

    # draw foci on image for quality control
    if debug_foci:
        outputdir = os.path.join(params['ana_dir'], 'debug_foci')
        if not os.path.isdir(outputdir):
            os.makedirs(outputdir)

        # print(np.min(gfit_area), np.max(gfit_area), gfit_median, avg_int, peak)
        # processing of image
        fig = plt.figure(figsize=(12,12))
        ax = fig.add_subplot(1,5,1)
        plt.title('fluor image')
        plt.imshow(img_foci, interpolation='nearest', cmap='gray')
        ax = fig.add_subplot(1,5,2)
        ax.set_title('segmented image')
        ax.imshow(img, interpolation='nearest', cmap='gray')

        ax = fig.add_subplot(1,5,3)
        ax.set_title('DoG blobs')
        ax.imshow(img_foci, interpolation='nearest', cmap='gray')
        # add circles for where the blobs are
        for i, spot in enumerate(x_blob):
            foci_center = Ellipse([x_blob[i], y_blob[i]], r_blob[i], r_blob[i],
                                  color=(1.0, 1.0, 0), linewidth=2, fill=False, alpha=0.5)
            ax.add_patch(foci_center)

        # show the shape of the gaussian for recorded foci
        ax = fig.add_subplot(1,5,4)
        ax.set_title('final foci')
        ax.imshow(img_foci, interpolation='nearest', cmap='gray')
        # print foci that pass and had gaussians fit
        for i, spot in enumerate(x_gaus):
            foci_ellipse = Ellipse([x_gaus[i], y_gaus[i]], w_gaus[i], w_gaus[i],
                                    color=(0, 1.0, 0.0), linewidth=2, fill=False, alpha=0.5)
            ax.add_patch(foci_ellipse)

        ax = fig.add_subplot(1,5,5)
        ax.set_title('overlay')
        ax.imshow(img, interpolation='nearest', cmap='gray')
        # print foci that pass and had gaussians fit
        for i, spot in enumerate(x_gaus):
            foci_ellipse = Ellipse([x_gaus[i], y_gaus[i]], 3, 3,
                                    color=(1.0, 1.0, 0), linewidth=2, fill=False, alpha=0.5)
            ax.add_patch(foci_ellipse)

        #plt.show()
        filename = 'foci_' + cell.id + '_time{:04d}'.format(t) + '.pdf'
        fileout = os.path.join(outputdir,filename)
        fig.savefig(fileout, bbox_inches='tight', pad_inches=0)
        print (fileout)
        plt.close('all')
        nblobs = len(blobs)
        print ("nblobs = {:d}".format(nblobs))

    # img_overlay = img
    # for i, spot in enumerate(xx):
    #     y_temp = int(yy[i])
    #     x_temp = int(xx[i])
    #
    #     img_overlay[y_temp-1,x_temp-1] = 12
    #     img_overlay[y_temp-1,x_temp] = 12
    #     img_overlay[y_temp-1,x_temp+1] = 12
    #     img_overlay[y_temp,x_temp-1] = 12
    #     img_overlay[y_temp,x_temp] = 12
    #     img_overlay[y_temp,x_temp+1] = 12
    #     img_overlay[y_temp+1,x_temp-1] = 12
    #     img_overlay[y_temp+1,x_temp] = 12
    #     img_overlay[y_temp+1,x_temp+1] = 12

    return disp_l, disp_w, foci_h

# actual worker function for foci detection
def foci_info_unet(foci, Cells, specs, time_table, channel_name='sub_c2'):
    '''foci_info_unet operates on cells in which foci have been found using
    using Unet.

    Parameters
    ----------
    Foci : empty dictionary for Focus objects to be placed into
    Cells : dictionary of Cell objects to which foci will be added
    specs : dictionary containing information on which fov/peak ids
        are to be used, and which are to be excluded from analysis
    time_table : dictionary containing information on which time
        points correspond to which absolute times in seconds
    channel_name : name of fluorescent channel for reading in
        fluorescence images for focus quantification

    Returns
    -------
    Updates cell information in Cells in-place.
    Cells must have .foci attribute
    '''

    # iterate over each fov in specs
    for fov_id,fov_peaks in specs.items():

        # keep cells with this fov_id
        fov_cells = filter_cells(Cells, attr='fov', val=fov_id)

        # iterate over each peak in fov
        for peak_id,peak_value in fov_peaks.items():

            # print(fov_id, peak_id)
            # keep cells with this peak_id
            peak_cells = filter_cells(fov_cells, attr='peak', val=peak_id)

            # if peak_id's value is not 1, go to next peak
            if peak_value != 1:
                continue

            print("Analyzing foci in experiment {}, channel {}, fov {}, peak {}.".format(params['experiment_name'], channel_name, fov_id, peak_id))
            # Load fluorescent images and segmented images for this channel
            fl_stack = load_stack(fov_id, peak_id, color=channel_name)
            seg_foci_stack = load_stack(fov_id, peak_id, color='foci_seg_unet')
            seg_cell_stack = load_stack(fov_id, peak_id, color='seg_unet')

            # loop over each frame
            for frame in range(fl_stack.shape[0]):

                fl_img = fl_stack[frame, ...]
                seg_foci_img = seg_foci_stack[frame, ...]
                seg_cell_img = seg_cell_stack[frame, ...]

                # if there are no foci in this frame, move to next frame
                if np.max(seg_foci_img) == 0:
                    continue
                # if there are no cells in this fov/peak/frame, move to next frame
                if np.max(seg_cell_img) == 0:
                    continue

                t = frame+1
                frame_cells = filter_cells_containing_val_in_attr(peak_cells, attr='times', val=t)
                # loop over focus regions in this frame
                focus_regions = measure.regionprops(seg_foci_img)

                # compare this frame's foci to prior frame's foci for tracking
                if frame > 0:
                    prior_seg_foci_img = seg_foci_stack[frame-1, ...]

                    fov_foci = filter_cells(foci,
                                            attr='fov',
                                            val=fov_id)
                    peak_foci = filter_cells(fov_foci,
                                             attr='peak',
                                             val=peak_id)
                    prior_frame_foci = filter_cells_containing_val_in_attr(peak_foci, attr='times', val=t-1)

                    # if there were foci in prior frame, do stuff
                    if len(prior_frame_foci) > 0:
                        prior_regions = measure.regionprops(prior_seg_foci_img)

                        # compare_array is prior_focus_number x this_focus_number
                        #   contains dice indices for each pairwise comparison
                        #   between focus positions
                        compare_array = np.zeros((np.max(prior_seg_foci_img),
                                                np.max(seg_foci_img)))
                        # populate the array with dice indices
                        for prior_focus_idx in range(np.max(prior_seg_foci_img)):

                            prior_focus_mask = np.zeros(seg_foci_img.shape)
                            prior_focus_mask[prior_seg_foci_img == (prior_focus_idx + 1)] = 1

                            # apply gaussian blur with sigma=1 to prior focus mask
                            sig = 1
                            gaus_1 = filters.gaussian(prior_focus_mask, sigma=sig)

                            for this_focus_idx in range(np.max(seg_foci_img)):

                                this_focus_mask = np.zeros(seg_foci_img.shape)
                                this_focus_mask[seg_foci_img == (this_focus_idx + 1)] = 1

                                # apply gaussian blur with sigma=1 to this focus mask
                                gaus_2 = filters.gaussian(this_focus_mask, sigma=sig)
                                # multiply the two images and place max into campare_array
                                product = gaus_1 * gaus_2
                                compare_array[prior_focus_idx, this_focus_idx] = np.max(product)

                        # which rows of each column are maximum jaccard index?
                        max_inds = np.argmax(compare_array, axis=0)
                        # because np.argmax returns zero if all rows are equal, we
                        #   need to evaluate if all rows are equal.
                        #   If std_dev is zero, then all were equal,
                        #   and we omit that index from consideration for
                        #   focus tracking.
                        sd_vals = np.std(compare_array, axis=0)
                        tracked_inds = np.where(sd_vals > 0)[0]
                        # if there is an index from a tracked focus, do this
                        if tracked_inds.size > 0:

                            for tracked_idx in tracked_inds:
                                # grab this frame's region belonging to tracked focus
                                tracked_label = tracked_idx + 1
                                (tracked_region_idx, tracked_region) = [(_,reg) for _,reg in enumerate(focus_regions) if reg.label == tracked_label][0]
                                # pop the region from focus_regions
                                del focus_regions[tracked_region_idx]

                                # grab prior frame's region belonging to tracked focus
                                prior_tracked_label = max_inds[tracked_idx] + 1
                                # prior_tracked_region = [reg for reg in prior_regions if reg.label == prior_tracked_label][0]

                                # grab the focus for which the prior_tracked_label is in
                                #   any of the labels in the prior focus from the prior time
                                prior_tracked_foci = filter_foci(
                                    prior_frame_foci,
                                    label=prior_tracked_label,
                                    t = t-1,
                                    debug=False
                                )

                                prior_tracked_focus = [val for val in prior_tracked_foci.values()][0]

                                # determine which cell this focus belongs to
                                for cell_id,cell in frame_cells.items():

                                    cell_idx = cell.times.index(t)
                                    cell_label = cell.labels[cell_idx]

                                    masked_cell_img = np.zeros(seg_cell_img.shape)
                                    masked_cell_img[seg_cell_img == cell_label] = 1

                                    masked_focus_img = np.zeros(seg_foci_img.shape)
                                    masked_focus_img[seg_foci_img == tracked_region.label] = 1

                                    intersect_img = masked_cell_img + masked_focus_img

                                    pixels_two = len(np.where(intersect_img == 2))
                                    pixels_one = len(np.where(masked_focus_img == 1))

                                    # if over half the focus is within this cell, do the following
                                    if pixels_two/pixels_one >= 0.5:

                                        prior_tracked_focus.grow(
                                            region=tracked_region,
                                            t=t,
                                            seg_img=seg_foci_img,
                                            intensity_image=fl_img,
                                            current_cell=cell
                                        )

                # after tracking foci, those that were tracked have been removed from focus_regions list
                # now we check if any regions remain in the list
                # if there are any remaining, instantiate new foci
                if len(focus_regions) > 0:
                    new_ids = []

                    for focus_region in focus_regions:

                        # make the focus_id
                        new_id = create_focus_id(
                            region = focus_region,
                            t = t,
                            peak = peak_id,
                            fov = fov_id,
                            experiment_name = params['experiment_name'])
                        # populate list for later checking if any are missing
                        # from foci dictionary's keys
                        new_ids.append(new_id)

                        # determine which cell this focus belongs to
                        for cell_id,cell in frame_cells.items():

                            cell_idx = cell.times.index(t)
                            cell_label = cell.labels[cell_idx]

                            masked_cell_img = np.zeros(seg_cell_img.shape)
                            masked_cell_img[seg_cell_img == cell_label] = 1

                            masked_focus_img = np.zeros(seg_foci_img.shape)
                            masked_focus_img[seg_foci_img == focus_region.label] = 1

                            intersect_img = masked_cell_img + masked_focus_img

                            pixels_two = len(np.where(intersect_img == 2))
                            pixels_one = len(np.where(masked_focus_img == 1))

                            # if over half the focus is within this cell, do the following
                            if pixels_two/pixels_one >= 0.5:
                                # set up the focus
                                # if no foci in cell, just add this one.

                                foci[new_id] = Focus(cell = cell,
                                                     region = focus_region,
                                                     seg_img = seg_foci_img,
                                                     intensity_image = fl_img,
                                                     t = t)

                    for new_id in new_ids:
                        # if new_id is not a key in the foci dictionary,
                        #   that suggests the focus doesn't overlap well
                        #   with any cells in this frame, so we'll relabel
                        #   this frame of seg_foci_stack to zero for that
                        #   focus to avoid trying to track a focus
                        #   that doesn't exist.
                        if new_id not in foci:

                            # get label of new_id's region
                            this_label = int(new_id[-2:])
                            # set pixels in this frame that match this label to 0
                            seg_foci_stack[frame, seg_foci_img == this_label] = 0

    return

def update_cell_foci(cells, foci):
    '''Updates cells' .foci attribute in-place using information
    in foci dictionary
    '''
    for focus_id, focus in foci.items():
        for cell in focus.cells:

            cell_id = cell.id
            cells[cell_id].foci[focus_id] = focus

# finds best fit for 2d gaussian using functin above
def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit
    if params are not provided, they are calculated from the moments
    params should be (height, x, y, width_x, width_y)"""
    gparams = moments(data) # create guess parameters.
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) - data)
    p, success = leastsq(errorfunction, gparams)
    return p

# calculate dice coefficient for two blobs
def dice_coeff_foci(mask_1_f, mask_2_f):
    '''Accepts two flattened numpy arrays from
    binary masks of two blobs and compares them
    using the dice metric.

    Returns a single dice score.
    '''
    intersection = np.sum(mask_1_f * mask_2_f)
    score = (2. * intersection) / (np.sum(mask_1_f) + np.sum(mask_2_f))
    return score

# returnes a 2D gaussian function
def gaussian(height, center_x, center_y, width):
    '''Returns a gaussian function with the given parameters. It is a circular gaussian.
    width is 2*sigma x or y
    '''
    # return lambda x,y: height*np.exp(-(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)
    return lambda x,y: height*np.exp(-(((center_x-x)/width)**2+((center_y-y)/width)**2)/2)

# moments of a 2D gaussian
def moments(data):
    '''
    Returns (height, x, y, width_x, width_y)
    The (circular) gaussian parameters of a 2D distribution by calculating its moments.
    width_x and width_y are 2*sigma x and sigma y of the guassian.
    '''
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width = float(np.sqrt(abs((np.arange(col.size)-y)**2*col).sum()/col.sum()))
    row = data[int(x), :]
    # width_y = np.sqrt(abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width

# returns a 1D gaussian function
def gaussian1d(x, height, mean, sigma):
    '''
    x : data
    height : height
    mean : center
    sigma : RMS width
    '''
    return height * np.exp(-(x-mean)**2 / (2*sigma**2))

# analyze ring fluroescence.
def ring_analysis(fov_id, peak_id, Cells, ring_plane='c2'):
    '''Add information to the Cell objects about the location of the Z ring. Sums the fluorescent channel along the long axis of the cell. This can be plotted directly to give a good idea about the development of the ring. Also fits a gaussian to the profile.

    Parameters
    ----------
    fov_id : int
        FOV number of the lineage to analyze.
    peak_id : int
        Peak number of the lineage to analyze.
    Cells : dict of Cell objects (from a Lineages dictionary)
        Cells should be prefiltered to match fov_id and peak_id.
    ring_plane : str
        The suffix of the channel to analyze. 'c1', 'c2', 'sub_c2', etc.

    Usage
    -----
    for fov_id, peaks in Lineages.iteritems():
        for peak_id, Cells in peaks.iteritems():
            mm3.ring_analysis(fov_id, peak_id, Cells, ring_plane='sub_c2')
    '''

    peak_width_guess = 2

    # Load data
    ring_stack = load_stack(fov_id, peak_id, color=ring_plane)
    seg_stack = load_stack(fov_id, peak_id, color='seg_unet')

    # Load time table to determine first image index.
    time_table = load_time_table()
    times_all = np.array(np.sort(time_table[fov_id].keys()), np.int_)
    t0 = times_all[0] # first time index

    # Loop through cells
    for Cell in Cells.values():

        # initialize ring data arrays for cell
        Cell.ring_locs = []
        Cell.ring_heights = []
        Cell.ring_widths = []
        Cell.ring_medians = []
        Cell.ring_profiles = []

        # loop through each time point for this cell
        for n, t in enumerate(Cell.times):
            # Make mask of fluorescent channel using segmented image
            ring_image_masked = np.copy(ring_stack[t-t0])
            ring_image_masked[seg_stack[t-t0] != Cell.labels[n]] = 0

            # Sum along long axis, use the profile_line function from skimage
            # Use orientation of cell as calculated from the ellipsoid fit,
            # the known length of the cell from the feret diameter,
            # and a width that is greater than the cell width.

            # find endpoints of line
            centroid = Cell.centroids[n]
            orientation = Cell.orientations[n]
            length = Cell.lengths[n]
            width = Cell.widths[n] * 1.25

            # give 2 pixel buffer to each end to capture area outside cell.
            p1 = (centroid[0] - np.sin(orientation) * (length+4)/2,
                  centroid[1] - np.cos(orientation) * (length+4)/2)
            p2 = (centroid[0] + np.sin(orientation) * (length+4)/2,
                  centroid[1] + np.cos(orientation) * (length+4)/2)

            # ensure old pole is always first point
            if p1[0] > p2[0]:
                p1, p2 = p2, p1 # python is cool

            profile = profile_line(ring_image_masked, p1, p2, linewidth=width,
                                   order=1, mode='constant', cval=0)
            profile_indicies = np.arange(len(profile))

            # subtract median from profile, using non-zero values for median
            profile_median = np.median(profile[np.nonzero(profile)])
            profile_sub = profile - profile_median
            profile_sub[profile_sub < 0] = 0

            # find peak position simply using maximum.
            peak_index = np.argmax(profile)
            peak_height = profile[peak_index]
            peak_height_sub = profile_sub[peak_index]

            try:
                # Fit gaussian
                p_guess = [peak_height_sub, peak_index, peak_width_guess]
                popt, pcov = curve_fit(gaussian1d, profile_indicies,
                                       profile_sub, p0=p_guess)

                peak_width = popt[2]
            except:
                # information('Ring gaussian fit failed. {} {} {}'.format(fov_id, peak_id, t))
                peak_width = np.float('NaN')

            # Add data to cells
            Cell.ring_locs.append(peak_index - 3) # minus 3 because we added 2 before and line_profile adds 1.
            Cell.ring_heights.append(peak_height)
            Cell.ring_widths.append(peak_width)
            Cell.ring_medians.append(profile_median)
            Cell.ring_profiles.append(profile) # append whole profile

    return

# Calculate Y projection intensity of a fluorecent channel per cell
def profile_analysis(fov_id, peak_id, Cells, profile_plane='c2'):
    '''Calculate profile of plane along cell and add information to Cell object. Sums the fluorescent channel along the long axis of the cell.

    Parameters
    ----------
    fov_id : int
        FOV number of the lineage to analyze.
    peak_id : int
        Peak number of the lineage to analyze.
    Cells : dict of Cell objects (from a Lineages dictionary)
        Cells should be prefiltered to match fov_id and peak_id.
    profile_plane : str
        The suffix of the channel to analyze. 'c1', 'c2', 'sub_c2', etc.

    Usage
    -----

    '''

    # Load data
    fl_stack = load_stack(fov_id, peak_id, color=profile_plane)
    seg_stack = load_stack(fov_id, peak_id, color='seg_unet')

    # Load time table to determine first image index.
    # load_time_table()
    times_all = []
    for fov in params['time_table']:
        times_all = np.append(times_all, list(params['time_table'][fov].keys()))
    times_all = np.unique(times_all)
    times_all = np.sort(times_all)
    times_all = np.array(times_all,np.int_)
    t0 = times_all[0] # first time index

    # Loop through cells
    for Cell in Cells.values():

        # initialize ring data arrays for cell
        fl_profiles = []

        # loop through each time point for this cell
        for n, t in enumerate(Cell.times):
            # Make mask of fluorescent channel using segmented image
            image_masked = np.copy(fl_stack[t-t0])
            image_masked[seg_stack[t-t0] != Cell.labels[n]] = 0

            # Sum along long axis, use the profile_line function from skimage
            # Use orientation of cell as calculated from the ellipsoid fit,
            # the known length of the cell from the feret diameter,
            # and a width that is greater than the cell width.

            # find endpoints of line
            centroid = Cell.centroids[n]
            orientation = Cell.orientations[n]
            length = Cell.lengths[n]
            width = Cell.widths[n] * 1.25

            # give 2 pixel buffer to each end to capture area outside cell.
            p1 = (centroid[0] - np.sin(orientation) * (length+4)/2,
                  centroid[1] - np.cos(orientation) * (length+4)/2)
            p2 = (centroid[0] + np.sin(orientation) * (length+4)/2,
                  centroid[1] + np.cos(orientation) * (length+4)/2)

            # ensure old pole is always first point
            if p1[0] > p2[0]:
                p1, p2 = p2, p1 # python is cool

            profile = profile_line(image_masked, p1, p2, linewidth=width,
                                   order=1, mode='constant', cval=0)

            fl_profiles.append(profile)

        # append whole profile, using plane name
        setattr(Cell, 'fl_profiles_'+profile_plane, fl_profiles)

    return

# Calculate X projection at midcell and quarter position
def x_profile_analysis(fov_id, peak_id, Cells, profile_plane='sub_c2'):
    '''Calculate profile of plane along cell and add information to Cell object. Sums the fluorescent channel along the long axis of the cell.

    Parameters
    ----------
    fov_id : int
        FOV number of the lineage to analyze.
    peak_id : int
        Peak number of the lineage to analyze.
    Cells : dict of Cell objects (from a Lineages dictionary)
        Cells should be prefiltered to match fov_id and peak_id.
    profile_plane : str
        The suffix of the channel to analyze. 'c1', 'c2', 'sub_c2', etc.

    '''

    # width to sum over in pixels
    line_width = 6

    # Load data
    fl_stack = load_stack(fov_id, peak_id, color=profile_plane)
    seg_stack = load_stack(fov_id, peak_id, color='seg_unet')

    # Load time table to determine first image index.
    time_table = load_time_table()
    t0 = times_all[0] # first time index

    # Loop through cells
    for Cell in Cells.values():

        # print(Cell.id)

        # initialize data arrays for cell
        midcell_fl_profiles = []
        midcell_pts = []
        quarter_fl_profiles = []
        quarter_pts = []

        # loop through each time point for this cell
        for n, t in enumerate(Cell.times):
            # Make mask of fluorescent channel using segmented image
            image_masked = np.copy(fl_stack[t-t0])
            # image_masked[seg_stack[t-t0] != Cell.labels[n]] = 0

            # Sum along short axis, use the profile_line function from skimage
            # Use orientation of cell as calculated from the ellipsoid fit,
            # the known length of the cell from the feret diameter,
            # and a width that is greater than the cell width.

            # find end points for summing
            centroid = Cell.centroids[n]
            orientation = Cell.orientations[n]
            length = Cell.lengths[n]
            width = Cell.widths[n]

            # midcell
            # give 2 pixel buffer to each end to capture area outside cell.
            md_p1 = (centroid[0] - np.cos(orientation) * (width+8)/2,
                     centroid[1] - np.sin(orientation) * (width+8)/2)
            md_p2 = (centroid[0] + np.cos(orientation) * (width+8)/2,
                     centroid[1] + np.sin(orientation) * (width+8)/2)

            # ensure lower x point is always first
            if md_p1[1] > md_p2[1]:
                md_p1, md_p2 = md_p2, md_p1 # python is cool
            midcell_pts.append((md_p1, md_p2))

            # print(t, centroid, orientation, md_p1, md_p2)
            md_profile = profile_line(image_masked, md_p1, md_p2,
                                      linewidth=line_width,
                                      order=1, mode='constant', cval=0)
            midcell_fl_profiles.append(md_profile)

            # quarter position, want to measure at mother end
            if orientation > 0:
                yq = centroid[0] - np.sin(orientation) * 0.5 * (length * 0.5)
                xq = centroid[1] + np.cos(orientation) * 0.5 * (length * 0.5)
            else:
                yq = centroid[0] + np.sin(orientation) * 0.5 * (length * 0.5)
                xq = centroid[1] - np.cos(orientation) * 0.5 * (length * 0.5)

            q_p1 = (yq - np.cos(orientation) * (width+8)/2,
                    xq - np.sin(orientation) * (width+8)/2)
            q_p2 = (yq + np.cos(orientation) * (width+8)/2,
                    xq + np.sin(orientation) * (width+8)/2)

            if q_p1[1] > q_p2[1]:
                q_p1, q_p2 = q_p2, q_p1
            quarter_pts.append((q_p1, q_p2))

            q_profile = profile_line(image_masked, q_p1, q_p2,
                                     linewidth=line_width,
                                     order=1, mode='constant', cval=0)
            quarter_fl_profiles.append(q_profile)

        # append whole profile, using plane name
        setattr(Cell, 'fl_md_profiles_'+profile_plane, midcell_fl_profiles)
        setattr(Cell, 'midcell_pts', midcell_pts)
        setattr(Cell, 'fl_quar_profiles_'+profile_plane, quarter_fl_profiles)
        setattr(Cell, 'quarter_pts', quarter_pts)

    return

# Calculate X projection at midcell and quarter position
def constriction_analysis(fov_id, peak_id, Cells, plane='sub_c1'):
    '''Calculate profile of plane along cell and add information to Cell object. Sums the fluorescent channel along the long axis of the cell.

    Parameters
    ----------
    fov_id : int
        FOV number of the lineage to analyze.
    peak_id : int
        Peak number of the lineage to analyze.
    Cells : dict of Cell objects (from a Lineages dictionary)
        Cells should be prefiltered to match fov_id and peak_id.
    plane : str
        The suffix of the channel to analyze. 'c1', 'c2', 'sub_c2', etc.

    '''

    # Load data
    sub_stack = load_stack(fov_id, peak_id, color=plane)
    seg_stack = load_stack(fov_id, peak_id, color='seg_unet')

    # Load time table to determine first image index.
    time_table = load_time_table()
    t0 = times_all[0] # first time index

    # Loop through cells
    for Cell in Cells.values():

        # print(Cell.id)

        # initialize data arrays for cell
        midcell_imgs = [] # Just a small image of the midcell
        midcell_sums = [] # holds sum of pixel values in midcell area
        midcell_vars = [] # variances

        coeffs_2nd = [] # coeffiients for fitting

        # loop through each time point for this cell
        for n, t in enumerate(Cell.times):
            # Make mask of subtracted image
            image_masked = np.copy(sub_stack[t-t0])
            image_masked[seg_stack[t-t0] != Cell.labels[n]] = 0

            # make a box aroud the midcell from which to calculate stats
            centroid = Cell.centroids[n]
            orientation = Cell.orientations[n]
            length = Cell.lengths[n]
            slice_l = np.around(length/4).astype('int')
            width = Cell.widths[n]
            slice_w = np.around(width/2).astype('int') + 3
            slice_l = slice_w

            # rotate box and then slice out area around centroid
            if orientation > 0:
                rot_angle = 90 - orientation * (180 / np.pi)
            else:
                rot_angle = -90 - orientation * (180 / np.pi)

            rotated = rotate(image_masked, rot_angle, resize=False,
                             center=centroid, mode='constant', cval=0)
            centroid = [int(coord) for coord in centroid]
            cropped_md = rotated[centroid[0]-slice_l:centroid[0]+slice_l,
                                 centroid[1]-slice_w:centroid[1]+slice_w]

            # sum across with widths
            md_widths = np.array([np.around(sum(row),5) for row in cropped_md])

            # fit widths
            x_pixels = np.arange(1, len(md_widths)+1) - (len(md_widths)+1)/2
            p_guess = (1, 1, 1)
            popt, pcov = curve_fit(poly2o, x_pixels, md_widths, p0=p_guess)
            a, b, c = popt
            # save coefficients
            coeffs_2nd.append(a)

            # go backwards through coeeficients and find at which index the coeff becomes negative.
            constriction_index = None
            for i, coeff in enumerate(reversed(coeffs_2nd), start=0):
                if coeff < 0:
                    constriction_index = i
                    break

            # fix index
            if constriction_index == None:
                constriction_index = len(coeffs_2nd) - 1 # make it last point if it was not found
            else:
                constriction_index = len(coeffs_2nd) - constriction_index - 1

            # midcell_imgs.append(cropped_md)
            # midcell_sums.append(np.sum(cropped_md))
            # midcell_vars.append(np.var(cropped_md))

        # append whole profile, using plane name
        # setattr(Cell, 'md_image_'+plane, midcell_imgs)
        # setattr(Cell, 'md_sums', midcell_sums)
        # setattr(Cell, 'md_vars', midcell_vars)

        setattr(Cell, 'constriction_time', Cell.times[constriction_index])

    return

# Calculate pole age of cell and add as attribute
def calculate_pole_age(Cells):
    '''Finds the pole age of each end of the cell. Adds this information to the cell object.

    This should maybe move to helpers
    '''

    # run through once and set up default
    for cell_id, cell_tmp in six.iteritems(Cells):
        cell_tmp.poleage = None

    for cell_id, cell_tmp in six.iteritems(Cells):
        # start from r1 cells which have r1 parents in the list.
        # these cells are old pole mothers.
    #     if cell_tmp.parent in Cells and cell_tmp.birth_label == 1:

        # less stringent requirement that the cell just r1
        if cell_tmp.birth_label == 1:

            # label this cell
            cell_tmp.poleage = (1000, 0) # closed end age first, 1000 for old pole.

            # label the daughter cell 01 if it is in the list
            if cell_tmp.daughters[1] in Cells:
                # sets poleage of this cell and recursively goes through descendents.
                Cells = set_poleages(cell_tmp.daughters[1], 1, Cells)

    return Cells

def set_poleages(cell_id, daughter_index, Cells):
    '''Determines pole ages for cells. Only for cells which are not old-pole mother.'''

    parent_poleage = Cells[Cells[cell_id].parent].poleage

    # the lower daughter
    if daughter_index == 0:
        Cells[cell_id].poleage = (parent_poleage[0]+1, 0)
    elif daughter_index == 1:
        Cells[cell_id].poleage = (0, parent_poleage[1]+1)

    for i, daughter_id in enumerate(Cells[cell_id].daughters):
        if daughter_id in Cells:
            Cells = set_poleages(daughter_id, i, Cells)

    return Cells

def poly2o(x, a, b, c):
    '''Second order polynomial of the form
       y = a*x^2 + bx + c'''

    return a*x**2 + b*x + c
