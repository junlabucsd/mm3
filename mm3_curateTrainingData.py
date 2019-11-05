#! /usr/bin/env python3
from __future__ import print_function, division

from PyQt5.QtWidgets import QApplication, QMainWindow, QMenuBar, QRadioButton, QMenu, QAction, QButtonGroup, QFileDialog, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QGridLayout, QAction, QDockWidget, QPushButton
from PyQt5.QtGui import QIcon, QImage, QPainter, QPen, QPixmap, qGray, QColor
from PyQt5.QtCore import Qt, QPoint, QRectF
from skimage import io, img_as_ubyte, color, draw
from matplotlib import pyplot as plt

# import modules
import six
import sys
import os
import glob
import re
# import time
import inspect
import argparse
import yaml
from pprint import pprint # for human readable file output
try:
    import cPickle as pickle
except:
    import pickle
import numpy as np
from scipy.io import savemat

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

import mm3_helpers as mm3
import mm3_GUI_helpers as GUI


if __name__ == "__main__":

    # set switches and parameters
    parser = argparse.ArgumentParser(prog='python mm3_Segment.py',
                                     description='Segment cells and create lineages.')
    parser.add_argument('-f', '--paramfile', type=str,
                        required=True, help='Yaml file containing parameters.')
    parser.add_argument('-o', '--fov', type=str,
                        required=False, help='List of fields of view to analyze. Input "1", "1,2,3", etc. ')
    parser.add_argument('-t', '--traindir', type=str,
                        required=True, help='Absolute path to the directory where you want your "images" and "masks" training data directories to be created and images to be saved.')
    parser.add_argument('-c', '--channel', type=int,
                        required=False, default=1,
                        help='Which channel, e.g. phase or some fluorescence image, should be used for creating masks. \
                            Accepts integers. Default is 1, which is usually your phase contrast images.')
    parser.add_argument('-n', '--no_prior_mask', action='store_true',
                        help='Apply this argument is you are making masks de novo, i.e., if no masks exist yet for your images.')
    namespace = parser.parse_args()

    # Load the project parameters file
    mm3.information('Loading experiment parameters.')
    training_dir = namespace.traindir
    if not os.path.exists(training_dir):
        mm3.warning('Training directory not found, making directory.')
        os.makedirs(training_dir)

    if namespace.paramfile:
        param_file_path = namespace.paramfile
    else:
        mm3.warning('No param file specified. Using 100X template.')
        param_file_path = 'yaml_templates/params_SJ110_100X.yaml'
    p = mm3.init_mm3_helpers(param_file_path) # initialized the helper library
    GUI.init_params(param_file_path)

    if namespace.fov:
        user_spec_fovs = [int(val) for val in namespace.fov.split(",")]
    else:
        user_spec_fovs = []

    if not os.path.exists(p['seg_dir']):
        sys.exit("Exiting: Segmentation directory, {}, not found.".format(p['seg_dir']))
    if not os.path.exists(p['chnl_dir']):
        sys.exit("Exiting: Channel directory, {}, not found.".format(p['chnl_dir']))

    specs = mm3.load_specs()
    # make list of FOVs to process (keys of channel_mask file)
    fov_id_list = sorted([fov_id for fov_id in specs.keys()])
    # remove fovs if the user specified so
    if user_spec_fovs:
        fov_id_list[:] = [fov for fov in fov_id_list if fov in user_spec_fovs]

    # set segmentation image name for segmented images
    seg_suffix_finder = re.compile(r'.+(seg_.+)\.tif$')
    test_name = glob.glob(os.path.join(p['seg_dir'],'*xy{:0=3}*.tif'.format(fov_id_list[0])))[0]
    mat = seg_suffix_finder.match(test_name)
    p['seg_img'] = mat.groups()[0] ## be careful here, it is lookgin for segmented images

    # get paired phase file names and mask file names for each fov
    fov_filename_dict = {}
    for fov_id in fov_id_list:
        if namespace.no_prior_mask:
            mask_filenames = None
        else:
            mask_filenames = [os.path.join(p['seg_dir'],fname) for fname in glob.glob(os.path.join(p['seg_dir'],'*xy{:0=3}*{}.tif'.format(fov_id, p['seg_img'])))]
            
        image_filenames = [fname.replace(p['seg_dir'], p['chnl_dir']).replace(p['seg_img'], 'c{}'.format(namespace.channel)) for fname in mask_filenames]

        fov_filename_dict[fov_id] = []
        if mask_filenames is not None:
            for i in range(len(mask_filenames)):
                fov_filename_dict[fov_id].append((image_filenames[i],mask_filenames[i]))
        else:
            fov_filename_dict[fov_id].append((image_filenames[i], None))

    # print([names for names in fov_filename_dict[1]]) # for debugging

    app = QApplication(sys.argv)
    window = GUI.Window(imgPaths=fov_filename_dict, fov_id_list=fov_id_list, training_dir=training_dir)
    window.show()
    app.exec_() # exec is a reserved word in python2, so this is exec_
