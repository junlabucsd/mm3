#! /usr/bin/env python3
from __future__ import print_function, division

from PyQt5.QtWidgets import (QApplication, QMainWindow, QMenuBar, QRadioButton,
    QMenu, QAction, QButtonGroup, QFileDialog, QVBoxLayout, QHBoxLayout, QWidget,
    QLabel, QGridLayout, QAction, QDockWidget, QPushButton, QInputDialog,
    QGraphicsScene,
    QCheckBox,
    QGraphicsItem,
    QGridLayout, QGraphicsLineItem, QGraphicsPathItem, QGraphicsPixmapItem,
    QGraphicsEllipseItem, QGraphicsTextItem)
from PyQt5.QtGui import QIcon, QImage, QPainter, QPen, QPixmap, qGray, QColor
from PyQt5.QtCore import Qt, QPoint, QRectF
from skimage import io, img_as_ubyte, color, draw, measure
import numpy as np
import sys
import re
import os
import inspect
import yaml
import multiprocessing

import argparse
import glob
from pprint import pprint # for human readable file output
import pickle as pickle
import random
import pandas as pd
import skimage.morphology
import math
from skimage.measure import profile_line # used for ring an nucleoid analysis
from skimage.exposure import equalize_adapthist
import tifffile as tiff

# realpath() will make your script run, even if you symlink it
cmd_folder = os.path.realpath(os.path.abspath(
                              os.path.split(inspect.getfile(inspect.currentframe()))[0]))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

# This makes python look for modules in directory above this one
mm3_dir = os.path.realpath(os.path.abspath(
                                 os.path.join(os.path.split(inspect.getfile(
                                 inspect.currentframe()))[0], '../..')))
if mm3_dir not in sys.path:
    sys.path.insert(0, mm3_dir)

import mm3_helpers as mm3
import mm3_plots
import mm3_GUI_helpers as GUI

if __name__ == "__main__":

    # set switches and parameters
    parser = argparse.ArgumentParser(prog='python mm3_updateTrackingData.py',
                                     description='Update existing tracking data for use as training data by neural network.')
    parser.add_argument('-f', '--paramfile', type=str,
                        required=True, help='Yaml file containing parameters.')
    parser.add_argument('-c', '--cellfile', type=str,
                        required=False, help='pkl file containing cell objects')
    parser.add_argument('-t', '--tracefile', type=str,
                        required=False, help='pickle file containing replication trace objects')

    namespace = parser.parse_args()

    # Load the project parameters file
    mm3.information('Loading experiment parameters.')

    if namespace.paramfile:
        param_file_path = namespace.paramfile
    else:
        mm3.warning('No param file specified. Using 100X template.')
        param_file_path = 'yaml_templates/params_SJ110_100X.yaml'

    global params
    params = mm3.init_mm3_helpers(param_file_path)

    if namespace.cellfile:
        cell_file_path = os.path.join(params['cell_dir'], namespace.cellfile)
    else:
        cell_file_path = os.path.join(params['cell_dir'], 'complete_cells_foci.pkl')

    if namespace.tracefile:
        trace_file_path = os.path.join(params['cell_dir'], namespace.tracefile)
    else:
        trace_file_path = os.path.join(params['cell_dir'], 'rep_traces.pkl')


    app = QApplication(sys.argv)
    window = GUI.FocusTrackWindow(params,cell_file_path, trace_file_path)
    window.show()
    app.exec_()
