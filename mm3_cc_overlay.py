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


import mm3_helpers as mm3
import mm3_plots
import mm3_GUI_helpers as GUI

if __name__ == "__main__":

    # set switches and parameters
    parser = argparse.ArgumentParser(prog='python mm3_updateTrackingData.py',
                                     description='Update existing tracking data for use as training data by neural network.')
    parser.add_argument('-f', '--paramfile', type=str,
                        required=True, help='Yaml file containing parameters.')
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

    app = QApplication(sys.argv)
    cell_file = 'complete_cells_test_foci.pkl'
    trace_file = 'complete_cells_test_foci_rep_traces.pkl'
    window = GUI.FocusTrackWindow(params,cell_file, trace_file)
    window.show()
    app.exec_()
