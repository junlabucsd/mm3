#! /usr/bin/env python3
from __future__ import print_function, division

from PyQt5.QtWidgets import (QApplication, QMainWindow, QGraphicsScene, QGraphicsView,
                             QRadioButton, QButtonGroup, QVBoxLayout, QHBoxLayout,
                             QWidget, QLabel, QAction, QDockWidget, QPushButton, QGraphicsItem,
                             QGridLayout, QGraphicsLineItem, QGraphicsPathItem, QGraphicsPixmapItem,
                             QGraphicsEllipseItem, QGraphicsTextItem)
from PyQt5.QtGui import (QIcon, QImage, QPainter, QPen, QPixmap, qGray, QColor, QPainterPath, QBrush,
                         QTransform, QPolygonF, QFont)
from PyQt5.QtCore import Qt, QPoint, QRectF, QLineF
from skimage import io, img_as_ubyte, color, draw, measure
import numpy as np
import glob
from pprint import pprint # for human readable file output
import pickle as pickle
import sys
import re
import os
import random
import pandas as pd
from IPython.display import display, HTML
import yaml
import multiprocessing

sys.path.insert(0, '/home/wanglab/src/mm3/') # Jeremy's path to mm3 folder
sys.path.insert(0, '/home/wanglab/src/mm3/aux/')

import mm3_helpers as mm3
import mm3_plots

from matplotlib import pyplot as plt

# def init_params(param_file_path):
#     # load all the parameters into a global dictionary
#     global params
#     with open(param_file_path, 'r') as param_file:
#         params = yaml.safe_load(param_file)
#
#     # set up how to manage cores for multiprocessing
#     params['num_analyzers'] = multiprocessing.cpu_count()
#
#     # useful folder shorthands for opening files
#     params['TIFF_dir'] = os.path.join(params['experiment_directory'], params['image_directory'])
#     params['ana_dir'] = os.path.join(params['experiment_directory'], params['analysis_directory'])
#     params['hdf5_dir'] = os.path.join(params['ana_dir'], 'hdf5')
#     params['chnl_dir'] = os.path.join(params['ana_dir'], 'channels')
#     params['empty_dir'] = os.path.join(params['ana_dir'], 'empties')
#     params['sub_dir'] = os.path.join(params['ana_dir'], 'subtracted')
#     params['seg_dir'] = os.path.join(params['ana_dir'], 'segmented')
#     params['cell_dir'] = os.path.join(params['ana_dir'], 'cell_data')
#     params['track_dir'] = os.path.join(params['ana_dir'], 'tracking')
#
#     # use jd time in image metadata to make time table. Set to false if no jd time
#     if params['TIFF_source'] == 'elements' or params['TIFF_source'] == 'nd2ToTIFF':
#         params['use_jd'] = True
#     else:
#         params['use_jd'] = False

class Window(QMainWindow):

    def __init__(self, parent=None, training_dir=None):
        super(Window, self).__init__(parent)

        top = 400
        left = 400
        width = 800
        height = 600

        self.setWindowTitle("Brent is not a doo-doo head!")
        self.setGeometry(top,left,width,height)

        # load specs file
        with open(os.path.join(params['ana_dir'], 'specs.yaml'), 'r') as specs_file:
            specs = yaml.safe_load(specs_file)

        self.threeFrames = ThreeFrameImgWidget(specs=specs, training_dir=training_dir)
        # make scene the central widget
        self.setCentralWidget(self.threeFrames)

        eventButtonGroup = QButtonGroup()
        migrateButton = QRadioButton("Migration")
        migrateButton.setShortcut("Ctrl+M")
        migrateButton.clicked.connect(self.threeFrames.scene.set_migration)
        migrateButton.click()
        eventButtonGroup.addButton(migrateButton)

        childrenButton = QRadioButton("Children")
        childrenButton.setShortcut("Ctrl+C")
        childrenButton.clicked.connect(self.threeFrames.scene.set_children)
        eventButtonGroup.addButton(childrenButton)

        bornButton = QRadioButton("Born")
        bornButton.setShortcut("Ctrl+C")
        bornButton.clicked.connect(self.threeFrames.scene.set_born)
        eventButtonGroup.addButton(bornButton)

        dieButton = QRadioButton("Die")
        dieButton.setShortcut("Ctrl+D")
        dieButton.clicked.connect(self.threeFrames.scene.set_die)
        eventButtonGroup.addButton(dieButton)

        appearButton = QRadioButton("Appear")
        appearButton.setShortcut("Ctrl+A")
        appearButton.clicked.connect(self.threeFrames.scene.set_appear)
        eventButtonGroup.addButton(appearButton)

        disappearButton = QRadioButton("Disappear")
        disappearButton.clicked.connect(self.threeFrames.scene.set_disappear)
        eventButtonGroup.addButton(disappearButton)

        eventButtonLayout = QVBoxLayout()
        eventButtonLayout.addWidget(migrateButton)
        eventButtonLayout.addWidget(childrenButton)
        eventButtonLayout.addWidget(bornButton)
        eventButtonLayout.addWidget(dieButton)
        eventButtonLayout.addWidget(appearButton)
        eventButtonLayout.addWidget(disappearButton)

        eventButtonGroupWidget = QWidget()
        eventButtonGroupWidget.setLayout(eventButtonLayout)

        eventButtonDockWidget = QDockWidget()
        eventButtonDockWidget.setWidget(eventButtonGroupWidget)
        self.addDockWidget(Qt.LeftDockWidgetArea, eventButtonDockWidget)

        advanceFrameButton = QPushButton("Next frame")
        advanceFrameButton.setShortcut("Ctrl+F")
        advanceFrameButton.clicked.connect(self.threeFrames.scene.advance_frame)

        priorFrameButton = QPushButton("Prior frame")
        priorFrameButton.clicked.connect(self.threeFrames.scene.prior_frame)

        advancePeakButton = QPushButton("Next peak")
        advancePeakButton.setShortcut("Ctrl+P")
        advancePeakButton.clicked.connect(self.threeFrames.scene.next_peak)

        priorPeakButton = QPushButton("Prior peak")
        priorPeakButton.clicked.connect(self.threeFrames.scene.prior_peak)

        advanceFOVButton = QPushButton("Next FOV")
        advanceFOVButton.clicked.connect(self.threeFrames.scene.next_fov)

        priorFOVButton = QPushButton("Prior FOV")
        priorFOVButton.clicked.connect(self.threeFrames.scene.prior_fov)

        fileAdvanceLayout = QVBoxLayout()
        fileAdvanceLayout.addWidget(advanceFrameButton)
        fileAdvanceLayout.addWidget(priorFrameButton)
        fileAdvanceLayout.addWidget(advancePeakButton)
        fileAdvanceLayout.addWidget(priorPeakButton)
        fileAdvanceLayout.addWidget(advanceFOVButton)
        fileAdvanceLayout.addWidget(priorFOVButton)
        # fileAdvanceLayout.addWidget(saveAndNextButton)

        fileAdvanceGroupWidget = QWidget()
        fileAdvanceGroupWidget.setLayout(fileAdvanceLayout)

        fileAdvanceDockWidget = QDockWidget()
        fileAdvanceDockWidget.setWidget(fileAdvanceGroupWidget)
        self.addDockWidget(Qt.RightDockWidgetArea, fileAdvanceDockWidget)

class ThreeFrameImgWidget(QWidget):
    # class for setting three frames side-by-side as a central widget in a QMainWindow object
    def __init__(self,specs,training_dir):
        super(ThreeFrameImgWidget, self).__init__()

        # add images and cell regions as ellipses to each frame in a QGraphicsScene object
        self.scene = ThreeFrameItem(specs,training_dir)
        self.view = QGraphicsView(self)
        self.view.setScene(self.scene)


class ThreeFrameItem(QGraphicsScene):
    # add more functionality for setting event type, i.e., parent-child, migrate, death, leave frame, etc..

    def __init__(self,specs,training_dir):
        super(ThreeFrameItem, self).__init__()

        self.specs = specs
        # add QImages to scene (try three frames)
        self.fov_id_list = [fov_id for fov_id in specs.keys()]
        self.center_frame_index = 1

        self.fovIndex = 0
        self.fov_id = self.fov_id_list[self.fovIndex]

        self.peak_id_list_in_fov = [peak_id for peak_id in specs[self.fov_id].keys()]

        self.peakIndex = 0
        self.peak_id = self.peak_id_list_in_fov[self.peakIndex]

        # construct image stack file names from params
        self.phaseImgPath = os.path.join(params['chnl_dir'], "{}_xy{:0=3}_p{:0=4}_{}.tif".format(params['experiment_name'], self.fov_id, self.peak_id, params['phase_plane']))
        self.labelImgPath = os.path.join(params['seg_dir'], "{}_xy{:0=3}_p{:0=4}_seg_unet.tif".format(params['experiment_name'], self.fov_id, self.peak_id))

        # labelImgPath = imgPaths[self.fov_id][self.imgIndex][1]
        self.labelStack = io.imread(self.labelImgPath)
        # phaseImgPath = imgPaths[self.fov_id][self.imgIndex][0]
        self.phaseStack = io.imread(self.phaseImgPath)

        time_int = params['moviemaker']['seconds_per_time_index']/60

        cell_filename = os.path.join(params['cell_dir'], 'complete_cells.pkl')
        cell_filename_all = os.path.join(params['cell_dir'], 'all_cells.pkl')

        with open(cell_filename, 'rb') as cell_file:
            self.Cells = pickle.load(cell_file)
        mm3.calculate_pole_age(self.Cells) # add poleage

        with open(cell_filename_all, 'rb') as cell_file:
            self.All_Cells = pickle.load(cell_file)

        plot_dir = os.path.join(params['cell_dir'], '20190312_plots')
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        lin_dir = os.path.join(plot_dir, 'lineage_plots_full')
        if not os.path.exists(lin_dir):
            os.makedirs(lin_dir)

        self.regions_and_events_by_time = self.create_tracking_information()

        # keep in mind that regions_and_events_by_time is 1-indexed, whereas the phaseStack and labelStack are 0-indexed
        leftFrame, leftRegions = self.phase_img_and_regions(frame_index=self.center_frame_index-1)
        centerFrame, centerRegions = self.phase_img_and_regions(frame_index=self.center_frame_index)
        rightFrame, rightRegions = self.phase_img_and_regions(frame_index=self.center_frame_index+1)

        self.brushSize = 2
        self.brushColor = QColor('black')
        self.lastPoint = QPoint()
        self.pen = QPen()

        # class options
        self.migration = False
        self.die = False
        self.children = False
        self.birth = False
        self.appear = False
        self.disappear = False

        # set the scene...
        self.set_scene(leftFrame, centerFrame, rightFrame, leftRegions, centerRegions, rightRegions)


    def set_scene(self, leftFrame, centerFrame, rightFrame, leftRegions, centerRegions, rightRegions):

        self.clear()
        # Add each image to the scene
        self.leftFrame = self.addPixmap(leftFrame)
        self.centerFrame = self.addPixmap(centerFrame)
        self.rightFrame = self.addPixmap(rightFrame)

        # Loop through images and shift each by appropriate x-distance to get them side-by-side, rather than stacked
        xPositions = []
        xPos = 0
        for item in self.items(order=Qt.AscendingOrder):
            item.setPos(xPos, 0)
            xPositions.append(xPos)
            xPos += item.pixmap().width()

        # add cell regions to each frame
        self.add_regions_to_frame(leftRegions, self.leftFrame)
        self.add_regions_to_frame(centerRegions, self.centerFrame)
        self.add_regions_to_frame(rightRegions, self.rightFrame)

        self.draw_cell_events()

    def advance_frame(self):
        try:
            self.center_frame_index += 1
            # redefine frames and regions
            leftFrame, leftRegions = self.phase_img_and_regions(frame_index=self.center_frame_index-1)
            centerFrame, centerRegions = self.phase_img_and_regions(frame_index=self.center_frame_index)
            rightFrame, rightRegions = self.phase_img_and_regions(frame_index=self.center_frame_index+1)

        except KeyError:
            sys.exit("You've already edited the last frame's mask. Write in functionality to increment to next peak_id now!")

        self.set_scene(leftFrame, centerFrame, rightFrame, leftRegions, centerRegions, rightRegions)

    def prior_frame(self):
        try:
            self.center_frame_index -= 1
            print(self.center_frame_index)
            # redefine frames and regions
            leftFrame, leftRegions = self.phase_img_and_regions(frame_index=self.center_frame_index-1)
            centerFrame, centerRegions = self.phase_img_and_regions(frame_index=self.center_frame_index)
            rightFrame, rightRegions = self.phase_img_and_regions(frame_index=self.center_frame_index+1)

        except KeyError:
            sys.exit("You've already edited the last frame's mask. Write in functionality to increment to next peak_id now!")

        self.set_scene(leftFrame, centerFrame, rightFrame, leftRegions, centerRegions, rightRegions)

    def next_peak(self):
        self.center_frame_index = 1
        self.peakIndex += 1
        self.peak_id = self.peak_id_list_in_fov[self.peakIndex]
        print(self.peak_id)

        # construct image stack file names from params
        self.phaseImgPath = os.path.join(params['chnl_dir'], "{}_xy{:0=3}_p{:0=4}_{}.tif".format(params['experiment_name'], self.fov_id, self.peak_id, params['phase_plane']))
        self.labelImgPath = os.path.join(params['seg_dir'], "{}_xy{:0=3}_p{:0=4}_seg_unet.tif".format(params['experiment_name'], self.fov_id, self.peak_id))
        print(self.phaseImgPath)
        print(self.labelImgPath)

        self.labelStack = io.imread(self.labelImgPath)
        self.phaseStack = io.imread(self.phaseImgPath)

        self.regions_and_events_by_time = self.create_tracking_information()

        # keep in mind that regions_and_events_by_time is 1-indexed, whereas the phaseStack and labelStack are 0-indexed
        leftFrame, leftRegions = self.phase_img_and_regions(frame_index=self.center_frame_index-1)
        centerFrame, centerRegions = self.phase_img_and_regions(frame_index=self.center_frame_index)
        rightFrame, rightRegions = self.phase_img_and_regions(frame_index=self.center_frame_index+1)

        # set the scene...
        self.set_scene(leftFrame, centerFrame, rightFrame, leftRegions, centerRegions, rightRegions)

    def prior_peak(self):
        self.center_frame_index = 1
        self.peakIndex -= 1
        self.peak_id = self.peak_id_list_in_fov[self.peakIndex]
        print(self.peak_id)

        # construct image stack file names from params
        self.phaseImgPath = os.path.join(params['chnl_dir'], "{}_xy{:0=3}_p{:0=4}_{}.tif".format(params['experiment_name'], self.fov_id, self.peak_id, params['phase_plane']))
        self.labelImgPath = os.path.join(params['seg_dir'], "{}_xy{:0=3}_p{:0=4}_seg_unet.tif".format(params['experiment_name'], self.fov_id, self.peak_id))
        print(self.phaseImgPath)
        print(self.labelImgPath)

        self.labelStack = io.imread(self.labelImgPath)
        self.phaseStack = io.imread(self.phaseImgPath)

        self.regions_and_events_by_time = self.create_tracking_information()

        # keep in mind that regions_and_events_by_time is 1-indexed, whereas the phaseStack and labelStack are 0-indexed
        leftFrame, leftRegions = self.phase_img_and_regions(frame_index=self.center_frame_index-1)
        centerFrame, centerRegions = self.phase_img_and_regions(frame_index=self.center_frame_index)
        rightFrame, rightRegions = self.phase_img_and_regions(frame_index=self.center_frame_index+1)

        # set the scene...
        self.set_scene(leftFrame, centerFrame, rightFrame, leftRegions, centerRegions, rightRegions)

    def next_fov(self):
        self.center_frame_index = 1

        self.fovIndex += 1
        self.fov_id = self.fov_id_list[self.fovIndex]

        self.peak_id_list_in_fov = [peak_id for peak_id in self.specs[self.fov_id].keys()]

        self.peakIndex = 0
        self.peak_id = self.peak_id_list_in_fov[self.peakIndex]

        # construct image stack file names from params
        self.phaseImgPath = os.path.join(params['chnl_dir'], "{}_xy{:0=3}_p{:0=4}_{}.tif".format(params['experiment_name'], self.fov_id, self.peak_id, params['phase_plane']))
        self.labelImgPath = os.path.join(params['seg_dir'], "{}_xy{:0=3}_p{:0=4}_seg_unet.tif".format(params['experiment_name'], self.fov_id, self.peak_id))
        print(self.phaseImgPath)
        print(self.labelImgPath)

        self.labelStack = io.imread(self.labelImgPath)
        self.phaseStack = io.imread(self.phaseImgPath)

        self.regions_and_events_by_time = self.create_tracking_information()

        # keep in mind that regions_and_events_by_time is 1-indexed, whereas the phaseStack and labelStack are 0-indexed
        leftFrame, leftRegions = self.phase_img_and_regions(frame_index=self.center_frame_index-1)
        centerFrame, centerRegions = self.phase_img_and_regions(frame_index=self.center_frame_index)
        rightFrame, rightRegions = self.phase_img_and_regions(frame_index=self.center_frame_index+1)

        # set the scene...
        self.set_scene(leftFrame, centerFrame, rightFrame, leftRegions, centerRegions, rightRegions)

    def prior_fov(self):
        self.center_frame_index = 1

        self.fovIndex -= 1
        self.fov_id = self.fov_id_list[self.fovIndex]

        self.peak_id_list_in_fov = [peak_id for peak_id in self.specs[self.fov_id].keys()]

        self.peakIndex = 0
        self.peak_id = self.peak_id_list_in_fov[self.peakIndex]

        # construct image stack file names from params
        self.phaseImgPath = os.path.join(params['chnl_dir'], "{}_xy{:0=3}_p{:0=4}_{}.tif".format(params['experiment_name'], self.fov_id, self.peak_id, params['phase_plane']))
        self.labelImgPath = os.path.join(params['seg_dir'], "{}_xy{:0=3}_p{:0=4}_seg_unet.tif".format(params['experiment_name'], self.fov_id, self.peak_id))
        print(self.phaseImgPath)
        print(self.labelImgPath)

        self.labelStack = io.imread(self.labelImgPath)
        self.phaseStack = io.imread(self.phaseImgPath)

        self.regions_and_events_by_time = self.create_tracking_information()

        # keep in mind that regions_and_events_by_time is 1-indexed, whereas the phaseStack and labelStack are 0-indexed
        leftFrame, leftRegions = self.phase_img_and_regions(frame_index=self.center_frame_index-1)
        centerFrame, centerRegions = self.phase_img_and_regions(frame_index=self.center_frame_index)
        rightFrame, rightRegions = self.phase_img_and_regions(frame_index=self.center_frame_index+1)

        # set the scene...
        self.set_scene(leftFrame, centerFrame, rightFrame, leftRegions, centerRegions, rightRegions)

    def phase_img_and_regions(self, frame_index):

        time = frame_index+1
        phaseImg = self.phaseStack[frame_index,:,:]
        maskImg = self.labelStack[frame_index,:,:]
        originalImgMax = np.max(phaseImg)
        phaseImg = phaseImg/originalImgMax
        phaseImg = color.gray2rgb(phaseImg)
        RGBImg = (phaseImg*255).astype('uint8')

        originalHeight, originalWidth, originalChannelNumber = RGBImg.shape
        phaseQimage = QImage(RGBImg, originalWidth, originalHeight,
                             RGBImg.strides[0], QImage.Format_RGB888)#.scaled(512, 512, aspectRatioMode=Qt.KeepAspectRatio)
        phaseQpixmap = QPixmap(phaseQimage)
        # maskQpixmap = QGraphicsPixmapItem(phaseQpixmap)

        # create transparent rbg overlay to grab colors from for drawing cell regions as QGraphicsPathItems
        RGBLabelImg = color.label2rgb(maskImg, bg_label=0)
        RGBLabelImg = (RGBLabelImg*255).astype('uint8')
        originalHeight, originalWidth, RGBLabelChannelNumber = RGBLabelImg.shape
        RGBLabelImg = QImage(RGBLabelImg, originalWidth, originalHeight, RGBLabelImg.strides[0], QImage.Format_RGB888)#.scaled(512, 512, aspectRatioMode=Qt.KeepAspectRatio)
        # pprint(regions)
        time_regions_and_events = self.regions_and_events_by_time[time]
        regions = time_regions_and_events['regions']
        time_regions_and_events['time'] = time

        for region_id in regions.keys():
            brush = QBrush()
            brush.setStyle(Qt.SolidPattern)
            pen = QPen()
            pen.setStyle(Qt.SolidLine)
            props = regions[region_id]['props']
            min_row, min_col, max_row, max_col = props.bbox
            label = props.label
            # coords = props.coords
            # rr = coords[:,0]
            # cc = coords[:,1]
            centroidY,centroidX = props.centroid
            brushColor = RGBLabelImg.pixelColor(centroidX,centroidY)
            brushColor.setAlphaF(0.25)
            brush.setColor(brushColor)
            # brush.setColor(QColor('red'))
            pen.setColor(brushColor)
            # brush.setColor(QColor('red'))

            # for i in range(len(rr)):
            #     x = cc[i]
            #     y = rr[i]
            #     point = QPoint(x,y)
            #     if i == 0:
            #         path = QPainterPath(point)
            #     else:
            #         path.moveTo(point)
            # path.setFillRule(Qt.WindingFill)
            # region_graphic = QGraphicsPathItem(path)
            # region_graphic.setPen(pen)
            # region_graphic.setBrush(brush)
            regions[region_id]['region_graphic'] = {'top_y':min_row, 'bottom_y':max_row,
                                                    'left_x':min_col, 'right_x':max_col,
                                                    #'path':path,
                                                    'pen':pen, 'brush':brush}

        return(phaseQpixmap, time_regions_and_events)

    def create_tracking_information(self):

        Complete_Lineages = mm3_plots.organize_cells_by_channel(self.Cells, self.specs)
        All_Lineages = mm3_plots.organize_cells_by_channel(self.All_Cells, self.specs)

        t_adj = 1

        regions_by_time = {frame+t_adj: measure.regionprops(self.labelStack[frame,:,:]) for frame in range(self.labelStack.shape[0])}
        regions_and_events_by_time = {frame+t_adj : {'regions' : {}, 'matrix' : None} for frame in range(self.labelStack.shape[0])}

        # loop through regions and add them to the main dictionary.
        for t, regions in regions_by_time.items():
            # this is a list, while we want it to be a dictionary with the region label as the key
            for region in regions:
                default_events = np.zeros(7, dtype=np.int)
                default_events[6] = 1 # set N to 1
                regions_and_events_by_time[t]['regions'][region.label] = {'props' : region,
                                                                      'events' : default_events}
        # create default interaction matrix
        # Now that we know how many regions there are per time point, we will create a default matrix which indicates how regions are connected to each between this time point, t, and the next one, t+1. The row index will be the region label of the region in t, and the column index will be the region label of the region in t+1.
        # If a region migrates from t to t+1, its row should have a sum of 1 corresponding from which region (row) to which region (column) it moved. If the region divided, then both of the daughter columns will get value 1.
        # Note that the regions are labeled from 1, but Numpy arrays and indexed from zero. We can use this to store some additional informaiton. If a region disappears, it will receive a 1 in the column with index 0.
        # In the last time point all regions will be connected to the disappear column
        for t, t_data in regions_and_events_by_time.items():
            n_regions_in_t = len(regions_by_time[t])
            if t+1 in regions_by_time:
                n_regions_in_t_plus_1 = len(regions_by_time[t+1])
            else:
                n_regions_in_t_plus_1 = 0

            t_data['matrix'] = np.zeros((n_regions_in_t+1, n_regions_in_t_plus_1+1), dtype=np.int)

        # Loop over cells and edit event information
        # We will use the cell dictionary All_Cells. Complete_Cells is a subset of All_Cells which is just those cells that have both a mother and a daughter. These are the brighter lines on the lineage plot.
        # Each cell object has a number of attributes that are useful to us. For example find cell f01p0077t0003r02 on the lineage plot and then take a look at its attributes below.
        # We will go through each cell by its time points and edit the events associated with that region.
        # We will also edit the matrix when appropriate.
        # pull out only the cells in of this FOV
        cells_tmp = mm3_plots.find_cells_of_fov_and_peak(self.All_Cells, self.fov_id, self.peak_id)
        print('There are {} cells for this channel'.format(len(cells_tmp)))

        for cell_id, cell_tmp in cells_tmp.items():

            # Check for when cell has less time points than it should
            if (cell_tmp.times[-1] - cell_tmp.times[0])+1 > len(cell_tmp.times):
                print('Cell {} has less time points than it should, skipping.'.format(cell_id))
                continue

            # Go over the time points of this cell and edit appropriate information main dictionary
            for i, t in enumerate(cell_tmp.times):

                # get the region label
                label_tmp = cell_tmp.labels[i]

                # M migration, event 0
                # If the cell has another time point after this one then it must have migrated
                if i != len(cell_tmp.times)-1:
                    regions_and_events_by_time[t]['regions'][label_tmp]['events'][0] = 1

                    # update matrix using this region label and the next one
                    regions_and_events_by_time[t]['matrix'][label_tmp, cell_tmp.labels[i+1]] = 1

                # S division, 1
                if cell_tmp.daughters and i == len(cell_tmp.times)-1:
                    regions_and_events_by_time[t]['regions'][label_tmp]['events'][1] = 1

                    # daughter 1 and 2 label
                    d1_label = self.All_Cells[cell_tmp.daughters[0]].labels[0]
                    d2_label = self.All_Cells[cell_tmp.daughters[1]].labels[0]

                    regions_and_events_by_time[t]['matrix'][label_tmp, d1_label] = 1
                    regions_and_events_by_time[t]['matrix'][label_tmp, d2_label] = 1

                # A apoptosis, 2
                # skip for now.

                # B birth, 3
                if cell_tmp.parent and i == 0:
                    regions_and_events_by_time[t]['regions'][label_tmp]['events'][3] = 1

                # I appears, 4
                if not cell_tmp.parent and i == 0:
                    regions_and_events_by_time[t]['regions'][label_tmp]['events'][4] = 1

                # O disappears, 5
                if not cell_tmp.daughters and i == len(cell_tmp.times)-1:
                    regions_and_events_by_time[t]['regions'][label_tmp]['events'][5] = 1
                    regions_and_events_by_time[t]['matrix'][label_tmp, 0] = 1

                # N no data, 6 - Set this to zero as this region as been checked.
                regions_and_events_by_time[t]['regions'][label_tmp]['events'][6] = 0

        # Set remaining regions to event space [0 0 0 0 1 1]
        # Also make their appropriate matrix value 1, which should be in the first column.
        for t, t_data in regions_and_events_by_time.items():
            for region, region_data in t_data['regions'].items():
                if region_data['events'][6] == 1:
                    region_data['events'][4:] = 1, 1, 0

                    t_data['matrix'][region, 0] = 1

        return(regions_and_events_by_time)

    def set_frames(self, leftFrame, centerFrame, rightFrame):
        self.leftFrame = self.addPixmap(leftFrame)
        self.centerFrame = self.addPixmap(centerFrame)
        self.rightFrame = self.addPixmap(rightFrame)

    def add_regions_to_frame(self, regions_and_events, frame):
        # loop through cells within this frame and add their ellipses as children of their corresponding qpixmap object
        regions = regions_and_events['regions']
        # print(regions_and_events)
        frame_time = regions_and_events['time']
        for region_id in regions.keys():
            region = regions[region_id]
            # pprint(region)
            # construct the ellipse
            graphic = region['region_graphic']
            top_left = QPoint(graphic['left_x'],graphic['top_y'])
            bottom_right = QPoint(graphic['right_x'],graphic['bottom_y'])
            rect = QRectF(top_left,bottom_right)
            ellipse = QGraphicsEllipseItem(rect, frame)

            # add cell information to the QGraphicsEllipseItem
            ellipse.cellMatrix = regions_and_events['matrix']
            ellipse.cellEvents = regions_and_events['regions'][region_id]['events']
            ellipse.cellProps = regions_and_events['regions'][region_id]['props']
            ellipse.time = frame_time
            ellipse.setBrush(graphic['brush'])
            ellipse.setPen(graphic['pen'])

    def draw_cell_events(self):
        # Here is where we will draw the intial lines and symbols representing
        #   cell events and linkages between cells.

        # regions_and_events_by_time is a dictionary, the keys of which are 1-indexed frame numbers
        # for each frame, there is a dictionary with the following keys: 'matrix' and 'regions'
        #   'matrix' is a 2D array, for which the row index is the region label at time t, and the column index is the region label at time t+1
        #      If a region disappears from t to t+1, it will receive a 1 in the column with index 0.
        #   'regions' is a dictionary with each region's label as a separate key.
        #      Each region in 'regions' is another dictionary, with 'events', which contains a 1D array identifying the events that correspond to the connections in 'matrix',
        #                                                      and 'props', which contains all region properties that measure.regionprops retuns for the labelled image at time t.
        # The 'events' array is binary. The events are [migration, division, death, birth, appearance, disappearance, no_data], where a 0 at a given position in the 'events'
        #      array indicates the given event did not occur, and a 1 indicates it did occur.

        if len(self.leftFrame.childItems()) > 0:
            for self.startItem in self.leftFrame.childItems():
                # test if this is an ellipse item. If it is, draw event symbols.
                if self.startItem.type() == 4:
                    cell_properties = self.startItem.cellProps
                    cell_label = cell_properties.label
                    cell_interactions = self.startItem.cellMatrix[cell_label,:]
                    cell_events = self.startItem.cellEvents
                    # print(cell_events)
                    # print(cell_interactions)
                    # get centroid of cell represented by this qgraphics item
                    firstPointY = cell_properties.centroid[0]
                    firstPointX = cell_properties.centroid[1] + self.startItem.parentItem().x()
                    self.firstPoint = QPoint(firstPointX, firstPointY)
                    # which events happened to this cell?
                    event_indices = np.where(cell_events == 1)[0]

                    for self.endItem in self.centerFrame.childItems():
                        # if the item is an ellipse, move on to look into it further
                        if self.endItem.type() == 4:
                            end_cell_properties = self.endItem.cellProps
                            end_cell_label = end_cell_properties.label
                            # test whether this ellipse represents the
                            #  cell that interacts with the cell represented
                            #  by self.startItem
                            if cell_interactions[end_cell_label] == 1:
                                # if this is the cell that interacts with the former frame's cell, draw the line.
                                endPointY = end_cell_properties.centroid[0]
                                endPointX = end_cell_properties.centroid[1] + self.endItem.parentItem().x()
                                self.lastPoint = QPoint(endPointX, endPointY)
                                if 0 in event_indices:
                                    # If the zero-th element in event_indices was 1, the cell migrates in the next frame
                                    #  get the information from cell_matrix to figure out to which region
                                    #  in the next frame it migrated
                                    # set self.migration = True
                                    self.set_migration()
                                if 1 in event_indices:
                                    self.set_children()

                                self.eventItem = self.set_event_item()
                                self.addItem(self.eventItem)

        if len(self.centerFrame.childItems()) > 0:
            for self.startItem in self.centerFrame.childItems():
                # test if this is an ellipse item. If it is, draw event symbols.
                if self.startItem.type() == 4:
                    cell_properties = self.startItem.cellProps
                    cell_label = cell_properties.label
                    cell_interactions = self.startItem.cellMatrix[cell_label,:]
                    cell_events = self.startItem.cellEvents
                    # print(cell_events)
                    # print(cell_interactions)
                    # get centroid of cell represented by this qgraphics item
                    firstPointY = cell_properties.centroid[0]
                    firstPointX = cell_properties.centroid[1] + self.startItem.parentItem().x()
                    self.firstPoint = QPoint(firstPointX, firstPointY)
                    # which events happened to this cell?
                    event_indices = np.where(cell_events == 1)[0]

                    # Start by determining whether the cell dies here
                    if 2 in event_indices:
                        # If the second element in event_indices was 1, the cell dies between this frame and the next.
                        self.set_die()
                        self.draw_death()

                    for self.endItem in self.rightFrame.childItems():
                        # if the item is an ellipse, move on to look into it further
                        if self.endItem.type() == 4:
                            end_cell_properties = self.endItem.cellProps
                            end_cell_label = end_cell_properties.label
                            # test whether this ellipse represents the
                            #  cell that interacts with the cell represented
                            #  by self.startItem
                            if cell_interactions[end_cell_label] == 1:
                                # if this is the cell that interacts with the former frame's cell, draw the line.
                                endPointY = end_cell_properties.centroid[0]
                                endPointX = end_cell_properties.centroid[1] + self.endItem.parentItem().x()
                                self.lastPoint = QPoint(endPointX, endPointY)
                                if 0 in event_indices:
                                    # If the zero-th element in event_indices was 1, the cell migrates in the next frame
                                    self.set_migration()
                                if 1 in event_indices:
                                    # If the one-th element in event_indices was 1, the cell divides into children in the next frame
                                    self.set_children()
                                # if 3 in event_indices:
                                    #

                                self.eventItem = self.set_event_item()
                                self.addItem(self.eventItem)

    # function for finding the ellipse under your mouse click or mouse release,
    #  since items can be stacked, a line you previously drew can obscure the
    #  ellipse you intended to select
    def get_ellipse(self, point):
        items = self.items(point)
        ellipseEncountered = False
        for item in items:

            itemType = item.type()
            if itemType == 4:
                ellipseEncountered = True
                # once we find an ellipse object under our mouse event we return the ellipse
                if ellipseEncountered:
                    return(item)

        # if an ellipse was never found after burrowing through the items, return None
        print("No cell detected underneath your selection. Ignoring selection.")
        return(None)

    # def get_event_items(self, point):
    #     valid_event_types = ["MigrationLine","ChildLine","DieSymbol",
    #                          "AppearSymbol","DisappearSymbol","BornSymbol"]
    #     items = self.items(point)
    #     event_types = [item.type() for item in items if item.type() in valid_event_types]
    #     event_items = [item for item in items if item.type() in valid_event_types]
    #
    #     event_type_counts = {}
    #     for event_type in valid_event_types:
    #         event_type_counts[event_type] = 0
    #
    #     for event_type in event_types:
    #         event_type_counts[event_type] += 1
    #
    #     return(item_type_counts, event_items)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.firstPoint = event.scenePos()
            self.lastPoint = event.scenePos()
            self.startItem = self.get_ellipse(point=self.firstPoint)
            if self.startItem is not None:
                # get the centroid position for the cell that was clicked
                firstPointY = self.startItem.cellProps.centroid[0]
                # here we add the x-position of the detected ellipse' frame, because
                #   the centroid of each cell is just its centroid within its own frame
                #   Therefore, by adding the x-offset of the frame in which the cell
                #   exists, we shift our x-value of our line's start or end-point by the appropriate distance.
                firstPointX = self.startItem.cellProps.centroid[1] + self.startItem.parentItem().x()
                self.firstPoint = QPoint(firstPointX, firstPointY)
                self.eventItem = self.set_event_item()
                self.addItem(self.eventItem)

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) & self.drawing:
            if self.startItem is not None:
                self.lastPoint = event.scenePos()
                self.removeItem(self.eventItem)
                self.eventItem = self.set_event_item()
                self.addItem(self.eventItem)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton & self.drawing:
            self.lastPoint = event.scenePos()
            self.endItem = self.get_ellipse(point=self.lastPoint)
            if self.startItem is not None:
                if self.endItem is None:
                    self.removeItem(self.eventItem)

            if self.endItem is not None:
                if self.startItem is not None:
                    if (self.startItem.parentItem() == self.endItem.parentItem()) and self.eventItem.type() in ["MigrationLine","ChildLine"]:
                        self.removeItem(self.eventItem)
                        print("Cannot link cells in a single frame as migrated or children. Ignoring selection.")
                    else:
                        self.removeItem(self.eventItem)
                        # get the centroid position for the cell that was clicked
                        endPointY = self.endItem.cellProps.centroid[0]
                        # here we add the x-position of the detected ellipse' frame, because
                        #   the centroid of each cell is just its centroid within its own frame
                        #   Therefore, by adding the x-offset of the frame in which the cell
                        #   exists, we shift our x-value of our line's start or end-point by the appropriate distance.
                        endPointX = self.endItem.cellProps.centroid[1] + self.endItem.parentItem().x()
                        self.lastPoint = QPoint(endPointX, endPointY)
                        self.eventItem = self.set_event_item()
                        if self.eventItem.type() == "MigrationLine":
                            print(self.eventItem.startItem.scenePos().x(), self.eventItem.endItem.scenePos().x())

###############################  TO DO: SORT OUT HOW TO REPLACE EXISTING ANNOTATIONS AND UPDATE UNDERLYING DATA STRUCTURE  ################################

                        # self.end_item_event_counts = self.get_event_items(point=self.lastPoint)
                        # self.start_item_event_counts = self.get_event_items(point=self.firstPoint)
                        # print("Start event count: ", self.start_item_event_counts)
                        # print("End event count: ", self.end_item_event_counts)
                        self.addItem(self.eventItem)
                        self.update_cell_info()

            self.drawing = False

    def update_cell_info(self):
        # This function updates the matrix and events for self.startItem and self.endItem,
        #  given the lines you've drawn from, and to them, respectively.
        pass

    # I need to test whether the cell already has an event of the chosen type
    #   so that the existing one can be replaced by the new one I'm currently drawing.

    def set_event_item(self):
        if self.migration:
            eventItem = MigrationLine(self.firstPoint, self.lastPoint, self.startItem, self.endItem)
        if self.children:
            eventItem = ChildLine(self.firstPoint, self.lastPoint, self.startItem, self.endItem)
        if self.die:
            eventItem = DieSymbol(self.firstPoint)
        if self.birth:
            eventItem = BornSymbol(self.firstPoint)
        if self.appear:
            eventItem = AppearSymbol(self.firstPoint)
        if self.disappear:
            eventItem = DisappearSymbol(self.firstPoint)

        return(eventItem)

    def set_migration(self):
        # print('clicked set_migration')
        self.migration = True
        self.die = False
        self.children = False
        self.birth = False
        self.appear = False
        self.disappear = False

    def set_children(self):
        # print('clicked set_children')
        self.migration = False
        self.die = False
        self.children = True
        self.birth = False
        self.appear = False
        self.disappear = False

    def set_die(self):
        # print('clicked set_die')
        self.migration = False
        self.die = True
        self.children = False
        self.birth = False
        self.appear = False
        self.disappear = False

    def set_appear(self):
        self.migration = False
        self.die = False
        self.children = False
        self.birth = False
        self.appear = True
        self.disappear = False

    def set_disappear(self):
        self.migration = False
        self.die = False
        self.children = False
        self.birth = False
        self.appear = False
        self.disappear = True

    def set_born(self):
        self.migration = False
        self.die = False
        self.children = False
        self.birth = True
        self.appear = False
        self.disappear = False

class MigrationLine(QGraphicsLineItem):
    # A class for helping to draw and organize migration events
    #  within a QGraphicsScene
    def __init__(self, firstPoint, lastPoint, startItem, endItem):
        super(MigrationLine, self).__init__()

        brushColor = QColor(1*255,1*255,1*255)
        brushSize = 2
        pen = QPen()
        firstPointX = firstPoint.x()
        lastPointX = lastPoint.x()
        if firstPointX < lastPointX:
            self.start = firstPoint
            self.startItem = startItem
            self.end = lastPoint
            self.endItem = endItem
        else:
            self.start = lastPoint
            self.startItem = endItem
            self.end = firstPoint
            self.endItem = startItem
        line = QLineF(self.start,self.end)
        pen.setColor(brushColor)
        pen.setWidth(brushSize)
        self.setPen(pen)
        self.setLine(line)

    def type(self):
        return("MigrationLine")

class DieSymbol(QGraphicsTextItem):

    def __init__(self, point):
        super(DieSymbol, self).__init__()

        textColor = QColor(1*255,0*255,0*255)
        textFont = QFont()
        textFont.setFamily("Times")
        textFont.setPixelSize(16)
        string = "X"
        textPosition = QPoint(point.x()-9, point.y()-9)

        self.setPlainText(string)
        self.setFont(textFont)
        self.setPos(textPosition)
        self.setDefaultTextColor(textColor)

    def type(self):
        return("DieSymbol")

class ChildLine(QGraphicsLineItem):
    # A class for helping to draw and organize migration events
    #  within a QGraphicsScene
    def __init__(self, firstPoint, lastPoint, startItem, endItem):
        super(ChildLine, self).__init__()

        brushColor = QColor(0*255,1*255,0*255)
        brushSize = 2
        pen = QPen()
        firstPointX = firstPoint.x()
        lastPointX = lastPoint.x()
        if firstPointX < lastPointX:
            self.start = firstPoint
            self.startItem = startItem
            self.end = lastPoint
            self.endItem = endItem
        else:
            self.start = lastPoint
            self.startItem = endItem
            self.end = firstPoint
            self.endItem = startItem
        line = QLineF(self.start,self.end)
        pen.setColor(brushColor)
        pen.setWidth(brushSize)
        self.setPen(pen)
        self.setLine(line)

    def type(self):
        return("ChildLine")

class BornSymbol(QGraphicsTextItem):

    def __init__(self, point):
        super(BornSymbol, self).__init__()

        textColor = QColor(0*255,1*255,1*255)
        textFont = QFont()
        textFont.setFamily("Times")
        textFont.setPixelSize(20)
        textFont.setWeight(75) # bold
        string = "o"
        textPosition = QPoint(point.x()-8, point.y()-17)

        self.setPlainText(string)
        self.setFont(textFont)
        self.setPos(textPosition)
        self.setDefaultTextColor(textColor)

    def type(self):
        return("BornSymbol")

class AppearSymbol(QGraphicsTextItem):

    def __init__(self, point):
        super(AppearSymbol, self).__init__()

        textColor = QColor(1*255,0*255,1*255)
        textFont = QFont()
        textFont.setFamily("Times")
        textFont.setPixelSize(24)
        textFont.setWeight(75) # Bold
        string = "+"
        textPosition = QPoint(point.x()-10, point.y()-15)

        self.setPlainText(string)
        self.setFont(textFont)
        self.setPos(textPosition)
        self.setDefaultTextColor(textColor)

    def type(self):
        return("AppearSymbol")

class DisappearSymbol(QGraphicsLineItem):

    def __init__(self, point):
        super(DisappearSymbol, self).__init__()

        brushColor = QColor(1*255,0*255,1*255)
        brushSize = 2
        pen = QPen()
        firstPoint = QPoint(point.x()-4, point.y())
        lastPoint = QPoint(point.x()+4, point.y())
        line = QLineF(firstPoint,lastPoint)
        pen.setColor(brushColor)
        pen.setWidth(brushSize)
        self.setPen(pen)
        self.setLine(line)

    def type(self):
        return("DisappearSymbol")



class LabelTransparencyWidget(QWidget):

        def __init__(self,imgPaths,fov_id_list,label_dir,frame_index):
                super(LabelTransparencyWidget, self).__init__()

                self.label_dir = label_dir
                self.frameIndex = frame_index

                self.imgPaths = imgPaths
                self.fov_id_list = fov_id_list
                self.fovIndex = 0
                self.fov_id = self.fov_id_list[self.fovIndex]

                self.imgIndex = 0
                self.labelImgPath = self.imgPaths[self.fov_id][self.imgIndex][1]
                # print(self.labelImgPath)

                # TO DO: check if re-annotated mask exists in training_dir, and present that instead of original mask
                #        make indicator appear if we're re-editing the mask again.
                experiment_name = params['experiment_name']
                original_file_name = self.labelImgPath
                pat = re.compile(r'.+(xy\d{3,4})_(p\d{3,4})_.+') # supports 3- or 4-digit naming
                mat = pat.match(original_file_name)
                fovID = mat.groups()[0]
                peakID = mat.groups()[1]
                fileBaseName = '{}_{}_{}_t{:0=4}.tif'.format(experiment_name, fovID, peakID, self.frameIndex+1)
                savePath = os.path.join(self.label_dir,fileBaseName)

                self.labelStack = io.imread(self.labelImgPath)
                if os.path.isfile(savePath):
                    print('Re-annotated mask exists in training directory. Loading it.')
                    # add widget to express whether this mask is one you already re-annotated
                    self.labelStack[self.frameIndex,:,:] = io.imread(savePath)
                    overwriteSegFile = True
                else:
                    overwriteSegFile = False

                img = self.labelStack[self.frameIndex,:,:]
                RGBImg = color.label2rgb(img, bg_label=0).astype('uint8')
                self.RGBImg = RGBImg*255

                alphaFloat = 0.25
                alphaArray = np.zeros(img.shape, dtype='uint8')
                alphaArray = np.expand_dims(alphaArray, -1)
                self.alpha = int(255*alphaFloat)
                alphaArray[...] = self.alpha
                self.RGBAImg = np.append(self.RGBImg, alphaArray, axis=-1)

                self.originalHeight, self.originalWidth, self.originalChannelNumber = self.RGBAImg.shape
                self.maskQimage = QImage(self.RGBAImg, self.originalWidth, self.originalHeight, self.RGBAImg.strides[0], QImage.Format_RGBA8888).scaled(512, 512, aspectRatioMode=Qt.KeepAspectRatio)
                self.maskQpixmap = QPixmap(self.maskQimage)

                self.label = QLabel(self)
                self.label.setPixmap(self.maskQpixmap)

                self.drawing = False
                self.brushSize = 2
                self.brushColor = QColor(0,0,0,self.alpha)
                self.lastPoint = QPoint()

        def mousePressEvent(self, event):
                if event.button() == Qt.LeftButton:
                        self.drawing = True
                        self.lastPoint = event.pos()

        def mouseMoveEvent(self, event):
                if (event.buttons() & Qt.LeftButton) & self.drawing:

                        # make the mouse position the center of a circle whose radius is defined as self.brushSize
                        rr,cc = draw.circle(event.y(), event.x(), self.brushSize)
                        for pix in zip(rr,cc):
                                rowIndex = pix[0]
                                colIndex = pix[1]
                                self.maskQimage.setPixelColor(colIndex, rowIndex, self.brushColor)

                        self.maskQpixmap = QPixmap(self.maskQimage)
                        self.label.setPixmap(self.maskQpixmap)
                        self.lastPoint = event.pos()
                        self.update()

        def mouseReleaseEvent(self, event):
                if event.button == Qt.LeftButton:
                        self.drawing = False

        def buttonSave(self):
                experiment_name = params['experiment_name']
                original_file_name = self.maskImgPath
                pat = re.compile(r'.+(xy\d{3,4})_(p\d{3,4})_.+') # supports 3- or 4-digit naming
                mat = pat.match(original_file_name)
                fovID = mat.groups()[0]
                peakID = mat.groups()[1]
                fileBaseName = '{}_{}_{}_t{:0=4}.tif'.format(experiment_name, fovID, peakID, self.frameIndex+1)
                savePath = os.path.join(self.label_dir,fileBaseName)
                labelSavePath = os.path.join(params['seg_dir'],fileBaseName)
                print("Saved binary mask image as: ", savePath)

                if not os.path.isdir(self.label_dir):
                        os.makedirs(self.label_dir)

                saveImg = self.maskQimage.convertToFormat(QImage.Format_Grayscale8).scaled(self.originalWidth,self.originalHeight,aspectRatioMode=Qt.KeepAspectRatio)
                qimgHeight = saveImg.height()
                qimgWidth = saveImg.width()

                saveArr = np.zeros((qimgHeight,qimgWidth),dtype='uint8')
                for rowIndex in range(qimgHeight):

                        for colIndex in range(qimgWidth):
                                pixVal = qGray(saveImg.pixel(colIndex,rowIndex))
                                if pixVal > 0:
                                        saveArr[rowIndex,colIndex] = 1

                io.imsave(savePath, saveArr)

        def reset(self):
                self.maskQimage = QImage(self.RGBAImg, self.originalWidth, self.originalHeight, self.RGBAImg.strides[0], QImage.Format_RGBA8888).scaled(512, 512, aspectRatioMode=Qt.KeepAspectRatio)
                self.maskQpixmap = QPixmap(self.maskQimage)
                self.label.setPixmap(self.maskQpixmap)
                self.update()

        def clear(self):
                self.imgFill = QColor(0, 0, 0, self.alpha)
                self.maskQimage = QImage(self.RGBAImg, self.originalWidth, self.originalHeight, self.RGBAImg.strides[0], QImage.Format_RGBA8888).scaled(512, 512, aspectRatioMode=Qt.KeepAspectRatio)
                self.maskQimage.fill(self.imgFill)
                self.maskQpixmap = QPixmap(self.maskQimage)
                self.label.setPixmap(self.maskQpixmap)
                self.update()

        def threePx(self):
                self.brushSize = 3

        def fivePx(self):
                self.brushSize = 5

        def sevenPx(self):
                self.brushSize = 7

        def ninePx(self):
                self.brushSize = 9

        def blackColor(self):
                self.brushColor = QColor(0, 0, 0, self.alpha)

        def redColor(self):
                self.brushColor = QColor(255, 0, 0, self.alpha)

        def whiteColor(self):
                self.brushColor = QColor(255, 255, 255, self.alpha)

        def setImg(self, img):
                self.RGBImg = color.label2rgb(img)
                # img[img>0] = 255
                # self.RGBImg = color.gray2rgb(img).astype('uint8')
                # self.RGBImg[:,:,1:] = 0 # set GB channels to 0 to make the transarency mask red
                alphaFloat = 0.25
                alphaArray = np.zeros(img.shape, dtype='uint8')
                alphaArray = np.expand_dims(alphaArray, -1)
                self.alpha = int(255*alphaFloat)
                alphaArray[...] = self.alpha
                self.RGBAImg = np.append(self.RGBImg, alphaArray, axis=-1)

                self.originalHeight, self.originalWidth, self.originalChannelNumber = self.RGBAImg.shape
                self.maskQimage = QImage(self.RGBAImg, self.originalWidth, self.originalHeight, self.RGBAImg.strides[0], QImage.Format_RGBA8888).scaled(512, 512, aspectRatioMode=Qt.KeepAspectRatio)
                self.maskQpixmap = QPixmap(self.maskQimage)
                self.label.setPixmap(self.maskQpixmap)

        def next_frame(self):
                self.frameIndex += 1
                try:
                    experiment_name = params['experiment_name']
                    original_file_name = self.maskImgPath
                    pat = re.compile(r'.+(xy\d{3,4})_(p\d{3,4})_.+') # supports 3- or 4-digit naming
                    mat = pat.match(original_file_name)
                    fovID = mat.groups()[0]
                    peakID = mat.groups()[1]
                    fileBaseName = '{}_{}_{}_t{:0=4}.tif'.format(experiment_name, fovID, peakID, self.frameIndex+1)
                    savePath = os.path.join(self.label_dir,fileBaseName)

                    if os.path.isfile(savePath):
                        print('Re-annotated mask exists in training directory. Loading it.')
                        # add widget to express whether this mask is one you already re-annotated
                        self.maskStack[self.frameIndex,:,:] = io.imread(savePath)
                    img = self.maskStack[self.frameIndex,:,:]
                except IndexError:
                    sys.exit("You've already edited the final frame's mask. Write in functionality to increment to next peak_id now!")

                self.setImg(img)

        def prior_frame(self):
                self.frameIndex -= 1
                try:
                    experiment_name = params['experiment_name']
                    original_file_name = self.maskImgPath
                    pat = re.compile(r'.+(xy\d{3,4})_(p\d{3,4})_.+') # supports 3- or 4-digit naming
                    mat = pat.match(original_file_name)
                    fovID = mat.groups()[0]
                    peakID = mat.groups()[1]
                    fileBaseName = '{}_{}_{}_t{:0=4}.tif'.format(experiment_name, fovID, peakID, self.frameIndex+1)
                    savePath = os.path.join(self.label_dir,fileBaseName)

                    if os.path.isfile(savePath):
                        print('Re-annotated mask exists in training directory. Loading it.')
                        # add widget to express whether this mask is one you already re-annotated
                        self.maskStack[self.frameIndex,:,:] = io.imread(savePath)
                    img = self.maskStack[self.frameIndex,:,:]
                except IndexError:
                        sys.exit("You've already edited the last frame's mask. Write in functionality to increment to next peak_id now!")
                self.setImg(img)

        def next_peak(self):
                self.imgIndex += 1
                self.maskImgPath = self.imgPaths[self.fov_id][self.imgIndex][1]
                self.maskStack = io.imread(self.maskImgPath)

                self.frameIndex = 0

                experiment_name = params['experiment_name']
                original_file_name = self.maskImgPath
                pat = re.compile(r'.+(xy\d{3,4})_(p\d{3,4})_.+') # supports 3- or 4-digit naming
                mat = pat.match(original_file_name)
                fovID = mat.groups()[0]
                peakID = mat.groups()[1]
                fileBaseName = '{}_{}_{}_t{:0=4}.tif'.format(experiment_name, fovID, peakID, self.frameIndex+1)
                savePath = os.path.join(self.label_dir,fileBaseName)

                if os.path.isfile(savePath):
                    print('Re-annotated mask exists in training directory. Loading it.')
                    # add widget to express whether this mask is one you already re-annotated
                    self.maskStack[self.frameIndex,:,:] = io.imread(savePath)

                img = self.maskStack[self.frameIndex,:,:]
                self.setImg(img)

        def prior_peak(self):
                self.imgIndex -= 1
                self.maskImgPath = self.imgPaths[self.fov_id][self.imgIndex][1]
                self.maskStack = io.imread(self.maskImgPath)

                self.frameIndex = 0

                experiment_name = params['experiment_name']
                original_file_name = self.maskImgPath
                pat = re.compile(r'.+(xy\d{3,4})_(p\d{3,4})_.+') # supports 3- or 4-digit naming
                mat = pat.match(original_file_name)
                fovID = mat.groups()[0]
                peakID = mat.groups()[1]
                fileBaseName = '{}_{}_{}_t{:0=4}.tif'.format(experiment_name, fovID, peakID, self.frameIndex+1)
                savePath = os.path.join(self.label_dir,fileBaseName)

                if os.path.isfile(savePath):
                    print('Re-annotated mask exists in training directory. Loading it.')
                    # add widget to express whether this mask is one you already re-annotated
                    self.maskStack[self.frameIndex,:,:] = io.imread(savePath)
                img = self.maskStack[self.frameIndex,:,:]
                self.setImg(img)

        def next_fov(self):
                self.fovIndex += 1
                self.fov_id = self.fov_id_list[self.fovIndex]

                self.imgIndex = 0
                self.maskImgPath = self.imgPaths[self.fov_id][self.imgIndex][1]
                self.maskStack = io.imread(self.maskImgPath)

                self.frameIndex = 0

                experiment_name = params['experiment_name']
                original_file_name = self.maskImgPath
                pat = re.compile(r'.+(xy\d{3,4})_(p\d{3,4})_.+') # supports 3- or 4-digit naming
                mat = pat.match(original_file_name)
                fovID = mat.groups()[0]
                peakID = mat.groups()[1]
                fileBaseName = '{}_{}_{}_t{:0=4}.tif'.format(experiment_name, fovID, peakID, self.frameIndex+1)
                savePath = os.path.join(self.label_dir,fileBaseName)

                if os.path.isfile(savePath):
                    print('Re-annotated mask exists in training directory. Loading it.')
                    # add widget to express whether this mask is one you already re-annotated
                    self.maskStack[self.frameIndex,:,:] = io.imread(savePath)
                img = self.maskStack[self.frameIndex,:,:]
                self.setImg(img)

        def prior_fov(self):
                self.fovIndex -= 1
                self.fov_id = self.fov_id_list[self.fovIndex]

                self.imgIndex = 0
                self.maskImgPath = self.imgPaths[self.fov_id][self.imgIndex][1]
                self.maskStack = io.imread(self.maskImgPath)

                self.frameIndex = 0
                experiment_name = params['experiment_name']
                original_file_name = self.maskImgPath
                pat = re.compile(r'.+(xy\d{3,4})_(p\d{3,4})_.+') # supports 3- or 4-digit naming
                mat = pat.match(original_file_name)
                fovID = mat.groups()[0]
                peakID = mat.groups()[1]
                fileBaseName = '{}_{}_{}_t{:0=4}.tif'.format(experiment_name, fovID, peakID, self.frameIndex+1)
                savePath = os.path.join(self.label_dir,fileBaseName)

                if os.path.isfile(savePath):
                    print('Re-annotated mask exists in training directory. Loading it.')
                    # add widget to express whether this mask is one you already re-annotated
                    self.maskStack[self.frameIndex,:,:] = io.imread(savePath)
                img = self.maskStack[self.frameIndex,:,:]
                self.setImg(img)

class PhaseWidget(QWidget):

        def __init__(self,imgPaths,fov_id_list,image_dir,frame_index):
                super(PhaseWidget, self).__init__()

                # print(imgPaths)
                self.image_dir = image_dir

                self.imgPaths = imgPaths
                self.fov_id_list = fov_id_list
                self.fovIndex = 0
                self.fov_id = self.fov_id_list[self.fovIndex]

                self.imgIndex = 0
                self.phaseImgPath = self.imgPaths[self.fov_id][self.imgIndex][0]
                self.phaseStack = io.imread(self.phaseImgPath)

                self.frameIndex = frame_index
                img = self.phaseStack[self.frameIndex,:,:]
                self.originalImgMax = np.max(img)
                originalRGBImg = color.gray2rgb(img/2**16*2**8).astype('uint8')
                self.originalPhaseQImage = QImage(originalRGBImg, originalRGBImg.shape[1], originalRGBImg.shape[0], originalRGBImg.strides[0], QImage.Format_RGB888)

                rescaledImg = img/self.originalImgMax*255
                RGBImg = color.gray2rgb(rescaledImg).astype('uint8')
                self.originalHeight, self.originalWidth, self.originalChannelNumber = RGBImg.shape
                self.phaseQimage = QImage(RGBImg, RGBImg.shape[1], RGBImg.shape[0], RGBImg.strides[0], QImage.Format_RGB888).scaled(512, 512, aspectRatioMode=Qt.KeepAspectRatio)
                self.phaseQpixmap = QPixmap(self.phaseQimage)

                self.label = QLabel(self)
                self.label.setPixmap(self.phaseQpixmap)

        def setImg(self, img):
                self.originalImgMax = np.max(img)
                originalRGBImg = color.gray2rgb(img/2**16*2**8).astype('uint8')
                self.originalPhaseQImage = QImage(originalRGBImg, originalRGBImg.shape[1], originalRGBImg.shape[0], originalRGBImg.strides[0], QImage.Format_RGB888)

                rescaledImg = img/np.max(img)*255
                RGBImg = color.gray2rgb(rescaledImg).astype('uint8')
                self.phaseQimage = QImage(RGBImg, RGBImg.shape[1], RGBImg.shape[0], RGBImg.strides[0], QImage.Format_RGB888).scaled(512, 512, aspectRatioMode=Qt.KeepAspectRatio)
                self.phaseQpixmap = QPixmap(self.phaseQimage)
                self.label.setPixmap(self.phaseQpixmap)

        def next_frame(self):
                self.frameIndex += 1

                try:
                        img = self.phaseStack[self.frameIndex,:,:]
                except IndexError:
                        sys.exit("You've already edited the last frame's mask. Write in functionality to increment to next peak_id now!")
                self.setImg(img)

        def prior_frame(self):
                self.frameIndex -= 1

                try:
                        img = self.phaseStack[self.frameIndex,:,:]
                except IndexError:
                        sys.exit("You've already edited the last frame's mask. Write in functionality to increment to next peak_id now!")
                self.setImg(img)

        def next_peak(self):

                self.imgIndex += 1
                self.phaseImgPath = self.imgPaths[self.fov_id][self.imgIndex][0]
                self.phaseStack = io.imread(self.phaseImgPath)

                self.frameIndex = 0
                img = self.phaseStack[self.frameIndex,:,:]
                self.setImg(img)

        def prior_peak(self):

                self.imgIndex -= 1
                self.phaseImgPath = self.imgPaths[self.fov_id][self.imgIndex][0]
                self.phaseStack = io.imread(self.phaseImgPath)

                self.frameIndex = 0
                img = self.phaseStack[self.frameIndex,:,:]
                self.setImg(img)

        def next_fov(self):
                self.fovIndex += 1
                self.fov_id = self.fov_id_list[self.fovIndex]

                self.imgIndex = 0
                self.phaseImgPath = self.imgPaths[self.fov_id][self.imgIndex][0]
                self.phaseStack = io.imread(self.phaseImgPath)

                self.frameIndex = 0
                img = self.phaseStack[self.frameIndex,:,:]
                self.setImg(img)

        def prior_fov(self):
                self.fovIndex -= 1
                self.fov_id = self.fov_id_list[self.fovIndex]

                self.imgIndex = 0
                self.phaseImgPath = self.imgPaths[self.fov_id][self.imgIndex][0]
                self.phaseStack = io.imread(self.phaseImgPath)

                self.frameIndex = 0
                img = self.phaseStack[self.frameIndex,:,:]
                self.setImg(img)

        def buttonSave(self):
                experiment_name = params['experiment_name']
                original_file_name = self.phaseImgPath
                pat = re.compile(r'.+(xy\d{3,4})_(p\d{3,4})_.+')
                mat = pat.match(original_file_name)
                fovID = mat.groups()[0]
                peakID = mat.groups()[1]
                fileBaseName = '{}_{}_{}_t{:0=4}.tif'.format(experiment_name, fovID, peakID, self.frameIndex+1)
                savePath = os.path.join(self.image_dir,fileBaseName)
                print("Saved phase image as: ", savePath)

                if not os.path.isdir(self.image_dir):
                        os.makedirs(self.image_dir)

                saveImg = self.originalPhaseQImage.convertToFormat(QImage.Format_Grayscale8)
                saveImg.save(savePath)

if __name__ == "__main__":

    # imgPaths = {1:[('/home/wanglab/sandbox/trackingGUI/testset1/analysis_20190312/channels/testset1_xy001_p0033_c1.tif',
    #              '/home/wanglab/sandbox/trackingGUI/testset1/analysis_20190312/segmented/testset1_xy001_p0033_seg_unet.tif'),
    #             ('/home/wanglab/sandbox/trackingGUI/testset1/analysis_20190312/channels/testset1_xy001_p0077_c1.tif',
    #              '/home/wanglab/sandbox/trackingGUI/testset1/analysis_20190312/channels/testset1_xy001_p0077_seg_unet.tif')],
    #             2:[('/home/wanglab/sandbox/trackingGUI/testset1/analysis_20190312/channels/testset1_xy002_p0078_c1.tif',
    #                 '/home/wanglab/sandbox/trackingGUI/testset1/analysis_20190312/segmented/testset1_xy002_p0078_seg_unet.tif'),
    #                ('/home/wanglab/sandbox/trackingGUI/testset1/analysis_20190312/channels/testset1_xy002_p0121_c1.tif',
    #                 '/home/wanglab/sandbox/trackingGUI/testset1/analysis_20190312/segmented/testset1_xy002_p0121_seg_unet.tif')]}
    #
    # fov_id_list = [1,2]

    training_dir = '/home/wanglab/sandbox/trackingGUI'

    global params
    param_file_path = '/home/wanglab/sandbox/trackingGUI/testset1/analysis_20190312/testset1_params_20190312.yaml'
    params = mm3.init_mm3_helpers(param_file_path)

    app = QApplication(sys.argv)
    window = Window(training_dir=training_dir)
    window.show()
    app.exec_()
