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

import argparse
import glob
from pprint import pprint # for human readable file output
import pickle as pickle
import sys
import re
import os
import random
import yaml
import multiprocessing

sys.path.insert(0, '/home/wanglab/src/mm3/') # Jeremy's path to mm3 folder
sys.path.insert(0, '/home/wanglab/src/mm3/aux/')

import mm3_helpers as mm3
import mm3_plots


class Window(QMainWindow):

    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        top = 400
        left = 400
        width = 800
        height = 600

        self.setWindowTitle("Go ahead. Update your tracking results.")
        self.setGeometry(top,left,width,height)

        # load specs file
        with open(os.path.join(params['ana_dir'], 'specs.yaml'), 'r') as specs_file:
            specs = yaml.safe_load(specs_file)

        self.frames = FrameImgWidget(specs=specs)
        # make scene the central widget
        self.setCentralWidget(self.frames)

        eventButtonGroup = QButtonGroup()
        migrateButton = QRadioButton("Migration")
        migrateButton.setShortcut("Ctrl+M")
        migrateButton.clicked.connect(self.frames.scene.set_migration)
        migrateButton.click()
        eventButtonGroup.addButton(migrateButton)

        childrenButton = QRadioButton("Children")
        childrenButton.setShortcut("Ctrl+C")
        childrenButton.clicked.connect(self.frames.scene.set_children)
        eventButtonGroup.addButton(childrenButton)

        # born button isn't really necessary, since we can get all that information
        #   from the children events that are already present
        # bornButton = QRadioButton("Born")
        # bornButton.setShortcut("Ctrl+C")
        # bornButton.clicked.connect(self.threeFrames.scene.set_born)
        # eventButtonGroup.addButton(bornButton)

        dieButton = QRadioButton("Die")
        dieButton.setShortcut("Ctrl+D")
        dieButton.clicked.connect(self.frames.scene.set_die)
        eventButtonGroup.addButton(dieButton)

        appearButton = QRadioButton("Appear")
        appearButton.setShortcut("Ctrl+A")
        appearButton.clicked.connect(self.frames.scene.set_appear)
        eventButtonGroup.addButton(appearButton)

        disappearButton = QRadioButton("Disappear")
        disappearButton.clicked.connect(self.frames.scene.set_disappear)
        eventButtonGroup.addButton(disappearButton)

        removeEventsButton = QRadioButton("Remove events")
        removeEventsButton.clicked.connect(self.frames.scene.remove_all_cell_events)
        eventButtonGroup.addButton(removeEventsButton)

        eventButtonLayout = QVBoxLayout()
        eventButtonLayout.addWidget(migrateButton)
        eventButtonLayout.addWidget(childrenButton)
        # eventButtonLayout.addWidget(bornButton)
        eventButtonLayout.addWidget(dieButton)
        eventButtonLayout.addWidget(appearButton)
        eventButtonLayout.addWidget(disappearButton)
        eventButtonLayout.addWidget(removeEventsButton)

        eventButtonGroupWidget = QWidget()
        eventButtonGroupWidget.setLayout(eventButtonLayout)

        eventButtonDockWidget = QDockWidget()
        eventButtonDockWidget.setWidget(eventButtonGroupWidget)
        self.addDockWidget(Qt.LeftDockWidgetArea, eventButtonDockWidget)

        advancePeakButton = QPushButton("Next peak")
        advancePeakButton.setShortcut("Ctrl+P")
        advancePeakButton.clicked.connect(self.frames.scene.next_peak)

        priorPeakButton = QPushButton("Prior peak")
        priorPeakButton.clicked.connect(self.frames.scene.prior_peak)

        advanceFOVButton = QPushButton("Next FOV")
        advanceFOVButton.clicked.connect(self.frames.scene.next_fov)

        priorFOVButton = QPushButton("Prior FOV")
        priorFOVButton.clicked.connect(self.frames.scene.prior_fov)

        saveUpdatedTracksButton = QPushButton("Save updated tracking info\nfor neural net training data")
        saveUpdatedTracksButton.clicked.connect(self.frames.scene.save_updates)

        fileAdvanceLayout = QVBoxLayout()
        fileAdvanceLayout.addWidget(advancePeakButton)
        fileAdvanceLayout.addWidget(priorPeakButton)
        fileAdvanceLayout.addWidget(advanceFOVButton)
        fileAdvanceLayout.addWidget(priorFOVButton)
        fileAdvanceLayout.addWidget(saveUpdatedTracksButton)

        fileAdvanceGroupWidget = QWidget()
        fileAdvanceGroupWidget.setLayout(fileAdvanceLayout)

        fileAdvanceDockWidget = QDockWidget()
        fileAdvanceDockWidget.setWidget(fileAdvanceGroupWidget)
        self.addDockWidget(Qt.RightDockWidgetArea, fileAdvanceDockWidget)

class FrameImgWidget(QWidget):
    # class for setting three frames side-by-side as a central widget in a QMainWindow object
    def __init__(self,specs):
        super(FrameImgWidget, self).__init__()

        # add images and cell regions as ellipses to each frame in a QGraphicsScene object
        self.scene = FrameItem(specs)
        self.view = QGraphicsView(self)
        self.view.setScene(self.scene)

class FrameItem(QGraphicsScene):
    # add more functionality for setting event type, i.e., parent-child, migrate, death, leave frame, etc..

    def __init__(self,specs):
        super(FrameItem, self).__init__()

        self.specs = specs
        # add QImages to scene (try three frames)
        self.fov_id_list = [fov_id for fov_id in specs.keys()]
        self.center_frame_index = 1

        self.fovIndex = 0
        self.fov_id = self.fov_id_list[self.fovIndex]

        self.peak_id_list_in_fov = [peak_id for peak_id in specs[self.fov_id].keys() if specs[self.fov_id][peak_id] == 1]

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

        self.pickle_file_name = os.path.join(params['cell_dir'], 'updated_tracks.pkl')
        # look for previously updated tracking information and load that if it is found.
        if not os.path.isfile(self.pickle_file_name):
            # get tracking information in a format usable and updatable by qgraphicsscene
            self.regions_and_events_by_time = self.create_tracking_information()
        else:
            with open(self.pickle_file_name, 'rb') as pickle_file:
                try:
                    print("Found updated track information in {}. Uploading and plotting it.".format(self.pickle_file_name))
                    self.regions_and_events_by_time = pickle.load(pickle_file)
                except Exception as e:
                    print("Could not load pickle file specified.")
                    print(e)
        # TO DO: enable zooming in/out with mouse wheel
        self.all_frames_by_time_dict = self.all_phase_img_and_regions()

        self.drawing = False
        self.remove_events = False
        self.brushSize = 2
        self.brushColor = QColor('black')
        self.lastPoint = QPoint()
        self.pen = QPen()

        # class options
        self.event_types_list = ["ChildLine","MigrationLine","DieSymbol","AppearSymbol","BornSymbol","DisappearSymbol"]
        self.line_events_list = ["ChildLine","MigrationLine"]
        # the below lookup table may need reworked to handle migration and child lines to/from a cell
        #   or the handling may be better done in the update_cell_info function
        self.event_type_index_lookup = {"MigrationLine":0, "ChildLine":1,
                                        "DieSymbol":2, "BornSymbol":3,
                                        "AppearSymbol":4, "DisappearSymbol":5}
        # given an event type for a cell, what are the event types that are incompatible within that same cell?
        self.forbidden_events_lookup = {"MigrationStart":["ChildStart","DisappearSymbol","DieSymbol","MigrationStart"],
                                           "MigrationEnd":["ChildEnd","AppearSymbol","BornSymbol","MigrationEnd"],
                                           "ChildStart":["MigrationStart","DisappearSymbol","DieSymbol","ChildStart"],
                                           "ChildEnd":["MigrationEnd","AppearSymbol","ChildEnd"],
                                           "BornSymbol":["MigrationEnd","AppearSymbol","BornSymbol"],
                                           "AppearSymbol":["MigrationEnd","ChildEnd","BornSymbol","AppearSymbol"],
                                           "DisappearSymbol":["MigrationStart","ChildStart","DieSymbol","DisappearSymbol"],
                                           "DieSymbol":["MigrationStart","ChildStart","DisappearSymbol","DieSymbol"]}
        self.migration = False
        self.die = False
        self.children = False
        self.birth = False
        self.appear = False
        self.disappear = False

        # apply cell events to the scene
        self.draw_cell_events()

    def save_updates(self):
        track_info = self.regions_and_events_by_time
        pickle_info = {}
        for time, stuff in track_info.items():
            time_info = track_info[time]
            pickle_info[time] = {}
            pickle_info[time]['matrix'] = time_info['matrix']
            pickle_info[time]['regions'] = {}

            for region_label, region in time_info['regions'].items():
                time_region_info = time_info['regions']
                pickle_info[time]['regions'][region_label] = {}
                pickle_info[time]['regions'][region_label]['events'] = time_region_info[region_label]['events']
                pickle_info[time]['regions'][region_label]['props'] = time_region_info[region_label]['props']

        with open(self.pickle_file_name, 'wb') as track_file:
            try:
                pickle.dump(pickle_info, track_file)
                track_file.close()
                print("Saved updated tracking information to {}.".format(os.path.join(params['cell_dir'], 'updated_tracks.pkl')))
            except Exception as e:
                track_file.close()
                print(str(e))

    def next_peak(self):
        # start by removing all current graphics items from the scene, the scene here being 'self'
        self.clear()

        self.center_frame_index = 1
        self.peakIndex += 1
        self.peak_id = self.peak_id_list_in_fov[self.peakIndex]
        # print(self.peak_id)

        # construct image stack file names from params
        self.phaseImgPath = os.path.join(params['chnl_dir'], "{}_xy{:0=3}_p{:0=4}_{}.tif".format(params['experiment_name'], self.fov_id, self.peak_id, params['phase_plane']))
        self.labelImgPath = os.path.join(params['seg_dir'], "{}_xy{:0=3}_p{:0=4}_seg_unet.tif".format(params['experiment_name'], self.fov_id, self.peak_id))

        self.labelStack = io.imread(self.labelImgPath)
        self.phaseStack = io.imread(self.phaseImgPath)

        self.regions_and_events_by_time = self.create_tracking_information()
        self.all_frames_by_time_dict = self.all_phase_img_and_regions()

        self.draw_cell_events()

    def prior_peak(self):
        # start by removing all current graphics items from the scene, the scene here being 'self'
        self.clear()

        self.center_frame_index = 1
        self.peakIndex -= 1
        self.peak_id = self.peak_id_list_in_fov[self.peakIndex]
        # print(self.peak_id)

        # construct image stack file names from params
        self.phaseImgPath = os.path.join(params['chnl_dir'], "{}_xy{:0=3}_p{:0=4}_{}.tif".format(params['experiment_name'], self.fov_id, self.peak_id, params['phase_plane']))
        self.labelImgPath = os.path.join(params['seg_dir'], "{}_xy{:0=3}_p{:0=4}_seg_unet.tif".format(params['experiment_name'], self.fov_id, self.peak_id))
        # print(self.phaseImgPath)
        # print(self.labelImgPath)

        self.labelStack = io.imread(self.labelImgPath)
        self.phaseStack = io.imread(self.phaseImgPath)

        self.regions_and_events_by_time = self.create_tracking_information()

        self.all_frames_by_time_dict = self.all_phase_img_and_regions()
        self.draw_cell_events()

    def next_fov(self):
        # start by removing all current graphics items from the scene, the scene here being 'self'
        self.clear()

        self.center_frame_index = 1

        self.fovIndex += 1
        self.fov_id = self.fov_id_list[self.fovIndex]

        self.peak_id_list_in_fov = [peak_id for peak_id in self.specs[self.fov_id].keys() if self.specs[self.fov_id][peak_id] == 1]

        self.peakIndex = 0
        self.peak_id = self.peak_id_list_in_fov[self.peakIndex]

        # construct image stack file names from params
        self.phaseImgPath = os.path.join(params['chnl_dir'], "{}_xy{:0=3}_p{:0=4}_{}.tif".format(params['experiment_name'], self.fov_id, self.peak_id, params['phase_plane']))
        self.labelImgPath = os.path.join(params['seg_dir'], "{}_xy{:0=3}_p{:0=4}_seg_unet.tif".format(params['experiment_name'], self.fov_id, self.peak_id))
        # print(self.phaseImgPath)
        # print(self.labelImgPath)

        self.labelStack = io.imread(self.labelImgPath)
        self.phaseStack = io.imread(self.phaseImgPath)

        self.regions_and_events_by_time = self.create_tracking_information()

        self.all_frames_by_time_dict = self.all_phase_img_and_regions()
        self.draw_cell_events()

    def prior_fov(self):
        # start by removing all current graphics items from the scene, the scene here being 'self'
        self.clear()

        self.center_frame_index = 1

        self.fovIndex -= 1
        self.fov_id = self.fov_id_list[self.fovIndex]

        self.peak_id_list_in_fov = [peak_id for peak_id in self.specs[self.fov_id].keys() if self.specs[self.fov_id][peak_id] == 1]

        self.peakIndex = 0
        self.peak_id = self.peak_id_list_in_fov[self.peakIndex]

        # construct image stack file names from params
        self.phaseImgPath = os.path.join(params['chnl_dir'], "{}_xy{:0=3}_p{:0=4}_{}.tif".format(params['experiment_name'], self.fov_id, self.peak_id, params['phase_plane']))
        self.labelImgPath = os.path.join(params['seg_dir'], "{}_xy{:0=3}_p{:0=4}_seg_unet.tif".format(params['experiment_name'], self.fov_id, self.peak_id))
        # print(self.phaseImgPath)
        # print(self.labelImgPath)

        self.labelStack = io.imread(self.labelImgPath)
        self.phaseStack = io.imread(self.phaseImgPath)

        self.regions_and_events_by_time = self.create_tracking_information()

        self.all_frames_by_time_dict = self.all_phase_img_and_regions()
        self.draw_cell_events()

    def all_phase_img_and_regions(self):

        frame_dict_by_time = {}

        xPos = 0
        for time in self.regions_and_events_by_time.keys():
            frame_index = time-1
            frame, regions = self.phase_img_and_regions(frame_index)
            frame_dict_by_time[time] = self.addPixmap(frame)
            frame_dict_by_time[time].time = time
            self.add_regions_to_frame(regions, frame_dict_by_time[time])
            frame_dict_by_time[time].setPos(xPos, 0)
            xPos += frame_dict_by_time[time].pixmap().width()

        return(frame_dict_by_time)

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
        phaseQpixmap.time = time

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

            ####### NOTE: WOULD BE NICE TO GET A QGRAPHICS PATH ITEM WORKING, BUT IT LOOKS PRETTY INVOLVED ########
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

    def draw_cell_events(self, start_time=1, end_time=None, update=False):

        # Here is where we will draw the intial lines and symbols representing
        #   cell events and linkages between cells.
        # Set up a list of time points at which we will re-draw events
        # the ability to create this list greatkt increases the speed of re-drawing
        #   events after an update.
        if end_time is not None:
            max_time = end_time
        else: max_time = np.max([item.time for item in self.items() if item.type() == 7])

        valid_times = [i for i in range(start_time, max_time+1)]

        # regions_and_events_by_time is a dictionary, the keys of which are 1-indexed frame numbers
        # for each frame, there is a dictionary with the following keys: 'matrix' and 'regions'
        #   'matrix' is a 2D array, for which the row index is the region label at time t, and the column index is the region label at time t+1
        #      If a region disappears from t to t+1, it will receive a 1 in the column with index 0.
        #   'regions' is a dictionary with each region's label as a separate key.
        #      Each region in 'regions' is another dictionary, with 'events', which contains a 1D array identifying the events that correspond to the connections in 'matrix',
        #                                                      and 'props', which contains all region properties that measure.regionprops retuns for the labelled image at time t.
        # The 'events' array is binary. The events are [migration, division, death, birth, appearance, disappearance, no_data], where a 0 at a given position in the 'events'
        #      array indicates the given event did not occur, and a 1 indicates it did occur.

        # loop through frames at valid times
        for time in valid_times:
            if time in self.all_frames_by_time_dict:

                frame = self.all_frames_by_time_dict[time]
                # if the frame has child items
                if len(frame.childItems()) > 0:

                    for self.startItem in frame.childItems():
                        # test if this is an ellipse item. If it is, draw event symbols.
                        if self.startItem.type() == 4:
                            # remove all previously drawn events belonging entirely to,
                            #   or emanating from, this cell.
                            if update:
                                items, items_list = self.get_all_cell_event_items(self.startItem, return_forbidden_items_list=True)
                                for i, item_type in enumerate(items_list):
                                    if item_type not in ["ChildEnd","MigrationEnd"]:
                                        self.removeItem(items[i])

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

                            if 2 in event_indices:
                                # If the second element in event_indices was 1, the cell dies between this frame and the next.
                                self.set_die()
                                self.eventItem = self.set_event_item()
                                self.addItem(self.eventItem)

                            # Ignoring "born" events in the GUI, and keeping their management
                            #   strictly in the back-end, since we can infer "born" from
                            #   whether a cell has a ChildLine that terminates within it.

                            # if 3 in event_indices:
                            #     # if the third element in event_indices was 1,
                            #     #   the cell cell was born between this frame and the previous one.
                            #     self.set_born()
                            #     self.eventItem = self.set_event_item()
                            #     self.addItem(self.eventItem)

                            if 4 in event_indices:
                                # if the fourth element in event_indices was 1,
                                #   the cell cell appeared between this frame and the previous one.
                                self.set_appear()
                                self.eventItem = self.set_event_item()
                                self.addItem(self.eventItem)

                            if 5 in event_indices:
                                # if the fifth element in event_indices was 1,
                                #   the cell cell disappeared between this frame and the previous one.
                                self.set_disappear()
                                self.eventItem = self.set_event_item()
                                self.addItem(self.eventItem)

                            try:
                                nextFrame = self.all_frames_by_time_dict[time+1]
                                for self.endItem in nextFrame.childItems():
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
                            except KeyError:
                                continue

    def get_ellipse(self, point):
        # function for finding the ellipse under your mouse click or mouse release,
        #  since items can be stacked, a line you previously drew can obscure the
        #  ellipse you intended to select
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

    def get_all_cell_event_items(self, cell, return_forbidden_items_list=False):
        # Fetch all items that collide with the clicked ellipse,
        #   evalute whether they are "events".
        # Further evaluate whether the event items' items are the same as
        #   the clicked ellipse. The reason for this is that some text items take more
        #   space than they appear to occupy in the GUI, so they may collide with
        #   ellipses neighboring the clicked ellipse

        items_colliding_with_cell = cell.collidingItems()
        event_items = []
        event_types = []

        for item in items_colliding_with_cell:
            itemType = item.type()
            if itemType in self.event_types_list:
                if itemType in self.line_events_list:
                    # determine whether the child or migration line's start or end point
                    #   belongs to the clicked cell
                    if (item.startItem == cell or item.endItem == cell):
                        # if we're calling this function with return_forbidden_items_list=True,
                        #   we need to figure out whether each migration and child line in this
                        #   cell represent the endpoint of the line or the startpoint of the line.
                        if return_forbidden_items_list:
                            # is this cell the endItem or the startItem for this item?
                            if itemType == "MigrationLine":
                                if item.startItem == cell:
                                    item_name = "MigrationStart"
                                elif item.endItem == cell:
                                    item_name = "MigrationEnd"

                            elif itemType == "ChildLine":
                                if item.startItem == cell:
                                    item_name = "ChildStart"
                                elif item.endItem == cell:
                                    item_name = "ChildEnd"

                            event_types.append(item_name)

                        event_items.append(item)
                else:
                    # determine whether the die, born, appear, or disappear event
                    #    belongs to the clicked cell
                    if item.item == cell:
                        if return_forbidden_items_list:
                            item_name = itemType
                            event_types.append(item_name)
                        event_items.append(item)

        if return_forbidden_items_list:
            return(event_items, event_types)

        return(event_items)

    def update_tracking_info(self):
        # This function updates the matrix and events for all cells in your scene.
        # First, get the maximum time so we don't get a key error later on
        max_time = np.max([key for key in self.all_frames_by_time_dict.keys()])

        # Work with the newly added eventItem
        ###################### TO DO: update so that this works with event removal ###########################
        # If the event was either child or migration, do this stuff
        if self.eventItem.type() in self.line_events_list:
            # Identify immediatly affected cells
            cell = self.eventItem.startItem
            frame = self.all_frames_by_time_dict[cell.time]
            self.update_frame_info(frame)

            cell = self.eventItem.endItem
            frame = self.all_frames_by_time_dict[cell.time]
            self.update_frame_info(frame)

        else:
            cell = self.eventItem.item
            frame = self.all_frames_by_time_dict[cell.time]
            self.update_frame_info(frame)

    def update_frame_info(self, frame):
        # This function takes a cell as input and grabs its information from the original
        #   tracking information, then updates the 'matrix' and 'events' objects for the proper frame
        #   and events.
        frame_time = frame.time
        # get all cells in the frame to update each one, in case they were indirectly affected by the event that was drawn
        for cell in frame.childItems():
            # print(cell_time)
            cell_label = cell.cellProps.label
            # print(cell_label)
            # grab all currently-drawn events for the cell of interest
            events, event_items = self.get_all_cell_event_items(cell, return_forbidden_items_list=True)
            # print("Events for cell {} at time {}: \n".format(cell_label, frame_time), event_items)
            # print([event.type() for event in events])

            #  # Fetch the cell's original information
            #    'matrix' is a 2D array, for which the row index is the region label at time t, and the column index is the region label at time t+1
            #     If a region disappears from t to t+1, it will receive a 1 in the column with index 0.
            time_matrix = self.regions_and_events_by_time[frame_time]['matrix']
            cell_events = self.regions_and_events_by_time[frame_time]['regions'][cell_label]['events']

            # print("Events and matrix for cell {} at time {}: \n\n".format(cell_label, frame_time),
            #       cell_events, "\n\n", time_matrix, "\n")
            # relabel all interactions for this cell as zero to later fill in appropriate 1's based on updated events

            time_matrix[cell_label,:] = 0
            # print(cell_events) # for debugging to verify changes are being made appropriately.
            # set all cell events temporarily to zero to fill back in with appropriate 1's
            cell_events[...] = 0

            # If an event is a migration or a child line, determine whether
            #   the migration line starts at the cell to be updated, and whether
            #   the child line begins or ends at the current cell.
            #   Update events array and linkage matrix accordingly
            for event in events:
                event_type = event.type()
                # if the event is a migration or child line
                if event_type in self.line_events_list:
                    # if the line starts with the cell to be updated, set the appropriate
                    #  index in cell_events to 1.

                    if event.startItem == cell:
                        cell_events[self.event_type_index_lookup[event_type]] = 1
                        start_cell_label = event.startItem.cellProps.label
                        end_cell_label = event.endItem.cellProps.label
                        # print(start_cell_label, end_cell_label)
                        time_matrix[start_cell_label,end_cell_label] = 1

                    if event_type == "ChildLine":
                        # if the event is a child line that terminates with this cell,
                        #   set the 'born' index of cell_events to 1
                        if event.endItem == cell:
                            cell_events[self.event_type_index_lookup["BornSymbol"]] = 1
                # if the event is either appear, disappear, or die, do this stuff
                else:
                    cell_events[self.event_type_index_lookup[event_type]] = 1
                    if event_type == "DissapearSymbol":
                        cell_label = event.item.cellProps.label
                        time_matrix[cell_label,0] = 1

            # print("New events and matrix for cell {} at time {}: \n\n".format(cell_label, frame_time),
                  # cell_events, "\n\n", time_matrix, "\n")

    def remove_old_conflicting_events(self, event):
        # This function derives the cells involved in a given newly-annotated event and evaluates
        #   whether a cell has conflicting events, such as two migrations to the
        #   next frame, or three children, etc.. It then attempts to resolve the conflict by removing
        #   the older annotation.
        event_type = event.type()
        # I need to set this up in a way that will allow me to interrogate whether an event type triggers
        #   the incompatible event type lookup in __init__. This will save a lot of headache in coding of logic.
        #
        # Therefore, prioritize establishing list of cell events for cells that interact with the newly drawn annotation


        if event_type in self.line_events_list:
            start_cell = event.startItem
            end_cell = event.endItem
            old_end_cell_events, old_end_cell_event_types = self.get_all_cell_event_items(end_cell, return_forbidden_items_list=True)

            # Name the key for fobidden event lookup.
            if event_type == "MigrationLine":
                forbidden_event_lookup_key = "MigrationEnd"
            elif event_type == "ChildLine":
                forbidden_event_lookup_key = "ChildEnd"
            else:
                forbidden_event_lookup_key = event_type

            # retrieve list of forbidden event types, given our drawn event
            forbidden_events_list = self.forbidden_events_lookup[forbidden_event_lookup_key]

            for i, old_end_cell_event_type in enumerate(old_end_cell_event_types):
                if old_end_cell_event_type in forbidden_events_list:
                    self.removeItem(old_end_cell_events[i])

        else:
            start_cell = event.item

        old_start_cell_events, old_start_cell_event_types = self.get_all_cell_event_items(start_cell, return_forbidden_items_list=True)

        # Name the key for fobidden event lookup.
        if event_type == "MigrationLine":
            forbidden_event_lookup_key = "MigrationStart"
        elif event_type == "ChildLine":
            forbidden_event_lookup_key = "ChildStart"
        else:
            forbidden_event_lookup_key = event_type

        # retrieve list of forbidden event types, given our drawn event
        forbidden_events_list = self.forbidden_events_lookup[forbidden_event_lookup_key]

        child_start_count = 0
        for i, old_start_cell_event_type in enumerate(old_start_cell_event_types):
            if old_start_cell_event_type in forbidden_events_list:
                if old_start_cell_event_type == "ChildStart":
                    if forbidden_event_lookup_key == "ChildStart":
                        if child_start_count == 0:
                            first_child_start_index = i
                        child_start_count += 1
                        if child_start_count == 2:
                            self.removeItem(old_start_cell_events[i])
                            self.removeItem(old_start_cell_events[first_child_start_index])
                    else:
                        self.removeItem(old_start_cell_events[i])
                else:
                    self.removeItem(old_start_cell_events[i])

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:

            self.drawing = True
            self.firstPoint = event.scenePos()
            self.lastPoint = event.scenePos()
            self.startItem = self.get_ellipse(point=self.firstPoint)
            if self.startItem is not None:

                # if we want to remove all events, do it here
                if self.remove_events:
                    events = self.get_all_cell_event_items(self.startItem)
                    for event in events:
                        self.removeItem(event)
                    self.update_tracking_info()

                # if we do not want to remove_events, we are adding an event, so do the following
                else:
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
        if (event.buttons() & Qt.LeftButton) & self.drawing and not self.remove_events:
            if self.startItem is not None:
                self.lastPoint = event.scenePos()
                self.removeItem(self.eventItem)
                self.eventItem = self.set_event_item()
                self.addItem(self.eventItem)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing and not self.remove_events:
            self.lastPoint = event.scenePos()
            self.endItem = self.get_ellipse(point=self.lastPoint)
            if self.startItem is not None:
                if self.endItem is None:
                    if self.migration:
                        self.set_migration()
                    elif self.children:
                        self.set_children()
                    elif self.die:
                        self.set_die()
                    elif self.appear:
                        self.set_appear()
                    elif self.disappear:
                        self.set_disappear()
                    self.removeItem(self.eventItem)


            if self.endItem is not None:
                if self.startItem is not None:
                    start_time = self.startItem.parentItem().time
                    end_time = self.endItem.parentItem().time
                    # A migration or child event cannot go from one frame to the same frame.
                    if (self.startItem.parentItem() == self.endItem.parentItem()) and self.eventItem.type() in self.line_events_list:
                        self.removeItem(self.eventItem)
                        print("Cannot link cells in a single frame as migrated or children. Ignoring selection.")
                    elif abs(start_time - end_time) > 1:
                        self.removeItem(self.eventItem)
                        print("Cannot link cells separated by more than a single frame. Ignoring selection.")
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

                        self.remove_old_conflicting_events(self.eventItem)
                        self.addItem(self.eventItem)
                        # query the currently-drawn annotations in the scene and update all cells' information
                        self.update_tracking_info()
                        # remove old events and draw the newly-updated ones. This is inefficient, but helps you
                        #   to ensure you haven't just messed up the underlying tracking data
                        self.draw_cell_events(start_time=start_time-2, end_time=end_time+2, update=True)

                        ## TO DO: FIGURE OUT WHY EVENT TYPE CHANGES AFTER REDRAWING CELL EVENTS, EVEN THOUGH I HAVE THIS CODE BELOW ###############
                        if self.eventItem.type() == "MigrationLine":
                            self.set_migration()
                        elif self.eventItem.type() == "ChildLine":
                            self.set_children()
                        elif self.eventItem.type() == "DieSymbol":
                            self.set_die()
                        elif self.eventItem.type() == "AppearSymbol":
                            self.set_appear()
                        elif self.eventItem.type() == "DisappearSymbol":
                            self.set_disappear()

            self.drawing = False

    def set_event_item(self):
        if self.migration:
            eventItem = MigrationLine(self.firstPoint, self.lastPoint, self.startItem, self.endItem)
        if self.children:
            eventItem = ChildLine(self.firstPoint, self.lastPoint, self.startItem, self.endItem)
        if self.die:
            eventItem = DieSymbol(self.firstPoint, self.startItem)
        # if self.birth:
        #     eventItem = BornSymbol(self.firstPoint, self.startItem)
        if self.appear:
            eventItem = AppearSymbol(self.firstPoint, self.startItem)
        if self.disappear:
            eventItem = DisappearSymbol(self.firstPoint, self.startItem)

        return(eventItem)

    def remove_all_cell_events(self):
        self.remove_events = True
        self.migration = False
        self.die = False
        self.children = False
        self.birth = False
        self.appear = False
        self.disappear = False

    def set_migration(self):
        # print('clicked set_migration')
        self.remove_events = False
        self.migration = True
        self.die = False
        self.children = False
        self.birth = False
        self.appear = False
        self.disappear = False

    def set_children(self):
        # print('clicked set_children')
        self.remove_events = False
        self.migration = False
        self.die = False
        self.children = True
        self.birth = False
        self.appear = False
        self.disappear = False

    def set_die(self):
        # print('clicked set_die')
        self.remove_events = False
        self.migration = False
        self.die = True
        self.children = False
        self.birth = False
        self.appear = False
        self.disappear = False

    def set_appear(self):
        self.remove_events = False
        self.migration = False
        self.die = False
        self.children = False
        self.birth = False
        self.appear = True
        self.disappear = False

    def set_disappear(self):
        self.remove_events = False
        self.migration = False
        self.die = False
        self.children = False
        self.birth = False
        self.appear = False
        self.disappear = True

    def set_born(self):
        self.remove_events = False
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
        # ensure that lines' starts are to the left of their ends
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

    def __init__(self, point, item):
        super(DieSymbol, self).__init__()

        self.item = item

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
        # ensure that lines' starts are to the left of their ends
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

# class BornSymbol(QGraphicsTextItem):
    #
    # def __init__(self, point, item):
    #     super(BornSymbol, self).__init__()
    #
    #     self.item = item
    #
    #     textColor = QColor(0*255,1*255,1*255)
    #     textFont = QFont()
    #     textFont.setFamily("Times")
    #     textFont.setPixelSize(20)
    #     textFont.setWeight(75) # bold
    #     string = "o"
    #     textPosition = QPoint(point.x()-8, point.y()-17)
    #
    #     self.setPlainText(string)
    #     self.setFont(textFont)
    #     self.setPos(textPosition)
    #     self.setDefaultTextColor(textColor)
    #
    # def type(self):
    #     return("BornSymbol")

class AppearSymbol(QGraphicsTextItem):

    def __init__(self, point, item):
        super(AppearSymbol, self).__init__()

        self.item = item

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

    def __init__(self, point, item):
        super(DisappearSymbol, self).__init__()

        self.item = item

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
    window = Window()
    window.show()
    app.exec_()
