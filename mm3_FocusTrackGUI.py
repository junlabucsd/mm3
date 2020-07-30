#! /usr/bin/env python3
from __future__ import print_function, division

from PyQt5.QtWidgets import (QApplication, QMainWindow, QGraphicsScene, QGraphicsView,
                             QRadioButton, QButtonGroup, QVBoxLayout, QHBoxLayout,
                             QWidget, QLabel, QAction, QDockWidget, QPushButton, QGraphicsItem,
                             QGridLayout, QGraphicsLineItem, QGraphicsPathItem, QGraphicsPixmapItem,
                             QGraphicsEllipseItem, QGraphicsTextItem, QInputDialog)
from PyQt5.QtGui import (QIcon, QImage, QPainter, QPen, QPixmap, qGray, QColor, QPainterPath, QBrush,
                         QTransform, QPolygonF, QFont, QPaintDevice)
from PyQt5.QtCore import Qt, QPoint, QRectF, QLineF
from skimage import io, img_as_ubyte, color, draw, measure, morphology, feature
import numpy as np
from matplotlib import pyplot as plt

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
import pandas as pd

sys.path.insert(0, '/home/wanglab/src/mm3/') # Jeremy's path to mm3 folder
sys.path.insert(0, '/home/wanglab/src/mm3/aux/')

import mm3_helpers as mm3
import mm3_plots


class Window(QMainWindow):

    def __init__(self, channel=1, parent=None):
        super(Window, self).__init__(parent)

        top = 400
        left = 400
        width = 800
        height = 600

        self.setWindowTitle("Update your tracking results and save for deep learning.")
        self.setGeometry(top,left,width,height)

        # load specs file
        with open(os.path.join(params['ana_dir'], 'specs.yaml'), 'r') as specs_file:
            specs = yaml.safe_load(specs_file)

        self.frames = FrameImgWidget(specs=specs, channel=channel)
        # make scene the central widget
        self.setCentralWidget(self.frames)

        eventButtonGroup = QButtonGroup()

        migrateButton = QRadioButton("Migration")
        migrateButton.setShortcut("Ctrl+V")
        migrateButton.setToolTip("(Ctrl+V) Draw white migration lines\nbetween foci in adjacent frames or\nin frames separated by one timepoint.")
        migrateButton.clicked.connect(self.frames.scene.set_migration)
        migrateButton.click()
        eventButtonGroup.addButton(migrateButton)

        childrenButton = QRadioButton("Children")
        childrenButton.setShortcut("Ctrl+C")
        childrenButton.setToolTip("(Ctrl+C) Draw green split lines\nbetween a parent focus and any number of children.")
        childrenButton.clicked.connect(self.frames.scene.set_children)
        eventButtonGroup.addButton(childrenButton)

        joinButton = QRadioButton("Join")
        joinButton.setShortcut("Ctrl+J")
        joinButton.setToolTip("(Ctrl+J) Draw red join lines.")
        joinButton.clicked.connect(self.frames.scene.set_join)
        eventButtonGroup.addButton(joinButton)

        appearButton = QRadioButton("Appear")
        appearButton.setShortcut("Ctrl+A")
        appearButton.setToolTip("(Ctrl+A) Denotes a focus appeared,\nor entered the field of view at this frame.")
        appearButton.clicked.connect(self.frames.scene.set_appear)
        eventButtonGroup.addButton(appearButton)

        disappearButton = QRadioButton("Disappear")
        disappearButton.setShortcut("Ctrl+D")
        disappearButton.setToolTip("(Ctrl+D) Indicate a focus disappears or leaves\nthe field of view after this frame.")
        disappearButton.clicked.connect(self.frames.scene.set_disappear)
        eventButtonGroup.addButton(disappearButton)

        removeEventsButton = QRadioButton("Remove events")
        removeEventsButton.setShortcut("Ctrl+R")
        removeEventsButton.setToolTip("(Ctrl+R) Eliminate events belonging entirely to this\nfocus, and emantating from this focus.")
        removeEventsButton.clicked.connect(self.frames.scene.remove_all_focus_events)
        eventButtonGroup.addButton(removeEventsButton)

        eventButtonLayout = QVBoxLayout()
        eventButtonLayout.addWidget(migrateButton)
        eventButtonLayout.addWidget(childrenButton)
        eventButtonLayout.addWidget(joinButton)
        eventButtonLayout.addWidget(appearButton)
        eventButtonLayout.addWidget(disappearButton)
        eventButtonLayout.addWidget(removeEventsButton)

        eventButtonGroupWidget = QWidget()
        eventButtonGroupWidget.setLayout(eventButtonLayout)

        eventButtonDockWidget = QDockWidget()
        eventButtonDockWidget.setWidget(eventButtonGroupWidget)
        self.addDockWidget(Qt.LeftDockWidgetArea, eventButtonDockWidget)

        goToPeakButton = QPushButton('Go to fov/peak')
        goToPeakButton.clicked.connect(self.frames.get_fov_peak_dialog)

        advancePeakButton = QPushButton("Next peak")
        advancePeakButton.clicked.connect(self.frames.scene.next_peak)

        priorPeakButton = QPushButton("Prior peak")
        priorPeakButton.clicked.connect(self.frames.scene.prior_peak)

        advanceFOVButton = QPushButton("Next FOV")
        advanceFOVButton.clicked.connect(self.frames.scene.next_fov)

        priorFOVButton = QPushButton("Prior FOV")
        priorFOVButton.clicked.connect(self.frames.scene.prior_fov)

        saveUpdatedTracksButton = QPushButton("Save updated tracking info\nfor neural net training data")
        saveUpdatedTracksButton.setShortcut("Ctrl+S")
        saveUpdatedTracksButton.clicked.connect(self.frames.scene.save_updates)

        fileAdvanceLayout = QVBoxLayout()
        fileAdvanceLayout.addWidget(goToPeakButton)
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
    def __init__(self, specs, channel):
        super(FrameImgWidget, self).__init__()
        print("Starting in track edit mode.")
        # add images and focus regions as ellipses to each frame in a QGraphicsScene object
        self.specs = specs
        self.channel = channel
        self.scene = TrackItem(specs, channel)
        self.view = View(self)
        self.view.setScene(self.scene)

    def get_fov_peak_dialog(self):

        fov_id, pressed = QInputDialog.getInt(self,
                                              "Type your desired FOV",
                                              "fov_id (should be an integer):")

        if pressed:
            fov_id = fov_id

        peak_id, pressed = QInputDialog.getInt(self,
                                               "Go to peak",
                                               "peak_id (should be an integer):")

        if pressed:
            peak_id = peak_id

        self.scene.go_to_fov_and_peak_id(fov_id,peak_id)

    # def enter_mask_edit_mode(self):
    #     print("Entering mask edit mode.")
    #     ###### NOTE: consider prompting to save track edits prior to switching modes
    #     self.scene.clear()
    #     self.scene = MaskItem(self.specs)
    #     self.view = View(self)
    #     self.view.setScene(self.scene)
    #
    # def enter_track_edit_mode(self):
    #     print("Entering track edit mode.")
    #     ###### NOTE: consider prompting to save new masks prior to switching modes
    #     self.scene.clear()
    #     self.scene = TrackItem(self.specs)
    #     self.view = View(self)
    #     self.view.setScene(self.scene)

class View(QGraphicsView):
    '''
    Re-implementation of QGraphicsView to accept mouse+Ctrl event
    as a zoom transformation
    '''
    def __init__(self, parent):
        super(View, self).__init__(parent)
        # set upper and lower bounds on zooming
        self.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
        self.maxScale = 2.5
        self.minScale = 0.3

    def wheelEvent(self, event):
        if event.modifiers() == Qt.ControlModifier:
            self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
            m11 = self.transform().m11() # horizontal scale factor
            m12 = self.transform().m12()
            m13 = self.transform().m13()
            m21 = self.transform().m21()
            m22 = self.transform().m22() # vertizal scale factor
            m23 = self.transform().m23()
            m31 = self.transform().m31()
            m32 = self.transform().m32()
            m33 = self.transform().m33()

            adjust = event.angleDelta().y()/120 * 0.1
            if (m11 >= self.maxScale) and (adjust > 0):
                m11 = m11
                m22 = m22
            elif (m11 <= self.minScale) and (adjust < 0):
                m11 = m11
                m22 = m22
            else:
                m11 += adjust
                m22 += adjust
            self.setTransform(QTransform(m11, m12, m13, m21, m22, m23, m31, m32, m33))

        elif event.modifiers() == Qt.AltModifier:
            self.setTransformationAnchor(QGraphicsView.NoAnchor)
            adjust = event.angleDelta().x()/120 * 50
            self.translate(adjust,0)

        else:
            QGraphicsView.wheelEvent(self, event)

class TrackItem(QGraphicsScene):
    # add more functionality for setting event type, i.e., parent-child, migrate, death, leave frame, etc..

    def __init__(self, specs, channel):
        super(TrackItem, self).__init__()

        self.specs = specs
        self.channel = channel
        # add QImages to scene
        self.fov_id_list = [fov_id for fov_id in self.specs.keys()]

        self.fovIndex = 0
        self.fov_id = self.fov_id_list[self.fovIndex]

        self.peak_id_list_in_fov = [peak_id for peak_id in self.specs[self.fov_id].keys() if self.specs[self.fov_id][peak_id] == 1]

        self.peakIndex = 0
        self.peak_id = self.peak_id_list_in_fov[self.peakIndex]
        print(self.peak_id)
        print(self.fov_id)

        # construct image stack file names from params
        self.phaseImgPath = os.path.join(params['chnl_dir'], "{}_xy{:0=3}_p{:0=4}_c{}.tif".format(params['experiment_name'], self.fov_id, self.peak_id, self.channel))
        self.labelImgPath = os.path.join(params['foci_seg_dir'], "{}_xy{:0=3}_p{:0=4}_foci_seg_unet.tif".format(params['experiment_name'], self.fov_id, self.peak_id))
        self.cellLabelImgPath = os.path.join(params['seg_dir'], "{}_xy{:0=3}_p{:0=4}_seg_unet.tif".format(params['experiment_name'], self.fov_id, self.peak_id))

        self.labelStack = io.imread(self.labelImgPath)
        self.phaseStack = io.imread(self.phaseImgPath)
        self.cellLabelStack = io.imread(self.cellLabelImgPath)

        # read in cell tracking info
        cell_filename_all = os.path.join(params['cell_dir'], 'all_cells.pkl')
        with open(cell_filename_all, 'rb') as cell_file:
            self.All_Cells = pickle.load(cell_file)

        self.fov_Cells = mm3.filter_cells(self.All_Cells, attr="fov", val=self.fov_id)
        self.these_Cells = mm3.filter_cells(self.fov_Cells, attr="peak", val=self.peak_id)

        # read in focus tracking info
        focus_filename_all = os.path.join(params['foci_track_dir'], 'all_foci.pkl')
        with open(focus_filename_all, 'rb') as focus_file:
            self.All_Foci = pickle.load(focus_file)

        self.fov_Foci = mm3.filter_cells(self.All_Foci, attr="fov", val=self.fov_id)
        self.these_Foci = mm3.filter_cells(self.fov_Foci, attr="fov", val=self.peak_id)

        plot_dir = os.path.join(params['foci_track_dir'], 'plots')
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        lin_dir = os.path.join(plot_dir, 'lineage_plots_full')
        if not os.path.exists(lin_dir):
            os.makedirs(lin_dir)

        # look for previously edited info and load it if it is found
        self.get_track_pickle()
        # self.no_track_pickle_lookup()

        self.all_frames_by_time_dict = self.all_phase_img_and_regions()

        self.drawing = False
        self.remove_events = False
        self.brushSize = 2
        self.brushColor = QColor('black')
        self.lastPoint = QPoint()
        self.pen = QPen()

        # class options
        self.event_types_list = ["ChildLine",
                                "MigrationLine",
                                "AppearSymbol",
                                "DisappearSymbol",
                                "JoinLine"]
        self.line_events_list = ["ChildLine","MigrationLine","JoinLine"]
        self.end_check_events_list = ["AppearSymbol","ZeroFocusSymbol"]
        # the below lookup table may need reworked to handle migration and child lines to/from a focus
        #   or the handling may be better done in the update_focus_info function
        self.event_type_index_lookup = {"MigrationLine":0,
                                        "ChildLine":1,        
                                        "AppearSymbol":2,
                                        "DisappearSymbol":3,
                                        "JoinLine":4,
                                        "NoData":5
                                        }
        # given an event type for a focus, what are the event types that are incompatible within that same focus?
        self.forbidden_events_lookup = {"MigrationStart":["ChildStart","DisappearSymbol","DieSymbol","MigrationStart","JoinStart","ZeroFocusSymbol"],
                                           "MigrationEnd":["ChildEnd","AppearSymbol","BornSymbol","MigrationEnd","JoinEnd","ZeroFocusSymbol"],
                                           "ChildStart":["MigrationStart","DisappearSymbol","DieSymbol","JoinStart","ZeroFocusSymbol"],
                                           "ChildEnd":["MigrationEnd","AppearSymbol","ChildEnd","JoinEnd","ZeroFocusSymbol"],
                                           "BornSymbol":["MigrationEnd","AppearSymbol","BornSymbol","JoinEnd","ZeroFocusSymbol"],
                                           "AppearSymbol":["MigrationEnd","ChildEnd","BornSymbol","AppearSymbol","JoinEnd","ZeroFocusSymbol"],
                                           "DisappearSymbol":["MigrationStart","ChildStart","DieSymbol","DisappearSymbol","JoinStart","ZeroFocusSymbol"],
                                           "DieSymbol":["MigrationStart","ChildStart","DisappearSymbol","DieSymbol","JoinStart","ZeroFocusSymbol"],
                                           "JoinStart":["MigrationStart","ChildStart","DisappearSymbol","DieSymbol","JoinStart","ZeroFocusSymbol"],
                                           "JoinEnd":["ChildEnd","AppearSymbol","BornSymbol","MigrationEnd","ZeroFocusSymbol"],
                                           "ZeroFocusSymbol":["OneFocusSymbol","TwoFocusSymbol","ThreeFocusSymbol","MigrationStart",
                                                             "MigrationEnd","ChildStart","ChildEnd","BornSymbol",
                                                             "AppearSymbol","DisappearSymbol","DieSymbol","JoinStart","JoinEnd"],
                                           "OneFocusSymbol":["ZeroFocusSymbol","TwoFocusSymbol","ThreeFocusSymbol"],
                                           "TwoFocusSymbol":["ZeroFocusSymbol","OneFocusSymbol","ThreeFocusSymbol"],
                                           "ThreeFocusSymbol":["ZeroFocusSymbol","OneFocusSymbol","TwoFocusSymbol"]}
        self.migration = False
        self.children = False
        self.appear = False
        self.disappear = False
        self.join = False

        # apply focus events to the scene
        self.draw_focus_events()

    def save_updates(self):

        track_info = self.track_info

        for t, time_info in track_info.items():
            for cell_id, cell_info in time_info.items():
                for region_label, region in cell_info['regions'].items():

                    if 'region_graphic' in region:
                        if 'pen' in region['region_graphic']:
                            region['region_graphic'].pop('pen')
                        if 'brush' in region['region_graphic']:
                            region['region_graphic'].pop('brush')

        with open(self.pickle_file_name, 'wb') as track_file:
            try:
                pickle.dump(track_info, track_file)
                track_file.close()
                print("Saved updated tracking information to {}.".format(self.pickle_file_name))
            except Exception as e:
                track_file.close()
                print(str(e))

        df_file_name = '/home/wanglab/src/mm3/focus_track_training_file_paths.csv'
        # df_file_name = '/Users/jt/code/mm3/focus_track_training_file_paths.csv'

        if os.path.isfile(df_file_name):

            track_file_name_df = pd.read_csv(df_file_name)

            # if the current training data isn't yet in the dataframe, add it and save the updated dataframe
            if not self.pickle_file_name in track_file_name_df.file_path.values:

                print("Appending file name {} as new row in {}.".format(self.pickle_file_name, df_file_name))
                track_file_name_df = track_file_name_df.append({'file_path':self.pickle_file_name, 'include':1}, ignore_index=True)
                track_file_name_df.to_csv(df_file_name,index=False)

        else:
            track_file_name_df = pd.DataFrame(data={
                                                    'file_path':[self.pickle_file_name],
                                                    'include':[1]
                                                    })
            track_file_name_df.to_csv(df_file_name,index=False)

    def no_track_pickle_lookup(self):
        self.track_info = self.create_tracking_information(self.fov_id, self.peak_id, self.labelStack, self.phaseStack, self.cellLabelStack, self.these_Cells)

    def get_track_pickle(self):

        self.pickle_file_name = os.path.join(params['foci_track_dir'], '{}_xy{:0=3}_p{:0=4}_updated_tracks.pkl'.format(params['experiment_name'], self.fov_id, self.peak_id))
        # look for previously updated tracking information and load that if it is found.
        if not os.path.isfile(self.pickle_file_name):
            # get tracking information in a format usable and updatable by qgraphicsscene
            self.track_info = self.create_tracking_information(self.fov_id, self.peak_id, self.labelStack, self.phaseStack, self.cellLabelStack, self.these_Cells)
        else:
            with open(self.pickle_file_name, 'rb') as pickle_file:
                try:
                    print("Found updated track information in {}. Uploading and plotting it.".format(self.pickle_file_name))
                    self.track_info = pickle.load(pickle_file)
                except Exception as e:
                    print("Could not load pickle file specified.")
                    print(e)

    def go_to_fov_and_peak_id(self, fov_id, peak_id):
        # start by removing all current graphics items from the scene, the scene here being 'self'
        self.clear()

        self.fov_id = fov_id
        self.fovIndex = self.fov_id_list.index(self.fov_id)

        self.peak_id_list_in_fov = [peak_id for peak_id in self.specs[self.fov_id].keys() if self.specs[self.fov_id][peak_id] == 1]

        self.peak_id = peak_id
        print(self.peak_id)
        # Now we'll look up the peakIndex
        self.peakIndex = self.peak_id_list_in_fov.index(self.peak_id)

        # construct image stack file names from params
        self.phaseImgPath = os.path.join(params['chnl_dir'], "{}_xy{:0=3}_p{:0=4}_c{}.tif".format(params['experiment_name'], self.fov_id, self.peak_id, self.channel))
        self.labelImgPath = os.path.join(params['foci_seg_dir'], "{}_xy{:0=3}_p{:0=4}_foci_seg_unet.tif".format(params['experiment_name'], self.fov_id, self.peak_id))
        self.cellLabelImgPath = os.path.join(params['seg_dir'], "{}_xy{:0=3}_p{:0=4}_seg_unet.tif".format(params['experiment_name'], self.fov_id, self.peak_id))

        self.labelStack = io.imread(self.labelImgPath)
        self.phaseStack = io.imread(self.phaseImgPath)
        self.cellLabelStack = io.imread(self.cellLabelImgPath)

        self.fov_Cells = mm3.filter_cells(self.All_Cells, attr="fov", val=self.fov_id)
        self.these_Cells = mm3.filter_cells(self.fov_Cells, attr="peak", val=self.peak_id)

        # look for previously edited info and load it if it is found
        self.get_track_pickle()

        self.all_frames_by_time_dict = self.all_phase_img_and_regions()

        self.draw_focus_events()

    def next_peak(self):
        # start by removing all current graphics items from the scene, the scene here being 'self'
        self.clear()

        # self.center_frame_index = 1
        self.peakIndex += 1
        self.peak_id = self.peak_id_list_in_fov[self.peakIndex]
        # print(self.peak_id)

        # construct image stack file names from params
        self.phaseImgPath = os.path.join(params['chnl_dir'], "{}_xy{:0=3}_p{:0=4}_c{}.tif".format(params['experiment_name'], self.fov_id, self.peak_id, self.channel))
        self.labelImgPath = os.path.join(params['foci_seg_dir'], "{}_xy{:0=3}_p{:0=4}_foci_seg_unet.tif".format(params['experiment_name'], self.fov_id, self.peak_id))
        self.cellLabelImgPath = os.path.join(params['seg_dir'], "{}_xy{:0=3}_p{:0=4}_seg_unet.tif".format(params['experiment_name'], self.fov_id, self.peak_id))

        self.labelStack = io.imread(self.labelImgPath)
        self.phaseStack = io.imread(self.phaseImgPath)
        self.cellLabelStack = io.imread(self.cellLabelImgPath)

        self.these_Cells = mm3.filter_cells(self.fov_Cells, attr="peak", val=self.peak_id)

        # look for previously edited info and load it if it is found
        self.get_track_pickle()

        self.all_frames_by_time_dict = self.all_phase_img_and_regions()

        self.draw_focus_events()

    def prior_peak(self):
        # start by removing all current graphics items from the scene, the scene here being 'self'
        self.clear()

        # self.center_frame_index = 1
        self.peakIndex -= 1
        self.peak_id = self.peak_id_list_in_fov[self.peakIndex]
        # print(self.peak_id)

        # construct image stack file names from params
        self.phaseImgPath = os.path.join(params['chnl_dir'], "{}_xy{:0=3}_p{:0=4}_c{}.tif".format(params['experiment_name'], self.fov_id, self.peak_id, self.channel))
        self.labelImgPath = os.path.join(params['foci_seg_dir'], "{}_xy{:0=3}_p{:0=4}_foci_seg_unet.tif".format(params['experiment_name'], self.fov_id, self.peak_id))
        self.cellLabelImgPath = os.path.join(params['seg_dir'], "{}_xy{:0=3}_p{:0=4}_seg_unet.tif".format(params['experiment_name'], self.fov_id, self.peak_id))

        self.labelStack = io.imread(self.labelImgPath)
        self.phaseStack = io.imread(self.phaseImgPath)
        self.cellLabelStack = io.imread(self.cellLabelImgPath)

        self.these_Cells = mm3.filter_cells(self.fov_Cells, attr="peak", val=self.peak_id)

        # look for previously edited info and load it if it is found
        self.get_track_pickle()

        self.all_frames_by_time_dict = self.all_phase_img_and_regions()

        self.draw_focus_events()

    def next_fov(self):
        # start by removing all current graphics items from the scene, the scene here being 'self'
        self.clear()

        # self.center_frame_index = 1

        self.fovIndex += 1
        self.fov_id = self.fov_id_list[self.fovIndex]

        self.peak_id_list_in_fov = [peak_id for peak_id in self.specs[self.fov_id].keys() if self.specs[self.fov_id][peak_id] == 1]

        self.peakIndex = 0
        self.peak_id = self.peak_id_list_in_fov[self.peakIndex]

        # construct image stack file names from params
        self.phaseImgPath = os.path.join(params['chnl_dir'], "{}_xy{:0=3}_p{:0=4}_c{}.tif".format(params['experiment_name'], self.fov_id, self.peak_id, self.channel))
        self.labelImgPath = os.path.join(params['foci_seg_dir'], "{}_xy{:0=3}_p{:0=4}_foci_seg_unet.tif".format(params['experiment_name'], self.fov_id, self.peak_id))
        self.cellLabelImgPath = os.path.join(params['seg_dir'], "{}_xy{:0=3}_p{:0=4}_seg_unet.tif".format(params['experiment_name'], self.fov_id, self.peak_id))

        self.labelStack = io.imread(self.labelImgPath)
        self.phaseStack = io.imread(self.phaseImgPath)
        self.cellLabelStack = io.imread(self.cellLabelImgPath)

        self.fov_Cells = mm3.filter_cells(self.All_Cells, attr="fov", val=self.fov_id)
        self.these_Cells = mm3.filter_cells(self.fov_Cells, attr="peak", val=self.peak_id)

        # look for previously edited info and load it if it is found
        self.get_track_pickle()

        self.all_frames_by_time_dict = self.all_phase_img_and_regions()
        self.draw_focus_events()

    def prior_fov(self):
        # start by removing all current graphics items from the scene, the scene here being 'self'
        self.clear()

        # self.center_frame_index = 1

        self.fovIndex -= 1
        self.fov_id = self.fov_id_list[self.fovIndex]

        self.peak_id_list_in_fov = [peak_id for peak_id in self.specs[self.fov_id].keys() if self.specs[self.fov_id][peak_id] == 1]

        self.peakIndex = 0
        self.peak_id = self.peak_id_list_in_fov[self.peakIndex]

        # construct image stack file names from params
        self.phaseImgPath = os.path.join(params['chnl_dir'], "{}_xy{:0=3}_p{:0=4}_c{}.tif".format(params['experiment_name'], self.fov_id, self.peak_id, self.channel))
        self.labelImgPath = os.path.join(params['foci_seg_dir'], "{}_xy{:0=3}_p{:0=4}_foci_seg_unet.tif".format(params['experiment_name'], self.fov_id, self.peak_id))
        self.cellLabelImgPath = os.path.join(params['seg_dir'], "{}_xy{:0=3}_p{:0=4}_seg_unet.tif".format(params['experiment_name'], self.fov_id, self.peak_id))

        self.labelStack = io.imread(self.labelImgPath)
        self.phaseStack = io.imread(self.phaseImgPath)
        self.cellLabelStack = io.imread(self.cellLabelImgPath)

        self.fov_Cells = mm3.filter_cells(self.All_Cells, attr="fov", val=self.fov_id)
        self.these_Cells = mm3.filter_cells(self.fov_Cells, attr="peak", val=self.peak_id)

        # look for previously edited info and load it if it is found
        self.get_track_pickle()

        self.all_frames_by_time_dict = self.all_phase_img_and_regions()
        self.draw_focus_events()

    def all_phase_img_and_regions(self):

        frame_dict_by_time = {}

        xPos = 0
        for time in self.track_info.keys():
            frame_index = time-1
            frame, regions = self.phase_img_and_regions(frame_index)
            frame_dict_by_time[time] = self.addPixmap(frame)
            frame_dict_by_time[time].time = time
            self.add_regions_to_frame(regions, frame_dict_by_time[time])
            frame_dict_by_time[time].setPos(xPos, 0)
            xPos += frame_dict_by_time[time].pixmap().width()

        return(frame_dict_by_time)

    def phase_img_and_regions(self, frame_index, watershed=False):

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

        # create transparent rbg overlay to grab colors from for drawing focus regions as QGraphicsPathItems
        RGBLabelImg = color.label2rgb(maskImg, bg_label=0)
        RGBLabelImg = (RGBLabelImg*255).astype('uint8')
        originalHeight, originalWidth, RGBLabelChannelNumber = RGBLabelImg.shape
        RGBLabelImg = QImage(RGBLabelImg, originalWidth, originalHeight, RGBLabelImg.strides[0], QImage.Format_RGB888)#.scaled(512, 512, aspectRatioMode=Qt.KeepAspectRatio)
        
        time_info = self.track_info[time]
        time_info['time'] = time

        for cell_info in time_info.values():
            regions = cell_info['regions']
            for region in regions.values():
                brush = QBrush()
                brush.setStyle(Qt.SolidPattern)
                pen = QPen()
                pen.setStyle(Qt.SolidLine)
                props = region['props']
                coords = props.coords
                min_row, min_col, max_row, max_col = props.bbox
                label = props.label
                centroidY,centroidX = props.centroid
                brushColor = RGBLabelImg.pixelColor(centroidX,centroidY)
                brushColor.setAlphaF(0.15)
                brush.setColor(brushColor)
                pen.setColor(brushColor)

                region['region_graphic'] = {'top_y':min_row, 'bottom_y':max_row,
                                            'left_x':min_col, 'right_x':max_col,
                                            'coords':coords,
                                            'pen':pen, 'brush':brush}

        return(phaseQpixmap, time_info)

    # def get_foci_regions_by_time(self, focus_label_stack):
    #     '''This function provides support to detect to which cell each focus belongs, and to
    #     add additional information to each focus' regionprops based on that focus' position
    #     within its cell.
    #     '''

    #     t_adj = 1
    #     foci_by_time = {frame+t_adj: {} for frame in range(focus_label_stack.shape[0])}
    #     cell_id_list = [k for k in self.these_Cells.keys()]

    #     # get to first timepoint containing cells
    #     for t in foci_by_time.keys():

    #         t_foci = mm3.filter_cells_containing_val_in_attr(self.these_Cells, attr='times', val=t)
    #         if t_cells:
    #             break





    #     skipped_cells = []
    #     # loop over cells, begining with cells at current time
    #     for cell_id,cell in t_cells.items():

    #         foci_in_cell = cell.foci

    #         for t in cell.times:
                
    #             cell_idx = cell.times.index(t)
    #             foci_in_cell = cell.foci

    #             these_foci = {focus_id:focus for focus_id,focus in foci_in_cell.items() if t in focus.times}


    #         t_foci = mm3.filter_cells_containing_val_in_attr(self.these_Foci, attr='times', val=t)
            
    #         if len(t_foci) > 0:
                
    #             for focus_id,focus in t_foci.items():

    #                 focus_idx = focus.times.index(t)
    #                 # loop over cells to which this focus belongs
    #                 for cell in focus.cells:
    #                     cell_times = cell.times
    #                     # if the current time is in this cell grab the index
    #                     #  for this cell and its id
    #                     if t in cell_times:
    #                         cell_idx = cell.times.index(t)
    #                         cell_id = cell.id
    #                         break
                    
    #                 cell_y,cell_x = cell.centroids[cell_idx]
    #                 focus_y,focus_x = focus.regions[focus_idx].centroid
                    
    #                 focus.regions[focus_idx].cell_y_pos = cell_y-focus_y
    #                 focus.regions[focus_idx].cell_x_pos = cell_x-focus_x
    #                 focus.daughter_cell_ids = []
    #                 if cell.daughters:
    #                     for daughter_cell in cell.daughters:
    #                         focus.daughter_cell_ids.append(daughter_cell.id)
    #                 cell_data[t].append(focus)

    #     return(foci_by_cell_and_time)


    def create_tracking_information(self, fov_id, peak_id, label_stack, img_stack, cell_label_stack, Cells):

        t_adj = 1
        foci_tmp = mm3_plots.find_cells_of_fov_and_peak(self.All_Foci, fov_id, peak_id)
        print('There are {} foci for this trap'.format(len(foci_tmp)))
        
        regions_and_events_by_time = {frame+t_adj: {} for frame in range(label_stack.shape[0])}
        # iterate through all times, collecting focus information and adding
        # cell-centric tracking info to the dictionary as we go
        for t,t_info in regions_and_events_by_time.items():

            # keep only foci with this time in their 'times' attribute
            t_foci = mm3.filter_cells_containing_val_in_attr(foci_tmp, attr='times', val=t)
            print('\n',t)

            for focus_id,focus in t_foci.items():

                focus_idx = focus.times.index(t)
                focus_cell_labels = focus.cell_labels
                focus_cell_label = focus_cell_labels[focus_idx]
                focus_region = focus.regions[focus_idx]
                
                for cell in focus.cells:
                    cell_times = cell.times
                    # if the current time is in this cell grab the index
                    #  for this cell and its id, then break the loop
                    if t in cell_times:
                        cell_idx = cell.times.index(t)
                        cell_id = cell.id
                        break

                # print('Focus {} is in cell {}.'.format(focus_id, cell_id))
                # print([c.id for c in focus.cells])
                # instantiate the regions and matrix info dictionary.
                #  Will add regionprops and events in the 'regions' dict later.
                if not cell_id in t_info:
                    t_info[cell_id] = {'regions' : {}, 'matrix' : None}

                    cell_info = t_info[cell_id]

                    # We need to build the interaction matrix for each cell.
                    # First we need to know the number of regions in cell at time t
                    #  so that we can instantiate an interaction matrix with correct 
                    #  firt dimension
                    # Loop through cell's foci, filtering them by time
                    n_regions_in_t = 0
                    # print(cell.foci)
##########################################################################################
# #######################################################################################
# somehow I'm occasionally dropping a focus!!!!!!!!!!!!!!!!
# I think the issue must be in mm3_helpers.py, lines 6853-8127                    
# maybe if a focus is small its overlap with pior foci is small compared to larger
# foci that are also present and overlapping. In this case, maybe my code just drops
#  the small focus.
                    for cell_focus in cell.foci.values():
                        print(cell_focus.id)
                        print(cell_focus.times)
                        if t in cell_focus.times:
                            n_regions_in_t += 1
                            print('n_regions_in_t: ', n_regions_in_t)

                    # check next timepoint in cell to get second dimension of matrix.
                    n_regions_in_t_plus_1 = 0
                    # if the cell has another time beyond this one, do the following

                    # print('current time: ', t)
                    # print('cell times: ', cell.times)
                    # print('cell daughters: ', cell.daughters)
                    if t+1 in cell.times:
                        # loop through foci in the cell
                        for cell_focus in cell.foci.values():
                            print(cell_focus.times)
                            # if the focus is present in the cell's next time,
                            # add one to n_regions_in_t_plus_1
                            if t+1 in cell_focus.times:
                                n_regions_in_t_plus_1 += 1
                                print('focus in next t: ', n_regions_in_t_plus_1)
                    # if the next timpoint doesn't exist in the cell, 
                    # the cell could have divided.
                    # check then if the cell divided and add daughter cell
                    # foci to second dimension of matrix 
                    elif cell.daughters is not None:
                        for daughter_cell in cell.daughters:
                            for cell_focus in daughter_cell.foci.values():
                                if t+1 in cell_focus.times:
                                    n_regions_in_t_plus_1 += 1
                                    print('focus in daughter, next t: ', n_regions_in_t_plus_1)

###################################################################################################################
###################################################################################################################
################################### I need to update to count correctly ###########################################
###################################################################################################################
###################################################################################################################

                    cell_info['matrix'] = np.zeros(
                        (n_regions_in_t+1, n_regions_in_t_plus_1+1),
                        dtype=np.int
                    )

                    print(cell_info['matrix'])
                    # print(t)

                # create default events list and add to the information for this
                #   cell/focus combination
                default_events = np.zeros(6, dtype=np.int)
                default_events[5] = 1 # set N to 1
                # add information at 'focus_cell_label'
                cell_info['regions'][focus_cell_label] = {
                    'props' : focus_region,
                    'events' : default_events,
                }

                # Check for when focus has less time points than it should
                # if (focus.times[-1] - focus.times[0])+1 > len(focus.times):
                #     print('Focus {} has less time points than it should, skipping.'.format(focus_id))
                #     continue
                
                # Now we can populate the cell_info matrix with the focus' interaction info
                #  We can also update the focus' events list

                # M migration, event 0
                # If the focus has another time point after this one then it must have migrated
                max_focus_time = np.max(focus.times)
                # print('focus {} has max time {}.'.format(focus_id,max_focus_time))
                if t < max_focus_time:
                    cell_info['regions'][focus_cell_label]['events'][0] = 1

                    # update matrix using this region label and the next one
                    print(focus_cell_label, focus_cell_labels[focus_idx+1], cell_info['matrix'])
                    cell_info['matrix'][focus_cell_label, focus_cell_labels[focus_idx+1]] = 1

                # S division, 1
                if focus.daughters and t == max_focus_time:
                    cell_info['regions'][focus_cell_label]['events'][1] = 1

                    # daughter 1 and 2 label
                    d1_label = self.All_Foci[focus.daughters[0]].cell_labels[0]

                    try:
                        d2_label = self.All_Foci[focus.daughters[1]].cell_labels[0]
                        cell_info['matrix'][focus_cell_label, d1_label] = 1
                        cell_info['matrix'][focus_cell_label, d2_label] = 1

                    except IndexError as e:
                        print("At timepoint {} there was an index error in assigning daughters: {}".format(t,e))

                # I appears, 2
                if not t == 1:
                    if not focus.parent and focus_idx == 0:
                        cell_info['regions'][focus_cell_label]['events'][2] = 1

                # O disappears, 3
                if not focus.daughters and t == max_focus_time:
                    cell_info['regions'][focus_cell_label]['events'][3] = 1
                    cell_info['matrix'][focus_cell_label, 0] = 1

                # N no data, 4 - Set this to zero as this region as been checked.
                cell_info['regions'][focus_cell_label]['events'][5] = 0

        # If any focus has still not been visited,
        #  make their appropriate matrix value 1, which should be in the first column.
        for t,t_data in regions_and_events_by_time.items():
            for cell_id,cell_data in t_data.items():
                for region,region_data in cell_data['regions'].items():
                    if region_data['events'][5] == 1:
                        cell_data['matrix'][region, 0] = 1

            # if not cell_id in regions_and_events_by_cell_and_time:
            #     regions_and_events_by_cell_and_time[cell_id] = {}
            
            # cell_data = regions_and_events_by_cell_and_time[cell_id]
            
            # for t,t_region_list in this_cell_time_info.items():

            #     if not t in cell_data:
            #         cell_data[t] = {'regions' : {}, 'matrix' : None}

            #     for i,region in enumerate(t_region_list):
                
            #         default_events = np.zeros(6, dtype=np.int)
            #         default_events[5] = 1 # set N to 1
            #         cell_data[t]['regions'][i+1] = {
            #             'props' : region,
            #             'events' : default_events,
            #         }
                
        # create default interaction matrix
        # Now that we know how many regions there are per time point, we will create a default matrix which indicates how regions are connected to each between this time point, t, and the next one, t+1. The row index will be the region label of the region in t, and the column index will be the region label of the region in t+1.
        # If a region migrates from t to t+1, its row should have a sum of 1 corresponding from which region (row) to which region (column) it moved. If the region divided, then both of the daughter columns will get value 1.
        # Note that the regions are labeled from 1, but Numpy arrays and indexed from zero. We can use this to store some additional informaiton. If a region disappears, it will receive a 1 in the column with index 0.
        # In the last time point all regions will be connected to the disappear column
        # for cell_id, this_cell_time_info in regions_and_events_by_cell_and_time.items():
        #     for t,t_data in this_cell_time_info.items():
        #         n_regions_in_t = len(t_data['regions'])
        #         if t+1 in this_cell_time_info:
        #             n_regions_in_t_plus_1 = len(this_cell_time_info[t+1]['regions'])
        #         else:
        #             n_regions_in_t_plus_1 = 0

        #         t_data['matrix'] = np.zeros((n_regions_in_t+1, n_regions_in_t_plus_1+1), dtype=np.int)
        

        # regions_and_events_by_time = {frame+t_adj : {'regions' : {}, 'matrix' : None} for frame in range(label_stack.shape[0])}

        # # loop through regions and add them to the main dictionary.
        # for t, regions in regions_by_time.items():
        #     # this is a list, while we want it to be a dictionary with the region label as the key
        #     for region in regions:
        #         default_events = np.zeros(6, dtype=np.int)
        #         default_events[5] = 1 # set N to 1
        #         regions_and_events_by_time[t]['regions'][region.label] = {'props' : region,
        #                                                                 'events' : default_events}
        # create default interaction matrix
        # Now that we know how many regions there are per time point, we will create a default matrix which indicates how regions are connected to each between this time point, t, and the next one, t+1. The row index will be the region label of the region in t, and the column index will be the region label of the region in t+1.
        # If a region migrates from t to t+1, its row should have a sum of 1 corresponding from which region (row) to which region (column) it moved. If the region divided, then both of the daughter columns will get value 1.
        # Note that the regions are labeled from 1, but Numpy arrays and indexed from zero. We can use this to store some additional informaiton. If a region disappears, it will receive a 1 in the column with index 0.
        # In the last time point all regions will be connected to the disappear column
        # for t, t_data in regions_and_events_by_time.items():
        #     n_regions_in_t = len(regions_by_time[t])
        #     if t+1 in regions_by_time:
        #         n_regions_in_t_plus_1 = len(regions_by_time[t+1])
        #     else:
        #         n_regions_in_t_plus_1 = 0

        #     t_data['matrix'] = np.zeros((n_regions_in_t+1, n_regions_in_t_plus_1+1), dtype=np.int)

        # Loop over foci and edit event information
        # We will use the focus dictionary All_Foci.
        # Each focus object has a number of attributes that are useful to us. 
        # We will go through each focus by its time points and edit the events associated with that region.
        # We will also edit the matrix when appropriate.
        # pull out only the foci in of this FOV

        # for focus_id,focus_tmp in foci_tmp.items():

        #     cell_id = focus_id_cell_id_lut[focus_id]
        #     cell_data = regions_and_events_by_cell_and_time[cell_id]
        #     # Check for when focus has less time points than it should
        #     # if (focus_tmp.times[-1] - focus_tmp.times[0])+1 > len(focus_tmp.times):
        #     #     print('Focus {} has less time points than it should, skipping.'.format(focus_id))
        #     #     continue
        #     unique_times = list(np.unique(focus_tmp.times))
        #     focus_tmp_labels = [focus_tmp.cell_labels[focus_tmp.times.index(t)] for t in unique_times]

        #     # Go over the time points of this focus and edit appropriate information main dictionary
        #     for i, t in enumerate(unique_times):

        #         # get the region label
        #         label_tmp = focus_tmp_labels[i]

        #         # M migration, event 0
        #         # If the focus has another time point after this one then it must have migrated
        #         if i != len(unique_times)-1:
        #             cell_data[t]['regions'][label_tmp]['events'][0] = 1
        #             # regions_and_events_by_time[t]['regions'][label_tmp]['events'][0] = 1

        #             # update matrix using this region label and the next one
        #             # print(label_tmp, focus_tmp_labels[i+1], regions_and_events_by_time[t]['matrix'])
        #             cell_data[t]['matrix'][label_tmp, focus_tmp_labels[i+1]] = 1
        #             # regions_and_events_by_time[t]['matrix'][label_tmp, focus_tmp_labels[i+1]] = 1

        #         # S division, 1
        #         if focus_tmp.daughters and i == len(unique_times)-1:
        #             cell_data[t]['regions'][label_tmp]['events'][1] = 1
        #             # regions_and_events_by_time[t]['regions'][label_tmp]['events'][1] = 1

        #             # daughter 1 and 2 label
        #             # d1_label = self.All_Foci[focus_tmp.daughters[0].id].labels[0]
        #             d1_label = self.All_Foci[focus_tmp.daughters[0]].labels[0]

        #             try:
        #                 # d2_label = self.All_Foci[focus_tmp.daughters[1].id].labels[0]
        #                 d2_label = self.All_Foci[focus_tmp.daughters[1]].labels[0]
        #                 cell_data[t]['matrix'][label_tmp, d1_label] = 1
        #                 cell_data[t]['matrix'][label_tmp, d2_label] = 1
        #                 # regions_and_events_by_time[t]['matrix'][label_tmp, d1_label] = 1
        #                 # regions_and_events_by_time[t]['matrix'][label_tmp, d2_label] = 1

        #             except IndexError as e:
        #                 print("At timepoint {} there was an index error in assigning daughters: {}".format(t,e))

        #         # I appears, 2
        #         if not t == 1:
        #             if not focus_tmp.parent and i == 0:
        #                 cell_data[t]['regions'][label_tmp]['events'][2] = 1
        #                 # regions_and_events_by_time[t]['regions'][label_tmp]['events'][2] = 1

        #         # O disappears, 3
        #         if not focus_tmp.daughters and i == len(unique_times)-1:
        #             cell_data[t]['regions'][label_tmp]['events'][3] = 1
        #             cell_data[t]['matrix'][label_tmp, 0] = 1
        #             # regions_and_events_by_time[t]['regions'][label_tmp]['events'][3] = 1
        #             # regions_and_events_by_time[t]['matrix'][label_tmp, 0] = 1

        #         # N no data, 4 - Set this to zero as this region as been checked.
        #         cell_data[t]['regions'][label_tmp]['events'][5] = 0
        #         # regions_and_events_by_time[t]['regions'][label_tmp]['events'][5] = 0

        # # Set remaining regions to event space [0 0 0 0 1 1]
        # # Also make their appropriate matrix value 1, which should be in the first column.
        # for cell_id,cell_data in regions_and_events_by_cell_and_time.items():
        #     for t, t_data in cell_data.items():
        #         for region, region_data in t_data['regions'].items():
        #             if region_data['events'][5] == 1:

        #                 t_data['matrix'][region, 0] = 1

        # for t, t_data in regions_and_events_by_time.items():
        #     for region, region_data in t_data['regions'].items():
        #         if region_data['events'][5] == 1:

        #             t_data['matrix'][region, 0] = 1

        return(regions_and_events_by_cell_and_time)

    def add_regions_to_frame(self, regions_and_events, frame):
        # loop through foci within this frame and add their ellipses as children of their corresponding qpixmap object
        regions = regions_and_events['regions']
        # print(regions_and_events)
        frame_time = regions_and_events['time']
        for region_id,region in regions.items():
            # construct the ellipse
            graphic = region['region_graphic']
            top_left = QPoint(graphic['left_x'],graphic['top_y'])
            bottom_right = QPoint(graphic['right_x'],graphic['bottom_y'])
            rect = QRectF(top_left,bottom_right)

            # painter_path = graphic['path']
            pen = graphic['pen']
            brush = graphic['brush']

            # focus = FocusItem(region, parent=frame)

            # instantiate a QGraphicsEllipseItem
            ellipse = QGraphicsEllipseItem(rect, frame)
            # add focus information to the QGraphicsEllipseItem
            ellipse.focusMatrix = regions_and_events['matrix']
            ellipse.focusEvents = region['events']
            ellipse.focusProps = region['props']
            print(region['props'])
            ellipse.time = frame_time
            ellipse.setBrush(brush)
            ellipse.setPen(pen)

            # # add focus information to the QGraphicsPathItem
            # path = QGraphicsPathItem(painter_path, frame)
            # path.focusMatrix = regions_and_events['matrix']
            # path.focusEvents = regions_and_events['regions'][region_id]['events']
            # path.focusProps = regions_and_events['regions'][region_id]['props']
            # path.time = frame_time
            # path.setBrush(brush)
            # path.setPen(pen)

            # set up QPainter object to actually paint the QGraphicsPathItem
            # painter = QPainter()
            # paint_device = QPaintDevice()
            # painter.begin(paint_device)
            # painter.setPen(pen)
            # painter.setBrush(brush)
            # painter.drawPath(painter_path)
            # painter.end()

            # path.paint()
            #

    def draw_focus_events(self, start_time=1, end_time=None, update=False, original_event_type=None):

        # Here is where we will draw the intial lines and symbols representing
        #   focus events and linkages between foci.
        # Set up a list of time points at which we will re-draw events
        # the ability to create this list greatkt increases the speed of re-drawing
        #   events after an update.
        if end_time is not None:
            max_time = end_time
        else: max_time = np.max([item.time for item in self.items() if item.type() == 7])

        valid_times = [i for i in range(start_time, max_time+1)]

        # track_info is a dictionary, the keys of which are 1-indexed frame numbers
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

                    for startItem in frame.childItems():
                        # test if this is an ellipse item. If it is, draw event symbols.
                        if startItem.type() == 4:
                            # remove all previously drawn events belonging entirely to,
                            #   or emanating from, this focus.
                            if update:
                                items, items_list = self.get_all_focus_event_items(startItem, return_forbidden_items_list=True)
                                for i, item_type in enumerate(items_list):
                                    if item_type not in ["ChildEnd","MigrationEnd","JoinEnd"]:
                                        self.removeItem(items[i])

                            focus_properties = startItem.focusProps
                            focus_label = focus_properties.label
                            focus_interactions = startItem.focusMatrix[focus_label,:]
                            focus_events = startItem.focusEvents
                            # print(focus_events)
                            # print(focus_interactions)
                            # get centroid of focus represented by this qgraphics item
                            firstPointY = focus_properties.centroid[0]
                            firstPointX = focus_properties.centroid[1] + startItem.parentItem().x()
                            firstPoint = QPoint(firstPointX, firstPointY)
                            # which events happened to this focus?
                            event_indices = np.where(focus_events == 1)[0]

                            if 2 in event_indices:
                                # if the second element in event_indices was 1,
                                #   the focus appeared between this frame and the previous one.
                                self.set_appear()
                                eventItem = self.set_event_item(firstPoint=firstPoint, startItem=startItem)
                                self.addItem(eventItem)

                            if 3 in event_indices:
                                # if the fifth element in event_indices was 1,
                                #   the focus focus disappeared between this frame and the previous one.
                                self.set_disappear()
                                eventItem = self.set_event_item(firstPoint=firstPoint, startItem=startItem)
                                self.addItem(eventItem)

                            try:
                                nextFrame = self.all_frames_by_time_dict[time+1]
                                for endItem in nextFrame.childItems():
                                    # if the item is an ellipse, move on to look into it further
                                    if endItem.type() == 4:
                                        end_focus_properties = endItem.focusProps
                                        end_focus_label = end_focus_properties.label
                                        # test whether this ellipse represents the
                                        #  focus that interacts with the focus represented
                                        #  by self.startItem
                                        if focus_interactions[end_focus_label] == 1:
                                            # if this is the focus that interacts with the former frame's focus, draw the line.
                                            endPointY = end_focus_properties.centroid[0]
                                            endPointX = end_focus_properties.centroid[1] + endItem.parentItem().x()
                                            lastPoint = QPoint(endPointX, endPointY)
                                            if 0 in event_indices:
                                                # If the zero-th element in event_indices was 1, the focus migrates in the next frame
                                                #  get the information from focus_matrix to figure out to which region
                                                #  in the next frame it migrated
                                                # set self.migration = True
                                                self.set_migration()

                                            if 1 in event_indices:
                                                self.set_children()

                                            if 4 in event_indices:
                                                self.set_join()

                                            eventItem = self.set_event_item(firstPoint=firstPoint, startItem=startItem, lastPoint=lastPoint, endItem=endItem)
                                            self.addItem(eventItem)
                            except KeyError:
                                continue

        if original_event_type is not None:
            if original_event_type == "MigrationLine":
                self.set_migration()
            elif original_event_type == "ChildLine":
                self.set_children()
            elif original_event_type == "AppearSymbol":
                self.set_appear()
            elif original_event_type == "DisappearSymbol":
                self.set_disappear()
            elif original_event_type == "JoinLine":
                self.set_join()
            elif original_event_type == "Removal":
                self.remove_all_focus_events()

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
        print("No focus detected underneath your selection. Ignoring selection.")
        return(None)

    def get_all_focus_event_items(self, focus, return_forbidden_items_list=False):
        # Fetch all items that collide with the clicked ellipse,
        #   evalute whether they are "events".
        # Further evaluate whether the event items' items are the same as
        #   the clicked ellipse. The reason for this is that some text items take more
        #   space than they appear to occupy in the GUI, so they may collide with
        #   ellipses neighboring the clicked ellipse

        items_colliding_with_focus = focus.collidingItems()
        event_items = []
        event_types = []

        for item in items_colliding_with_focus:
            itemType = item.type()
            if itemType in self.event_types_list:
                if itemType in self.line_events_list:
                    # determine whether the child or migration line's start or end point
                    #   belongs to the clicked focus
                    if (item.startItem == focus or item.endItem == focus):
                        # if we're calling this function with return_forbidden_items_list=True,
                        #   we need to figure out whether each migration and child line in this
                        #   focus represent the endpoint of the line or the startpoint of the line.
                        if return_forbidden_items_list:
                            # is this focus the endItem or the startItem for this item?
                            if itemType == "MigrationLine":
                                if item.startItem == focus:
                                    item_name = "MigrationStart"
                                elif item.endItem == focus:
                                    item_name = "MigrationEnd"

                            elif itemType == "ChildLine":
                                if item.startItem == focus:
                                    item_name = "ChildStart"
                                elif item.endItem == focus:
                                    item_name = "ChildEnd"

                            elif itemType == "JoinLine":
                                if item.startItem == focus:
                                    item_name = "JoinStart"
                                elif item.endItem == focus:
                                    item_name = "JoinEnd"

                            event_types.append(item_name)

                        event_items.append(item)
                else:
                    # determine whether the die, born, appear, or disappear event
                    #    belongs to the clicked focus
                    if item.item == focus:
                        if return_forbidden_items_list:
                            item_name = itemType
                            event_types.append(item_name)
                        event_items.append(item)

        if return_forbidden_items_list:
            return(event_items, event_types)

        return(event_items)

    def update_tracking_info(self):
        # This function updates the matrix and events for all foci in your scene.
        # First, get the maximum time so we don't get a key error later on
        max_time = np.max([key for key in self.all_frames_by_time_dict.keys()])
        if self.remove_events:
            focus = self.startItem
            frame = self.all_frames_by_time_dict[focus.time]
            self.update_frame_info(frame)

        # Work with the newly added eventItem
        # If the event was either child or migration, do this stuff
        elif self.eventItem.type() in self.line_events_list:

            # Identify immediately affected foci
            focus = self.eventItem.startItem
            frame = self.all_frames_by_time_dict[focus.time]
            self.update_frame_info(frame)

            focus = self.eventItem.endItem
            frame = self.all_frames_by_time_dict[focus.time]
            self.update_frame_info(frame)

        else:
            focus = self.eventItem.item
            frame = self.all_frames_by_time_dict[focus.time]
            self.update_frame_info(frame)

    def update_frame_info(self, frame):
        # This function takes a focus as input and grabs its information from the original
        #   tracking information, then updates the 'matrix' and 'events' objects for the proper frame
        #   and events.
        frame_time = frame.time
        # get all foci in the frame to update each one, in case they were indirectly affected by the event that was drawn
        for focus in frame.childItems():

            #############################################################################################
            #############################################################################################
            #############  THIS PART ALSO NEEDS REWRITTEN TO ACCOMODATE CELL-CENTRIC CHANGES!!!! ########
            #############################################################################################
            #############################################################################################
            # CHANGE LABEL TO CELL-CENTRIC LABEL!!!!!!!!!!!!
            # GET THE CELL TO WHICH THE FOCUS BELONGS!!!!!!!!!
            focus_label = focus.focusProps.label
            # print(focus_label)
            # grab all currently-drawn events for the focus of interest
            events, event_items = self.get_all_focus_event_items(focus, return_forbidden_items_list=True)
            # print("Events for focus {} at time {}: \n".format(focus_label, frame_time), event_items)
            # print([event.type() for event in events])

            #  # Fetch the focus's original information
            #    'matrix' is a 2D array, for which the row index is the region label at time t, and the column index is the region label at time t+1
            #     If a region disappears from t to t+1, it will receive a 1 in the column with index 0.
            # GO THROUGH DICTIONARY OF TRACK_INFO IN CELL_CENTRIC WAY!!!!!!

            t_info = self.track_info[frame_time]
            for cell_info in t_info.values():


                time_matrix = cell_info['matrix']
                focus_events = cell_info['regions'][focus_label]['events']

                # print("Events and matrix for focus {} at time {}: \n\n".format(focus_label, frame_time),
                #       focus_events, "\n\n", time_matrix, "\n")
                # relabel all interactions for this focus as zero to later fill in appropriate 1's based on updated events

                time_matrix[focus_label,:] = 0
                # print(focus_events) # for debugging to verify changes are being made appropriately.
                # set all focus events temporarily to zero to fill back in with appropriate 1's
                focus_events[...] = 0

                # If an event is a migration or a child line, determine whether
                #   the migration line starts at the focus to be updated, and whether
                #   the child line begins or ends at the current focus.
                #   Update events array and linkage matrix accordingly
                for event in events:
                    event_type = event.type()
                    # if the event is a migration or child line
                    if event_type in self.line_events_list:
                        # if the line starts with the focus to be updated, set the appropriate
                        #  index in focus_events to 1.

                        if event.startItem == focus:
                            focus_events[self.event_type_index_lookup[event_type]] = 1
                            start_focus_label = event.startItem.focusProps.label
                            end_focus_label = event.endItem.focusProps.label
                            # print(start_focus_label, end_focus_label)
                            time_matrix[start_focus_label,end_focus_label] = 1

                        # if event_type == "ChildLine":
                        #     # if the event is a child line that terminates with this focus,
                        #     #   set the 'born' index of focus_events to 1
                        #     if event.endItem == focus:
                        #         focus_events[self.event_type_index_lookup["BornSymbol"]] = 1

                    # # If the event is zeroFocusSymbol or Appear, do this
                    # elif event_type in self.end_check_events_list:

                    #     for i,old_event_type in enumerate(events):
                    #         print(i, old_event_type)

                    # if the event is either disappear or die, do this stuff
                    else:
                        focus_events[self.event_type_index_lookup[event_type]] = 1
                        if event_type == "DissapearSymbol":
                            focus_label = event.item.focusProps.label
                            time_matrix[focus_label,0] = 1

            # print("New events and matrix for focus {} at time {}: \n\n".format(focus_label, frame_time),
                    # focus_events, "\n\n", time_matrix, "\n")

    def remove_old_conflicting_events(self, event):
        # This function derives the foci involved in a given newly-annotated event and evaluates
        #   whether a focus has conflicting events, such as two migrations to the
        #   next frame, or three children, etc.. It then attempts to resolve the conflict by removing
        #   the older annotation.
        event_type = event.type()

        if event_type in self.line_events_list:

            start_focus = event.startItem
            end_focus = event.endItem
            old_end_focus_events, old_end_focus_event_types = self.get_all_focus_event_items(end_focus, return_forbidden_items_list=True)

            # Name the key for fobidden event lookup.
            if event_type == "MigrationLine":
                forbidden_event_lookup_key = "MigrationEnd"
            elif event_type == "ChildLine":
                forbidden_event_lookup_key = "ChildEnd"
            elif event_type == "JoinLine":
                forbidden_event_lookup_key = "JoinEnd"
            else:
                forbidden_event_lookup_key = event_type

            # retrieve list of forbidden event types, given our drawn event
            forbidden_events_list = self.forbidden_events_lookup[forbidden_event_lookup_key]

            for i, old_end_focus_event_type in enumerate(old_end_focus_event_types):
                if old_end_focus_event_type in forbidden_events_list:
                    self.removeItem(old_end_focus_events[i])

        else:
            start_focus = event.item

        old_start_focus_events, old_start_focus_event_types = self.get_all_focus_event_items(start_focus, return_forbidden_items_list=True)
        # print(old_start_focus_events, old_start_focus_event_types)

        # NOTE: removal not working as desired for zero_foci and appear. Lines into zero and appear detections are removed, then redrawn due to lack of updating background data
        # Name the key for fobidden event lookup.
        if event_type == "MigrationLine":
            forbidden_event_lookup_key = "MigrationStart"
        elif event_type == "ChildLine":
            forbidden_event_lookup_key = "ChildStart"
        elif event_type == "JoinLine":
            forbidden_event_lookup_key = "JoinStart"
        else:
            forbidden_event_lookup_key = event_type

        # retrieve list of forbidden event types, given our drawn event
        forbidden_events_list = self.forbidden_events_lookup[forbidden_event_lookup_key]
        # print(forbidden_events_list)

        child_start_count = 0
        for i, old_start_focus_event_type in enumerate(old_start_focus_event_types):
            # print(i, old_start_focus_event_type)
            if old_start_focus_event_type in forbidden_events_list:
                # print(i, "removal triggered")
                if old_start_focus_event_type == "ChildStart":
                    # if forbidden_event_lookup_key == "ChildStart":
                    #     if child_start_count == 0:
                    #         first_child_start_index = i
                    #     child_start_count += 1
                    #     if child_start_count == 2:
                    #         self.removeItem(old_start_focus_events[i])
                    #         self.removeItem(old_start_focus_events[first_child_start_index])
                    # else:
                    self.removeItem(old_start_focus_events[i])
                else:
                    # print(i, "removal of {} triggered".format(old_start_focus_events[i]))
                    self.removeItem(old_start_focus_events[i])

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:

            self.drawing = True
            self.firstPoint = event.scenePos()
            self.lastPoint = event.scenePos()
            self.startItem = self.get_ellipse(point=self.firstPoint)
            if self.startItem is not None:

                # if we want to remove all events, do it here
                if self.remove_events:
                    events = self.get_all_focus_event_items(self.startItem)
                    for event in events:
                        self.removeItem(event)
                    self.update_tracking_info()
                    # remove old events and draw the newly-updated ones. This helps you
                    #   to ensure you haven't just messed up the underlying tracking data
                    start_time = self.startItem.parentItem().time
                    end_time = self.startItem.parentItem().time
                    self.draw_focus_events(start_time=start_time-2, end_time=end_time+2, update=True, original_event_type="Removal")
                    print("Removed outgoing events from clicked focus.")
                    self.drawing = False

                # if we do not want to remove_events, we are adding an event, so do the following
                else:
                    # get the centroid position for the focus that was clicked
                    firstPointY = self.startItem.focusProps.centroid[0]
                    # here we add the x-position of the detected ellipse' frame, because
                    #   the centroid of each focus is just its centroid within its own frame
                    #   Therefore, by adding the x-offset of the frame in which the focus
                    #   exists, we shift our x-value of our line's start or end-point by the appropriate distance.
                    firstPointX = self.startItem.focusProps.centroid[1] + self.startItem.parentItem().x()
                    self.firstPoint = QPoint(firstPointX, firstPointY)
                    self.eventItem = self.set_event_item(firstPoint=self.firstPoint, startItem=self.startItem, lastPoint=self.lastPoint)
                    self.addItem(self.eventItem)

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) & self.drawing:
            if self.startItem is not None:
                self.lastPoint = event.scenePos()
                self.removeItem(self.eventItem)
                self.eventItem = self.set_event_item(firstPoint=self.firstPoint, startItem=self.startItem, lastPoint=self.lastPoint)
                self.addItem(self.eventItem)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing and not self.remove_events:

            self.lastPoint = event.scenePos()
            self.endItem = self.get_ellipse(point=self.lastPoint)
            if self.startItem is not None:
                if self.endItem is None:
                    self.removeItem(self.eventItem)

            if self.endItem is not None:
                if self.startItem is not None:
                    start_time = self.startItem.parentItem().time
                    end_time = self.endItem.parentItem().time
                    # A migration or child event cannot go from one frame to the same frame.
                    if (self.startItem.parentItem() == self.endItem.parentItem()) and self.eventItem.type() in self.line_events_list:
                        self.removeItem(self.eventItem)
                        print("Cannot link foci within a single frame as migrated or children. Ignoring selection.")

                    elif abs(start_time - end_time) > 1:

                        self.removeItem(self.eventItem)
                        # get the centroid position for the focus that was clicked
                        endPointY = self.endItem.focusProps.centroid[0]
                        # here we add the x-position of the detected ellipse' frame, because
                        #   the centroid of each focus is just its centroid within its own frame
                        #   Therefore, by adding the x-offset of the frame in which the focus
                        #   exists, we shift our x-value of our line's start or end-point by the appropriate distance.
                        endPointX = self.endItem.focusProps.centroid[1] + self.endItem.parentItem().x()
                        self.lastPoint = QPoint(endPointX, endPointY)
                        self.eventItem = self.set_event_item(firstPoint=self.firstPoint, startItem=self.startItem, lastPoint=self.lastPoint, endItem=self.endItem)

                        # get the centroid position for the focus that was clicked
                        endPointY = self.endItem.focusProps.centroid[0]
                        # here we add the x-position of the detected ellipse' frame, because
                        #   the centroid of each focus is just its centroid within its own frame
                        #   Therefore, by adding the x-offset of the frame in which the focus
                        #   exists, we shift our x-value of our line's start or end-point by the appropriate distance.
                        endPointX = self.endItem.focusProps.centroid[1] + self.endItem.parentItem().x()
                        self.lastPoint = QPoint(endPointX, endPointY)
                        self.eventItem = self.set_event_item(firstPoint=self.firstPoint, startItem=self.startItem, lastPoint=self.lastPoint, endItem=self.endItem)

                        # if it's a migration line we're drawing, it can span multiple frames.
                        #   We'll just split it up into its component migrations, frame-to-frame
                        if self.eventItem.type() == "MigrationLine":
                            # Need to get all foci with which this line collides.
                            foci_under_line = self.get_foci_under_item(item=self.eventItem)
                            # No garuantee these foci are sorted by time, sort them now
                            foci_under_line.sort(key=self.get_time)
                            # Make a list of the times to evaluate whether a timepoint is missing.
                            focus_times = [focus.time for focus in foci_under_line]
                            time_diffs = np.diff(focus_times)

                            # loop over sorted foci to add migration events to scene and background data
                            for focus_index,focus in enumerate(foci_under_line):

                                if focus_index < len(foci_under_line)-1: # do nothing if we're at the final focus

                                    time_diff = time_diffs[focus_index]
                                    # if a timepoint was missed, break the loop
                                    if time_diff > 1:
                                        # self.removeItem(self.eventItem)
                                        break

                                    self.startItem = focus
                                    self.endItem = foci_under_line[focus_index+1]

                                    # get the centroid position for the focus that was clicked
                                    endPointY = self.endItem.focusProps.centroid[0]
                                    # here we add the x-position of the detected ellipse' frame, because
                                    #   the centroid of each focus is just its centroid within its own frame
                                    #   Therefore, by adding the x-offset of the frame in which the focus
                                    #   exists, we shift our x-value of our line's start or end-point by the appropriate distance.
                                    endPointX = self.endItem.focusProps.centroid[1] + self.endItem.parentItem().x()
                                    self.lastPoint = QPoint(endPointX, endPointY)
                                    self.eventItem = self.set_event_item(firstPoint=self.firstPoint, startItem=self.startItem, lastPoint=self.lastPoint, endItem=self.endItem)

                                    self.remove_old_conflicting_events(self.eventItem)
                                    self.addItem(self.eventItem)
                                    # query the currently-drawn annotations in the scene and update all foci' information
                                    self.update_tracking_info()
                                    # remove old events and draw the newly-updated ones. This is inefficient, but helps you
                                    #   to ensure you haven't just messed up the underlying tracking data
                                    self.draw_focus_events(start_time=start_time-2, end_time=end_time+2, update=True, original_event_type=self.eventItem.type())

                        else:
                            self.removeItem(self.eventItem)
                            print("Cannot link foci separated by more than a single frame. Ignoring selection.")

                    else:
                        self.removeItem(self.eventItem)
                        # get the centroid position for the focus that was clicked
                        endPointY = self.endItem.focusProps.centroid[0]
                        # here we add the x-position of the detected ellipse' frame, because
                        #   the centroid of each focus is just its centroid within its own frame
                        #   Therefore, by adding the x-offset of the frame in which the focus
                        #   exists, we shift our x-value of our line's start or end-point by the appropriate distance.
                        endPointX = self.endItem.focusProps.centroid[1] + self.endItem.parentItem().x()
                        self.lastPoint = QPoint(endPointX, endPointY)
                        self.eventItem = self.set_event_item(firstPoint=self.firstPoint, startItem=self.startItem, lastPoint=self.lastPoint, endItem=self.endItem)

                        self.remove_old_conflicting_events(self.eventItem)
                        self.addItem(self.eventItem)
                        # query the currently-drawn annotations in the scene and update all foci' information
                        self.update_tracking_info()
                        # remove old events and draw the newly-updated ones. This is inefficient, but helps you
                        #   to ensure you haven't just messed up the underlying tracking data
                        self.draw_focus_events(start_time=start_time-2, end_time=end_time+2, update=True, original_event_type=self.eventItem.type())

            self.drawing = False

    def get_foci_under_item(self, item):
        '''
        A function which returns all focus objects underneath a QGraphicsItem.
        '''

        # get all colliding items, this will include QPixmapItems and QEllipseItems
        collisions = self.collidingItems(item)
        foci = []
        # evaluate whether each item is an ellipse (focus)
        for item in collisions:
            item_type = item.type()

            if item_type == 4:
                foci.append(item)

        return(foci)

    def get_time(self, focus):
        return(focus.time)

    def set_event_item(self, firstPoint, startItem, lastPoint=None, endItem=None):
        if self.migration:
            eventItem = MigrationLine(firstPoint, lastPoint, startItem, endItem)
        if self.children:
            eventItem = ChildLine(firstPoint, lastPoint, startItem, endItem)
        if self.join:
            eventItem = JoinLine(firstPoint, lastPoint, startItem, endItem)
        # if self.birth:
        #     eventItem = BornSymbol(self.firstPoint, self.startItem)
        if self.appear:
            eventItem = AppearSymbol(firstPoint, startItem)
        if self.disappear:
            eventItem = DisappearSymbol(firstPoint, startItem)
        if self.remove_events:
            eventItem = None

        return(eventItem)

    def clear_all_events(self):
        print("Destroyed all tracking information")
        for item in self.items():
            # if the item isn't an qgraphicsellipseitem, 4, or a qgraphicspixmapitem, 7, remove it from the scene
            if item.type() not in [4,7]:
                # clear every item we found
                self.removeItem(item)
        self.update_tracking_info()
        # for debugging
        self.draw_focus_events()

    def remove_all_focus_events(self):
        self.remove_events = True
        self.migration = False
        self.die = False
        self.children = False
        self.birth = False
        self.appear = False
        self.disappear = False
        self.join = False

    def set_migration(self):
        # print('clicked set_migration')
        self.remove_events = False
        self.migration = True
        self.die = False
        self.children = False
        self.birth = False
        self.appear = False
        self.disappear = False
        self.join = False

    def set_join(self):
        self.remove_events = False
        self.migration = False
        self.die = False
        self.children = False
        self.birth = False
        self.appear = False
        self.disappear = False
        self.join = True

    def set_children(self):
        # print('clicked set_children')
        self.remove_events = False
        self.migration = False
        self.die = False
        self.children = True
        self.birth = False
        self.appear = False
        self.disappear = False
        self.join = False

    def set_die(self):
        # print('clicked set_die')
        self.remove_events = False
        self.migration = False
        self.die = True
        self.children = False
        self.birth = False
        self.appear = False
        self.disappear = False
        self.join = False

    def set_appear(self):
        self.remove_events = False
        self.migration = False
        self.die = False
        self.children = False
        self.birth = False
        self.appear = True
        self.disappear = False
        self.join = False

    def set_disappear(self):
        self.remove_events = False
        self.migration = False
        self.die = False
        self.children = False
        self.birth = False
        self.appear = False
        self.disappear = True
        self.join = False

    def set_born(self):
        self.remove_events = False
        self.migration = False
        self.die = False
        self.children = False
        self.birth = True
        self.appear = False
        self.disappear = False
        self.join = False

class OneFocusSymbol(QGraphicsTextItem):

    def __init__(self, point, item):
        super(OneFocusSymbol, self).__init__()

        self.item = item

        textColor = QColor(1*255,0*255,0*255)
        textFont = QFont()
        textFont.setFamily("Times")
        textFont.setPixelSize(16)
        string = "1"
        textPosition = QPoint(point.x()-9, point.y()-9)

        self.setPlainText(string)
        self.setFont(textFont)
        self.setPos(textPosition)
        self.setDefaultTextColor(textColor)

    def type(self):
        return("OneFocusSymbol")

class TwoFocusSymbol(QGraphicsTextItem):

    def __init__(self, point, item):
        super(TwoFocusSymbol, self).__init__()

        self.item = item

        textColor = QColor(1*255,0*255,0*255)
        textFont = QFont()
        textFont.setFamily("Times")
        textFont.setPixelSize(16)
        string = "2"
        textPosition = QPoint(point.x()-9, point.y()-9)

        self.setPlainText(string)
        self.setFont(textFont)
        self.setPos(textPosition)
        self.setDefaultTextColor(textColor)

    def type(self):
        return("TwoFocusSymbol")

class ThreeFocusSymbol(QGraphicsTextItem):

    def __init__(self, point, item):
        super(ThreeFocusSymbol, self).__init__()

        self.item = item

        textColor = QColor(1*255,0*255,0*255)
        textFont = QFont()
        textFont.setFamily("Times")
        textFont.setPixelSize(16)
        string = "3"
        textPosition = QPoint(point.x()-9, point.y()-9)

        self.setPlainText(string)
        self.setFont(textFont)
        self.setPos(textPosition)
        self.setDefaultTextColor(textColor)

    def type(self):
        return("ThreeFocusSymbol")

class ZeroFocusSymbol(QGraphicsTextItem):

    def __init__(self, point, item):
        super(ZeroFocusSymbol, self).__init__()

        self.item = item

        textColor = QColor(1*255,0*255,0*255)
        textFont = QFont()
        textFont.setFamily("Times")
        textFont.setPixelSize(16)
        string = "0"
        textPosition = QPoint(point.x()-9, point.y()-9)

        self.setPlainText(string)
        self.setFont(textFont)
        self.setPos(textPosition)
        self.setDefaultTextColor(textColor)

    def type(self):
        return("ZeroFocusSymbol")

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

class JoinLine(QGraphicsLineItem):
    '''
    A class for handling segmentation errors in tracking algorithm.
    '''
    def __init__(self, firstPoint, lastPoint, startItem, endItem):
        super(JoinLine, self).__init__()

        brushColor = QColor(1*255,0*255,0*255)
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
        return("JoinLine")

if __name__ == "__main__":

    # set switches and parameters
    parser = argparse.ArgumentParser(prog='python mm3_updateTrackingData.py',
                                     description='Update existing tracking data for use as training data by neural network.')
    parser.add_argument('-f', '--paramfile', type=str,
                        required=True, help='Yaml file containing parameters.')
    parser.add_argument('-c', '--channel', type=int,
                        required=False, default=1,
                        help='Which channel, e.g. phase or some fluorescence image, should be used for creating masks. \
                            Accepts integers. Default is 1, which is usually your phase contrast images.')
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
    window = Window(channel=namespace.channel)
    window.show()
    app.exec_()
