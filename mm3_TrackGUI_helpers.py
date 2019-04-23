#! /usr/bin/env python3
from __future__ import print_function, division

from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsView, QRadioButton, QButtonGroup, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QAction, QDockWidget, QPushButton, QGridLayout, QGraphicsLineItem
from PyQt5.QtGui import QIcon, QImage, QPainter, QPen, QPixmap, qGray, QColor
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

        self.setWindowTitle("You got this!")
        self.setGeometry(top,left,width,height)

        # load specs file
        with open(os.path.join(params['ana_dir'], 'specs.yaml'), 'r') as specs_file:
            specs = yaml.safe_load(specs_file)

        self.threeFrames = ThreeFrameImgWidget(self, specs=specs, training_dir=training_dir)
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

        dieButton = QRadioButton("Die")
        dieButton.setShortcut("Ctrl+D")
        dieButton.clicked.connect(self.threeFrames.scene.set_die)
        eventButtonGroup.addButton(dieButton)

        eventButtonLayout = QVBoxLayout()
        eventButtonLayout.addWidget(migrateButton)
        eventButtonLayout.addWidget(childrenButton)
        eventButtonLayout.addWidget(dieButton)

        eventButtonGroupWidget = QWidget()
        eventButtonGroupWidget.setLayout(eventButtonLayout)

        eventButtonDockWidget = QDockWidget()
        eventButtonDockWidget.setWidget(eventButtonGroupWidget)
        self.addDockWidget(Qt.LeftDockWidgetArea, eventButtonDockWidget)

class ThreeFrameImgWidget(QWidget):
    # class for setting three frames side-by-side
    def __init__(self,parent,specs,training_dir):
        super(ThreeFrameImgWidget, self).__init__(parent)

        # add QImages to scene (try three frames)
        self.center_frame_index = 1
        self.fov_id_list = [fov_id for fov_id in specs.keys()]

        self.fovIndex = 0
        self.fov_id = self.fov_id_list[self.fovIndex]

        self.peak_id_list_in_fov = [peak_id for peak_id in specs[self.fov_id].keys()]

        self.peakIndex = 0
        self.peak_id = self.peak_id_list_in_fov[self.peakIndex]

        # construct image stack file names from params
        phaseImgPath = os.path.join(params['chnl_dir'], "{}_xy{:0=3}_p{:0=4}_{}.tif".format(params['experiment_name'], self.fov_id, self.peak_id, params['phase_plane']))
        labelImgPath = os.path.join(params['seg_dir'], "{}_xy{:0=3}_p{:0=4}_seg_unet.tif".format(params['experiment_name'], self.fov_id, self.peak_id))

        # labelImgPath = imgPaths[self.fov_id][self.imgIndex][1]
        labelStack = io.imread(labelImgPath)
        # phaseImgPath = imgPaths[self.fov_id][self.imgIndex][0]
        phaseStack = io.imread(phaseImgPath)

        time_int = params['moviemaker']['seconds_per_time_index']/60

        cell_filename = os.path.join(params['cell_dir'], 'complete_cells.pkl')
        cell_filename_all = os.path.join(params['cell_dir'], 'all_cells.pkl')

        with open(cell_filename, 'rb') as cell_file:
            Cells = pickle.load(cell_file)
        mm3.calculate_pole_age(Cells) # add poleage

        with open(cell_filename_all, 'rb') as cell_file:
            All_Cells = pickle.load(cell_file)

        plot_dir = os.path.join(params['cell_dir'], '20190312_plots')
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        lin_dir = os.path.join(plot_dir, 'lineage_plots_full')
        if not os.path.exists(lin_dir):
            os.makedirs(lin_dir)

        # regions_and_events_by_time is a dictionary, the keys of which are 1-indexed frame numbers
        # for each frame, there is a dictionary with the following keys: 'matrix' and 'regions'
        #   'matrix' is a 2D array, for which the row index is the region label at time t, and the column index is the region label at time t+1
        #      If a region disappears from t to t+1, it will receive a 1 in the column with index 0.
        #   'regions' is a dictionary with each region's label as a separate key.
        #      Each region in 'regions' is another dictionary, with 'events', which contains a 1D array identifying the events that correspond to the connections in 'matrix',
        #                                                      and 'props', which contains all region properties that measure.regionprops retuns for the labelled image at time t.
        # The 'events' array is binary. The events are [migration, division, death, birth, appearance, disappearance, no_data], where a 0 at a given position in the 'events'
        #      array indicates the given event did not occur, and a 1 indicates it did occur.
        regions_and_events_by_time = self.create_tracking_information(labelStack, specs, Cells, All_Cells)
        # pprint(regions_and_events_by_time)

        # make the frames QGraphicsItems instead of qpixmaps so that I can map regions to frames.
        # keep in mind that regions_and_events_by_time is 1-indexed, whereas the phaseStack and labelStack are 0-indexed
        leftFrame = self.overlay_imgs_pixmap(phaseStack=phaseStack,labelStack=labelStack,frame_index=self.center_frame_index-1,regions_and_events_by_time[self.center_frame_index])
        centerFrame = self.overlay_imgs_pixmap(phaseStack=phaseStack,labelStack=labelStack,frame_index=self.center_frame_index,regions_and_events_by_time[self.center_frame_index+1])
        rightFrame = self.overlay_imgs_pixmap(phaseStack=phaseStack,labelStack=labelStack,frame_index=self.center_frame_index+1,regions_and_events_by_time[self.center_frame_index+2])

        self.scene = ThreeFrameScene(leftFrame,centerFrame,rightFrame,regions_and_events_by_time,self.center_frame_index)
        self.view = QGraphicsView(self)
        self.view.setScene(self.scene)

    def overlay_imgs_pixmap(phaseStack, labelStack, frame_index):

        maskImg = labelStack[frame_index,:,:]
        phaseImg = phaseStack[frame_index,:,:]
        originalImgMax = np.max(phaseImg)
        phaseImg = phaseImg/originalImgMax

        RGBImg = color.label2rgb(maskImg, phaseImg, alpha=0.25, bg_label=0)
        RGBImg = (RGBImg*255).astype('uint8')

        originalHeight, originalWidth, originalChannelNumber = RGBImg.shape
        maskQimage = QImage(RGBImg, originalWidth, originalHeight, RGBImg.strides[0], QImage.Format_RGB888).scaled(512, 512, aspectRatioMode=Qt.KeepAspectRatio)
        maskQpixmap = QPixmap(maskQimage)

        return(maskQpixmap)

    def create_tracking_information(self, labelStack, specs, Cells, All_Cells):

        Complete_Lineages = mm3_plots.organize_cells_by_channel(Cells, specs)
        All_Lineages = mm3_plots.organize_cells_by_channel(All_Cells, specs)

        t_adj = 1

        regions_by_time = {frame+t_adj: measure.regionprops(labelStack[frame,:,:]) for frame in range(labelStack.shape[0])}
        regions_and_events_by_time = {frame+t_adj : {'regions' : {}, 'matrix' : None} for frame in range(labelStack.shape[0])}

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
        cells_tmp = mm3_plots.find_cells_of_fov_and_peak(All_Cells, self.fov_id, self.peak_id)
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
                    d1_label = All_Cells[cell_tmp.daughters[0]].labels[0]
                    d2_label = All_Cells[cell_tmp.daughters[1]].labels[0]

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

class ThreeFrameScene(QGraphicsScene):

    # add more functionality for setting event type, i.e., parent-child, migrate, death, leave frame, etc..

    def __init__(self,leftFrame, centerFrame, rightFrame, regions_and_events_by_time, center_frame_index):
        super(ThreeFrameScene, self).__init__()

        self.center_frame_index = center_frame_index

        self.addPixmap(leftFrame)
        self.addPixmap(centerFrame)
        self.addPixmap(rightFrame)

        xPos = 0
        for item in self.items(order=Qt.AscendingOrder):
            item.setPos(xPos, 0)
            xPos += item.pixmap().width()

        # populate dictionary with cell track info
        self.regions_and_events = {self.center_frame_index+t: regions_and_events_by_time[self.center_frame_index+t] for t in range(3)}
        self.

        self.brushSize = 2
        self.brushColor = QColor('black')
        self.lastPoint = QPoint()
        self.origPoint = QPoint()
        self.pen = QPen()

        # class options
        self.migration = False
        self.die = False
        self.children = False
        self.birth = False
        self.appear = False
        self.disappear = False

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.firstPoint = event.scenePos()
            self.lastPoint = event.scenePos()
            self.set_line()
            self.addItem(self.line)

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) & self.drawing:

            self.lastPoint = event.scenePos()
            self.removeItem(self.line)
            self.set_line()
            self.addItem(self.line)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.lastPoint = event.scenePos()
            self.removeItem(self.line)
            self.set_line()
            self.addItem(self.line)
            self.drawing = False

    # I need to test whether the cell already has an event of the chosen type
    #   so that the existing one can be replaced by the new one I'm currently drawing.
    def set_line(self):
        self.line = QGraphicsLineItem(QLineF(self.firstPoint,self.lastPoint))
        if self.migration:
            self.brushColor = QColor('white')
        elif self.children:
            self.brushColor = QColor('green')
        elif self.die:
            self.brushColor = QColor('red')
        self.pen.setColor(self.brushColor)
        self.pen.setWidth(self.brushSize)
        self.line.setPen(self.pen)

    def set_migration(self):
        # print('clicked set_migration')
        self.migration = True
        self.children = False
        self.die = False

    def set_children(self):
        # print('clicked set_children')
        self.children = True
        self.die = False
        self.migration = False

    def set_die(self):
        # print('clicked set_die')
        self.die = True
        self.migration = False
        self.children = False

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
