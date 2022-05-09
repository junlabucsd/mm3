#! /usr/bin/env python3
from __future__ import print_function, division

from PyQt5.QtWidgets import (QApplication, QMainWindow, QMenuBar, QRadioButton,
    QMenu, QAction, QButtonGroup, QFileDialog, QVBoxLayout, QHBoxLayout, QWidget,
    QLabel, QGridLayout, QAction, QDockWidget, QPushButton, QInputDialog,QGraphicsView,
    QGraphicsScene,
    QCheckBox,
    QGraphicsItem,
    QGridLayout, QGraphicsLineItem, QGraphicsPathItem, QGraphicsPixmapItem,
    QGraphicsEllipseItem, QGraphicsTextItem)
from PyQt5.QtGui import QIcon, QImage, QPainter, QPen, QPixmap, qGray, QColor, QBrush
from PyQt5.QtCore import Qt, QPoint, QRectF,QLineF
from skimage import io, img_as_ubyte, color, draw, measure
import numpy as np
import sys
import re
import os
import yaml
import multiprocessing
try:
    import cPickle as pickle # loading and saving python objects
except:
    import pickle

import mm3_helpers as mm3
import mm3_plots
import tifffile as tiff
import math


def init_params(param_file_path):
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
    params['cell_dir'] = os.path.join(params['ana_dir'], 'cell_data')
    params['track_dir'] = os.path.join(params['ana_dir'], 'tracking')

    # use jd time in image metadata to make time table. Set to false if no jd time
    if params['TIFF_source'] == 'elements' or params['TIFF_source'] == 'nd2ToTIFF':
        params['use_jd'] = True
    else:
        params['use_jd'] = False

class cc_picking(QMainWindow):
    def __init__(self,params,cell_file):
        # super(Window, self).__init__(cell_dir,cell_file)
        super().__init__()

        top = 10
        left = 10
        width = 1100
        height = 700

        #self.setWindowTitle("")
        self.setGeometry(top,left,width,height)

        # load specs file
        with open(os.path.join(params['ana_dir'], 'specs.yaml'), 'r') as specs_file:
            specs = yaml.safe_load(specs_file)

        self.frames = FrameImgWidget(specs,cell_file,params)
        #make scene the central widget
        self.setCentralWidget(self.frames)

        eventButtonGroup = QButtonGroup()

        init1Button = QRadioButton("1")
        init1Button.setShortcut("1")
        init1Button.setToolTip("(1) No overlapping cell cycles")
        init1Button.clicked.connect(self.frames.scene.set_init1)
        eventButtonGroup.addButton(init1Button)

        init2Button = QRadioButton("2")
        init2Button.setShortcut("2")
        init2Button.setToolTip("(2) 2 overlapping cell cycles")
        init2Button.clicked.connect(self.frames.scene.set_init2)
        eventButtonGroup.addButton(init2Button)

        init3Button = QRadioButton("3")
        init3Button.setShortcut("3")
        init3Button.setToolTip("(3) 3 overlapping cell cycles")
        init3Button.clicked.connect(self.frames.scene.set_init3)
        eventButtonGroup.addButton(init3Button)

        # initButton = QRadioButton("Initiate")
        # initButton.setShortcut("I")
        # initButton.setToolTip("(I) Indicate initiate of replication")
        # initButton.clicked.connect(self.frames.scene.set_init1)
        # eventButtonGroup.addButton(init1Button)

        # resetButton = QRadioButton("Reset")
        # resetButton.setShortcut("R")
        # resetButton.setToolTip("Reset click-through")
        # resetButton.clicked.connect(self.frames.scene.reset_cc)
        # eventButtonGroup.addButton(resetButton)

        resetButton = QPushButton("Reset")
        resetButton.setShortcut("R")
        resetButton.setToolTip("Reset click-through")
        resetButton.clicked.connect(self.frames.scene.reset_cc)

        undoButton = QPushButton("Clear events")
        undoButton.setShortcut("U")
        undoButton.setToolTip("(U) Clear events from peak")
        undoButton.clicked.connect(self.frames.scene.clear_cc_events)

        flagButton = QPushButton("Flag cell")
        flagButton.setShortcut("X")
        flagButton.setToolTip("(X) Flag cell to check later")
        flagButton.clicked.connect(self.frames.scene.flag_cell)

        olButton = QPushButton("Toggle overlay")
        olButton.setShortcut("T")
        olButton.setToolTip("(T) Toggle foci overlay")
        olButton.clicked.connect(self.frames.scene.toggle_overlay)

        saveUpdatedTracksButton = QPushButton("Save updated initiations")
        saveUpdatedTracksButton.setShortcut("S")
        saveUpdatedTracksButton.clicked.connect(self.frames.scene.save_output)

        eventButtonLayout = QVBoxLayout()

        eventButtonLayout.addWidget(resetButton)

        eventButtonLayout.addWidget(init1Button)
        eventButtonLayout.addWidget(init2Button)
        eventButtonLayout.addWidget(init3Button)

        eventButtonLayout.addWidget(undoButton)
        eventButtonLayout.addWidget(flagButton)
        eventButtonLayout.addWidget(olButton)

        eventButtonGroupWidget = QWidget()
        eventButtonGroupWidget.setLayout(eventButtonLayout)

        eventButtonDockWidget = QDockWidget()
        eventButtonDockWidget.setWidget(eventButtonGroupWidget)
        self.addDockWidget(Qt.LeftDockWidgetArea, eventButtonDockWidget)

        advancePeakButton = QPushButton("Next peak")
        advancePeakButton.setShortcut("P")
        advancePeakButton.clicked.connect(self.frames.scene.next_peak)

        priorPeakButton = QPushButton("Prior peak")
        priorPeakButton.clicked.connect(self.frames.scene.prior_peak)

        advanceFOVButton = QPushButton("Next FOV")
        advanceFOVButton.setShortcut("F")
        advanceFOVButton.clicked.connect(self.frames.scene.next_fov)

        priorFOVButton = QPushButton("Prior FOV")
        priorFOVButton.clicked.connect(self.frames.scene.prior_fov)

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
    def __init__(self,specs,cell_file,params):
        super(FrameImgWidget, self).__init__()
        self.specs = specs
        self.scene = TrackItem(self.specs,cell_file,params)
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

class TrackItem(QGraphicsScene):

    def __init__(self,specs,cell_file,params):
        # super(TrackItem, self).__init__()
        super().__init__()
        self.items = []

        #these set the size of the kymograph display in pixels in phase_imgs_and_regions
        self.y_scale = 700
        self.x_scale = 1100

        self.specs = specs

        self.fov_id_list = [fov_id for fov_id in specs.keys()]

        self.fovIndex = 0
        self.fov_id = self.fov_id_list[self.fovIndex]

        self.peak_id_list_in_fov = [peak_id for peak_id in specs[self.fov_id].keys() if specs[self.fov_id][peak_id] == 1]

        self.peakIndex = 0
        self.peak_id = self.peak_id_list_in_fov[self.peakIndex]

        self.color = params['foci']['foci_plane']



        with open(os.path.join(params['cell_dir'],cell_file), 'rb') as cf:
            all_cells = pickle.load(cf)

        ## filter for cells with doubling time above some threshold, to eliminate all_cell artefacts. Maybe filter for cells with mothers only?

        # complete_cells = {}
        # #
        # for cell_id in all_cells:
        #     if all_cells[cell_id].parent and len(all_cells[cell_id].times) > 4.:
        #         complete_cells[cell_id] = all_cells[cell_id]

        # self.Cells = complete_cells
        self.Cells = all_cells

        self.Cells_by_peak = mm3_plots.organize_cells_by_channel(self.Cells,specs)
        self.cell_id_list_in_peak = [cell_id for cell_id in self.Cells_by_peak[self.fov_id][self.peak_id].keys()]

        self.init1 = False
        self.init2 = False
        self.init3 = False
        #self.remove_events = False
        self.remove_last = False
        self.reset = False
        self.flag = False
        self.overlay_cc = True

        self.clicks = 0
        self.ccf = []

        img_dir = params['experiment_directory']+ params['analysis_directory'] + 'kymograph'

        img_filename = params['experiment_name'] + '_xy%03d_p%04d_%s.tif' % (self.fov_id, self.peak_id, self.color)
        #img_filename = params['experiment_name'] + '_xy%03d_p%04d_seg.tif' % (self.fov_id, self.peak_id)

        with tiff.TiffFile(os.path.join(img_dir, img_filename)) as tif:
            fl_proj = tif.asarray()

        print('FOV '+str(self.fov_id))
        print('Peak '+str(self.peak_id))

        mm3.load_time_table()

        # determine absolute time index
        time_table = params['time_table']
        times_all = []
        abs_times = []
        for fov in params['time_table']:
            times_all = np.append(times_all, [int(x) for x in time_table[fov].keys()])
            abs_times = np.append(abs_times, [int(x) for x in time_table[fov].values()])
        times_all = np.unique(times_all)
        times_all = np.sort(times_all)
        abs_times = np.unique(abs_times)
        abs_times = np.sort(abs_times)

        times_all = np.array(times_all,np.int_)
        abs_times = np.array(abs_times,np.int_)

        t0 = times_all[0] # first time index

        self.times_all = times_all
        self.abs_times = abs_times

        self.x_px = len(fl_proj[0])
        self.y_px = len(fl_proj[:,0])

        self.fl_kymo = fl_proj

        overlay = self.phase_img_and_regions()
        self.addPixmap(overlay)

        self.label_divs()

        ## draw predicted replication tracks


    def next_peak(self):
        # start by removing all current graphics items from the scene, the scene here being 'self'
        # write overwrite cell dictionary only if new events were added to the scene
        if len(self.items)>0.:
            self.match_cc_events()
            print('writing new cc data')

        self.clear()
        specs = self.specs
        self.peakIndex += 1
        self.ccf = []
        self.clicks = 0
        self.items = []

        self.flag = False

        try:
            self.peak_id = self.peak_id_list_in_fov[self.peakIndex]
            print('Peak '+str(self.peak_id))

        except IndexError:
            print('go to next FOV')
            return

        try:
            self.cell_id_list_in_peak = [cell_id for cell_id in self.Cells_by_peak[self.fov_id][self.peak_id].keys()]

        except KeyError:
            print('no cells in this peak')
            self.peakIndex+=1
            return

        img_dir = params['experiment_directory']+ params['analysis_directory'] + 'kymograph'

        img_filename = params['experiment_name'] + '_xy%03d_p%04d_%s.tif' % (self.fov_id, self.peak_id, self.color)

        #img_filename = params['experiment_name'] + '_xy%03d_p%04d_seg.tif' % (self.fov_id, self.peak_id)

        with tiff.TiffFile(os.path.join(img_dir, img_filename)) as tif:
            fl_proj = tif.asarray()

        print('FOV '+str(self.fov_id))
        print('Peak '+str(self.peak_id))

        self.fl_kymo = fl_proj

        overlay = self.phase_img_and_regions()
        self.addPixmap(overlay)

        self.label_divs()

    def prior_peak(self):
        # start by removing all current graphics items from the scene, the scene here being 'self'
        self.clear()
        specs = self.specs
        self.peakIndex -= 1
        self.ccf = []
        self.clicks = 0
        self.items = []
        self.flag = False

        try:
            self.peak_id = self.peak_id_list_in_fov[self.peakIndex]

        except IndexError:
            print('go to previous FOV')

        img_dir = params['experiment_directory']+ params['analysis_directory'] + 'kymograph'

        img_filename = params['experiment_name'] + '_xy%03d_p%04d_%s.tif' % (self.fov_id, self.peak_id, self.color)

        with tiff.TiffFile(os.path.join(img_dir, img_filename)) as tif:
            fl_proj = tif.asarray()

        print('FOV '+str(self.fov_id))
        print('Peak '+str(self.peak_id))

        self.fl_kymo = fl_proj

        overlay = self.phase_img_and_regions()
        self.addPixmap(overlay)

        self.label_divs()

    def next_fov(self):
        #overwrite cc events for this peak only if new data was added
        if len(self.items)>0:
            self.match_cc_events()
            print('writing new cc data')

        self.clear()

        self.ccf = []
        self.items = []

        specs = self.specs
        self.clicks = 0

        self.fovIndex += 1
        self.flag = False

        # if self.fovIndex == 20:
        #     self.fovIndex +=1
        try:
            self.fov_id = self.fov_id_list[self.fovIndex]
        except IndexError:
            print('FOV not found')
            while self.fovIndex < len(self.fov_id_list):
                try:
                    self.fovIndex+=1
                    self.fov_id = self.fov_id_list[self.fovIndex]
                except:
                    pass
                break
            if self.fovIndex == len(self.fov_id_list):
                print('no FOVs remaining')
                return


        self.peak_id_list_in_fov = [peak_id for peak_id in self.specs[self.fov_id].keys() if self.specs[self.fov_id][peak_id] == 1]

        self.peakIndex = 0
        self.peak_id = self.peak_id_list_in_fov[self.peakIndex]

        img_dir = params['experiment_directory']+ params['analysis_directory'] + 'kymograph'

        img_filename = params['experiment_name'] + '_xy%03d_p%04d_%s.tif' % (self.fov_id, self.peak_id, self.color)
        #img_filename = params['experiment_name'] + '_xy%03d_p%04d_seg.tif' % (self.fov_id, self.peak_id)

        with tiff.TiffFile(os.path.join(img_dir, img_filename)) as tif:
            fl_proj = tif.asarray()

        print('FOV '+str(self.fov_id))
        print('Peak '+str(self.peak_id))

        self.fl_kymo = fl_proj

        overlay = self.phase_img_and_regions()
        self.addPixmap(overlay)

        self.label_divs()

    def prior_fov(self):
        self.clear()

        specs = self.specs
        self.ccf = []
        self.items = []
        self.clicks = 0

        self.flag = False

        self.fovIndex -= 1
        self.fov_id = self.fov_id_list[self.fovIndex]

        self.peak_id_list_in_fov = [peak_id for peak_id in self.specs[self.fov_id].keys() if self.specs[self.fov_id][peak_id] == 1]

        self.peakIndex = 0
        self.peak_id = self.peak_id_list_in_fov[self.peakIndex]

        img_dir = params['experiment_directory']+ params['analysis_directory'] + 'kymograph'

        img_filename = params['experiment_name'] + '_xy%03d_p%04d_%s.tif' % (self.fov_id, self.peak_id, self.color)

        with tiff.TiffFile(os.path.join(img_dir, img_filename)) as tif:
            fl_proj = tif.asarray()

        print('FOV '+str(self.fov_id))
        print('Peak '+str(self.peak_id))

        self.fl_kymo = fl_proj

        overlay = self.phase_img_and_regions()
        self.addPixmap(overlay)

        self.label_divs()

    def phase_img_and_regions(self):

        phaseImg = self.fl_kymo
        originalImgMax = np.max(phaseImg)
        phaseImgN = phaseImg/originalImgMax
        phaseImg = color.gray2rgb(phaseImgN)
        RGBLabelImg = (phaseImg*255).astype('uint8')

        originalHeight, originalWidth, RGBLabelChannelNumber = RGBLabelImg.shape
        RGBLabelImg = QImage(RGBLabelImg, originalWidth, originalHeight, RGBLabelImg.strides[0], QImage.Format_RGB888).scaled(self.x_scale,self.y_scale, aspectRatioMode=Qt.IgnoreAspectRatio)
        labelQpixmap = QPixmap(RGBLabelImg)

        return(labelQpixmap)

    def label_divs(self):
        try:
            cells_tmp = self.Cells_by_peak[self.fov_id][self.peak_id]
        except KeyError:
            return
        self.divs_p = []
        self.divs_t = []

        pen = QPen()
        brush = QBrush()
        brush.setStyle(Qt.SolidPattern)
        brush.setColor(QColor("white"))

        pen.setWidth(3)

        for cell_id, cell in cells_tmp.items():
            if cell.division_time:
                #convert division time (in frames) to window position (x_scale / x_px = pixels / time frame)
                x = cell.division_time*self.x_scale/self.x_px

                #convert centroid position at division to window position (y_scale / y_px = window pixel / camera pixel)
                # y = cell.centroids[-1][0]*self.y_scale/self.y_px
                y = cell.centroids[-1][0]*self.y_scale/self.y_px

                ld = cell.lengths[-1]*1*self.y_scale/self.y_px

                try:
                    init = QGraphicsEllipseItem(x-5,(y-5),10,10)
                except TypeError:
                    continue

                pen.setColor(QColor("red"))

                init.setPen(pen)
                init.setBrush(brush)
                self.divs_p.append([x,y])
                self.divs_t.append([cell.division_time, cell.centroids[-1][0]])
                self.addItem(init)

                eventItem = self.set_event_item(QPoint(x,(y + ld/2)),QPoint(x,(y - ld/2)), color="white")
                self.addItem(eventItem)

            times = np.array(cell.times)*self.x_scale/self.x_px
            lengths = np.array(cell.lengths)*self.y_scale/self.y_px
            cents = np.array(cell.centroids)[:,0]*self.y_scale/self.y_px

            # birth = QGraphicsEllipseItem(times[0]-5, cents[0]-5, 10, 10)
            # pen.setColor(QColor("green"))
            # birth.setPen(pen)
            # birth.setBrush(brush)
            # self.addItem(birth)


            for i in range(len(cell.times)-1):
                eventItem = self.set_event_item(QPoint(times[i],cents[i] - lengths[i]/2),QPoint(times[i+1],cents[i+1] - lengths[i+1]/2),color="yellow")
                self.addItem(eventItem)
                eventItem = self.set_event_item(QPoint(times[i],cents[i] + lengths[i]/2),QPoint(times[i+1],cents[i+1] + lengths[i+1]/2), color="yellow")
                self.addItem(eventItem)
            self.overlay_cc = True
            if self.overlay_cc:
                # (c1,c2,c3) = np.random.rand(3,)

                # try:
                if cell.disp_l:
                    print(type(cell.disp_l))
                    print(type(self.y_scale))
                    print(type(self.y_px))
                    # disps = np.array(cell.disp_l) * self.y_scale/self.y_px
                    disps = [np.array(x)*self.y_scale/self.y_px for x in cell.disp_l]
                    for t, c, l in zip(times,cents,cell.disp_l):
                        for i in range(len(l)):
                            focus = QGraphicsEllipseItem(t-3,c+l[i]* self.y_scale/self.y_px-3,6,6)
                            penColor= QColor("orange")
                            # penColor.setAlphaF(0.5)
                            pen.setColor(penColor)
                            focus.setPen(pen)
                            #focus.setBrush(brush)
                            self.addItem(focus)
                if cell.termination_time:
                    penColor= QColor('blue')
                    pen.setColor(penColor)
                    #mark termination in same color but bolded
                    focus = QGraphicsEllipseItem(cell.termination_time-4,cell.termination_y* self.y_scale/self.y_px-4,8,8)
                    focus.setPen(pen)

                    self.addItem(focus)
                    print('adding termination time')
                if cell.initiation_time:
                    penColor= QColor('green')
                    pen.setColor(penColor)
                    #mark initiation in same color but bolded
                    focus = QGraphicsEllipseItem(cell.initiation_time-4,cell.initiation_y* self.y_scale/self.y_px-4,8,8)
                    focus.setPen(pen)

                    self.addItem(focus)
                    print('adding initiation time')


                # except AttributeError:
                #         pass


    def save_output(self):
        cell_filename = os.path.join(params['cell_dir'], 'complete_cells.pkl')
        cells_out = {}
        for fov_id in self.fov_id_list:
            peak_id_list_in_fov = [peak_id for peak_id in self.specs[fov_id].keys() if self.specs[fov_id][peak_id] == 1]
            for peak_id in peak_id_list_in_fov:
                try:
                    cells_out.update(self.Cells_by_peak[fov_id][peak_id])
                except KeyError:
                    print('Missing ' +str((fov_id, peak_id)))
        with open(os.path.join(params['cell_dir'], cell_filename[:-4] + '_test_foci.pkl'), 'wb') as cell_file:
            pickle.dump(cells_out, cell_file, protocol=pickle.HIGHEST_PROTOCOL)
            print('saved updated initiations to pickle file')


    def clear_cc_events(self):
        #clear data for this peak
        for item in self.items:
            self.removeItem(item)
        self.items = []
        self.ccf = []
        self.clicks = 0

    def mousePressEvent(self, event):
        #default should be no button selected
        # if self.remove_last:
        #     #clear data for this peak
        #     for item in self.items:
        #         self.removeItem(item)
        #     self.items = []
        #     self.ccf = []
        #     return

        # if self.reset:
        #     self.clicks = 0
            #return

        # if #self.remove_events:
        #     for item in self.items:
        #         self.removeItem(item)
        #         self.items.remove(item)
        #
        #     return
        self.reset = False
        if self.init1: #edit to add button that indicates init
            self.nc = 1

        elif self.init2:
            self.nc = 2

        elif self.init3:
            self.nc = 3

        else:
            print('Defaulting to two overlapping cell cycle')
            self.nc = 2

        ccf = self.ccf

        if self.clicks == 0:
            #initiation event
            cckeys = ['init_time','init_pos', 'term_time','term_pos','div_time','div_pos']
            ccf.append(dict.fromkeys(cckeys,None))#add new cell dict to list
            col = "green"

            self.drawing = True

            x = event.scenePos().x()
            y = event.scenePos().y()

            self.init_pos = QPoint(x,y)

            init = QGraphicsEllipseItem(x-5,y-5,10,10)
            pen = QPen()
            pen.setWidth(3)
            pen.setColor(QColor(col))
            init.setPen(pen)

            self.addItem(init)
            self.items.append(init)
            init_tmp = math.ceil(x/self.x_scale*self.x_px)
            init_y = math.ceil(y/self.y_scale*self.y_px)
            ccf[-1]['init_time'] = init_tmp
            ccf[-1]['init_pos'] = init_y
            ccf[-1]['n_oc'] = self.nc

            self.clicks +=1

        elif self.clicks == 1:
            #termination event
            #do not add new dict, append to old
            col = 'blue'
            x = event.scenePos().x()
            y = event.scenePos().y()

            term = QGraphicsEllipseItem(x-5,y-5,10,10)
            pen = QPen()
            pen.setWidth(3)
            pen.setColor(QColor(col))
            term.setPen(pen)

            self.term_pos = QPoint(x,y)

            self.addItem(term)
            self.items.append(term)

            eventItem = self.set_event_item(self.init_pos,self.term_pos)
            self.addItem(eventItem)
            self.items.append(eventItem)

            term_t = math.ceil(x*self.x_px/self.x_scale)
            term_y = math.ceil(y*self.y_px/self.y_scale)

            ccf[-1]['term_time'] = term_t
            ccf[-1]['term_pos'] = term_y

            self.clicks +=1

        else:
            #division event
            #do not add new dict, append to old
            col = "red"
            x = event.scenePos().x()
            y = event.scenePos().y()

            divs_x = np.array(self.divs_p)[:,0] - x
            divs_y = np.array(self.divs_p)[:,1] - y
            diffs = [divs_x**2 + divs_y**2 for x,y in zip(divs_x,divs_y)]

            x1 = np.array(self.divs_p)[np.argmin(diffs),0]
            y1 = np.array(self.divs_p)[np.argmin(diffs),1]

            #snap to nearest division event

            div = QGraphicsEllipseItem(x1-5,(y1-5),10,10)
            pen = QPen()
            pen.setWidth(3)
            pen.setColor(QColor(col))
            div.setPen(pen)

            brush = QBrush()
            brush.setColor(QColor("white"))
            div.setBrush(brush)

            self.addItem(div)

            div_t_tmp = np.array(self.divs_t)[np.argmin(diffs),0]
            div_y_tmp = np.array(self.divs_t)[np.argmin(diffs),1]

            ccf[-1]['div_time'] = div_t_tmp
            ccf[-1]['div_pos'] = div_y_tmp

            div_pos = QPoint(x1,y1)

            eventItem = self.set_event_item(self.term_pos,div_pos)
            self.addItem(eventItem)
            self.items.append(eventItem)

            self.clicks = 0

        self.ccf = ccf

    def match_cc_events(self):
        cc = self.ccf
        cells_tmp = self.Cells_by_peak[self.fov_id][self.peak_id]

        required_attr = {'initiation_size_n','initiation_size','initiation_length_n','initiation_length','termination_time', 'termination_time_n','B','C','D','B_min','C_min','D_min','initiation_time','initiation_time_n','n_oc'}
        for key,cell in cells_tmp.items():
            for attr in required_attr:
                try:
                    cell.attr
                except AttributeError:
                    setattr(cell,attr,None)

        try:
            mpf = params['min_per_frame']
        except:
            print('no min per frame found, defaulting to 1')
            mpf = 1

        for cell_dict in cc:

            div_time, div_pos, term_time, term_pos, init_time, init_pos, init_s, init_l, init_s_n, init_l_n = [None] * 10

            init_time = cell_dict['init_time']
            init_pos = cell_dict['init_pos']
            term_time = cell_dict['term_time']
            term_pos = cell_dict['term_pos']
            div_time = cell_dict['div_time']
            div_pos = cell_dict['div_pos']
            nc = cell_dict['n_oc']

            found = False
            if div_time:
                for (cell_id, cell) in cells_tmp.items():
                    #first try to match cells by division time
                    if cell.division_time:
                        if cell.division_time == div_time and cell.centroids[-1][0] == div_pos:
                            print('matched cell by division')
                            found = True
                            # compute initiation mass, C & D periods
                            B = init_time - cell.birth_time
                            C = term_time - init_time
                            D = div_time - term_time

                            #get absolute time of initiation -> subtract cell birth time -> take cell volume / n_oc
                            try:
                                if nc == 1:
                                    if cell.birth_time < init_time < cell.division_time:
                                        init_s = cell.volumes_w_div[init_time - cell.birth_time]/2**(nc-1)
                                        init_l = cell.lengths_w_div[init_time - cell.birth_time]/2**(nc-1)

                                if nc == 2:
                                    if cell.parent in cells_tmp:
                                        cell_m = cells_tmp[cell.parent]
                                        if cell_m.birth_time < init_time < cell_m.division_time:
                                            init_s = cell_m.volumes_w_div[init_time - cell_m.birth_time]/2**(nc-1)
                                            init_l = cell_m.lengths_w_div[init_time - cell_m.birth_time]/2**(nc-1)

                                            if cell.labels[0] == 1:
                                                init_l_n = cell_m.lengths_w_div[init_time - cell_m.birth_time]/2**(nc-1)
                                                init_s_n = cell_m.volumes_w_div[init_time - cell_m.birth_time]/2**(nc-1)
                                                cell_m.initiation_size_n = init_s_n
                                                cell_m.initiation_length_n = init_l_n
                                                cell_m.initiation_time_n = [init_time]
                                                cell_m.termination_time_n = [term_time]

                                if nc == 3:
                                    if cell.parent in cells_tmp:
                                        if cells_tmp[cell.parent].parent in cells_tmp:
                                            cell_g = cells_tmp[cells_tmp[cell.parent].parent]
                                            if cell_g.birth_time < init_time < cell_g.division_time:
                                                init_s = cell_g.volumes_w_div[init_time - cell_g.birth_time]/2**(nc-1)
                                                init_l = cell_g.lengths_w_div[init_time - cell_g.birth_time]/2**(nc-1)
                                                if cell.labels[0] == 1:
                                                    init_l_n = cell_g.lengths_w_div[init_time - cell_g.birth_time]/2**(nc-1)
                                                    init_s_n = cell_g.volumes_w_div[init_time - cell_g.birth_time]/2**(nc-1)
                                                    cell_g.initiation_size_n = init_s_n
                                                    cell_g.initiation_length_n = init_l_n
                                                    cell_g.initiation_time_n = [init_time]
                                                    cell_g.termination_time_n = [term_time]

                            except IndexError:
                                pass
                            cell.initiation_size = init_s
                            cell.initiation_length = init_l
                            cell.initiation_time = init_time
                            #cell.initiation_time_n = init_time_n
                            cell.termination_time = term_time
                            cell.B = B
                            cell.C = C
                            cell.D = D
                            cell.B_min = cell.B * mpf
                            cell.C_min = cell.C * mpf
                            cell.D_min = cell.D * mpf
                            cell.n_oc = nc
                            cell.flag = self.flag

                            break

            else:
                if term_time:
                    found = False
                    for (cell_id, cell) in cells_tmp.items():
                        if cell.birth_time < term_time < cell.times[-1]:
                            try:
                                y_dist = abs(cell.centroids[term_time-cell.birth_time][0] - term_pos)
                                y_bound = cell.lengths[term_time-cell.birth_time]
                            except IndexError:
                                continue
                            #find the cell containing the termination event
                            if abs(cell.centroids[term_time-cell.birth_time][0] - term_pos) < cell.lengths[term_time-cell.birth_time]:
                                print('matched by termination time')
                                found = True

                                B = init_time - cell.birth_time
                                C = term_time - init_time
                                #get absolute time of initiation -> subtract cell birth time -> take cell volume / n_oc
                                try:
                                    if nc == 1:
                                        # this is the simplest case
                                        init_s = cell.volumes[init_time - cell.birth_time]*params['pxl2um']**3/2**(nc-1)
                                        init_l = cell.lengths[init_time - cell.birth_time]*params['pxl2um']/2**(nc-1)

                                        cell.initiation_time = init_time
                                        cell.termination_time = term_time
                                        cell.initiation_size = init_s
                                        cell.initiation_length = init_l
                                        cell.B = B
                                        cell.C = C
                                        cell.B_min = cell.B * mpf
                                        cell.C_min = cell.C * mpf
                                        print('saved init mass, time, B & C')

                                    if nc == (2 or 3) and cell.birth_time < init_time < cell.times[-1]:
                                        # if we have overlapping cell cycles and the initiation & termination happened in this generation, store as init_n and term_n
                                        # i.e. these replication events would not correspond to segregation at (hypothetical) cell division
                                        init_l_n = cell.lengths[init_time - cell.birth_time]*params['pxl2um']/2**(nc-1)
                                        init_s_n = cell.volumes[init_time - cell.birth_time]*params['pxl2um']**3/2**(nc-1)
                                        cell.initiation_size_n = init_s_n
                                        cell.initiation_length_n = init_l_n
                                        # check if this cell already has an initiation assigned
                                        if cell.initiation_time_n == None:
                                            cell.initiation_time_n = [init_time]
                                            cell.termination_time_n = [term_time]

                                        # if it does, add this one rather than overwriting
                                        else:
                                            cell.initiation_time_n.append(init_time)
                                            cell.termination_time_n.append(term_time)
                                        print('saved init mass, time, B & C')

                                    elif nc == 2:
                                        # if initiation happened before birth, assume this cycle links to cell division and store as regular initiation
                                        cell.initiation_time = init_time
                                        cell.termination_time = term_time
                                        cell.B = B
                                        cell.C = C
                                        cell.B_min = cell.B * mpf
                                        cell.C_min = cell.C * mpf

                                        if cell.parent in cells_tmp:
                                            cell_m = cells_tmp[cell.parent]
                                            if cell_m.birth_time < init_time < cell_m.division_time:
                                                init_s = cell_m.volumes[init_time - cell_m.birth_time]*params['pxl2um']**3/2**(nc-1)
                                                init_l = cell_m.lengths[init_time - cell_m.birth_time]*params['pxl2um']/2**(nc-1)
                                                cell.initiation_size = init_s
                                                cell.initiation_length = init_l
                                                if cell.labels[0] == 1:
                                                    init_l_n = cell_m.lengths[init_time - cell_m.birth_time]*params['pxl2um']/2**(nc-1)
                                                    init_s_n = cell_m.volumes[init_time - cell_m.birth_time]*params['pxl2um']**3/2**(nc-1)
                                                    cell_m.initiation_size_n = init_s_n
                                                    cell_m.initiation_length_n = init_l_n
                                                    cell_m.initiation_time_n = [init_time]
                                                    cell_m.termination_time_n = [term_time]
                                        print('saved init mass, time, B & C')

                                    elif nc == 3:
                                        # again if initiation happened before birth, assume this cycle links to cell division and store as regular initiation
                                        ## for the case of D period longer than doubling time, this needs to be modified
                                        cell.B = B
                                        cell.C = C
                                        cell.B_min = cell.B * mpf
                                        cell.C_min = cell.C * mpf
                                        cell.initiation_time = init_time
                                        cell.termination_time = term_time

                                        if cell.parent in cells_tmp:
                                            if cells_tmp[cell.parent].parent in cells_tmp:
                                                cell_g = cells_tmp[cells_tmp[cell.parent].parent]
                                                if cell_g.birth_time < init_time < cell_g.division_time:
                                                    init_s = cell_g.volumes[init_time - cell_g.birth_time]*params['pxl2um']**3/2**(nc-1)
                                                    init_l = cell_g.lengths[init_time - cell_g.birth_time]*params['pxl2um']/2**(nc-1)
                                                    cell.initiation_size = init_s
                                                    cell.initiation_length = init_l
                                                    if cell.labels[0] == 1:
                                                        init_l_n = cell_g.lengths[init_time - cell_g.birth_time]*params['pxl2um']/2**(nc-1)
                                                        init_s_n = cell_g.volumes[init_time - cell_g.birth_time]*params['pxl2um']**3/2**(nc-1)
                                                        cell_g.initiation_size_n = init_s_n
                                                        cell_g.initiation_length_n = init_l_n
                                                        cell_g.initiation_time_n = [init_time]
                                                        cell_g.termination_time_n = [term_time]
                                        print('saved init mass, time, B & C')

                                except:
                                    pass


                                cell.n_oc = nc
                                cell.flag = self.flag

                                break

                if init_time and (found==False):
                    for (cell_id, cell) in cells_tmp.items():

                        if cell.birth_time < init_time < cell.times[-1]:
                            try:
                                y_dist = abs(cell.centroids[init_time-cell.birth_time][0] - init_pos)
                                y_bound = cell.lengths[init_time - cell.birth_time]
                            except IndexError:
                                continue
                            if abs(cell.centroids[init_time-cell.birth_time][0] - init_pos) < cell.lengths[init_time - cell.birth_time]:
                                print('matched by initiation time')
                                # compute initiation mass
                                try:
                                    if nc == 1:
                                        #store this initiation as init i.e. part of this cell's cycle
                                        init_s = cell.volumes[init_time - cell.birth_time]*params['pxl2um']**3/2**(nc-1)
                                        init_l = cell.lengths[init_time - cell.birth_time]*params['pxl2um']/2**(nc-1)
                                        cell.initiation_size = init_s
                                        cell.initiation_length = init_l
                                        cell.initiation_time = init_time

                                        cell.B = init_time - cell.birth_time
                                        cell.B_min = cell.B * mpf
                                        # don't know if there is a termination event for this cycle
                                        try:
                                            cell.termination_time = term_time
                                            cell.C = term_time - init_time
                                            cell.C_min = cell.C * mpf
                                        except:
                                            pass
                                        print('saved init mass and time')

                                    elif nc == 2:
                                        #this initiation occured in the cell's lifetime but not linked to its division. store as init_n
                                        init_s_n = cell.volumes[init_time - cell.birth_time]*params['pxl2um']**3/2**(nc-1)
                                        init_l_n = cell.lengths[init_time - cell.birth_time]*params['pxl2um']/2**(nc-1)
                                        cell.initiation_size_n = init_s_n
                                        cell.initiation_length_n = init_l_n
                                        if cell.initiation_time_n == None:
                                            cell.initiation_time_n = [init_time]
                                            # don't know if there is a termination event for this cycle
                                            try:
                                                cell.termination_time_n = [term_time]
                                            except:
                                                cell.termination_time_n = [None]

                                        else:
                                            cell.initiation_time_n.append(init_time)
                                            try:
                                                cell.termination_time_n.append(term_time)
                                            except:
                                                cell.termination_time_n.append(None)
                                        print('saved init mass and time')


                                    elif nc == 3:
                                        #this initiation occured in the cell's lifetime but not linked to its division. store as init_n
                                        init_l_n = cell.lengths[init_time - cell.birth_time]*params['pxl2um']/2**(nc-1)
                                        init_s_n = cell.volumes[init_time - cell.birth_time]*params['pxl2um']**3/2**(nc-1)
                                        cell.initiation_size_n = init_s_n
                                        cell.initiation_length_n = init_l_n
                                        if cell.initiation_time_n == None:
                                            cell.initiation_time_n = [init_time]
                                            try:
                                                cell.termination_time_n = [term_time]
                                            except:
                                                cell.termination_time_n = [None]
                                        else:
                                            cell.initiation_time_n.append(init_time)
                                            try:
                                                cell.termination_time_n.append(term_time)
                                            except:
                                                cell.termination_time_n.append(None)
                                        print('saved init mass and time')


                                except:
                                    pass

                                cell.n_oc = nc
                                cell.flag = self.flag

                                break
                                # try:
                                #     setattr(cells_tmp[cell_id], attr, val)
                                # except: continue

        self.Cells_by_peak[self.fov_id][self.peak_id] = cells_tmp
        # for (cell_id, cell) in self.Cells_by_peak[1][self.peak_id].items():
        #     print(cell.C_min)
        #     print(cell.D_min)
        #     print(cell.initiation_size)
        #     print(cell.initiation_length)


    def get_time(self, cell):
        return(cell.time)

    def set_init1(self):
        #self.remove_events = False
        self.init1 = True
        self.init2 = False
        self.init3 = False
        self.remove_last = False
        self.reset = False

    def set_init2(self):
        #self.remove_events = False
        self.init1 = False
        self.init2 = True
        self.init3 = False
        self.remove_last = False
        self.reset = False

    def set_init3(self):
        #self.remove_events = False
        self.init1 = False
        self.init2 = False
        self.init3 = True
        self.remove_last = False
        self.reset = False

    def reset_cc(self):
        #self.remove_events = False
        # self.init1 = False
        # self.init2 = False
        # self.init3 = False
        self.remove_last = False
        self.reset = True
        self.clicks = 0

    def flag_cell(self):
        if self.flag == True:
            self.flag = False
            print('unflagging peak')
        elif self.flag == False:
            self.flag = True
            print('flagging peak')

    def toggle_overlay(self):
        if self.overlay_cc == False:
            self.overlay_cc = True
        elif self.overlay_cc == True:
            self.overlay_cc = False

    def set_event_item(self,firstPoint,lastPoint, color="white"):
        eventItem = RepLine(firstPoint, lastPoint, color)

        return(eventItem)

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

class RepLine(QGraphicsLineItem):
    # A class for helping to draw and organize migration events
    #  within a QGraphicsScene
    def __init__(self, firstPoint, lastPoint, color):
        super(RepLine, self).__init__()

        brushColor = QColor(color)
        brushSize = 3
        pen = QPen()
        firstPointX = firstPoint.x()
        lastPointX = lastPoint.x()
        # ensure that lines' starts are to the left of their ends
        if firstPointX < lastPointX:
            self.start = firstPoint
            #self.startItem = startItem
            self.end = lastPoint
            #self.endItem = endItem
        else:
            self.start = lastPoint
            #self.startItem = endItem
            self.end = firstPoint
            #self.endItem = startItem
        line = QLineF(self.start,self.end)
        brushColor.setAlphaF(0.5)
        pen.setColor(brushColor)
        pen.setWidth(brushSize)
        self.setPen(pen)
        self.setLine(line)

    def type(self):
        return("RepLine")

class Window(QMainWindow):

    def __init__(self, parent=None, imgPaths=None, fov_id_list=None, training_dir=None):
        super(Window, self).__init__(parent)

        top = 400
        left = 400
        width = 400
        height = 1200

        self.setWindowTitle("You got this!")
        self.setGeometry(top,left,width,height)

        self.ImgsWidget = OverlayImgsWidget(self, imgPaths=imgPaths, fov_id_list=fov_id_list, training_dir=training_dir)
        self.setCentralWidget(self.ImgsWidget)

        brushSizeGroup = QButtonGroup()
        onePxButton = QRadioButton("1px")
        onePxButton.setShortcut("Ctrl+1")
        onePxButton.clicked.connect(self.ImgsWidget.mask_widget.onePx)
        brushSizeGroup.addButton(onePxButton)

        threePxButton = QRadioButton("3px")
        threePxButton.setShortcut("Ctrl+3")
        threePxButton.clicked.connect(self.ImgsWidget.mask_widget.threePx)
        brushSizeGroup.addButton(threePxButton)

        fivePxButton = QRadioButton("5px")
        fivePxButton.setShortcut("Ctrl+5")
        fivePxButton.clicked.connect(self.ImgsWidget.mask_widget.fivePx)
        brushSizeGroup.addButton(fivePxButton)

        sevenPxButton = QRadioButton("7px")
        sevenPxButton.setShortcut("Ctrl+7")
        sevenPxButton.clicked.connect(self.ImgsWidget.mask_widget.sevenPx)
        brushSizeGroup.addButton(sevenPxButton)

        ninePxButton = QRadioButton("9px")
        ninePxButton.setShortcut("Ctrl+9")
        ninePxButton.clicked.connect(self.ImgsWidget.mask_widget.ninePx)
        brushSizeGroup.addButton(ninePxButton)

        brushSizeLayout = QVBoxLayout()
        brushSizeLayout.addWidget(onePxButton)
        brushSizeLayout.addWidget(threePxButton)
        brushSizeLayout.addWidget(fivePxButton)
        brushSizeLayout.addWidget(sevenPxButton)
        brushSizeLayout.addWidget(ninePxButton)

        brushSizeGroupWidget = QWidget()
        brushSizeGroupWidget.setLayout(brushSizeLayout)

        brushSizeDockWidget = QDockWidget()
        brushSizeDockWidget.setWidget(brushSizeGroupWidget)
        self.addDockWidget(Qt.LeftDockWidgetArea, brushSizeDockWidget)

        brushColorGroup = QButtonGroup()
        # whiteButton = QRadioButton("White")
        # whiteButton.setShortcut("Ctrl+W")
        # whiteButton.clicked.connect(self.ImgsWidget.mask_widget.whiteColor)
        # brushColorGroup.addButton(whiteButton)

        cellButton = QRadioButton("Cell")
        cellButton.setShortcut("Ctrl+C")
        cellButton.clicked.connect(self.ImgsWidget.mask_widget.redColor)
        brushColorGroup.addButton(cellButton)

        notCellButton = QRadioButton("Not cell")
        notCellButton.setShortcut("Ctrl+N")
        notCellButton.clicked.connect(self.ImgsWidget.mask_widget.blackColor)
        brushColorGroup.addButton(notCellButton)

        resetButton = QPushButton("Reset mask")
        resetButton.setShortcut("Ctrl+R")
        resetButton.clicked.connect(self.ImgsWidget.mask_widget.reset)

        clearButton = QPushButton("Clear mask")
        clearButton.clicked.connect(self.ImgsWidget.mask_widget.clear)

        brushColorLayout = QVBoxLayout()
        # brushColorLayout.addWidget(whiteButton)
        brushColorLayout.addWidget(cellButton)
        brushColorLayout.addWidget(notCellButton)
        brushColorLayout.addWidget(resetButton)
        brushColorLayout.addWidget(clearButton)

        brushColorGroupWidget = QWidget()
        brushColorGroupWidget.setLayout(brushColorLayout)

        brushColorDockWidget = QDockWidget()
        brushColorDockWidget.setWidget(brushColorGroupWidget)
        self.addDockWidget(Qt.LeftDockWidgetArea, brushColorDockWidget)

        advanceFrameButton = QPushButton("Next frame")
        advanceFrameButton.setShortcut("Ctrl+F")
        advanceFrameButton.clicked.connect(self.ImgsWidget.mask_widget.next_frame)
        advanceFrameButton.clicked.connect(self.ImgsWidget.img_widget.next_frame)

        priorFrameButton = QPushButton("Prior frame")
        priorFrameButton.clicked.connect(self.ImgsWidget.mask_widget.prior_frame)
        priorFrameButton.clicked.connect(self.ImgsWidget.img_widget.prior_frame)

        advancePeakButton = QPushButton("Next peak")
        advancePeakButton.setShortcut("Ctrl+P")
        advancePeakButton.clicked.connect(self.ImgsWidget.mask_widget.next_peak)
        advancePeakButton.clicked.connect(self.ImgsWidget.img_widget.next_peak)

        priorPeakButton = QPushButton("Prior peak")
        priorPeakButton.clicked.connect(self.ImgsWidget.mask_widget.prior_peak)
        priorPeakButton.clicked.connect(self.ImgsWidget.img_widget.prior_peak)

        advanceFOVButton = QPushButton("Next FOV")
        advanceFOVButton.clicked.connect(self.ImgsWidget.mask_widget.next_fov)
        advanceFOVButton.clicked.connect(self.ImgsWidget.img_widget.next_fov)

        priorFOVButton = QPushButton("Prior FOV")
        priorFOVButton.clicked.connect(self.ImgsWidget.mask_widget.prior_fov)
        priorFOVButton.clicked.connect(self.ImgsWidget.img_widget.prior_fov)

        saveAndNextButton = QPushButton("Save and next frame")
        saveAndNextButton.setShortcut("Ctrl+S")
        saveAndNextButton.clicked.connect(self.ImgsWidget.mask_widget.buttonSave)
        saveAndNextButton.clicked.connect(self.ImgsWidget.img_widget.buttonSave)
        saveAndNextButton.clicked.connect(self.ImgsWidget.mask_widget.next_frame)
        saveAndNextButton.clicked.connect(self.ImgsWidget.img_widget.next_frame)

        fileAdvanceLayout = QVBoxLayout()
        fileAdvanceLayout.addWidget(advanceFrameButton)
        fileAdvanceLayout.addWidget(priorFrameButton)
        fileAdvanceLayout.addWidget(advancePeakButton)
        fileAdvanceLayout.addWidget(priorPeakButton)
        fileAdvanceLayout.addWidget(advanceFOVButton)
        fileAdvanceLayout.addWidget(priorFOVButton)
        fileAdvanceLayout.addWidget(saveAndNextButton)

        fileAdvanceGroupWidget = QWidget()
        fileAdvanceGroupWidget.setLayout(fileAdvanceLayout)

        fileAdvanceDockWidget = QDockWidget()
        fileAdvanceDockWidget.setWidget(fileAdvanceGroupWidget)
        self.addDockWidget(Qt.RightDockWidgetArea, fileAdvanceDockWidget)

    #     setFrameInputWidget = FrameSetter()
    #     setPeakInputWidget = PeakSetter()
    #     setFOVInputWidget = FOVSetter()
    #
    #     setFOVPeakFrameInputLayout = QVBoxLayout()
    #     setFOVPeakFrameInputLayout.addWidget(setFrameInputWidget)
    #     setFOVPeakFrameInputLayout.addWidget(setPeakInputWidget)
    #     setFOVPeakFrameInputLayout.addWidget(setFOVInputWidget)
    #
    #     setFOVPeakFrameWidget = QWidget()
    #     setFOVPeakFrameWidget.setLayout(setFOVPeakFrameInputLayout)
    #
    #     setFOVPeakFrameDockWidget = QDockWidget()
    #     setFOVPeakFrameDockWidget.setWidget(setFrameInputWidget)
    #     self.addDockWidget(Qt.RightDockWidgetArea, setFOVPeakFrameDockWidget)
    #
    # def FrameSetter(self):
    #     i, okPressed = QInputDialog.getInt(self, "Jump to frame", "Frame index (0-indexed):", 0, 0)
    #     if okPressed:
    #         self.

class OverlayImgsWidget(QWidget):

        def __init__(self,parent,imgPaths,fov_id_list,training_dir):
                super(OverlayImgsWidget, self).__init__(parent)

                self.imgLayout = QGridLayout()

                mask_dir = os.path.join(training_dir, 'masks/cells')
                image_dir = os.path.join(training_dir, 'images/cells')

                # self.maskImgPaths = [paths[1] for paths in imgPaths]
                # self.phaseImgPaths = [paths[0] for paths in imgPaths]

                names = ['image','mask'] # image has to be after mask to place image on top
                positions = [(0,0),(0,0)]

                for name,position in zip(names,positions):
                        if name == '':
                                continue
                        if name == 'mask':
                                test_key = [k for k in imgPaths.keys()][0]
                                if not imgPaths[test_key][0][1] is None:
                                        self.mask_widget = MaskTransparencyWidget(self, imgPaths=imgPaths, fov_id_list=fov_id_list, mask_dir=mask_dir)
                                        self.imgLayout.addWidget(self.mask_widget, *position)
                                else:
                                        # Write some code to make a blank MaskTransparencyWidget
                                        #######################################################################################################################
                                        # NOTE: write the blank widget code!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                                        #######################################################################################################################
                                        self.mask_widget = BlankMaskTransparencyWidget(self, fov_id_list=fov_id_list, mask_dir=mask_dir)

                        elif name == 'image':
                                self.img_widget = PhaseWidget(self, imgPaths=imgPaths, fov_id_list=fov_id_list, image_dir=image_dir)
                                self.imgLayout.addWidget(self.img_widget, *position)

                self.setLayout(self.imgLayout)

class MaskTransparencyWidget(QWidget):

        def __init__(self,parent,imgPaths,fov_id_list,mask_dir):
                super(MaskTransparencyWidget, self).__init__(parent)

                self.mask_dir = mask_dir
                self.frameIndex = 0

                self.imgPaths = imgPaths
                self.fov_id_list = fov_id_list
                self.fovIndex = 0
                self.fov_id = self.fov_id_list[self.fovIndex]

                self.imgIndex = 0
                self.maskImgPath = self.imgPaths[self.fov_id][self.imgIndex][1]

                # TO DO: check if re-annotated mask exists in training_dir, and present that instead of original mask
                #        make indicator appear if we're re-editing the mask again.
                experiment_name = params['experiment_name']
                original_file_name = self.maskImgPath
                pat = re.compile(r'.+(xy\d{3,4})_(p\d{3,4})_.+') # supports 3- or 4-digit naming
                mat = pat.match(original_file_name)
                fovID = mat.groups()[0]
                peakID = mat.groups()[1]
                fileBaseName = '{}_{}_{}_t{:0=4}.tif'.format(experiment_name, fovID, peakID, self.frameIndex+1)
                savePath = os.path.join(self.mask_dir,fileBaseName)

                self.maskStack = io.imread(self.maskImgPath)
                if os.path.isfile(savePath):
                    print('Re-annotated mask exists in training directory. Loading it.')
                    # add widget to express whether this mask is one you already re-annotated
                    # print(self.frameIndex)
                    self.maskStack[self.frameIndex,:,:] = io.imread(savePath)
                    overwriteSegFile = True
                else:
                    overwriteSegFile = False

                img = self.maskStack[self.frameIndex,:,:]
                img[img>0] = 255
                self.RGBImg = color.gray2rgb(img).astype('uint8')
                self.RGBImg[:,:,1:] = 0 # set GB channels to 0 to make the transarency mask red
                alphaFloat = 0.15
                alphaArray = np.zeros(img.shape, dtype='uint8')
                alphaArray = np.expand_dims(alphaArray, -1)
                self.alpha = int(255*alphaFloat)
                alphaArray[...] = self.alpha
                self.RGBAImg = np.append(self.RGBImg, alphaArray, axis=-1)

                self.originalHeight, self.originalWidth, self.originalChannelNumber = self.RGBAImg.shape
                self.maskQimage = QImage(self.RGBAImg, self.originalWidth, self.originalHeight, self.RGBAImg.strides[0], QImage.Format_RGBA8888).scaled(1024, 1024, aspectRatioMode=Qt.KeepAspectRatio)
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
                        rr,cc = draw.disk((event.y(), event.x()), self.brushSize)
                        for pix in zip(rr,cc):
                                rowIndex = pix[0]
                                colIndex = pix[1]
                                self.maskQimage.setPixel(colIndex, rowIndex, self.brushColor.rgba())

                        self.maskQpixmap = QPixmap(self.maskQimage)
                        self.label.setPixmap(self.maskQpixmap)
                        self.lastPoint = event.pos()
                        self.update()

        def mouseReleaseEvent(self, event):
                if event.button == Qt.LeftButton:
                        self.drawing = False

        def save(self):
                filePath, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG(*.png);;JPEG(*.jpg *.jpeg);;TIFF(*.tif *.tiff);;ALL FILES(*.*)")
                if filePath == "":
                        return
                saveImg = self.maskQimage.convertToFormat(QImage.Format_Grayscale8).scaled(self.originalWidth,self.originalHeight)
                qimgHeight = saveImg.height()
                qimgWidth = saveImg.width()

                for rowIndex in range(qimgHeight):

                        for colIndex in range(qimgWidth):
                                pixVal = qGray(saveImg.pixel(colIndex,rowIndex))
                                if pixVal > 0:
                                        saveImg.setPixelColor(colIndex,rowIndex,QColor(1,1,1))
                                        pixVal = qGray(saveImg.pixel(colIndex, rowIndex))

                saveImg.save(filePath)

        def buttonSave(self):
                experiment_name = params['experiment_name']
                original_file_name = self.maskImgPath
                pat = re.compile(r'.+(xy\d{3,4})_(p\d{3,4})_.+') # supports 3- or 4-digit naming
                mat = pat.match(original_file_name)
                fovID = mat.groups()[0]
                peakID = mat.groups()[1]
                fileBaseName = '{}_{}_{}_t{:0=4}.tif'.format(experiment_name, fovID, peakID, self.frameIndex+1)
                savePath = os.path.join(self.mask_dir,fileBaseName)
                # labelSavePath = os.path.join(params['seg_dir'],fileBaseName)
                print("Saved binary mask image as: ", savePath)

                if not os.path.isdir(self.mask_dir):
                    os.makedirs(self.mask_dir)

                # saveImg = self.maskQimage.convertToFormat(QImage.Format_Grayscale8)

                # This was bugging out and making the image not the same size as it started
                # saveImg = self.maskQimage.convertToFormat(QImage.Format_Grayscale8).scaled(self.originalWidth,self.originalHeight,aspectRatioMode=Qt.KeepAspectRatio)

                saveImg = self.maskQimage.convertToFormat(QImage.Format_Grayscale8).scaled(self.originalWidth,self.originalHeight)

                qimgHeight = saveImg.height()
                qimgWidth = saveImg.width()

                print(self.originalHeight, self.originalWidth, qimgHeight, qimgWidth)

                saveArr = np.zeros((qimgHeight,qimgWidth),dtype='uint8')
                for rowIndex in range(qimgHeight):

                        for colIndex in range(qimgWidth):
                                pixVal = qGray(saveImg.pixel(colIndex,rowIndex))
                                if pixVal > 0:
                                        saveArr[rowIndex,colIndex] = 1

                io.imsave(savePath, saveArr)
                # labelArr = measure.label(saveArr, connectivity=1)
                # labelArr = labelArr.astype('uint8')

                # print(labelSavePath)
                # io.imsave(labelSavePath,labelArr)


        def reset(self):
                self.maskQimage = QImage(self.RGBAImg, self.originalWidth, self.originalHeight, self.RGBAImg.strides[0], QImage.Format_RGBA8888).scaled(1024, 1024, aspectRatioMode=Qt.KeepAspectRatio)
                self.maskQpixmap = QPixmap(self.maskQimage)
                self.label.setPixmap(self.maskQpixmap)
                self.update()

        def clear(self):
                self.imgFill = QColor(0, 0, 0, self.alpha)
                self.maskQimage = QImage(self.RGBAImg, self.originalWidth, self.originalHeight, self.RGBAImg.strides[0], QImage.Format_RGBA8888).scaled(1024, 1024, aspectRatioMode=Qt.KeepAspectRatio)
                self.maskQimage.fill(self.imgFill)
                self.maskQpixmap = QPixmap(self.maskQimage)
                self.label.setPixmap(self.maskQpixmap)
                self.update()

        def onePx(self):
                self.brushSize = 1

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

        # def setFOVPeakFrameIndex(self,frame_index,peak_index,fov_id):
        #     self.frameIndex = frame_index
        #     self.fov_id = fov_id
        #     self.imgIndex = peak_id


        def setImg(self, img):
                img[img>0] = 255
                self.RGBImg = color.gray2rgb(img).astype('uint8')
                self.RGBImg[:,:,1:] = 0 # set green and blue channels to 0 to make the transarency mask red
                alphaFloat = 0.25
                alphaArray = np.zeros(img.shape, dtype='uint8')
                alphaArray = np.expand_dims(alphaArray, -1)
                self.alpha = int(255*alphaFloat)
                alphaArray[...] = self.alpha
                self.RGBAImg = np.append(self.RGBImg, alphaArray, axis=-1)

                self.originalHeight, self.originalWidth, self.originalChannelNumber = self.RGBAImg.shape
                self.maskQimage = QImage(self.RGBAImg, self.originalWidth, self.originalHeight, self.RGBAImg.strides[0], QImage.Format_RGBA8888).scaled(1024, 1024, aspectRatioMode=Qt.KeepAspectRatio)
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
                    savePath = os.path.join(self.mask_dir,fileBaseName)

                    if os.path.isfile(savePath):
                        print('Re-annotated mask exists in training directory. Loading it.')
                        # add widget to express whether this mask is one you already re-annotated
                        self.maskStack[self.frameIndex,:,:] = io.imread(savePath)
                    img = self.maskStack[self.frameIndex,:,:]
                except IndexError:
                    sys.exit("You've already edited the last frame's mask. Write in functionality to increment to next peak_id now!")

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
                    savePath = os.path.join(self.mask_dir,fileBaseName)

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
                savePath = os.path.join(self.mask_dir,fileBaseName)

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
                savePath = os.path.join(self.mask_dir,fileBaseName)

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
                savePath = os.path.join(self.mask_dir,fileBaseName)

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
                savePath = os.path.join(self.mask_dir,fileBaseName)

                if os.path.isfile(savePath):
                    print('Re-annotated mask exists in training directory. Loading it.')
                    # add widget to express whether this mask is one you already re-annotated
                    self.maskStack[self.frameIndex,:,:] = io.imread(savePath)
                img = self.maskStack[self.frameIndex,:,:]
                self.setImg(img)

class PhaseWidget(QWidget):

        def __init__(self, parent,imgPaths,fov_id_list,image_dir):#,frame_index,peak_id,fov_id):
                super(PhaseWidget, self).__init__(parent)

                self.image_dir = image_dir

                self.imgPaths = imgPaths
                self.fov_id_list = fov_id_list
                self.fovIndex = 0
                self.fov_id = self.fov_id_list[self.fovIndex]

                self.imgIndex = 0
                self.phaseImgPath = self.imgPaths[self.fov_id][self.imgIndex][0]
                self.phaseStack = io.imread(self.phaseImgPath)

                self.frameIndex = 0
                self.img = self.phaseStack[self.frameIndex,:,:]
                # self.originalImgMax = np.max(self.img)
                self.originalImgMax = np.max(self.phaseStack)
                originalRGBImg = color.gray2rgb(self.img/2**16*2**8).astype('uint8')
                self.originalPhaseQImage = QImage(originalRGBImg, originalRGBImg.shape[1], originalRGBImg.shape[0], originalRGBImg.strides[0], QImage.Format_RGB888)

                rescaledImg = self.img/self.originalImgMax*255
                RGBImg = color.gray2rgb(rescaledImg).astype('uint8')
                self.originalHeight, self.originalWidth, self.originalChannelNumber = RGBImg.shape
                self.phaseQimage = QImage(RGBImg, RGBImg.shape[1], RGBImg.shape[0], RGBImg.strides[0], QImage.Format_RGB888).scaled(1024, 1024, aspectRatioMode=Qt.KeepAspectRatio)
                self.phaseQpixmap = QPixmap(self.phaseQimage)

                self.label = QLabel(self)
                self.label.setPixmap(self.phaseQpixmap)

        def setImg(self):
                # self.originalImgMax = np.max(self.img)
                originalRGBImg = color.gray2rgb(self.img/2**16*2**8).astype('uint8')
                self.originalPhaseQImage = QImage(originalRGBImg, originalRGBImg.shape[1], originalRGBImg.shape[0], originalRGBImg.strides[0], QImage.Format_RGB888)

                # rescaledImg = self.img/np.max(self.img)*255
                rescaledImg = self.img/self.originalImgMax*255
                RGBImg = color.gray2rgb(rescaledImg).astype('uint8')
                self.phaseQimage = QImage(RGBImg, RGBImg.shape[1], RGBImg.shape[0], RGBImg.strides[0], QImage.Format_RGB888).scaled(1024, 1024, aspectRatioMode=Qt.KeepAspectRatio)
                self.phaseQpixmap = QPixmap(self.phaseQimage)
                self.label.setPixmap(self.phaseQpixmap)

        def next_frame(self):
                self.frameIndex += 1

                try:
                        self.img = self.phaseStack[self.frameIndex,:,:]
                except IndexError:
                        sys.exit("You've already edited the last frame's mask. Write in functionality to increment to next peak_id now!")
                self.setImg()

        def prior_frame(self):
                self.frameIndex -= 1

                try:
                        self.img = self.phaseStack[self.frameIndex,:,:]
                except IndexError:
                        sys.exit("You've already edited the last frame's mask. Write in functionality to increment to next peak_id now!")
                self.setImg()

        def next_peak(self):

                self.imgIndex += 1
                self.phaseImgPath = self.imgPaths[self.fov_id][self.imgIndex][0]
                self.phaseStack = io.imread(self.phaseImgPath)

                self.frameIndex = 0
                self.img = self.phaseStack[self.frameIndex,:,:]
                self.setImg()

        def prior_peak(self):

                self.imgIndex -= 1
                self.phaseImgPath = self.imgPaths[self.fov_id][self.imgIndex][0]
                self.phaseStack = io.imread(self.phaseImgPath)

                self.frameIndex = 0
                self.img = self.phaseStack[self.frameIndex,:,:]
                self.setImg()

        def next_fov(self):
                self.fovIndex += 1
                self.fov_id = self.fov_id_list[self.fovIndex]

                self.imgIndex = 0
                self.phaseImgPath = self.imgPaths[self.fov_id][self.imgIndex][0]
                self.phaseStack = io.imread(self.phaseImgPath)

                self.frameIndex = 0
                self.img = self.phaseStack[self.frameIndex,:,:]
                self.setImg()

        def prior_fov(self):
                self.fovIndex -= 1
                self.fov_id = self.fov_id_list[self.fovIndex]

                self.imgIndex = 0
                self.phaseImgPath = self.imgPaths[self.fov_id][self.imgIndex][0]
                self.phaseStack = io.imread(self.phaseImgPath)

                self.frameIndex = 0
                self.img = self.phaseStack[self.frameIndex,:,:]
                self.setImg()

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

                # saveImg = self.originalPhaseQImage.convertToFormat(QImage.Format_Grayscale8)
                # saveImg.save(savePath)
                io.imsave(savePath, self.img)

if __name__ == "__main__":

        imgPaths = {1:[('/home/wanglab/Users_local/Jeremy/Imaging/20190214/analysis/channels/20190214_JDW3418_xy001_p0028_c1.tif',
                     '/home/wanglab/Users_local/Jeremy/Imaging/20190214/analysis/segmented/20190214_JDW3418_xy001_p0028_seg_unet.tif'),
                    ('/home/wanglab/Users_local/Jeremy/Imaging/20190214/analysis/channels/20190214_JDW3418_xy001_p0039_c1.tif',
                     '/home/wanglab/Users_local/Jeremy/Imaging/20190214/analysis/segmented/20190214_JDW3418_xy001_p0039_seg_unet.tif')],
                    2:[('/home/wanglab/Users_local/Jeremy/Imaging/20190214/analysis/channels/20190214_JDW3418_xy001_p0127_c1.tif',
                        '/home/wanglab/Users_local/Jeremy/Imaging/20190214/analysis/segmented/20190214_JDW3418_xy001_p0127_seg_unet.tif'),
                       ('/home/wanglab/Users_local/Jeremy/Imaging/20190214/analysis/channels/20190214_JDW3418_xy001_p0104_c1.tif',
                        '/home/wanglab/Users_local/Jeremy/Imaging/20190214/analysis/segmented/20190214_JDW3418_xy001_p0104_seg_unet.tif')]}

        fov_id_list = [1,2]

        training_dir = '/home/wanglab/sandbox/pyqtpainter/commonDir'

        init_params('/home/wanglab/Users_local/Jeremy/Imaging/20190214/20190214_params_Unet.yaml')

        app = QApplication(sys.argv)
        window = Window(imgPaths=imgPaths, fov_id_list=fov_id_list, training_dir=training_dir)
        window.show()
        app.exec_()
