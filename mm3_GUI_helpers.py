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
from PyQt5.QtGui import QIcon, QImage, QPainter, QPen, QPixmap, qGray, QColor, QBrush, QTransform
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

class FrameImgWidget(QWidget):
    def __init__(self,specs,cell_file,trace_file,params):
        super(FrameImgWidget, self).__init__()
        self.specs = specs
        self.scene = TrackItem(self.specs,cell_file,trace_file,params)
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

class FocusTrackWindow(QMainWindow):
    def __init__(self,params,cell_file_path,trace_file_path):
        # super(Window, self).__init__(cell_dir,cell_file)
        super().__init__()

        # self.setStyleSheet("background-color: gray;")

        top = 10
        left = 10
        width = 1100
        height = 700

        #self.setWindowTitle("")
        self.setGeometry(top,left,width,height)

        # load specs file
        with open(os.path.join(params['ana_dir'], 'specs.yaml'), 'r') as specs_file:
            specs = yaml.safe_load(specs_file)

        self.frames = FrameImgWidget(specs,cell_file_path,trace_file_path,params)
        #make scene the central widget
        self.setCentralWidget(self.frames)

        eventButtonGroup = QButtonGroup()

        removeButton = QPushButton("Remove trace")
        removeButton.setCheckable(True)
        removeButton.setShortcut("W")
        removeButton.setToolTip("Enter mode to remove existing trace")
        removeButton.clicked.connect(self.frames.scene.set_remove)
        eventButtonGroup.addButton(removeButton)

        # resetButton = QPushButton("Reset")
        # resetButton.setShortcut("R")
        # resetButton.setToolTip("Reset click-through")
        # resetButton.clicked.connect(self.frames.scene.reset_cc)

        # undoButton = QPushButton("Clear events")
        # undoButton.setShortcut("U")
        # undoButton.setToolTip("(U) Clear all events from peak")
        # undoButton.clicked.connect(self.frames.scene.clear_cc_events)

        overlayButton = QPushButton("Toggle overlay")
        overlayButton.setShortcut("T")
        overlayButton.setToolTip("(T) Toggle foci overlay")
        overlayButton.clicked.connect(self.frames.scene.toggle_overlay)

        saveUpdatedTracksButton = QPushButton("Save updated initiations")
        saveUpdatedTracksButton.setShortcut("S")
        saveUpdatedTracksButton.clicked.connect(self.frames.scene.save_output)

        eventButtonLayout = QVBoxLayout()

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

        fileAdvanceLayout.addWidget(removeButton)
        fileAdvanceLayout.addWidget(overlayButton)
        fileAdvanceLayout.addWidget(saveUpdatedTracksButton)

        fileAdvanceGroupWidget = QWidget()
        fileAdvanceGroupWidget.setLayout(fileAdvanceLayout)

        fileAdvanceDockWidget = QDockWidget()
        fileAdvanceDockWidget.setWidget(fileAdvanceGroupWidget)
        self.addDockWidget(Qt.LeftDockWidgetArea, fileAdvanceDockWidget)

class TrackItem(QGraphicsScene):

    def __init__(self,specs,cell_file_path,trace_file_path,params):
        # super(TrackItem, self).__init__()
        super().__init__()
        self.items = []

        #these set the size of the kymograph display in pixels in phase_imgs_and_regions
        self.y_scale = 1400
        self.x_scale = 1500

        self.specs = specs

        self.fov_id_list = [fov_id for fov_id in specs.keys()]

        self.fovIndex = 0
        self.fov_id = self.fov_id_list[self.fovIndex]

        self.peak_id_list_in_fov = [peak_id for peak_id in specs[self.fov_id].keys() if specs[self.fov_id][peak_id] == 1]

        self.peakIndex = 0
        self.peak_id = self.peak_id_list_in_fov[self.peakIndex]

        self.color = params['foci']['foci_plane']

        self.params = params

        self.setBackgroundBrush(Qt.black)

        with open(cell_file_path, 'rb') as cf:
            self.Cells = pickle.load(cf)

        with open(trace_file_path, 'rb') as tf:
            self.Traces = pickle.load(tf)

        self.Cells_by_peak = mm3_plots.organize_cells_by_channel(self.Cells,specs)

        self.cell_id_list_in_peak = [cell_id for cell_id in self.Cells_by_peak[self.fov_id][self.peak_id].keys()]

        self.init1 = False
        self.init2 = False
        self.init3 = False
        self.remove = False
        self.reset = False
        self.overlay_fl = False

        self.select = False

        self.clicks = 0

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

        img_dir = params['experiment_directory']+ params['analysis_directory'] + 'kymograph'
        img_filename = params['experiment_name'] + '_xy%03d_p%04d_%s.tif' % (self.fov_id, self.peak_id, self.color)
        with tiff.TiffFile(os.path.join(img_dir, img_filename)) as tif:
            self.fl_kymo = tif.asarray()

        self.x_px = len(self.fl_kymo[0])
        self.y_px = len(self.fl_kymo[:,0])

        if self.overlay_fl:
            overlay = self.phase_img_and_regions()
            self.addPixmap(overlay)

        ## draw inferred replication tracks
        self.label_traces()

    def toggle_reload(self):
        self.clear()
        specs = self.specs
        self.ccf = []
        self.clicks = 0
        self.items = []

        params = self.params

        self.peak_id = self.peak_id_list_in_fov[self.peakIndex]

        try:
            self.cell_id_list_in_peak = [cell_id for cell_id in self.Cells_by_peak[self.fov_id][self.peak_id].keys()]

        except KeyError:
            print('no cells in this peak')
            self.peakIndex+=1
            return

        print('FOV '+str(self.fov_id))
        print('Peak '+str(self.peak_id))

        if self.overlay_fl:
            try:
                img_dir = params['experiment_directory']+ params['analysis_directory'] + 'kymograph'
                img_filename = params['experiment_name'] + '_xy%03d_p%04d_%s.tif' % (self.fov_id, self.peak_id, self.color)
                with tiff.TiffFile(os.path.join(img_dir, img_filename)) as tif:
                    self.fl_kymo = tif.asarray()
                overlay = self.phase_img_and_regions()
                self.addPixmap(overlay)

            except:
                pass

        self.label_traces()

    def next_peak(self):
        # start by removing all current graphics items from the scene, the scene here being 'self'
        self.clear()
        specs = self.specs
        self.peakIndex += 1
        self.ccf = []
        self.clicks = 0
        self.items = []

        params = self.params

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

        print('FOV '+str(self.fov_id))
        print('Peak '+str(self.peak_id))

        if self.overlay_fl:
            try:
                img_dir = params['experiment_directory']+ params['analysis_directory'] + 'kymograph'
                img_filename = params['experiment_name'] + '_xy%03d_p%04d_%s.tif' % (self.fov_id, self.peak_id, self.color)
                with tiff.TiffFile(os.path.join(img_dir, img_filename)) as tif:
                    self.fl_kymo = tif.asarray()
                overlay = self.phase_img_and_regions()
                self.addPixmap(overlay)

            except:
                pass

        self.label_traces()

    def prior_peak(self):
        # start by removing all current graphics items from the scene, the scene here being 'self'
        self.clear()
        specs = self.specs
        params = self.params
        self.peakIndex -= 1
        self.clicks = 0
        self.items = []

        try:
            self.peak_id = self.peak_id_list_in_fov[self.peakIndex]

        except IndexError:
            print('go to previous FOV')

        print('FOV '+str(self.fov_id))
        print('Peak '+str(self.peak_id))

        if self.overlay_fl:
            try:
                img_dir = params['experiment_directory']+ params['analysis_directory'] + 'kymograph'
                img_filename = params['experiment_name'] + '_xy%03d_p%04d_%s.tif' % (self.fov_id, self.peak_id, self.color)
                with tiff.TiffFile(os.path.join(img_dir, img_filename)) as tif:
                    self.fl_kymo = tif.asarray()
                overlay = self.phase_img_and_regions()
                self.addPixmap(overlay)

            except:
                pass

        self.label_traces()

    def next_fov(self):
        self.save_output()
        self.clear()
        self.items = []

        specs = self.specs
        self.clicks = 0
        self.fovIndex += 1

        params = self.params

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

        print('FOV '+str(self.fov_id))
        print('Peak '+str(self.peak_id))

        if self.overlay_fl:
            try:
                img_dir = params['experiment_directory']+ params['analysis_directory'] + 'kymograph'
                img_filename = params['experiment_name'] + '_xy%03d_p%04d_%s.tif' % (self.fov_id, self.peak_id, self.color)
                with tiff.TiffFile(os.path.join(img_dir, img_filename)) as tif:
                    self.fl_kymo = tif.asarray()
                overlay = self.phase_img_and_regions()
                self.addPixmap(overlay)

            except:
                pass

        self.label_traces()

    def prior_fov(self):
        self.save_output()
        self.clear()

        specs = self.specs
        params = self.params
        self.ccf = []
        self.items = []
        self.clicks = 0

        # self.flag = False

        self.fovIndex -= 1
        self.fov_id = self.fov_id_list[self.fovIndex]

        self.peak_id_list_in_fov = [peak_id for peak_id in self.specs[self.fov_id].keys() if self.specs[self.fov_id][peak_id] == 1]

        self.peakIndex = 0
        self.peak_id = self.peak_id_list_in_fov[self.peakIndex]

        print('FOV '+str(self.fov_id))
        print('Peak '+str(self.peak_id))

        if self.overlay_fl:
            img_dir = params['experiment_directory']+ params['analysis_directory'] + 'kymograph'
            img_filename = params['experiment_name'] + '_xy%03d_p%04d_%s.tif' % (self.fov_id, self.peak_id, self.color)

            with tiff.TiffFile(os.path.join(img_dir, img_filename)) as tif:
                self.fl_kymo = tif.asarray()
            overlay = self.phase_img_and_regions()
            self.addPixmap(overlay)

        self.label_traces()

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

    def label_traces(self):
        try:
            cells_p = self.Cells_by_peak[self.fov_id][self.peak_id]
            traces_p = self.Traces[self.fov_id][self.peak_id]
        except KeyError:
            return
        self.divs_p = []
        self.divs_t = []

        pen = QPen()
        brush = QBrush()
        brush.setStyle(Qt.SolidPattern)
        brush.setColor(QColor("white"))

        pen.setWidth(3)
        if not self.overlay_fl:
            for cell_id, cell in cells_p.items():
                times = np.array(cell.times)*self.x_scale/self.x_px
                lengths = np.array(cell.lengths)*self.y_scale/self.y_px
                cents = np.array(cell.centroids)[:,0]*self.y_scale/self.y_px

                if cell.disp_l:
                    for t, c, l in zip(times,cents,cell.disp_l):
                        for i in range(len(l)):
                            focus = QGraphicsEllipseItem(t-3,c+l[i]* self.y_scale/self.y_px-3,6,6)
                            penColor= QColor("gray")
                            penColor.setAlphaF(1)
                            pen.setColor(penColor)
                            focus.setPen(pen)
                            #focus.setBrush(brush)
                            self.addItem(focus)

        self.tracks = {}

        for (trace_id,trace) in traces_p.items():
            x_pos = [t*self.x_scale/self.x_px for t in trace.times]
            y_pos = [p[1]*self.y_scale/self.y_px for p in trace.positions]

            painter = QPainter()
            points = [QPoint(t,y) for t,y in zip(x_pos,y_pos)]
            Color = QColor(np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
            trace_pts = []
            for i in range(len(points)-1):
                eventItem = RepLine(points[i],points[i+1], color=Color)
                self.addItem(eventItem)
                trace_pts.append(eventItem)
            self.tracks[trace_id] = trace_pts

    def clear_cc_events(self):
        #clear data for this peak
        for item in self.items:
            self.removeItem(item)
        self.items = []

        self.clicks = 0


    def mousePressEvent(self, event):
        cells_p = self.Cells_by_peak[self.fov_id][self.peak_id]
        # traces_p = self.Traces[self.fov_id][self.peak_id]

        def match_cells(init_y,init_t):
            matched_id = None
            min_dist = np.inf
            for cell_id, cell in cells_p.items():
                init_age = init_t-cell.birth_time
                try:
                    if (cell.birth_time < init_t < cell.times[-1]
                        and abs(cell.centroids[init_age][0] - init_y) < cell.lengths[init_age]/2.):
                        if abs(cell.centroids[init_age][0] - init_y) < min_dist:
                            matched_id = cell_id
                            min_dist = abs(cell.centroids[init_age][0] - init_y)
                except IndexError:
                    pass
            return(matched_id)

        if self.remove == True:
            x = event.scenePos().x()
            y = event.scenePos().y()
            min_dist = np.inf
            min_id = None
            for (trace_id,trace) in self.Traces[self.fov_id][self.peak_id].items():
                x_pos = [t*self.x_scale/self.x_px for t in trace.times]
                y_pos = [p[1]*self.y_scale/self.y_px for p in trace.positions]
                diff = [np.sqrt((x1-x)**2+(y1-y)**2) for (x1,y1) in zip(x_pos,y_pos)]
                curr_min = min(diff)
                if curr_min < min_dist:
                    min_dist = curr_min
                    min_id = trace_id

            sel_trace = self.Traces[self.fov_id][self.peak_id][min_id]
            self.sel_trace_id = min_id

            self.Traces[self.fov_id][self.peak_id].pop(self.sel_trace_id)
            for item in self.tracks[self.sel_trace_id]:
                self.removeItem(item)

        elif self.clicks == 0:

            col = "white"

            self.drawing = True

            x = event.scenePos().x()
            y = event.scenePos().y()

            self.trace_pts = []

            self.init_pos = QPoint(x,y)

            init = QGraphicsEllipseItem(x-5,y-5,10,10)
            pen = QPen()
            pen.setWidth(3)
            pen.setColor(QColor(col))
            init.setPen(pen)

            self.addItem(init)
            self.items.append(init)
            self.trace_pts.append(init)
            self.init_t = np.int64(math.ceil(x/self.x_scale*self.x_px))
            self.init_y = np.int64(math.ceil(y/self.y_scale*self.y_px))

            self.clicks +=1

        elif self.clicks == 1:
            #termination event
            col = "white"
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
            self.trace_pts.append(term)

            Color = QColor(np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
            eventItem = RepLine(self.init_pos,self.term_pos,color=Color)
            self.addItem(eventItem)
            self.trace_pts.append(eventItem)

            self.term_t = np.int64(math.ceil(x*self.x_px/self.x_scale))
            self.term_y = np.int64(math.ceil(y*self.y_px/self.y_scale))

            id_n = mm3.create_rep_id(0,self.init_y,self.init_t,self.peak_id,self.fov_id)
            cell_id_init = match_cells(self.init_y,self.init_t)
            if cell_id_init is not None:
                trace_n = mm3.ReplicationTrace(id_n,0,self.init_y,self.init_t,cell_id_init)
                cell_id_term = match_cells(self.term_y,self.term_t)

                for t in range(self.init_t,self.term_t,1):
                    yp = self.interp(t,self.init_t,self.init_y,self.term_t,self.term_y)
                    cell_id_curr = match_cells(yp,t)
                    if cell_id_curr is None:
                        cell_id_curr = cell_id_init
                    trace_n.process(None,yp,t,cell_id_curr)

                trace_n.terminate(self.term_t)
                self.Traces[self.fov_id][self.peak_id][id_n] = trace_n
                self.tracks[id_n] = self.trace_pts
                self.clicks = 0
            else:
                self.clicks = 0

    def interp(self,t, t1,i1,t2,i2):
        return i1 + (i2 - i1)/(t2 - t1) * (t - t1)

    def get_time(self, cell):
        return(cell.time)

    def set_init1(self):
        self.remove = False
        self.reset = False
        self.init1 = True
        self.init2 = False
        self.init3 = False
        self.reset = False

    def set_init2(self):
        self.remove = False
        self.reset = False
        self.init1 = False
        self.init2 = True
        self.init3 = False
        self.reset = False

    def set_init3(self):
        self.remove = False
        self.reset = False
        self.init1 = False
        self.init2 = False
        self.init3 = True
        self.reset = False

    def set_remove(self):
        if self.remove == False:
            self.remove = True
        else:
            self.remove = False

    def reset_cc(self):
        self.remove = False
        self.reset = True
        self.clicks = 0

    def toggle_overlay(self):
        if self.overlay_fl == False:
            self.overlay_fl = True
        elif self.overlay_fl == True:
            self.overlay_fl = False
        self.toggle_reload()

    def save_output(self):
        with open(os.path.join(self.params['cell_dir'], 'rep_traces_mod.pkl'), 'wb') as trace_file:
            pickle.dump(self.Traces, trace_file, protocol=pickle.HIGHEST_PROTOCOL)
        print('Saved updated tracks')

class RepLine(QGraphicsLineItem):
    # A class for helping to draw replication traces
    #  within a QGraphicsScene
    def __init__(self, firstPoint, lastPoint, color,alpha=1,width=3):
        super(RepLine, self).__init__()

        self.setFlag(QGraphicsLineItem.ItemIsSelectable, True)

        brushColor = QColor(color)
        brushSize = width
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
        brushColor.setAlphaF(alpha)
        pen.setColor(brushColor)
        pen.setWidth(brushSize)
        self.setPen(pen)
        self.setLine(line)
        self.color = QColor(color)

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
