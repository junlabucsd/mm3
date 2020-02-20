#! /usr/bin/env python3
from __future__ import print_function, division

from PyQt5.QtWidgets import QApplication, QMainWindow, QMenuBar, QRadioButton, QMenu, QAction, QButtonGroup, QFileDialog, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QGridLayout, QAction, QDockWidget, QPushButton, QInputDialog
from PyQt5.QtGui import QIcon, QImage, QPainter, QPen, QPixmap, qGray, QColor
from PyQt5.QtCore import Qt, QPoint, QRectF
from skimage import io, img_as_ubyte, color, draw, measure
import numpy as np
import sys
import re
import os
import yaml
import multiprocessing

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
                self.originalImgMax = np.max(self.img)
                # self.originalImgMax = np.max(self.phaseStack)
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
                self.originalImgMax = np.max(self.img)
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
