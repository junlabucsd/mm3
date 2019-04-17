#! /usr/bin/env python3
from __future__ import print_function, division

from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsView, QRadioButton, QButtonGroup, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QAction, QDockWidget, QPushButton, QGridLayout, QGraphicsLineItem
from PyQt5.QtGui import QIcon, QImage, QPainter, QPen, QPixmap, qGray, QColor
from PyQt5.QtCore import Qt, QPoint, QRectF, QLineF
from skimage import io, img_as_ubyte, color, draw, measure
import numpy as np
import sys
import re
import os
import yaml
import multiprocessing

from matplotlib import pyplot as plt

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

class Window(QMainWindow):

    def __init__(self, parent=None, imgPaths=None, fov_id_list=None, training_dir=None):
        super(Window, self).__init__(parent)

        top = 400
        left = 400
        width = 800
        height = 600

        # icon = "icons/pain.png"

        self.setWindowTitle("You got this!")
        self.setGeometry(top,left,width,height)

        self.threeFrames = ThreeFrameImgWidget(self, imgPaths=imgPaths, fov_id_list=fov_id_list, training_dir=training_dir)
        # make scene the central widget
        self.setCentralWidget(self.threeFrames)

class ThreeFrameImgWidget(QWidget):
        # class for setting three frames side-by-side
        def __init__(self,parent,imgPaths,fov_id_list,training_dir):
                super(ThreeFrameImgWidget, self).__init__(parent)

                # self.frameLayout = QHBoxLayout()
                # add layout to scene

                # add QImages to scene (try three frames)
                self.center_frame_index = 20
                self.fov_id_list = fov_id_list

                self.fovIndex = 0
                self.fov_id = self.fov_id_list[self.fovIndex]

                self.imgIndex = 0
                labelImgPath = imgPaths[self.fov_id][self.imgIndex][1]
                labelStack = io.imread(labelImgPath)
                print(labelImgPath)

                phaseImgPath = imgPaths[self.fov_id][self.imgIndex][0]
                phaseStack = io.imread(phaseImgPath)

                #New idea: add each cell object as an item in the graphics scene at its appropriate location so that I can use
                #     the tutorial at https://doc.qt.io/qt-5/qtwidgets-graphicsview-diagramscene-example.html to help
                leftFrame = overlay_imgs_pixmap(phaseStack=phaseStack,labelStack=labelStack,frame_index=self.center_frame_index-1)
                centerFrame = overlay_imgs_pixmap(phaseStack=phaseStack,labelStack=labelStack,frame_index=self.center_frame_index)
                rightFrame = overlay_imgs_pixmap(phaseStack=phaseStack,labelStack=labelStack,frame_index=self.center_frame_index+1)

                self.scene = ThreeFrameScene(leftFrame,centerFrame,rightFrame)
                self.view = QGraphicsView(self)
                self.view.setScene(self.scene)
                # self.scene.addPixmap(leftFrame)
                # self.scene.addPixmap(centerFrame)
                # self.scene.addPixmap(rightFrame)
                #
                # xPos = 0
                # for item in self.scene.items(order=Qt.AscendingOrder):
                #         item.setPos(xPos, 0)
                #         xPos += item.pixmap().width()

                # self.view = QGraphicsView(self)
                # self.view.setScene(self.scene)


# need a way to track line and destroy/redraw it as mouse is dragged around scene


class ThreeFrameScene(QGraphicsScene):

        # add more functionality for setting event type, i.e., parent-child, migrate, death, leave frame, etc..

        def __init__(self,leftFrame, centerFrame, rightFrame):
                super(ThreeFrameScene, self).__init__()

                self.addPixmap(leftFrame)
                self.addPixmap(centerFrame)
                self.addPixmap(rightFrame)

                xPos = 0
                for item in self.items(order=Qt.AscendingOrder):
                        item.setPos(xPos, 0)
                        xPos += item.pixmap().width()

                self.brushSize = 2
                self.brushColor = QColor(0,0,0)
                self.pen = QPen()
                self.pen.setWidth(self.brushSize)
                self.lastPoint = QPoint()
                self.origPoint = QPoint()

        def mousePressEvent(self, event):
                if event.button() == Qt.LeftButton:
                        self.drawing = True
                        self.firstPoint = event.scenePos()
                        self.lastPoint = event.scenePos()

                        self.line = QGraphicsLineItem(QLineF(self.firstPoint,self.lastPoint))
                        self.line.setPen(self.pen)
                        self.addItem(self.line)
                        # print(event.scenePos())

        def mouseMoveEvent(self, event):
                if (event.buttons() & Qt.LeftButton) & self.drawing:

                        self.lastPoint = event.scenePos()
                        self.removeItem(self.line)
                        self.line = QGraphicsLineItem(QLineF(self.firstPoint,self.lastPoint))
                        self.line.setPen(self.pen)
                        self.addItem(self.line)

        def mouseReleaseEvent(self, event):
                if event.button() == Qt.LeftButton:
                        # print(event.scenePos())
                        self.lastPoint = event.scenePos()
                        self.removeItem(self.line)
                        self.line = QGraphicsLineItem(QLineF(self.firstPoint,self.lastPoint))
                        self.line.setPen(self.pen)
                        self.addItem(self.line)
                        self.drawing = False

#
# class OverlayImgsPixmap(QWidget):
#
#         def __init__(self,imgPaths,fov_id_list,training_dir,frame_index):
#                 super(OverlayImgsPixmap, self).__init__()
#
#                 self.imgLayout = QGridLayout()
#                 # print(imgPaths)
#
#                 # self.maskImgPaths = [paths[1] for paths in imgPaths]
#                 # self.phaseImgPaths = [paths[0] for paths in imgPaths]
#
#                 label_dir = os.path.join(training_dir, 'masks/cells')
#                 image_dir = os.path.join(training_dir, 'images/cells')
#
#                 self.image_dir = image_dir
#                 self.label_dir = label_dir
#                 self.frameIndex = frame_index
#
#                 self.imgPaths = imgPaths
#                 self.fov_id_list = fov_id_list
#                 self.fovIndex = 0
#                 self.fov_id = self.fov_id_list[self.fovIndex]
#
#                 self.imgIndex = 0
#                 self.labelImgPath = self.imgPaths[self.fov_id][self.imgIndex][1]
#
#                 self.phaseImgPath = self.imgPaths[self.fov_id][self.imgIndex][0]
#                 self.phaseStack = io.imread(self.phaseImgPath)
#
#                 # print(self.labelImgPath)
#
#                 # TO DO: check if re-annotated mask exists in training_dir, and present that instead of original mask
#                 #        make indicator appear if we're re-editing the mask again.
#                 experiment_name = params['experiment_name']
#                 original_file_name = self.labelImgPath
#                 pat = re.compile(r'.+(xy\d{3,4})_(p\d{3,4})_.+') # supports 3- or 4-digit naming
#                 mat = pat.match(original_file_name)
#                 fovID = mat.groups()[0]
#                 peakID = mat.groups()[1]
#                 fileBaseName = '{}_{}_{}_t{:0=4}.tif'.format(experiment_name, fovID, peakID, self.frameIndex+1)
#                 savePath = os.path.join(self.label_dir,fileBaseName)
#
#                 self.labelStack = io.imread(self.labelImgPath)
#                 if os.path.isfile(savePath):
#                     print('Re-annotated mask exists in training directory. Loading it.')
#                     # add widget to express whether this mask is one you already re-annotated
#                     self.labelStack[self.frameIndex,:,:] = io.imread(savePath)
#                     overwriteSegFile = True
#                 else:
#                     overwriteSegFile = False
#
#                 maskImg = self.labelStack[self.frameIndex,:,:]
#                 phaseImg = self.phaseStack[self.frameIndex,:,:]
#                 self.originalImgMax = np.max(phaseImg)
#                 phaseImg = phaseImg/self.originalImgMax
#
#                 RGBImg = color.label2rgb(maskImg, phaseImg, alpha=0.25, bg_label=0)
#                 self.RGBImg = (RGBImg*255).astype('uint8')
#
#                 self.originalHeight, self.originalWidth, self.originalChannelNumber = self.RGBImg.shape
#                 self.maskQimage = QImage(self.RGBImg, self.originalWidth, self.originalHeight, self.RGBImg.strides[0], QImage.Format_RGB888).scaled(512, 512, aspectRatioMode=Qt.KeepAspectRatio)
#                 self.maskQpixmap = QPixmap(self.maskQimage)
#
#                 self.label = QLabel(self)
#                 self.label.setPixmap(self.maskQpixmap)
#
#                 # names = ['image','mask'] # image has to be after mask to place image on top
#                 # positions = [(0,0),(0,0)]
#                 #
#                 # for name,position in zip(names,positions):
#                 #         if name == '':
#                 #                 continue
#                 #         if name == 'mask':
#                 #                 self.mask_widget = LabelTransparencyWidget(imgPaths=imgPaths, fov_id_list=fov_id_list, label_dir=label_dir, frame_index=frame_index)
#                 #                 self.imgLayout.addWidget(self.mask_widget, *position)
#                 #
#                 #         elif name == 'image':
#                 #                 self.img_widget = PhaseWidget(imgPaths=imgPaths, fov_id_list=fov_id_list, image_dir=image_dir, frame_index=frame_index)
#                 #                 self.imgLayout.addWidget(self.img_widget, *position)
#                 #
#                 # self.setLayout(self.imgLayout)

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

    imgPaths = {1:[('/home/wanglab/sandbox/trackingGUI/testset1/analysis_20190312/channels/testset1_xy001_p0033_c1.tif',
                 '/home/wanglab/sandbox/trackingGUI/testset1/analysis_20190312/segmented/testset1_xy001_p0033_seg_unet.tif'),
                ('/home/wanglab/sandbox/trackingGUI/testset1/analysis_20190312/channels/testset1_xy001_p0077_c1.tif',
                 '/home/wanglab/sandbox/trackingGUI/testset1/analysis_20190312/channels/testset1_xy001_p0077_seg_unet.tif')],
                2:[('/home/wanglab/sandbox/trackingGUI/testset1/analysis_20190312/channels/testset1_xy002_p0078_c1.tif',
                    '/home/wanglab/sandbox/trackingGUI/testset1/analysis_20190312/segmented/testset1_xy002_p0078_seg_unet.tif'),
                   ('/home/wanglab/sandbox/trackingGUI/testset1/analysis_20190312/channels/testset1_xy002_p0121_c1.tif',
                    '/home/wanglab/sandbox/trackingGUI/testset1/analysis_20190312/segmented/testset1_xy002_p0121_seg_unet.tif')]}

    fov_id_list = [1,2]

    training_dir = '/home/wanglab/sandbox/trackingGUI'

    init_params('/home/wanglab/sandbox/trackingGUI/testset1/analysis_20190312/testset1_params_20190312.yaml')

    app = QApplication(sys.argv)
    window = Window(imgPaths=imgPaths, fov_id_list=fov_id_list, training_dir=training_dir)
    window.show()
    app.exec_()
