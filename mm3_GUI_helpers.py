#! /usr/bin/env python3

from PyQt5.QtWidgets import QApplication, QMainWindow, QMenuBar, QRadioButton, QMenu, QAction, QButtonGroup, QFileDialog, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QGridLayout, QAction, QDockWidget, QPushButton
from PyQt5.QtGui import QIcon, QImage, QPainter, QPen, QPixmap, qGray, QColor
from PyQt5.QtCore import Qt, QPoint, QRectF
from skimage import io, img_as_ubyte, color, draw
import numpy as np
import sys
import re
import os

class Window(QMainWindow):

    def __init__(self, parent=None, imgPaths=None, fov_id_list=None, training_dir=None):
        super(Window, self).__init__(parent)

        top = 400
        left = 400
        width = 800
        height = 600

        icon = "icons/pain.png"

        self.setWindowTitle("Paint Application")
        self.setGeometry(top,left,width,height)
        self.setWindowIcon(QIcon(icon))

        self.ImgsWidget = OverlayImgsWidget(self, imgPaths=imgPaths, fov_id_list=fov_id_list, training_dir=training_dir)
        self.setCentralWidget(self.ImgsWidget)

        brushSizeGroup = QButtonGroup()
        threePxButton = QRadioButton("3px")
        threePxButton.clicked.connect(self.ImgsWidget.mask_widget.threePx)
        brushSizeGroup.addButton(threePxButton)

        fivePxButton = QRadioButton("5px")
        fivePxButton.clicked.connect(self.ImgsWidget.mask_widget.fivePx)
        brushSizeGroup.addButton(fivePxButton)

        sevenPxButton = QRadioButton("7px")
        sevenPxButton.clicked.connect(self.ImgsWidget.mask_widget.sevenPx)
        brushSizeGroup.addButton(sevenPxButton)

        ninePxButton = QRadioButton("9px")
        ninePxButton.clicked.connect(self.ImgsWidget.mask_widget.ninePx)
        brushSizeGroup.addButton(ninePxButton)

        brushSizeLayout = QVBoxLayout()
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
        cellButton.clicked.connect(self.ImgsWidget.mask_widget.redColor)
        brushColorGroup.addButton(cellButton)

        notCellButton = QRadioButton("Not cell")
        notCellButton.clicked.connect(self.ImgsWidget.mask_widget.blackColor)
        brushColorGroup.addButton(notCellButton)

        resetButton = QPushButton("Reset mask")
        resetButton.setShortcut("Ctrl+R")
        resetButton.clicked.connect(self.ImgsWidget.mask_widget.reset)

        clearButton = QPushButton("Clear mask")
        clearButton.setShortcut("Ctrl+C")
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

        advanceFOVButton = QPushButton("Next FOV")
        advanceFOVButton.clicked.connect(self.ImgsWidget.mask_widget.next_fov)
        advanceFOVButton.clicked.connect(self.ImgsWidget.img_widget.next_fov)

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
        fileAdvanceLayout.addWidget(advanceFOVButton)
        fileAdvanceLayout.addWidget(saveAndNextButton)

        fileAdvanceGroupWidget = QWidget()
        fileAdvanceGroupWidget.setLayout(fileAdvanceLayout)

        fileAdvanceDockWidget = QDockWidget()
        fileAdvanceDockWidget.setWidget(fileAdvanceGroupWidget)
        self.addDockWidget(Qt.RightDockWidgetArea, fileAdvanceDockWidget)


class ImgsWidget(QWidget):

        def __init__(self,parent):
                super(ImgsWidget, self).__init__(parent)

                self.imgLayout = QGridLayout()

                names = ['image','mask']
                positions = [(0,0),(0,1)]

                for name,position in zip(names,positions):
                        if name == '':
                                continue
                        if name == 'mask':
                                self.mask_widget = MaskWidget(self)
                                self.imgLayout.addWidget(self.mask_widget, *position)

                        elif name == 'image':
                                self.img_widget = PhaseWidget(self)
                                self.imgLayout.addWidget(self.img_widget, *position)

                self.setLayout(self.imgLayout)

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
                                self.mask_widget = MaskTransparencyWidget(self, imgPaths=imgPaths, fov_id_list=fov_id_list, mask_dir=mask_dir)
                                self.imgLayout.addWidget(self.mask_widget, *position)

                        elif name == 'image':
                                self.img_widget = PhaseWidget(self, imgPaths=imgPaths, fov_id_list=fov_id_list, image_dir=image_dir)
                                self.imgLayout.addWidget(self.img_widget, *position)

                self.setLayout(self.imgLayout)

class PhaseTransparencyWidget(QWidget):

        def __init__(self,parent):
                super(PhaseTransparencyWidget, self).__init__(parent)

                self.label = QLabel(self)

                img = io.imread('/home/wanglab/src/motherMachineSegger/data_from_deepLearning/commonDir/images/cells/20181011_JDW3308_StagePosition01_68_Channel1-0103.tif')
                rescaledImg = img/np.max(img)*255
                self.RGBImg = color.gray2rgb(rescaledImg).astype('uint8')
                alphaFloat = 0.75
                alphaArray = np.zeros(img.shape, dtype='uint8')
                alphaArray = np.expand_dims(alphaArray, -1)
                alpha = int(255*alphaFloat)
                alphaArray[...] = alpha
                self.RGBAImg = np.append(self.RGBImg, alphaArray, axis=-1)

                self.originalHeight, self.originalWidth, self.originalChannelNumber = self.RGBAImg.shape
                self.maskQimage = QImage(self.RGBAImg, self.originalWidth, self.originalHeight, self.RGBAImg.strides[0], QImage.Format_RGBA8888).scaled(512, 512, aspectRatioMode=Qt.KeepAspectRatio)
                self.maskQpixmap = QPixmap(self.maskQimage)
                self.label.setPixmap(self.maskQpixmap)

class MaskTransparencyWidget(QWidget):

        def __init__(self,parent,imgPaths,fov_id_list,mask_dir):
                super(MaskTransparencyWidget, self).__init__(parent)

                self.mask_dir = mask_dir

                self.imgPaths = imgPaths
                self.fov_id_list = fov_id_list
                self.fovIndex = 0
                self.fov_id = self.fov_id_list[self.fovIndex]

                self.imgIndex = 0
                self.maskImgPath = self.imgPaths[self.fov_id][self.imgIndex][1]
                self.maskStack = io.imread(self.maskImgPath)

                self.frameIndex = 0
                img = self.maskStack[self.frameIndex,:,:]
                img[img>0] = 255
                self.RGBImg = color.gray2rgb(img).astype('uint8')
                self.RGBImg[:,:,1:] = 0 # set GB channels to 0 to make the transarency mask red
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

        def save(self):
                filePath, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG(*.png);;JPEG(*.jpg *.jpeg);;TIFF(*.tif *.tiff);;ALL FILES(*.*)")
                if filePath == "":
                        return
                saveImg = self.maskQimage.convertToFormat(QImage.Format_Grayscale8).scaled(self.originalWidth,self.originalHeight,aspectRatioMode=Qt.KeepAspectRatio)
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
                experiment_name = 'test'
                original_file_name = self.maskImgPath
                peak_id_pat = re.compile(r'.+\_(p\d{4})_.+')
                mat = peak_id_pat.match(original_file_name)
                peakID = mat.groups()[0]
                fileBaseName = '{}_xy{:0=3}_{}_t{:0=4}.tif'.format(experiment_name, self.fov_id, peakID, self.frameIndex+1)
                savePath = os.path.join(self.mask_dir,fileBaseName)
                print("Saved binary mask image as: ", savePath)

                if not os.path.isdir(self.mask_dir):
                        os.makedirs(self.mask_dir)

                saveImg = self.maskQimage.convertToFormat(QImage.Format_Grayscale8).scaled(self.originalWidth,self.originalHeight,aspectRatioMode=Qt.KeepAspectRatio)
                qimgHeight = saveImg.height()
                qimgWidth = saveImg.width()

                for rowIndex in range(qimgHeight):

                        for colIndex in range(qimgWidth):
                                pixVal = qGray(saveImg.pixel(colIndex,rowIndex))
                                if pixVal > 0:
                                        saveImg.setPixelColor(colIndex,rowIndex,QColor(1,1,1))
                                        pixVal = qGray(saveImg.pixel(colIndex, rowIndex))

                saveImg.save(savePath)

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

        def next_frame(self):
                self.frameIndex += 1
                try:
                        img = self.maskStack[self.frameIndex,:,:]
                except IndexError:
                        sys.exit("You've already edited the last frame's mask. Write in functionality to increment to next peak_id now!")

                img[img>0] = 255
                self.RGBImg = color.gray2rgb(img).astype('uint8')
                self.RGBImg[:,:,1:] = 0 # set GB channels to 0 to make the transarency mask red
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

        def prior_frame(self):
                self.frameIndex -= 1
                try:
                        img = self.maskStack[self.frameIndex,:,:]
                except IndexError:
                        sys.exit("You've already edited the last frame's mask. Write in functionality to increment to next peak_id now!")

                img[img>0] = 255
                self.RGBImg = color.gray2rgb(img).astype('uint8')
                self.RGBImg[:,:,1:] = 0 # set GB channels to 0 to make the transarency mask red
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

        def next_peak(self):
                self.imgIndex += 1
                self.maskImgPath = self.imgPaths[self.fov_id][self.imgIndex][1]
                self.maskStack = io.imread(self.maskImgPath)

                self.frameIndex = 0
                img = self.maskStack[self.frameIndex,:,:]
                img[img>0] = 255
                self.RGBImg = color.gray2rgb(img).astype('uint8')
                self.RGBImg[:,:,1:] = 0 # set GB channels to 0 to make the transarency mask red
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

        def next_fov(self):
                self.fovIndex += 1
                self.fov_id = self.fov_id_list[self.fovIndex]

                self.imgIndex = 0
                self.maskImgPath = self.imgPaths[self.fov_id][self.imgIndex][1]
                self.maskStack = io.imread(self.maskImgPath)

                self.frameIndex = 0
                img = self.maskStack[self.frameIndex,:,:]
                img[img>0] = 255
                self.RGBImg = color.gray2rgb(img).astype('uint8')
                self.RGBImg[:,:,1:] = 0 # set GB channels to 0 to make the transarency mask red
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

class MaskWidget(QWidget):

        def __init__(self, parent):
                super(MaskWidget, self).__init__(parent)

                self.label = QLabel(self)

                img = io.imread('/home/wanglab/src/motherMachineSegger/data_from_deepLearning/commonDir/masks/cells/20181011_JDW3308_StagePosition01_68_Channel1-0103.tif')
                img[img>0] = 255
                self.RGBImg = color.gray2rgb(img).astype('uint8')
                self.originalHeight = self.RGBImg.shape[0]
                self.originalWidth = self.RGBImg.shape[1]
                self.maskQimage = QImage(self.RGBImg, self.originalWidth, self.originalHeight, self.RGBImg.strides[0], QImage.Format_RGB888).scaled(512, 512, aspectRatioMode=Qt.KeepAspectRatio)
                self.maskQpixmap = QPixmap(self.maskQimage)
                self.label.setPixmap(self.maskQpixmap)

                self.drawing = False
                self.brushSize = 2
                self.brushColor = Qt.black
                self.lastPoint = QPoint()

        def mousePressEvent(self, event):
                if event.button() == Qt.LeftButton:
                        self.drawing = True
                        self.lastPoint = event.pos()

        def mouseMoveEvent(self, event):
                if (event.buttons() & Qt.LeftButton) & self.drawing:
                        painter = QPainter(self.maskQimage)
                        painter.setPen(QPen(self.brushColor, self.brushSize, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                        painter.drawLine(self.lastPoint, event.pos())
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
                self.maskQimage.convertToFormat(QImage.Format_Grayscale8).scaled(self.originalWidth,self.originalHeight,aspectRatioMode=Qt.KeepAspectRatio).save(filePath)

        def clear(self):
                self.maskQimage = QImage(self.RGBImg, self.originalWidth, self.originalHeight, self.RGBImg.strides[0], QImage.Format_RGB888).scaled(512, 512, aspectRatioMode=Qt.KeepAspectRatio)
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
                self.brushColor = Qt.black

        def whiteColor(self):
                self.brushColor = Qt.white

        def redColor(self):
                self.brushColor = Qt.red

class PhaseWidget(QWidget):

        def __init__(self, parent,imgPaths,fov_id_list,image_dir):
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

        def next_frame(self):
                self.frameIndex += 1

                try:
                        img = self.phaseStack[self.frameIndex,:,:]
                except IndexError:
                        sys.exit("You've already edited the last frame's mask. Write in functionality to increment to next peak_id now!")

                rescaledImg = img/np.max(img)*255
                RGBImg = color.gray2rgb(rescaledImg).astype('uint8')
                self.phaseQimage = QImage(RGBImg, RGBImg.shape[1], RGBImg.shape[0], RGBImg.strides[0], QImage.Format_RGB888).scaled(512, 512, aspectRatioMode=Qt.KeepAspectRatio)
                self.phaseQpixmap = QPixmap(self.phaseQimage)
                self.label.setPixmap(self.phaseQpixmap)

        def prior_frame(self):
                self.frameIndex -= 1

                try:
                        img = self.phaseStack[self.frameIndex,:,:]
                except IndexError:
                        sys.exit("You've already edited the last frame's mask. Write in functionality to increment to next peak_id now!")

                rescaledImg = img/np.max(img)*255
                RGBImg = color.gray2rgb(rescaledImg).astype('uint8')
                self.phaseQimage = QImage(RGBImg, RGBImg.shape[1], RGBImg.shape[0], RGBImg.strides[0], QImage.Format_RGB888).scaled(512, 512, aspectRatioMode=Qt.KeepAspectRatio)
                self.phaseQpixmap = QPixmap(self.phaseQimage)
                self.label.setPixmap(self.phaseQpixmap)

        def next_peak(self):

                self.imgIndex += 1
                self.phaseImgPath = self.imgPaths[self.fov_id][self.imgIndex][0]
                self.phaseStack = io.imread(self.phaseImgPath)

                self.frameIndex = 0
                img = self.phaseStack[self.frameIndex,:,:]
                rescaledImg = img/np.max(img)*255
                RGBImg = color.gray2rgb(rescaledImg).astype('uint8')
                self.phaseQimage = QImage(RGBImg, RGBImg.shape[1], RGBImg.shape[0], RGBImg.strides[0], QImage.Format_RGB888).scaled(512, 512, aspectRatioMode=Qt.KeepAspectRatio)
                self.phaseQpixmap = QPixmap(self.phaseQimage)
                self.label.setPixmap(self.phaseQpixmap)

        def next_fov(self):
                self.fovIndex += 1
                self.fov_id = self.fov_id_list[self.fovIndex]

                self.imgIndex = 0
                self.phaseImgPath = self.imgPaths[self.fov_id][self.imgIndex][0]
                self.phaseStack = io.imread(self.phaseImgPath)

                self.frameIndex = 0
                img = self.phaseStack[self.frameIndex,:,:]
                rescaledImg = img/np.max(img)*255
                RGBImg = color.gray2rgb(rescaledImg).astype('uint8')
                self.phaseQimage = QImage(RGBImg, RGBImg.shape[1], RGBImg.shape[0], RGBImg.strides[0], QImage.Format_RGB888).scaled(512, 512, aspectRatioMode=Qt.KeepAspectRatio)
                self.phaseQpixmap = QPixmap(self.phaseQimage)
                self.label.setPixmap(self.phaseQpixmap)

        def buttonSave(self):
                experiment_name = 'test'
                original_file_name = self.phaseImgPath
                peak_id_pat = re.compile(r'.+\_(p\d{4})_.+')
                mat = peak_id_pat.match(original_file_name)
                peakID = mat.groups()[0]
                fileBaseName = '{}_xy{:0=3}_{}_t{:0=4}.tif'.format(experiment_name, self.fov_id, peakID, self.frameIndex+1)
                savePath = os.path.join(self.image_dir,fileBaseName)
                print("Saved phase image as: ", savePath)

                if not os.path.isdir(self.image_dir):
                        os.makedirs(self.image_dir)

                saveImg = self.originalPhaseQImage.convertToFormat(QImage.Format_Grayscale8)
                saveImg.save(savePath)


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

        app = QApplication(sys.argv)
        window = Window(imgPaths=imgPaths, fov_id_list=fov_id_list, training_dir=training_dir)
        window.show()
        app.exec()
