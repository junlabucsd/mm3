'''
APLibrary contains functions for agarPad.

Created 20151110 by jt, original code by Yonggun
Edited 20151111 by jt - moved threshold function to APPhase.py
Edited 20151209 by jt - added index function
'''

# import modules
import numpy as np
import cv2 # openCV
from skimage.util import img_as_ubyte
from scipy import ndimage
import matplotlib
import matplotlib.pyplot as plt

################################################################################
### functions
def find_contours(img, params):
    '''find_contours finds contours in a phase contrast image. Uses the openCV
    library.

    Parameters
    ----------
    img : 2D np.array
         phase contrast or bright field image
    params : dict
         dictionary with parameters from .yaml

    Returns
    -------
    contours
        list of contours
    '''

    # find all the cells using contouring
    kernel = np.ones((3,3), np.uint8)
    img_preprocess = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img8 = img_as_ubyte(img_preprocess)

    # process the imaging for contour finding
    blur = cv2.GaussianBlur(img8,(5,5),0)
    th3 = cv2.adaptiveThreshold(blur, 100, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY,121,2)
    ret2,global_otsu_inv = cv2.threshold(th3, 128, 255,
                                         cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    opening = cv2.morphologyEx(global_otsu_inv, cv2.MORPH_OPEN, kernel)
    closing = cv2.erode(opening, kernel, iterations = 1)
    closing = cv2.dilate(closing, kernel, iterations = 1)
    img_inv = closing

    # find the contours of all the cells or possible cells.
    # CHAIN_APPROX_SIMPLE saves the whole contour information in a compressed
    # way, saving memory.
    contours, hierarchy = cv2.findContours(img_inv, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE, (0,0))

    if params['debug']:
        print('Number of found contours:', len(contours))

    return contours

def filter_contours(img, contours, params):
    '''Filters out contours which are likely not cells based on the user
    defined .yaml file. Returns a shortened list of contours as well as the
    box length and width around those contours. Filters contours that are near
    the edge of the image, ones that are to large or wide, or have a strange
    aspect ratio

    Parameters
    ----------
    img : 2D np.array
         phase contrast or bright field image
    contours
        list of contours
    params : dict
         dictionary with parameters from .yaml

    Returns
    -------
    contours
        list of contours
    lth_box : np.array
        list of box widths in pixels corresponding to the contours
    wth_box : np.array
        list of box widths in pixels corresponding to the contours
    '''

    # record contours before for debug
    if params['debug']:
        con_img_before = plot_contours(img, contours)
        n_cnt_before = len(contours)

    # these numpy arrays hold the output data for the remaining cells
    lth_box = [] # crude height of cell box
    wth_box = [] # crude width of cell box

    # get rid of cells that are in the boundary of the image
    im_height, im_width = img.shape

    for h, cnt in reversed(list(enumerate(contours))):
        for tmp in cnt:
            if (tmp[0][1] < params['boundary'] or tmp[0][1] > im_height - params['boundary'] or tmp[0][0] < params['boundary'] or tmp[0][0] > im_width - params['boundary']):
                del contours[h]
                break

    # get rid of cells based on shape and size
    for h, cnt in reversed(list(enumerate(contours))):
        # calculate contour information
        rect, box, length, width, angle, area = contour_stats1(cnt)

        ### filter contours that are not cell
        # if the area is too small then delete that cell
        if area < params['min_area']:
            del contours[h]
            continue

        # if the cell has a crazy shape delete it
        rectangle = length * width
        if rectangle / area > params['rect_to_area']:
            del contours[h]
            continue

        # delete cells with weird aspect ratio
        if length / width < params['min_aspect_ratio'] or length / width > params['max_aspect_ratio']:
           del contours[h]
           continue

        # excluding short and long cells
        if length > params['con_max_l'] or length < params['con_min_l']:
            del contours[h]
            continue

        # excluding based on width limits
        if width > params['con_max_w'] or width < params['con_min_w']:
            del contours[h]
            continue

        # record height and width of cell if it made the cut
        lth_box = np.append(lth_box, length)
        wth_box = np.append(wth_box, width)

    ### end loop throwing out contours

    # print number of filtered contours per image
    if params['debug']:
        con_img_after = plot_contours(img, contours)
        print('Number of found cells after initial filtering:', len(contours))
        plt.figure(figsize=(10,5), dpi=100)
        plt.subplot(121)
        plt.title('Before filtering, number of contours = %d' % n_cnt_before)
        plt.imshow(con_img_before, cmap=plt.cm.gray, interpolation='nearest')
        plt.subplot(122)
        plt.title('After filtering, number of contours = %d' % len(contours))
        plt.imshow(con_img_after, cmap=plt.cm.gray, interpolation='nearest')
        plt.show()

    return contours, lth_box, wth_box

def index_cells(contours, current_cellN):
    '''Returns a list of integers for how many cells there are in contours'''
    ind_pc = np.array(range(current_cellN+1,
                      len(contours)+current_cellN+1)).astype(int)
    return ind_pc

def contour_area(contours):
    '''Returns a list of contour areas in pixels'''
    con_area = np.zeros(len(contours))
    for h, cnt in enumerate(contours):
        con_area[h] = cv2.contourArea(cnt)
    return con_area

def plot_contours(img, contours):
    '''Draws contours on an image. Usually the image should be the phase
    contrast image and contours are a list of correspond contours. Returns the
    new compiled image, which you still need to plot separately.'''
    contour_overlay = (img*255).copy() # recale image so the colors work
    cv2.drawContours(contour_overlay, contours, -1, 255, 2)
    return contour_overlay

def contour_stats1(contour):
    '''Takes a single contour and returns the box information around the
    contour, rough lenght and width, angle (rotated so the cell is sitting hot
    dog), and the area'''

    # find shape around cell
    rect = cv2.minAreaRect(contour) # create rectangle around cell
    box = cv2.cv.BoxPoints(rect) # turn that rect object into 4 points
    box = np.int0(box)

    # determine the angle the cell is pointed as a rough estimate of
    # length and width based on the bounding box
    if rect[1][0] > rect[1][1]:
        length = np.int0(rect[1][0])
        width = np.int0(rect[1][1])
        angle = rect[2]
    else:
        length = np.int0(rect[1][1])
        width = np.int0(rect[1][0])
        angle = rect[2] + 90 # there is a difference if this is 90 or 270

    area = cv2.contourArea(contour) # area of the cell based on contour

    return rect, box, length, width, angle, area
    
def contour_stats2(contour):
    '''Takes a single contour and returns the box information around the
    contour, rough lenght and width, angle (rotated so the cell is sitting hot
    dog), and the area'''

    # find shape around cell
    rect = cv2.minAreaRect(contour) # create rectangle around cell
    box = cv2.cv.BoxPoints(rect) # turn that rect object into 4 points
    box = np.int0(box)

    # determine the angle the cell is pointed as a rough estimate of
    # length and width based on the bounding box
    if rect[1][0] > rect[1][1]:
        length = np.int0(rect[1][0])
        width = np.int0(rect[1][1])
        angle = rect[2]
    else:
        length = np.int0(rect[1][1])
        width = np.int0(rect[1][0])
        angle = rect[2] + 270 # there is a difference if this is 90 or 270

    area = cv2.contourArea(contour) # area of the cell based on contour

    return rect, box, length, width, angle, area
    

def crop_cell_box(img, contour, len_pad, width_pad, params):
    '''Takes an image and crops out a smaller section around a cell and rotates the cell so it is hotdog. The length is the long/longitudnial/x axis, the width is the short/transverse/y axis.
    Parameters
    ----------
    img : 2D nd.array
        image to be cropped
    contour : cv2 contour object
        contour of cell
    len_pad : int
        number of pixels to pad on left and right of cell
    width_pad : int
        number of pixels to have above and below cell
    Returns
    -------
    img_cropped : 2D nd.array
        cropped image
    '''

    # flag for bugs
    error_flag = False

    # calculate contour stats
    rect, box, length, width, angle, area = contour_stats2(contour)

    # first cut out smaller section for rotating
    rotation_pad = (length / 2) + params['boundary']

    # make sure we are not at cell boundaries, and shrink crop size if so
    under_edge = min(np.int0(rect[0][1]-rotation_pad),np.int0(rect[0][1]+rotation_pad))
    over_col = img.shape[0] - np.int0(rect[0][1]+rotation_pad)
    over_row = img.shape[1] - np.int0(rect[0][0]+rotation_pad)
    over_edge_by = min(under_edge, over_col, over_row)
    if over_edge_by < 0:
        # print('must crop smaller', over_edge_by)
        rotation_pad += over_edge_by

    first_crop = img[np.int0(rect[0][1]-rotation_pad):np.int0(rect[0][1]+rotation_pad), np.int0(rect[0][0]-rotation_pad):np.int0(rect[0][0]+rotation_pad)]

    # rotate image so cell is horizontal
    try:
        rows, cols = first_crop.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
        rotation_crop = cv2.warpAffine(first_crop,M,(cols,rows),
                                   borderValue=np.median(first_crop))
    except:
        error_flag = True
        if params['debug']: print('Affine warp failed, rotating')
#        rotation_crop = ndimage.rotate(first_crop, angle, mode='constant', cval=np.median(first_crop))
#        plt.imshow(img, interpolation='nearest', cmap='gray')
        rotation_crop = ndimage.rotate(first_crop, angle, mode='constant', cval=np.median(first_crop))

    # determine pad sizes, which is a function of the cell (rectangle) size
    x_pad = length/2 + len_pad
    y_pad = width/2 + width_pad

    # pad image so it can surely be cut
    rotation_crop_padded = np.pad(rotation_crop, max(len_pad, width_pad),
                        mode='edge')

    # cut down image to specified pad (and remove black spaces after rotation)
    sz_imgR = rotation_crop_padded.shape[0] # find height (rows) of first crop
    sz_imgC = rotation_crop_padded.shape[1] # find width (columns) of first crop
    img_cropped = rotation_crop_padded[sz_imgR/2-y_pad:sz_imgR/2+y_pad,
                                sz_imgC/2-x_pad:sz_imgC/2+x_pad]

    # This plot the process. It is not on a debug flag because it happens so often it gets annoying.
    if error_flag:
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(4,1,1)
        plt.imshow(first_crop, cmap=plt.cm.gray, interpolation='nearest')
        ax.set_title('initial crop')
        ax = fig.add_subplot(4,1,2)
        plt.imshow(rotation_crop, cmap=plt.cm.gray, interpolation='nearest')
        ax.set_title('rotated image')
        ax = fig.add_subplot(4,1,3)
        plt.imshow(rotation_crop_padded, cmap=plt.cm.gray, interpolation='nearest')
        ax.set_title('padded rotated image')
        ax = fig.add_subplot(4,1,4)
        plt.imshow(img_cropped, cmap=plt.cm.gray, interpolation='nearest')
        ax.set_title('final crop used for analysis')
        plt.show()

    return img_cropped


















################################################################################
