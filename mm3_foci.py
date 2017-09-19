'''
APFoci contains functions for foci analysis for agarPad

Created 20151210 by jt, original code by Yonggun/Fangwei
Edited 20151217 by jt, added laplacian/difference of gaussian method
'''

# import modules
import numpy as np
from numpy import unravel_index
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import cv2 # openCV
from scipy import ndimage, optimize
from scipy.optimize import leastsq
import scipy.ndimage.filters as filters
from skimage.feature import blob_dog, blob_log

################################################################################
### functions
def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*np.exp(-(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments
    width_x and width_y are 2*sigma x and sigma y of the guassian
    """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y

def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit
    if params are not provided, they are calculated from the moments
    params should be (height, x, y, width_x, width_y)"""
    gparams = moments(data) # create guess parameters.
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) -data)
    p, success = optimize.leastsq(errorfunction, gparams)
    return p

def foci_lap(img, img_foci, bbox, orientation, centroid, params):
    '''foci_dog finds foci using a laplacian convolution then fits a 2D
    Gaussian.

    The returned information are the parameters of this Gaussian.
    All the information is returned in the form of np.arrays which are the
    length of the number of found foci across all cells in the image.

    Parameters
    ----------
    img : 2D np.array
        phase contrast or bright field image
    img_foci : 2D np.array
        fluorescent image with foci hopefully
    contours
        list of contours
    params : dict
         dictionary with parameters from .yaml

    Returns
    -------
    disp_l : 1D np.array
        displacement on long axis, in um, of a foci from the center of the cell
    disp_w : 1D np.array
        displacement on short axis, in um, of a foci from the center of the cell
    foci_wx : 1D np.array
        width of foxi in long axis direction in pixels
    foci_wy : 1D np.array
        width of foci in sort axis direction in pixels
    foci_h : 1D np.array
        height of foci (intensity value of fluorecent image at Gaussian peak)
    '''

    # declare arrays which will hold foci data
    disp_l = [] # displacement in length of foci from cell center
    disp_w = [] # displacement in width of foci from cell center
    foci_h4 = [] # foci total amount (from raw image)

    # define parameters for foci finding
    minsig = params['foci_log_minsig']
    maxsig = params['foci_log_maxsig']
    thresh = params['foci_log_thresh']
    peak_med_ratio = params['foci_log_peak_med_ratio']

    c = img_foci
    c_pc = img

    # calculate median cell intensity. Used to filter foci
    int_mask = np.zeros(img_foci.shape, np.uint8)
    avg_int = cv2.mean(img_foci, mask = int_mask)
    avg_int = avg_int[0]

    # transform image before foci detection?
    cc = c
    c_subtract_gaus = cc
    c_subtract_gaus[c_subtract_gaus > 10000] = 0

    # find blobs using difference of gaussian
    over_lap = .95 # if two blobs overlap by more than this fraction, smaller blob is cut
    numsig = maxsig - minsig + 1 # number of division to consider (height of z cube) set this heigh so it considers all pixels
    blobs = blob_log(c_subtract_gaus, min_sigma=minsig, max_sigma=maxsig, overlap=over_lap, num_sigma=numsig, threshold=thresh)

    # these will hold information abou foci position temporarily
    x, y, r = [], [], []
    xx, yy, xxw, yyw = [], [], [], []

    # loop through each potenial foci
    for blob in blobs:
        yloc, xloc, sig = blob # x location, y location, and sigma of gaus

        if yloc > np.int16(bbox[0]) and yloc < np.int16(bbox[2]) and xloc > np.int16(bbox[1]) and xloc < np.int16(bbox[3]):
            radius = np.ceil(np.sqrt(2)*sig)
            x.append(xloc) # for plotting
            y.append(yloc) # for plotting
            r.append(radius)

            sz_fit = radius # increase the size around the foci for gaussian fitting

        #        #remove blob if not in cell box
        #        if (xloc < sz_imgC[1]/2-length/2 or xloc > sz_imgC[1]/2+length/2 or
        #            yloc < sz_imgC[0]/2-width/2 or yloc > sz_imgC[0]/2+width/2):
        #            if params['debug_foci']: print('blob not in cell area')
        #            continue

            # cut out a small image from origincal image to fit gaussian
            gfit_area = cc[yloc-sz_fit:yloc+sz_fit, xloc-sz_fit:xloc+sz_fit]
            gfit_rows, gfit_cols = gfit_area.shape

            gfit_area_0 = c[max(0,yloc-1*sz_fit):min(c.shape[0],yloc+1*sz_fit), max(0,xloc-1*sz_fit):min(c.shape[1],xloc+1*sz_fit)]

            # fit gaussian to proposed foci in small box
            p = fitgaussian(gfit_area)
            (peak, xc, yc, width_x, width_y) = p


            if xc <= 0 or xc >= gfit_cols or yc <= 0 or yc >= gfit_rows:
                if params['debug_foci']: print('throw out foci (gaus fit not in gfit_area)')
                continue
            elif peak/avg_int < peak_med_ratio:
                if params['debug_foci']: print('peak does not pass height test')
                continue
            else:
                # find x an y position
                xxx = xloc - sz_fit + xc
                yyy = yloc - sz_fit + yc
                xx = np.append(xx, xxx) # for plotting
                yy = np.append(yy, yyy) # for plotting
                xxw = np.append(xxw, width_x) # for plotting
                yyw = np.append(yyw, width_y) # for plotting

                # calculate distance of foci from middle of cell (scikit image)
                if orientation<0:
                    orientation = np.pi+orientation
                disp_y = (yyy-centroid[0])*np.sin(orientation) - (xxx-centroid[1])*np.cos(orientation)
                disp_x = (yyy-centroid[0])*np.cos(orientation) + (xxx-centroid[1])*np.sin(orientation)

                # append foci information to the list
                disp_l = np.append(disp_l, disp_y*params['pxl2um'])
                disp_w = np.append(disp_w, disp_x*params['pxl2um'])
                foci_h4 = np.append(foci_h4, np.sum(gfit_area_0))

                if params['debug_foci']:
                    print(disp_x, width_x)
                    print(disp_y, width_y)

    # draw foci on image for quality control
    if params['debug_foci']:
    #    print(np.min(gfit_area), np.max(gfit_area), gfit_median, avg_int, peak)
        # processing of image
        fig = plt.figure(figsize=(12,12))
        ax = fig.add_subplot(1,5,1)
        plt.title('fluor image')
        plt.imshow(c, interpolation='nearest', cmap='gray')
        ax = fig.add_subplot(1,5,2)
        plt.title('segmented image')
        plt.imshow(c_pc, interpolation='nearest', cmap='gray')
    #    ax = fig.add_subplot(1,5,3)
    #    plt.title('gaussian blur')
    #    plt.imshow(c_blur_gaus, interpolation='nearest', cmap='gray')
    #    ax = fig.add_subplot(1,6,5)
    #    plt.title('gaussian subtraction')
    #    plt.imshow(c_subtract_gaus, interpolation='nearest', cmap='gray')


        ax = fig.add_subplot(1,5,3)
        plt.title('DoG blobs')
        plt.imshow(c_subtract_gaus, interpolation='nearest', cmap='gray')
        # add circles for where the blobs are
        for i, max_spot in enumerate(x):
            foci_center = Ellipse([x[i],y[i]],r[i],r[i],color=(1.0, 1.0, 0), linewidth=2, fill=False, alpha=0.5)
            ax.add_patch(foci_center)

        # show the shape of the gaussian for recorded foci
        ax = fig.add_subplot(1,5,4)
        plt.title('final foci')
        plt.imshow(c, interpolation='nearest', cmap='gray')
        # print foci that pass and had gaussians fit
        for i, spot in enumerate(xx):
            foci_ellipse = Ellipse([xx[i],yy[i]], xxw[i], yyw[i],color=(0, 1.0, 0.0), linewidth=2, fill=False, alpha=0.5)
            ax.add_patch(foci_ellipse)

        ax6 = fig.add_subplot(1,5,5)
        plt.title('overlay')
        plt.imshow(c_pc, interpolation='nearest', cmap='gray')
        # print foci that pass and had gaussians fit
        for i, spot in enumerate(xx):
            foci_ellipse = Ellipse([xx[i],yy[i]], 3, 3,color=(1.0, 1.0, 0), linewidth=2, fill=False, alpha=0.5)
            ax6.add_patch(foci_ellipse)

    img_overlay = c_pc
    for i, spot in enumerate(xx):
        img_overlay[yy[i]-1,xx[i]-1] = 12
        img_overlay[yy[i]-1,xx[i]] = 12
        img_overlay[yy[i]-1,xx[i]+1] = 12
        img_overlay[yy[i],xx[i]-1] = 12
        img_overlay[yy[i],xx[i]] = 12
        img_overlay[yy[i],xx[i]+1] = 12
        img_overlay[yy[i]+1,xx[i]-1] = 12
        img_overlay[yy[i]+1,xx[i]] = 12
        img_overlay[yy[i]+1,xx[i]+1] = 12

    return disp_l, disp_w, foci_h4, img_overlay
