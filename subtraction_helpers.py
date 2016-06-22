#!/usr/bin/env python
__version__ = '2015.02.18'
#__docformat__ = 'restructuredtext en'
__all__ = ['pad_nans_tblr', 'pad_zeros_rb', 'trim_zeros_2d', 'trim_false_2d', 'rect_false_2d', 'trim_nans_2d', 'trim_zeros_2d3d', 'trim_false_2d3d']

import numpy as np

# tblr -> TOP BOTTOM LEFT RIGHT
def pad_nans_tblr(array, amounts):
    '''Puts NaNs around an image'''
    # where amounts is a tuple of (top, bottom, left, right)
    # asser is a shorthand way of triggering an error output if a condition is not met
    assert len(amounts) == 4, "Amounts was not a tuple of length 4."
    # deal with NEGATIVE values for top bottom left and right
    # in each case resize the array trimming of the amount that the boundary is negative
    # i.e. if top is -2 then 2 rows of values are removed from the top
    # after each change is made the value is set to zero
    if amounts[0] < 0:
        array = array[(-1*amounts[0]):,:]
        amounts[0] = 0
    if amounts[1] < 0:
        array = array[:amounts[1],:]
        amounts[1] = 0
    if amounts[2] < 0:
        array = array[:,(-1*amounts[2]):]
        amounts[2] = 0
    if amounts[3] < 0:
        array = array[:,:amounts[3]]
        amounts[3] = 0
    # creat a 2D array of zeros as tall as the input array and as wide as the 'left' value
    pad_1 = np.zeros((array.shape[0], amounts[2]))
    # reassigne its zeros to be NANs instead
    pad_1[pad_1 == 0] = np.nan
    # repeat this for the right side
    pad_2 = np.zeros((array.shape[0], amounts[3]))
    pad_2[pad_2 == 0] = np.nan
    # sandwich the array between these 'pads' from the left and right
    array = np.concatenate((np.concatenate((pad_1, array), axis = 1), pad_2), axis = 1)
    # now use the same method as in the previous lines to sanwhich it from the top and botom with
    # pads of NAN's of the appropriate width
    pad_3 = np.zeros((amounts[0], array.shape[1]))
    pad_3[pad_3 == 0] = np.nan
    pad_4 = np.zeros((amounts[1], array.shape[1]))
    pad_4[pad_4 == 0] = np.nan
    array = np.concatenate((np.concatenate((pad_3, array), axis = 0), pad_4), axis = 0)
    #return the padded array
    return array

# this function is like a version of pad_nans_tblr only just for right and bottom
# one major differenc is that it does not do anything with negative values in amounts
def pad_zeros_rb(array, amounts):
    # where amounts is a tuple of (right, bottom)
    assert len(amounts) == 2, "Amounts was not a tuple of length 2."
    if amounts[0] > 0:
        array = np.concatenate((array, np.zeros((amounts[0], array.shape[1]))), axis = 0)
    if amounts[1] > 0:
        array = np.concatenate((array, np.zeros((array.shape[0], amounts[1]))), axis = 1)
    return array

# remove margins of zeros
def trim_zeros_2d(array):
    # make the array equal to the sub array which has columns of all zeros removed
    # "all" looks along an axis and says if all of the valuse are such and such for each row or column
    # ~ is the inverse operator
    # using logical indexing
    array = array[~np.all(array == 0, axis = 1)]
    # transpose the array
    array = array.T
    # make the array equal to the sub array which has columns of all zeros removed
    array = array[~np.all(array == 0, axis = 1)]
    # transpose the array again
    array = array.T
    # return the array
    return array

# version of trim_zeros_2d for multichannel images
def trim_zeros_2d3d(array):
    if len(array.shape) > 2:
        array = array[~np.all(array[:,:,0] == 0, axis = 1)]
        array = array.T
        array = array[~np.all(array[:,:,0] == 0, axis = 1)]
        array = array.T
    else:
        array = array[~np.all(array == 0, axis = 1)]
        array = array.T
        array = array[~np.all(array == 0, axis = 1)]
        array = array.T
    return array

# remove margins of boolean value False
def trim_false_2d(array):
    array = array[~np.all(array == False, axis = 1)]
    array = array.T
    array = array[~np.all(array == False, axis = 1)]
    array = array.T
    return array

# version of trim_false_2d for multichannel images
def trim_false_2d3d(array):
    newarray = []
    if len(array.shape) < 3:
    	array = np.expand_dims(array, axis = 2)
    for layer in np.dsplit(array, array.shape[2]):
        temparray = layer.reshape(array[:,:,0].shape)
        temparray = temparray[~np.all(temparray == False, axis = 1)]
        temparray = temparray.T
        temparray = temparray[~np.all(temparray == False, axis = 1)]
        temparray = temparray.T
        newarray.append(temparray)
    try:
        return np.dstack(newarray)
    except:
        for array in newarray:
            print array.shape
        raise

# This function returns a boolean mask with an inner rectangle of Trues coreesponding to the largest rectangle to contane no rowns or crolumns with only Falses in the given image
def rect_false_2d(array):
    # only used in empty averaging
    # test operation "not all" along axis one of the boolean array retuned by the logic test array == False
    # this means that if a full row is all False then there will be a false in s1_array
    s1_array = ~np.all(array == False, axis = 1)
    # same only for columns on s2_array because of the .T transpose operation
    s2_array = ~np.all(array.T == False, axis = 1)
    # need to transpose s2_array
    s2_array = s2_array.T
    # use the numpy repeat function to creat an s3_array as wide as s1_array and as tall as s2_array
    # reshape allows the array to be reinterpreted/ reformulated into an array of the orriginal data indexed with new dimentsions
    # the defaults is row major order (C-STYLE)
    s3_array = np.reshape(np.repeat(s1_array, repeats = s2_array.shape[0], axis = 0), array.shape)
    # same data only with column major order (fourtran-style indexing order)
    s4_array = np.reshape(np.repeat(s2_array, repeats = s1_array.shape[0], axis = 0), array.shape, order = 'F')
    #dstack is supposed to let you stack into the third dimension in a fashion simlare to vstack (axis=0) and hstack(axis=1)
    # the axis = 2 should be unnecessary, no?
    # retunr the true values for where there are all values which evaluate tor True in all layers the array (i.e. looking along axis 2)
    array = np.all(np.dstack((s3_array, s4_array)), axis = 2)
    # return this truth array
    return array

# a mutlilayer verstion of rect_false_2d
def rect_false_2d3d(array):
    # only used in empty averaging
    s1_array = ~np.all(array == False, axis = 1)
    s2_array = ~np.all(array.T == False, axis = 1)
    s2_array = s2_array.T
    s3_array = np.reshape(np.repeat(s1_array, repeats = s2_array.shape[0], axis = 0), array.shape)
    s4_array = np.reshape(np.repeat(s2_array, repeats = s1_array.shape[0], axis = 0), array.shape, order = 'F')
    array = np.all(np.dstack((s3_array, s4_array)), axis = 2)
    return array

#trim NAN's of of an array in a fashion similar to trim_false_2d
def trim_nans_2d(array):
    # only used in empty averaging
    array = array[~np.all(array == np.nan, axis = 1)]
    array = array.T
    array = array[~np.all(array == np.nan, axis = 1)]
    array = array.T
    return array
