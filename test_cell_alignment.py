#%%
from skimage import io, transform, measure
import os
import sys
import numpy as np

from matplotlib import pyplot as plt

#%%
home_direc = '/home/wanglab/Users_local/Jeremy/Imaging/20190726/analysis'
cell_seg_stack = io.imread(os.path.join(home_direc,'segmented','20190726_JDW3705-inv_xy001_p0126_seg_unet.tif'))
focus_seg_stack = io.imread(os.path.join(home_direc,'segmented_foci','20190726_JDW3705-inv_xy001_p0126_foci_seg_unet.tif'))
fluor_stack = io.imread(os.path.join(home_direc,'channels','20190726_JDW3705-inv_xy001_p0126_c2.tif'))
phase_seg_stack = io.imread(os.path.join(home_direc,'channels','20190726_JDW3705-inv_xy001_p0126_c1.tif'))


#%%
test_frame = 10
cell_seg_img = cell_seg_stack[test_frame,...]
focus_seg_img = focus_seg_stack[test_frame,...]
fluor_img = fluor_stack[test_frame,...]
phase_seg_img = phase_seg_stack[test_frame,...]

#%%
regions = measure.regionprops(cell_seg_img)
region = [reg for reg in regions if reg.label==3][0]
orientation = region.orientation
#%%
if orientation > 0:
    deg = 90 - (orientation * 180 / np.pi)
else:
    deg = -90 - (orientation * 180 / np.pi)

#%%
cell_seg_img = transform.rotate(
        cell_seg_img,
        angle = deg,
        # center = (centroid[1], centroid[0]),
        preserve_range = True,
        resize = True
    ).astype('uint8') # cast back to integer

focus_seg_img = transform.rotate(
        focus_seg_img,
        angle = deg,
        # center = (centroid[1], centroid[0]),
        preserve_range = True,
        resize = True
    ).astype('uint8') # cast back to integer        

fluor_img = transform.rotate(
        fluor_img,
        angle = deg,
        # center = (centroid[1], centroid[0]),
        preserve_range = True,
        resize = True
    ).astype('uint16') # cast back to integer   

#%%
io.imshow(fluor_img)

#%%
# get regionprops for each region
focus_regions = measure.regionprops(focus_seg_img)

#%%
len(measure.regionprops(focus_seg_stack[test_frame,...]))
#%%
np.max(focus_seg_img)
#%%
a = np.array([[0,0,2],
              [0,1,3]])
b = np.where(a > 0)
print(b)

#%%
c = np.array([[0,0,0],
             [0,0,0]])
#%%
print(np.max(a, axis=0))
print(np.argmax(a, axis=0))
print(np.std(a, axis=0))
print(np.where(np.std(a, axis=0) > 0)[0])
#%%
d = np.where(np.std(c, axis=0) >0 )[0]

#%%
# which rows of each column are maximum jaccard index?
max_inds = np.argmax(a, axis=0)
# because np.argmax returns zero if all rows are equal, we
#   need to evaluate if all rows are equal.
#   If std_dev is zero, then all were equal,
#   and we omit that index from consideration for 
#   focus tracking.
sd_vals = np.std(a, axis=0)
tracked_inds = np.where(sd_vals > 0)[0]


#%%
tracked_inds