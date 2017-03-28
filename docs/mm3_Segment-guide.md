# mm3_Segment.py

This script segments the subtracted images and then creates cell lives (lineages) by linking those segments across time.

**Input**
* TIFF channel stacks (subtracted images).
* specs.pkl file.

**Output**
* Segmented channels stacks. Saved in the `segmented/` subfolder in the analysis directory.
* Data for all cells. Saved as `complete_cells.pkl` in the `cell_data` subfolder in the analysis directory.
* (Optional) Lineage images. Lineage and segmentation information overlayed on the subtracted images across time frames.

## Usage
Run in terminal or iPython session. The -f option is required followed by the path to your parameter .yaml file.

> python ./mm3_Segment.py -f /path/to/parameter/file.yaml

**Options**

* -o "1,2,3" : Only these FOVs. Use a list of numbers separated by commas to only process these FOVs.

**Parameters File**

There are some key parameters which influence segmentation as well as building the lineages. The first four parameters are important for finding markers in order to do watershedding/diffusion for segmentation. They should be changed depending on cell size and magnification/imaging conditions.

The rest of the parameters are concerned with rules for linking segmented regions to create the lineages. They are not necessarily changed from experiment to experiment.

* `first_opening_size` : Size in pixels of first morphological opening during segmentation.
* `distance_threshold` : Distance in pixels which thresholds distance transform of binary cell image.
* `second_opening_size` : Size in pixels of second morphological opening.
* `min_object_size` : Objects smaller than this area in pixels will be removed before labeling.
* `print_lineages` : If set to true, images are printed overlaying segmentations and lineages over the subtracted images across time, one for each channel. Very slow but useful for debugging.
* `lost_cell_time` : If this many time points pass and a region has not yet been linked to a future region, it is dropped.
* `max_growth_length` : If a region is to be connected to a previous region, it cannot be larger in length by more than this ratio.
* `min_growth_length` : If a region is to be connected to a previous region, it cannot be smaller in length by less than this ratio.
* `max_growth_area` : If a region is to be connected to a previous region, it cannot be larger in area by more than this ratio.
* `min_growth_area` : If a region is to be connected to a previous region, it cannot be smaller in area by less than this ratio.
* `new_cell_y_cutoff` : distance in pixels from closed end of image above which new regions are not considered for starting new cells.

**Hardcoded parameters**

There are few hardcoded parameters at the start of the executable Python script (right after __main__).

* `do_segmentation` : Segment the subtracted channel stacks. If False attempt to load them.
* `do_lineages` : Create lineages from segmented images and save cell data.

## Notes on use

mm3_Segment.py consists of two parts, segmenting individual images, and then looking across time at those segments and linking them together to create growing cells.

Use the IPython notebook `mm3_Segment.ipynb` in the folder `notebooks` to decide which parameters to use during segmentation. You can start an IPython notebook session by typing `ipython notebook` in Terminal and navigating to the notebook using the browser.

Lineages are made by connecting the segmented regions into complete cells using basic rules. These rules include that a region in one time point must be a similar size and overlap with a region in a previous time point to which it will link. For a cell to divide, the two daughter cells' combined size must be similar and also overlap. For a cell to be considered complete, it must have both a mother and two daughters. 
