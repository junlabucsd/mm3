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

There are some key parameters which influence segmentation as well as building the lineages.

* `first_opening_size`
* `distance_threshold`
* `second_opening_size`
* `min_object_size`
* `lost_cell_time`
* `print_lineages`
* `max_growth_length`
* `min_growth_length`
* `max_growth_area`
* `min_growth_area`
* `new_cell_y_cutoff`

**Hardcoded parameters**

There are few hardcoded parameters at the start of the executable Python script (right after __main__).

* `do_segmentation` : Segment the subtracted channel stacks. If False attempt to load them.
* `do_lineages` : Create lineages from segmented images and save cell data.

## Notes on use

Use the IPython notebook `mm3_Segment.ipynb` in the folder `notebooks` to decide which parameters to use during segmentation. 
