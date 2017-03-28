# Description of cell data

When cells are made during lineage creation, the information is stored per cell in an object called Cell. The Cell object is the fundamental unit of data produced by mm3. Every Cell object has a unique identifier (`id`) as well as all the other pertinent information. Below is a description of the information contained in a Cell object and how to export it to other formats. For an overview on classes in Python see [here](https://learnpythonthehardway.org/book/ex40.html).

## Cell object attributes

The following is a list of the attributes of a Cell.

* `Cell.id` : The cell id is a string in the form `f0p0t0r0` which represents the FOV, channel peak number, time point the cell came from as well as which segmented region it is in that image.
* `Cell.fov` : FOV the cell came from.
* `Cell.peak` : Channel peak number the cell came from.
* `Cell.birth_label` : The segmented region number the cell was born at. The regions are numbered from the closed end of the channel, so mother cells should have a birth label of 1.
* `Cell.parent` : This cell's mother cell's id.
* `Cell.daughters` : A list of the ids of this cell's two daughter cells.
* `Cell.birth_time` : Nominal time point at time of birth.
* `Cell.division_time` : Nominal division time of cell. Note that this is equal to the birth time of the daughters.
* `Cell.times` : A list of time points for which this cell grew. Includes the first time point but does not include the last time point. It is the same length as the following attributes, but it may not be sequential because of dropped segmentations.
* `Cell.labels` : The segmented region labels over time.
* `Cell.bboxes` : The bounding boxes of each region in the segmented channel image over time.
* `Cell.areas` : The areas of the segmented regions over time in pixels^2.
* `Cell.x_positions` : The x positions in pixels of the centroid of the regions over time.
* `Cell.y_positions` : The y positions in pixels of the centroid of the regions over time.
* `Cell.lengths` : The long axis length in pixels of the regions over time.
* `Cell.widths` : The long axis width in pixels of the regions over time.
* `Cell.times_w_div` : Same as Cell.times but includes the division time.
* `Cell.lengths_w_div` : The long axis length in microns of the regions over time, including the division length.
* `Cell.sb` : Birth length of cell in microns.
* `Cell.sd` : Division length of cell in microns. The division length is the combined birth length of the daugthers.
* `Cell.delta` : Cell.sd - Cell.sb. Simply for convenience.
* `Cell.tau` : Nominal generation time of the cell.
* `Cell.elong_rate` : Elongation rate of the cell using a linear fit of the log lengths.
* `Cell.sum_cov` : Sum of the covarience matrix for the fit.
* `Cell.septum_position` : The birth length of the first daughter (closer to closed end) divided by the division length.  

## Additional data outputs

The default data format save by mm3_Segment.py is a dictionary of these cell objecs, where the keys are each cell id and the values are the object itself. Use the script mm3_OutputData.py to save the data in different ways, such as a Matlab .mat file or .csv. There are also functions/examples there that show how to filter cells (for example, just extracting the mother cells) or filtering by time point. The script is run the same way as the others, but you should manually edit the flags in the script to turn on what you want. There is also some basic plotting functions which require the [Seaborn](https://seaborn.pydata.org/) package.
