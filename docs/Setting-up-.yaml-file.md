# Setting up .yaml file

This is a guide for editing the .yaml file prior to analysis. The .yaml file contains information about the location of the images, and where to save output data. It also contains parameters related to the size of the growth channels (to aid in channel locating) and segmentation. There has been an effort to reduce the number of hardcoded parameters in the mm3 scripts, and instead load them from the parameters file. It is thus very important to fill out the parameters file correctly. The script itself is also commented with descriptions of the parameters.

Not all parameters are used in all scripts, and this can lead to confusing results. An effort here is made to indicate what scripts use which parameters.

The parameters .yaml file should be placed in your experimental directory, though that is not explicitly required. There are template parameter files in the mm3 directory `yaml_templates`.

### Name your experiment and folders.

`experiment_name: 'exp_name'`

_exp_name_ can be any arbitrary name, but will be the TIFF file prefix for images saved by mm3_Compile.py and other scripts. Downstream scripts thus look for channel stacks with this prefix. Don't change the name halfway through analysis!

`experiment_directory: '/path/to/exp/folder/'`

Root directory for the experiment. Must end in "/". All the following folders and files should be in this directory. See **Guide to folders and files** for more information.

The following are subfolders to be placed in the experiment directory. End them with a trailing "/" as well. If you do not makes the folders, the scripts will initialize them for you.

`image_directory: 'TIFF/'`

Subdirectory for the original TIFFs. If you are using mm3_nd2ToTIFF.py, then the images will be save to this folder.

`analysis_directory: 'analysis/'`

Subdirectory for storing analysis data. This will contain all your sliced channel TIFFs, empty channels, subtracted channels, segmented channels, metadata, and curated cell data.

### Indicate TIFF image source.

`TIFF_source: 'nd2ToTIFF'`

Options: `'nd2ToTIFF'` or `'elements'`

This is important for how mm3_Compile.py tries to read the metadata. Choose `'nd2ToTIFF'` if you used mm3_nd2ToTIFF.py to export TIFF images from an .nd2 file. Use `'elements'` if you exported your TIFF images from Nikon Elements.

### Indicate how processed images should be saved.

`output: 'TIFF'`

Options: `'TIFF'` or `'HDF5'`

mm3 supports saving processed images (sliced, empty, subtracted, and segmented channel stacks) to either TIFF stacks per channel or into a single HDF5 file per one FOV. TIFF stacks are a little more familiar for debugging. Using HDF5 is a little faster and the final file size is smaller. HDF5 is required if doing real-time analysis.

### Indicate picture interval.

`seconds_per_time_index: 60`

This number is needed for creating the time stamps during movie making.

### Indicate which color channel (plane) has the phase images.

`phase_plane: 'c1'`

Put in the file postfix of the plane which contains the phase contrast images. Used by the mm3_ChannelPicker.py and mm3_Subtract.py.

### Indicate conversion factor from pixels to micron.

`pxl2um: 0.108`

Options 0.065 (100X) or 0.108 (60X)

This number will be used when converting from pixels to microns when curating the cell data.

### Indicate channel orientation.

`image_orientation: 'auto'`

Options: `auto`, `up`, or `down`

The analysis code works on the convention that the channel opening is facing down (this ensures that the mother cell is close to 0,0 in the image array). Set this parameter so that your images can be flipped appropriately. If you have some FOVs opening up, and some opening down, use `auto`.

### Put in info about the size of the channels, use imageJ to figure out exact values.

`channel_width: 12`

The width of the channel in pixels. Used to help find the channels.

`channel_separation: 45`

Distance between channels (midpoint to midpoint) in pixels.

### Signal to noise ratio threshold for channel findings.

`channel_detection_snr: 1`

This is used in channel finding. Lower numbers will find more (and possible false) channels. Higher numbers will find less (and possibly miss) channels. This is used by mm3_Compile.py

### Set pad size around channels for slicing.

`channel_length_pad: 15`

Padding along y in pixels (applied to both ends).

`channel_width_pad: 10`

Padding along x in pixels (applied to both sides).

### Determining empty and full channels based on cross correlation.

`channel_picking_threshold: 0.97`

This is used by mm3_ChannelPicker.py to help determine if a channel is full or not. It is a measure of correlation between a series of images, so a value of 1 would mean the same image over and over. Channels with values above this value (like empty channels) will be designated as empty before the user selection GUI.

### Set the pad used when aligning channels.

`alignment_pad: 10`

This is the value in pixels that images will be scanned over to match them during cross-correlation determination and subtraction. Use large values if your channels move a lot during the experiment (will slow subtraction down).

### Set parameters for segmentation.

The following parameters are used in the segmentation of a single subtracted image. Check out the IPython notebook mm3_Segment.ipynb in the notebooks folder for a walkthrough on segmentation. You should edit these based on your experiment, with magnification and cell size determining what values work best.

`OTSU_threshold: 1.0`

Float greater than 0. The OTSU threshold will be multiplied by this number. 1.0 means no change. Use higher vales (<1.5 usually) when you want to separate segments more.

`first_opening_size: 2`

The radius of the disk that is used in the first morphological opening. Unit is pixels. 3 is good for 100X and 2 is good for 60X

`distance_threshold: 2`

Pixels closer than this distance to the edge of a cell will be zeroed. Unit is pixels. 3 is good for 100X and 2 is good for 60X

`second_opening_size: 0`

The radius, in pixels, of the size of the disk that is used in the second morphological. Unit is pixels. 2 is good for 100X and 0 (no second opening) is good for 60X

`min_object_size: 20`

Markers for potential cells smaller than this area will be zeroed. Unit is pixels^2. 40 is good for 100X and 20 is good for 60X.

### Set parameters for lineage creation.

These parameters have to do with creating cell lineages from segmentations across time. The should not have to be changed between experiments.

`lost_cell_time: 3`

Amount of frames after which a cell is dropped because no new regions linked to it.

`max_growth_length: 1.3`
`min_growth_length: 0.8`
`max_growth_area: 1.3`
`min_growth_area: 0.8`

Parameters for the minimum and maximum a region can when linking new regions to existing potential cell. Unit is ratio.

`new_cell_y_cutoff: 150`

Regions only less than this value down the channel from the closed end will be considered to start potential new cells. Does not apply to daughters. Unit is pixels.

`new_cell_region_cutoff: 2`

Regions with values only less than or equal to this number will be considered to start new cells. The mother cell has region value 1. 

### Set movie making parameters.

The following parameters are only used by mm3_MovieMaker.py

`movie_directory: 'movies/`

Directory that .mpg4 files are files. Appended to the experimental directory.

`fps: 24`

Frame rate per seconds of the movie.

`image_start: 1`

Frame to start movie making at. This is gleaned from the "t" position in the TIFF file.

`image_end: 1000`

Last frame of movie. Frames after this value will be ignored.
