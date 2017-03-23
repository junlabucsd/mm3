# Setting up .yaml file

This is a guide for editing the .yaml file prior to analysis. The .yaml file contains information about the location of the images, and where to save output data. It also contains parameters related to the size of the growth channels. There has been an effort to reduce the number of hardcoded parameters in the mm3 scripts, and instead load them from the parameters file. It is thus very important to fill out the parameters file correctly. The script itself is also commented with descriptions of the parameters.

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

### Indicate conversion factor from pixels to micron

`pxl2um: 0.108`

Options 0.065 (100X) or 0.108 (60X)

This number will be used when converting from pixels to microns when curating the cell data.

### Indicate channel orientation

`image_orientation: 'auto'`

Options: `auto`, `up`, or `down`

The analysis code works on the convention that the channel opening is facing down (this ensures that the mother cell is close to 0,0 in the image array). Set this parameter so that your images can be flipped appropriately. If you have some FOVs opening up, and some opening down, use `auto`.

### Put in info about the size of the channels, use imageJ to figure out exact values.

`channel_width: 12`

The width of the channel in pixels. Used to help find the channels.

`channel_separation: 45`

Distance between channels (midpoint to midpoint) in pixels.

### Set pad size around channels for slicing

`channel_length_pad: 15`

Padding along y in pixels (applied to both ends).

`channel_width_pad: 10`

Padding along x in pixels (applied to both sides).

### Signal to noise ratio threshold for channel findings

`channel_detection_snr: 1`

This is used in channel finding. Lower numbers will find more (and possible false) channels. Higher numbers will find less (and possibly miss) channels. This is used by mm3_Compile.py

### Determining empty and full channels based on cross correlation

`channel_picking_threshold: 0.97`

This is used by mm3_ChannelPicker.py to help determine if a channel is full or not. It is a measure of correlation between a series of images, so a value of 1 would mean the same image over and over. Channels with values above this value (like empty channels) will be designated as empty before the user selection GUI.

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
