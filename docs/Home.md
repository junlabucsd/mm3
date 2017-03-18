# mm3 Overview as of March 2017

This is an overview of how to use mm3. Refer to the individual guides referenced in this document for more specific information about usage and function of each script. This guide is current for March 2017.

mm3 is a set of python scripts designed to facilitate analyzing time-lapse mother machine experiments. This can be thought of in two general tasks which are mostly independent. The first task is the bookkeeping of taking raw data (image files), identifying cell-containing growth channels, and creating image stacks that contain a single channel. mm3 Supports reading TIFF files from Nikon Elements, and supports saving to TIFF stacks or HDF5 datasets. The second task is to take those image stacks and actually identify cells and features to create analyzed data (curated cells).

## Workflow

Generally, there is one script for one process. The mm3 library file mm3_helpers.py contains the functions that do the actual heavy lifting. 

**Basic workflow is as follows:**

1. Create experimental folder and choose parameters.
2. Curate input data.
3. (Optional) Make movies (mm3_MovieMaker.py).
4. Locate channels, create channel stacks, and return metadata (mm3_Compile.py).
5. User guided selection of empty and full channels (mm3_ChannelPicker.py).
6. Subtract phase contrast images (mm3_Subtract.py).
7. Segment images and create cell lineages (mm3_Segment.py).
8. (Optional) Add fluorescent or additional image-based analysis (mm3_Colors.py).

### 1. Create experimental folder and choose parameters.

mm3 Python scripts are run from the terminal and point to a parameter file (.yaml file) that contains all the pertinent information about the experiment. See **Setting up .yaml file** for a guide to the parameters in the .yaml file. The most important information is arguably the paths to the images and where the analyzed images should be saved. See **Guide to folders and files** for more information about the organization of the raw and analyzed data in the experiment folder.

### 2. Curate input data.

mm3 currently currently takes individual TIFF images as its input. If there are multiple color layers, then each TIFF image should be a stack of planes corresponding to a color. There is a script to convert Nikon Elements .nd2 files into TIFF images of this form. See **mm3_nd2ToTIFF guide** for usage of that script. The quality of your images is important for mm3 to work properly. See **Input images guidelines** for more information.

### 3. Make movies per FOV (mm3_MovieMaker.py).

Though making movies of your data is not strictly required for analysis, it's a good idea to do so! There is a script to make .mpg4 movies from TIFF files. See **mm3_MovieMaker guide** for usage and details.

### 4. Locate channels, create channel stacks, and return metadata (mm3_Compile.py).

mm3_Compile.py is responsible for the initial bookkeeping. It attempts to automatically identify and crop out individual growth channels. Images corresponding to a specific channel are then stacked in time, and these "channel stacks" are the basis of further analysis. If there are multiple colors, a channel stack is made for each color for each channel.

It is also at this time that metadata is drawn from the images and saved. See **mm3_Compile guide** for usage and details.

### 5. User guided selection of empty and full channels (mm3_ChannelPicker.py).

mm3_Compile.py identifies all growth channels, regardless of if they contain or do not contain cells. mm3_ChannelPicker.py first attempts to guess, and then presents the user with a GUI to decide which channels should be analyzed, which channels should be ignored, and which channels should be used as empty channels during subtraction. See **mm3_ChannelPicker guide** for usage and details.

### 6. Subtract phase contrast images (mm3_Subtract.py).

Downstream analysis of phase contrast (brightfield) images requires background subtraction to remove artifacts of the PDMS device in the images. See **mm3_Subtract guide**.

### 7. Segmented images and create cell lineages (mm3_Segment.py).

mm3 Uses a relies on Otsu thresholding and watershedding algorithms to locate cells from the subtracted images. After cells are found for each channel in each time point, these labeled cells are connected across time to create complete cells and lineages. See **mm3_Segment guide** for usage and details.

## Other scripts

mm3 relies on a number of other python modules to run. Some of these are contained in the repository in the folder /external_lib. Most other modules are standard and should already be installed with your python distribution, especially if you are using [Anaconda](https://www.continuum.io/) (recommended). If you get an error and you need to install a module that you do not have, attempt to install it with pip:

`pip install module_name`
