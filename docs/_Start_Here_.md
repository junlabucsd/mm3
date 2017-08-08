# mm3 Overview as of March 2017

This is an overview of how to use mm3. Refer to the individual guides referenced in this document for more specific information about usage and function of each script. This guide is current for March 2017.

mm3 is a set of python scripts designed to facilitate analyzing time-lapse mother machine experiments. This can be thought of in two general tasks which are mostly independent. The first task is the bookkeeping of taking raw data (image files), identifying cell-containing growth channels, and creating image stacks that contain a single channel. mm3 supports reading TIFF files from Nikon Elements, and supports saving to TIFF stacks or HDF5 datasets. The second task is to take those image stacks and actually identify cells and features to create analyzed data (curated cells). This is done via segmentation of subtracted images and lineage recreation. A parameter (.yaml) file is used to pass parameters specific to the experiment to the scripts of mm3.

## Installation

See the script **Install-guide** for requirements and installation procedure.

## Workflow

Generally, there is one script for one process. The mm3 library file mm3_helpers.py contains the functions that do the actual heavy lifting. Scripts are best run from a Python session started in Terminal:

> python /path/to/mm3_script.py -f /path/to/parameter/file.yaml

**Basic workflow is as follows:**

1. Create experimental folder and choose parameters.
2. Curate input data.
3. Locate channels, create channel stacks, and return metadata (mm3_Compile.py).
4. User guided selection of empty and full channels (mm3_ChannelPicker.py).
5. Subtract phase contrast images (mm3_Subtract.py).
6. Segment images and create cell lineages (mm3_Segment.py).

### 1. Create experimental folder and choose parameters.

mm3 Python scripts are run from the Terminal and point to a parameter file (.yaml file) that contains all the pertinent information about the experiment. See **Setting up .yaml file** for a guide to the parameters in the .yaml file. The most important information is arguably the paths to the images and where the analyzed images should be saved. See **Guide to folders and files** for more information about the organization of the raw and analyzed data in the experiment folder.

As an example, we'll assume that the current working directory contains:
```
.
├── 20170720_SJ388_mopsgluc12aa.nd2
└── params.yaml
```

### 2. Curate input data.

mm3 currently currently takes individual TIFF images as its input. If there are multiple color layers, then each TIFF image should be a stack of planes corresponding to a color. There is a script to convert Nikon Elements .nd2 files into TIFF images of this form. See **mm3_nd2ToTIFF guide** for usage of that script. The quality of your images is important for mm3 to work properly. See **Input images guidelines** for more information.

The working directory now contains:
.
├── 20170720_SJ388_mopsgluc12aa.nd2
├── TIFF
└── params.yaml

### 4. Locate channels, create channel stacks, and return metadata (mm3_Compile.py).

mm3_Compile.py is responsible for the initial bookkeeping. It attempts to automatically identify and crop out individual growth channels. Images corresponding to a specific channel are then stacked in time, and these "channel stacks" are the basis of further analysis. If there are multiple colors, a channel stack is made for each color for each channel.

It is also at this time that metadata is drawn from the images and saved. See **mm3_Compile guide** for usage and details.

The working directory now contains:
.
├── 20170720_SJ388_mopsgluc12aa.nd2
├── TIFF
├── analysis
│   ├── TIFF_metadata.pkl
│   ├── TIFF_metadata.txt
│   ├── channel_masks.pkl
│   ├── channel_masks.txt
│   └── channels
└── params.yaml

### 5. User guided selection of empty and full channels (mm3_ChannelPicker.py).

mm3_Compile.py identifies all growth channels, regardless of if they contain or do not contain cells. mm3_ChannelPicker.py first attempts to guess, and then presents the user with a GUI to decide which channels should be analyzed, which channels should be ignored, and which channels should be used as empty channels during subtraction. See **mm3_ChannelPicker guide** for usage and details.

The working directory is now:
.
├── 20170720_SJ388_mopsgluc12aa.nd2
├── TIFF
├── analysis
│   ├── TIFF_metadata.pkl
│   ├── TIFF_metadata.txt
│   ├── channel_masks.pkl
│   ├── channel_masks.txt
│   ├── channels
│   ├── crosscorrs.pkl
│   ├── crosscorrs.txt
│   ├── specs.pkl
│   └── specs.txt
└── params.yaml

### 6. Subtract phase contrast images (mm3_Subtract.py).

Downstream analysis of phase contrast (brightfield) images requires background subtraction to remove artifacts of the PDMS device in the images. See **mm3_Subtract guide**.

The working directory is now:
.
├── 20170720_SJ388_mopsgluc12aa.nd2
├── TIFF
├── analysis
│   ├── TIFF_metadata.pkl
│   ├── TIFF_metadata.txt
│   ├── channel_masks.pkl
│   ├── channel_masks.txt
│   ├── channels
│   ├── crosscorrs.pkl
│   ├── crosscorrs.txt
│   ├── empties
│   ├── specs.pkl
│   ├── specs.txt
│   └── subtracted
└── params.yaml

### 7. Segment images and create cell lineages (mm3_Segment.py).

mm3 Uses a relies on Otsu thresholding and watershedding algorithms to locate cells from the subtracted images. After cells are found for each channel in each time point, these labeled cells are connected across time to create complete cells and lineages. See **mm3_Segment guide** for usage and details.

The working directory is now:
.
├── 20170720_SJ388_mopsgluc12aa.nd2
├── TIFF
├── analysis
│   ├── TIFF_metadata.pkl
│   ├── TIFF_metadata.txt
│   ├── cell_data
│   │   └── complete_cells.pkl
│   ├── channel_masks.pkl
│   ├── channel_masks.txt
│   ├── channels
│   ├── crosscorrs.pkl
│   ├── crosscorrs.txt
│   ├── empties
│   ├── segmented
│   ├── specs.pkl
│   ├── specs.txt
│   └── subtracted
└── params.yaml

## Other scripts

There are some optional and additional processes that you may want to perform based on you experiment.

**Additional processes:**

* Make movies (mm3_MovieMaker.py).
* Add fluorescent or additional image-based analysis (mm3_Colors.py).
* Output data to various formats (mm3_OutputData.py).
* mm3 can process images in real-time (mm3_Agent.py).

### Make movies per FOV (mm3_MovieMaker.py).

Though making movies of your data is not strictly required for analysis, it's a good idea to do so! There is a script to make .mpg4 movies from TIFF files. See **mm3_MovieMaker guide** for usage and details.

### Add additional image analysis to cell data (mm3_Colors.py).

The cell data output by mm3_Segment.py contains information about all cells in the experiment, including which images and locations in the images they came from. You can use this to go back to additional image planes (colors) for secondary analysis, such as fluorescence levels or foci detection. mm3_Colors.py provides an example of how to add average fluorescent intensity information to the cell data.

### Output data to various formats (mm3_OutputData.py).

The cell information output by mm3_Segment.py is in the form of a dictionary of Cell objects which describe individual cells. You can use mm3_OutputData.py to both filter that dictionary and to save the data in different formats. See **Cell_data_description** for more information.
