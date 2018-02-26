# mm3 Overview
# Updated October 2017

This is an overview of how to use mm3. Refer to the individual guides referenced in this document for more specific information about usage and function of each script.

mm3 is a set of python scripts designed to facilitate analyzing time-lapse mother machine experiments. This can be thought of in two general tasks which are mostly independent. The first task is the bookkeeping of taking raw data (image files), identifying cell-containing growth channels, and creating image stacks that contain a single channel. mm3 supports reading .nd2 or TIFF files from Nikon Elements, and supports saving to TIFF stacks or HDF5 datasets. The second task is to take those image stacks and actually identify cells and features to create analyzed data (curated cells). This is done via segmentation of subtracted images and lineage creation. A parameter (.yaml) file is used to pass parameters specific to the experiment to the scripts of mm3.

## Installation

See the script **Install-guide** for requirements and installation procedure.

## Workflow

Generally, there is one script for one process. The mm3 library file mm3_helpers.py contains the functions that do the actual heavy lifting. Scripts are best run from a Python session started in Terminal in the following general format:

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
```
.
├── 20170720_SJ388_mopsgluc12aa.nd2
├── TIFF
└── params.yaml
```

### 3. Locate channels, create channel stacks, and return metadata (mm3_Compile.py).

mm3_Compile.py is responsible for the initial bookkeeping. It attempts to automatically identify and crop out individual growth channels. Images corresponding to a specific channel are then stacked in time, and these "channel stacks" are the basis of further analysis. If there are multiple colors, a channel stack is made for each color for each channel.

It is also at this time that metadata is drawn from the images and saved. See **mm3_Compile guide** for usage and details.

The working directory now contains:
```
.
├── 20170720_SJ388_mopsgluc12aa.nd2
├── TIFF
├── analysis
│   ├── time_table.pkl
│   ├── time_table.txt
│   ├── TIFF_metadata.pkl
│   ├── TIFF_metadata.txt
│   ├── channel_masks.pkl
│   ├── channel_masks.txt
│   └── channels
└── params.yaml
```

### 4. User guided selection of empty and full channels (mm3_ChannelPicker.py).

mm3_ChannelPicker.py identifies all growth channels, regardless of if they contain or do not contain cells. mm3_ChannelPicker.py first attempts to guess, and then presents the user with a GUI to decide which channels should be analyzed, which channels should be ignored, and which channels should be used as empty channels during subtraction. This information is contained within the specs.pkl file. See **mm3_ChannelPicker guide** for usage and details.

The working directory is now:
```
.
├── 20170720_SJ388_mopsgluc12aa.nd2
├── TIFF
├── analysis
│   ├── time_table.pkl
│   ├── time_table.txt
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
```

### 5. Subtract phase contrast images (mm3_Subtract.py).

Downstream analysis of phase contrast (brightfield) images requires background subtraction to remove artifacts of the PDMS device in the images. See **mm3_Subtract guide**.

The working directory is now:
```
.
├── 20170720_SJ388_mopsgluc12aa.nd2
├── TIFF
├── analysis
│   ├── time_table.pkl
│   ├── time_table.txt
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
```

### 6. Segment images and create cell lineages (mm3_Segment.py).

mm3 relies on Otsu thresholding, morphological operations and watershedding to locate cells from the subtracted images. After cells are found for each channel in each time point, these labeled cells are connected across time to create complete cells and lineages. See **mm3_Segment guide** for usage and details.

The working directory is now:
```
.
├── 20170720_SJ388_mopsgluc12aa.nd2
├── TIFF
├── analysis
│   ├── time_table.pkl
│   ├── time_table.txt
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
```

## Other scripts

There are some optional and additional processes that you may want to perform based on you experiment.

**Additional processes:**

* Make movies (mm3_MovieMaker.py).
* Add fluorescent or additional image-based analysis (mm3_Colors.py).
* Foci detection (mm3_Foci.py)
* Output data to various formats and plot data (mm3_OutputData.py).
* mm3 can process images in real-time (mm3_Agent.py).

### Make movies per FOV (mm3_MovieMaker.py).

Though making movies of your data is not strictly required for analysis, it's a good idea to do so! There is a script to make .mpg4 movies from TIFF files. See **mm3_MovieMaker guide** for usage and details.

### Add additional image analysis to cell data (mm3_Colors.py).

The cell data output by mm3_Segment.py contains information about all cells in the experiment, including which images and locations in the images they came from. You can use this to go back to additional image planes (colors) for secondary analysis, such as fluorescence levels or foci detection. mm3_Colors.py provides an example of how to add average fluorescent intensity information to the cell data.

### Output data to various formats (mm3_OutputData.py).

The cell information output by mm3_Segment.py is in the form of a dictionary of Cell objects which describe individual cells. You can use mm3_OutputData.py to both filter that dictionary and to save the data in different formats. See **Cell_data_description** for more information.

### Filtering of the cells pickle file (mm3_postprocessing.py)
The `all_cells.pkl` file may contain problematic cells. For example incomplete generations, or simply non-healthy cells. The `mm3_postprocessing.py` script implements a straightforward pipeline for postprocessing the pickle file generated by mm3. For help, type `python mm3_postprocessing.py -h`.

* Filtering
	1. Removal of incomplete cells (*i.e.* without parent and daughters).
	2. Retaining certain label types (*e.g.* mother cell label type).
	3. Filtering based on scalar quantities such as cell size at birth or generation time.
	4. Retaining cells belonging to continuous lineages of at least a given length.

* Computations
	1. Compute key quantities based on the cell cycle information (*e.g.* C period).
	2. Convert times and rates in minutes based on the minute per frame information (mpf) in the param file (see below).

* Cell cycle
	1. Load the cell cycle information obtained with the Matlab GUI in the cell pickle file.
	2. Optionally retain only cells which are mapped to a complete cell cycle (*i.e.* initiation --> division).

All these operations are parametrized through the yaml file given as input (see the template named `params_postprocessing.yaml`). The syntax for applying the filtering is:
```
python mm3_postprocessing.py all_cells.pkl -f path/to/params_postprocessing.yaml [options]
```

The output will be:
* a filtered pickle file (*e.g.* `all_cells_filtered.pkl`)
* a file containing the list of lineages of a given length (*e.g.* `all_cells_lineages.pkl`). Note that the filtered cells pickle file will be restrained to those continuous lineages only if the option `keep_continuous_only` in the parameters file is set to `True`.

### Plots based on the cells pickle file (mm3_plots_alternative.py)
The `mm3_plots_alternative.py` has the following syntax:
```
python mm3_plots_alternative.py all_cells_filtered.pkl -f path/to/params_postprocessing.yaml [--distributions] [--crosscorrelations] [--autocorrelations] [-l lineages.pkl]
```

Each option has the following function:
* `--distributions`: plot the distributions of the selected variables.
* `--crosscorrelations`: plot the cross-correlations of the selected variables.
* `--autocorrelations`: plot the autocorrelations of the selected variables.
* `-l lineages.pkl`: plots on a per lineage basis.

All options for plots are passed through the `params_postprocessing.yaml` file.

