# mm3_nd2ToTIFF.py
This script takes an .nd2 file from Nikon Elements and makes individual TIFFs for every FOV and time point. Colors are saved into a stack for that timepoint (multi-page TIFF). 

The script uses the [pims_nd2](https://github.com/soft-matter/pims_nd2) package based off the Nikon SDK to be able to open .nd2 and read the metadata. Saves TIFF files using the package tifffile.py (contained in the `external_lib/` directory. 

**Input**
* .nd2 file as produced by Nikon Elements
* parameter.yaml file

**Output**
* Individual TIFF files. 

## Usage
Run in terminal or iPython session. The -f option is required followed by the path to your parameter .yaml file. 

> python ./mm3_nd2_to_TIFF.py -f /path/to/parameter/file.yaml

**Options**

* -x /path/to/nd2/file : eXternal file. Use this to point to a specific folder which contains your .nd2. Otherwise the experimental directory will be searched
* -o "1,2,3" : Only these FOVs. Use a list of numbers separated by commas to only process these FOVs.
* -s "5" : Start FOV. Put in a number to start processing at a certain FOV. 
* -n "1" : FOV Number offset. You can use this to save the FOV number of the TIFF file increased by an arbitrary value. 

**Parameters File**

Fill out the parameters file normally, the most important thing is that you have the file and folder names right.

**Hardcoded parameters**

There are few hardcoded parameters at the start of the executable Python script (right after __main__). 

* `number_of_rows` : If there are two rows of channels, put 2 here, so two TIFF files, each containing a single row, is cropped out for each TIFF. Otherwise put 1. 
* `vertical_crop` : This is used for cropping out a section of the image vertically when saving the TIFF files. The form is (y1, y2) for a single row, and ((r1y1, r1y2), (r2y1, r2y2)) for two rows if `number_of_rows` is 2. It is best to crop the rows so there are ~50 pixels above and below the rows, but not too much extra, which complicates channel finding. See **Input image guidelines** for more advice. 
* `tif_compress` : A number between 0 and 9 which indicates if the TIFF files should be compressed when saved. 0 is no compression, 9 is the most.

## Notes on use

When converting from .nd2 to TIFF, you may need to run the script multiple times for different sets of images because of different cropping parameters, etc. This means that the number of FOVs you have in your original .nd2 file may change (for example when analyzing double row images). Be mindful when using the `specify_fovs` and `fov_naming_start` parameters. 

mm3_nd2ToTIFF.py reads the metadata directly from the .nd2 and then writes it into the header of the TIFF file when saving. The format is a json representation of a Python dictionary, and is recognized later by mm3_Compile.m3.  
