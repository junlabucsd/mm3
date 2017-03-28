# mm3_Compile.py

This script takes raw TIFF files, finds growth channels, cuts them out, and saves image stacks across time for each growth channel. It does this by first locating the growth channels, then creating a cutting mask for each FOV that is applied to all time points. In addition, it extracts and saves image metadata.

**Input**
* Individual TIFF files. TIFFs should be separated by time point and FOV, but multiple colors should be in one image.

**Output**
* Stacked TIFFs through time for each channel (colors saved in separate stacks). These are saved to the `channels/` subfolder in the analysis directory.
* Metadata for each TIFF. These are saved as `TIFF_metadata.pkl` and `.txt`. A Python dictionary of metadata associated with each TIFF file.
* Channel masks for each FOV. These are saved as `channel_masks.pkl` and `.txt`. A Python dictionary that records the location of the channels in each FOV. Is a nested dictionaries of FOVs and then channel peaks. The final values are 4 pixel coordinates, ((y1, y2), (x1, x2)).

## Usage
Run in terminal or iPython session. The -f option is required followed by the path to your parameter .yaml file.

> python ./mm3_Compile.py -f /path/to/parameter/file.yaml

**Options**

* -o "1,2,3" : Only these FOVs. Use a list of numbers separated by commas to only process these FOVs.

**Parameters File**

Fill out the parameters file normally. Pay special attention to the following:

* `TIFF_source` needs to be specified to indicate how the script should look for TIFF metadata. Choices are `elements` and `nd2ToTIFF`.
* `channel_width`, `channel_separation`, and `channel_detection_snr`, which are used to help find the channels.
* `channel_length_pad` and `channel_width_pad` will increase the size of your channel slices.

**Hardcoded parameters**

There are few hardcoded parameters at the start of the executable Python script (right after __main__).

* `do_metadata` : Determine metadata. If this is False, it will attempt to load the metadata from a previous run of mm3_Compile.py
* `do_channel_masks` : Calculate consensus channel masks. Again, if False it will look to load this information.
* `t_end` : Will only analyze images up to this time point. Useful for debugging.

## Notes on use

Two types of TIFF files can be handled by this script. One are TIFFs that have been exported by mm3_nd2ToTIFF.py. These TIFFs have their metadata saved in the header of each file. The second is TIFF files which have been exported by Nikon Elements. This TIFF files must be exported as multi-page TIFFs if there are multiple planes (colors).
