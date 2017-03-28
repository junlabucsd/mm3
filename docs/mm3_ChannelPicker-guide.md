# mm3_ChannelPicker.py

This script takes the channel stacks as created by mm3_Compile.py and identifies which channels are full and empty. It does this first by calculating the cross correlation value across time of the images in the stack to the first image in the stack. Images which have high correlation are considered empty, lower values are considered full. It then presents a GUI to the user who can manually curate which channels should be analyzed, which ones should be ignored, and which ones should be used as empty channels for subtraction.

**Input**
* TIFF channel stacks (phase contrast only).

**Output**
* crosscorrs.pkl and .txt : Python dictionary that contains image correlation value for channels over time. Used to guess if a channel is full or empty. Same structure as channel_masks.
* specs.pkl and .txt : Python dictionary which is the specifications of channels as full (1), empty (0), or ignore (-1). Same structure as channel_masks.

## Usage
Run in terminal or iPython session. The -f option is required followed by the path to your parameter .yaml file.

> python ./mm3_ChannelPicker.py -f /path/to/parameter/file.yaml

**Options**

* -o "1,2,3" : Only these FOVs. Use a list of numbers separated by commas to only process these FOVs.

**Parameters File**

Fill out the parameters file normally. Pay special attention to the following:

* `channel_picking_threshold` is a measure of correlation between a series of images, so a value of 1 would mean the same image over and over. Channels with values above this value (like empty channels) will be designated as empty before the user selection GUI.

**Hardcoded parameters**

There are few hardcoded parameters at the start of the executable Python script (right after __main__).

* `do_crosscorrs` : Determine cross correlations. If False try to load them.
* `do_picking` : Set to `False` if you just want to calculate cross correlations.

## Notes on use

When the cross correlations are calculated or loaded, the GUI is then launched. The user is asked to click on the channels to change their designation between analyze (green), empty (blue) and ignore (red).

The GUI shows all the channel for one FOV in columns. The top row has the first image from that channel. The second row has the last image from that channel, colored with the guess of if it is full or empty. The bottom row shows the cross correlation value (X, between 0.8 and 1), across time (Y, 0-100 where 0 is the start of the experiment and 100 is the end).

Click on the colored channels until they are as you wish. To go to the next FOV close the window or press enter in the Terminal (depends on Matplotlib version). The script will output the specs file with channels indicated as analyzed (green, 1), empty for subtraction (blue, 0), or ignore (red, -1).
