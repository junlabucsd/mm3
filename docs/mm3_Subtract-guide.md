# mm3_Subtract.py

This script, according the specs file, creates and average empty channel to be used for subtraction, and then subtracts this empty channel from the specified channel in the phase contrast plane.

**Input**
* TIFF channel stacks (phase contrast only).
* specs.pkl file.

**Output**
* Averaged empty stack. Saved in the `empties/` subfolder in the analysis directory.
* Subtracted channel stacks. Saved in the `subtracted/` subfolder in the analysis directory.

## Usage
Run in terminal or iPython session. The -f option is required followed by the path to your parameter .yaml file.

> python ./mm3_Subtract.py -f /path/to/parameter/file.yaml

**Options**

* -o "1,2,3" : Only these FOVs. Use a list of numbers separated by commas to only process these FOVs.

**Parameters File**

Fill out the parameters file normally.

**Hardcoded parameters**

There are few hardcoded parameters at the start of the executable Python script (right after __main__).

* `do_empties` : Calculate averaged empty channels. If False attempt to load them.
* `do_subtraction` : Subtract phase constrast images or not.

## Notes on use

If for a specific FOV there are multiple empty channels designated, then those channels are averaged together by timepoint to create an averaged empty channel. If only one channel is designated in the specs file as empty, then it will simply be copied over. If no channels are designated as empty, than this FOV is skipped, and the user is required to copy one of the empty channels from `empties/` subfolder and rename with the absent FOV ID.
