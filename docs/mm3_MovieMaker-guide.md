# mm3_MovieMaker.py

This script takes a series of TIFF images and saves them into .mp4 videos. It uses [ffmpeg](https://trac.ffmpeg.org/wiki/CompilationGuide/MacOSX) which must be installed separately. The Python script, after getting the image information from the experimental folder, is actually initiating a shell command of ffmpeg.

In addition it uses the Python packages freetype to make labels, and tifffile to open the TIFFs.

### Usage
Run in terminal or ipython session. The -f option is required followed by the path to your parameter .yaml file.
> python ./mm3_MovieMaker.py -f /path/to/parameter/file.yaml

**Options**
* -o "1,2,3" : Only these FOVs. Use a list of numbers separated by commas to only process these FOVs.
* -s "5" : Start FOV. Put in a number to start processing at a certain FOV.

**Parameters File**

Fill out the parameters file normally, the most important thing is that you have the file and folder names right. Here are some things to be aware of:
* `seconds_per_time_index` This is used for the labeling (not the actual time in the image metadata).
* `fps` Frames per second of the movie.
* `image_start` Frame number (zero indexed) at which to start movie making.
* `image_end` Same but for end frame to include.
