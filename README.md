# mm3
## doing the bookkeeping so you don't have to since 2014.

same optimization applies as in `motherMachine2`: patch `numpy` to use FFTW!

no automatic segmentation output yet. using `pyMakeSegmentMaps.py` from the
`analyzer_dev` branch of `motherMachine2` should work as a stop-gap. s.d.b.
wrote new (much faster) recursion-based code in an IPython notebook and hasn't
converted it to a reasonable script yet. `mm3-Segmentation.py` has the nuts and
bolts but won't dump out to a nice structured array yet.

## how to use it

1. set up a YAML file with experiment parameters as in `motherMachine2`
2. once you start syncing files from the microscope computer, launch
`mm3-CompileSubtractAgent.py`
3. when somewhere between 30-100 timepoints have elapsed, open another terminal
and launch `mm3-ChannelPicker.py`. this code will run fine with a remote X
session so you can pick channels from the comfort of your recliner, but use the
`-C` switch on `ssh` to make the speed a little more reasonable.

## notes & warnings

1. image subdirectory polling only works on Linux for now. on OS X, put every
TIFF in `image_directory`.
2. this software will only process data correctly if it has multi-plane TIFFs.
3. killing and restarting `mm3-CompileSubtractAgent` does **NOT** work unless
`originals/`, `subtracted/` and `known_files.mshl` are deleted first.
