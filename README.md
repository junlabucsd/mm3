# mm3
## doing the bookkeeping so you don't have to since 2014.

same optimization applies as in `motherMachine2`: patch `numpy` to use FFTW!

no automatic segmentation output yet. using `pyMakeSegmentMaps.py` from the
`analyzer_dev` branch of `motherMachine2` should work as a stop-gap. s.d.b.
wrote new (much faster) recursion-based code in an IPython notebook and hasn't
converted it to a reasonable script yet. `mm3-Segmentation.py` has the nuts and
bolts but won't dump out to a nice structured array yet.
