# mm3 Installation Guide

mm3 runs on Python 2.7. Follow the guide below to get it running.

Python comes native on your computer, but it is recommended to install the popular distribution Anaconda. This is a free distribution that includes most of the common scientific and engineering packages.

mm3 has only been tested on MacOSX.

## Install Anaconda Python 2.7

1. Go to the [Anaconda homepage](https://www.continuum.io/downloads)
2. Download the **Python 2.7** graphical installer.
3. Follow the GUI to install.

### Check the install

1. Open Terminal.
2. Type `which python`.
3. It should say something like `/Users/username/anaconda/bin/python`.

Now when you type `python` in the Terminal it will open the Anaconda version, as opposed to your native version. If it doesn't link to the right Python, then you need to open your `.bash_profile` or similar file and put in the line `export PATH="/Users/username/anaconda/bin:$PATH"` after your $PATH is initially set. See [this thread](http://stackoverflow.com/questions/22773432/mac-using-default-python-despite-anaconda-install).

mm3 relies on a number of other python modules to run. Some of these are contained in the repository in the folder /external_lib. Most other modules and should be included with your Anacondas distribution. If you did not use Anaconda or you get an error and you need to install a module that you do not have, attempt to install it with pip:

>pip install module_name

## Install FFmpeg and Freetype

mm3_MovieMaker.py relies on [FFmpeg](https://ffmpeg.org/) to create .mp4 movies, and [Freetype](https://www.freetype.org/) to create labels. These may already be installed on your computer, but if not, follow these directions. You will need the package manager [Homebrew](https://brew.sh/) as well. You can test if you have Homebrew and FFmpeg by typing `which brew` and `which FFmpeg` into Terminal, respectively.

### Install FFmpeg

1. Open Terminal
2. Type `brew install ffmpeg`

See [here](https://trac.ffmpeg.org/wiki/CompilationGuide/MacOSX) for more information.

### Install Freetype and its Python binding

1. Open Terminal
2. Type `brew install freetype`

The Freetype library should now be installed on your computer. Now we need the Python bindings so Python can use the library. Unfortunately there is not an "easy" way to install this, so you have to do it the classic way.

1. The package is in the `external_lib` folder in mm3, or you can download the package .tar file [here](https://pypi.python.org/pypi/freetype-py).
2. Navigate to the folder in 1 using Terminal.
3. Type `python setup.py build`
4. Type `python setup.py install`

It should now be installed. You can test it by starting a Python session from Terminal and typing `import freetype`. If nothing happens, then you're good to go!
