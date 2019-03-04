#!/bin/bash

# This is done in the Dockerfile
# echo "update sources and upgrade distro"
# apt-get update
# apt dist-upgrade -y

# These things are already installed with the tf image, except tk
# -y means yes
# echo "install python version in repositories"
# apt-get install -y python python-dev python-pip python-tk
apt-get install -y python3-tk

echo "install other packages for imaging and plotting"
apt-get install -y libreadline-dev libsqlite3-dev libbz2-dev libssl-dev
apt-get install -y libblas-dev liblapack-dev libatlas-dev
apt-get install -y libpng-dev libfreetype6-dev tk-dev pkg-config
apt-get install -y ffmpeg

echo "install additional python packages with pip"
# Note: pip softlinks to pip3
# python -m ensurepip
# python -m pip install pip --upgrade
# python -m pip install wheel

# python -m pip install numpy
python -m pip install scipy
python -m pip install matplotlib
python -m pip install seaborn
python -m pip install Pillow
python -m pip install scikit-image
python -m pip install pyYAML
# python -m pip install h5py
python -m pip install pandas
python -m pip install pims_nd2
# python -m pip install opencv-python

## python binding
python -m pip install freetype-py

# install jupyter
# python -m pip install jupyter

python -m pip install PyQt5
sudo apt-get -y install python3-pyqt5
sudo apt-get -y install pyqt5-dev-tools
sudo apt-get -y install qttools5-dev-tools
