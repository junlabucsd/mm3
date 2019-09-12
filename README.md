# mm3
### Segmentation and tracking software for mother machine experiments

mm3 is a image analysis pipeline for mother machine experiments. It takes as input raw timelapse images and produces as output cell objects containing information such as elongation rate, birth size, etc. There is support for fluorescent planes.

The pipeline is modular. Segmentation is done by either and Otsu based method or a "U-net" convolution network. There are two tracking algorithms which act on the segmented images.

The pipeline is implemented in Python 3.5. A Docker container for the appropriate Python environment is provided. The docs folder contains a guide for setting up the Docker and using the pipeline. 
