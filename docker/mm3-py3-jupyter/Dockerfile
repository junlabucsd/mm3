# Tensorflow image based on Ubuntu 16.04 with Python 3 and Jupyter
FROM tensorflow/tensorflow:1.13.1-py3-jupyter

# update repositories
RUN apt-get update && \
    apt -y dist-upgrade

# Choose locales. Set to US.
RUN apt-get install -y locales tzdata
RUN locale-gen en_US.UTF-8 # en_GB.UTF-8 eu_FR.UTF-8

# set timezone. Set to PST.
RUN rm -f /etc/localtime /etc/timezone && echo "tzdata tzdata/Areas select America" > myfile && echo "tzdata tzdata/Zones/America select Los_Angeles" >> myfile && debconf-set-selections myfile && dpkg-reconfigure -f noninteractive tzdata && rm myfile

# Install minimal packages
RUN apt-get install -y sudo vim less

# Copy installation script with Python packages into container and execute
COPY install_mm3_dependencies.sh /root/
RUN /bin/bash /root/install_mm3_dependencies.sh
RUN rm -f /root/install_mm3_dependencies.sh
