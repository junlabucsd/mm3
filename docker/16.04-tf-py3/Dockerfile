FROM tensorflow/tensorflow:1.13.1-py3
#FROM tensorflow/tensorflow:1.13.1-py3-jupyter
#FROM tensorflow/tensorflow:1.13.1-jupyter

# update repositories
RUN apt-get update && \
    apt -y dist-upgrade

# locales
RUN apt-get install -y locales #tzdata
RUN locale-gen en_US.UTF-8 # en_GB.UTF-8 eu_FR.UTF-8

# set timezone
# RUN rm -f /etc/localtime /etc/timezone && echo "tzdata tzdata/Areas select America" > myfile && echo "tzdata tzdata/Zones/America select Los_Angeles" >> myfile && debconf-set-selections myfile && dpkg-reconfigure -f noninteractive tzdata && rm myfile

# install some packages
RUN apt-get install -y sudo vim less # not sure if it is a good idea to install sudo

# copy installation script into container and execute
COPY install.sh /root/
RUN /bin/bash /root/install.sh
RUN rm -f /root/install.sh
