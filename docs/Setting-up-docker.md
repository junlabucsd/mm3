# Setting up Docker for mm3

_This tutorial is adapted from one by Guillaume le Treut_

## Introduction

Docker is a containerization software that allows programs such as mm3 to be run in a "container," similar to a virtual machine though a bit lighter. The advantage of the container is that it they have their own dependencies and libraries, such that you can run mm3 without having to modify your system's configuration.

You can check out the [Docker docs](https://docs.docker.com/), though they don't have a great explanation of the general concept. I'll try to do that in the guide below.

## Quick guide

1. Install [Docker](https://www.docker.com/).
2. Build Docker image for mm3.
* Navigate to `mm3/docker/mm3-py3/` and run `docker build -t local/mm3-py3:root .`
3. Run mm3 container.
* Linux: `docker run -it --rm -p 8888:8888 -v /home/USERNAME:/home/USERNAME local/mm3-py3:root`
* OSX: `docker run -it --rm -p 8888:8888 -v /Users/USERNAME:/home/USERNAME local/mm3-py3:root`
* Enter your username where the above commands say `USERNAME`. You can figure out your username by typing the command `whoami`.

You should be able to run mm3 with the above commands, as long as mm3 and your data are in your home folder on your computer. To be able to interact with the GUIs, you need to follow additional steps in **3.5 Using GUIs with Docker containers**.

## Notes before starting

The files required to create the mm3 Docker image are located in `mm3/docker/`. However, you will likely want to edit these files directly before building from them. For that reason, please copy that directory to another location if you plan to push your branch to the main repository, as to not overwrite the defaults in the Dockerfiles.

## 1. Install Docker

### 1.1 Download and install

Install the community engine version for your system:

* [Linux (Ubuntu)](https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-docker-ce)
* [Mac OSX](https://docs.docker.com/docker-for-mac/install/)
* [Windows](https://docs.docker.com/docker-for-windows/install/)

You can test if your installation was successful by running:

`docker run -it --rm hello-world`

### 1.2 Additional configuration

Docker requires root privileges to run. When installing the OSX desktop version, you can give permission and no additional configuration is necessary.

On Linux systems, you need to do some additional steps if you want to avoid entering sudo for every docker command. It is not strictly necessary to do this, but if becomes annoying there are two strategies I know of:

* You can create a docker group and give that group root privileges. See [this](https://docs.docker.com/install/linux/linux-postinstall/).
* You can create a bash alias to replace the `docker` command with `sudo docker`, and give your user name permission to not require a password when executing `sudo docker`. Read below for instructions on the latter strategy.

#### 1.2.1 Avoid entering sudo with bash alias

In order to avoid having to enter `sudo` command every time, we can create an alias. Locate your `.bash_aliases` file in your home (if it does not exist yet, create it). For instance on Mac OSX, just do:

`open -a textedit ~/.bash_aliases`

Then add the following line, save the file and exit:

`alias docker='sudo docker'`

We will do other modifications to the `.bash_aliases` file later, and give a minimal running version at the end of this tutorial.

Finally, we need to make sure that your OS load the aliases written in the `.bash_aliases` file. For this, open the `bash_profile` file:

`open -a textedit ~/.bash_profile`

And make sure you have the following line somewhere (if not add it anywhere):

`. ~/.bash_aliases`

Where the first `.` is important, as well as the following space. You may now close and re-open a new terminal window to have the changes applied.

#### 1.2.2. Avoid entering password each time

First, identify your user name and host name. You obtain your username by entering the command:

`users`

Find your hostname by entering the command:

`hostname`

We now need to grant your user the privilege to execute the `sudo docker` command without having to enter a password. Open (or create) the following file using admin privileges:

`sudo open -a textedit /etc/sudoers.d/docker`

Then, add the following line:

`USERNAME HOSTNAME = NOPASSWD: /usr/local/bin/docker`

where `USERNAME` and `HOSTNAME` need to be replaced by your own values for the user name and host name, that you have determined above.

You can now run docker without entering password, and without the `sudo` command. Try for instance:

`docker run -it --rm hello-world`

## 2. Build Docker image for mm3

### 2.1 Overview

An image is a template system, including its OS and all the dependencies you want to include. We "build" an image once, and then launch a container from it every time we want to run a program, in our case mm3. Unlike a virtual machine, a container usually just runs one program at a time. Images are build from a set of instructions, called a [Dockerfile](https://docs.docker.com/engine/reference/builder/).

Docker has a GitHub like service called [Docker Hub](https://hub.docker.com/). Organizations and individuals can thus upload and share Dockerfiles. You can also use these Dockerfiles as the basis of your own image so that you don't have to build the whole systems from the ground up. For example, you may start a Dockerfile with `FROM ubuntu:16.04` to build upon an [Ubuntu xenial image](https://hub.docker.com/_/ubuntu).

Because we are using TensorFlow, we will build upon the official [TensorFlow image](https://hub.docker.com/r/tensorflow/tensorflow). These images are build on Ubuntu 16.04 and can come with Python 3 as well as Jupyter, so most of the work is already done for us. We just have to add on the additional Python packages that are required by mm3. You can check additional tags for the TensorFlow image, for example, in order to enable GPU usage.

### 2.2 Create Docker image using provided Dockerfile

#### 2.2.1. Edit the pre-made Dockerfiles

Enter the `mm3/docker/mm3-py3/` directory. This directory contains two files, a Dockerfile and an additional file named `install_mm3_dependencies.sh` which contains the packages specific to mm3. You may want to edit the Dockerfile, for example, to change the time zone of the image.

_Feel free to copy and edit the mm3/docker/ directory and run the following commands on your edited version. However, avoid uploading personal changes to the Dockerfiles to the mm3 repository._

#### 2.2.2. Build the Docker image

In the same directory as the Dockerfile, type:

`docker build -t local/mm3-py3:root .`

This should take a few minutes because you are installing everything from Ubuntu to TensorFlow to the Python packages. Note the final `.`, to indicate that the Dockerfile is in this directory. The `-t` option is for "tag", which is the name of the image. The format is usually `repository/name:version`. You will reference the tag when running the image to make a container.

You can check the images on your system by running:

`docker images`

## 3. Run mm3 container

Now that we have an image, we can run a container from that image. The container is like the virtual machine in which you run mm3. You can open multiple containers from a single image at the same time (and are in fact encouraged to, as usually one container does one job at a time).

### 3.1 Launch your container

Run:

`docker run -it --rm -p 8888:8888 local/mm3-py3:root`

* `-it` is for interactive mode so you can use the terminal (very often used).
* `--rm` is to remove the container automatically when it closes (very often used).
* `-p` publishes container's port to the host computer. This may only be required for the Jupyter notebook. Should test and edit the directions below as needed.
* `local/mm3-py3:root` is the tag of our image we chose when building.
* Exit the container with `CTRL + D`

You can now navigate the Docker container, start a Python session, etc.

### 3.2 Run with access to your drives

You'll notice that by using the above `run` command, the container starts in the `/tf` directory. You can move around to the other directories, but you have no access to the directories of your computer. This is by design, so you cannot mess up your system. However, you obviously want to be able to access some of those files (such as mm3), and be able to save your work which otherwise will be destroyed when the container is closed. This is done with the `--volume-` or `-v` option.

We will link the home directory of our computer (the host) to the home folder of the container. In the command below, change `USERNAME` to the username on your computer.

On Linux run:

`docker run -it --rm -p 8888:8888 -v /home/USERNAME:/home/USERNAME local/mm3-py3:root`

On Mac run:

`docker run -it --rm -p 8888:8888 -v /Users/USERNAME:/home/USERNAME local/mm3-py3:root`

In the container, use `cd` to navigate to your home directory. You should see all your files. Whatever is saved in this volume in your Docker session will be available after the container is closed.

The `-v` flag is of the form `/host/directory:/container/directory`. You can bind as many volumes as you want. Here are some examples of other volumes you may want to bind.

On Mac: `docker run -it --rm -p 8888:8888 -v /Users/USERNAME:/home/USERNAME -v /Users/USERNAME:/Users/USERNAME local/mm3-py3:root`. Because Ubuntu and OSX have different home folder names, this will make it so paths you specified before to `/Users/USERNAME` will not break in the container.

Bind an external disk (Linux): `docker run -it --rm -p 8888:8888 -v /home/USERNAME:/home/USERNAME -v /media/DRIVENAME:/media/DRIVENAME local/mm3-py3:root`

Bind GoogleDrive (OSX): `docker run -it --rm -p 8888:8888 -v /Users/USERNAME:/home/USERNAME -v /Users/USERNAME:/Users/USERNAME -v /Volumes/GoogleDrive:/media/GoogleDrive -v /Volumes/GoogleDrive:/Volumes/GoogleDrive local/mm3-py3:root`. Note how OSX calls the `/media` directory `/Volumes`, so we use the same trick as before.

### 3.3 Using your own username within Docker containers

By default, containers are run as the root user. This is the reason we put root at the end of our image tag.

Generally, it's not a huge deal, but can be inconvenient as all files created by the container will owned naturally by root. This may cause permission issues when accessing those files on your host computer. In fact, the TensorFlow Docker image complains of this upon start up.

The image suggest running the container with the `-u $(id -u):$(id -g)` command, which is perfectly fine as far as I understand.

Read on for another, perhaps more permanent solution where we will build another image based on the previous on, but using your username. Building one image from another is a common practice with Docker.

#### 3.3.1 Find your username

We are going to build a Docker image upon the previous one but with the user information of our host computer. To determine your username, user id (uid), group name, and group id (gid), run:

`id`

The four variables should be on the first line in the format `uid=USERID(USERNAME) gid=GROUPID(GROUPNAME)`. Navigate to the `docker/username_linux` or `docker/username_mac` directory depending on your system, and edit the Dockerfile with your information. On OSX, the information is likely `uid=504(janedoe) gid=20(staff)`.

#### 3.3.2 Remove inconvenient IDs from the docker image for OSX

_Only do this step if you are on Mac OSX_

The Docker image runs a Linux Ubuntu OS. Like all Linux based system, IDs smaller than 1000 are reserved for system users.  This is somehow not respected by Mac OS: you can see that both 504 (the user ID) and 20 (the group ID) are smaller than 1000. Therefore, we need to include another part to the Dockerfile to delete the user and group in the Docker image that are duplicate of the user and group ID on OS X. Then we can create a user and a group in the image that match our own. Another way to do would be to modify all the files on your Mac computer to change your IDs, but it safer to alter the Docker image, which is a virtual object.

The `docker/username_mac/Dockerfile` contains code to fix this issue. In the present case, only group ID (20) exists in the mm3-py3 Docker image. If your group ID is different, you may need to adapt this code. Otherwise, the the code should be sufficient.

#### 3.3.3 Build image using your username

Navigate to either `docker/username_linux` or `docker/username_mac` with your edited Dockerfile and run with your username in place of `USERNAME`, which will become part of the image tag:

`docker build -t local/mm3-py3:USERNAME .`

Building this container should not take long because we are building it upon the image local/mm3-py3:root. When you run this container, with `docker run -it --rm -p 8888:8888 -v /home/USERNAME:/home/USERNAME local/mm3-py3:USERNAME`, all files created by the container should be of the same user at your host computer. TensorFlow should no longer throw an error upon start up.

### 3.5 Using GUIs with Docker containers

mm3 has a number of GUIs. If you want to use those GUIs in the container, we have to connect to the display of the host computer. We do this when we launch the container by setting the environmental variable of the container with the `-e` option. The method is different for Linux and OSX as OSX does not natively have X11.

#### 3.5.1 Connecting to the host's display in Linux

Because Linux distributions have X11 natively, we can use that directly. For your run command use:

`docker run -it --rm -p 8888:8888 -e DISPLAY=$DISPLAY -v /home/USERNAME:/home/USERNAME local/mm3-py3:USERNAME`

You should be good to go.

#### 3.5.2 Connecting to the host's display in OSX

We need to install X11 with XQuartz as well as socat. You can install both with brew. See [this tutorial](https://cntnr.io/running-guis-with-docker-on-mac-os-x-a14df6a76efc).

With those installed, we start an X11 tunnel in a separate terminal window with:

`socat TCP-LISTEN:6000,reuseaddr,fork UNIX-CLIENT:\"$DISPLAY\"`

Then we use the run command with our IP address (note where it should be entered below at `_IP_ADDRESS_`):

`docker run -it --rm -p 8888:8888 -e DISPLAY=_IP_ADDRESS_:0 -v /home/USERNAME:/home/USERNAME local/mm3-py3:USERNAME`

The `bash_aliases` file  described in **3.5 Setting up shortcuts** has two useful commands, `x11-tunnel` and `get_IP` that can be used to help simplify the commands above.

### 3.4 Running a Jupyter notebook

TBA

`docker build -t local/mm3-py3-jupyter:root .`


`docker build -t local/mm3-py3-jupyter:USERNAME`


`docker run -it --rm -p 8888:8888 -e DISPLAY=$(get_IP):0 -w=$PWD -v /Users/jt:/home/jt -v /Users/jt:/Users/jt -v /Volumes/JunLabSSD_04:/media/JunLabSSD_04 -v /Volumes/JunLabSSD_04:/Volumes/JunLabSSD_04 local/mm3-py3-jupyter:jt /bin/bash`

for some reason this does not require x11-tunnel

### 3.5 Setting up shortcuts

Typing the big run command every time is annoying, so to get around that add an alias to your `.bash_aliases` file in your home folder. Look in section **1.2.1 Avoid entering sudo with bash alias** for directions on setting up a `.bash_aliases` file.

See the file `mm3/docker/bash_aliases` for examples. Edit the username, mounts, etc, for your system.

Finally, see the [Docker run docs](https://docs.docker.com/engine/reference/commandline/run/) to see some other options you may want to include to your `docker run` command. Here are some suggestions:

* `-w=$PWD` will launch the container in your current directory.
* `-h=mm3` will change the host name inside the container to `mm3`.
* `--name=mm3-py3` will name your container `mm3-py3` such that you can launch it with `docker start`. This is in a way an alternative to setting up an alias.
