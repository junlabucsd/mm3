#!/usr/bin/env python3

# import modules
import sys
import os
import argparse
import yaml
import inspect
from getpass import getpass
import glob
try:
    import cPickle as pickle
except:
    import pickle

import paramiko


# user modules
# realpath() will make your script run, even if you symlink it
cmd_folder = os.path.realpath(os.path.abspath(
                              os.path.split(inspect.getfile(inspect.currentframe()))[0]))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

# This makes python look for modules in ./external_lib
cmd_subfolder = os.path.realpath(os.path.abspath(
                                 os.path.join(os.path.split(inspect.getfile(
                                 inspect.currentframe()))[0], "external_lib")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

# this is the mm3 module with all the useful functions and classes
import mm3_helpers as mm3


# set switches and parameters
parser = argparse.ArgumentParser(prog='python mm3_Compile.py',
                                    description='Identifies and slices out channels into individual TIFF stacks through time.')
parser.add_argument('-f', '--paramfile',  type=str,
                    required=True, help='Yaml file containing parameters.')
namespace = parser.parse_args()

# Load the project parameters file
mm3.information('Loading experiment parameters.')
if namespace.paramfile:
    param_file_path = namespace.paramfile
else:
    mm3.warning('No param file specified. Using 100X template.')
    param_file_path = 'yaml_templates/params_SJ110_100X.yaml'
p = mm3.init_mm3_helpers(param_file_path) # initialized the helper library

# load specs file
specs = mm3.load_specs()

# identify files to be copied to chtc
files_to_transfer = []

for fov_id,peak_ids in specs.items():
    for peak_id,val in peak_ids.items():
        if val == 1:
            base_name = '{}_xy{:0=3}_p{:0=4}_{}.tif'.format(
                p['experiment_name'],
                fov_id,
                peak_id,
                p['phase_plane']
            )
            fname = os.path.join(p['chnl_dir'],base_name)
            files_to_transfer.append(fname)

# connect to chtc
ssh = paramiko.SSHClient() 
ssh.load_host_keys(os.path.expanduser(os.path.join("~", ".ssh", "known_hosts")))
server = input("Hostname: ")
username = input("Username: ")
password = getpass("Password for {}@{}: ".format(username,server))
ssh.connect(server, username=username, password=password)

# copy files
sftp = ssh.open_sftp()
for localpath in files_to_transfer:

    print(localpath)
    remotepath = localpath.split('/')[-1]
    sftp.put(localpath, remotepath)

sftp.close()
ssh.close()