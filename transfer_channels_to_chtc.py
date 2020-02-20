#!/usr/bin/env python3

# import modules
import sys
import os
import argparse
import yaml
import inspect
from pprint import pprint
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
parser.add_argument('-s', '--transfer_segmentation',  action='store_true',
                    required=False, help='Add this option at command line to send segmentation files.')
parser.add_argument('-c', '--transfer_all_channels',  action='store_true',
                    required=False, help='Add this option at command line to send all channels, not just phase.')
parser.add_argument('-j', '--transfer_job_file',  action='store_true',
                    required=False, help='Add this option at command line to compile text file containing file names for job submission at chtc.')
namespace = parser.parse_args()

# Load the project parameters file
mm3.information('Loading experiment parameters.')
if namespace.paramfile:
    param_file_path = namespace.paramfile
else:
    mm3.warning('No param file specified. Using 100X template.')
    param_file_path = 'yaml_templates/params_SJ110_100X.yaml'

param_file_path = os.path.join(os.getcwd(), param_file_path)
p = mm3.init_mm3_helpers(param_file_path) # initialized the helper library

# load specs file
specs = mm3.load_specs()

# identify files to be copied to chtc
files_to_transfer = []
if namespace.transfer_job_file:
    job_file_name = '{}_files_list.txt'.format(os.path.basename(param_file_path.split('.')[0]))
    job_file = open(job_file_name,'w')
    spec_file_name = os.path.join(p['ana_dir'], 'specs.yaml')
    new_spec_file_name = '{}_specs.yaml'.format(p['experiment_name'])
    time_file_name = os.path.join(p['ana_dir'], 'time_table.yaml')
    new_time_file_name = '{}_time.yaml'.format(p['experiment_name'])
    new_param_file_name = '{}_params.yaml'.format(p['experiment_name'])

for fov_id,peak_ids in specs.items():
    for peak_id,val in peak_ids.items():
        if val == 1:

            if namespace.transfer_all_channels:
                base_name = '{}_xy{:0=3}_p{:0=4}_*.tif'.format(
                    p['experiment_name'],
                    fov_id,
                    peak_id
                )
                match_list = glob.glob(os.path.join(p['chnl_dir'],base_name))
                match_list.sort()

            else:
                base_name = '{}_xy{:0=3}_p{:0=4}_{}.tif'.format(
                    p['experiment_name'],
                    fov_id,
                    peak_id,
                    p['phase_plane']
                )
                match_list = glob.glob(os.path.join(p['chnl_dir'],base_name))

            if namespace.transfer_segmentation:
                base_name = '{}_xy{:0=3}_p{:0=4}_{}.tif'.format(
                    p['experiment_name'],
                    fov_id,
                    peak_id,
                    'seg_unet'
                )
                fname = os.path.join(p['seg_dir'],base_name)
                match_list.append(fname)

            # pprint(match_list)

            files_to_transfer.extend(match_list)
            match_base_names = [os.path.basename(fname) for fname in match_list]
            
            if namespace.transfer_job_file:

                match_base_names.append(new_spec_file_name)
                match_base_names.append(new_time_file_name)
                match_base_names.append(param_file_path.split('/')[-1])
                    
                line_to_write = ','.join(match_base_names)
                line_to_write = line_to_write + '\n'

                job_file.write(line_to_write)


if namespace.transfer_job_file:
    job_file.close()
    files_to_transfer.append(job_file_name)
            
# files_to_transfer.append(param_file_path)

print("You'll be sending {} files total to chtc.".format(len(files_to_transfer)))

# connect to chtc
ssh = paramiko.SSHClient() 
ssh.load_host_keys(os.path.expanduser(os.path.join("~", ".ssh", "known_hosts")))
username = input("Username: ")
server = input("Hostname: ")
password = getpass("Password for {}@{}: ".format(username,server))
ssh.connect(server, username=username, password=password)

# copy files
sftp = ssh.open_sftp()
for localpath in files_to_transfer:

    print(localpath)
    remotepath = localpath.split('/')[-1]
    sftp.put(localpath, remotepath)

sftp.put(param_file_path, new_param_file_name)

if namespace.transfer_job_file:

    sftp.put(spec_file_name, new_spec_file_name)
    sftp.put(time_file_name, new_time_file_name)

sftp.close()
ssh.close()