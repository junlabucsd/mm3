#!/usr/bin/env python3
from __future__ import print_function, division
import six

# import modules
import sys
import os
# import time
import re
import inspect
import argparse
import yaml
from pprint import pprint # for human readable file output
try:
    import cPickle as pickle
except:
    import pickle
import numpy as np
from scipy.io import savemat

from skimage import measure, io
from tensorflow.keras import models

from matplotlib import pyplot as plt # for debugging

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

import mm3_helpers as mm3

def extract_fov_and_peak_ids(infile_name):

    fov_id = mm3.get_fov(infile_name)
    peak_id = mm3.get_peak(infile_name)

    return (fov_id,peak_id)

def track_single_file(
    phase_file_name,
    seg_file_name,
    params,
    namespace):

    mm3.information("Tracking cells in {}.".format(seg_file_name))
    # load model to pass to algorithm
    mm3.information("Loading model...")

    params['tracking']['migrate_model'] = namespace.migrate_modelfile
    params['tracking']['child_model'] = namespace.child_modelfile
    params['tracking']['appear_model'] = namespace.appear_modelfile
    params['tracking']['die_model'] = namespace.die_modelfile
    params['tracking']['disappear_model'] = namespace.disappear_modelfile
    params['tracking']['born_model'] = namespace.born_modelfile
    
    model_dict = mm3.get_tracking_model_dict()

    fov_id,peak_id = extract_fov_and_peak_ids(phase_file_name)

    tracks = {}
    track_loop(
        fov_id,
        peak_id,
        params,
        tracks,
        model_dict,
        phase_file_name=phase_file_name,
        seg_file_name=seg_file_name
    )

    track_file_name = "{}_xy{:0=3}_p{:0=4}_tracks.pkl".format(
        params['experiment_name'],
        fov_id,
        peak_id
    )

    with open(track_file_name, 'wb') as cell_file:
        pickle.dump(tracks, cell_file)
    
    sys.exit("Completed tracking cells in stack {}.".format(seg_file_name))

def run_cells(tracks,
              peak_id,
              fov_id,
              params,
              predictions_dict,
              regions_by_time,
              born_threshold = 0.85,
              appear_threshold = 0.85):

    G,graph_df = mm3.initialize_track_graph(peak_id=peak_id,
                                            fov_id=fov_id,
                                            experiment_name=params['experiment_name'],
                                            predictions_dict=predictions_dict,
                                            regions_by_time = regions_by_time,
                                            born_threshold=born_threshold,
                                            appear_threshold=appear_threshold)

    tracks.update(mm3.create_lineages_from_graph(G, graph_df, fov_id, peak_id))

def run_foci(tracks,
             peak_id,
             fov_id,
             params,
             predictions_dict,
             regions_by_time,
             Cells,
             appear_threshold = 0.85,
             max_cell_number = 6):

    G,graph_df = mm3.initialize_focus_track_graph(
        peak_id=peak_id,
        fov_id=fov_id,
        experiment_name=params['experiment_name'],
        predictions_dict=predictions_dict,
        regions_by_time = regions_by_time,
        appear_threshold=appear_threshold,
    )

    tracks.update(mm3.create_focus_lineages_from_graph(G, graph_df, fov_id, peak_id, Cells, max_cell_number))


def track_loop(
    fov_id,
    peak_id,
    params,
    tracks,
    model_dict,
    cell_number = 6, 
    data_number = 9,
    img_file_name = None,
    seg_file_name = None,
    track_type = 'cells',
    max_cell_number = 6):

    if img_file_name is None:

        if track_type == 'cells':
            seg_stack = mm3.load_stack(fov_id, peak_id, color=params['seg_img'])
            img_stack = mm3.load_stack(fov_id, peak_id, color=params['phase_plane'])
        elif track_type == 'foci':
            seg_stack = mm3.load_stack(fov_id, peak_id, color=params['seg_img'])
            img_stack = mm3.load_stack(fov_id, peak_id, color=params['foci']['foci_plane'])

    else:
        seg_stack = io.imread(seg_file_name)
        img_stack = io.imread(img_file_name)

    # run predictions for each tracking class
    # consider only the top six cells for a given trap when doing tracking
    frame_number = seg_stack.shape[0]

    # sometimes a phase contrast image is missed and has no signal.
    # This is a workaround for that problem
    no_signal_frames = []
    for k,img in enumerate(img_stack):
        if track_type == 'foci':
            if np.max(img) < 100:
                no_signal_frames.append(k)
        elif track_type == 'cells':
            # if the mean phase image signal is less than 200, add its index to list
            if np.mean(img) < 200:
                no_signal_frames.append(k)

    # loop through segmentation stack and replace frame from missed phase image
    #   with the prior frame.
    for k,label_img in enumerate(seg_stack):
        if k in no_signal_frames:
            seg_stack[k,...] = seg_stack[k-1,...]

    if track_type == 'cells':
        regions_by_time = [measure.regionprops(label_image=img) for img in seg_stack]
    elif track_type == 'foci':
        with open(p['cell_dir'] + '/all_cells.pkl', 'rb') as cell_file:
            Cells = pickle.load(cell_file)
        regions_by_time = []
        for i,img in enumerate(seg_stack):
            regs = measure.regionprops(label_image=img, intensity_image=img_stack[i,:,:])
            regs_sorted = mm3.sort_regions_in_list(regs)
            regions_by_time.append(regs_sorted)

    if track_type == 'cells':
        # have generator yield info for top six cells in all frames
        prediction_generator = mm3.PredictTrackDataGenerator(regions_by_time, batch_size=frame_number, dim=(cell_number,5,data_number), track_type=track_type)
    elif track_type == 'foci':
        prediction_generator = mm3.PredictTrackDataGenerator(
            regions_by_time,
            batch_size=frame_number,
            dim=(cell_number,5,data_number),
            track_type=track_type,
            img_stack=img_stack,
            images=True,
            img_dim=(5,256,32)
        )
    cell_info = prediction_generator.__getitem__(0)

    predictions_dict = {}
    # run data through each classification model
    for key,mod in model_dict.items():

        # Run predictions and add to dictionary
        if key in ['zero_cell_model', 'one_cell_model' , 'two_cell_model', 'geq_three_cell_model']:
            continue

        mm3.information('Predicting probability of {} events in FOV {}, trap {}.'.format('_'.join(key.split('_')[:-1]), fov_id, peak_id))
        predictions_dict['{}_predictions'.format(key)] =  mod.predict(cell_info)

    if track_type == 'cells':
        run_cells(
            tracks,
            peak_id,
            fov_id,
            params,
            predictions_dict,
            regions_by_time,
        )

    elif track_type == 'foci':
        pred_dict = {}
        (
            outbound1,
            outbound2,
            outbound3,
            outbound4,
            outbound5,
            outbound6,
            pred_dict['appear_model_predictions']
        ) = predictions_dict['all_model_predictions']
        # for this in predictions_dict['all_model_predictions']:
        #     print(this.shape)
        # pred_dict['appear_model_predictions'],pred_dict['disappear_model_predictions'],pred_dict['appear_model_predictions'] = predictions_dict['all_model_predictions']

        # take the -2nd element of each outbound array. the -1st is for "no focus", -2nd is for 'disappear, 0:6 are for migrate.
        pred_dict['disappear_model_predicitons'] = np.transpose(np.array(
            [outbound1[:,-2],outbound2[:,-2],outbound3[:,-2],outbound4[:,-2],outbound5[:,-2],outbound6[:,-2]]
        ))

        # take the 0:6 elements of each outbound prediction result. 
        pred_dict['migrate_model_predictions'] = np.concatenate(
            [
                outbound1[:,:6],
                outbound2[:,:6],
                outbound3[:,:6],
                outbound4[:,:6],
                outbound5[:,:6],
                outbound6[:,:6],
            ],
            axis=1
        )

        # print(pred_dict['migrate_model_predictions'].shape)

        run_foci(
            tracks,
            peak_id,
            fov_id,
            params,
            pred_dict,
            regions_by_time,
            Cells,
            max_cell_number=max_cell_number,
            appear_threshold=0.85
        )

# when using this script as a function and not as a library the following will execute
if __name__ == "__main__":

    # set switches and parameters
    parser = argparse.ArgumentParser(
        prog='python mm3_Track.py',
        description='Track cells or fluroescent foci and create lineages.'
    )
    subparsers = parser.add_subparsers(help='commands', dest='command')

    # cells
    cell_parser = subparsers.add_parser(
        'cells',
        help = "Track cells",
    )

    # foci
    focus_parser = subparsers.add_parser(
        'foci',
        help = "Track fluorescent foci"
    )

    parser.add_argument(
        '-f',
        '--paramfile',
        type=str,
        required=True,
        help='Yaml file containing parameters.'
    )
    parser.add_argument(
        '-o',
        '--fov',
        type=str,
        required=False,
        help='List of fields of view to analyze. Input "1", "1,2,3", or "1-10", etc.'
    )
    parser.add_argument(
        '--peak',
        type=str,
        required=False,
        help='List of peaks to analyze. Input "1", "1,2,3", or "1-10", etc.'
    )
    # parser.add_argument(
    #     '-j',
    #     '--nproc',
    #     type=int,
    #     required=False,
    #     help='Number of processors to use.'
    # )
    parser.add_argument(
        '-r',
        '--chtc',
        action='store_true',
        required=False,
        help='Add this flag at the command line if the job will run at chtc.'
    )
    cell_parser.add_argument(
        '-p',
        '--phase_file_name',
        type=str,
        required=False,
        help='Name of file containing stack of images for a single fov/peak'
    )
    focus_parser.add_argument(
        '-fl',
        '--fluor_file_name',
        type=str,
        required=False,
        help='Name of file containing stack of fluorescent images for a single fov/peak'
    )
    parser.add_argument(
        '-s',
        '--seg_file_name',
        type=str,
        required=False,
        help='Name of file containing stack of images for a single fov/peak'
    )
    parser.add_argument(
        '--migrate_modelfile',
        type=str,
        required=False,
        help='Path to trained migration model.'
    )
    cell_parser.add_argument(
        '--child_modelfile',
        type=str,
        required=False,
        help='Path to trained child model.'
    )
    parser.add_argument(
        '--appear_modelfile',
        type=str,
        required=False,
        help='Path to trained appear model.'
    )
    cell_parser.add_argument(
        '--die_modelfile',
        type=str,
        required=False,
        help='Path to trained die model.'
    )
    parser.add_argument(
        '--disappear_modelfile',
        type=str,
        required=False,
        help='Path to trained disappear model.'
    )
    cell_parser.add_argument(
        '--born_modelfile',
        type=str,
        required=False,
        help='Path to trained born model.'
    )
    parser.add_argument(
        '--specfile',
        type=str,
        required=False,
        help='Path to specs file.'
    )
    parser.add_argument(
        '--timefile',
        type=str,
        required=False,
        help='Path to file containing time table.'
    )

    namespace = parser.parse_args()

    # Load the project parameters file
    mm3.information('Loading experiment parameters.')
    if namespace.paramfile:
        param_file_path = namespace.paramfile
    else:
        mm3.warning('No param file specified. Using 100X template.')
        param_file_path = 'yaml_templates/params_SJ110_100X.yaml'
    p = mm3.init_mm3_helpers(param_file_path) # initialized the helper library

    if namespace.fov:
        if '-' in namespace.fov:
            user_spec_fovs = range(int(namespace.fov.split("-")[0]),
                                   int(namespace.fov.split("-")[1])+1)
        else:
            user_spec_fovs = [int(val) for val in namespace.fov.split(",")]
    else:
        user_spec_fovs = []

    if namespace.peak:
        if '-' in namespace.peak:
            user_spec_peaks = range(int(namespace.peak.split("-")[0]),
                                   int(namespace.peak.split("-")[1])+1)
        else:
            user_spec_peaks = [int(val) for val in namespace.peak.split(",")]
    else:
        user_spec_peaks = []

    # set segmentation image name for saving and loading segmented images
    if namespace.command == 'cells':
        p['seg_img'] = 'seg_unet'
    elif namespace.command == 'foci':
        p['seg_img'] = 'foci_seg_unet'

    # load specs file
    if namespace.chtc:
        specs = mm3.load_specs(fname=namespace.specfile)
        mm3.load_time_table(fname=namespace.timefile)
    else:
        specs = mm3.load_specs()
        mm3.load_time_table()

    if namespace.command == 'cells':
        if namespace.phase_file_name:
            track_single_file(
                namespace.phase_file_name,
                namespace.seg_file_name,
                p,
                namespace
            )

    if not os.path.exists(p['cell_dir']):
        os.makedirs(p['cell_dir'])

    # make list of FOVs to process (keys of channel_mask file)
    fov_id_list = sorted([fov_id for fov_id in specs.keys()])

    # remove fovs if the user specified so
    if user_spec_fovs:
        fov_id_list[:] = [fov for fov in fov_id_list if fov in user_spec_fovs]

    mm3.information("Processing %d FOVs." % len(fov_id_list))

    mm3.information("Creating cell lineages.")
    mm3.information("Reading track models. This could take a few minutes.")

    # read in models as dictionary
    # keys are 'migrate_model', 'child_model', 'appear_model', 'die_model', 'disappear_model', etc.
    # NOTE on 2019-07-15: For now, some of the models are ignored by the tracking algorithm, as they don't yet perform well
    if namespace.command == 'cells':
        model_dict = mm3.get_tracking_model_dict()
    elif namespace.command == 'foci':
        model_dict = mm3.get_focus_tracking_model_dict()

    # do lineage creation per fov, per trap
    tracks = {}
    for i,fov_id in enumerate(fov_id_list):
        # tracks[fov_id] = {}
        # update will add the output from make_lineages_function, which is a
        # dict of Cell entries, into Cells
        ana_peak_ids = [peak_id for peak_id in specs[fov_id].keys() if specs[fov_id][peak_id] == 1]
        if user_spec_peaks:
            ana_peak_ids[:] = [peak for peak in ana_peak_ids if peak in user_spec_peaks]

        for j,peak_id in enumerate(ana_peak_ids):

            if namespace.command == 'cells':

                track_loop(
                    fov_id,
                    peak_id,
                    p,
                    tracks,
                    model_dict,
                    track_type = namespace.command
                )

            elif namespace.command == 'foci':

                track_loop(
                    fov_id,
                    peak_id,
                    p,
                    tracks,
                    model_dict,
                    data_number = 11,
                    track_type = namespace.command,
                    max_cell_number = 6
                )

    mm3.information("Finished lineage creation.")

    ### Now prune and save the data.
    if namespace.command == 'cells':
        mm3.information("Saving cell data.")

        ### save the cell data. Use the script mm3_OutputData for additional outputs.
        # All cell data (includes incomplete cells)
        if not os.path.isdir(p['cell_dir']):
            os.mkdir(p['cell_dir'])

        with open(p['cell_dir'] + '/all_cells.pkl', 'wb') as cell_file:
            pickle.dump(tracks, cell_file, protocol=pickle.HIGHEST_PROTOCOL)

        if os.path.isfile(os.path.join(p['cell_dir'], 'complete_cells.pkl')):
            os.remove(os.path.join(p['cell_dir'], 'complete_cells.pkl'))

        os.symlink(
            os.path.join(p['cell_dir'], 'all_cells.pkl'),
            os.path.join(p['cell_dir'], 'complete_cells.pkl')
        )

        mm3.information("Finished curating and saving cell data.")

    elif namespace.command == 'foci':
        mm3.information("Saving focus track data.")

        if not os.path.isdir(p['foci_track_dir']):
            os.mkdir(p['foci_track_dir'])

        with open(os.path.join(p['foci_track_dir'], 'all_foci.pkl'), 'wb') as foci_file:
            pickle.dump(tracks, foci_file, protocol=pickle.HIGHEST_PROTOCOL)

        # with open(os.path.join(p['cell_dir'],'all_cells_with_foci.pkl'), 'wb') as cell_file:
        #     pickle.dump(Cells, cell_file, protocol=pickle.HIGHEST_PROTOCOL)

        mm3.information("Finished curating and saving focus data in {} and updated cell data in {}.".format(os.path.join(p['foci_track_dir'], 'all_foci.pkl'),
                                                                                                            os.path.join(p['cell_dir'], 'all_cells_with_foci.pkl')))
