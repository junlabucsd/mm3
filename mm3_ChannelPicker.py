#!/usr/bin/python
from __future__ import print_function

# import modules
import sys
import os
import time
import inspect
import getopt
import yaml
from pprint import pprint # for human readable file output
try:
    import cPickle as pickle
except:
    import pickle
import numpy as np
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# global settings mpl
plt.rcParams['axes.linewidth']=0.5

from skimage.exposure import rescale_intensity # for displaying in GUI
from scipy.misc import imresize
import multiprocessing
from multiprocessing import Pool
import warnings
import h5py

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

# supress the warning this always gives
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import tifffile as tiff

# this is the mm3 module with all the useful functions and classes
import mm3_helpers as mm3

### functions
def fov_plot_channels(fov_id, crosscorrs, specs, outputdir='.',phase_plane='c1'):
    '''
    Creates a plot with the channels with guesses for empties and full channels.
    The plot is saved in PDF format.


    Parameters
    fov_id : str
        file name of the hdf5 file name in originals
    crosscorrs : dictionary
        dictionary for cross correlation values for all fovs.
    specs: dictionary
        dictionary for channal assignment (Analyze/Don't Analyze/Background).

    '''

    mm3.information("Plotting channels for FOV %d." % fov_id)

    # set up figure for user assited choosing
    n_peaks = len(specs[fov_id].keys())
    axw=1
    axh=4*axw
    nrows=3
    ncols=int(n_peaks)
    fig = plt.figure(num='none', facecolor='w',figsize=(ncols*axw,nrows*axh))
    gs = gridspec.GridSpec(nrows,ncols,wspace=0.5,hspace=0.1,top=0.90)

    # plot the peaks peak by peak using sorted list
    sorted_peaks = sorted([peak_id for peak_id in specs[fov_id].keys()])
    npeaks = len(sorted_peaks)

    for n, peak_id in enumerate(sorted_peaks):
        if crosscorrs:
            peak_xc = crosscorrs[fov_id][peak_id] # get cross corr data from dict

        # load data for figure
        image_data = mm3.load_stack(fov_id, peak_id, color=phase_plane)

        first_img = rescale_intensity(image_data[0,:,:]) # phase image at t=0
        last_img = rescale_intensity(image_data[-1,:,:]) # phase image at end

        # append an axis handle to ax list while adding a subplot to the figure which has a
        axhi = fig.add_subplot(gs[0,n])
        axmid = fig.add_subplot(gs[1,n])
        axlo = fig.add_subplot(gs[2,n])

        # plot the first image in each channel in top row
        ax=axhi
        ax.imshow(first_img,cmap=plt.cm.gray, interpolation='nearest')
        ax.axis('off')
        ax.set_title(str(peak_id), fontsize = 12)
        if n == 0:
            ax.set_ylabel("first time point")

        # plot middle row using last time point with highlighting for empty/full
        ax=axmid
        ax.axis('off')
        #ax.imshow(last_img,cmap=plt.cm.gray, interpolation='nearest')
        #H,W = last_img.shape
        #img = np.zeros((H,W,3))
        if specs[fov_id][peak_id] == 1: # 1 means analyze, show green
            #img[:,:,1]=last_img
            cmap=plt.cm.Greens_r
        elif specs[fov_id][peak_id] == 0: # 0 means reference, show blue
            #img[:,:,2]=last_img
            cmap=plt.cm.Blues_r
        else: # otherwise show red, means don't analyze
            #img[:,:,0]=last_img
            cmap=plt.cm.Reds_r
        ax.imshow(last_img,cmap=cmap, interpolation='nearest')

        # format
        if n == 0:
            ax.set_ylabel("last time point")

        # finally plot the cross correlations a cross time
        ax=axlo
        if crosscorrs: # don't try to plot if it's not there.
            ccs = peak_xc['ccs'] # list of cc values
            ax.plot(ccs,range(len(ccs)))
            ax.set_title('avg=%1.2f' % peak_xc['cc_avg'], fontsize = 8)
        else:
            ax.plot(np.zeros(10), range(10))

        ax.get_xaxis().set_ticks([0.8,0.9,1.0])
        ax.set_xlim((0.8,1))
        ax.tick_params('x',labelsize=8)
        if not n == 0:
            ax.set_yticks([])
        else:
            ax.set_ylabel("time index, CC on X")


    fig.suptitle("FOV {:d}".format(fov_id),fontsize=14)
    fileout=os.path.join(outputdir,'fov_xy{:03d}.pdf'.format(fov_id))
    fig.savefig(fileout,bbox_inches='tight',pad_inches=0)
    plt.close('all')
    mm3.information("Written FOV {}'s channels in {}".format(fov_id,fileout))

    return specs

# funtion which makes the UI plot
def fov_choose_channels_UI(fov_id, crosscorrs, specs, UI_images):
    '''Creates a plot with the channels with guesses for empties and full channels,
    and requires the user to choose which channels to use for analysis and which to
    average for empties and subtraction.

    Parameters
    fov_file : str
        file name of the hdf5 file name in originals
    fov_xcorrs : dictionary
        dictionary for cross correlation values for all fovs.

    Returns
    bgdr_peaks : list
        list of peak id's (int) of channels to be used for subtraction
    spec_file_pkl : pickle file
        saves the lists cell_peaks, bgrd_peaks, and drop_peaks to a pkl file

    '''

    mm3.information("Starting channel picking for FOV %d." % fov_id)

    # define functions here so they have access to variables
    # for UI. change specification of channel
    def onclick_cells(event):
        try:
            peak_id = int(event.inaxes.get_title())
        except AttributeError:
            return

        # reset image to be updated based on user clicks
        ax_id = sorted_peaks.index(peak_id) * 3 + 1
        new_img = last_imgs[sorted_peaks.index(peak_id)]
        ax[ax_id].imshow(new_img, cmap=plt.cm.gray, interpolation='nearest')

        # if it says analyze, change to empty
        if specs[fov_id][peak_id] == 1:
            specs[fov_id][peak_id] = 0
            ax[ax_id].imshow(np.dstack((ones_array*0.1, ones_array*0.1, ones_array)), alpha=0.25)
            #mm3.information("peak %d now set to empty." % peak_id)

        # if it says empty, change to don't analyze
        elif specs[fov_id][peak_id] == 0:
            specs[fov_id][peak_id] = -1
            ax[ax_id].imshow(np.dstack((ones_array, ones_array*0.1, ones_array*0.1)), alpha=0.25)
            #mm3.information("peak %d now set to ignore." % peak_id)

        # if it says don't analyze, change to analyze
        elif specs[fov_id][peak_id] == -1:
            specs[fov_id][peak_id] = 1
            ax[ax_id].imshow(np.dstack((ones_array*0.1, ones_array, ones_array*0.1)), alpha=0.25)
            #mm3.information("peak %d now set to analyze." % peak_id)

        plt.draw()
        return

    # set up figure for user assited choosing
    n_peaks = len(specs[fov_id].keys())
    fig = plt.figure(figsize=(int(n_peaks/2), 12))
    fig.set_size_inches(int(n_peaks/2),12)
    ax = [] # for axis handles

    # plot the peaks peak by peak using sorted list
    sorted_peaks = sorted([peak_id for peak_id in specs[fov_id].keys()])
    npeaks = len(sorted_peaks)
    last_imgs = [] # list that holds last images for updating figure

    for n, peak_id in enumerate(sorted_peaks, start=1):
        if crosscorrs:
            peak_xc = crosscorrs[fov_id][peak_id] # get cross corr data from dict

        # load data for figure
        # image_data = mm3.load_stack(fov_id, peak_id, color='c1')

        # first_img = rescale_intensity(image_data[0,:,:]) # phase image at t=0
        # last_img = rescale_intensity(image_data[-1,:,:]) # phase image at end
        last_imgs.append(UI_images[fov_id][peak_id]['last']) # append for updating later
        # del image_data # clear memory (maybe)

        # append an axis handle to ax list while adding a subplot to the figure which has a
        # column for each peak and 3 rows

        # plot the first image in each channel in top row
        ax.append(fig.add_subplot(3, npeaks, n))
        ax[-1].imshow(UI_images[fov_id][peak_id]['first'],
                      cmap=plt.cm.gray, interpolation='nearest')
        ax = format_channel_plot(ax, peak_id) # format axis and title
        if n == 1:
            ax[-1].set_ylabel("first time point")

        # plot middle row using last time point with highlighting for empty/full
        ax.append(fig.add_subplot(3, npeaks, n + npeaks))
        ax[-1].imshow(UI_images[fov_id][peak_id]['last'],
                      cmap=plt.cm.gray, interpolation='nearest')

        # color image based on if it is thought empty or full
        ones_array = np.ones_like(UI_images[fov_id][peak_id]['last'])
        if specs[fov_id][peak_id] == 1: # 1 means analyze, show green
            ax[-1].imshow(np.dstack((ones_array*0.1, ones_array, ones_array*0.1)), alpha=0.25)
        else: # otherwise show red, means don't analyze
            ax[-1].imshow(np.dstack((ones_array, ones_array*0.1, ones_array*0.1)), alpha=0.25)

        # format
        ax = format_channel_plot(ax, peak_id)
        if n == 1:
            ax[-1].set_ylabel("last time point")

        # finally plot the cross correlations a cross time
        ax.append(fig.add_subplot(3, npeaks, n + 2*npeaks))
        if crosscorrs: # don't try to plot if it's not there.
            ccs = peak_xc['ccs'] # list of cc values
            ax[-1].plot(ccs, range(len(ccs)))
            ax[-1].set_title('avg=%1.2f' % peak_xc['cc_avg'], fontsize = 8)
        else:
            ax[-1].plot(np.zeros(10), range(10))

        ax[-1].set_xlim((0.8,1))
        ax[-1].get_xaxis().set_ticks([])
        if not n == 1:
            ax[-1].get_yaxis().set_ticks([])
        else:
            ax[-1].set_ylabel("time index, CC on X")

    # show the plot finally
    fig.suptitle("FOV %d" % fov_id)

    # enter user input
    # ask the user to correct cell/nocell calls
    cells_handler = fig.canvas.mpl_connect('button_press_event', onclick_cells)
    # matplotlib has difefrent behavior for interactions in different versions.
    if float(mpl.__version__[:3]) < 1.5: # check for verions less than 1.5
        plt.show(block=False)
        raw_input("Click colored channels to toggle between analyze (green), use for empty (blue), and ignore (red).\nPrees enter to go to the next FOV.")
    else:
        print("Click colored channels to toggle between analyze (green), use for empty (blue), and ignore (red).\nClose figure to go to the next FOV.")
        plt.show(block=True)
    fig.canvas.mpl_disconnect(cells_handler)
    plt.close()

    return specs

# function for better formatting of channel plot
def format_channel_plot(ax, peak_id):
    '''Removes axis and puts peak as title from plot for channels'''
    ax[-1].get_xaxis().set_ticks([])
    ax[-1].get_yaxis().set_ticks([])
    ax[-1].set_title(str(peak_id), fontsize = 8)
    return ax

# function to preload all images for all FOVs, hopefully saving time
def preload_images(specs, fov_id_list):
    '''This dictionary holds the first and last image
    for all channels in all FOVS. It is passed to the UI so that the
    figures can be populated much faster
    '''
    global p

    # Intialized the dicionary
    UI_images = {}

    for fov_id in fov_id_list:
        UI_images[fov_id] = {}
        for peak_id in specs[fov_id].keys():
            image_data = mm3.load_stack(fov_id, peak_id, color=p['phase_plane'])
            UI_images[fov_id][peak_id] = {'first' : None, 'last' : None} # init dictionary
             # phase image at t=0. Rescale intenstiy and also cut the size in half
            UI_images[fov_id][peak_id]['first'] = imresize(image_data[0,:,:], 0.5)
            # phase image at end
            UI_images[fov_id][peak_id]['last'] = imresize(image_data[-1,:,:], 0.5)

    return UI_images

### For when this script is run from the terminal ##################################
if __name__ == "__main__":
    # hardcoded parameters
    do_crosscorrs = True
    interactive = True
    nproc = 6
    specfile = None

    # get switches and parameters
    try:
        unixoptions="f:o:j:s:ci"
        gnuoptions=["paramfile=","fov=","nproc=","specfile=","saved-cross-correlations","interactive"]
        opts, args = getopt.getopt(sys.argv[1:],unixoptions,gnuoptions)
        # switches which may be overwritten
        user_spec_fovs = []
        param_file_path = 'yaml_templates/params_SJ110_100X.yaml'
    except getopt.GetoptError:
        mm3.warning('No arguments detected (-f -o).')

    for opt, arg in opts:
        if opt in ['-f','--paramfile']:
            param_file_path = arg # parameter file path
        if opt in ['-o','--fov']:
            try:
                for fov_to_proc in arg.split(","):
                    user_spec_fovs.append(int(fov_to_proc))
            except:
                mm3.warning("Couldn't convert argument to an integer:",arg)
                raise ValueError
        if opt in ['-j', '--nproc']:
            try:
                nproc = int(arg)
            except ValueError:
                mm3.warning("Could not convert \"{}\" to an integer".format(arg))
        if opt in ['-s', '--specfile']:
            try:
                specfile = os.path.relpath(arg)
                if not os.path.isfile(specfile):
                    raise ValueError
            except ValueError:
                mm3.warning("\"{}\" is not a regular file or does not exist".format(specfile))
        if opt in ['-i','--interactive']:
            interactive=True
        if opt in ['-c', '--saved-cross-correlations']:
            do_crosscorrs=False

    # Load the project parameters file
    if len(param_file_path) == 0:
        raise ValueError("A parameter file must be specified (-f <filename>).")
    mm3.information('Loading experiment parameters.')
    with open(param_file_path, 'r') as param_file:
        p = yaml.safe_load(param_file) # load parameters into dictionary

    mm3.init_mm3_helpers(param_file_path) # initialized the helper library

    # assign shorthand directory names
    ana_dir = os.path.join(p['experiment_directory'], p['analysis_directory'])
    chnl_dir = os.path.join(p['experiment_directory'], p['analysis_directory'], 'channels')
    hdf5_dir = os.path.join(p['experiment_directory'], p['analysis_directory'], 'hdf5')

    # load channel masks
    try:
        with open(os.path.join(ana_dir,'channel_masks.pkl'), 'r') as cmask_file:
            channel_masks = pickle.load(cmask_file)
    except:
        mm3.warning('Could not load channel mask file.')
        raise ValueError

    # make list of FOVs to process (keys of channel_mask file), but only if there are channels
    fov_id_list = sorted([fov_id for fov_id, peaks in channel_masks.items() if peaks])

    # remove fovs if the user specified so
    if (len(user_spec_fovs) > 0):
        fov_id_list = [fov for fov in fov_id_list if fov in user_spec_fovs]

    mm3.information("Found %d FOVs to process." % len(fov_id_list))

    ### Cross correlations ########################################################################
    if not do_crosscorrs: # load precalculate ones if indicated
        mm3.information('Loading precalculated cross-correlations.')

        try:
            with open(os.path.join(ana_dir,'crosscorrs.pkl'), 'r') as xcorrs_file:
                crosscorrs = pickle.load(xcorrs_file)
        except:
            crosscorrs = None
            mm3.information('Precalculated cross-correlations not found.')

    else:
        # a nested dict to hold cross corrs per channel per fov.
        crosscorrs = {}

        # for each fov find cross correlations (sending to pull)
        for fov_id in fov_id_list:
            mm3.information("Calculating cross correlations for FOV %d." % fov_id)

            # nested dict keys are peak_ids and values are cross correlations
            crosscorrs[fov_id] = {}

            # initialize pool for analyzing image metadata
            pool = Pool(nproc)

            # find all peak ids in the current FOV
            for peak_id in sorted(channel_masks[fov_id].keys()):
                mm3.information("Calculating cross correlations for peak %d." % peak_id)

                # linear loop
                # crosscorrs[fov_id][peak_id] = mm3.channel_xcorr(fov_id, peak_id)

                # # multiprocessing verion
                crosscorrs[fov_id][peak_id] = pool.apply_async(mm3.channel_xcorr,
                                                               args=(fov_id, peak_id,))

            mm3.information('Waiting for cross correlation pool to finish for FOV %d.' % fov_id)

            pool.close() # tells the process nothing more will be added.
            pool.join() # blocks script until everything has been processed and workers exit

            mm3.information("Finished cross correlations for FOV %d." % fov_id)

        # get results from the pool and put the results in the dictionary if succesful
        for fov_id, peaks in crosscorrs.iteritems():
            for peak_id, result in peaks.iteritems():
                if result.successful():
                    # put the results, with the average, and a guess if the channel
                    # is full into the dictionary
                    crosscorrs[fov_id][peak_id] = {'ccs' : result.get(),
                                                   'cc_avg' : np.average(result.get()),
                                                   'full' : np.average(result.get()) < p['channel_picking_threshold']}
                else:
                    crosscorrs[fov_id][peak_id] = False # put a false there if it's bad

        # write cross-correlations to pickle and text
        mm3.information("Writing cross correlations file.")
        with open(os.path.join(ana_dir,"crosscorrs.pkl"), 'w') as xcorrs_file:
            pickle.dump(crosscorrs, xcorrs_file, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(ana_dir,"crosscorrs.txt"), 'w') as xcorrs_file:
            pprint(crosscorrs, stream=xcorrs_file)
        mm3.information("Wrote cross correlations files.")

    ### User selection (channel picking) #####################################################
    if (specfile == None):
        mm3.information('Initializing specifications file.')
        # nested dictionary of {fov : {peak : spec ...}) for if channel should
        # be analyzed, used for empty, or ignored.
        specs = {}

        # if there is cross corrs, use it. Otherwise, just make everything -1
        if crosscorrs:
            # update dictionary on initial guess from cross correlations
            for fov_id, peaks in crosscorrs.items():
                specs[fov_id] = {}
                for peak_id, xcorrs in peaks.items():
                    # update the guess incase the parameters file was changed
                    xcorrs['full'] = xcorrs['cc_avg'] < p['channel_picking_threshold']

                    if xcorrs['full'] == True:
                        specs[fov_id][peak_id] = 1
                    else: # default to don't analyze
                        specs[fov_id][peak_id] = -1
        else: # just set everything to 1 and go forward.
            for fov_id, peaks in channel_masks.items():
                specs[fov_id] = {peak_id: 1 for peak_id in peaks.keys()}

    else:
        with open(specfile,'r') as fin:
            specs = yaml.load(fin)

    if interactive:
        # preload the images
        mm3.information('Preloading images.')
        UI_images = preload_images(specs, fov_id_list)

        mm3.information('Starting channel picking.')
        # go through the fovs again, same as above
        for fov_id in fov_id_list:
            specs = fov_choose_channels_UI(fov_id, crosscorrs, specs, UI_images)
    else:
        outputdir = os.path.join(ana_dir, "fovs")
        if not os.path.isdir(outputdir):
            os.makedirs(outputdir)
        for fov_id in fov_id_list:
            specs = fov_plot_channels(fov_id, crosscorrs, specs,outputdir=outputdir,phase_plane=p['phase_plane'])

    # write specfications to pickle and text
    mm3.information("Writing specifications file.")
    with open(os.path.join(ana_dir,"specs.pkl"), 'w') as specs_file:
        pickle.dump(specs, specs_file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(ana_dir,"specs.txt"), 'w') as specs_file:
        pprint(specs, stream=specs_file)
    # with open(os.path.join(ana_dir,"specs.yaml"), 'w') as specs_file:
    #     yaml.dump(data=specs, stream=specs_file, default_flow_style=False, tags=None)
    mm3.information("Finished.")
