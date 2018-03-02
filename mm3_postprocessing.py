import os,sys,glob
import cPickle as pkl
import argparse
import yaml
import numpy as np
import time
import shutil
import scipy.io as spio

import mm3_helpers

# yaml formats
npfloat_representer = lambda dumper,value: dumper.represent_float(float(value))
nparray_representer = lambda dumper,value: dumper.represent_list(value.tolist())
float_representer = lambda dumper,value: dumper.represent_scalar(u'tag:yaml.org,2002:float', "{:<.6g}".format(value))
yaml.add_representer(float,float_representer)
yaml.add_representer(np.float_,npfloat_representer)
yaml.add_representer(np.ndarray,nparray_representer)

################################################
# functions
################################################
def mad(data,axis=None):
    """
    return the Median Absolute Deviation.
    """
    return np.median(np.absolute(data-np.median(data,axis)),axis)

def get_cutoffs(X, method='mean', plo=1, phi=1):
    if len(X.shape) > 1:
        raise ValueError ("X must be a vector")

    if (method == 'mean'):
        mu = np.mean(X)
        delta = np.std(X)
        xlo = mu - plo*delta
        xhi = mu + phi*delta
    elif (method == 'median'):
        mu = np.median(X)
        delta = mad(X)
        xlo = mu - plo*delta
        xhi = mu + phi*delta
    elif (method == 'median-min'):
        mu = np.median(X)
        delta = mad(X)
        xlo = mu + plo*delta
        xhi = np.nan
    elif (method == 'logmean'):
        idx = (X > 0.)
        mu = np.mean(np.log(X[idx]))
        delta = np.std(np.log(X[idx]))
        xlo = np.exp(mu - plo*delta)
        xhi = np.exp(mu + phi*delta)
    else:
        raise ValueError("Non recognized method argument")

    return xlo,xhi

def print_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def filter_by_generation_index(data, labels):
    data_new = {}
    for key in data:
        cell = data[key]
#        label = np.mean(np.array(cell.labels,dtype=np.float_))
#        label = int(np.around(label))
        label = cell.birth_label
        if (label in labels):
            data_new[key] = data[key]

    return data_new

def filter_by_fovs_peaks(data, fovpeak):
    if fovpeak == None:
        return data

    data_new = {}
    for key in data:
        cell = data[key]
        fov = cell.fov
        peak = cell.peak
        if not (fov in fovpeak):
            continue
        elif (fovpeak[fov] != None) and not (peak in fovpeak[fov]):
            continue
        else:
            data_new[key] = data[key]

    return data_new

def make_lineage(descents, cells):
    if descents == []:
        return descents

    elder = descents[-1]
    ancestor = cells[elder].parent
    if (ancestor == None) or not (ancestor in cells):
        return descents
    else:
        return make_lineage(descents + [ancestor], cells)

def get_lineages_mother_cells(cells):
    lineages = []
    for key in cells:
        cell = cells[key]

        if (np.all([not (keyd in cells) for keyd in cell.daughters])) and (cell.birth_label == 1):
            lineage = make_lineage([key],cells)
            lineage.reverse() # put oldest first
            lineages.append(lineage)

    return lineages

def select_lineages(lineages, min_gen):
    selection = []
    for lin in lineages:
        if (len(lin) >= min_gen):
            selection.append(lin)
    return selection

def filter_cells(cells, par=None):
    # if no parameters passed, then return identical dictionary
    if (type(par) != dict) or (len(par) == 0):
        return cells

    # find list of admissible attributes
    keyref = cells.keys()[0]
    cellref = cells[keyref]
    cellattributes = vars(cellref).keys()
    obs_admissible = []
#    print "Admissible keys:"
#    for x in cellattributes:
#        typ = type(x)
#        if (typ == float) or (typ == int) or (typ == list):
#            print "{:<4s}{:<s}".format("",key)

    # start by selecting all cells
    idx = [True for key in cells.keys()]
    for obs in par:
        #print "Observable \'{}\'".format(obs)
        if not obs in par:
            print "Key \'{}\' not in admissible list of keys:".format(obs)
            for y in obs_admissible:
                print "{:<4s}{:<s}".format("",y)
            continue

        # make scalar array that will undergo selection
        try:
            ind = np.int_(par[obs]['ind'])
            X=[vars(cells[key])[obs][ind] for key in cells.keys()]
        except (TypeError, KeyError, ValueError):
            ind =  None
            X=[vars(cells[key])[obs] for key in cells.keys()]

        # make filtering
        X = np.array(X,dtype=np.float_)
        #xlo, xhi = get_cutoffs(X, method=par[obs]['method'], p=par[obs]['pcut'])
        xlo, xhi = get_cutoffs(X, **par[obs])
        idx = idx & ~(X < xlo) & ~( X > xhi)
        par[obs]['xlo']=np.around(xlo,decimals=4)
        par[obs]['xhi']=np.around(xhi,decimals=4)
        #print "{}: xlo = {:.4g}    xhi = {:.4g}\n".format(obs,xlo,xhi)

    # return new dict
    return {key: cells[key] for key in np.array(cells.keys())[idx]}

def compute_growth_rate_minute(cell, mpf=1):
    """
    Compute the growth in minute for the input cell object.
    """
    cell.times_min = np.array(cell.times) * mpf # create a time index in minutes

    if (len(cell.times_min) < 2):
        cell.growth_rate = np.nan
        cell.growth_rate_intercept = np.nan
        return

    X = np.array(cell.times_min, dtype=np.float_)
    Y = np.array(cell.lengths, dtype=np.float_)
    idx = np.argsort(X)
    X = X[idx]
    Y = Y[idx]
    Z = np.log(Y)
    pf = np.polyfit(X-X[0],Z,deg=1) # fit to a line of the log-length
    gr = pf[0]  # growth rate in per minute
    cell.growth_rate = gr
    cell.growth_rate_intercept = pf[1]

    return

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict
################################################
# main
################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Processing of MM3 pickle file.")
    parser.add_argument('pklfile', metavar='cell pickle file', type=file, help='Pickle file containing the cell dictionary.')
    parser.add_argument('-f', '--paramfile',  type=file, required=True, help='Yaml file containing parameters.')
    parser.add_argument('--trunc',  nargs=2, metavar='t', type=int, help='Make a truncated pkl file for debugging purpose.')
    parser.add_argument('--nofilters',  action='store_true', help='Disable the filters.')
    parser.add_argument('--nocomputations',  action='store_true', help='Disable the computation of extra-quantities (some cells attributes may be overwritten).')
    parser.add_argument('-c', '--cellcycledir',  metavar='picked', type=str, help='Directory containing all Matlab files (lineages) with cell cycle information.')
    parser.add_argument('--complete_cc',  action='store_true', help='If passed, remove cells not mapped to cell cycle through a initiation --> division correspondence.')
    namespace = parser.parse_args(sys.argv[1:])
    paramfile = namespace.paramfile.name
    allparams = yaml.load(namespace.paramfile)
    data = pkl.load(namespace.pklfile)
    dataname = os.path.splitext(os.path.basename(namespace.pklfile.name))[0]
    ddir = os.path.dirname(namespace.pklfile.name)
    print "{:<20s}{:<s}".format('data dir', ddir)

    ncells = len(data)
    print "ncells = {:d}".format(ncells)

    # first initialization of parameters
    params = allparams['filters']

################################################
# Make test set
################################################
    print print_time(), "Writing test file..."
    if namespace.trunc != None:
        print namespace.trunc
        n0 = namespace.trunc[0]
        n1 = namespace.trunc[1]

        datanew={}
        keys = data.keys()
        for key in keys[n0:n1+1]:
            datanew[key] = data[key]

        fileout = os.path.join(ddir, dataname + '_test_n{:d}-{:d}'.format(n0,n1) + '.pkl')
        with open(fileout,'w') as fout:
            pkl.dump(datanew, fout)
        print "{:<20s}{:<s}".format('fileout', fileout)

        sys.exit('Test pkl file written. Exit')

################################################
# cell cycle
################################################
    if namespace.cellcycledir != None:
        print print_time(), 'Loading cell cycle information'
        if not os.path.isdir(namespace.cellcycledir):
            sys.exit('Directory doesn\'t exist: {}'.format(namespace.cellcycledir))

        # initialize new attributes
        #'initiation_mass', 'initiation_mass_n', 'termination_time', 'initiation_time', 'initiation_time_n'
        required_attr = {'initiation_mass_n':'unit_size', 'termination_time':'termination_time', 'initiation_time':'initiation_time'}
        for key,cell in data.items():
            for attrcc,attr in required_attr.items():
                setattr(cell,attr,None)

        for cellcyclefile in glob.glob(os.path.join(namespace.cellcycledir,'*.mat')):
            ccdata = loadmat(cellcyclefile)['cell_list']
            for key in ccdata.keys():
                if not (key in data):
                    print "Skipping cell from cell cycle data: {}".format(key)
                    continue

                cell_cc = ccdata[key]
                cell = data[key]

                for attrcc,attr in required_attr.items():
                    try:
                        setattr(cell,attr,cell_cc[attrcc])
                    except KeyError:
                        pass

        # if argument passed, remove cells without a cell cycle mapped.
        if namespace.complete_cc:
            has_cc = lambda cell: not ((cell.termination_time is None) or (cell.initiation_time is None))
            data = {key: data[key] for key in data.keys() if has_cc(data[key])}

            ncells = len(data)
            print "ncells = {:d}".format(ncells)

################################################
# filters
################################################
    if not namespace.nofilters:
        # Remove cells without mother or without daughters
        print print_time(), "Removing cells without mother or daughters..."
        data_new = {}
        for key in data:
            cell = data[key]
            if (cell.parent != None) and (cell.daughters != None):
                data_new[key] = cell
        data = data_new

        ncells = len(data)
        print "ncells = {:d}".format(ncells)

        # cutoffs filtering
        print print_time(), "Applying cutoffs filtering..."
        if ('cutoffs' in params):
            data = filter_cells(data, params['cutoffs'])

        ncells = len(data)
        print "ncells = {:d}".format(ncells)

        # Generation index
        print print_time(), "Selecting cell indexes..."
        try:
            labels_selection=params['cell_generation_index']
            print "Labels: ", labels_selection
            data = filter_by_generation_index(data, labels=labels_selection)
            ncells = len(data)
            print "ncells = {:d}".format(ncells)

        except KeyError:
            print "Not applied."

        # FOVs and peaks
        print print_time(), "Selecting by FOVs and peaks..."
        try:
            fovpeak = params['fovpeak']
            data = filter_by_fovs_peaks(data, fovpeak)
            ncells = len(data)
            print "ncells = {:d}".format(ncells)

        except KeyError:
            print "Not applied."

        # continuous lineages
        ## list all lineages
        print print_time(), "Selecting by continuous lineages..."
        lineages = get_lineages_mother_cells(data)
        bname = "{}_lineages.pkl".format(dataname)
        fileout = os.path.join(ddir,bname)
        with open(fileout,'w') as fout:
            pkl.dump(lineages, fout)
        print "{:<20s}{:<s}".format("fileout",fileout)

        ## keep only long enough lineages
        try:
            lineages = select_lineages(lineages, **params['lineages']['args'])
            bname = "{}_lineages_selection_min{:d}.pkl".format(dataname, params['lineages']['args']['min_gen'])
            fileout = os.path.join(ddir,bname)
            with open(fileout,'w') as fout:
                pkl.dump(lineages, fout)
            print "{:<20s}{:<s}".format("fileout",fileout)
        except KeyError:
            pass

        try:
            if bool(params['lineages']['keep_continuous_only']):
                selection = []
                for lin in lineages:
                    selection += lin
                selection = np.unique(selection)
                data_new = {key: data[key] for key in selection}
                data = data_new
                ncells = len(data)
                print "ncells = {:d}".format(ncells)
        except KeyError, e :
            print e

        for key in data:
            cell = data[key]

################################################
# compute extra-quantities
################################################
    # second initialization of parameters
    if (not namespace.nocomputations):
        print print_time(), "Compute extra-quantities..."
        # cell cycle quantities
        for key in data.keys():
            cell = data[key]
            try:
                cell.B = cell.initiation_time - cell.birth_time
                cell.C = cell.termination_time - cell.initiation_time
                cell.D = cell.division_time - cell.termination_time
                cell.taucyc = cell.C + cell.D
            except (TypeError, AttributeError):
                cell.B = None
                cell.C = None
                cell.D = None
                cell.taucyc = None

        # other quantities
        if ('computations' in allparams):
            params = allparams['computations']

            # Several quantities in minutes
            try:
                mpf = float(params['min_per_frame'])
            except KeyError:
                print "Could not read the min_per_frame parameter. Default to mpf=1"
                mpf = 1.

            for key in data:
                cell = data[key]
                compute_growth_rate_minute(cell, mpf=mpf)

                try:
                    cell.B_min = cell.B * mpf
                    cell.D_min = cell.D * mpf
                    cell.C_min = cell.C * mpf
                    cell.taucyc_min = cell.taucyc * mpf
                except TypeError:
                    pass

            # compute cell volumes
            try:
                upp = float(params['um_per_pixel'])
            except KeyError:
                print "Could not read the um_per_pixel parameter. Default to upp=1."
                upp = 1.
            for key in data:
                cell = data[key]
                w = np.array(cell.widths, dtype=np.float_)*upp
                L = np.array(cell.lengths, dtype=np.float_)*upp
                cell.volumes = np.pi/4. * w**2*L - np.pi/12. * w**3 # cylinder with hemispherical caps of length L and width w

            # compute fluorescence per volume
            for key in data:
                cell = data[key]
                try:
                    cell.fl_pervolume = np.array(cell.fl_tots, dtype=np.float_) / cell.volumes
                except AttributeError:
                    pass

################################################
# write dictionary
################################################
    print print_time(), "Writing new datafile..."
    ncells = len(data)
    if (ncells == 0 ):
        sys.exit("Filtered data is empty!")

    fileout = os.path.join(ddir, dataname + '_filtered' + '.pkl')
    with open(fileout,'w') as fout:
        pkl.dump(data, fout)
    print "{:<20s}{:<s}".format('fileout', fileout)

# copy parameters
    dest = os.path.join(ddir, os.path.basename(paramfile))
    with open(dest,'w') as fout:
        yaml.dump(allparams,stream=fout,default_flow_style=False, tags=None)
    print "{:<20s}{:<s}".format('fileout', dest)

#try:
#    shutil.copyfile(paramfile,dest)
#    print "{:<20s}{:<s}".format('fileout', dest)
#except shutil.Error:
#    pass
