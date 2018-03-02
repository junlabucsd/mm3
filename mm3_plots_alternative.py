import os,sys
import cPickle as pkl
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker
from scipy.stats.stats import pearsonr
from scipy.stats.stats import spearmanr
from scipy.stats import iqr
from scipy import signal
from scipy.optimize import least_squares

import mm3_helpers as mm3

origin = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(origin)
from fitmodel import FitRes

# global settings
plt.rcParams['axes.linewidth']=0.5

############################################################################
# functions
############################################################################
def fit_xy(x ,y ,p_init, funcfit_f, least_squares_args={'loss':'cauchy'}, funcfit_df=None):
    """
    1) Extract x- (y-) coordinates from attribute key_x (key_y).
    2) Fit the resulting data set according to a model function funcfit_f
    """
    # define FitRes object -- define the residuals
    fitres = FitRes(x,y,funcfit_f=funcfit_f, funcfit_df=funcfit_df)

    # perform the least_square minimization
    try:
        if (funcfit_df == None):
            res = least_squares(x0=p_init, fun=fitres.residual_f, **least_squares_args)
        else:
            res = least_squares(x0=p_init, fun=fitres.residual_f, jac=fitres.residual_df, **least_squares_args)
        par=res.x
    except ValueError as e:
        print e
        sys.exit(1)
#    print res
    return par

def exp_f(par,xi):
    """
    f(x) =  a exp( b x )
    """
    #a = par[0]
    b = par[0]

    #return a * np.exp( b*xi )
    return  np.exp( b*xi )

def exp_df(par,xi):
    """
    f(x) =  a exp( b x )
    """
    #return np.array([np.exp(par[1]*xi), xi*par[0]*np.exp(par[1]*xi)])
    return np.array([xi*np.exp(par[0]*xi)])

def correlation(x,y):
    N=len(x)
    xft=np.fft.fft(x)
    yft=np.fft.fft(y)
    zft=xft*np.conjugate(yft)
    mass=np.real(np.sum(np.fft.fft(np.ones(N))))
    x0=np.float_(np.real(xft[0]/mass))
    y0=np.float_(np.real(yft[0]/mass))
    x2=np.float_(np.real(np.sum(xft*np.conjugate(xft))))/mass**2
    y2=np.float_(np.real(np.sum(yft*np.conjugate(yft))))/mass**2
    z=np.float_(np.real(np.fft.ifft(zft))/mass)

    return (z-x0*y0)

def correlation_pearsonr(x,y):
    N=len(x)
    xft=np.fft.fft(x)
    yft=np.fft.fft(y)
    zft=xft*np.conjugate(yft)
    mass=np.real(np.sum(np.fft.fft(np.ones(N))))
    x0=np.float_(np.real(xft[0]/mass))
    y0=np.float_(np.real(yft[0]/mass))
    x2=np.float_(np.real(np.sum(xft*np.conjugate(xft))))/mass**2
    y2=np.float_(np.real(np.sum(yft*np.conjugate(yft))))/mass**2
    z=np.float_(np.real(np.fft.ifft(zft))/mass)

    vx=x2-x0**2
    vy=y2-y0**2

    return (z-x0*y0)/np.sqrt(vx*vy)

def histogram(X,density=True):
    valmax = np.max(X)
    valmin = np.min(X)
    iqrval = iqr(X)
    nbins_fd = (valmax-valmin)*np.float_(len(X))**(1./3)/(2.*iqrval)
    if (nbins_fd < 1.0e4):
        return np.histogram(X,bins='auto',density=density)
    else:
        return np.histogram(X,bins='sturges',density=density)

def make_binning(x,y,bincount_min):
    """
    Given a graph (x,y), return a graph (x_binned, y_binned) which is a binned version
    of the input graph, along x. Standard deviations per bin for the y direction are also returned.
    """
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]

    valmax = np.max(x)
    valmin = np.min(x)
    iqrval = iqr(x)
    nbins_fd = (valmax-valmin)*np.float_(len(x))**(1./3)/(2.*iqrval)
    if (nbins_fd < 1.0e6):
        histx, bins = np.histogram(x,bins='auto')
    else:
        histx, bins = np.histogram(x,bins='sturges')
#    bins = np.linspace(x[0],x[-1],nbins)
    digitized = np.digitize(x,bins)
    x_binned=[]
    y_binned=[]
    std_binned=[]
    for i in range(1,len(bins)):
        ypts = y[digitized == i]

        if (len(ypts) < bincount_min):
            continue

        x_binned.append(float(0.5*(bins[i-1] + bins[i])))
        y_binned.append(float(np.mean(ypts)))
        std_binned.append(float(np.sqrt(np.var(ypts))))

    res = {}
    res['x'] = x_binned
    res['y'] = y_binned
    res['err'] = std_binned
    return res

def get_derivative(X,Y,p=3,deg=1, fits=False):
    X2 = []
    Y2 = []
    Xfit = []
    Yfit = []
    for i in np.arange(len(X))[p:-p]:
        Xtp = X[i-p:i+p+1]
        Ytp = Y[i-p:i+p+1]
        pf = np.polyfit(Xtp,Ytp,deg)
        pfd = np.poly1d(np.polyder(pf,1))
        xfit = np.linspace(Xtp[0],Xtp[-1],10)
        yfit = np.poly1d(pf)(xfit)
        X2.append(X[i])
        Y2.append(pfd(X[i]))
        Xfit.append(xfit)
        Yfit.append(yfit)
    if fits:
        return np.array(X2),np.array(Y2),Xfit,Yfit
    else:
        return np.array(X2),np.array(Y2)

def lineage_byfov_bypeak(lineages, cells, fov=None, peaks=None):
    if (fov == None):
        return lineages

    selection = []
    for lin in lineages:
        cellref = cells[lin[0]]
        if (cellref.fov == fov):
            if not (peaks is None):
                if (cellref.peak in peaks):
                    selection.append(lin)
            else:
                selection.append(lin)

    return selection

def plot_lineages_byfov(lineages,cells,fileoutspl, color='black', lw=0.5, ax_height=3, ax_width_per_hour=2, fovs=None):
    # all cells
    if (fovs is None):
        all_lineages = np.concatenate(lineages)
        fovs = {fov: None for fov in np.unique([cells[key].fov for key in all_lineages])}

    for fov in fovs:
        # determine correct lineages

        if fovs[fov] == None:
            peaks = np.unique([cells[key].peak for key in np.concatenate(lineage_byfov_bypeak(lineages,cells,fov=fov))])
            fovs[fov] = peaks
        peaks = fovs[fov]
        selection = lineage_byfov_bypeak(lineages,cells, fov=fov, peaks=peaks)
        nlin = len(selection)
        npeaks = len(peaks)
        min_bypeak = {}
        max_bypeak = {}
        for lin in selection:
            cellref = cells[lin[0]]
            peak = cellref.peak
            tstart = np.min([cells[key].times_min[0] for key in lin])
            tend = np.max([cells[key].times_min[-1] for key in lin])
            try:
                tmin = min_bypeak[peak]
                if (tstart < tmin):
                    min_bypeak[peak] = tstart
            except KeyError:
                min_bypeak[peak] = tstart
            try:
                tmax = max_bypeak[peak]
                if (tend > tmax):
                    max_bypeak[peak] = tend
            except KeyError:
                max_bypeak[peak] = tend

        deltamax = np.max([max_bypeak[p] - min_bypeak[p] for p in peaks]) / 60.
        figsize = deltamax*ax_width_per_hour,npeaks*ax_height
        if figsize[0] < ax_width_per_hour:
            figsize = ax_width_per_hour, figsize[1]
        fig = plt.figure(num='none', facecolor='w', figsize=figsize)
        gs = gridspec.GridSpec(npeaks,1)

        cell = cells[lineages[0][0]]
        scale = cell.sb / cell.lengths[0]

        for i,peak in enumerate(peaks):
            ax = fig.add_subplot(gs[i,0])
            for lin in lineage_byfov_bypeak(lineages, cells, fov=fov, peaks=[peak]):
                for key in lin:
                    cell = cells[key]
                    X = np.array(cell.times_min)
                    Y = np.array(cell.lengths)
                    Y *= scale
                    ax.plot(X, Y, '-', color=color, lw=lw)
                for keym,keyd in zip(lin[:-1],lin[1:]):
                    cellm = cells[keym]
                    celld = cells[keyd]
                    x0 = np.array(cellm.times_min)[-1]
                    y0 = np.array(cellm.lengths)[-1]
                    x1 = np.array(celld.times_min)[0]
                    y1 = np.array(celld.lengths)[0]
                    y0 *= scale
                    y1 *= scale
                    ax.plot([x0,x1],[y0, y1], '--', color=color, lw=lw)

            ax.annotate("peak = {:d}".format(cell.peak), xy=(0.,0.98), xycoords='axes fraction', ha = 'left', va='top', fontsize='x-small')
            ax.set_xlabel('time [min]', fontsize='x-small')
            ax.set_ylabel('length $[\mu m]$',fontsize='x-small')
            #ax.tick_params(length=2)
            ax.tick_params(axis='both', labelsize='xx-small', pad=2)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        fig.suptitle("FOV {:d}".format(fov), fontsize='medium')
        rect = [0.,0.,1.,0.98]
        gs.tight_layout(fig,rect=rect)
        fileout = "{}_xy{:03d}.pdf".format(fileoutspl,fov)
        print "{:<20s}{:<s}".format('fileout',fileout)
        fig.savefig(fileout,bbox_inches='tight',pad_inches=0)
        plt.close('all')
    return

def plot_lineage_with_growth_rate(lineage, cells, fileoutspl, stitch=False, color='black', color1='darkblue', color2='darkgreen', lw=0.5, ms=1, logscale=True, pfit=2, showfits=False, acf_dtau_max=2, T_filter=None):
    """
    plot lineage.
    plot growth rate.
    plot growth rate distribution.
    plot growth rate autocorrelation.
    """

    # some information
    cell_tp = cells[lineage[0]]
    fov = cell_tp.fov
    peak = cell_tp.peak
    ncell = len(lineage)
    tstart = np.min([cells[key].birth_time for key in lineage])
    tend = np.max([cells[key].division_time for key in lineage])

    # figure
    fig = plt.figure(num='none', facecolor='w')
    gs = gridspec.GridSpec(2,2, width_ratios=[3,1.5])
    ax_left_top = fig.add_subplot(gs[0,0])
    ax_left_bot = fig.add_subplot(gs[1,0])
    ax_right_top = fig.add_subplot(gs[0,1])
    ax_right_bot = fig.add_subplot(gs[1,1])

    # compute average growth rate
    Xpop = []
    for key in cells:
        cell = cells[key]
        try:
            x = np.float_(cell.growth_rate)
            if np.isfinite(x):
                Xpop.append(x)
        except ValueError:
            continue
    Xpop = np.array(Xpop)
    gr_mean_pop = np.mean(Xpop)
    gr_std_pop = np.std(Xpop)
    gr_cv_pop = gr_std_pop / gr_mean_pop

    # fill-in first axes
    ## plot growth curves
    ax = ax_left_top
    XX = []
    YY = []
    YYs = []
    ZZ = [] # hold line for constant gr per generation
    ZZs = []
    GR = []

    cell = cells[lineage[0]]
    Sref = cell.sb
    scale = cell.sb / cell.lengths[0]

    for key in lineage:
        cell = cells[key]
        X = np.array(cell.times_min)
        Y = np.array(cell.lengths) * scale
        gr = cell.growth_rate
        y0 = np.exp(cell.growth_rate_intercept) * scale
        Z = y0*np.exp(gr*(X-X[0]))
        GR.append(gr)

        fac = Sref/Y[0]
        Ys = Y*fac
        Sref = Ys[-1]
        Zs = y0*fac*np.exp(gr*(X-X[0]))

        XX.append(X)
        YY.append(Y)
        YYs.append(Ys)
        ZZ.append(Z)
        ZZs.append(Zs)

        if not stitch:
            ax.plot(X, Y, '.', color=color, ms=ms)
            ax.plot(X, Z, '-', color=color2, lw=lw)

    for i in range(ncell-1):
        x0 = XX[i][-1]
        x1 = XX[i+1][0]
        if stitch:
            ax.axvline(x=0.5*(x0+x1), linestyle='--', color=color, lw=lw)
        else:
            y0 = YY[i][-1]
            y1 = YY[i+1][0]
            ax.plot([x0,x1],[y0, y1], '--', color=color, lw=lw)

    Xs = []
    Ys = []
    Zs = []
    for x, ys, zs in zip(XX,YYs,ZZs):
        Xs = np.append(Xs,x)
        Ys = np.append(Ys,ys)
        Zs = np.append(Zs,zs)

    if stitch:
        ax.plot(Xs, Ys, '.', color=color, ms=ms)
        ax.plot(Xs, Zs, '-', color=color2, lw=lw)


    if logscale:
        ax.set_yscale('log', basey=2)

    ax.set_xlabel('time [min]', fontsize='x-small')
    ax.set_ylabel('length $[\mu m]$',fontsize='x-small')
    #ax.tick_params(length=2)
    ax.tick_params(axis='both', labelsize='xx-small', pad=2)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ## plot growth rate
    ax = ax_left_bot
    # plot generation growth rates
    for x, gr in zip(XX,GR):
        ax.plot([x[0],x[-1]],[gr*60,gr*60], '-', color=color2, lw=lw)

    # compute instantaneous growth rates
    Zs = np.log(Ys)
    pf = np.polyfit(Xs-Xs[0],Zs,deg=1)
    Xfit = np.linspace(np.min(Xs-Xs[0]),np.max(Xs-Xs[0]),100)
    Zfit = np.poly1d(pf)(Xfit)
    Xfit += Xs[0]
    Yfit = np.exp(Zfit)
    gr_glb = pf[0]
    gr_intercept = pf[1]

    X1, Z1, X1fits, Z1fits = get_derivative(Xs,Zs,p=pfit,deg=1, fits=True)

    ax.plot(X1, Z1*60, '--', color=color1, lw=lw)

    # filter
    #wn = 0.05
    tmax = np.max(X1)
    tmin = np.min(X1)
    tnyq = (tmax-tmin)/len(X1) # sampling time
    if (T_filter == None):
        T_filter = np.log(2.)/gr_glb
    fn = 0.5/T_filter
    fnyq = 0.5/ tnyq # the fastest frequency is when a cosine performs one half-cycle per sampling time. There is no sense in authorizing fluctuations faster than the sampling interval.
    wn = fn/fnyq
    #print "wn = {:.4f}".format(wn)
    b, a = signal.butter(3, wn) # second argument is in unit of the nyquist frequency = 1/2 1/N (N = len(sample)). The input given in minutes is therefore the half period of the fastest sine wave. A good rule of thumb seems to be choosing a half-period which is ~1 generation time. first argument is the order of the low-pass filter.
    Z1_fil = signal.filtfilt(b,a, Z1,method='gust')
    ax.plot(X1, Z1_fil*60, '-', color=color1, lw=2.*lw)
    """
    Ys_fil = signal.filtfilt(b,a, Ys ,method='gust')
    Zs_fil = np.log(Ys_fil)
    X1_fil, Z1_fil = get_derivative(Xs,Zs_fil,p=pfit,deg=1)
    ax.plot(X1_fil, Z1_fil*60, '-', color=color, lw=2.*lw)
    ax_left_top.plot(Xs,Ys_fil,'b-', lw=lw)
    """
    # filter

    if stitch:
        ax_left_top.plot(Xfit,Yfit,'r--', lw=lw, label='$\lambda = {:.2f}$ $[h^{{-1}}]$'.format(gr_glb*60))
        ax_left_top.plot(Xs, np.exp(gr_mean_pop*(Xs-Xs[0]) + gr_intercept), '-.', color=color, lw=lw, label='$\\lambda_{{pop}} = {:.2f}$ $[h^{{-1}}]$'.format(gr_mean_pop*60.))
        ax.axhline(y=gr_mean_pop*60, color='k', linestyle='-.', lw=lw)
        ax_left_top.legend(loc='best', fontsize='x-small')

        if showfits:
            #for x1fit, z1fit in zip(X1fits,Z1fits)[::2*(pfit + 1)]:
            Zs_fil = [Zs[pfit]]
            Xs_fil = [Xs[pfit]]
            z = Zs_fil[-1]
            for x0,x1,dz in zip(Xs[pfit:-pfit-1], Xs[pfit+1:-pfit], Z1_fil):
                dx = x1-x0
                z += dz*dx
                Zs_fil.append(z)
                Xs_fil.append(x1)

            ax_left_top.plot(Xs_fil, np.exp(Zs_fil), '-b', lw=lw)
            for x1fit, z1fit in zip(X1fits,Z1fits):
                y1fit = np.exp(z1fit)
                ax_left_top.plot(x1fit,y1fit, '-', color=color1, lw=lw)

    for i in range(ncell-1):
        x0 = XX[i][-1]
        x1 = XX[i+1][0]
        ax.axvline(x=0.5*(x0+x1), linestyle='--', color=color, lw=lw)

    time = X1[:]
    growth_rates_raw = Z1[:]
    growth_rates_fil = Z1_fil[:]
    growth_rates_gen = np.array(GR)[:]
    gr_mean_gen = np.mean(growth_rates_gen)
    gr_std_gen = np.std(growth_rates_gen)
    gr_cv_gen = gr_std_gen / gr_mean_gen
    gr_mean_fil = np.mean(growth_rates_fil)
    gr_std_fil = np.std(growth_rates_fil)
    gr_cv_fil = gr_std_fil / gr_mean_fil
    tau = np.log(2.)/gr_glb
    ax.axhline(y=gr_glb*60, color='red', linestyle='--', lw=lw, label="$<\lambda> = {:.2f}$ $[h^{{-1}}]$\n$\\tau = {:.0f}$ [min]".format(gr_glb*60, tau))
    ax.legend(loc='best', fontsize='x-small')

    ax.set_xlabel('time [min]', fontsize='x-small')
    ax.set_ylabel('growth rate $[h^{-1}]$',fontsize='x-small')
    #ax.tick_params(length=2)
    ax.tick_params(axis='both', labelsize='xx-small', pad=2)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # fill-in other axes

    ## histogram
    ax = ax_right_top

    hist,edges = histogram(growth_rates_gen*60., density=True)
    #ax.bar(left=edges[:-1], height=hist, width=edges[1:]-edges[:-1], linestyle='-', color='none', edgecolor=color, lw=lw)
    label = "$\mu$ = {:.2f}, CV = {:<.0f}%".format(gr_mean_gen*60,gr_cv_gen*100.)
    ax.bar(left=edges[:-1], height=hist, width=edges[1:]-edges[:-1], linestyle='-', color=color2, edgecolor='none', lw=0., alpha=0.5, label=label)

    hist,edges = histogram(growth_rates_fil*60., density=True)
    #ax.plot(edges[:-1], hist, '-', color=color, lw=lw)
    #ax.bar(left=edges[:-1], height=hist, width=edges[1:]-edges[:-1], linestyle='-', color='none', edgecolor=color, lw=lw)
    label = "$\mu$ = {:.2f}, CV = {:<.0f}%".format(gr_mean_fil*60,gr_cv_fil*100.)
    ax.bar(left=edges[:-1], height=hist, width=edges[1:]-edges[:-1], linestyle='-', color=color1, edgecolor='none', lw=0., alpha=0.5, label=label)

    #text = "Mean = {:.2f} ({:.2f}) $[h^{{-1}}]$\nCV = {:<.0f}% ({:<.0f}%)".format(gr_mean*60,gr_mean_fil*60, gr_cv*100., gr_cv_fil*100)
    #ax.set_title(text, fontsize='x-small')
    ax.legend(loc='best', fontsize='xx-small')
    ax.set_yticks([])
    ax.set_xlabel('growth rate $[h^{-1}]$', fontsize='x-small')
    ax.set_ylabel('pdf', fontsize='x-small')
    #ax.tick_params(length=2)
    ax.tick_params(axis='both', labelsize='xx-small', pad=2)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


    ax = ax_right_bot
    X_gen = np.arange(len(growth_rates_gen))+1
    Y = growth_rates_gen[:]*60.
    ax.plot(X_gen,Y,'-o', color=color2, lw=lw, ms=2*ms)
    ax.set_xlabel('generation', fontsize='x-small')
    ax.set_ylabel('growth rate $[h^{-1}]$', fontsize='x-small')
    #ax.tick_params(length=2)
    ax.tick_params(axis='both', labelsize='xx-small', pad=2)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

#    ## autocorrelation function
#    ax = ax_right_bot
#    #r_gr=correlation_pearsonr(growth_rates,growth_rates)
#    r_gr_fil=correlation_pearsonr(growth_rates_fil,growth_rates_fil)
#
#    idx = ((time - time[0]) <= acf_dtau_max*tau)
#    X=time[idx] - time[0]
#    Y=r_gr[idx]
#    Y_fil=r_gr_fil[idx]
#    M = len(X)
#    npts=1000
#    dn = max(1,np.int_(np.float_(M)/npts))
#    X=X[::dn]
#    Y=Y[::dn]
#    ax.plot(X,Y,'-.', color=color, lw=lw)
#    ax.plot(X,Y_fil,'-', color=color, lw=lw)
#
#    k0 = np.argmin(np.abs(X-tau))
#    x0 = X[k0]
#    y0 = Y[k0]
#    ax.axvline(x=x0, lw=lw, linestyle='--', color=color)
#    text = "$r_{{PE}}(\\tau) = {:.1f}$".format(y0)
#    ax.annotate(text, xy=(x0,y0), xycoords='data', xytext=(1.,0.98), textcoords='axes fraction', fontsize='x-small', ha='right', va='top')
#    ax.set_xlabel('time [min]', fontsize='x-small')
#    ax.set_ylabel('ACF', fontsize='x-small')
#    #ax.tick_params(length=2)
#    ax.tick_params(axis='both', labelsize='xx-small', pad=2)
#    ax.spines['right'].set_visible(False)
#    ax.spines['top'].set_visible(False)

    fig.suptitle("FOV {:d}{:4s}peak = {:d}".format(fov,',',peak), fontsize='medium')
    rect = [0.,0.,1.,0.98]
    gs.tight_layout(fig,rect=rect)
    fileout = "{}_xy{:03d}p{:04d}t{:d}-{:d}.pdf".format(fileoutspl,fov,peak,tstart,tend)
    print "{:<20s}{:<s}".format('fileout',fileout)
    fig.savefig(fileout,bbox_inches='tight',pad_inches=0)
    plt.close('all')
    return

def plot_lineage_variables(lineage, cells, fileoutspl, attrdict=None, stitch=False, color='black', color1='darkgreen', lw=0.5, ms=1, logscale=True):
    """
    plot lineage.
    plot choosen attributes:
        o cv across lineage
        o evolution during lineage
    """
    # preliminary check
    if (attrdict is None) or (type(attrdict) != dict) or (len(attrdict) == 0):
        print "List of observables empty!"
        return

    # some information
    cell_tp = cells[lineage[0]]
    Sref = cell_tp.sb
    scale = cell_tp.sb / cell_tp.lengths[0]
    fov = cell_tp.fov
    peak = cell_tp.peak
    ncell = len(lineage)
    tstart = np.min([cells[key].birth_time for key in lineage])
    tend = np.max([cells[key].division_time for key in lineage])
    attributes = attrdict.keys()
    nattr = len(attributes)
    ncol = nattr

    # compute average growth rate
    Xpop = []
    for key in cells:
        cell = cells[key]
        try:
            x = np.float_(cell.growth_rate)
            if np.isfinite(x):
                Xpop.append(x)
        except ValueError:
            continue
    Xpop = np.array(Xpop)
    gr_mean_pop = np.mean(Xpop)
    gr_std_pop = np.std(Xpop)
    gr_cv_pop = gr_std_pop / gr_mean_pop

    # figure
    fig = plt.figure(num='none', facecolor='w')
    gs = gridspec.GridSpec(3,ncol, height_ratios=[1.5,1.5,2])
    ax_trace = fig.add_subplot(gs[2,:])
    axdict = {key: [fig.add_subplot(gs[0,i]), fig.add_subplot(gs[1,i])] for i,key in enumerate(attributes)}

    # fill-in first axes
    ## plot growth curves
    ax = ax_trace
    XX = []
    YY = []
    YYs = []

    for key in lineage:
        cell = cells[key]
        X = np.array(cell.times_min)
        Y = np.array(cell.lengths) * scale

        fac = Sref/Y[0]
        Ys = Y*fac
        Sref = Ys[-1]

        XX.append(X)
        YY.append(Y)
        YYs.append(Ys)

        if not stitch:
            ax.plot(X, Y, '-', color=color, lw=lw, ms=ms)

    for i in range(ncell-1):
        x0 = XX[i][-1]
        x1 = XX[i+1][0]
        if stitch:
            ax.axvline(x=0.5*(x0+x1), linestyle='--', color=color, lw=lw)
        else:
            y0 = YY[i][-1]
            y1 = YY[i+1][0]
            ax.plot([x0,x1],[y0, y1], '--', color=color, lw=lw)

    Xs = []
    Ys = []
    for x, ys in zip(XX,YYs):
        Xs = np.append(Xs,x)
        Ys = np.append(Ys,ys)

    if stitch:
        ax.plot(Xs, Ys, '-', color=color, lw=lw, ms=ms)
        ax.plot(Xs, Ys[0]*np.exp(gr_mean_pop*(Xs-Xs[0])), '-.', color=color, lw=lw, label='$\\lambda_{{pop}} = {:.2f}$ $[h^{{-1}}]$'.format(gr_mean_pop*60.))
        ax.legend(loc='best', fontsize='xx-small')

    if logscale:
        ax.set_yscale('log', basey=2)

    ax.set_xlabel('time [min]', fontsize='x-small')
    ax.set_ylabel('length $[\mu m]$',fontsize='x-small')
    #ax.tick_params(length=2)
    ax.tick_params(axis='both', labelsize='xx-small', pad=2)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # plot per attrbute
    for i, attr in enumerate(attributes):
        ax_top, ax_bot = axdict[attr]

        # build data
        X = []
        for key in lineage:
            cell = cells[key]
            try:
                x = np.float_(getattr(cell,attr))
                if np.isfinite(x):
                    X.append(x)
            except ValueError:
                continue
        X = np.array(X)

        Xpop = []
        for key in cells:
            cell = cells[key]
            try:
                x = np.float_(getattr(cell,attr))
                if np.isfinite(x):
                    Xpop.append(x)
            except ValueError:
                continue
        Xpop = np.array(Xpop)

        # rescale
        try:
            scale = attrdict[attr]['scale']
            X = X *scale
            Xpop = Xpop *scale
        except KeyError:
            pass

        #print len(X)
        mean = np.mean(X)
        std = np.std(X)
        cv = std/mean
        mean_pop = np.mean(Xpop)
        std_pop = np.std(Xpop)

        # axis label
        try:
            axis_label = attrdict[attr]['label']
        except KeyError:
            axis_label = attr

        # histogram
        ax = ax_top
        hist,edges = histogram(X)
        left = edges[:-1]
        right = edges[1:]
        idx = (hist != 0.)
        label = "$\mu$ = {:.2f}, CV = {:<.0f}%".format(mean,cv*100)
        ax.bar(left=left, height=hist, width=right-left, linestyle='-', color=color1, edgecolor='none', lw=0., label=label, alpha=0.5)
        ax.axvline(x=mean_pop-std_pop, linestyle='--', color='k', lw=lw)
        ax.axvline(x=mean_pop+std_pop, linestyle='--', color='k', lw=lw)
        ax.axvline(x=mean_pop, linestyle='-', color='k', lw=lw)
        ax.legend(loc='best', fontsize='xx-small')
        ax.set_yticks([])
        ax.set_xlabel(axis_label, fontsize='x-small')
        ax.set_ylabel('pdf', fontsize='x-small')
        #ax.tick_params(length=2)
        ax.tick_params(axis='both', labelsize='xx-small', pad=2)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # generation trace
        ax = ax_bot
        X_gen = np.arange(len(X))+1
        ax.plot(X_gen,X,'-o', color=color1, lw=lw, ms=2*ms)
        ax.set_xlabel('generation', fontsize='x-small')
        ax.set_ylabel(axis_label, fontsize='x-small')
        ax.axhline(y=mean_pop-std_pop, linestyle='--', color='k', lw=lw)
        ax.axhline(y=mean_pop+std_pop, linestyle='--', color='k', lw=lw)
        ax.axhline(y=mean_pop, linestyle='-', color='k', lw=lw, label='popul.')
        #ax.tick_params(length=2)
        ax.xaxis.set_major_locator(matplotlib.ticker.IndexLocator(base=2, offset=0))
        #ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x,pos: int(x+0.5)))
        ax.tick_params(axis='both', labelsize='xx-small', pad=2)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.legend(loc='best', fontsize='xx-small')

    fig.suptitle("FOV {:d}{:4s}peak = {:d}".format(fov,',',peak), fontsize='medium')
    rect = [0.,0.,1.,0.98]
    gs.tight_layout(fig,rect=rect)
    fileout = "{}_xy{:03d}p{:04d}t{:d}-{:d}.pdf".format(fileoutspl,fov,peak,tstart,tend)
    print "{:<20s}{:<s}".format('fileout',fileout)
    fig.savefig(fileout,bbox_inches='tight',pad_inches=0)
    plt.close('all')
    return

def plot_lineage_correlations(lineage, cells, fileoutspl, attrdict=None, stitch=False, color='black', color1='darkgreen', lw=0.5, ms=2, logscale=True):
    """
    plot lineage.
    plot attributes correlations
    """
    # preliminary check
    if (attrdict is None) or (type(attrdict) != dict) or (len(attrdict) == 0):
        print "List of observables empty!"
        return

    # some information
    cell_tp = cells[lineage[0]]
    Sref = cell_tp.sb
    scale = cell_tp.sb / cell_tp.lengths[0]
    fov = cell_tp.fov
    peak = cell_tp.peak
    ncell = len(lineage)
    tstart = np.min([cells[key].birth_time for key in lineage])
    tend = np.max([cells[key].division_time for key in lineage])
    attributes = np.array(attrdict.keys())
    nattr = len(attributes)
    ncol = nattr

    # compute average growth rate
    Xpop = []
    for key in cells:
        cell = cells[key]
        try:
            x = np.float_(cell.growth_rate)
            if np.isfinite(x):
                Xpop.append(x)
        except ValueError:
            continue
    Xpop = np.array(Xpop)
    gr_mean_pop = np.mean(Xpop)
    gr_std_pop = np.std(Xpop)
    gr_cv_pop = gr_std_pop / gr_mean_pop

    # figure
    r = 4./3.
    axdim=1.5
    figsize=nattr*r*axdim, (nattr + 1 + 1)*axdim
    #fig = plt.figure(num='none', facecolor='w')
    #gs = gridspec.GridSpec(nattr+1+1,nattr, height_ratios=[1.5]*(nattr+1)+[2])
    fig = plt.figure(num='none', facecolor='w', figsize=figsize)
    gs = gridspec.GridSpec(nattr+1+1,nattr)
    ax_trace = fig.add_subplot(gs[-1,:])

    # fill-in first axes
    ## plot growth curves
    ax = ax_trace
    XX = []
    YY = []
    YYs = []

    for key in lineage:
        cell = cells[key]
        X = np.array(cell.times_min)
        Y = np.array(cell.lengths) * scale

        fac = Sref/Y[0]
        Ys = Y*fac
        Sref = Ys[-1]

        XX.append(X)
        YY.append(Y)
        YYs.append(Ys)

        if not stitch:
            ax.plot(X, Y, '-', color=color, lw=lw, ms=ms)

    for i in range(ncell-1):
        x0 = XX[i][-1]
        x1 = XX[i+1][0]
        if stitch:
            ax.axvline(x=0.5*(x0+x1), linestyle='--', color=color, lw=lw)
        else:
            y0 = YY[i][-1]
            y1 = YY[i+1][0]
            ax.plot([x0,x1],[y0, y1], '--', color=color, lw=lw)

    Xs = []
    Ys = []
    for x, ys in zip(XX,YYs):
        Xs = np.append(Xs,x)
        Ys = np.append(Ys,ys)

    if stitch:
        ax.plot(Xs, Ys, '-', color=color, lw=lw, ms=ms)
        ax.plot(Xs, Ys[0]*np.exp(gr_mean_pop*(Xs-Xs[0])), '-.', color=color, lw=lw, label='$\\lambda_{{pop}} = {:.2f}$ $[h^{{-1}}]$'.format(gr_mean_pop*60.))
        ax.legend(loc='best', fontsize='xx-small')

    if logscale:
        ax.set_yscale('log', basey=2)

    ax.set_xlabel('time [min]', fontsize='x-small')
    ax.set_ylabel('length $[\mu m]$',fontsize='x-small')
    #ax.tick_params(length=2)
    ax.tick_params(axis='both', labelsize='xx-small', pad=2)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # plot per attrbute
    for i, attr_y in enumerate(attributes): # row is y-axis
        # get label data
        try:
            axis_labely = attrdict[attr_y]['label']
        except KeyError:
            axis_labely = attr_y

        # build data for Y
        Y = []
        for key in lineage:
            cell = cells[key]
            try:
                y = np.float_(getattr(cell,attr_y))
                if np.isfinite(y):
                    Y.append(y)
            except ValueError:
                continue
        Y = np.array(Y)

        # build data for average population
        Ypop = []
        for key in cells:
            cell = cells[key]
            try:
                y = np.float_(getattr(cell,attr_y))
                if np.isfinite(y):
                    Ypop.append(y)
            except ValueError:
                continue
        Ypop = np.array(Ypop)

        # rescale
        try:
            scale = attrdict[attr_y]['scale']
            Y = Y *scale
            Ypop = Ypop *scale
        except KeyError:
            pass

        ymean = np.mean(Y)
        ystd = np.std(Y)
        ycv = ystd/ymean
        ymean_pop = np.mean(Ypop)
        ystd_pop = np.std(Ypop)
        Yticks = [ymean - ystd, ymean, ymean + ystd]

        for j, attr_x in enumerate(attributes): # col is x-axis
            # get label data
            try:
                axis_labelx = attrdict[attr_x]['label']
            except KeyError:
                axis_labelx = attr_x

            # plot histogram in diagonals
            if (j==i):
                ax = fig.add_subplot(gs[i,i])
                hist,edges = histogram(Y)
                left = edges[:-1]
                right = edges[1:]
                label = "$\mu$ = {:.2f}, CV = {:<.0f}%".format(ymean,ycv*100)
                ax.bar(left=left, height=hist, width=right-left, linestyle='-', color=color1, edgecolor='none', lw=0., label=label, alpha=0.5)

                ax.set_xticks(Yticks)
                ax.set_yticks([])
                ax.tick_params(axis='x', which='both', bottom='on', top='off', labelsize='xx-small')
                ax.legend(loc='best', fontsize='xx-small')
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                #ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=nbins_max))
                ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:.2g}'))

            else:
                # build data for X
                X = []
                for key in lineage:
                    cell = cells[key]
                    try:
                        x = np.float_(getattr(cell,attr_x))
                        if np.isfinite(x):
                            X.append(x)
                    except ValueError:
                        continue
                X = np.array(X)

                # build data for average population
                Xpop = []
                for key in cells:
                    cell = cells[key]
                    try:
                        x = np.float_(getattr(cell,attr_x))
                        if np.isfinite(x):
                            Xpop.append(x)
                    except ValueError:
                        continue
                Xpop = np.array(Xpop)

                # rescale
                try:
                    scale = attrdict[attr_x]['scale']
                    X = X * scale
                    Xpop = Xpop * scale
                except KeyError:
                    pass

                xmean = np.mean(X)
                xstd = np.std(X)
                xcv = xstd/xmean
                xmean_pop = np.mean(Xpop)
                xstd_pop = np.std(Xpop)
                Xticks = [xmean - xstd, xmean, xmean + xstd]

                # axes label
                ax = fig.add_subplot(gs[i,j])
                ax.plot(X,Y,'o', color=color1, lw=lw, ms=ms)
                ax.axvline(x=xmean_pop-xstd_pop, linestyle='-.', color='k', lw=lw, label='popul.')
                ax.axvline(x=xmean_pop+xstd_pop, linestyle='-.', color='k', lw=lw)
                ax.axvline(x=xmean_pop, linestyle='--', color='k', lw=lw)
                ax.axhline(y=ymean_pop-ystd_pop, linestyle='-.', color='k', lw=lw)
                ax.axhline(y=ymean_pop+ystd_pop, linestyle='-.', color='k', lw=lw)
                ax.axhline(y=ymean_pop, linestyle='--', color='k', lw=lw)

                ax.set_xticks(Xticks)
                ax.set_yticks(Yticks)
                ax.tick_params(axis='x', which='both', bottom='on', top='off', labelsize='xx-small')
                ax.tick_params(axis='y', which='both', left='on', right='off', labelsize='xx-small')
                ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:.2g}'))
                ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:.2g}'))
                #ax.legend(loc='best', fontsize='xx-small')
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)

            if (j == 0):
                ax.annotate(axis_labely, xy=(-0.30,0.5), xycoords='axes fraction', va='center', ha='right', fontsize='x-small', rotation='vertical')
            if (i == 0):
                ax.annotate(axis_labelx, xy=(0.5,1.10), xycoords='axes fraction', ha='center', va='bottom', fontsize='x-small')


    # plot autocorrelations
    for j, attr in enumerate(attributes):
        # get label data
        try:
            axis_labely = attrdict[attr]['label_d']
        except KeyError:
            axis_labely = attr_y
        try:
            axis_labelx = attrdict[attr]['label_m']
        except KeyError:
            axis_labelm = attr_m

        # build data
        X = []
        Y = []
        for keyd in lineage:
            try:
                celld = cells[keyd]
                keym=celld.parent
                cellm = cells[keym]
                x = np.float_(getattr(cellm,attr))
                y = np.float_(getattr(celld,attr))
                if np.isfinite(x) and np.isfinite(y):
                    X.append(x)
                    Y.append(y)
            except ValueError:
                # error in isfinite tests
                continue
            except KeyError:
                # error in cellm=cells[keym] statement
                continue

        X = np.array(X)
        Y = np.array(Y)

        # rescale
        try:
            scale = attrdict[attr]['scale']
            X = X * scale
            Y = Y * scale
        except KeyError:
            pass

        xmean = np.mean(X)
        xstd = np.std(X)
        xcv = xstd/xmean
        Xticks = [xmean - xstd, xmean, xmean + xstd]

        # add plot
        ax = fig.add_subplot(gs[nattr,j])
        ax.plot(X,Y,'o', color=color1, ms=ms, lw=lw)

        ax.set_xticks(Xticks)
        ax.set_yticks(Xticks)
        ax.tick_params(axis='x', which='both', bottom='on', top='off', labelsize='xx-small')
        ax.tick_params(axis='y', which='both', left='on', right='off', labelsize='xx-small')
        ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:.2g}'))
        ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:.2g}'))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylabel(attrdict[attr]['label_d'], fontsize='x-small')
        ax.set_xlabel(attrdict[attr]['label_m'], fontsize='x-small')

    fig.suptitle("FOV {:d}{:4s}peak = {:d}".format(fov,',',peak), fontsize='medium')
    rect = [0.,0.,1.,0.92]
    gs.tight_layout(fig,rect=rect, w_pad=0.1, h_pad=0.1)
    fileout = "{}_xy{:03d}p{:04d}t{:d}-{:d}.pdf".format(fileoutspl,fov,peak,tstart,tend)
    print "{:<20s}{:<s}".format('fileout',fileout)
    fig.savefig(fileout,bbox_inches='tight',pad_inches=0)
    plt.close('all')
    return

def plot_lineages_acf_old(lineages, cells, fileoutspl, fovs=None, attrdict=None, color='black', lw=0.5, ms=2, minnorm=100):
    """
    plot autocorrelation curves (overlaid) for different cell attributes.
    """
    # preliminary check
    if (attrdict is None) or (type(attrdict) != dict) or (len(attrdict) == 0):
        print "List of observables empty!"
        return

    # filter lineages
    if (fovs is None):
        all_lineages = np.concatenate(lineages)
        fovs = {fov: None for fov in np.unique([cells[key].fov for key in all_lineages])}

    selection=[]
    for fov in fovs:
        # determine correct lineages
        if fovs[fov] == None:
            peaks = np.unique([cells[key].peak for key in np.concatenate(lineage_byfov_bypeak(lineages,cells,fov=fov))])
            fovs[fov] = peaks
        peaks = fovs[fov]
        subselection = lineage_byfov_bypeak(lineages,cells, fov=fov, peaks=peaks)
        nlin = len(selection)
        selection.append(subselection)

    selection = np.concatenate(selection)

    # some information
    attributes = np.array(attrdict.keys())
    nattr = len(attributes)
    ncol = nattr
    tau_mean = np.mean([cell.tau for cell in cells.values()])

    # figure
    r = 4./3.
    axdim=3
    figsize=nattr*r*axdim, axdim
    fig = plt.figure(num='none', facecolor='w', figsize=figsize)
    gs = gridspec.GridSpec(1,nattr)

    # plot per attrbute
    for i, attr_y in enumerate(attributes): # row is y-axis
        # get label data
        try:
            axis_labely = attrdict[attr_y]['label']
        except KeyError:
            axis_labely = attr_y

        # add plot
        ax = fig.add_subplot(gs[0,i])

        Ytot=[]
        Xtot=[]
        for lineage in selection:
            # build data
            XX = []
            YY = []
            for key in lineage:
                cell = cells[key]
                try:
                    x = np.array(getattr(cell,'times_min'), dtype=np.float_)
                    y = np.array(getattr(cell,attr_y), dtype=np.float_)
                    idx = np.isfinite(x)
                    if np.isfinite(x).all() and np.isfinite(y).all():
                        XX.append(x)
                        YY.append(y)
                except ValueError:
                    continue
            X = np.concatenate(XX)
            Y = np.concatenate(YY)

            # rescale
            try:
                scale = attrdict[attr_y]['scale']
                Y = Y *scale
            except KeyError:
                pass

            # shift time origin
            x0 = X[0]
            X-=x0 # start at lag = 0
            Xtot.append(X)
            Ytot.append(Y)

        # computing the acf
        kmax = np.argmax([len(x) for x in Xtot])
        X = np.array(Xtot[kmax])
        T=len(X)
        NORM = np.zeros(T)   # normalizations
        S = np.zeros(T)     # 2-points: 0,t
        MU2 = np.zeros(T)   # 2-points: t,t
        MU1 = np.zeros(T)   # 1-point: t
        for y in Ytot:
            for k in range(len(y)):
                S[k] += y[k]*y[0]
                MU2[k] += y[k]*y[k]
                MU1[k] += y[k]
                NORM[k] += 1.
        S /= NORM
        MU2 /= NORM
        MU1 /= NORM
        VAR = MU2 - MU1**2
        Z = (S - MU1 * MU1[0])
        #Z /= Z[0]
        Z /= np.sqrt(VAR[0]*VAR)
        idx = NORM > minnorm
        X=X[idx]
        Z=Z[idx]
        #print VAR[idx]
        ax.plot(X,Z,'-', color=color, ms=ms, lw=3*lw)
        #ax.plot(X,np.sqrt(VAR[idx]),'-', color=color, ms=ms, lw=3*lw)
        ax.axvline(x=tau_mean, color='k', linestyle='--', lw=lw, label='$\\tau={:.0f}$'.format(tau_mean))
        #ax.axhline(y=0, color='k', linestyle='-', lw=lw)

        # fitting
        #idx = (np.isfinite(Z)) & (X>0)
        idx = (np.isfinite(Z)) & (Z > 0.)
        Xfit = X[idx]
        Zfit = Z[idx]

        #"""
        ## exponential
        #ax.plot(Xfit,Zfit,'-g', lw=3*lw)
        a = -1. / (np.sum(Zfit) * np.diff(Xfit)[0])
        par0 = [a]
        try:
            par = fit_xy(Xfit,Zfit, par0, funcfit_f=exp_f)
        except:
            par = par0
        xx=np.linspace(Xfit[0],Xfit[-1],1000)
        zz = np.array([exp_f(par,x) for x in xx])
        a = par[0]
        #"""
        """
        ## line
        Yfit = np.log(Zfit)
        #ax.plot(Xfit,Zfit,'-g', lw=3*lw)
        a,b = np.polyfit(Xfit, Yfit, deg=1)
        xx=np.linspace(Xfit[0],Xfit[-1],1000)
        yy=a*xx+b
        zz=np.exp(yy)
        #"""

        ax.plot(xx,zz,'--r', lw=3*lw, label='$\\tau={:.0f}$'.format(-np.log(2.)/a))

        # adjust plot parameters
        ax.tick_params(axis='x', which='both', bottom='on', top='off', labelsize='xx-small')
        ax.tick_params(axis='y', which='both', left='on', right='off', labelsize='xx-small')
        ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:.0f}'))
        ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:.2g}'))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlabel('times [min]', fontsize='x-small')
        ax.set_ylabel(axis_labely, fontsize='x-small')
        ax.legend(loc='best', fontsize='x-small')

    rect = [0.,0.,1.,0.92]
    gs.tight_layout(fig,rect=rect, w_pad=0.1, h_pad=0.1)
    fileout = "{}.pdf".format(fileoutspl)
    print "{:<20s}{:<s}".format('fileout',fileout)
    fig.savefig(fileout,bbox_inches='tight',pad_inches=0)
    plt.close('all')
    return

def get_pearsonr_piecewise_vectors(lag, vectors):
    """
    Return the pearson coefficient of points such as the indexes are separated by j-i=lag.
    vectors=[v1, v2, v3,..., vN].
    Each vector has the form vi=[x1,x2,...,xM], where M can vary.
    """
    N = len(vectors)
    Mi = [len(v) for v in vectors]
    M = np.max(Mi)
    if not (lag < M):
        return np.nan

    pairs=[]
    for i in range(N):
        v = vectors[i]
        Ni = len(v)
        for j in range(Ni-lag):
            pairs.append([v[j],v[j+lag]])

    pairs = np.array(pairs, dtype=np.float_)
    r, pvalue = pearsonr(pairs[:,0], pairs[:,1])
    return r

def plot_lineages_acf(lineages, cells, fileoutspl, fovs=None, attrdict=None, color='black', lw=0.5, ms=2, minnorm=100):
    """
    plot autocorrelation curves (overlaid) for different cell attributes.
    """
    # preliminary check
    if (attrdict is None) or (type(attrdict) != dict) or (len(attrdict) == 0):
        print "List of observables empty!"
        return

    # filter lineages
    if (fovs is None):
        all_lineages = np.concatenate(lineages)
        fovs = {fov: None for fov in np.unique([cells[key].fov for key in all_lineages])}

    selection=[]
    for fov in fovs:
        # determine correct lineages
        if fovs[fov] == None:
            peaks = np.unique([cells[key].peak for key in np.concatenate(lineage_byfov_bypeak(lineages,cells,fov=fov))])
            fovs[fov] = peaks
        peaks = fovs[fov]
        subselection = lineage_byfov_bypeak(lineages,cells, fov=fov, peaks=peaks)
        nlin = len(selection)
        selection.append(subselection)

    selection = np.concatenate(selection)

    # some information
    attributes = np.array(attrdict.keys())
    nattr = len(attributes)
    ncol = nattr
    tau_mean = np.mean([cell.tau for cell in cells.values()])

    # figure
    r = 4./3.
    axdim=3
    figsize=nattr*r*axdim, axdim
    fig = plt.figure(num='none', facecolor='w', figsize=figsize)
    gs = gridspec.GridSpec(1,nattr)

    # plot per attrbute
    for i, attr_y in enumerate(attributes): # row is y-axis
        # get label data
        try:
            axis_labely = attrdict[attr_y]['label']
        except KeyError:
            axis_labely = attr_y

        # add plot
        ax = fig.add_subplot(gs[0,i])

        Ytot=[]
        Xtot=[]
        for lineage in selection:
            # build data
            XX = []
            YY = []
            for key in lineage:
                cell = cells[key]
                try:
                    x = np.array(getattr(cell,'times_min'), dtype=np.float_)
                    y = np.array(getattr(cell,attr_y), dtype=np.float_)
                    idx = np.isfinite(x)
                    if np.isfinite(x).all() and np.isfinite(y).all():
                        XX.append(x)
                        YY.append(y)
                except ValueError:
                    continue
            X = np.concatenate(XX)
            Y = np.concatenate(YY)

            # rescale
            try:
                scale = attrdict[attr_y]['scale']
                Y = Y *scale
            except KeyError:
                pass

            # shift time origin
            x0 = X[0]
            X-=x0 # start at lag = 0
            Xtot.append(X)
            Ytot.append(Y)

        # computing the acf
        kmax = np.argmax([len(x) for x in Xtot])
        X = np.array(Xtot[kmax])
        T=len(X)
        X=X[:T/2]
        T=len(X)
        lags = np.arange(T)
        Z = [get_pearsonr_piecewise_vectors(lag, Ytot) for lag in lags]
        Z = np.array(Z, dtype=np.float_)
        ax.plot(X,Z,'o-', color=color, ms=ms, lw=lw)
        #ax.plot(X,np.sqrt(VAR[idx]),'-', color=color, ms=ms, lw=3*lw)
        ax.axhline(y=0, color='k', linestyle='-', lw=lw)
        ax.axvline(x=tau_mean, color='k', linestyle='--', lw=lw, label='$\\tau={:.0f}$'.format(tau_mean))
        ktau = np.argmin(np.abs(X-tau_mean))
        ax.axhline(y=Z[ktau], color='k', linestyle='-.', lw=lw, label='$r(\\tau)={:.2f}$'.format(Z[ktau]))


        # fitting
        #idx = (np.isfinite(Z)) & (X>0)
        idx = (np.isfinite(Z)) & (Z > 0.)
        Xfit = X[idx]
        Zfit = Z[idx]

        #"""
        ## exponential
        #ax.plot(Xfit,Zfit,'-g', lw=3*lw)
        a = -1. / (np.sum(Zfit) * np.diff(Xfit)[0])
        par0 = [a]
        try:
            par = fit_xy(Xfit,Zfit, par0, funcfit_f=exp_f)
        except:
            par = par0
        xx=np.linspace(Xfit[0],Xfit[-1],1000)
        zz = np.array([exp_f(par,x) for x in xx])
        a = par[0]
        #"""
        """
        ## line
        Yfit = np.log(Zfit)
        #ax.plot(Xfit,Zfit,'-g', lw=3*lw)
        a,b = np.polyfit(Xfit, Yfit, deg=1)
        xx=np.linspace(Xfit[0],Xfit[-1],1000)
        yy=a*xx+b
        zz=np.exp(yy)
        #"""

        ax.plot(xx,zz,'--r', lw=3*lw, label='$\\tau={:.0f}$'.format(-np.log(2.)/a))

        # adjust plot parameters
        ax.tick_params(axis='x', which='both', bottom='on', top='off', labelsize='xx-small')
        ax.tick_params(axis='y', which='both', left='on', right='off', labelsize='xx-small')
        ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:.0f}'))
        ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:.2g}'))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlabel('times [min]', fontsize='x-small')
        ax.set_ylabel(axis_labely, fontsize='x-small')
        ax.legend(loc='best', fontsize='x-small')

    rect = [0.,0.,1.,0.92]
    gs.tight_layout(fig,rect=rect, w_pad=0.1, h_pad=0.1)
    fileout = "{}.pdf".format(fileoutspl)
    print "{:<20s}{:<s}".format('fileout',fileout)
    fig.savefig(fileout,bbox_inches='tight',pad_inches=0)
    plt.close('all')
    return

def plot_distributions(cells, attrdict, fileout, color='darkblue', nbins_max=8):
    if (type(attrdict) != dict) or (len(attrdict) == 0):
        print "List of observables empty!"
        return

    # make figure
    attributes = attrdict.keys()
    ncol = len(attributes)
    r = 1.
    fig = plt.figure(num='none',facecolor='w', figsize=(ncol*r*3,3))

    gs = gridspec.GridSpec(1,ncol,wspace=0.0,hspace=0.0)
    for col in range(ncol):
        # choose attribute
        print "col {:d}".format(col)
        attr = attributes[col]

        # build data
        X = []
        for key,cell in cells.items():
            try:
                x = np.float_(getattr(cell,attr))
                if np.isfinite(x):
                    X.append(x)
            except ValueError:
                continue
        X = np.array(X)

        # rescale
        try:
            scale = attrdict[attr]['scale']
            X = X *scale
        except KeyError:
            pass

        #print len(X)
        mean = np.mean(X)
        std = np.std(X)
        cv = std/mean
        hist,edges = histogram(X)
        left = edges[:-1]
        right = edges[1:]
        idx = (hist != 0.)

        # add plot
        ax = fig.add_subplot(gs[0,col])
        #ax.bar(left=left, height=hist, width=right-left, color='none', lw=0.0, edgecolor=thecolor, alpha=0.6)
        ax.plot(left[idx], hist[idx], '-', color=color, lw=1)

        # annotations
        text = "Mean = {:.4g}\nCV = {:<.0f}%".format(mean,cv*100.)
        ax.set_title(text, fontsize='small')
        #ax.annotate(text,xy=(0.05,0.98),xycoords='axes fraction',ha='left',va='top',color=thecolor, fontsize=fontsize)
        xticks = [mean-std,mean,mean+std]
        #ax.set_xticks(xticks)
        ax.set_yticks([])
        try:
            ax.set_xlabel(attrdict[attr]['label'], fontsize='medium')
        except KeyError:
            ax.set_xlabel(attr, fontsize='medium')
        #ax.set_ylabel('length',fontsize='x-small')
        #ax.tick_params(length=2)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=nbins_max))
        ax.tick_params(axis='both', labelsize='xx-small', pad=2)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    rect = [0.,0.,1.,0.98]
    gs.tight_layout(fig,rect=rect)
    print "{:<20s}{:<s}".format('fileout',fileout)
    fig.savefig(fileout,bbox_inches='tight',pad_inches=0)
    plt.close('all')

    return

def plot_cross_correlations(cells, attrdict, fileout, color1='darkblue', color2='black', nbins_max=8, method='pearsonr', scatter_max_pts=1000, ms=2, bincount_min=10):
    if (type(attrdict) != dict) or (len(attrdict) == 0):
        print "List of observables empty!"
        return

    # make figure
    attributes = attrdict.keys()
    n = len(attributes)
    r = 1.
    fig = plt.figure(num='none',facecolor='w', figsize=(n*r*3,n*3))
    gs = gridspec.GridSpec(n,n,wspace=0.0,hspace=0.2)

    for row in range(n):
        for col in range(n):
            # choose attribute
            print "row {:d} col {:d}".format(row,col)
            attr_row = attributes[row]
            attr_col = attributes[col]

            # build data
            X = []
            Y = []
            for key,cell in cells.items():
                try:
                    x = np.float_(getattr(cell,attr_col))
                    y = np.float_(getattr(cell,attr_row))
                    if np.isfinite(x) and np.isfinite(y):
                        X.append(x)
                        Y.append(y)
                except ValueError:
                    continue
            X = np.array(X)
            #print len(X)
            Y = np.array(Y)

            # rescale
            try:
                scale = attrdict[attr_col]['scale']
                X = X * scale
            except KeyError:
                pass

            try:
                scale = attrdict[attr_row]['scale']
                Y = Y * scale
            except KeyError:
                pass

            xmean = np.mean(X)
            xstd = np.std(X)
            ymean = np.mean(Y)
            ystd = np.std(Y)

            # add plot
            ax = fig.add_subplot(gs[row,col])

            if (col == row):
                hist,edges = histogram(X)
                left = edges[:-1]
                right = edges[1:]
                idx = (hist != 0.)
                ax.plot(left[idx], hist[idx], '-', color=color1, lw=1)

                xticks = [xmean - xstd, xmean, xmean + xstd]
                ax.set_xticks(xticks)
                ax.tick_params(axis='x', which='both', bottom='on', top='off')
                #ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=nbins_max))
                ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:.2g}'))

            else:
                X = (X-xmean)/xstd
                Y = (Y-ymean)/ystd
                idx = np.random.permutation(np.arange(len(X)))[:scatter_max_pts]
                ax.plot(X[idx],Y[idx],'.', ms=ms,color=color1,alpha=0.8)

                if (method == 'pearsonr'):
                    us = 'PE'
                    corr,pvalue = pearsonr(X,Y)
                elif (method == 'spearmanr'):
                    us = 'SP'
                    corr,pvalue = spearmanr(X,Y)
                else:
                    raise ValueError
                ax.annotate('$r_{{{:s}}} = {:<.2f}$'.format(us,corr),xy=(0.5,1.0),xycoords='axes fraction', ha='center',va='top', color=color2, fontsize='small')

                ## make linear fit to binned data
                ### define the binned data set
                res = make_binning(X, Y, bincount_min)
                x_binned = res['x']
                y_binned = res['y']
                ax.plot(x_binned,y_binned, '-o', color=color2, ms=3*ms, lw=1, alpha=1.0)

                ax.set_xticks([])
                ax.tick_params(axis='x', which='both', bottom='off', top='off')
                ax.axis('equal')
            # end if statement
            ax.set_yticks([])
            ax.tick_params(axis='both', labelsize='xx-small', pad=2)
            ax.tick_params(axis='y', which='both', left='off', right='off')

            ax.spines['right'].set_visible(True)
            ax.spines['top'].set_visible(True)
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_visible(True)

            ## write labels
            if (col == 0):
                label = attrdict[attr_row]['label']
                ax.annotate(label, xy=(-0.10,0.5), xycoords='axes fraction', ha='right', va='center', fontsize='medium')

            if (row == 0):
                label = attrdict[attr_col]['label']
                ax.annotate(label, xy=(0.5,1.10), xycoords='axes fraction', ha='center', va='bottom', fontsize='medium')

    rect = [0.,0.,1.,0.98]
    #gs.tight_layout(fig,rect=rect)
    print "{:<20s}{:<s}".format('fileout',fileout)
    fig.savefig(fileout,bbox_inches='tight',pad_inches=0)
    plt.close('all')

    return

def plot_autocorrelations(cells, attrdict, fileout, color1='darkblue', color2='black', nbins_max=8, method='pearsonr', scatter_max_pts=1000, ms=2, bincount_min=10):
    if (type(attrdict) != dict) or (len(attrdict) == 0):
        print "List of observables empty!"
        return

    # make figure
    attributes = attrdict.keys()
    n = len(attributes)
    r = 1.
    fig = plt.figure(num='none',facecolor='w', figsize=(n*r*3,3))
    gs = gridspec.GridSpec(1,n,wspace=0.0,hspace=0.2)

    for col in range(n):
        # choose attribute
        print "col {:d}".format(col)
        attr = attributes[col]

        # build data
        X = []
        Y = []
        for key,cell in cells.items():
            try:
                keym=cell.parent
                cellm = cells[keym]
                x = np.float_(getattr(cellm,attr))
                y = np.float_(getattr(cell,attr))
                if np.isfinite(x) and np.isfinite(y):
                    X.append(x)
                    Y.append(y)
            except ValueError:
                # error in isfinite tests
                continue
            except KeyError:
                # error in cellm=cells[keym] statement
                continue
        X = np.array(X)
        Y = np.array(Y)

        # rescale
        try:
            scale = attrdict[attr]['scale']
            X = X * scale
            Y = Y * scale
        except KeyError:
            pass

        # add plot
        ax = fig.add_subplot(gs[0,col])

        idx = np.random.permutation(np.arange(len(X)))[:scatter_max_pts]
        ax.plot(X[idx],Y[idx],'.', ms=ms,color=color1,alpha=0.8)

        if (method == 'pearsonr'):
            us = 'PE'
            corr,pvalue = pearsonr(X,Y)
        elif (method == 'spearmanr'):
            us = 'SP'
            corr,pvalue = spearmanr(X,Y)
        else:
            raise ValueError
        ax.annotate('$r_{{{:s}}} = {:<.2f}$'.format(us,corr),xy=(0.5,1.0),xycoords='axes fraction', ha='center',va='top', color='k', fontsize='small')

        ## make linear fit to binned data
        ### define the binned data set
        res = make_binning(X, Y, bincount_min)
        x_binned = res['x']
        y_binned = res['y']
        ax.plot(x_binned,y_binned, '-o', color=color2, ms=3*ms, lw=1, alpha=1.0)

        try:
            ax.set_xlabel(attrdict[attr]['label_m'], fontsize='medium')
            ax.set_ylabel(attrdict[attr]['label_d'], fontsize='medium')
        except KeyError:
            ax.set_xlabel(attr + ' mother', fontsize='medium')
            ax.set_ylabel(attr + ' daughter', fontsize='medium')
        ax.axis('equal')
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=nbins_max))
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=nbins_max))
        ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:.2g}'))
        ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:.2g}'))
        #ax.set_xticks([])
        #ax.set_yticks([])
        ax.tick_params(axis='both', labelsize='xx-small', pad=2)
        ax.tick_params(axis='x', which='both', bottom='on', top='off')
        ax.tick_params(axis='y', which='both', left='on', right='off')

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    rect = [0.,0.,1.,0.98]
    gs.tight_layout(fig,rect=rect)
    print "{:<20s}{:<s}".format('fileout',fileout)
    fig.savefig(fileout,bbox_inches='tight',pad_inches=0)
    plt.close('all')

    return

############################################################################
# main
############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Plots of cells measurements.")
    parser.add_argument('pklfile', type=file, help='Pickle file containing the cell dictionary.')
    parser.add_argument('-f', '--paramfile',  type=file, required=True, help='Yaml file containing parameters.')
    parser.add_argument('--distributions',  action='store_true', help='Plot the distributions of cell variables.')
    parser.add_argument('--crosscorrelations',  action='store_true', help='Plot the cross-correlation of cell variables.')
    parser.add_argument('--autocorrelations',  action='store_true', help='Plot the autocorrelation of cell variables.')
    parser.add_argument('-l', '--lineagesfile',  type=file, help='Pickle file containing the list of lineages.')
    namespace = parser.parse_args(sys.argv[1:])
    paramfile = namespace.paramfile.name
    allparams = yaml.load(namespace.paramfile)
    params = allparams['plots']
    cells = pkl.load(namespace.pklfile)
    plot_dist = namespace.distributions
    plot_crosscorr = namespace.crosscorrelations
    plot_autocorr = namespace.autocorrelations

    tdir = os.path.dirname(namespace.pklfile.name)
    print "{:<20s}{:<s}".format('data dir', tdir)
    cellname = os.path.basename(namespace.pklfile.name)
    cellnamespl = os.path.splitext(cellname)[0]
    plotdir = os.path.join(tdir,'plots')
    if not os.path.isdir(plotdir):
        os.makedirs(plotdir)

# plot general statistics
    if plot_dist:
        mm3.information ('Plotting distributions.')
        popdir = os.path.join(plotdir,'population')
        if not os.path.isdir(popdir):
            os.makedirs(popdir)
        try:
            fileout = os.path.join(popdir,'{}_distributions.pdf'.format(cellnamespl))
            plot_distributions(cells, attrdict=params['distributions']['attributes'], fileout=fileout)
        except:
            print "Error with distributions plotting."

    if plot_crosscorr:
        mm3.information ('Plotting cross-correlations.')
        popdir = os.path.join(plotdir,'population')
        if not os.path.isdir(popdir):
            os.makedirs(popdir)
        try:
            fileout = os.path.join(popdir,'{}_cross_correlations.pdf'.format(cellnamespl))
            plot_cross_correlations(cells, attrdict=params['cross correlations']['attributes'], fileout=fileout, **params['cross correlations']['args'])
        except:
            print "Error with cross-correlations plotting."

    if plot_autocorr:
        mm3.information ('Plotting autocorrelations.')
        popdir = os.path.join(plotdir,'population')
        if not os.path.isdir(popdir):
            os.makedirs(popdir)
        try:
            fileout = os.path.join(popdir,'{}_autocorrelations.pdf'.format(cellnamespl))
            plot_autocorrelations(cells, attrdict=params['autocorrelations']['attributes'], fileout=fileout, **params['autocorrelations']['args'])
        except:
            print "Error with autocorrelations plotting."

# lineages
    if namespace.lineagesfile != None:
        mm3.information ('Plotting lineages.')
        lineages = pkl.load(namespace.lineagesfile)

        if 'plot_lineages_byfov' in params:
            mm3.information ('Plotting lineages -- by fov.')
            lindir = os.path.join(plotdir,'lineages_byfov')
            if not os.path.isdir(lindir):
                os.makedirs(lindir)
            if 'fovs' in params['plot_lineages_byfov']:
                fovs = params['plot_lineages_byfov']['fovs']
                fileoutspl = os.path.join(lindir,'{}_lineages'.format(cellnamespl))
                plot_lineages_byfov(lineages,cells,fileoutspl, fovs=fovs, **params['plot_lineages_byfov']['args'])


        if 'plot_lineages_with_growth_rate' in params:
            mm3.information ('Plotting lineages individually -- growth rate.')
            lindir = os.path.join(plotdir,'lineages_with_growth_rate')
            if not os.path.isdir(lindir):
                os.makedirs(lindir)
            fileoutspl = os.path.join(lindir,'{}_lineages'.format(cellnamespl))
            if 'fovs' in params['plot_lineages_with_growth_rate']:
                fovs = params['plot_lineages_with_growth_rate']['fovs']
                selection = []
                if not (fovs is None):
                    for fov in fovs:
                        peaks = fovs[fov]
                        selection = lineage_byfov_bypeak(lineages, cells, fov=fov, peaks=peaks)
            else:
                selection = lineages

            for lineage in selection:
                plot_lineage_with_growth_rate(lineage, cells, fileoutspl, **params['plot_lineages_with_growth_rate']['args'])

        if 'plot_lineages_variables' in params:
            mm3.information ('Plotting lineages individually -- variables evolution.')
            lindir = os.path.join(plotdir,'lineages_variables')
            if not os.path.isdir(lindir):
                os.makedirs(lindir)
            fileoutspl = os.path.join(lindir,'{}_lineages'.format(cellnamespl))
            if 'fovs' in params['plot_lineages_variables']:
                fovs = params['plot_lineages_variables']['fovs']
                selection = []
                if not (fovs is None):
                    for fov in fovs:
                        peaks = fovs[fov]
                        selection = lineage_byfov_bypeak(lineages, cells, fov=fov, peaks=peaks)
            else:
                selection = lineages

            for lineage in selection:
                plot_lineage_variables(lineage, cells,fileoutspl, attrdict=params['plot_lineages_variables']['attributes'], **params['plot_lineages_variables']['args'])

        if 'plot_lineages_correlations' in params:
            mm3.information ('Plotting lineages individually -- variables correlations.')
            lindir = os.path.join(plotdir,'lineages_correlations')
            if not os.path.isdir(lindir):
                os.makedirs(lindir)
            fileoutspl = os.path.join(lindir,'{}_lineages'.format(cellnamespl))
            if 'fovs' in params['plot_lineages_correlations']:
                fovs = params['plot_lineages_correlations']['fovs']
                selection = []
                if not (fovs is None):
                    for fov in fovs:
                        peaks = fovs[fov]
                        selection = lineage_byfov_bypeak(lineages, cells, fov=fov, peaks=peaks)
            else:
                selection = lineages

            for lineage in selection:
                plot_lineage_correlations(lineage, cells,fileoutspl, attrdict=params['plot_lineages_correlations']['attributes'], **params['plot_lineages_correlations']['args'])

        if 'plot_lineages_acf' in params:
            mm3.information ('Plotting lineages -- autocorrelation functions.')
            lindir = os.path.join(plotdir,'lineages_acf')
            if not os.path.isdir(lindir):
                os.makedirs(lindir)
            fileoutspl = os.path.join(lindir,'{}_lineages_acf'.format(cellnamespl))

            if 'fovs' in params['plot_lineages_acf']:
                fovs = params['plot_lineages_acf']['fovs']
                plot_lineages_acf(lineages,cells,fileoutspl,attrdict=params['plot_lineages_acf']['attributes'],fovs=fovs, **params['plot_lineages_acf']['args'])
