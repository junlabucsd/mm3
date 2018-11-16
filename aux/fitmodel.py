import numpy as np
from copy import copy
from scipy.optimize import least_squares

class FitRes:
    """
    Object used to fit a data set to a particular function.
    Input:
        o x: x coordinates of the input data set.
        o y: y coordinates of the input data set.
        o s: standard deviations for the y values.
        o funcfit: fitting function.
    """
    def __init__(self, x, y,funcfit_f, funcfit_df, yerr=None):
        self.x = np.array(x)
        self.y = np.array(y)
        self.funcfit_f = funcfit_f
        self.funcfit_df = funcfit_df
        if (yerr == None):
            self.yerr = np.ones(self.x.shape)
        else:
            self.yerr = yerr

    def residual_f(self, par):
        """
        Return the vector of residuals.
        """
        fx = np.array([self.funcfit_f(par,xi) for xi in self.x])
        return (fx-self.y)/self.yerr

    def residual_df(self, par):
        dfx = np.array([np.array(self.funcfit_df(par,xi))/si for (xi,si) in zip(self.x,self.yerr)])
        return dfx

