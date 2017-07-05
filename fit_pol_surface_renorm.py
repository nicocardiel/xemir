from __future__ import division
from __future__ import print_function

import numpy as np


def normalize_to_pm_one(x):
    """Normalize x to [-1,1] range.

    The normalization is performed as explained in Appendix B1 of
    Cardiel (2009, MNRAS, 396, 680), where the direct and inverse
    transformations are given, respectively, by:
    x_norm = bx * x - cx
    x = (x_norm + cx)/bx

    Parameters
    ----------
    x : 1d numpy array, floats
        Array of values to be normalized.

    Returns
    -------
    xnorm : 1d numpy array, floats
        Array of normalized values.
    bx : float
        Coefficient bx.
    cx : float
        Coefficient cx.

    """

    xmin = x.min()
    xmax = x.max()
    bx = 2./(xmax-xmin)
    cx = bx * (xmin+xmax)/2.
    xnorm = bx * x - cx

    return xnorm, bx, cx


def fit_pol_surface_renorm(x, y, z, degx, degy, maskdeg=None):
    """Fit polynomial surface renormalizing data ranges prior to fit.

    The fitted data can be located arbitrarily in the plane X-Y.
    The data ranges are normalized to the [-1,1] interval in order to
    reduce the impact of numerical errors.

    Parameters
    ----------
    x : 1d numpy array, floats
        X values of the (x,y,z) points to be fitted.
    y : 1d numpy array, floats
        Y values of the (x,y,z) points to be fitted.
    z : 1d numpy array, floats
        Z values of the (x,y,z) points to be fitted.
    degx : int
        Maximum degree for X values.
    degy : int
        Maximum degree for Y values.
    maskdeg : None or 2d numpy array, bool
        If mask[ix, iy] == True, the term x**ix + y**iy is employed
        in the fit. If maskdeg == None, all the terms are fitted.

    Returns
    -------
    coeff : 1d numpy array, float
        Fitted coefficients.
    norm_factors : tuple of floats
        Coefficients bx, cx, by, cy, bz, and cz employed to normalize
        the (x,y,z) values to the [-1,1] interval.
    
    """

    # protections
    if type(x) is not np.ndarray:
        raise ValueError("x must be a numpy.ndarray")
    elif x.ndim != 1:
        raise ValueError("x.ndim is not 1")
    if type(y) is not np.ndarray:
        raise ValueError("y must be a numpy.ndarray")
    elif y.ndim != 1:
        raise ValueError("y.ndim is not 1")
    if type(y) is not np.ndarray:
        raise ValueError("y must be a numpy.ndarray")
    elif z.ndim != 1:
        raise ValueError("z.ndim is not 1")
    if len(x) != len(y) or len(x) != len(z):
        raise ValueError("x, y and z must have the same length")

    # normalize data ranges to [-1,1] interval
    xnorm, bx, cx = normalize_to_pm_one(x)
    ynorm, by, cy = normalize_to_pm_one(y)
    znorm, bz, cz = normalize_to_pm_one(z)
    norm_factors = (bx, cx, by, cy, bz, cz)

    # effective maskdeg
    if maskdeg is None:
        maskdeg = np.ones((degx+1, degy+1), dtype=bool)
    else:
        if maskdeg.shape != (degx+1, degy+1):
            raise ValueError("Unexpected maskdeg.shape=", maskdeg.shape)

    # matrix of the system of equations
    a_matrix = None
    for ix in range(degx + 1):
        for iy in range(degy + 1):
            if maskdeg[ix, iy]:
                if a_matrix is None:
                    # use np.vstack even in this case to guarantee
                    # that a_matrix is two-dimensional even when only
                    # a single unknown is to be fitted
                    a_matrix = np.vstack([xnorm**ix * ynorm**iy])
                else:
                    a_matrix = np.vstack([a_matrix, xnorm**ix * ynorm**iy])

    # transpose previous matrix
    a_matrix = a_matrix.T

    # perform fit using least squares fitting
    coeff = np.linalg.lstsq(a_matrix, znorm)[0]

    return coeff, norm_factors


def eval_pol_surface_renorm(x, y, coeff, norm_factors,
                            degx, degy, maskdeg=None):
    """Evaluates polynomial surface at points (x,y).

    This function evaluates the polynomial fit computed with the
    function fit_pol_surface_renorm(). The input parameters 'degx',
    'degy', and 'maskdeg' must be the same employed with the
    function fit_pol_surface_renorm(). In addition, the input
    parameters 'coeff', and 'norm_factors' must be the output of that
    function.

    Parameters
    ----------
    x : float or np.array
        X values where the function will be evaluated.
    y : float or np.array
        Y values where the function will be evaluated.
    coeff : 1d numpy array, float
        Coefficients previously fitted with fit_pol_surface_renorm().
    norm_factors : tuple of floats
        Coefficients bx, cx, by, cy, bz, and cz employed to normalize
        the (x,y,z) values to the [-1,1] interval.
    degx : int
        Maximum degree for X values.
    degy : int
        Maximum degree for Y values.
    maskdeg : None or 2d numpy array, bool
        If mask[ix, iy] == True, the term x**ix + y**iy was employed
        in the fit. If maskdeg == None, all the terms were fitted.

    Returns
    -------
    z : float or np.array
        Z values corresponding to the evaluation of the polynomial
        surface at the (x,y) values.

    """

    if type(x) is not np.ndarray:
        raise ValueError("x must be a numpy.ndarray")
    if type(y) is not np.ndarray:
        raise ValueError("y must be a numpy.ndarray")
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")

    # normalize data ranges to [-1,1] interval
    bx, cx, by, cy, bz, cz = norm_factors
    xnorm = bx * x - cx
    ynorm = by * y - cy

    # effective maskdeg
    if maskdeg is None:
        maskdeg = np.ones((degx+1, degy+1), dtype=bool)
    else:
        if maskdeg.shape != (degx+1, degy+1):
            raise ValueError("Unexpected maskdeg.shape=", maskdeg.shape)

    znorm = np.zeros_like(x)
    k = 0
    for ix in range(degx + 1):
        for iy in range(degy + 1):
            if maskdeg[ix, iy]:
                znorm += coeff[k] * xnorm**ix * ynorm**iy
                k += 1

    # revert normalization
    z = (znorm + cz) / bz

    return z
