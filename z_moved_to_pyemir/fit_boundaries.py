from __future__ import division
from __future__ import print_function

import argparse
from astropy.io import fits
from copy import deepcopy
from datetime import datetime
import json
from lmfit import Minimizer, Parameters, report_fit
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
from uuid import uuid4

from numina.array.display.pause_debugplot import pause_debugplot
from numina.array.display.ximshow import ximshow
from emirdrp.core import EMIR_NBARS

from ccd_line import SpectrumTrail

from emir_definitions import NAXIS1_EMIR
from emir_definitions import NAXIS2_EMIR
from numina.array.display.pause_debugplot import DEBUGPLOT_CODES

EXPECTED_PARAMETER_LIST = (
    'c2', 'c4', 'ff', 'slit_gap', 'slit_height',
    'theta0_origin', 'theta0_slope', 'x0', 'y0', 'y_baseline'
)

EXPECTED_PARAMETER_LIST_EXTENDED = (
    mainpar + '_' + subpar for mainpar in EXPECTED_PARAMETER_LIST
    for subpar in ['a0s', 'a1s', 'a2s']
)

FUNCTION_EVALUATIONS = 0


def integrity_check(bounddict):
    """Integrity check of 'bounddict' content.

    Parameters
    ----------
    bounddict : JSON structure
        Structure employed to store bounddict information.

    """

    if 'meta-info' not in bounddict.keys():
        raise ValueError('"meta-info" not found in JSON file')
    if 'description' not in bounddict['meta-info'].keys():
        raise ValueError('"description" not found in JSON file')
    if bounddict['meta-info']['description'] != \
            'slitlet boundaries from fits to continuum-lamp exposures':
        raise ValueError('Unexpected "description" in JSON file')

    grism = bounddict['tags']['grism']
    print('>>> grism...:', grism)
    spfilter = bounddict['tags']['filter']
    print('>>> filter..:', spfilter)

    valid_slitlets = ["slitlet" + str(i).zfill(2) for i in
                      range(1, EMIR_NBARS + 1)]
    read_slitlets = bounddict['contents'].keys()
    read_slitlets.sort()

    for tmp_slitlet in read_slitlets:
        if tmp_slitlet not in valid_slitlets:
            raise ValueError("Unexpected slitlet key: " + tmp_slitlet)
        # for each slitlet, check valid DATE-OBS (ISO 8601)
        read_dateobs = bounddict['contents'][tmp_slitlet].keys()
        read_dateobs.sort()
        for tmp_dateobs in read_dateobs:
            try:
                datetime.strptime(tmp_dateobs, "%Y-%m-%dT%H:%M:%S.%f")
            except ValueError:
                print("Unexpected date_obs key: " + tmp_dateobs)
                raise
            # for each DATE-OBS, check expected fields
            tmp_dict = bounddict['contents'][tmp_slitlet][tmp_dateobs]
            valid_keys = ["boundary_coef_lower",
                          "boundary_coef_upper",
                          "boundary_xmax_lower",
                          "boundary_xmax_upper",
                          "boundary_xmin_lower",
                          "boundary_xmin_upper",
                          "csu_bar_left",
                          "csu_bar_right",
                          "csu_bar_slit_center",
                          "csu_bar_slit_width",
                          "rotang",
                          "xdtu",
                          "xdtu_0",
                          "ydtu",
                          "ydtu_0",
                          "zdtu",
                          "zdtu_0",
                          "zzz_info1",
                          "zzz_info2"]
            read_keys = tmp_dict.keys()
            for tmp_key in read_keys:
                if tmp_key not in valid_keys:
                    print("ERROR:")
                    print("grism...:", grism)
                    print("slitlet.:", tmp_slitlet)
                    print("date_obs:", tmp_dateobs)
                    raise ValueError("Unexpected key " + tmp_key)
            for tmp_key in valid_keys:
                if tmp_key not in read_keys:
                    print("ERROR:")
                    print("grism...:", grism)
                    print("slitlet.:", tmp_slitlet)
                    print("date_obs:", tmp_dateobs)
                    raise ValueError("Expected key " + tmp_key + " not found")
    print("* Integrity check OK!")


def exvp_scalar(x, y, x0, y0, c2, c4, theta0, ff):
    """Convert virtual pixel to real pixel.

    Parameters
    ----------
    x : float
        X coordinate (pixel).
    y : float
        Y coordinate (pixel).
    x0 : float
        X coordinate of reference pixel, in units of 1E3.
    y0 : float
        Y coordinate of reference pixel, in units of 1E3.
    c2 : float
        Coefficient corresponding to the term r**2 in distortion
        equation, in units of 1E4.
    c4 : float
        Coefficient corresponding to the term r**4 in distortion
        equation, in units of 1E9
    theta0 : float
        Additional rotation angle (radians).
    ff : float
        Scaling factor to be applied to the Y axis.

    Returns
    -------
    xdist, ydist : tuple of floats
        Distorted coordinates.

    """

    # plate scale: 0.1944 arcsec/pixel
    # conversion factor (in radian/pixel)
    factor = 0.1944 * np.pi/(180.0*3600)
    # distance from image center (pixels)
    r_pix = np.sqrt((x - x0*1000)**2 + (y - y0*1000)**2)
    # distance from imagen center (radians)
    r_rad = factor * r_pix
    # radial distortion: this number is 1.0 for r=0 and increases
    # slightly (reaching values around 1.033) for r~sqrt(2)*1024
    # (the distance to the corner of the detector measured from the
    # center)
    rdist = (1 +
             c2 * 1.0E4 * r_rad**2 +
             c4 * 1.0E9 * r_rad**4)
    # angle measured from the Y axis towards the X axis
    theta = np.arctan((x - x0*1000)/(y - y0*1000))
    if y < y0*1000:
        theta = theta - np.pi
    # distorted coordinates
    xdist = (rdist * r_pix * np.sin(theta+theta0)) + x0*1000
    ydist = (ff * rdist * r_pix * np.cos(theta+theta0)) + y0*1000

    return xdist, ydist


def exvp(x, y, x0, y0, c2, c4, theta0, ff):
    """Convert virtual pixel(s) to real pixel(s).

    This function makes use of exvp_scalar(), which performs the
    conversion for a single point (x, y), over an array of X and Y
    values.

    Parameters
    ----------
    x : array-like
        X coordinate (pixel).
    y : array-like
        Y coordinate (pixel).
    x0 : float
        X coordinate of reference pixel.
    y0 : float
        Y coordinate of reference pixel.
    c2 : float
        Coefficient corresponding to the term r**2 in distortion
        equation.
    c4 : float
        Coefficient corresponding to the term r**4 in distortion
        equation.
    theta0 : float
        Additional rotation angle (radians).
    ff : float
        Scaling factor to be applied to the Y axis.

    Returns
    -------
    xdist, ydist : tuple of floats (or two arrays of floats)
        Distorted coordinates.

    """

    if all([np.isscalar(x), np.isscalar(y)]):
        xdist, ydist = exvp_scalar(x, y, x0=x0, y0=y0,
                                   c2=c2, c4=c4, theta0=theta0, ff=ff)
        return xdist, ydist
    elif any([np.isscalar(x), np.isscalar(y)]):
        raise ValueError("invalid mixture of scalars and arrays")
    else:
        xdist = []
        ydist = []
        for x_, y_ in zip(x, y):
            xdist_, ydist_ = exvp_scalar(x_, y_, x0=x0, y0=y0,
                                         c2=c2, c4=c4, theta0=theta0, ff=ff)
            xdist.append(xdist_)
            ydist.append(ydist_)
        return np.array(xdist), np.array(ydist)


def return_params(islitlet, csu_bar_slit_center, params, parmodel):
    """Return individual model parameters from object of type Parameters.

    Parameters
    ----------
    islitlet : int
        Number of slitlet.
    csu_bar_slit_center : float
        CSU bar slit center, in mm.
    params : :class:`~lmfit.parameter.Parameters`
        Parameters to be employed in the prediction of the distorted
        boundaries.
    parmodel : str
        Model to be assumed. Allowed values are 'longslit' and
        'multislit'.


    Returns
    -------
    c2 : float
        Coefficient corresponding to the term r**2 in distortion
        equation.
    c4 : float
        Coefficient corresponding to the term r**4 in distortion
        equation.
    ff : float
        Scaling factor to be applied to the Y axis.
    slit_gap : float
        Slit gap.
    slit_height : float
        Slit height.
    theta0 : float
        Additional rotation angle (radians).
    x0 : float
        X coordinate of reference pixel.
    y0 : float
        Y coordinate of reference pixel.
    y_baseline : float
        Y coordinate employed as baseline.

    """

    if parmodel == "longslit":
        # set each variable in EXPECTED_PARAMETER_LIST to the value
        # transferred through 'params'
        c2 = params['c2'].value
        c4 = params['c4'].value
        ff = params['ff'].value
        slit_gap = params['slit_gap'].value
        slit_height = params['slit_height'].value
        theta0_origin = params['theta0_origin'].value
        theta0_slope = params['theta0_slope'].value
        x0 = params['x0'].value
        y0 = params['y0'].value
        y_baseline = params['y_baseline'].value
    else:
        # set each variable in EXPECTED_PARAMETER_LIST_EXTENDED to the value
        # transferred through 'params'
        c2_a0s = params['c2_a0s'].value
        c2_a1s = params['c2_a1s'].value / 1E3
        c2_a2s = params['c2_a2s'].value / 1E6
        c2 = c2_a0s + \
             c2_a1s * csu_bar_slit_center + \
             c2_a2s * csu_bar_slit_center ** 2
        # ---
        c4_a0s = params['c4_a0s'].value
        c4_a1s = params['c4_a1s'].value / 1E3
        c4_a2s = params['c4_a2s'].value / 1E6
        c4 = c4_a0s + \
             c4_a1s * csu_bar_slit_center + \
             c4_a2s * csu_bar_slit_center ** 2
        # ---
        ff_a0s = params['ff_a0s'].value
        ff_a1s = params['ff_a1s'].value / 1E3
        ff_a2s = params['ff_a2s'].value / 1E6
        ff = ff_a0s + \
             ff_a1s * csu_bar_slit_center + \
             ff_a2s * csu_bar_slit_center ** 2
        # ---
        slit_gap_a0s = params['slit_gap_a0s'].value
        slit_gap_a1s = params['slit_gap_a1s'].value / 1E3
        slit_gap_a2s = params['slit_gap_a2s'].value / 1E6
        slit_gap = slit_gap_a0s + \
                   slit_gap_a1s * csu_bar_slit_center + \
                   slit_gap_a2s * csu_bar_slit_center ** 2
        # ---
        slit_height_a0s = params['slit_height_a0s'].value
        slit_height_a1s = params['slit_height_a1s'].value / 1E3
        slit_height_a2s = params['slit_height_a2s'].value / 1E6
        slit_height = slit_height_a0s + \
                      slit_height_a1s * csu_bar_slit_center + \
                      slit_height_a2s * csu_bar_slit_center ** 2
        # ---
        theta0_origin_a0s = params['theta0_origin_a0s'].value
        theta0_origin_a1s = params['theta0_origin_a1s'].value / 1E3
        theta0_origin_a2s = params['theta0_origin_a2s'].value / 1E6
        theta0_origin = theta0_origin_a0s + \
                        theta0_origin_a1s * csu_bar_slit_center + \
                        theta0_origin_a2s * csu_bar_slit_center ** 2
        # ---
        theta0_slope_a0s = params['theta0_slope_a0s'].value
        theta0_slope_a1s = params['theta0_slope_a1s'].value / 1E3
        theta0_slope_a2s = params['theta0_slope_a2s'].value / 1E6
        theta0_slope = theta0_slope_a0s + \
                       theta0_slope_a1s * csu_bar_slit_center + \
                       theta0_slope_a2s * csu_bar_slit_center ** 2
        # ---
        x0_a0s = params['x0_a0s'].value
        x0_a1s = params['x0_a1s'].value / 1E3
        x0_a2s = params['x0_a2s'].value / 1E6
        x0 = x0_a0s + \
             x0_a1s * csu_bar_slit_center + \
             x0_a2s * csu_bar_slit_center ** 2
        # ---
        y0_a0s = params['y0_a0s'].value
        y0_a1s = params['y0_a1s'].value / 1E3
        y0_a2s = params['y0_a2s'].value / 1E6
        y0 = y0_a0s + \
             y0_a1s * csu_bar_slit_center + \
             y0_a2s * csu_bar_slit_center ** 2
        # ---
        y_baseline_a0s = params['y_baseline_a0s'].value
        y_baseline_a1s = params['y_baseline_a1s'].value / 1E3
        y_baseline_a2s = params['y_baseline_a2s'].value / 1E6
        y_baseline = y_baseline_a0s + \
                     y_baseline_a1s * csu_bar_slit_center + \
                     y_baseline_a2s * csu_bar_slit_center ** 2

    theta0 = theta0_origin / 1E3 + theta0_slope / 1E4 * islitlet

    return c2, c4, ff, slit_gap, slit_height, theta0, x0, y0, y_baseline


def expected_distorted_boundaries(islitlet, csu_bar_slit_center,
                                  borderlist, params, parmodel,
                                  numpts, deg, debugplot=0):
    """Return polynomial coefficients of expected distorted boundaries.

    Parameters
    ----------
    islitlet : int
        Number of slitlet.
    csu_bar_slit_center : float
        CSU bar slit center, in mm.
    borderlist : list of floats
        Each float provides the fractional vertical location of the
        spectrum trail relative to the lower boundary. In other words,
        0.0 corresponds to the lower boundary, 1.0 to the upper
        boundary, and any number in the interval (0,1) will be a
        spectral trail in between.
    params : :class:`~lmfit.parameter.Parameters`
        Parameters to be employed in the prediction of the distorted
        boundaries.
    parmodel : str
        Model to be assumed. Allowed values are 'longslit' and
        'multislit'.
    numpts : int
        Number of points in which the X-range interval is subdivided
        before fitting the returned polynomial(s).
    deg : int
        Degree of the fitted polynomial.
    debugplot : int
        Debugging level for messages and plots. For details see
        'numina.array.display.pause_debugplot.py'.

    Returns
    -------
    list_spectrails : list of SpectrumTrail objects
        List containing the fitted spectrum trails.

    """

    c2, c4, ff, slit_gap, slit_height, theta0, x0, y0, y_baseline = \
        return_params(islitlet, csu_bar_slit_center, params, parmodel)

    xp = np.linspace(1, NAXIS1_EMIR, numpts)
    slit_dist = (slit_height * 10) + slit_gap

    # undistorted (constant) y-coordinate of the lower and upper boundaries
    ybottom = y_baseline * 100 + (islitlet - 1) * slit_dist
    ytop = ybottom + (slit_height * 10)

    list_spectrails = []
    for borderval in borderlist:
        yvalue = ybottom + borderval * (ytop - ybottom)
        # undistorted boundary
        yp_value = np.ones(numpts) * yvalue
        # distorted boundary
        xdist, ydist = exvp(xp, yp_value, x0=x0, y0=y0,
                            c2=c2, c4=c4, theta0=theta0, ff=ff)
        spectrail = SpectrumTrail()  # declare SpectrumTrail instance
        spectrail.fit(x=xdist, y=ydist, deg=deg, debugplot=debugplot)
        list_spectrails.append(spectrail)

    return list_spectrails


def expected_distorted_frontiers(islitlet, csu_bar_slit_center,
                                 params, parmodel,
                                 numpts, deg, debugplot=0):
    """Return polynomial coefficients of expected distorted frontiers.

    Note that the frontiers are computed as the polynomials that extend
    the slitlet region defined by its boundaries, encompassing half the
    slit gap between consecutive slitlets.

    Parameters
    ----------
    islitlet : int
        Number of slitlet.
    csu_bar_slit_center : float
        CSU bar slit center, in mm.
    params : :class:`~lmfit.parameter.Parameters`
        Parameters to be employed in the prediction of the distorted
        boundaries.
    parmodel : str
        Model to be assumed. Allowed values are 'longslit' and
        'multislit'.
    numpts : int
        Number of points in which the X-range interval is subdivided
        before fitting the returned polynomials.
    deg : int
        Degree of the fitted polynomial.
    debugplot : int
        Debugging level for messages and plots. For details see
        'numina.array.display.pause_debugplot.py'.

    Returns
    -------
    list_frontiers : list of SpectrumTrail objects
        List containing the polynomials defining the two slitlet
        frontiers (lower and upper, respectively).

    """

    c2, c4, ff, slit_gap, slit_height, theta0, x0, y0, y_baseline = \
        return_params(islitlet, csu_bar_slit_center, params, parmodel)

    xp = np.linspace(1, NAXIS1_EMIR, numpts)
    slit_dist = (slit_height * 10) + slit_gap

    # undistorted (constant) y-coordinate of the lower and upper frontiers
    ybottom = y_baseline * 100 + (islitlet - 1) * slit_dist - slit_gap / 2
    ytop = ybottom + (slit_height * 10) + slit_gap

    list_frontiers = []
    for i in range(2):
        if i == 0:
            yvalue = ybottom
        else:
            yvalue = ytop
        # undistorted boundary
        yp_value = np.ones(numpts) * yvalue
        # distorted boundary
        xdist, ydist = exvp(xp, yp_value, x0=x0, y0=y0,
                            c2=c2, c4=c4, theta0=theta0, ff=ff)
        spectrail = SpectrumTrail()  # declare SpectrumTrail instance
        spectrail.fit(x=xdist, y=ydist, deg=deg, debugplot=debugplot)
        list_frontiers.append(spectrail)

    return list_frontiers


def fun_residuals(params, parmodel, bounddict, numresolution,
                  islitmin, islitmax, debugplot):
    """Function to be minimised.

    Parameters
    ----------
    params : :class:`~lmfit.parameter.Parameters`
        Parameters to be employed in the prediction of the distorted
        boundaries.
    parmodel : str
        Model to be assumed. Allowed values are 'longslit' and
        'multislit'.
    bounddict : JSON structure
        Structure employed to store bounddict information.
    numresolution : int
        Number of points in which the X-range interval is subdivided
        before computing the residuals.
    islitmin : int
        Minimum slitlet number.
    islitmax : int
        Maximum slitlet number.
    debugplot : int
        Debugging level for messages and plots. For details see
        'numina.array.display.pause_debugplot.py'.

    Returns
    -------
    global_residual : float
        Squared root of the averaged sum of squared residuals.

    """

    global FUNCTION_EVALUATIONS
    global_residual = 0.0
    nsummed = 0

    read_slitlets = bounddict['contents'].keys()
    # read_slitlets.sort()  # this is not really necessary
    for tmp_slitlet in read_slitlets:
        islitlet = int(tmp_slitlet[7:])
        if islitmin <= islitlet <= islitmax:
            read_dateobs = bounddict['contents'][tmp_slitlet].keys()
            # read_dateobs.sort()  # this is not really necessary
            for tmp_dateobs in read_dateobs:
                tmp_dict = bounddict['contents'][tmp_slitlet][tmp_dateobs]
                csu_bar_slit_center = tmp_dict['csu_bar_slit_center']
                # expected boundaries using provided parameters
                list_spectrails = expected_distorted_boundaries(
                        islitlet, csu_bar_slit_center,
                        [0, 1], params, parmodel,
                        numpts=numresolution, deg=5, debugplot=0
                    )
                poly_lower_expected = list_spectrails[0].poly_funct
                poly_upper_expected = list_spectrails[1].poly_funct
                # measured lower boundary
                poly_lower_measured = np.polynomial.Polynomial(
                    tmp_dict['boundary_coef_lower']
                )
                xmin_lower_bound = tmp_dict['boundary_xmin_lower']
                xmax_lower_bound = tmp_dict['boundary_xmax_lower']
                dx = (xmax_lower_bound - xmin_lower_bound) / 20
                xdum_lower = np.linspace(xmin_lower_bound + dx,
                                         xmax_lower_bound - dx,
                                         num=numresolution)
                # distance between expected and measured polynomials
                poly_diff = poly_lower_expected - poly_lower_measured
                global_residual += np.sum(poly_diff(xdum_lower)**2)
                nsummed += numresolution
                # measured upper boundary
                poly_upper_measured = np.polynomial.Polynomial(
                    tmp_dict['boundary_coef_upper']
                )
                xmin_upper_bound = tmp_dict['boundary_xmin_upper']
                xmax_upper_bound = tmp_dict['boundary_xmax_upper']
                dx = (xmax_upper_bound - xmin_upper_bound) / 20
                xdum_upper = np.linspace(xmin_upper_bound + dx,
                                         xmax_upper_bound - dx,
                                         num=numresolution)
                # distance between expected and measured polynomials
                poly_diff = poly_upper_expected - poly_upper_measured
                global_residual += np.sum(poly_diff(xdum_upper)**2)
                nsummed += numresolution

    if nsummed > 0:
        global_residual = np.sqrt(global_residual/nsummed)
    if debugplot >= 10:
        FUNCTION_EVALUATIONS += 1
        print('-' * 79)
        print('>>> Number of function evaluations:', FUNCTION_EVALUATIONS)
        print('>>> global residual...............:', global_residual)
        params.pretty_print()
    return global_residual


def overplot_boundaries_from_bounddict(bounddict, micolors, linetype='-'):
    """Overplot boundaries on current plot.

    Parameters
    ----------
    bounddict : JSON structure
        Structure employed to store bounddict information.
    micolors : list of char
        List with two characters corresponding to alternating colors
        for odd and even slitlets.
    linetype : str
        Line type.

    """

    for islitlet in range(1, EMIR_NBARS + 1):
        tmpcolor = micolors[islitlet % 2]
        tmp_slitlet = 'slitlet' + str(islitlet).zfill(2)
        if tmp_slitlet in bounddict['contents'].keys():
            read_dateobs = bounddict['contents'][tmp_slitlet].keys()
            read_dateobs.sort()
            for tmp_dateobs in read_dateobs:
                tmp_dict = bounddict['contents'][tmp_slitlet][tmp_dateobs]
                # lower boundary
                pol_lower_measured = np.polynomial.Polynomial(
                    tmp_dict['boundary_coef_lower']
                )
                xdum = np.linspace(1, NAXIS1_EMIR, num=NAXIS1_EMIR)
                ydum = pol_lower_measured(xdum)
                plt.plot(xdum, ydum, tmpcolor + linetype)
                pol_upper_measured = np.polynomial.Polynomial(
                    tmp_dict['boundary_coef_upper']
                )
                ydum = pol_upper_measured(xdum)
                plt.plot(xdum, ydum, tmpcolor + linetype)


def overplot_boundaries_from_params(ax, params, parmodel,
                                    list_islitlet_lower,
                                    list_islitlet_upper,
                                    list_csu_bar_slit_center,
                                    micolors, linetype='--'):
    """Overplot boundaries computed from fitted parameters.

    Parameters
    ----------
    ax : matplotlib axes
        Current plot axes.
    params : :class:`~lmfit.parameter.Parameters`
        Parameters to be employed in the prediction of the distorted
        boundaries.
    parmodel : str
        Model to be assumed. Allowed values are 'longslit' and
        'multislit'.
    list_islitlet_lower : list of integers
        Slitlet numbers corresponding to the lower boundary of pseudo
        longslits.
    list_islitlet_upper : list of integers
        Slitlet numbers corresponding to the upper bounday of pseudo
        longslits.
    list_csu_bar_slit_center : list of floats
        CSU bar slit centers of the pseudo longslits.
    micolors : list of char
        List with two characters corresponding to alternating colors
        for odd and even slitlets.
    linetype : str
        Line type.

    """

    for islitlet_lower, islitlet_upper, csu_bar_slit_center in \
            zip(list_islitlet_lower,
                list_islitlet_upper,
                list_csu_bar_slit_center):
        tmpcolor = micolors[islitlet_lower % 2]
        pol_lower_expected = expected_distorted_boundaries(
            islitlet_lower, csu_bar_slit_center,
            [0], params, parmodel, numpts=101, deg=5, debugplot=0
        )[0].poly_funct
        pol_upper_expected = expected_distorted_boundaries(
            islitlet_upper, csu_bar_slit_center,
            [1], params, parmodel, numpts=101, deg=5, debugplot=0
        )[0].poly_funct
        xdum = np.linspace(1, NAXIS1_EMIR, num=NAXIS1_EMIR)
        ydum = pol_lower_expected(xdum)
        plt.plot(xdum, ydum, tmpcolor + linetype)
        ydum = pol_upper_expected(xdum)
        plt.plot(xdum, ydum, tmpcolor + linetype)
        # slitlet label
        yc_lower = pol_lower_expected(NAXIS1_EMIR / 2 + 0.5)
        yc_upper = pol_upper_expected(NAXIS1_EMIR / 2 + 0.5)
        if islitlet_lower == islitlet_upper:
            ax.text(NAXIS1_EMIR / 2 + 0.5, (yc_lower + yc_upper) / 2,
                    str(islitlet_lower),
                    fontsize=10, va='center', ha='center',
                    bbox=dict(boxstyle="round,pad=0.1",
                              fc="white", ec="grey"),
                    color=tmpcolor, fontweight='bold',
                    backgroundcolor='white')
        else:
            ax.text(NAXIS1_EMIR / 2 + 0.5, (yc_lower + yc_upper) / 2,
                    str(islitlet_lower) + '-' + str(islitlet_upper),
                    fontsize=10, va='center', ha='center',
                    bbox=dict(boxstyle="round,pad=0.1",
                              fc="white", ec="grey"),
                    color=tmpcolor, fontweight='bold',
                    backgroundcolor='white')


def save_boundaries_from_bounddict_ds9(bounddict, ds9_filename, numpix=100):
    """Export to ds9 region file the boundaries in bounddict.

    Parameters
    ----------
    bounddict : JSON structure
        Structure employed to store bounddict information.
    ds9_filename : str
        Output file name for the ds9 region file.
    numpix : int
        Number of points in which the X-range interval is subdivided
        in order to save each boundary as a connected set of line
        segments.

    """

    ds9_file = open(ds9_filename, 'w')

    ds9_file.write('# Region file format: DS9 version 4.1\n')
    ds9_file.write('global color=green dashlist=2 4 width=2 '
                   'font="helvetica 10 normal roman" select=1 '
                   'highlite=1 dash=1 fixed=0 edit=1 '
                   'move=1 delete=1 include=1 source=1\n')
    ds9_file.write('physical\n')

    uuid = bounddict['uuid']
    spfilter = bounddict['tags']['filter']
    grism = bounddict['tags']['grism']

    ds9_file.write('#\n# uuid (boundict file): {0}\n'.format(uuid))
    ds9_file.write('# filter..............: {0}\n'.format(spfilter))
    ds9_file.write('# grism...............: {0}\n'.format(grism))

    colorbox = ['green', 'green']
    for islitlet in range(1, EMIR_NBARS + 1):
        tmp_slitlet = 'slitlet' + str(islitlet).zfill(2)
        if tmp_slitlet in bounddict['contents'].keys():
            ds9_file.write('#\n# islitlet: {0}\n'.format(tmp_slitlet))
            read_dateobs = bounddict['contents'][tmp_slitlet].keys()
            read_dateobs.sort()
            for tmp_dateobs in read_dateobs:
                ds9_file.write('#\n# date-obs: {0}\n'.format(tmp_dateobs))
                tmp_dict = bounddict['contents'][tmp_slitlet][tmp_dateobs]
                # lower boundary
                pol_lower_measured = np.polynomial.Polynomial(
                    tmp_dict['boundary_coef_lower']
                )
                xmin_lower = tmp_dict['boundary_xmin_lower']
                xmax_lower = tmp_dict['boundary_xmax_lower']
                xdum = np.linspace(xmin_lower, xmax_lower, num=numpix)
                ydum = pol_lower_measured(xdum)
                for i in range(len(xdum) - 1):
                    ds9_file.write(
                        'line {0} {1} {2} {3}'.format(xdum[i], ydum[i],
                                                      xdum[i + 1], ydum[i + 1])
                    )
                    ds9_file.write(
                        ' # color={0}\n'.format(colorbox[islitlet % 2]))
                # upper boundary
                pol_upper_measured = np.polynomial.Polynomial(
                    tmp_dict['boundary_coef_upper']
                )
                xmin_upper = tmp_dict['boundary_xmin_upper']
                xmax_upper = tmp_dict['boundary_xmax_upper']
                xdum = np.linspace(xmin_upper, xmax_upper, num=numpix)
                ydum = pol_upper_measured(xdum)
                for i in range(len(xdum) - 1):
                    ds9_file.write(
                        'line {0} {1} {2} {3}'.format(xdum[i], ydum[i],
                                                      xdum[i + 1], ydum[i + 1])
                    )
                    ds9_file.write(
                        ' # color={0}\n'.format(colorbox[islitlet % 2]))
                # slitlet label
                xlabel = xmax_lower + xmax_upper + xmin_lower + xmin_upper
                xlabel /= 4
                yc_lower = pol_lower_measured(xlabel)
                yc_upper = pol_upper_measured(xlabel)
                ds9_file.write('text {0} {1} {{{2}}} # color={3} '
                               'font="helvetica 10 bold '
                               'roman"\n'.format(xlabel,
                                                 (yc_lower + yc_upper) / 2,
                                                 islitlet,
                                                 colorbox[islitlet % 2]))

    ds9_file.close()


def save_boundaries_from_params_ds9(params, parmodel,
                                    list_islitlet_lower,
                                    list_islitlet_upper,
                                    list_csu_bar_slit_center,
                                    uuid, grism, spfilter,
                                    ds9_filename, numpix=100):
    """Export to ds9 region file the boundaries parametrised with params.

    Parameters
    ----------
    params : :class:`~lmfit.parameter.Parameters`
        Parameters to be employed in the prediction of the distorted
        boundaries.
    parmodel : str
        Model to be assumed. Allowed values are 'longslit' and
        'multislit'.
    list_islitlet_lower : list of integers
        Slitlet numbers corresponding to the lower boundary of pseudo
        longslits.
    list_islitlet_upper : list of integers
        Slitlet numbers corresponding to the upper bounday of pseudo
        longslits.
    list_csu_bar_slit_center : list of floats
        CSU bar slit centers of the pseudo longslits.
    uuid: int
        UUID corresponding to the bounddict file that has been employed
        to fit the parameters 'params'.
    grism : str
        Employed grism.
    spfilter : str
        Employed filter.
    ds9_filename : str
        Output file name for the ds9 region file.
    numpix : int
        Number of points in which the X-range interval is subdivided
        in order to save each boundary as a connected set of line
        segments.

    """

    ds9_file = open(ds9_filename, 'w')

    ds9_file.write('# Region file format: DS9 version 4.1\n')
    ds9_file.write('global color=green dashlist=2 4 width=2 '
                   'font="helvetica 10 normal roman" select=1 '
                   'highlite=1 dash=1 fixed=0 edit=1 '
                   'move=1 delete=1 include=1 source=1\n')
    ds9_file.write('physical\n#\n')

    ds9_file.write('#\n# uuid (boundict file): {0}\n'.format(uuid))
    ds9_file.write('# filter..............: {0}\n'.format(spfilter))
    ds9_file.write('# grism...............: {0}\n'.format(grism))

    if parmodel == "longslit":
        for dumpar in EXPECTED_PARAMETER_LIST:
            parvalue = params[dumpar].value
            ds9_file.write('# {0}: {1}\n'.format(dumpar, parvalue))
    else:
        for dumpar in EXPECTED_PARAMETER_LIST_EXTENDED:
            parvalue = params[dumpar].value
            ds9_file.write('# {0}: {1}\n'.format(dumpar, parvalue))

    for islitlet_lower, islitlet_upper, csu_bar_slit_center in \
            zip(list_islitlet_lower,
                list_islitlet_upper,
                list_csu_bar_slit_center):
        if islitlet_lower % 2 == 0:
            colorbox = '#ff77ff'
        else:
            colorbox = '#4444ff'

        ds9_file.write(
            '#\n# islitlet_lower.....: {0}\n'.format(islitlet_lower)
        )
        ds9_file.write(
            '# islitlet_upper.....: {0}\n'.format(islitlet_upper)
        )
        ds9_file.write(
            '# csu_bar_slit_center: {0}\n'.format(csu_bar_slit_center)
        )
        pol_lower_expected = expected_distorted_boundaries(
            islitlet_lower, csu_bar_slit_center,
            [0], params, parmodel, numpts=101, deg=5, debugplot=0
        )[0].poly_funct
        pol_upper_expected = expected_distorted_boundaries(
            islitlet_upper, csu_bar_slit_center,
            [1], params, parmodel, numpts=101, deg=5, debugplot=0
        )[0].poly_funct
        xdum = np.linspace(1, NAXIS1_EMIR, num=numpix)
        ydum = pol_lower_expected(xdum)
        for i in range(len(xdum)-1):
            ds9_file.write(
                'line {0} {1} {2} {3}'.format(xdum[i], ydum[i],
                                              xdum[i+1], ydum[i+1])
            )
            ds9_file.write(' # color={0}\n'.format(colorbox))
        ydum = pol_upper_expected(xdum)
        for i in range(len(xdum)-1):
            ds9_file.write(
                'line {0} {1} {2} {3}'.format(xdum[i], ydum[i],
                                              xdum[i+1], ydum[i+1])
            )
            ds9_file.write(' # color={0}\n'.format(colorbox))
        # slitlet label
        yc_lower = pol_lower_expected(NAXIS1_EMIR / 2 + 0.5)
        yc_upper = pol_upper_expected(NAXIS1_EMIR / 2 + 0.5)
        if islitlet_lower == islitlet_upper:
            ds9_file.write('text {0} {1} {{{2}}} # color={3} '
                           'font="helvetica 10 bold '
                           'roman"\n'.format(NAXIS1_EMIR / 2 + 0.5,
                                             (yc_lower + yc_upper) / 2,
                                             islitlet_lower,
                                             colorbox))
        else:
            ds9_file.write('text {0} {1} {{{2}-{3}}} # color={4} '
                           'font="helvetica 10 bold '
                           'roman"\n'.format(NAXIS1_EMIR / 2 + 0.5,
                                            (yc_lower + yc_upper) / 2,
                                             islitlet_lower, islitlet_upper,
                                             colorbox))

    ds9_file.close()


def bound_params_from_dict(bound_param_dict):
    """Define `~lmfit.parameter.Parameters` object from dictionary.

    Parameters
    ----------
    bound_param_dict : dictionary
        Dictionary containing the JSON contents of a boundary
        parameter file.

    Returns
    -------
    params : :class:`~lmfit.parameter.Parameters`
        Parameters object.

    """

    params = Parameters()
    for mainpar in EXPECTED_PARAMETER_LIST:
        if mainpar not in bound_param_dict['contents'].keys():
            raise ValueError('Parameter ' + mainpar + ' not found!')
        if bound_param_dict['meta-info']['parmodel'] ==  "longslit":
            dumdict = bound_param_dict['contents'][mainpar]
            params.add(mainpar, value=dumdict["value"],
                       vary=dumdict["vary"])
        else:
            for subpar in ['a0s', 'a1s', 'a2s']:
                if subpar not in bound_param_dict['contents'][mainpar].keys():
                    raise ValueError('Subparameter ' + subpar + ' not found' +
                                     ' under parameter ' + mainpar)
                cpar = mainpar + '_' + subpar
                dumdict = bound_param_dict['contents'][mainpar][subpar]
                params.add(cpar, value=dumdict["value"],
                           vary=dumdict["vary"])
    return params


def main(args=None):
    parser = argparse.ArgumentParser(prog='fit_boundaries')
    parser.add_argument("bounddict",
                        help="Input JSON boundary file with fits to "
                             "continuum-lamp exposures",
                        type=argparse.FileType('r'))
    parser.add_argument("init_bound_param",
                        help="Input JSON with initial boundary parameters",
                        type=argparse.FileType('r'))
    parser.add_argument("--parmodel",
                        help="Parameter model: multislit (default) or "
                             "longslit",
                        default="multislit",
                        choices=("multislit", "longslit"))
    parser.add_argument("--fitted_bound_param",
                        help="Output JSON with fitted boundary parameters",
                        type=argparse.FileType('w'))
    parser.add_argument("--numresolution",
                        help="Number of points/boundary (default=101)",
                        type=int, default=101)
    parser.add_argument("--background_image",
                        help="Optional FITS image to display as background "
                             "image",
                        type=argparse.FileType('r'))
    parser.add_argument("--pickle_input",
                        help="Use previous pickle file instead of carrying "
                             "out the minimisation procces",
                        type=argparse.FileType('rb'))
    parser.add_argument("--debugplot",
                        help="Integer indicating plotting/debugging" +
                             " (default=0)",
                        type=int, default=0,
                        choices=DEBUGPLOT_CODES)
    parser.add_argument("--echo",
                        help="Display full command line",
                        action="store_true")
    args = parser.parse_args(args)

    if args.echo:
        print('\033[1m\033[31mExecuting: ' + ' '.join(sys.argv) + '\033[0m\n')

    if args.background_image is not None and args.debugplot % 10 == 0:
        raise ValueError("--background_image requires --debugplot value "
                         "compatible with plotting")

    # read bounddict file and check its contents
    bounddict = json.loads(open(args.bounddict.name).read())
    integrity_check(bounddict)
    save_boundaries_from_bounddict_ds9(bounddict, 'ds9_bounddict.reg')
    # save lists with individual slitlet number and csu_bar_slit_center value,
    # needed to save ds9 region file and plotting
    list_islitlet = []
    list_csu_bar_slit_center = []
    read_slitlets = bounddict['contents'].keys()
    read_slitlets.sort()
    for tmp_slitlet in read_slitlets:
        islitlet = int(tmp_slitlet[7:])
        list_islitlet.append(islitlet)
        read_dateobs = bounddict['contents'][tmp_slitlet].keys()
        read_dateobs.sort()
        for tmp_dateobs in read_dateobs:
            tmp_dict = bounddict['contents'][tmp_slitlet][tmp_dateobs]
            csu_bar_slit_center = tmp_dict['csu_bar_slit_center']
            list_csu_bar_slit_center.append(csu_bar_slit_center)

    uuid = bounddict['uuid']
    grism = bounddict['tags']['grism']
    spfilter = bounddict['tags']['filter']

    # read init_bound_param file
    init_bound_param = json.loads(open(args.init_bound_param.name).read())
    # check that grism and filter match
    grism_ = init_bound_param['tags']['grism']
    spfilter_ = init_bound_param['tags']['filter']
    if grism != grism_:
        raise ValueError("grism mismatch")
    if spfilter != spfilter_:
        raise ValueError("filter mismatch")
    islitlet_min = init_bound_param['tags']['islitlet_min']
    islitlet_max = init_bound_param['tags']['islitlet_max']
    # check that parameter model is correct
    parmodel_  = init_bound_param['meta-info']['parmodel']
    if args.parmodel != parmodel_:
        raise ValueError("Unexpected parmodel: ", parmodel_,
                         " in file ", args.init_bound_param.name)

    # establish initial parameters from init_bound_param dictionary
    params = bound_params_from_dict(init_bound_param)

    print('-' * 79)
    print('* INITIAL PARAMETERS')
    params.pretty_print()
    pause_debugplot(args.debugplot)

    # carry out minimisation process (or read previous pickle file)
    if args.pickle_input is not None:
        result = pickle.load(args.pickle_input)
    else:
        fitter = Minimizer(
            fun_residuals, params,
            fcn_args=(args.parmodel, bounddict, args.numresolution,
                      islitlet_min, islitlet_max, args.debugplot)
        )
        result = fitter.scalar_minimize(method='Nelder-Mead', tol=1E-7)
        pickle.dump(result, open('dum.pickle', 'wb'))

    global_residual = fun_residuals(result.params, args.parmodel, bounddict,
                                    args.numresolution,
                                    islitlet_min, islitlet_max,
                                    args.debugplot)
    print('\n>>> global residual', global_residual)
    result.params.pretty_print()
    print('>>> Number of function evaluations:', result.nfev)
    pause_debugplot(args.debugplot)
    report_fit(result.params, min_correl=0.5)
    pause_debugplot(args.debugplot)

    # export resulting boundaries to ds9 region file
    save_boundaries_from_params_ds9(result.params, args.parmodel,
                                    list_islitlet,
                                    list_islitlet,
                                    list_csu_bar_slit_center,
                                    uuid, grism, spfilter,
                                    'ds9_fittedpar.reg')

    if args.fitted_bound_param is not None:
        fitted_bound_param = deepcopy(init_bound_param)
        fitted_bound_param['meta-info']['creation_date'] = \
            datetime.now().isoformat()
        fitted_bound_param['meta-info']['description'] \
            = "fitted boundary parameters"
        fitted_bound_param['meta-info']['function_evaluations'] = result.nfev
        fitted_bound_param['meta-info']['global_residual'] = global_residual
        fitted_bound_param['meta-info']['origin'] = {}
        fitted_bound_param['meta-info']['origin']['bounddict_uuid'] = \
            bounddict['uuid']
        fitted_bound_param['meta-info']['origin']['init_bound_param_uuid'] = \
            init_bound_param['uuid']
        fitted_bound_param['uuid'] = str(uuid4())
        for mainpar in EXPECTED_PARAMETER_LIST:
            if mainpar not in init_bound_param['contents'].keys():
                raise ValueError('Parameter ' + mainpar +
                                 ' not found in ' + args.init_bound_param.name)
            if args.parmodel == "longslit":
                fitted_bound_param['contents'][mainpar]['value'] = \
                    result.params[mainpar].value
                fitted_bound_param['contents'][mainpar]['initial'] = \
                    params[mainpar].value
                # compute median csu_bar_slit_center (only in longslit mode)
                dumlist = []
                for islitlet, csu_bar_slit_center in \
                        zip(list_islitlet, list_csu_bar_slit_center):
                    if islitlet_min <= islitlet <= islitlet_max:
                        dumlist.append(csu_bar_slit_center)
                median_csu_bar_slit_center = np.median(np.array(dumlist))
                fitted_bound_param['meta-info']['median_csu_bar_slit_center']\
                    = median_csu_bar_slit_center
            else:
                for subpar in ['a0s', 'a1s', 'a2s']:
                    if subpar not in \
                            init_bound_param['contents'][mainpar].keys():
                        raise ValueError(
                            'Subparameter ' + subpar + ' not found in ' +
                            args.init_bound_param.name + ' under parameter ' +
                            mainpar
                        )
                    cpar = mainpar + '_' + subpar
                    fitted_bound_param['contents'][mainpar][subpar]['value']\
                        = result.params[cpar].value
                    fitted_bound_param['contents'][mainpar][subpar]['initial']\
                        = params[cpar].value
        with open(args.fitted_bound_param.name, 'w') as fstream:
            json.dump(fitted_bound_param, fstream, indent=2, sort_keys=True)

    if args.debugplot % 10 != 0:
        fig = plt.figure()
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(0, 0, 640, 480)
        if args.background_image is not None:
            # read input FITS file
            hdulist = fits.open(args.background_image.name)
            image2d = hdulist[0].data
            hdulist.close()
            if image2d.shape != (NAXIS2_EMIR, NAXIS1_EMIR):
                raise ValueError("Unexpected error with NAXIS1, NAXIS2")
            ax = ximshow(image2d=image2d,
                         title=args.background_image.name,
                         image_bbox=(1, NAXIS1_EMIR, 1, NAXIS2_EMIR),
                         show=False)
        else:
            ax = fig.add_subplot(111)
            ax.set_xlim([-0.5, NAXIS1_EMIR + 0.5])
            ax.set_ylim([-0.5, NAXIS2_EMIR + 0.5])
            ax.set_xlabel('X axis (from 1 to NAXIS1)')
            ax.set_xlabel('Y axis (from 1 to NAXIS2)')
            ax.set_title(args.bounddict.name)
        # boundaries from bounddict
        overplot_boundaries_from_bounddict(bounddict, ['r', 'b'])
        # expected boundaries
        overplot_boundaries_from_params(ax, result.params, args.parmodel,
                                        list_islitlet,
                                        list_islitlet,
                                        list_csu_bar_slit_center,
                                        ['m', 'c'], linetype='--')
        pause_debugplot(debugplot=args.debugplot, pltshow=True,
                        tight_layout=True)


if __name__ == "__main__":

    main()
