from __future__ import division
from __future__ import print_function

import argparse
from astropy.io import fits
from copy import deepcopy
from datetime import datetime
import json
from lmfit import minimize, Parameters, minimizer
import matplotlib.pyplot as plt
import numpy as np
from uuid import uuid4

from numina.array.display.polfit_residuals import polfit_residuals
from numina.array.display.pause_debugplot import pause_debugplot
from numina.array.display.ximshow import ximshow
from emirdrp.core import EMIR_NBARS

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
                          "z_info1",
                          "z_info2"]
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


def expected_distorted_boundaries(islitlet, csu_bar_slit_center,
                                  border, params, parmodel,
                                  numpts, deg, debugplot=0):
    """Return polynomial coefficients of expected distorted boundaries.

    """

    if parmodel == "longslit":
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
        raise ValueError("parmodel=" + str(parmodel) +
                         " is not implemented")

    theta0 = theta0_origin/10000 + theta0_slope/100000 * islitlet

    if border not in ['lower', 'upper', 'both']:
        raise ValueError('Unexpected border:', border)

    xp = np.linspace(1, NAXIS1_EMIR, numpts)
    slit_dist = (slit_height * 10) + slit_gap

    # undistorted (constant) y-coordinate of the lower and upper boundaries
    ybottom = y_baseline * 100 + (islitlet - 1) * slit_dist
    ytop = ybottom + (slit_height * 10)

    # avoid PyCharm warning (variables might by referenced before assignment)
    poly_lower = poly_upper = None  # avoid PyCharm warning

    if border in ['lower', 'both']:
        # undistorted boundary
        yp_bottom = np.ones(numpts) * ybottom
        # distorted boundary
        xdist, ydist = exvp(xp, yp_bottom, x0=x0, y0=y0,
                            c2=c2, c4=c4, theta0=theta0, ff=ff)
        poly_lower, dum = polfit_residuals(xdist, ydist, deg,
                                           debugplot=debugplot)
    if border in ['upper', 'both']:
        # undistorted boundary
        yp_top = np.ones(numpts) * ytop
        # distorted boundary
        xdist, ydist = exvp(xp, yp_top, x0=x0, y0=y0,
                            c2=c2, c4=c4, theta0=theta0, ff=ff)
        poly_upper, dum = polfit_residuals(xdist, ydist, deg,
                                           debugplot=debugplot)

    if border == 'lower':
        return poly_lower
    elif border == 'upper':
        return poly_upper
    else:
        return poly_lower, poly_upper


def fun_residuals(params, parmodel, bounddict, numresolution,
                  islitmin, islitmax):
    residuals = 0.0
    nsummed = 0

    read_slitlets = bounddict['contents'].keys()
    #read_slitlets.sort()  # this is not really necessary
    for tmp_slitlet in read_slitlets:
        islitlet = int(tmp_slitlet[7:])
        if islitmin <= islitlet <= islitmax:
            read_dateobs = bounddict['contents'][tmp_slitlet].keys()
            #read_dateobs.sort()  # this is not really necessary
            for tmp_dateobs in read_dateobs:
                tmp_dict = bounddict['contents'][tmp_slitlet][tmp_dateobs]
                csu_bar_slit_center = tmp_dict['csu_bar_slit_center']
                # expected boundaries using provided parameters
                poly_lower_expected, poly_upper_expected = \
                    expected_distorted_boundaries(
                        islitlet, csu_bar_slit_center,
                        'both', params, parmodel,
                        numpts=numresolution, deg=5, debugplot=0
                    )
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
                residuals += np.sum(poly_diff(xdum_lower)**2)
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
                residuals += np.sum(poly_diff(xdum_upper)**2)
                nsummed += numresolution

    if nsummed > 0:
        residuals = np.sqrt(residuals/nsummed)
    print('>>> residuals:', residuals)
    params.pretty_print()
    return residuals


def overplot_boundaries_from_bounddict(bounddict, micolors, linetype='-'):
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

    for islitlet_lower, islitlet_upper, csu_bar_slit_center in \
            zip(list_islitlet_lower,
                list_islitlet_upper,
                list_csu_bar_slit_center):
        tmpcolor = micolors[islitlet_lower % 2]
        pol_lower_expected = expected_distorted_boundaries(
            islitlet_lower, csu_bar_slit_center,
            'lower', params, parmodel, numpts=101, deg=5, debugplot=0)
        pol_upper_expected = expected_distorted_boundaries(
            islitlet_upper, csu_bar_slit_center,
            'upper', params, parmodel, numpts=101, deg=5, debugplot=0)
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

    ds9_file.write('#\n# uuid.......: {0}\n'.format(uuid))
    ds9_file.write('# filter.....: {0}\n'.format(spfilter))
    ds9_file.write('# grism......: {0}\n'.format(grism))

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
                                    ds9_filename, numpix=100):
    ds9_file = open(ds9_filename, 'w')

    ds9_file.write('# Region file format: DS9 version 4.1\n')
    ds9_file.write('global color=green dashlist=2 4 width=2 '
                   'font="helvetica 10 normal roman" select=1 '
                   'highlite=1 dash=1 fixed=0 edit=1 '
                   'move=1 delete=1 include=1 source=1\n')
    ds9_file.write('physical\n#\n')

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
            'lower', params, parmodel, numpts=101, deg=5, debugplot=0
        )
        pol_upper_expected = expected_distorted_boundaries(
            islitlet_upper, csu_bar_slit_center,
            'upper', params, parmodel, numpts=101, deg=5, debugplot=0
        )
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


def main(args=None):
    parser = argparse.ArgumentParser(prog='fit_boundaries')
    parser.add_argument("bounddict",
                        help="Input JSON boundary file with fits to "
                             "continuum-lamp exposures",
                        type=argparse.FileType('r'))
    parser.add_argument("initparam",
                        help="Input JSON with initial boundary parameters",
                        type=argparse.FileType('r'))
    parser.add_argument("--parmodel",
                        help="Parameter model: multislit (default) or "
                             "longslit",
                        default="multislit",
                        choices=("multislit", "longslit"))
    parser.add_argument("--fittedparam",
                        help="Output JSON with fitted boundary parameters",
                        type=argparse.FileType('w'))
    parser.add_argument("--numresolution",
                        help="Number of points/boundary (default=101)",
                        type=int, default=101)
    parser.add_argument("--background_image",
                        help="Optional FITS image to display as background "
                             "image",
                        type=argparse.FileType('r'))
    parser.add_argument("--debugplot",
                        help="Integer indicating plotting/debugging" +
                             " (default=0)",
                        type=int, default=0,
                        choices=DEBUGPLOT_CODES)
    args = parser.parse_args(args)

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

    grism = bounddict['tags']['grism']
    spfilter = bounddict['tags']['filter']

    # read initparam file
    initparam = json.loads(open(args.initparam.name).read())
    # check that grism and filter match
    grism_ = initparam['tags']['grism']
    spfilter_ = initparam['tags']['filter']
    if grism != grism_:
        raise ValueError("grism mismatch")
    if spfilter != spfilter_:
        raise ValueError("filter mismatch")
    islitlet_min = initparam['tags']['islitlet_min']
    islitlet_max = initparam['tags']['islitlet_max']

    params = Parameters()
    for mainpar in EXPECTED_PARAMETER_LIST:
        if mainpar not in initparam['contents'].keys():
            raise ValueError('Parameter ' + mainpar + ' not found in ' +
                             args.initparam.name)
        if args.parmodel == "longslit":
            dumdict = initparam['contents'][mainpar]
            params.add(mainpar, value=dumdict["value"],
                       vary=dumdict["vary"])
        else:
            for subpar in ['a0s', 'a1s', 'a2s']:
                if subpar not in initparam['contents'][mainpar].keys():
                    raise ValueError('Subparameter ' + subpar +
                                     ' not found in ' +
                                     args.initparam.name +
                                     ' under parameter ' + mainpar)
                cpar = mainpar + '_' + subpar
                dumdict = initparam['contents'][mainpar][subpar]
                params.add(cpar, value=dumdict["value"],
                           vary=dumdict["vary"])

    print('-' * 79)
    print('* INITIAL PARAMETERS')
    params.pretty_print()
    print('-' * 79)

    result = minimize(fun_residuals, params, method='nelder',
                      args=(args.parmodel, bounddict, args.numresolution,
                            islitlet_min, islitlet_max))
    global_residual = fun_residuals(result.params, args.parmodel, bounddict,
                             args.numresolution, islitlet_min, islitlet_max)
    print('\n>>> global residual', global_residual)
    result.params.pretty_print()

    # export resulting boundaries to ds9 region file
    save_boundaries_from_params_ds9(result.params, args.parmodel,
                                    list_islitlet,
                                    list_islitlet,
                                    list_csu_bar_slit_center,
                                    'ds9_fittedpar.reg')

    if args.fittedparam is not None:
        fittedparam = deepcopy(initparam)
        fittedparam['meta-info']['creation_date'] = datetime.now().isoformat()
        fittedparam['meta-info']['description'] \
            = "fitted boundary parameters"
        fittedparam['meta-info']['global_residual'] = global_residual
        fittedparam['meta-info']['uuid_bounddict'] = bounddict['uuid']
        fittedparam['meta-info']['uuid_initparam'] = initparam['uuid']
        fittedparam['meta-info']['parmodel'] = args.parmodel
        fittedparam['uuid'] = str(uuid4())
        if args.parmodel == "longslit":
            for mainpar in EXPECTED_PARAMETER_LIST:
                parvalue = result.params[mainpar].value
                fittedparam['contents'][mainpar]['value'] = parvalue
            # compute median csu_bar_slit_center
            dumlist = []
            for islitlet, csu_bar_slit_center in \
                    zip(list_islitlet, list_csu_bar_slit_center):
                if islitlet_min <= islitlet <= islitlet_max:
                    dumlist.append(csu_bar_slit_center)
            median_csu_bar_slit_center = np.median(np.array(dumlist))
            fittedparam['meta-info']['median_csu_bar_slit_center'] = \
                median_csu_bar_slit_center
        else:
            raise ValueError("parmodel " + args.parmodel +
                             " is not implemented")
        with open(args.fittedparam.name, 'w') as fstream:
            json.dump(fittedparam, fstream, indent=2, sort_keys=True)

    if args.debugplot % 10 != 0:
        fig = plt.figure()
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
            if args.bounddict is not None:
                ax.set_title(args.bounddict.name)
            else:
                ax.set_title(args.pickle_input.name)
        # boundaries from bounddict
        if args.bounddict is not None:
            overplot_boundaries_from_bounddict(bounddict, ['r', 'b'])
        # expected boundaries
        overplot_boundaries_from_params(ax, result.params, args.parmodel,
                                        list_islitlet,
                                        list_islitlet,
                                        list_csu_bar_slit_center,
                                        ['m', 'c'], linetype='--')
        pause_debugplot(debugplot=args.debugplot, pltshow=True)


if __name__ == "__main__":

    main()
