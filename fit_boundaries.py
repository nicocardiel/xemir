from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
from lmfit import minimize, Parameters

from numina.array.display.polfit_residuals import polfit_residuals
from numina.array.display.pause_debugplot import pause_debugplot
from numina.array.display.ximplot import ximplot
from emirdrp.core import EMIR_NBARS

from emir_definitions import NAXIS1_EMIR
from emir_definitions import NAXIS2_EMIR
from numina.array.display.pause_debugplot import DEBUGPLOT_CODES


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
    if bounddict['meta-info']['description'] != 'slitlet boundaries':
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


def exvp_scalar(x, y, x0, y0, c1, c2):
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
    c1 : float
        Coefficient corresponding to the term r**2 in distortion
        equation, in units of 1E4.
    c2 : float
        Coefficient corresponding to the term r**4 in distortion
        equation, in units of 1E9

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
    rdist = (1 + c1 * 1.0E4 * r_rad**2 + c2 * 1.0E9 * r_rad**4)
    # angle measured from the Y axis towards the X axis
    theta = np.arctan((x - x0*1000)/(y - y0*1000))
    if y < y0*1000:
        theta = theta - np.pi
    # distorted coordinates
    xdist = (rdist * r_pix * np.sin(theta)) + x0*1000
    ydist = (rdist * r_pix * np.cos(theta)) + y0*1000
    return xdist, ydist


def exvp(x, y, x0, y0, c1, c2):
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
    c1 : float
        Coefficient corresponding to the term r**2 in distortion
        equation.
    c2 : float
        Coefficient corresponding to the term r**4 in distortion
        equation.

    Returns
    -------
    xdist, ydist : tuple of floats (or two arrays of floats)
        Distorted coordinates.

    """

    if all([np.isscalar(x), np.isscalar(y)]):
        xdist, ydist = exvp_scalar(x, y, x0=x0, y0=y0, c1=c1, c2=c2)
        return xdist, ydist
    elif any([np.isscalar(x), np.isscalar(y)]):
        raise ValueError("invalid mixture of scalars and arrays")
    else:
        xdist = []
        ydist = []
        for x_, y_ in zip(x, y):
            xdist_, ydist_ = exvp_scalar(x_, y_, x0=x0, y0=y0, c1=c1, c2=c2)
            xdist.append(xdist_)
            ydist.append(ydist_)
        return np.array(xdist), np.array(ydist)


def expected_distorted_boundaries(islitlet, border, slit_height, slit_gap,
                                  y_baseline, x0, y0, c1, c2,
                                  numpts, deg,
                                  debugplot=0):
    """Return polynomial coefficients of expected distorted boundaries.

    """

    if border not in ['lower', 'upper', 'both']:
        raise ValueError('Unexpected border:', border)

    xp = np.linspace(1, NAXIS1_EMIR, numpts)
    slit_dist = slit_height + slit_gap

    # y-coordinates at x=1024.5
    ybottom = y_baseline + (islitlet - 1) * slit_dist
    ytop = ybottom + slit_height

    # avoid PyCharm warning (variables might by referenced before assignment)
    poly_lower = poly_upper = None  # avoid PyCharm warning

    if border in ['lower', 'both']:
        # undistorted boundary
        yp_bottom = np.ones(numpts) * ybottom
        # distorted boundary
        xdist, ydist = exvp(xp, yp_bottom, x0=x0, y0=y0, c1=c1, c2=c2)
        poly_lower, dum = polfit_residuals(xdist, ydist, deg,
                                           debugplot=debugplot)
    if border in ['upper', 'both']:
        # undistorted boundary
        yp_top = np.ones(numpts) * ytop
        # distorted boundary
        xdist, ydist = exvp(xp, yp_top, x0=x0, y0=y0, c1=c1, c2=c2)
        poly_upper, dum = polfit_residuals(xdist, ydist, deg,
                                           debugplot=debugplot)

    if border == 'lower':
        return poly_lower
    elif border == 'upper':
        return poly_upper
    else:
        return poly_lower, poly_upper


def residuals_polynomials(params, xdata, islitlet, border, poly_measured):
    slit_height = params['slit_height']
    slit_gap = params['slit_gap']
    y_baseline = params['y_baseline']
    x0 = params['x0']
    y0 = params['y0']
    c1 = params['c1']
    c2 = params['c2']
    # expected boundary using provided parameters
    poly_expected = expected_distorted_boundaries(
        islitlet, border, slit_height, slit_gap, y_baseline,
        x0, y0, c1, c2, numpts=101, deg=7, debugplot=0)
    # distance between measured and expected polynomials
    poly_diff = poly_expected - poly_measured
    residuals = poly_diff(xdata)
    return residuals


def main(args=None):
    parser = argparse.ArgumentParser(prog='fit_boundaries')
    parser.add_argument("boundict",
                        help="JSON boundary file",
                        type=argparse.FileType('r'))
    parser.add_argument("--debugplot",
                        help="Integer indicating plotting/debugging" +
                             " (default=0)",
                        type=int, default=0,
                        choices=DEBUGPLOT_CODES)
    args = parser.parse_args(args)

    bounddict = json.loads(open(args.boundict.name).read())

    integrity_check(bounddict)

    slit_height = 33.5  # 33.437  # 33.441  # 33.453  # 33.462  # 33.475
    slit_gap = 4.0  # 4.015  # 4.006  # 3.997  # 3.988  # 3.979
    # 7(H-H), 3(K-Ksp), -85(LR-HK), -87(LR-YJ)
    y_baseline = 2.0  # 2.347  # 2.295  # 2.244  # 2.195  # 2.147
    if bounddict['tags']['grism'] == 'J' and \
        bounddict['tags']['filter'] == 'J':
        x0 = 1.0245  # in units of 1E3
        y0 = 1.0245  # in units of 1E3
        c1 = 1.46067  # in units of 1E4
        c2 = 1.7397161151  # in units of 1E9
    else:
        raise ValueError("Distortion parameters are not available for this "
                         "combination of grism and filter")
    list_slit_height_fitted = []
    list_slit_gap_fitted = []
    list_y_baseline_fitted = []
    list_x0_fitted = []
    list_y0_fitted = []
    list_c1_fitted = []
    list_c2_fitted = []

    if args.debugplot % 10 != 0:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim([-0.5, NAXIS1_EMIR+0.5])
        ax.set_ylim([-0.5, NAXIS2_EMIR+0.5])
        ax.set_xlabel('X axis (from 1 to NAXIS1)')
        ax.set_xlabel('Y axis (from 1 to NAXIS2)')
        ax.set_title(args.boundict.name)
        micolors = ['r', 'b']
        micolors_exp = ['m', 'c']

    read_slitlets = bounddict['contents'].keys()
    read_slitlets.sort()
    for tmp_slitlet in read_slitlets:
        islitlet = int(tmp_slitlet[7:])
        print()
        print(79 * '-')
        print('>>> Reading slitlet ', islitlet)
        # expected boundaries
        pol_lower_expected, pol_upper_expected = expected_distorted_boundaries(
            islitlet, 'both', slit_height, slit_gap, y_baseline,
            x0, y0, c1, c2, numpts=101, deg=7, debugplot=0)
        read_dateobs = bounddict['contents'][tmp_slitlet].keys()
        read_dateobs.sort()
        for tmp_dateobs in read_dateobs:
            print('...', tmp_dateobs)
            tmp_dict = bounddict['contents'][tmp_slitlet][tmp_dateobs]
            # lower boundary
            pol_lower_measured = np.polynomial.Polynomial(
                tmp_dict['boundary_coef_lower']
            )
            xmin_lower_bound = tmp_dict['boundary_xmin_lower']
            xmax_lower_bound = tmp_dict['boundary_xmax_lower']
            xdum_lower = np.arange(xmin_lower_bound, xmax_lower_bound+1)
            ydum_lower_measured = pol_lower_measured(xdum_lower)
            ydum_lower_expected = pol_lower_expected(xdum_lower)
            # upper boundary
            pol_upper_measured = np.polynomial.Polynomial(
                tmp_dict['boundary_coef_upper']
            )
            xmin_upper_bound = tmp_dict['boundary_xmin_upper']
            xmax_upper_bound = tmp_dict['boundary_xmax_upper']
            xdum_upper = np.arange(xmin_upper_bound, xmax_upper_bound + 1)
            ydum_upper_measured = pol_upper_measured(xdum_upper)
            ydum_upper_expected = pol_upper_expected(xdum_upper)
            # compute optimal parameters for current islitlet
            params = Parameters()
            params.add('slit_height', value=slit_height, vary=True,
                       min=30, max=40)
            params.add('slit_gap', value=slit_gap, vary=True,
                       min=2, max=6)
            params.add('y_baseline', value=y_baseline, vary=True,
                       min=-3, max=7)
            params.add('x0', value=x0, vary=False)
            params.add('y0', value=y0, vary=False)
            params.add('c1', value=c1, vary=False)
            params.add('c2', value=c2, vary=True,
                       min=2, max=-6)
            print(39 * '- ' + '-')
            result = minimize(residuals_polynomials, params,
                              method='nelder',
                              args=(xdum_lower[::5], islitlet, 'lower',
                                    pol_lower_measured))
            print('... params (lower):')
            result.params.pretty_print()
            slit_height_fitted = result.params['slit_height'].value
            slit_gap_fitted = result.params['slit_gap'].value
            y_baseline_fitted = result.params['y_baseline'].value
            x0_fitted = result.params['x0'].value
            y0_fitted = result.params['y0'].value
            c1_fitted = result.params['c1'].value
            c2_fitted = result.params['c2'].value
            list_slit_height_fitted.append(slit_height_fitted)
            list_slit_gap_fitted.append(slit_gap_fitted)
            list_y_baseline_fitted.append(y_baseline_fitted)
            list_x0_fitted.append(x0_fitted)
            list_y0_fitted.append(y0_fitted)
            list_c1_fitted.append(c1_fitted)
            list_c2_fitted.append(c2_fitted)
            pol_lower_expected_fit = expected_distorted_boundaries(
                islitlet, 'lower',
                slit_height_fitted, slit_gap_fitted, y_baseline_fitted,
                x0_fitted, y0_fitted, c1_fitted, c2_fitted,
                numpts=101, deg=7, debugplot=0)
            ydum_lower_expected_fit = pol_lower_expected_fit(xdum_lower)
            result = minimize(residuals_polynomials, params,
                              method='nelder',
                              args=(xdum_upper[::5], islitlet, 'upper',
                                    pol_upper_measured))
            print('... params (upper):')
            result.params.pretty_print()
            slit_height_fitted = result.params['slit_height'].value
            slit_gap_fitted = result.params['slit_gap'].value
            y_baseline_fitted = result.params['y_baseline'].value
            x0_fitted = result.params['x0'].value
            y0_fitted = result.params['y0'].value
            c1_fitted = result.params['c1'].value
            c2_fitted = result.params['c2'].value
            list_slit_height_fitted.append(slit_height_fitted)
            list_slit_gap_fitted.append(slit_gap_fitted)
            list_y_baseline_fitted.append(y_baseline_fitted)
            list_x0_fitted.append(x0_fitted)
            list_y0_fitted.append(y0_fitted)
            list_c1_fitted.append(c1_fitted)
            list_c2_fitted.append(c2_fitted)
            pol_upper_expected_fit= expected_distorted_boundaries(
                islitlet, 'upper',
                slit_height_fitted, slit_gap_fitted, y_baseline_fitted,
                x0_fitted, y0_fitted, c1_fitted, c2_fitted,
                numpts=101, deg=7, debugplot=0)
            ydum_upper_expected_fit = pol_upper_expected_fit(xdum_upper)
            if args.debugplot % 10 != 0:
                tmpcolor = micolors[islitlet % 2]
                tmpcolor_exp = micolors_exp[islitlet % 2]
                plt.plot(xdum_lower, ydum_lower_measured, tmpcolor+'-')
                plt.plot(xdum_lower, ydum_lower_expected, tmpcolor_exp+'--')
                plt.plot(xdum_lower, ydum_lower_expected_fit, tmpcolor_exp+':')
                plt.plot(xdum_upper, ydum_upper_measured, tmpcolor + '-')
                plt.plot(xdum_upper, ydum_upper_expected, tmpcolor_exp + '--')
                plt.plot(xdum_upper, ydum_upper_expected_fit, tmpcolor_exp+':')
                yc_lower = pol_lower_measured(NAXIS1_EMIR / 2 + 0.5)
                yc_upper = pol_upper_measured(NAXIS1_EMIR / 2 + 0.5)
                ax.text(NAXIS1_EMIR / 2 + 0.5, (yc_lower + yc_upper) / 2,
                        str(islitlet), fontsize=10, va='center', ha='center',
                        bbox=dict(boxstyle="round,pad=0.1",
                                  fc="white", ec="grey"),
                        color=tmpcolor, fontweight='bold',
                        backgroundcolor='white')

    pause_debugplot(debugplot=args.debugplot, pltshow=True)

    slit_height_fitted = np.array(slit_height_fitted)
    slit_gap_fitted = np.array(slit_gap_fitted)
    y_baseline_fitted = np.array(y_baseline_fitted)
    c2_fitted = np.array(c2_fitted)

    print(78 * '-')
    print('>>> Median values: ', np.median(slit_height_fitted),
          np.median(slit_gap_fitted), np.median(y_baseline_fitted),
          np.median(c2_fitted))

    ximplot(slit_height_fitted, title='slit_height', debugplot=12)
    ximplot(slit_gap_fitted, title='slit_gap', debugplot=12)
    ximplot(y_baseline_fitted, title='y_baseline', debugplot=12)
    ximplot(c2_fitted, title='c2', debugplot=12)


if __name__ == "__main__":

    main()
