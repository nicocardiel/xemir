from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import json
from lmfit import minimize, Parameters, minimizer
import matplotlib.pyplot as plt
import numpy as np
import pickle

from numina.array.display.polfit_residuals import polfit_residuals
from numina.array.display.pause_debugplot import pause_debugplot
from numina.array.display.ximplot import ximplot
from numina.array import stats
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


def exvp_scalar(x, y, x0, y0, c2, c4, theta0):
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
    xdist = (rdist * r_pix * np.sin(theta+theta0/1000)) + x0*1000
    ydist = (rdist * r_pix * np.cos(theta+theta0/1000)) + y0*1000
    return xdist, ydist


def exvp(x, y, x0, y0, c2, c4, theta0):
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
        Additional rotation angle (radians)

    Returns
    -------
    xdist, ydist : tuple of floats (or two arrays of floats)
        Distorted coordinates.

    """

    if all([np.isscalar(x), np.isscalar(y)]):
        xdist, ydist = exvp_scalar(x, y, x0=x0, y0=y0,
                                   c2=c2, c4=c4, theta0=theta0)
        return xdist, ydist
    elif any([np.isscalar(x), np.isscalar(y)]):
        raise ValueError("invalid mixture of scalars and arrays")
    else:
        xdist = []
        ydist = []
        for x_, y_ in zip(x, y):
            xdist_, ydist_ = exvp_scalar(x_, y_, x0=x0, y0=y0,
                                         c2=c2, c4=c4, theta0=theta0)
            xdist.append(xdist_)
            ydist.append(ydist_)
        return np.array(xdist), np.array(ydist)


def expected_distorted_boundaries(islitlet, border, params,
                                  numpts, deg, debugplot=0):
    """Return polynomial coefficients of expected distorted boundaries.

    """

    slit_height = params['slit_height'].value
    slit_gap = params['slit_gap'].value
    y_baseline = params['y_baseline'].value
    x0 = params['x0'].value
    y0 = params['y0'].value
    c2 = params['c2'].value
    c4 = params['c4'].value
    theta0 = params['theta0'].value

    if border not in ['lower', 'upper', 'both']:
        raise ValueError('Unexpected border:', border)

    xp = np.linspace(1, NAXIS1_EMIR, numpts)
    slit_dist = slit_height * 10 + slit_gap

    # y-coordinates at x=1024.5
    ybottom = y_baseline + (islitlet - 1) * slit_dist
    ytop = ybottom + slit_height * 10

    # avoid PyCharm warning (variables might by referenced before assignment)
    poly_lower = poly_upper = None  # avoid PyCharm warning

    if border in ['lower', 'both']:
        # undistorted boundary
        yp_bottom = np.ones(numpts) * ybottom
        # distorted boundary
        xdist, ydist = exvp(xp, yp_bottom, x0=x0, y0=y0,
                            c2=c2, c4=c4, theta0=theta0)
        poly_lower, dum = polfit_residuals(xdist, ydist, deg,
                                           debugplot=debugplot)
    if border in ['upper', 'both']:
        # undistorted boundary
        yp_top = np.ones(numpts) * ytop
        # distorted boundary
        xdist, ydist = exvp(xp, yp_top, x0=x0, y0=y0,
                            c2=c2, c4=c4, theta0=theta0)
        poly_upper, dum = polfit_residuals(xdist, ydist, deg,
                                           debugplot=debugplot)

    if border == 'lower':
        return poly_lower
    elif border == 'upper':
        return poly_upper
    else:
        return poly_lower, poly_upper


def fun_residuals(params, bounddict, numresolution, islitmin, islitmax):
    residuals = 0.0
    nsummed = 0

    read_slitlets = bounddict['contents'].keys()
    read_slitlets.sort()
    for tmp_slitlet in read_slitlets:
        islitlet = int(tmp_slitlet[7:])
        if islitmin <= islitlet <= islitmax:
            # expected boundaries using provided parameters
            poly_lower_expected, poly_upper_expected = \
                expected_distorted_boundaries(
                    islitlet, 'both', params,
                    numpts=numresolution, deg=7, debugplot=0
                )
            # print(79 * '-')
            # print('>>> Reading slitlet ', islitlet)
            read_dateobs = bounddict['contents'][tmp_slitlet].keys()
            read_dateobs.sort()
            for tmp_dateobs in read_dateobs:
                # print('...', tmp_dateobs)
                tmp_dict = bounddict['contents'][tmp_slitlet][tmp_dateobs]
                # lower boundary
                poly_lower_measured = np.polynomial.Polynomial(
                    tmp_dict['boundary_coef_lower']
                )
                xmin_lower_bound = tmp_dict['boundary_xmin_lower']
                xmax_lower_bound = tmp_dict['boundary_xmax_lower']
                xdum_lower = np.linspace(xmin_lower_bound, xmax_lower_bound,
                                         num=numresolution)
                # distance between expected and measured polynomials
                poly_diff = poly_lower_expected - poly_lower_measured
                residuals += np.sum(poly_diff(xdum_lower)**2)
                nsummed += numresolution
                # upper boundary
                poly_upper_measured = np.polynomial.Polynomial(
                    tmp_dict['boundary_coef_upper']
                )
                xmin_upper_bound = tmp_dict['boundary_xmin_upper']
                xmax_upper_bound = tmp_dict['boundary_xmax_upper']
                xdum_upper = np.linspace(xmin_upper_bound, xmax_upper_bound,
                                         num=numresolution)
                # distance between expected and measured polynomials
                poly_diff = poly_upper_expected - poly_upper_measured
                residuals += np.sum(poly_diff(xdum_upper)**2)
                nsummed += numresolution

    if nsummed > 0:
        residuals = np.sqrt(residuals/nsummed)
    # print('>>> residuals:', residuals)
    # params.pretty_print()
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


def overplot_boundaries_from_params(ax, params, micolors, linetype='--'):
    for islitlet in range(1, EMIR_NBARS + 1):
        tmpcolor = micolors[islitlet % 2]
        pol_lower_expected, pol_upper_expected = \
            expected_distorted_boundaries(islitlet, 'both', params,
                                          numpts=101, deg=7, debugplot=0)
        xdum = np.linspace(1, NAXIS1_EMIR, num=NAXIS1_EMIR)
        ydum = pol_lower_expected(xdum)
        plt.plot(xdum, ydum, tmpcolor + linetype)
        ydum = pol_upper_expected(xdum)
        plt.plot(xdum, ydum, tmpcolor + linetype)
        # slitlet label
        yc_lower = pol_lower_expected(NAXIS1_EMIR / 2 + 0.5)
        yc_upper = pol_upper_expected(NAXIS1_EMIR / 2 + 0.5)
        ax.text(NAXIS1_EMIR / 2 + 0.5, (yc_lower + yc_upper) / 2,
                str(islitlet), fontsize=10, va='center', ha='center',
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
    ds9_file.write('# filter....: {0}\n'.format(spfilter))
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
                xdum = np.linspace(1, NAXIS1_EMIR, num=numpix)
                ydum = pol_lower_measured(xdum)
                for i in range(len(xdum) - 1):
                    ds9_file.write(
                        'line {0} {1} {2} {3}'.format(xdum[i], ydum[i],
                                                      xdum[i + 1], ydum[i + 1])
                    )
                    ds9_file.write(
                        ' # color={0}\n'.format(colorbox[islitlet % 2]))
                pol_upper_measured = np.polynomial.Polynomial(
                    tmp_dict['boundary_coef_upper']
                )
                ydum = pol_upper_measured(xdum)
                for i in range(len(xdum) - 1):
                    ds9_file.write(
                        'line {0} {1} {2} {3}'.format(xdum[i], ydum[i],
                                                      xdum[i + 1], ydum[i + 1])
                    )
                    ds9_file.write(
                        ' # color={0}\n'.format(colorbox[islitlet % 2]))
    ds9_file.close()


def save_boundaries_from_params_ds9(params, ds9_filename, numpix=100):
    # read individual parameters
    slit_height = params['slit_height'].value
    slit_gap = params['slit_gap'].value
    y_baseline = params['y_baseline'].value
    x0 = params['x0'].value
    y0 = params['y0'].value
    c2 = params['c2'].value
    c4 = params['c4'].value
    theta0 = params['theta0'].value

    ds9_file = open(ds9_filename, 'w')

    ds9_file.write('# Region file format: DS9 version 4.1\n')
    ds9_file.write('global color=green dashlist=2 4 width=2 '
                   'font="helvetica 10 normal roman" select=1 '
                   'highlite=1 dash=1 fixed=0 edit=1 '
                   'move=1 delete=1 include=1 source=1\n')
    ds9_file.write('physical\n')

    ds9_file.write('#\n# slit_height: {0}\n'.format(slit_height))
    ds9_file.write('# slit_gap...: {0}\n'.format(slit_gap))
    ds9_file.write('# y_baseline.: {0}\n'.format(y_baseline))
    ds9_file.write('# x0.........: {0}\n'.format(x0))
    ds9_file.write('# y0.........: {0}\n'.format(y0))
    ds9_file.write('# c2.........: {0}\n'.format(c2))
    ds9_file.write('# c4.........: {0}\n'.format(c4))
    ds9_file.write('# theta0.....: {0}\n'.format(theta0))

    colorbox = ['#ff77ff', '#4444ff']
    for islitlet in range(1, EMIR_NBARS + 1):
        ds9_file.write('#\n# islitlet: {0}\n'.format(islitlet))
        pol_lower_expected, pol_upper_expected = \
            expected_distorted_boundaries(islitlet, 'both', params,
                                          numpts=101, deg=7, debugplot=0)
        xdum = np.linspace(1, NAXIS1_EMIR, num=numpix)
        ydum = pol_lower_expected(xdum)
        for i in range(len(xdum)-1):
            ds9_file.write(
                'line {0} {1} {2} {3}'.format(xdum[i], ydum[i],
                                              xdum[i+1], ydum[i+1])
            )
            ds9_file.write(' # color={0}\n'.format(colorbox[islitlet % 2]))
        ydum = pol_upper_expected(xdum)
        for i in range(len(xdum)-1):
            ds9_file.write(
                'line {0} {1} {2} {3}'.format(xdum[i], ydum[i],
                                              xdum[i+1], ydum[i+1])
            )
            ds9_file.write(' # color={0}\n'.format(colorbox[islitlet % 2]))
        # slitlet label
        yc_lower = pol_lower_expected(NAXIS1_EMIR / 2 + 0.5)
        yc_upper = pol_upper_expected(NAXIS1_EMIR / 2 + 0.5)
        ds9_file.write('text {0} {1} {{{2}}} # color={3} '
                       'font="helvetica 10 bold '
                       'roman"\n'.format(NAXIS1_EMIR / 2 + 0.5,
                                        (yc_lower+yc_upper)/2, islitlet,
                                         colorbox[islitlet % 2]))
    ds9_file.close()


def main(args=None):
    parser = argparse.ArgumentParser(prog='fit_boundaries')
    parser.add_argument("boundict",
                        help="JSON boundary file",
                        type=argparse.FileType('r'))
    parser.add_argument("--numresolution",
                        help="Number of points/boundary (default=101)",
                        type=int, default=101)
    parser.add_argument("--debugplot",
                        help="Integer indicating plotting/debugging" +
                             " (default=0)",
                        type=int, default=0,
                        choices=DEBUGPLOT_CODES)
    args = parser.parse_args(args)

    # read boundict file and check its contents
    bounddict = json.loads(open(args.boundict.name).read())
    integrity_check(bounddict)
    save_boundaries_from_bounddict_ds9(bounddict, 'ds9_bound.reg')

    # initial parameters
    slit_height = 3.390  # in units of 1E1
    slit_gap = 3.578
    # 7(H-H), 3(K-Ksp), -85(LR-HK), -87(LR-YJ)
    y_baseline = 1.323
    theta0 = 0.6503  # in units of 1E-3 radians
    if bounddict['tags']['grism'] == 'J' and \
        bounddict['tags']['filter'] == 'J':
        x0 = 1.0245  # in units of 1E3
        y0 = 1.0245  # in units of 1E3
        c2 = 1.234   # in units of 1E4
        c4 = 1.786   # in units of 1E9
    else:
        raise ValueError("Distortion parameters are not available for this "
                         "combination of grism and filter")

    params = Parameters()
    params.add('slit_height', value=slit_height, vary=True, min=3, max=4)
    params.add('slit_gap', value=slit_gap, vary=True, min=2, max=6)
    params.add('y_baseline', value=y_baseline, vary=True, min=-3, max=7)
    params.add('x0', value=x0, vary=True)
    params.add('y0', value=y0, vary=True)
    params.add('c2', value=c2, vary=True)
    params.add('c4', value=c4, vary=True)
    params.add('theta0', value=theta0, vary=True)

    islitlet_min = 2
    islitlet_max = 54
    array_of_results = np.zeros((islitlet_max - islitlet_min + 1, 2),
                                dtype=minimizer.MinimizerResult)
    list_slitlets = range(islitlet_min, islitlet_max + 1)
    for idum, islitlet in enumerate(list_slitlets):
        result = minimize(fun_residuals, params, method='nelder',
                          args=(bounddict, args.numresolution,
                                islitlet, islitlet))
        # save_boundaries_from_params_ds9(result.params, 'ds9_param.reg')
        print('\n>>> islitlet, chisqr', islitlet, result.chisqr)
        result.params.pretty_print()

        array_of_results[idum, 0] = islitlet
        array_of_results[idum, 1] = result

    pickle.dump(array_of_results, open('array_of_results.p', 'wb') )

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim([0, EMIR_NBARS + 1])
    xplot = array_of_results[:, 0]
    for vardum in ['chisqr', 'slit_height', 'slit_gap', 'y_baseline',
                   'x0', 'y0', 'c2', 'c4', 'theta0']:
        if vardum == 'chisqr':
            yplot = [array_of_results[idum, 1].chisqr
                     for idum in range(array_of_results.shape[0])]
        else:
            yplot = [array_of_results[idum, 1].params[vardum].value
                     for idum in range(array_of_results.shape[0])]
        yplot = np.array(yplot)
        plt.plot(xplot, yplot, 'o', label=vardum)
        print('\n>>> ', vardum)
        stats.summary(yplot, debug=True)
    plt.legend()
    pause_debugplot(debugplot=12, pltshow=True)

    # if args.debugplot % 10 != 0:
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     ax.set_xlim([-0.5, NAXIS1_EMIR + 0.5])
    #     ax.set_ylim([-0.5, NAXIS2_EMIR + 0.5])
    #     ax.set_xlabel('X axis (from 1 to NAXIS1)')
    #     ax.set_xlabel('Y axis (from 1 to NAXIS2)')
    #     ax.set_title(args.boundict.name)
    #     overplot_boundaries_from_bounddict(bounddict, ['r', 'b'])
    #     # overplot_boundaries_from_params(ax, params, ['r', 'b'],
    #     #                                 linetype=':')
    #     overplot_boundaries_from_params(ax, result.params, ['m', 'c'],
    #                                     linetype='--')
    #     pause_debugplot(debugplot=args.debugplot, pltshow=True)


if __name__ == "__main__":

    main()