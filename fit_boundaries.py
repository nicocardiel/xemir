from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import json
import matplotlib.pyplot as plt
import numpy as np

from numina.array.display.polfit_residuals import polfit_residuals
from numina.array.display.pause_debugplot import pause_debugplot
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
    xdist, ydist : tuple of floats
        Distorted coordinates.

    """
    # plate scale: 0.1944 arcsec/pixel
    # conversion factor (in radian/pixel)
    factor = 0.1944 * np.pi/(180.0*3600)
    # distance from image center (pixels)
    r_pix = np.sqrt((x - x0)**2 + (y - y0)**2)
    # distance from imagen center (radians)
    r_rad = factor * r_pix
    # radial distortion: this number is 1.0 for r=0 and increases
    # slightly (reaching values around 1.033) for r~sqrt(2)*1024
    # (the distance to the corner of the detector measured from the
    # center)
    rdist = (1 + c1 * r_rad**2 + c2 * r_rad**4)
    # angle measured from the Y axis towards the X axis
    theta = np.arctan((x - x0)/(y - y0))
    if y < y0:
        theta = theta - np.pi
    # distorted coordinates
    xdist = (rdist * r_pix * np.sin(theta)) + x0
    ydist = (rdist * r_pix * np.cos(theta)) + y0
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


def expected_distorted_boundaries(islitlet, slit_height, slit_gap,
                                  y_baseline, x0, y0, c1, c2,
                                  numpts, deg,
                                  debugplot=0):
    """Return polynomial coefficients of expected distorted boundaries.

    """

    xp = np.linspace(1, NAXIS1_EMIR, numpts)
    slit_dist = slit_height + slit_gap

    # y-coordinates at x=1024.5
    ybottom = y_baseline + (islitlet - 1) * slit_dist
    ytop = ybottom + slit_height
    # undistorted lower and upper slitlet boundaries
    yp_bottom = np.ones(numpts) * ybottom
    yp_top = np.ones(numpts) * ytop
    # distorted lower boundary
    xdist, ydist = exvp(xp, yp_bottom, x0=x0, y0=y0, c1=c1, c2=c2)
    poly_lower, dum = polfit_residuals(xdist, ydist, deg, debugplot=debugplot)
    # distorted upper boundary
    xdist, ydist = exvp(xp, yp_top, x0=x0, y0=y0, c1=c1, c2=c2)
    poly_upper, dum = polfit_residuals(xdist, ydist, deg, debugplot=debugplot)

    return poly_lower, poly_upper


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

    slit_height = 33.5
    slit_gap = 3.95
    y_baseline = 2  # 1(J-J), 7(H-H), 3(K-Ksp), -85(LR-HK), -87(LR-YJ)
    if bounddict['tags']['grism'] == 'J' and \
        bounddict['tags']['filter'] == 'J':
        x0 = 1024.5
        y0 = 1024.5
        c1 = 14606.7
        c2 = 1739716115.1
    else:
        raise ValueError("Distortion parameters are not available for this "
                         "combination of grism and filter")

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
        print('>>> Reading slitlet ', islitlet)
        # expected boundaries
        pol_lower_expected, pol_upper_expected = expected_distorted_boundaries(
            islitlet, slit_height, slit_gap, y_baseline, x0, y0, c1, c2,
            numpts=101, deg=7, debugplot=0)
        read_dateobs = bounddict['contents'][tmp_slitlet].keys()
        read_dateobs.sort()
        for tmp_dateobs in read_dateobs:
            print('...', tmp_dateobs)
            tmp_dict = bounddict['contents'][tmp_slitlet][tmp_dateobs]
            # lower boundary
            pol_lower_bound = np.polynomial.Polynomial(
                tmp_dict['boundary_coef_lower']
            )
            xmin_lower_bound = tmp_dict['boundary_xmin_lower']
            xmax_lower_bound = tmp_dict['boundary_xmax_lower']
            xdum_lower = np.arange(xmin_lower_bound, xmax_lower_bound+1)
            ydum_lower_measured = pol_lower_bound(xdum_lower)
            ydum_lower_expected = pol_lower_expected(xdum_lower)
            # upper boundary
            pol_upper_bound = np.polynomial.Polynomial(
                tmp_dict['boundary_coef_upper']
            )
            xmin_upper_bound = tmp_dict['boundary_xmin_upper']
            xmax_upper_bound = tmp_dict['boundary_xmax_upper']
            xdum_upper = np.arange(xmin_upper_bound, xmax_upper_bound + 1)
            ydum_upper_measured = pol_upper_bound(xdum_upper)
            ydum_upper_expected = pol_upper_expected(xdum_upper)
            if args.debugplot % 10 != 0:
                tmpcolor = micolors[islitlet % 2]
                tmpcolor_exp = micolors_exp[islitlet % 2]
                plt.plot(xdum_lower, ydum_lower_measured, tmpcolor+'-')
                plt.plot(xdum_lower, ydum_lower_expected, tmpcolor_exp+'--')
                plt.plot(xdum_upper, ydum_upper_measured, tmpcolor + '-')
                plt.plot(xdum_upper, ydum_upper_expected, tmpcolor_exp + '--')
                yc_lower = pol_lower_bound(NAXIS1_EMIR / 2 + 0.5)
                yc_upper = pol_upper_bound(NAXIS1_EMIR/2 + 0.5)
                ax.text(NAXIS1_EMIR / 2 + 0.5, (yc_lower + yc_upper) / 2,
                        str(islitlet), fontsize=10, va='center', ha='center',
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="grey"),
                        color=tmpcolor, fontweight='bold', backgroundcolor='white')
                for xp, yp1, yp2 in zip(
                        xdum_lower[::50],
                        ydum_lower_measured[::50],
                        ydum_lower_expected[::50]):
                    plt.plot([xp, xp], [yp1, yp2], 'y-')
            # compute optimal parameters for current islitlet
            # seguir aqui
            # seguir aqui

    pause_debugplot(debugplot=args.debugplot, pltshow=True)


if __name__ == "__main__":

    main()
