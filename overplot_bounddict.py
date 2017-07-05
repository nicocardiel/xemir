from __future__ import division
from __future__ import print_function

import argparse
from astropy.io import fits
import json
import numpy as np
from numpy.polynomial import Polynomial
import os.path

from display_slitlet_arrangement import read_csup_from_header

from numina.array.display.pause_debugplot import pause_debugplot
from numina.array.display.polfit_residuals import polfit_residuals
from numina.array.display.ximshow import ximshow

from emirdrp.core import EMIR_NBARS

from emir_definitions import NAXIS1_EMIR
from emir_definitions import VALID_GRISMS
from emir_definitions import VALID_FILTERS


def get_boundaries(grism, spfilter, slitlet_number,
                   csu_bar_slit_center, nsampling=10,
                   deg_boundary=4,
                   debugplot=0):
    """Read the bounddict json file and compute boundaries for the slitlet.

    The boundaries are interpolated according to the requested
    csu_bar_slit_center value.

    Note that instead of fitting a 2d polynomial surface (which can
    be problematic considering that the polynomial degree of the
    boundaries is relatively high), two perpendicular 1d interpolations
    are performed. This approach is computationally slower but very
    robust.

    Parameters
    ----------
    grism : string
        Character string ("J", "H", "K" or LR) indicating the grism.
    spfilter : string
        Character string ("J", "H", "Ksp",...) indicating the filter.
    slitlet_number : int
        Number of slitlet.
    csu_bar_slit_center : float
        Middle point (mm) in between the two bars defining a slitlet.
    nsampling : int
        Sampling of the boundaries. Each boundary in the json file is
        sampled into nsampling points for each value of the
        corresponding csu_bar_slit_center. For each of this nsampling
        points, an interpolation polynomial (as a function of
        csu_bar_slit_center) is computed in order to estimate the
        corresponding point for the requested value of
        csu_bar_slit_center.
    deg_boundary : int
        Polynomial degree to fit boundaries.
    debugplot : int
        Determines whether intermediate computations and/or plots
        are displayed:
        00 : no debug, no plots
        01 : no debug, plots without pauses
        02 : no debug, plots with pauses
        10 : debug, no plots
        11 : debug, plots without pauses
        12 : debug, plots with pauses
        This parameter is ignored when show is False.

    Returns
    -------
    pol_lower_boundary : numpy polynomial
        Polynomial defining the lower boundary of the slitlet.
    pol_upper_boundary : numpy polynomial
        Polynomial defining the upper boundary of the slitlet.

    """

    # read bounddict_grism_?_filter_?.json
    json_filename = "bounddict_grism_" + grism + \
                 "_filter_" + spfilter + ".json"
    if os.path.isfile(json_filename):
        bounddict = json.loads(open(json_filename).read())
    else:
        raise ValueError("File " + json_filename + " not found!")
    # print(json.dumps(bounddict, indent=4, sort_keys=True))

    # return values in case the requested slitlet number is not defined
    pol_lower_boundary = None
    pol_upper_boundary = None

    # search the slitlet number in bounddict
    slitlet_label = "slitlet" + str(slitlet_number).zfill(2)
    if slitlet_label in bounddict['contents'].keys():
        list_date_obs = bounddict['contents'][slitlet_label].keys()
        list_date_obs.sort()
        num_date_obs = len(list_date_obs)
        if num_date_obs == 1:
            date_obs = list_date_obs[0]
            tmp_dict = bounddict['contents'][slitlet_label][date_obs]
            pol_lower_boundary = Polynomial(tmp_dict['boundary_coef_lower'])
            pol_upper_boundary = Polynomial(tmp_dict['boundary_coef_upper'])
        else:
            matrix_data_lower = np.zeros((num_date_obs, nsampling))
            matrix_data_upper = np.zeros((num_date_obs, nsampling))
            xfit = np.zeros(num_date_obs)
            ysampled_lower = np.zeros(nsampling)
            ysampled_upper = np.zeros(nsampling)
            xsampled = np.linspace(start=1, stop=NAXIS1_EMIR, num=nsampling)
            for i,date_obs in enumerate(list_date_obs):
                tmp_dict = bounddict['contents'][slitlet_label][date_obs]
                xfit[i] = tmp_dict['csu_bar_slit_center']
                pol_lower_boundary = Polynomial(
                    tmp_dict['coef_lower_boundary'])
                matrix_data_lower[i,:] = pol_lower_boundary(xsampled)
                pol_upper_boundary = Polynomial(
                    tmp_dict['coef_upper_boundary'])
                matrix_data_upper[i,:] = pol_upper_boundary(xsampled)
            num_unique_csu_values = np.unique(xfit).size
            if num_unique_csu_values == 1:
                deg_csu = 0
            elif num_unique_csu_values == 2:
                deg_csu = 1
            else:
                deg_csu = 2
            if deg_csu == 0:
                ysampled_lower = np.mean(matrix_data_lower, axis=0)
                ysampled_upper = np.mean(matrix_data_upper, axis=0)
            else:
                for j in range(nsampling):
                    yfit_lower = matrix_data_lower[:,j]
                    poly_lower, yres = polfit_residuals(
                        x=xfit, y=yfit_lower, deg=deg_csu,
                        debugplot=debugplot)
                    ysampled_lower[j] = poly_lower(csu_bar_slit_center)
                    yfit_upper = matrix_data_upper[:,j]
                    poly_upper, yres = polfit_residuals(
                        x=xfit, y=yfit_upper, deg=deg_csu,
                        debugplot=debugplot)
                    ysampled_upper[j] = poly_upper(csu_bar_slit_center)
            pol_lower_boundary, yres = polfit_residuals(
                x=xsampled, y=ysampled_lower, deg=deg_boundary,
                debugplot=debugplot)
            pol_upper_boundary, yres = polfit_residuals(
                x=xsampled, y=ysampled_upper, deg=deg_boundary,
                debugplot=debugplot)
    else:
        print("WARNING: slitlet number " + str(slitlet_number) +
              " is not available in " + json_filename)

    # return result
    return pol_lower_boundary, pol_upper_boundary


def main(args=None):
    # parse command-line options
    parser = argparse.ArgumentParser()
    parser.add_argument("filename",
                        help="FITS file name to be displayed",
                        type=argparse.FileType('r'))
    parser.add_argument("grism",
                        help="grism name",
                        choices=VALID_GRISMS)
    parser.add_argument("filter",
                        help="Filter name",
                        choices=VALID_FILTERS)
    parser.add_argument("tuple_slit_numbers",
                        help="Tuple n1[,n2[,step]] to define slitlet numbers")

    args = parser.parse_args()

    # read slitlet numbers to be computed
    tmp_str = args.tuple_slit_numbers.split(",")
    if len(tmp_str) == 3:
        if int(tmp_str[0]) < 1:
            raise ValueError("Invalid slitlet number < 1")
        if int(tmp_str[1]) > EMIR_NBARS:
            raise ValueError("Invalid slitlet number > EMIR_NBARS")
        list_slitlets = range(int(tmp_str[0]),
                              int(tmp_str[1])+1,
                              int(tmp_str[2]))
    elif len(tmp_str) == 2:
        if int(tmp_str[0]) < 1:
            raise ValueError("Invalid slitlet number < 1")
        if int(tmp_str[1]) > EMIR_NBARS:
            raise ValueError("Invalid slitlet number > EMIR_NBARS")
        list_slitlets = range(int(tmp_str[0]),
                              int(tmp_str[1])+1,
                              1)
    elif len(tmp_str) == 1:
        if int(tmp_str[0]) < 1:
            raise ValueError("Invalid slitlet number < 1")
        if int(tmp_str[0]) > EMIR_NBARS:
            raise ValueError("Invalid slitlet number > EMIR_NBARS")
        list_slitlets = [int(tmp_str[0])]
    else:
        raise ValueError("Invalid tuple for slitlet numbers")

    # read input FITS file
    hdulist = fits.open(args.filename.name)
    image_header = hdulist[0].header
    image2d = hdulist[0].data
    hdulist.close()

    naxis1 = image_header['naxis1']
    naxis2 = image_header['naxis2']

    if image2d.shape != (naxis2, naxis1):
        raise ValueError("Unexpected error with NAXIS1, NAXIS2")

    # remove path from filename
    sfilename = os.path.basename(args.filename.name)

    # check that the FITS file has been obtained with EMIR
    instrument = image_header['instrume']
    if instrument != 'EMIR':
        raise ValueError("INSTRUME keyword is not 'EMIR'!")

    # read CSU configuration for FITS header
    csu_bar_left, csu_bar_right, csu_bar_slit_center, csu_bar_slit_width = \
        read_csup_from_header(image_header=image_header, debugplot=0)

    # read grism
    grism_in_header = image_header['grism']
    if args.grism != grism_in_header:
        raise ValueError("GRISM keyword=" + grism_in_header +
                         " is not the expected value=" + args.grism)
    # read filter
    spfilter_in_header = image_header['filter']
    if args.filter != spfilter_in_header:
        raise ValueError("FILTER keyword=" + spfilter_in_header +
                         " is not the expected value=" + args.filter)
    # read rotator position angle
    rotang = image_header['rotang']

    # display full image
    ax = ximshow(image2d=image2d,
                 title=sfilename + "\ngrism=" + args.grism +
                       ", filter=" + args.filter +
                       ", rotang=" + str(round(rotang, 2)),
                 image_bbox=(1, naxis1, 1, naxis2), show=False)

    # overplot boundaries for each slitlet
    xp = np.linspace(start=1, stop=NAXIS1_EMIR, num=1000)
    for slitlet_number in list_slitlets:
        pol_lower_boundary, pol_upper_boundary = get_boundaries(
            args.grism, args.filter, slitlet_number,
            csu_bar_slit_center[slitlet_number-1],
            nsampling=100, deg_boundary=4, debugplot=0
        )
        if (pol_lower_boundary is not None) and \
                (pol_upper_boundary is not None):
            yp = pol_lower_boundary(xp)
            ax.plot(xp, yp, 'g-')
            yp = pol_upper_boundary(xp)
            ax.plot(xp, yp, 'b-')

    # show plot
    pause_debugplot(12, pltshow=True)


if __name__ == "__main__":

    main()
