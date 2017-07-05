from __future__ import division
from __future__ import print_function

import argparse
from astropy.io import fits
import json
import numpy as np
from numpy.polynomial import Polynomial
import os.path

from numina.array.display.pause_debugplot import pause_debugplot
from numina.array.display.ximshow import ximshow

from emirdrp.core import EMIR_NBARS

from display_slitlet_arrangement import display_slitlet_arrangement

from emir_definitions import VALID_GRISMS
from emir_definitions import VALID_FILTERS


def poly_from_dict(tmpdict):
    """Return numpy polynomial from dictionary storing polynomial coefficients.

    This function avoids problems due to the fact that information
    stored in python dictionaries are not sorted.

    Parameters
    ----------
    tmpdict : dictionary
        The dictionary must contain the different coefficients saved
        under keywords '0', '1', '2',...

    Returns
    -------
    poly : numpy polynomial
        Numpy polynomial defined with the coefficients read from the
        input dictionary.

    """

    coeff = np.array([tmpdict[str(i)] for i in range(len(tmpdict))])
    return Polynomial(coeff)


if __name__ == "__main__":

    # parse command-line options
    parser = argparse.ArgumentParser()
    parser.add_argument("filename",
                        help="FITS file name to be displayed")
    parser.add_argument("grism",
                        help="grism name (J, H, K, YJ or HK")
    parser.add_argument("filter",
                        help="Filter name ('J', 'H', 'Ksp',...)")
    args = parser.parse_args()

    # read input FITS file
    filename = args.filename
    if not os.path.isfile(filename):
        raise ValueError("File " + filename + " not found!")

    # read grism
    grism = args.grism
    if grism not in VALID_GRISMS:
        raise ValueError("Unexpected grism: " + grism)

    # read filter name
    spfilter = args.filter
    if spfilter not in VALID_FILTERS:
        raise ValueError(
            "Filter=" + spfilter + " is not in valid filter list")

    # read wvdict_grism_[grism]_filter_[filter].json
    main_label = "wvdict_grism_" + grism + "_filter_" + spfilter
    wvdict_file = main_label + ".json"
    if os.path.isfile(wvdict_file):
        wvdict = json.loads(open(wvdict_file).read())
        print('\n>>> Reading megadict from file:')
        print(wvdict_file)
        if wvdict.keys() != [main_label]:
            raise ValueError("Invalid initial key in " + wvdict_file)
    else:
        raise ValueError("File " + wvdict_file + " not found!")
    # print(json.dumps(wvdict, indent=4, sort_keys=True))

    # read input FITS file
    hdulist = fits.open(filename)
    image_header = hdulist[0].header
    image2d = hdulist[0].data
    hdulist.close()

    naxis1 = image_header['naxis1']
    naxis2 = image_header['naxis2']

    if image2d.shape != (naxis2, naxis1):
        raise ValueError("Unexpected error with NAXIS1, NAXIS2")

    if True:
        csu_bar_left, csu_bar_right, csu_bar_slit_center, \
        csu_bar_slit_width = display_slitlet_arrangement(filename)
    else:
        csu_bar_slit_center = np.zeros(EMIR_NBARS)
        csu_bar_slit_center[40 - 1] = 1.23100431 + 0.16390062 * 425.15

    # display full image
    ax = ximshow(image2d=image2d, title=filename,
                 image_bbox=(1, naxis1, 1, naxis2), show=False)

    # display (xpos, ypos)
    for i in range(EMIR_NBARS):
        ibar = i + 1
        slitlet_label = "slitlet" + str(ibar).zfill(2)
        if slitlet_label in wvdict[main_label].keys():
            list_wv = wvdict[main_label][slitlet_label].keys()
            list_wv.sort()
            xpos_0 = []
            ypos_0 = []
            xpos_1 = []
            ypos_1 = []
            xpos_2 = []
            ypos_2 = []
            markcolor = []
            marksize = []
            for wv in list_wv:
                wvdictz = wvdict[main_label][slitlet_label][wv]  # alias
                if 'data_csu_bar_slit_center_0' in wvdictz.keys():
                    min_csu_bar = wvdictz['data_csu_bar_slit_center_0'] - 10
                    max_csu_bar = wvdictz['data_csu_bar_slit_center_1'] + 10
                    if min_csu_bar <= csu_bar_slit_center[i] <= max_csu_bar:
                        poly_xpos_0 = poly_from_dict(
                            wvdictz['data_poly_xpos_0_vs_csu'])
                        poly_ypos_0 = poly_from_dict(
                            wvdictz['data_poly_ypos_0_vs_csu'])
                        xpos_0.append(poly_xpos_0(csu_bar_slit_center[i]))
                        ypos_0.append(poly_ypos_0(csu_bar_slit_center[i]))
                        #
                        poly_xpos_1 = poly_from_dict(
                            wvdictz['data_poly_xpos_1_vs_csu'])
                        poly_ypos_1 = poly_from_dict(
                            wvdictz['data_poly_ypos_1_vs_csu'])
                        xpos_1.append(poly_xpos_1(csu_bar_slit_center[i]))
                        ypos_1.append(poly_ypos_1(csu_bar_slit_center[i]))
                        #
                        poly_xpos_2 = poly_from_dict(
                            wvdictz['data_poly_xpos_2_vs_csu'])
                        poly_ypos_2 = poly_from_dict(
                            wvdictz['data_poly_ypos_2_vs_csu'])
                        xpos_2.append(poly_xpos_2(csu_bar_slit_center[i]))
                        ypos_2.append(poly_ypos_2(csu_bar_slit_center[i]))
                        markcolor.append('b')
                        marksize.append(50)
                if 'model_csu_bar_slit_center_0' in wvdictz.keys():
                    min_csu_bar = wvdictz['model_csu_bar_slit_center_0'] - 10
                    max_csu_bar = wvdictz['model_csu_bar_slit_center_1'] + 10
                    if min_csu_bar <= csu_bar_slit_center[i] <= max_csu_bar:
                        poly_xpos_0 = poly_from_dict(
                            wvdictz['model_poly_xpos_0_vs_csu'])
                        poly_ypos_0 = poly_from_dict(
                            wvdictz['model_poly_ypos_0_vs_csu'])
                        xpos_0.append(poly_xpos_0(csu_bar_slit_center[i]))
                        ypos_0.append(poly_ypos_0(csu_bar_slit_center[i]))
                        #
                        poly_xpos_1 = poly_from_dict(
                            wvdictz['model_poly_xpos_1_vs_csu'])
                        poly_ypos_1 = poly_from_dict(
                            wvdictz['model_poly_ypos_1_vs_csu'])
                        xpos_1.append(poly_xpos_1(csu_bar_slit_center[i]))
                        ypos_1.append(poly_ypos_1(csu_bar_slit_center[i]))
                        #
                        poly_xpos_2 = poly_from_dict(
                            wvdictz['model_poly_xpos_2_vs_csu'])
                        poly_ypos_2 = poly_from_dict(
                            wvdictz['model_poly_ypos_2_vs_csu'])
                        xpos_2.append(poly_xpos_2(csu_bar_slit_center[i]))
                        ypos_2.append(poly_ypos_2(csu_bar_slit_center[i]))
                        markcolor.append('g')
                        marksize.append(25)
            xpos_0 = np.array(xpos_0)
            ypos_0 = np.array(ypos_0)
            xpos_1 = np.array(xpos_1)
            ypos_1 = np.array(ypos_1)
            xpos_2 = np.array(xpos_2)
            ypos_2 = np.array(ypos_2)
            ax.scatter(xpos_0, ypos_0, marker='o', c=markcolor, s=marksize,
                       edgecolor='w')
            ax.scatter(xpos_1, ypos_1, marker='o', c=markcolor, s=marksize,
                       edgecolor='w')
            ax.scatter(xpos_2, ypos_2, marker='o', c=markcolor, s=marksize,
                       edgecolor='w')
            for i in range(len(xpos_1)):
                ax.plot([xpos_1[i], xpos_2[i]], [ypos_1[i], ypos_2[i]],
                        'y-')

    # show plot
    pause_debugplot(12, pltshow=True)

    # ToDo:
    # - Ver que hacer cuando las rendijas estan cerradas:
    #   csu_bar_slit_width sera inferior a un valor dado

