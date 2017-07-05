from __future__ import division
from __future__ import print_function

import argparse
from astropy.io import fits
import numpy as np
import os.path

from numina.array.display.pause_debugplot import pause_debugplot

from emirdrp.core import EMIR_NBARS


def read_csup_from_header(image_header, debugplot=0):
    """Read CSUP keywords from FITS header.

    Parameter
    ---------
    image_header : astropy image header
        Header of FITS image.
    debugplot : int
        Determines whether intermediate computations and/or plots
        are displayed:
        00 : no debug, no plots
        01 : no debug, plots without pauses
        02 : no debug, plots with pauses
        10 : debug, no plots
        11 : debug, plots without pauses
        12 : debug, plots with pauses

    Returns
    -------
    csu_bar_left : array, float
        Location (mm) of the left bar for each slitlet.
    csu_bar_right : array, float
        Location (mm) of the right bar for each slitlet, using the
        same origin employed for csu_bar_left (which is not the
        value stored in the FITS keywords.
    csu_bar_slit_center : array, float
        Middle point (mm) in between the two bars defining a slitlet.
    csu_bar_slit_width : array, float
        Slitlet width (mm), computed as the distance between the two
        bars defining the slitlet.

    """

    # declare arrays to store CSU bar configuration
    csu_bar_left = np.zeros(EMIR_NBARS)
    csu_bar_right = np.zeros(EMIR_NBARS)
    csu_bar_slit_center = np.zeros(EMIR_NBARS)
    csu_bar_slit_width = np.zeros(EMIR_NBARS)

    # loop to read all the CSUP keywords
    if debugplot >= 10:
        print("slit     left    right   center   width")
        print("====  =======  =======  =======   =====")

    for i in range(EMIR_NBARS):
        ibar = i + 1
        keyword = 'CSUP' + str(ibar)
        if keyword in image_header:
            csu_bar_left[i] = image_header[keyword]
        else:
            raise ValueError("Expected keyword " + keyword + " not found!")
        keyword = 'CSUP' + str(ibar + EMIR_NBARS)
        if keyword in image_header:
            csu_bar_right[i] = image_header[keyword]
            # set the same origin as the one employed for csu_bar_left
            csu_bar_right[i] = 341.5 - csu_bar_right[i]
        else:
            raise ValueError("Expected keyword " + keyword + " not found!")
        csu_bar_slit_center[i] = (csu_bar_left[i] + csu_bar_right[i])/2
        csu_bar_slit_width[i] = csu_bar_right[i] - csu_bar_left[i]
        if debugplot >= 10:
            print("{0:4d} {1:8.3f} {2:8.3f} {3:8.3f} {4:7.3f}".format(
                ibar, csu_bar_left[i], csu_bar_right[i],
                csu_bar_slit_center[i], csu_bar_slit_width[i]))

    # return results
    return csu_bar_left, csu_bar_right, csu_bar_slit_center, \
           csu_bar_slit_width


def display_slitlet_arrangement(filename, bbox=None, debugplot=0):
    """Display slitlet arrangment from CSUP keywords in FITS header.

    Parameters
    ----------
    filename : string
        FITS file name.
    bbox : tuple of 4 floats
        If not None, values for xmin, xmax, ymin and ymax.
    debugplot : int
        Determines whether intermediate computations and/or plots
        are displayed:
        00 : no debug, no plots
        01 : no debug, plots without pauses
        02 : no debug, plots with pauses
        10 : debug, no plots
        11 : debug, plots without pauses
        12 : debug, plots with pauses

    Returns
    -------
    csu_bar_left : array, float
        Location (mm) of the left bar for each slitlet.
    csu_bar_right : array, float
        Location (mm) of the right bar for each slitlet, using the
        same origin employed for csu_bar_left (which is not the
        value stored in the FITS keywords.
    csu_bar_slit_center : array, float
        Middle point (mm) in between the two bars defining a slitlet.
    csu_bar_slit_width : array, float
        Slitlet width (mm), computed as the distance between the two
        bars defining the slitlet.

    """

    # read input FITS file
    hdulist = fits.open(filename)
    image_header = hdulist[0].header
    image2d = hdulist[0].data
    hdulist.close()

    # image dimensions
    naxis1 = image_header['naxis1']
    naxis2 = image_header['naxis2']
    if image2d.shape != (naxis2, naxis1):
        raise ValueError("Unexpected error with NAXIS1, NAXIS2")

    # additional info from header
    grism = image_header['grism']
    spfilter = image_header['filter']
    rotang = image_header['rotang']

    # read CSU bar configuration
    csu_bar_left, csu_bar_right, csu_bar_slit_center, csu_bar_slit_width = \
        read_csup_from_header(image_header=image_header, debugplot=debugplot)

    # display slit arrangement
    if debugplot % 10 != 0:
        import matplotlib
        matplotlib.use('Qt5Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if bbox is None:
            ax.set_xlim([0., 341.5])
            ax.set_ylim([0, 56])
        else:
            ax.set_xlim([bbox[0], bbox[1]])
            ax.set_ylim([bbox[2], bbox[3]])
        ax.set_xlabel('csu_bar_position (mm)')
        ax.set_ylabel('slit number')
        for i in range(EMIR_NBARS):
            ibar = i + 1
            ax.add_patch(patches.Rectangle((csu_bar_left[i], ibar-0.5),
                                           csu_bar_slit_width[i], 1.0))
            ax.plot([0., csu_bar_left[i]], [ibar, ibar], 'o-')
            ax.plot([csu_bar_right[i], 341.5], [ibar, ibar], 'o-')
        plt.title("File: " + filename + "\ngrism=" + grism +
                  ", filter=" + spfilter + ", rotang=" + str(round(rotang,2)))
        pause_debugplot(debugplot, pltshow=True)

    # return results
    return csu_bar_left, csu_bar_right, csu_bar_slit_center, \
           csu_bar_slit_width


def main(args=None):

    # parse command-line options
    parser = argparse.ArgumentParser(prog='display_slitlet_arrangement')
    parser.add_argument("filename",
                        help="FITS file or txt file with list of FITS files")
    parser.add_argument("--bbox",
                        help="bounding box tuple xmin,xmax,ymin,ymax")
    parser.add_argument("--debugplot",
                        help="Integer indicating plotting/debugging" +
                             " (default=12)",
                        default=12, type=int,
                        choices=[0, 1, 2, 10, 11, 12, 21, 22])
    args = parser.parse_args(args)

    # check for input file
    filename = args.filename
    if not os.path.isfile(filename):
        raise ValueError("File " + filename + " not found!")

    # read bounding box
    if args.bbox is None:
        bbox = None
    else:
        tmp_bbox = args.bbox.split(",")
        xmin = int(tmp_bbox[0])
        xmax = int(tmp_bbox[1])
        ymin = int(tmp_bbox[2])
        ymax = int(tmp_bbox[3])
        bbox = xmin, xmax, ymin, ymax

    # if input file is a txt file, assume it is a list of FITS files
    if filename[-4:] == ".txt":
        with open(filename) as f:
            file_content = f.read().splitlines()
        list_fits_files = []
        for line in file_content:
            if len(line) > 0:
                if line[0] != '#':
                    tmpfile = line.split()[0]
                    if not os.path.isfile(tmpfile):
                        raise ValueError("File " + tmpfile + " not found!")
                    list_fits_files.append(tmpfile)
    else:
        list_fits_files = [filename]

    # total number of files to be examined
    nfiles = len(list_fits_files)

    # declare arrays to store CSU values
    csu_bar_left = np.zeros((nfiles, EMIR_NBARS))
    csu_bar_right = np.zeros((nfiles, EMIR_NBARS))
    csu_bar_slit_center = np.zeros((nfiles, EMIR_NBARS))
    csu_bar_slit_width = np.zeros((nfiles, EMIR_NBARS))

    # display CSU bar arrangement
    for ifile, file in enumerate(list_fits_files):
        print("\nFile " + str(ifile+1) + "/" + str(nfiles) + ": " + file)
        csu_bar_left[ifile, :], csu_bar_right[ifile, :], \
        csu_bar_slit_center[ifile, :], csu_bar_slit_width[ifile, :] = \
            display_slitlet_arrangement(file, bbox=bbox,
                                        debugplot=args.debugplot)

    # print summary of comparison between files
    if nfiles > 1:
        std_csu_bar_left = np.zeros(EMIR_NBARS)
        std_csu_bar_right = np.zeros(EMIR_NBARS)
        std_csu_bar_slit_center = np.zeros(EMIR_NBARS)
        std_csu_bar_slit_width = np.zeros(EMIR_NBARS)
        if args.debugplot >= 10:
            print("\n   STANDARD DEVIATION BETWEEN IMAGES")
            print("slit     left    right   center   width")
            print("====  =======  =======  =======   =====")
            for i in range(EMIR_NBARS):
                ibar = i + 1
                std_csu_bar_left[i] = np.std(csu_bar_left[:, i])
                std_csu_bar_right[i] = np.std(csu_bar_right[:, i])
                std_csu_bar_slit_center[i] = np.std(csu_bar_slit_center[:, i])
                std_csu_bar_slit_width[i] = np.std(csu_bar_slit_width[:, i])
                print("{0:4d} {1:8.3f} {2:8.3f} {3:8.3f} {4:7.3f}".format(
                    ibar,
                    std_csu_bar_left[i],
                    std_csu_bar_right[i],
                    std_csu_bar_slit_center[i],
                    std_csu_bar_slit_width[i]))
            print("====  =======  =======  =======   =====")
            print("MIN: {0:8.3f} {1:8.3f} {2:8.3f} {3:7.3f}".format(
                std_csu_bar_left.min(),
                std_csu_bar_right.min(),
                std_csu_bar_slit_center.min(),
                std_csu_bar_slit_width.min()))
            print("MAX: {0:8.3f} {1:8.3f} {2:8.3f} {3:7.3f}".format(
                std_csu_bar_left.max(),
                std_csu_bar_right.max(),
                std_csu_bar_slit_center.max(),
                std_csu_bar_slit_width.max()))
            print("====  =======  =======  =======   =====")
            print("Total number of files examined:", nfiles)

    # stop program execution
    if len(list_fits_files) > 1:
        pause_debugplot(12, optional_prompt="Press RETURN to STOP")


if __name__ == "__main__":
    main()
