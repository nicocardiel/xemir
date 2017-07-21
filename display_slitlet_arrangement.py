from __future__ import division
from __future__ import print_function

import argparse
from astropy.io import fits
import numpy as np
import os.path

from numina.array.display.pause_debugplot import pause_debugplot
from slitlet_arrangement import SlitletArrangement

from emirdrp.core import EMIR_NBARS
from numina.array.display.pause_debugplot import DEBUGPLOT_CODES


def display_slitlet_arrangement(fileobj, bbox=None, debugplot=0):
    """Display slitlet arrangment from CSUP keywords in FITS header.

    Parameters
    ----------
    fileobj : file object
        FITS file object.
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
    csu_bar_left : list of floats
        Location (mm) of the left bar for each slitlet.
    csu_bar_right : list of floats
        Location (mm) of the right bar for each slitlet, using the
        same origin employed for csu_bar_left (which is not the
        value stored in the FITS keywords.
    csu_bar_slit_center : list of floats
        Middle point (mm) in between the two bars defining a slitlet.
    csu_bar_slit_width : list of floats
        Slitlet width (mm), computed as the distance between the two
        bars defining the slitlet.

    """

    # read input FITS file
    hdulist = fits.open(fileobj.name)
    image_header = hdulist[0].header
    hdulist.close()

    # additional info from header
    grism = image_header['grism']
    spfilter = image_header['filter']
    rotang = image_header['rotang']

    # define slitlet arrangement
    slt_arrang = SlitletArrangement()
    slt_arrang.define_from_fits(fileobj)

    # display arrangement
    if debugplot >= 10:
        print("slit     left    right   center   width")
        print("====  =======  =======  =======   =====")
        for i in range(EMIR_NBARS):
            ibar = i + 1
            print("{0:4d} {1:8.3f} {2:8.3f} {3:8.3f} {4:7.3f}".format(
                ibar, slt_arrang.csu_bar_left[i], slt_arrang.csu_bar_right[i],
                slt_arrang.csu_bar_slit_center[i],
                slt_arrang.csu_bar_slit_width[i]))
        print(
            "---> {0:8.3f} {1:8.3f} {2:8.3f} {3:7.3f} <- mean (all)".format(
                np.mean(slt_arrang.csu_bar_left),
                np.mean(slt_arrang.csu_bar_right),
                np.mean(slt_arrang.csu_bar_slit_center),
                np.mean(slt_arrang.csu_bar_slit_width)
            )
        )
        print(
            "---> {0:8.3f} {1:8.3f} {2:8.3f} {3:7.3f} <- mean (odd)".format(
                np.mean(slt_arrang.csu_bar_left[::2]),
                np.mean(slt_arrang.csu_bar_right[::2]),
                np.mean(slt_arrang.csu_bar_slit_center[::2]),
                np.mean(slt_arrang.csu_bar_slit_width[::2])
            )
        )
        print(
            "---> {0:8.3f} {1:8.3f} {2:8.3f} {3:7.3f} <- mean (even)".format(
                np.mean(slt_arrang.csu_bar_left[1::2]),
                np.mean(slt_arrang.csu_bar_right[1::2]),
                np.mean(slt_arrang.csu_bar_slit_center[1::2]),
                np.mean(slt_arrang.csu_bar_slit_width[1::2])
            )
        )

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
            ax.add_patch(patches.Rectangle(
                (slt_arrang.csu_bar_left[i], ibar-0.5),
                slt_arrang.csu_bar_slit_width[i], 1.0))
            ax.plot([0., slt_arrang.csu_bar_left[i]], [ibar, ibar], 'o-')
            ax.plot([slt_arrang.csu_bar_right[i], 341.5], [ibar, ibar], 'o-')
        plt.title("File: " + fileobj.name + "\ngrism=" + grism +
                  ", filter=" + spfilter + ", rotang=" + str(round(rotang,2)))
        pause_debugplot(debugplot, pltshow=True)

    # return results
    return slt_arrang.csu_bar_left, \
           slt_arrang.csu_bar_right, \
           slt_arrang.csu_bar_slit_center, \
           slt_arrang.csu_bar_slit_width


def main(args=None):

    # parse command-line options
    parser = argparse.ArgumentParser(prog='display_slitlet_arrangement')
    parser.add_argument("filename",
                        help="FITS files (wildcards accepted) or single TXT "
                             "file with list of FITS files",
                        type=argparse.FileType('r'),
                        nargs='+')
    parser.add_argument("--bbox",
                        help="bounding box tuple xmin,xmax,ymin,"
                             "ymax indicating plot limits")
    parser.add_argument("--debugplot",
                        help="Integer indicating plotting & debugging options"
                             " (default=12)",
                        default=12, type=int,
                        choices=DEBUGPLOT_CODES)
    args = parser.parse_args(args)

    # read bounding box
    if args.bbox is None:
        bbox = None
    else:
        str_bbox = args.bbox.split(",")
        xmin, xmax, ymin, ymax = [int(str_bbox[i]) for i in range(4)]
        bbox = xmin, xmax, ymin, ymax

    list_fits_file_objects = []
    # if input file is a single txt file, assume it is a list of FITS files
    if len(args.filename) == 1:
        if args.filename[0].name[-4:] == ".txt":
            file_content = args.filename[0].read().splitlines()
            for line in file_content:
                if len(line) > 0:
                    if line[0] != '#':
                        tmpfile = line.split()[0]
                        if not os.path.isfile(tmpfile):
                            raise ValueError("File " + tmpfile + " not found!")
                        list_fits_file_objects.append(open(tmpfile, 'r'))
        else:
            list_fits_file_objects = [args.filename[0]]
    else:
        list_fits_file_objects = args.filename

    # total number of files to be examined
    nfiles = len(list_fits_file_objects)

    # declare arrays to store CSU values
    csu_bar_left = np.zeros((nfiles, EMIR_NBARS))
    csu_bar_right = np.zeros((nfiles, EMIR_NBARS))
    csu_bar_slit_center = np.zeros((nfiles, EMIR_NBARS))
    csu_bar_slit_width = np.zeros((nfiles, EMIR_NBARS))

    # display CSU bar arrangement
    for ifile, fileobj in enumerate(list_fits_file_objects):
        print("\nFile " + str(ifile+1) + "/" + str(nfiles) + ": " +
              fileobj.name)
        csu_bar_left[ifile, :], csu_bar_right[ifile, :], \
        csu_bar_slit_center[ifile, :], csu_bar_slit_width[ifile, :] = \
            display_slitlet_arrangement(fileobj, bbox=bbox,
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
    if len(list_fits_file_objects) > 1:
        pause_debugplot(12, optional_prompt="Press RETURN to STOP")


if __name__ == "__main__":
    main()
