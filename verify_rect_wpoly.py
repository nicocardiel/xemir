from __future__ import division
from __future__ import print_function

import argparse
import astropy.io.fits as fits
import json
import numpy as np
import os

from numina.array.wavecalib.__main__ import read_wv_master_file
from numina.array.wavecalib.check_wlcalib import check_wlcalib_sp

from numina.array.display.pause_debugplot import DEBUGPLOT_CODES


def main(args=None):
    # parse command-line options
    parser = argparse.ArgumentParser(prog='verify_rect_wpoly')
    # positional parameters
    parser.add_argument("fitsfile",
                        help="Rectified and wavelength calibrated FITS image",
                        type=argparse.FileType('r'))
    parser.add_argument("--rect_wpoly", required=True,
                        help="Initial JSON file with rectification and "
                             "wavelength calibration coefficients",
                        type=argparse.FileType('r'))
    parser.add_argument("--wv_master_file", required=True,
                        help="TXT file containing wavelengths",
                        type=argparse.FileType('r'))
    parser.add_argument("--verified_rect_wpoly", required=True,
                        help="Output JSON file with improved wavelength "
                             "calibration",
                        default=None,
                        type=argparse.FileType('w'))
    # optional arguments
    parser.add_argument("--geometry",
                        help="tuple x,y,dx,dy",
                        default="0,0,640,480")
    parser.add_argument("--debugplot",
                        help="integer indicating plotting/debugging" +
                             " (default=0)",
                        type=int, default=12,
                        choices=DEBUGPLOT_CODES)
    parser.add_argument("--echo",
                        help="Display full command line",
                        action="store_true")

    args = parser.parse_args(args=args)

    if args.echo:
        print('\033[1m\033[31m% ' + ' '.join(sys.argv) + '\033[0m\n')

    # geometry
    if args.geometry is None:
        geometry = None
    else:
        tmp_str = args.geometry.split(",")
        x_geom = int(tmp_str[0])
        y_geom = int(tmp_str[1])
        dx_geom = int(tmp_str[2])
        dy_geom = int(tmp_str[3])
        geometry = x_geom, y_geom, dx_geom, dy_geom

    # read calibration structure from JSON file
    rect_wpoly_dict = json.loads(open(args.rect_wpoly.name).read())
    islitlet_min = rect_wpoly_dict['tags']['islitlet_min']
    islitlet_max = rect_wpoly_dict['tags']['islitlet_max']

    # read FITS image and its corresponding header
    hdulist = fits.open(args.fitsfile)
    header = hdulist[0].header
    image2d_rectified_wv = hdulist[0].data
    hdulist.close()
    crpix1_enlarged = header['crpix1']
    crval1_enlarged = header['crval1']
    cdelt1_enlarged = header['cdelt1']

    # read master arc line wavelengths (whole data set)
    wv_master_all = read_wv_master_file(
        wv_master_file=args.verify_arc_lines,
        lines='all',
        debugplot=args.debugplot
    )

    # main loop
    for islitlet in range(islitlet_min, islitlet_max + 1):
        sltmin = header['sltmin' + str(islitlet).zfill(2)]
        sltmax = header['sltmax' + str(islitlet).zfill(2)]
        spmedian = np.median(image2d_rectified_wv[sltmin:(sltmax + 1)],
                             axis=0)
        polyres, ysummary  = check_wlcalib_sp(
            sp=spmedian,
            crpix1=crpix1_enlarged,
            crval1=crval1_enlarged,
            cdelt1=cdelt1_enlarged,
            wv_master=wv_master_all,
            threshold=3000,
            poldeg_residuals=5,
            use_r=False,
            title= os.path.basename(args.fitsfile.name) + '[slitlet #' +
                   str(islitlet).zfill(2) + ']',
            geometry=geometry,
            debugplot=12)

        # ToDo: use last result to modify the initial wavelength
        # calibration polynomial...


if __name__ == "__main__":

    main()
