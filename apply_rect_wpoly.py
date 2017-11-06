from __future__ import division
from __future__ import print_function

import argparse
from astropy.io import fits
import json
import sys
from uuid import uuid4

from csu_configuration import CsuConfiguration
from dtu_configuration import DtuConfiguration

from numina.array.display.pause_debugplot import DEBUGPLOT_CODES


def main(args=None):
    # parse command-line options
    parser = argparse.ArgumentParser(prog='apply_rect_wpoly')
    # required arguments
    parser.add_argument("fitsfile",
                        help="Input FITS file",
                        type=argparse.FileType('r'))
    parser.add_argument("--json_rect_wpoly", required=True,
                        help="Input JSON file with rectification and "
                             "wavelength calibration coefficients",
                        type=argparse.FileType('r'))

    # optional arguments
    parser.add_argument("--out_rectwv",
                        help="Rectified and wavelength calibrated output "
                             "FITS file",
                        type=argparse.FileType('w'))
    parser.add_argument("--debugplot",
                        help="Integer indicating plotting & debugging options"
                             " (default=0)",
                        default=0, type=int,
                        choices=DEBUGPLOT_CODES)
    parser.add_argument("--echo",
                        help="Display full command line",
                        action="store_true")
    args = parser.parse_args(args)

    if args.echo:
        print('\033[1m\033[31m% ' + ' '.join(sys.argv) + '\033[0m\n')

    # read the CSU configuration from the header of the input FITS file
    csu_conf = CsuConfiguration()
    csu_conf.define_from_fits(args.fitsfile)

    # read the DTU configuration from the header of the input FITS file
    dtu_conf = DtuConfiguration()
    dtu_conf.define_from_fits(args.fitsfile)

    # read calibration structure from JSON file
    rect_wpoly_dict = json.loads(open(args.json_rect_wpoly.name).read())

    # check that the DTU configuration employed to obtain the calibration
    # corresponds to the DTU configuration in the input FITS file
    dtu_conf_calib = DtuConfiguration()
    dtu_conf_calib.define_from_dictionary(rect_wpoly_dict['dtu_configuration'])
    if dtu_conf != dtu_conf_calib:
        print('>>> DTU configuration from FITS header:')
        print(dtu_conf)
        print('>>> DTU configuration from calibration JSON file:')
        print(dtu_conf_calib)
        raise ValueError("DTU configurations do not match!")
    if abs(args.debugplot) >= 10:
        print('>>> DTU Configuration math!')
        print(dtu_conf)

    # read FITS image and its corresponding header
    hdulist = fits.open(args.fitsfile)
    header = hdulist[0].header
    image2d = hdulist[0].data
    hdulist.close()

    # protections
    naxis2, naxis1 = image2d.shape
    if naxis1 != header['naxis1'] or naxis2 != header['naxis2']:
        print('Something is wrong with NAXIS1 and/or NAXIS2')
    if abs(args.debugplot) >= 10:
        print('>>> NAXIS1:', naxis1)
        print('>>> NAXIS2:', naxis2)

    # check that the input FITS file grism and filter match
    filter_name = header['filter']
    if filter_name != rect_wpoly_dict['tags']['filter']:
        raise ValueError("Filter name does not match!")
    grism_name = header['grism']
    if grism_name != rect_wpoly_dict['tags']['grism']:
        raise ValueError("Filter name does not match!")
    if abs(args.debugplot) >= 10:
        print('>>> grism.......:', grism_name)
        print('>>> filter......:', filter_name)

    # compute rectification and wavelength calibration coefficients for each
    # slitlet according to its csu_bar_slit_center value
    islitlet_min = rect_wpoly_dict['tags']['islitlet_min']
    islitlet_max = rect_wpoly_dict['tags']['islitlet_max']
    if abs(args.debugplot) >= 10:
        print('>>> islitlet_min:', islitlet_min)
        print('>>> islitlet_max:', islitlet_max)


if __name__ == "__main__":
    main()
