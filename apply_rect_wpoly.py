from __future__ import division
from __future__ import print_function

import argparse
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

if __name__ == "__main__":
    main()
