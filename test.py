from __future__ import division
from __future__ import print_function

import argparse
from astropy.io import fits
import numpy as np

from numina.array.display.ximshow import ximshow_file

from spec_2d_image import Spec2DImage

from numina.array.display.pause_debugplot import DEBUGPLOT_CODES


def main(args=None):

    # parse command-line options
    parser = argparse.ArgumentParser(prog='display_slitlet_arrangement')
    parser.add_argument("fitsfile",
                        help="FITS file",
                        type=argparse.FileType('r'))
    parser.add_argument("--debugplot",
                        help="Integer indicating plotting & debugging options"
                             " (default=12)",
                        default=12, type=int,
                        choices=DEBUGPLOT_CODES)
    args = parser.parse_args(args)

    ximshow_file(args.fitsfile.name, debugplot=args.debugplot)


if __name__ == "__main__":
    main()
